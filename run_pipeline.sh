#!/usr/bin/env bash
#
# Full vncompress training pipeline on an Ubuntu GPU box.
#
#   ./run_pipeline.sh                      # every stage, GPU 0
#   ./run_pipeline.sh --gpu 3              # every stage, GPU 3
#   ./run_pipeline.sh --gpu 0,1            # expose two GPUs
#   ./run_pipeline.sh --quick              # a few steps per stage, to prove the wiring
#   ./run_pipeline.sh --stages data,slm    # only those stages
#   ./run_pipeline.sh --dry-run            # print the commands, run nothing
#
# Stages, in order:
#   check    host, driver and GPU sanity
#   venv     create/reuse the virtualenv
#   deps     install torch for this GPU's CUDA, then requirements.txt
#   data     pull thanthienhai/vncompress-vi-v2 from the Hub
#   slm      train.py --mode slm            -> models/slm/final  (+ tone_probe.pt)
#   probe    train_relevance_probe.py (E4)  -> models/relevance/relevance_probe.pt
#   encoder  train_encoder_compressor.py    -> models/encoder_compressor
#   validate train.py --mode slm --validate
#   bench    VCC-Bench v2 sweep (E1/E2/E5/E7/E6) + tone-probe and E4
#            relevance-probe A/Bs, all against data already on disk --
#            no LLM-generated gold needed (see WAVE2_DATA_NOTES.md ss1)
#
# Everything below is overridable from the environment, e.g.
#   GPU=2 SLM_EPOCHS=5 ./run_pipeline.sh
#
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

# --- configuration -----------------------------------------------------------

GPU="${GPU:-0}"                       # which physical GPU(s); dynamic, default 0
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${LOG_DIR:-logs}"

# 12.8+ builds, so a cu124 wheel imports fine, reports the GPU, and then fails
# at the first kernel launch. cu128 is therefore the default, not cu124.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_SPEC="${TORCH_SPEC:-torch>=2.7.0}"

# thanthienhai/vncompress-vi-v2 is the rebuilt, entity-clean, Ha-Noi-held-out
# dataset (2026-09-17) -- see memory/compression-budget-miss.md and
# e4-probe-v2-wiring.md for why the original anhalu/vncompress-vi-v2 is no
# longer the default: it still has the corpus HTML-entity leak and Hà Nội
# sits in its train split despite overlapping VCC-Bench v2.
HF_DATASET="${HF_DATASET:-thanthienhai/vncompress-vi-v2}"
DATA_DIR="${DATA_DIR:-data/vncompress_vi_v2}"
# corpus.jsonl for the LM/encoder arms, qa.jsonl + qa_synthetic.jsonl (stage
# 2b, teacher-generated but span-verified) for the E4 relevance probe's
# document coverage. --all-data adds the compression/eval/extras files.
# NOTE: qa_synthetic.jsonl only exists on thanthienhai/vncompress-vi-v2 -- if
# you override HF_DATASET back to a repo without it, also override DATA_FILES
# or the download will 404.
DATA_FILES="${DATA_FILES:-corpus.jsonl qa.jsonl qa_synthetic.jsonl}"
DATA_FILES_ALL="corpus.jsonl qa.jsonl qa_synthetic.jsonl compression.jsonl eval/test.jsonl extras/compression_unanswerable.jsonl README.md"

SLM_EPOCHS="${SLM_EPOCHS:-3}"
SLM_BATCH="${SLM_BATCH:-8}"
SLM_MAX_LEN="${SLM_MAX_LEN:-256}"
SLM_DTYPE="${SLM_DTYPE:-bfloat16}"
SLM_OUT="${SLM_OUT:-models/slm}"

PROBE_EPOCHS="${PROBE_EPOCHS:-3}"
PROBE_BATCH="${PROBE_BATCH:-8}"
PROBE_MAX_LEN="${PROBE_MAX_LEN:-512}"
PROBE_OUT="${PROBE_OUT:-models/relevance}"
# G2 (docs/lacc_coling2027_tasklist.md): retrain E4 with focal loss. Wave 2's
# probe measured P=4.9% / R=59% -- inverse-frequency class weights bought
# recall at the cost of precision, and plain CE let the ~98% easy negatives
# own the gradient. gamma=2 is the usual starting point; 0 restores wave-2
# behaviour exactly. PROBE_CLASS_WEIGHT_CAP is the *other* lever on the same
# trade-off (lower cap = fewer false positives), deliberately left at the
# wave-2 value so focal loss is the only variable that changed between the two
# probes -- move one at a time or the comparison says nothing.
PROBE_FOCAL_GAMMA="${PROBE_FOCAL_GAMMA:-2.0}"
PROBE_CLASS_WEIGHT_CAP="${PROBE_CLASS_WEIGHT_CAP:-50.0}"

ENCODER_ID="${ENCODER_ID:-vinai/phobert-base}"
ENCODER_TEACHER="${ENCODER_TEACHER:-Qwen/Qwen2.5-0.5B-Instruct}"
ENCODER_RATIO="${ENCODER_RATIO:-4}"
ENCODER_EPOCHS="${ENCODER_EPOCHS:-2}"
ENCODER_BATCH="${ENCODER_BATCH:-8}"
ENCODER_MAX_TEXTS="${ENCODER_MAX_TEXTS:--1}"
ENCODER_OUT="${ENCODER_OUT:-models/encoder_compressor}"
# Held-out DOCUMENTS (not paragraphs -- corpus.jsonl averages ~17 paragraphs per
# document). Wave 2 reported E6 on a loss curve alone because the script had no
# split at all; 0 restores that.
ENCODER_VAL_FRACTION="${ENCODER_VAL_FRACTION:-0.05}"
ENCODER_SEED="${ENCODER_SEED:-42}"

# bench: VCC-Bench v2 sweep (the full paper arm list -- E1/E2/E5/E7/E8 plus
# the E6 encoder arm, scored with the trained SLM) and the two probe A/Bs
# (verify_tone_probe_e2e.py),
# one with the SLM's own tone_probe.pt, one with the E4 relevance_probe.pt
# swapped into the same slot (probe_kind is auto-detected from its sidecar
# meta json -- see scripts/train_relevance_probe.py's docstring). Everything
# here reads data already on disk; it does not touch compression.jsonl and
# does not need an LLM API key.
# Generation model that reads the compressed context and answers. Deliberately
# NOT $ENCODER_TEACHER: wave 2 read the benchmark with the 0.5B teacher and
# every arm collapsed into a 0.03-0.14 quality band -- uncompressed context
# itself scored 0.114 and *lost* to compressed arms, which is what a reader too
# weak to use its own context looks like. Nothing about compression is
# measurable in that regime. It also graded E6 with the exact model E6 was
# distilled from. 7B in fp16 needs ~16GB, so it fits beside the scorer on one
# H100; drop back to the 0.5B only for wiring smoke tests.
BENCH_MODEL="${BENCH_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
# Arm list finalized in docs/lacc_coling2027_tasklist.md G0 (2026-09-18): the
# full paper sweep, including 'llmlingua' (plain, non-contrastive baseline)
# and 'lacc_tone_gated' (E8) alongside the arms already here, plus the G2
# "dịch-rồi-nén" baseline (translate_then_compress[_llmlingua2]) -- survival
# baseline per *Lost in Compression*, without which the "Vietnamese needs a
# native compressor" premise is untested.
#
# 'lacc_tone' (tone always on, the wave-1 arm) is E8's control and is NOT
# optional: gated-vs-off is not the E8 question, gated-vs-always-on is. Note
# how to read that pair -- LACCCompressor.surface_tasks is
# {cross_lingual, translation, transliteration, quotation}, and VCC-Bench v2 is
# 220 long_document_qa / 120 needle / 48 multi_turn / 18 cross_lingual / 8
# agent_tool_calling, so the gate only changes behaviour on 18 of 414 samples.
# In the aggregate table 'lacc_tone_gated' will therefore look like a tone-off
# arm; the E8 claim lives in the cross_lingual per-task rows (n=18, wide CI),
# not in the headline number.
BENCH_METHODS="${BENCH_METHODS:-none,random,llmlingua,llmlingua_contrastive,lacc_ppl_contrastive,lacc_ppl_morph,lacc_cx_morph,lacc_sentence,lacc_classprop,lacc_tone,lacc_tone_gated,encoder,translate_then_compress,translate_then_compress_llmlingua2}"
BENCH_NLI_MODEL="${BENCH_NLI_MODEL:-MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7}"  # source_span_recoverability / unsupported_claim_rate; "" to skip
BENCH_RATIOS="${BENCH_RATIOS:-2,4,8}"
# One seed of `random` is a sample, not a floor (docs/eval_sweep_gate1.md SS4
# rule 3): without this the weakest arm in the table is a single draw.
BENCH_RANDOM_SEEDS="${BENCH_RANDOM_SEEDS:-1,2,3}"
# Empty = all five tasks. Set this to run the answer-shaped tasks first
# (needle_in_haystack,agent_tool_calling,cross_lingual): on v2,
# long_document_qa + multi_turn_conversation references are verbatim passages
# (median 347 words) rather than answers, so they cost 65% of the GPU time for
# a number that measures passage recovery -- see scripts/analyze_sweep.py.
BENCH_TASKS="${BENCH_TASKS:-}"
# Samples per model.generate() call. The full 5-task sweep is ~23k generations;
# one at a time on a 7B reader that is days, not hours. Lower it if the pod OOMs
# (a failed batch retries one sample at a time, so the cost is time, not data);
# set 1 if you need real per-sample generation latency.
BENCH_GEN_BATCH_SIZE="${BENCH_GEN_BATCH_SIZE:-8}"
BENCH_DATA_PATH="${BENCH_DATA_PATH:-data/benchmark/vcc_bench_v2.json}"
BENCH_OUT="${BENCH_OUT:-results/bench_wave2}"
PROBE_AB_RATIOS="${PROBE_AB_RATIOS:-2,4,8}"
PROBE_AB_MAX_SAMPLES="${PROBE_AB_MAX_SAMPLES:-100}"
BENCH_ABLATION="${BENCH_ABLATION:-1}"        # 0 to skip the ppl/tone/morph isolation sweep
BENCH_TRAINED_SCORER="${BENCH_TRAINED_SCORER:-1}"  # 0 reverts the lacc_* arms to scoring with the generation model

ALL_STAGES="check venv deps data slm probe encoder validate bench"
STAGES="${STAGES:-$ALL_STAGES}"
QUICK=0
DRY_RUN=0
SKIP_TESTS="${SKIP_TESTS:-0}"

# --- plumbing ----------------------------------------------------------------

BOLD=$'\033[1m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; GREEN=$'\033[32m'; OFF=$'\033[0m'
[ -t 1 ] || { BOLD=""; RED=""; YELLOW=""; GREEN=""; OFF=""; }

log()  { printf '%s[%s]%s %s\n' "$BOLD" "$(date +%H:%M:%S)" "$OFF" "$*"; }
warn() { printf '%s[%s] WARNING:%s %s\n' "$YELLOW" "$(date +%H:%M:%S)" "$OFF" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$RED" "$(date +%H:%M:%S)" "$OFF" "$*" >&2; exit 1; }
ok()   { printf '%s[%s] OK:%s %s\n' "$GREEN" "$(date +%H:%M:%S)" "$OFF" "$*"; }

run() {
    printf '  %s$%s %s\n' "$BOLD" "$OFF" "$*"
    [ "$DRY_RUN" = 1 ] && return 0
    "$@"
}

wants() { [[ " $STAGES " == *" $1 "* ]]; }

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)       GPU="$2"; shift 2 ;;
        --gpu=*)     GPU="${1#*=}"; shift ;;
        --stages)    STAGES="${2//,/ }"; shift 2 ;;
        --stages=*)  STAGES="${1#*=}"; STAGES="${STAGES//,/ }"; shift ;;
        --quick)     QUICK=1; shift ;;
        --all-data)  DATA_FILES="$DATA_FILES_ALL"; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        -h|--help)   usage ;;
        *)           die "unknown argument: $1 (try --help)" ;;
    esac
done

for stage in $STAGES; do
    [[ " $ALL_STAGES " == *" $stage "* ]] || die "unknown stage: $stage (valid: $ALL_STAGES)"
done

# The one line that makes the GPU choice dynamic: everything downstream --
# torch, transformers, bitsandbytes -- reads this, and inside the process the
# selected card is always cuda:0, so no script needs a device index.
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

if [ "$QUICK" = 1 ]; then
    SLM_EPOCHS=1; PROBE_EPOCHS=1; ENCODER_EPOCHS=1; ENCODER_MAX_TEXTS=200
    QUICK_SLM_ARGS=(--max-steps 30); QUICK_PROBE_ARGS=(--max-steps 30 --min-samples 8)
    PROBE_AB_MAX_SAMPLES=8
    QUICK_BENCH_ARGS=(--quick --limit-samples 20)
else
    QUICK_SLM_ARGS=(); QUICK_PROBE_ARGS=(); QUICK_BENCH_ARGS=()
fi

# --- stages ------------------------------------------------------------------

stage_check() {
    log "Stage: check"
    [ "$(uname -s)" = "Linux" ] || warn "this script targets Ubuntu; uname says $(uname -s)"
    command -v "$PYTHON_BIN" >/dev/null || die "$PYTHON_BIN not found. apt install python3 python3-venv python3-pip"
    "$PYTHON_BIN" -c 'import venv' 2>/dev/null || die "python venv module missing. sudo apt install -y python3-venv"
    command -v nvidia-smi >/dev/null || die "nvidia-smi not found -- no NVIDIA driver on this host."

    log "Visible GPU(s): CUDA_VISIBLE_DEVICES=$GPU"
    run nvidia-smi -i "$GPU" --query-gpu=index,name,memory.total,compute_cap,driver_version \
        --format=csv,noheader
    ok "host looks sane"
}

activate_venv_if_present() {
    [ -n "${VIRTUAL_ENV:-}" ] && return 0
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        log "activated $VENV_DIR"
    fi
    return 0
}

stage_venv() {
    log "Stage: venv ($VENV_DIR)"
    if [ -d "$VENV_DIR" ]; then
        log "reusing existing $VENV_DIR"
    else
        run "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    [ "$DRY_RUN" = 1 ] || source "$VENV_DIR/bin/activate"
    run python -m pip install --upgrade pip setuptools wheel
    ok "virtualenv ready"
}

stage_deps() {
    log "Stage: deps"
    # torch first, from the CUDA-specific index, so requirements.txt's
    # `torch>=2.1.0` is already satisfied and pip does not pull a generic wheel
    # over it.
    run python -m pip install --index-url "$TORCH_INDEX_URL" "$TORCH_SPEC"
    run python -m pip install -r requirements.txt
    # Not in requirements.txt but imported by the 4-bit paths
    # (vncompress/models.py, training.py) and by the Hub download below.
    run python -m pip install "huggingface_hub[cli]"
    # Only the --load-4bit paths import it, and its Blackwell wheels lag; a
    # failure here must not sink a run that never asks for 4-bit.
    run python -m pip install bitsandbytes || warn "bitsandbytes not installed -- do not pass --load-4bit"

    [ "$DRY_RUN" = 1 ] && return 0

    log "verifying torch sees the GPU"
    python - <<'PY'
import sys, torch
print(f"  torch {torch.__version__} (CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    sys.exit("torch cannot see a GPU -- check the driver and the CUDA build of torch.")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
arches = torch.cuda.get_arch_list()
print(f"  device 0: {name} (sm_{major}{minor})")
print(f"  torch was built for: {', '.join(arches)}")
# A Blackwell card with a pre-12.8 wheel imports and reports the GPU happily,
# then dies on the first kernel launch. Catch it here, not 40 minutes in.
if f"sm_{major}{minor}" not in arches and f"sm_{major}0" not in arches:
    sys.exit(
        f"This torch build has no kernels for sm_{major}{minor} ({name}).\n"
        "Reinstall with a matching CUDA index, e.g.\n"
        "  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./run_pipeline.sh --stages deps")
torch.randn(64, 64, device="cuda").matmul(torch.randn(64, 64, device="cuda")).sum().item()
print("  kernel launch: OK")
PY
    if [ "$SKIP_TESTS" != 1 ]; then
        log "running the test suite (CPU, ~1 min)"
        run python -m pytest -q
    fi
    ok "dependencies installed"
}

stage_data() {
    log "Stage: data ($HF_DATASET -> $DATA_DIR)"
    run mkdir -p "$DATA_DIR"
    local cli=""
    if command -v hf >/dev/null; then cli="hf"
    elif command -v huggingface-cli >/dev/null; then cli="huggingface-cli"
    fi

    for f in $DATA_FILES; do
        if [ -s "$DATA_DIR/$f" ]; then
            log "have $f, skipping"
            continue
        fi
        if [ "$cli" = "hf" ]; then
            run hf download "$HF_DATASET" "$f" --repo-type dataset --local-dir "$DATA_DIR"
        elif [ "$cli" = "huggingface-cli" ]; then
            run huggingface-cli download "$HF_DATASET" "$f" --repo-type dataset --local-dir "$DATA_DIR"
        else
            run mkdir -p "$DATA_DIR/$(dirname "$f")"
            run curl -fL --retry 3 -C - \
                "https://huggingface.co/datasets/$HF_DATASET/resolve/main/$f" \
                -o "$DATA_DIR/$f"
        fi
    done

    [ "$DRY_RUN" = 1 ] && return 0
    for f in $DATA_FILES; do
        [ -s "$DATA_DIR/$f" ] || die "$DATA_DIR/$f is missing or empty after download"
        case "$f" in *.jsonl) log "  $f: $(wc -l < "$DATA_DIR/$f") rows" ;; esac
    done
    ok "dataset in place"
}

stage_slm() {
    log "Stage: slm -> $SLM_OUT"
    # Reads corpus.jsonl's `train` split and holds out the benchmark documents
    # by default; both are decided inside load_training_texts.
    run python train.py --mode slm \
        --output-dir "$SLM_OUT" \
        --epochs "$SLM_EPOCHS" \
        --batch-size "$SLM_BATCH" \
        --max-length "$SLM_MAX_LEN" \
        --base-dtype "$SLM_DTYPE" \
        "${QUICK_SLM_ARGS[@]}"
    [ "$DRY_RUN" = 1 ] || [ -d "$SLM_OUT/final" ] || die "$SLM_OUT/final was not produced"
    ok "SLM adapter + tone probe saved"
}

stage_probe() {
    log "Stage: probe (E4) -> $PROBE_OUT"
    local adapter_args=()
    if [ -d "$SLM_OUT/final" ]; then
        adapter_args=(--adapter-dir "$SLM_OUT/final")
        log "using the adapter from the slm stage"
    else
        warn "$SLM_OUT/final not found -- training the probe on the plain base model"
    fi
    local data_path="$DATA_DIR/qa.jsonl"
    if [ -s "$DATA_DIR/qa_synthetic.jsonl" ]; then
        # E4's document coverage is the scarce thing (WAVE2_DATA_NOTES.md):
        # qa.jsonl alone is ~142 documents; qa_synthetic.jsonl (stage 2b)
        # adds ~3,393 more, teacher-generated but span-verified. Same row
        # shape (context/query/answer/answer_span/task/doc_id/split), so a
        # plain concat is a valid combined source -- the benchmark holdout is
        # re-derived downstream by document_key regardless of which file a
        # row came from.
        data_path="$DATA_DIR/qa_combined.jsonl"
        [ "$DRY_RUN" = 1 ] || cat "$DATA_DIR/qa.jsonl" "$DATA_DIR/qa_synthetic.jsonl" > "$data_path"
        log "training on qa.jsonl + qa_synthetic.jsonl combined ($data_path)"
    else
        warn "$DATA_DIR/qa_synthetic.jsonl not found -- training on qa.jsonl alone (limited document coverage)"
    fi
    run python scripts/train_relevance_probe.py \
        "${adapter_args[@]}" \
        --data-path "$data_path" \
        --output-dir "$PROBE_OUT" \
        --epochs "$PROBE_EPOCHS" \
        --batch-size "$PROBE_BATCH" \
        --max-length "$PROBE_MAX_LEN" \
        --focal-gamma "$PROBE_FOCAL_GAMMA" \
        --class-weight-cap "$PROBE_CLASS_WEIGHT_CAP" \
        "${QUICK_PROBE_ARGS[@]}"
    [ "$DRY_RUN" = 1 ] || [ -f "$PROBE_OUT/relevance_probe.pt" ] || die "relevance_probe.pt was not produced"
    # The validation F1 is the number that says whether the probe learned
    # anything; accuracy alone is ~0.96 for a probe that answers "irrelevant"
    # to everything.
    [ "$DRY_RUN" = 1 ] || run python -c "
import json; m=json.load(open('$PROBE_OUT/relevance_probe_meta.json'))
v=m.get('val_metrics') or {}
print('  validation:', {k: round(x,4) for k,x in v.items()} or 'none recorded')
print('  trained on:', m.get('num_train_samples'), 'samples /', m.get('num_train_documents'), 'documents')
"
    ok "relevance probe saved"
}

stage_encoder() {
    log "Stage: encoder (E6) -> $ENCODER_OUT"
    run python scripts/train_encoder_compressor.py \
        --encoder-id "$ENCODER_ID" \
        --teacher-model "$ENCODER_TEACHER" \
        --ratio "$ENCODER_RATIO" \
        --epochs "$ENCODER_EPOCHS" \
        --batch-size "$ENCODER_BATCH" \
        --max-texts "$ENCODER_MAX_TEXTS" \
        --val-fraction "$ENCODER_VAL_FRACTION" \
        --seed "$ENCODER_SEED" \
        --output-dir "$ENCODER_OUT"
    [ "$DRY_RUN" = 1 ] || run python -c "
import json; m=json.load(open('$ENCODER_OUT/encoder_compressor_meta.json'))
v=m.get('val_metrics') or {}
print('  held-out:', {k: round(x,4) for k,x in v.items() if isinstance(x,(int,float))} or 'none recorded')
print('  trained on:', m.get('num_train_texts'), 'texts /', m.get('num_train_documents'), 'documents')
"
    ok "encoder compressor saved"
}

stage_validate() {
    log "Stage: validate"
    [ "$DRY_RUN" = 1 ] || [ -d "$SLM_OUT/final" ] || die "nothing to validate: $SLM_OUT/final is missing (run the slm stage first)"
    run python train.py --mode slm --validate \
        --adapter-dir "$SLM_OUT/final" \
        --tone-probe "$SLM_OUT/tone_probe.pt"
    ok "validation done"
}

stage_bench() {
    log "Stage: bench -> $BENCH_OUT"

    # --- VCC-Bench v2 sweep: the full paper arm list (E1/E2/E5/E7/E8) ------
    # plus 'encoder' (E6) when a checkpoint exists, all scored with the trained
    # SLM (see scorer_args below). None of this reads compression.jsonl.
    local methods="$BENCH_METHODS"
    local encoder_args=()
    if [ -f "$ENCODER_OUT/config.json" ]; then
        encoder_args=(--encoder-path "$ENCODER_OUT")
    else
        # Both arms need the E6 checkpoint: 'encoder' directly, and
        # 'translate_then_compress_llmlingua2' (G2 baseline) as its inner step.
        warn "$ENCODER_OUT/config.json not found -- dropping 'encoder' and " \
             "'translate_then_compress_llmlingua2' from --methods (run the encoder stage first)"
        methods="$(echo ",$methods," \
            | sed 's/,encoder,/,/' \
            | sed 's/,translate_then_compress_llmlingua2,/,/' \
            | sed 's/^,//; s/,$//')"
    fi
    local nli_args=()
    [ -n "$BENCH_NLI_MODEL" ] && nli_args=(--nli-model "$BENCH_NLI_MODEL")
    local task_args=()
    [ -n "$BENCH_TASKS" ] && task_args=(--tasks "$BENCH_TASKS")
    local seed_args=()
    [ -n "$BENCH_RANDOM_SEEDS" ] && seed_args=(--random-seeds "$BENCH_RANDOM_SEEDS")
    local gen_args=(--gen-batch-size "$BENCH_GEN_BATCH_SIZE")

    # The lacc_* arms score with the SLM the slm stage trained -- the same
    # scorer the ablation below already uses. Wave 2 ran the sweep WITHOUT
    # these flags, and without a scorer LACC silently falls back to scoring
    # perplexity with the *generation* model (compression.py, the
    # `elif self.use_perplexity and self.model is not None` branch). So its
    # headline table scored with Qwen-0.5B while its ablation table scored with
    # the trained SLM -- two tables that cannot be read against each other, and
    # no arm anywhere measuring what wave 2 actually trained. The fallback also
    # ties scorer strength to reader strength, so raising BENCH_MODEL would
    # silently change the compressor too; passing the adapter keeps the reader
    # and the scorer independent. The rule-vs-probe question is a separate
    # axis, answered by the two probe A/Bs below.
    local scorer_args=()
    if [ "$BENCH_TRAINED_SCORER" = 1 ]; then
        if [ -d "$SLM_OUT/final" ]; then
            scorer_args=(--scorer-adapter-dir "$SLM_OUT/final")
            [ -f "$SLM_OUT/tone_probe.pt" ] && scorer_args+=(--tone-probe-path "$SLM_OUT/tone_probe.pt")
        else
            warn "$SLM_OUT/final not found -- lacc_* arms fall back to scoring perplexity with the " \
                 "GENERATION model ($BENCH_MODEL), not the trained SLM; the sweep table will not be " \
                 "comparable with the ablation table (run the slm stage first)"
        fi
    fi

    if [ -n "$methods" ]; then
        run python benchmark.py \
            --model "$BENCH_MODEL" \
            --methods "$methods" \
            --ratios "$BENCH_RATIOS" \
            --data-path "$BENCH_DATA_PATH" \
            --output-dir "$BENCH_OUT/sweep" \
            "${encoder_args[@]}" \
            "${scorer_args[@]}" \
            "${nli_args[@]}" \
            "${task_args[@]}" \
            "${seed_args[@]}" \
            "${gen_args[@]}" \
            "${QUICK_BENCH_ARGS[@]}"
        # Significance, from the per-sample rows the sweep just wrote. Arm means
        # alone cannot say whether a 0.002 gap is a result; this is also the only
        # per-task view, which is where the E8 (cross_lingual) read-out lives.
        [ "$DRY_RUN" = 1 ] || run python scripts/analyze_sweep.py "$BENCH_OUT/sweep" \
            --metric token_f1 --reference none --out "$BENCH_OUT/sweep/significance_token_f1.md"
    else
        warn "no bench methods left to run -- skipping the VCC-Bench sweep"
    fi

    # --- Probe A/Bs (verify_tone_probe_e2e.py): same SLM, same perplexity/
    # morphology signals, only the tone-term source differs (rule vs trained
    # probe). Run once per probe checkpoint that exists.
    if [ -f "$SLM_OUT/final/adapter_config.json" ] || [ -d "$SLM_OUT/final" ]; then
        if [ -f "$SLM_OUT/tone_probe.pt" ]; then
            run python scripts/verify_tone_probe_e2e.py \
                --generation-model "$BENCH_MODEL" \
                --scorer-adapter-dir "$SLM_OUT/final" \
                --tone-probe-path "$SLM_OUT/tone_probe.pt" \
                --data-path "$BENCH_DATA_PATH" \
                --ratios "$PROBE_AB_RATIOS" --max-samples "$PROBE_AB_MAX_SAMPLES" \
                --bertscore \
                --output-dir "$BENCH_OUT/tone_probe_ab"
        else
            warn "$SLM_OUT/tone_probe.pt not found -- skipping the tone-probe A/B"
        fi

        if [ -f "$PROBE_OUT/relevance_probe.pt" ]; then
            # Same script, relevance_probe.pt in the tone-probe slot: probe_kind
            # is auto-detected from relevance_probe_meta.json (see
            # scripts/train_relevance_probe.py docstring). This is the headline
            # E4 measurement -- expect worse tone preservation, better answers.
            run python scripts/verify_tone_probe_e2e.py \
                --generation-model "$BENCH_MODEL" \
                --scorer-adapter-dir "$SLM_OUT/final" \
                --tone-probe-path "$PROBE_OUT/relevance_probe.pt" \
                --data-path "$BENCH_DATA_PATH" \
                --ratios "$PROBE_AB_RATIOS" --max-samples "$PROBE_AB_MAX_SAMPLES" \
                --bertscore \
                --output-dir "$BENCH_OUT/relevance_probe_ab"
        else
            warn "$PROBE_OUT/relevance_probe.pt not found -- skipping the E4 relevance-probe A/B"
        fi

        # --- Ablation: isolate perplexity/tone/morphology as three single-
        # signal arms plus the combined 'lacc', all against VCC-Bench v2. Not
        # gated on the encoder checkpoint -- this is purely about the LACC
        # scorer's three signals, 'encoder' plays no part in it.
        if [ "$BENCH_ABLATION" = 1 ]; then
            run python benchmark.py \
                --ablation \
                --model "$BENCH_MODEL" \
                --scorer-adapter-dir "$SLM_OUT/final" \
                --tone-probe-path "$SLM_OUT/tone_probe.pt" \
                --ratios "$BENCH_RATIOS" \
                --data-path "$BENCH_DATA_PATH" \
                --output-dir "$BENCH_OUT/ablation" \
                "${nli_args[@]}" \
                "${task_args[@]}" \
                "${gen_args[@]}" \
                "${QUICK_BENCH_ARGS[@]}"
        fi
    else
        warn "$SLM_OUT/final not found -- skipping both probe A/Bs and the ablation sweep (run the slm stage first)"
    fi

    ok "bench sweep + probe A/Bs + ablation done"
}

# --- run ---------------------------------------------------------------------

main() {
    local mode=""; [ "$QUICK" = 1 ] && mode=" | quick"
    log "vncompress pipeline | GPU=$GPU | stages: $STAGES$mode"
    log "log file: $RUN_LOG"
    # Running a later stage on its own (e.g. --stages slm) must still use the
    # project's interpreter, not whatever `python` the login shell has.
    [ "$DRY_RUN" = 1 ] || activate_venv_if_present
    local started=$SECONDS
    for stage in $ALL_STAGES; do
        wants "$stage" || continue
        # venv/deps must run in this shell to keep the activated environment;
        # the rest inherit it.
        "stage_${stage}"
    done
    ok "pipeline finished in $(( (SECONDS - started) / 60 ))m $(( (SECONDS - started) % 60 ))s"
    log "artifacts: $SLM_OUT/final, $PROBE_OUT/relevance_probe.pt, $ENCODER_OUT, $BENCH_OUT"
}

main 2>&1 | tee -a "$RUN_LOG"
exit "${PIPESTATUS[0]}"
