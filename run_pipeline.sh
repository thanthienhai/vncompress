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
#   data     pull anhalu/vncompress-vi-v2 from the Hub
#   slm      train.py --mode slm            -> models/slm/final  (+ tone_probe.pt)
#   probe    train_relevance_probe.py (E4)  -> models/relevance/relevance_probe.pt
#   encoder  train_encoder_compressor.py    -> models/encoder_compressor
#   validate train.py --mode slm --validate
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

HF_DATASET="${HF_DATASET:-anhalu/vncompress-vi-v2}"
DATA_DIR="${DATA_DIR:-data/vncompress_vi_v2}"
# Only these two feed training: corpus.jsonl for the LM/encoder arms, qa.jsonl
# for the E4 relevance probe. --all-data adds the compression/eval/extras files.
DATA_FILES="${DATA_FILES:-corpus.jsonl qa.jsonl}"
DATA_FILES_ALL="corpus.jsonl qa.jsonl compression.jsonl eval/test.jsonl extras/compression_unanswerable.jsonl README.md"

SLM_EPOCHS="${SLM_EPOCHS:-3}"
SLM_BATCH="${SLM_BATCH:-8}"
SLM_MAX_LEN="${SLM_MAX_LEN:-256}"
SLM_DTYPE="${SLM_DTYPE:-bfloat16}"
SLM_OUT="${SLM_OUT:-models/slm}"

PROBE_EPOCHS="${PROBE_EPOCHS:-3}"
PROBE_BATCH="${PROBE_BATCH:-8}"
PROBE_MAX_LEN="${PROBE_MAX_LEN:-512}"
PROBE_OUT="${PROBE_OUT:-models/relevance}"

ENCODER_ID="${ENCODER_ID:-vinai/phobert-base}"
ENCODER_TEACHER="${ENCODER_TEACHER:-Qwen/Qwen2.5-0.5B-Instruct}"
ENCODER_RATIO="${ENCODER_RATIO:-4}"
ENCODER_EPOCHS="${ENCODER_EPOCHS:-2}"
ENCODER_BATCH="${ENCODER_BATCH:-8}"
ENCODER_MAX_TEXTS="${ENCODER_MAX_TEXTS:--1}"
ENCODER_OUT="${ENCODER_OUT:-models/encoder_compressor}"

ALL_STAGES="check venv deps data slm probe encoder validate"
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

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

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
else
    QUICK_SLM_ARGS=(); QUICK_PROBE_ARGS=()
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
    run python scripts/train_relevance_probe.py \
        "${adapter_args[@]}" \
        --output-dir "$PROBE_OUT" \
        --epochs "$PROBE_EPOCHS" \
        --batch-size "$PROBE_BATCH" \
        --max-length "$PROBE_MAX_LEN" \
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
        --output-dir "$ENCODER_OUT"
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
    log "artifacts: $SLM_OUT/final, $PROBE_OUT/relevance_probe.pt, $ENCODER_OUT"
}

main 2>&1 | tee -a "$RUN_LOG"
exit "${PIPESTATUS[0]}"
