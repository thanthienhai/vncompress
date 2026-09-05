# Pending experiments (handoff)

Minimal handoff for whoever runs the next training/eval jobs. **All code below
already exists and is tested — no coding needed, only GPU runs.** Nothing here
trains from scratch on our side; run the jobs and report the numbers back into
`results/training/<date>_<name>/README.md` (see `docs/training.md`'s "Filing a
training/eval report" section).

**First training target: `Qwen/Qwen3-4B`** (LoRA + Phonological Consistency
Loss), to evaluate how effective tone-aware training is on a strong base
before investing further in the smaller SLM tier. The tone probe auto-sizes
to the base's hidden dim and everything downstream is base-agnostic — no
code change needed to switch bases. `chronopt-research/vietnamese-gpt2-base`
(125.6M, 6GB GPU) remains the fallback/lightweight-tier default.

VRAM for Qwen3-4B: QLoRA 4-bit ~12-16GB (`--load-4bit`), bf16 LoRA ~24GB+
(`--base-dtype bfloat16`). Use the same precision flag across jobs 1-4 below.

## Status at a glance

| Item | State |
|---|---|
| Tone-probe trainer, GPT-2 SLM (`train.py --mode slm`) | done (3 runs, tone acc reached 92.9% marked) |
| Trainer supports Qwen3-4B (bf16 / 4-bit QLoRA) | code done, **needs a run** |
| Tone probe -> live compressor bridge (`LACCCompressor(tone_source='model')`) | code done, **needs a run** on Qwen3-4B |
| ROUGE-L Vietnamese tone-mark fix | done |
| Metric suite (per-task F1/ROUGE-L/needle/BLEU/BERTScore + paired CI) | done |
| Full VCC-Bench quality run (post-ROUGE-fix) | **TODO — run** |
| End-to-end tone-probe A/B (probe vs rule) on Qwen3-4B | **TODO — run** |
| Probe control study (LoRA vs frozen-base, selectivity) on Qwen3-4B | **TODO — run** |
| Calibrate LACC's blend weights (`vncompress.compression.calibrate_*`) instead of fixed 0.4/0.3/0.3 | not done |

## Datasets

| Purpose | File | Source / license | How to get |
|---|---|---|---|
| Train SLM+probe | `data/benchmark/training_corpus_v1.json` | UVW-2026 (CC-BY-SA) + Vietnamese poetry (MIT, gated) | `python scripts/build_training_corpus.py` (not committed by default) |
| Eval benchmark | `data/benchmark/vcc_bench_v1.json` | 243 samples, 5 tasks | committed |
| Eval real QA (optional) | `data/benchmark/vcc_bench_uit_viquad_qa.json` | UIT-ViQuAD2.0, eval-only, license unconfirmed | `python scripts/build_viquad_eval.py` (gitignored) |

Scale used for the best SLM run so far: ~20,000 UVW + ~2,200 poetry docs.
Split is deterministic (seed 42, 90/10); the held-out val split is saved to
`models/slm/final/val_split.json` at train time.

## Jobs to run (in order)

### 1. Train Qwen3-4B + tone probe (the real training job, first target)

```bash
# QLoRA 4-bit (~12-16 GB GPU):
python train.py --mode slm --model Qwen/Qwen3-4B --load-4bit \
    --output-dir models/qwen3 \
    --train-data-path data/benchmark/training_corpus_v1.json \
    --epochs 2 --batch-size 1 --max-length 512 --grad-accum 16 --lambda-tone 0.1
#   >=24 GB GPU, no quant:  --base-dtype bfloat16   (drop --load-4bit)

# Fallback / lightweight tier -- GPT-2 SLM on a 6 GB GPU:
python train.py --mode slm --output-dir models/slm \
    --train-data-path data/benchmark/training_corpus_v1.json \
    --epochs 2 --batch-size 1 --max-length 128 --grad-accum 8 --lambda-tone 0.1
```

Outputs (in `--output-dir`): `final/` (LoRA adapter + tokenizer + `val_split.json`),
`tone_probe.pt`, `tone_probe_meta.json` (records base model + dtype; the
inference loader, `vncompress.models.load_scorer()`, uses it to verify the
probe matches the base).

### 2. Evaluate perplexity + tone accuracy (the direct "does it work" number)

```bash
python train.py --mode slm --validate --adapter-dir models/qwen3/final \
    --tone-probe models/qwen3/tone_probe.pt --load-4bit   # match training precision
python train.py --mode slm --validate --adapter-dir models/qwen3/final --no-adapter --load-4bit
```

Report: perplexity (LoRA vs base, same split), tone accuracy on marked tones,
macro-F1, majority-class baseline. Evaluates whether the tone loss works on
Qwen3-4B. (For the GPT-2 SLM, drop `--load-4bit` and use `models/slm`.)

### 3. Probe control study (proves the probe learned, not memorized)

```bash
python scripts/train_probe_control.py --mode lora --adapter-dir models/qwen3/final --load-4bit
python scripts/train_probe_control.py --mode frozen_base --adapter-dir models/qwen3/final --load-4bit
python scripts/train_probe_control.py --mode lora --control-task --adapter-dir models/qwen3/final --load-4bit
python scripts/train_probe_control.py --mode frozen_base --control-task --adapter-dir models/qwen3/final --load-4bit
```

Trains only the probe (cheap, model frozen). Read: `lora - frozen_base` = what
LoRA/lambda_tone added; `real - control` = probe selectivity.

### 4. End-to-end tone-probe A/B (the headline measurement)

```bash
python scripts/verify_tone_probe_e2e.py \
    --generation-model Qwen/Qwen3-4B \
    --scorer-adapter-dir models/qwen3/final \
    --tone-probe-path   models/qwen3/tone_probe.pt --scorer-4bit \
    --ratios 2,4 --max-samples 40 --bertscore
# fast structural check, no generation model needed:
python scripts/verify_tone_probe_e2e.py --no-generation --max-samples 20 --scorer-4bit
```

(Qwen3-4B can serve as both scorer and generation model, as above.) Controlled
A/B (one SLM shared): `lacc_tone_probe` (trained probe) vs `lacc_tone_rule`
(heuristic), same everything else. Reports per-task quality + TPR + the
probe-rule delta with 95% bootstrap CI + p-value. Outputs go to
`results/tone_probe_e2e/*.json` + `*.md`.

### 5. Full VCC-Bench quality run

```bash
python benchmark.py --model Qwen/Qwen3-4B \
    --methods none,random,llmlingua,lacc \
    --scorer-adapter-dir models/qwen3/final --tone-probe-path models/qwen3/tone_probe.pt \
    --ratios 2,4,8
python benchmark.py --ablation --model Qwen/Qwen3-4B \
    --scorer-adapter-dir models/qwen3/final --tone-probe-path models/qwen3/tone_probe.pt \
    --ratios 2,4,8
```

Full 243 samples x {2,4,8}x with the corrected ROUGE-L. `benchmark.py` loads
the scorer in fp32 by default; for a 4B scorer, either pass `--scorer-dtype`
equivalent handling (see `vncompress.models.load_scorer`'s `dtype`/`load_4bit`
args) or run this on a larger GPU.

## Hardware / notes

- Training (job 1), Qwen3-4B: QLoRA 4-bit ~12-16GB, bf16 LoRA ~24GB+. GPT-2
  SLM fallback: 6GB. Keep the same precision flag on jobs 2-4 so the adapter
  loads onto a matching base.
- Jobs 2-3 are cheap (probe/eval only, base frozen). Jobs 4-5 also run a
  generation model -- Qwen3-4B can be both scorer and generator.
- Windows/CUDA: models load plain FP16/BF16 then `.to(cuda)` -- do not add
  `device_map` for unquantized loads (segfaults here; see `docs/benchmark.md`).
- Determinism: seed 42 everywhere. Log optimizer-step count + peak VRAM.
- `build_training_corpus.py` overwrites `training_corpus_v1.json`; if you
  retrain, the old `val_split.json` no longer matches -- always evaluate
  against the `val_split.json` saved by *that* run.

## Want-to-do (optional, later)

- Calibrate LACC's blend weights (`vncompress.compression.calibrate_lacc_compressor`
  / `calibrate_score_weights`) instead of the fixed 0.4/0.3/0.3 default.
- Grow under-represented VCC-Bench tasks (needle=9, agent=8 samples).
- Real-QA eval on UIT-ViQuAD2.0 (job 5 with
  `--data-path data/benchmark/vcc_bench_uit_viquad_qa.json`).

See `docs/training.md` for the full design of jobs 1 and 4.
