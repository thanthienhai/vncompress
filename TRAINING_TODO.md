# LACC — Training & Evaluation TODO (handoff)

Minimal handoff for the training team. Goal: produce the measured numbers the
paper still lists as pending. **All code below already exists and is tested — no
coding needed, only GPU runs.** Nothing here trains from scratch on our side; you
run the jobs and report the numbers.

**First training target: `Qwen/Qwen3-4B`** (LoRA + Phonological Consistency Loss),
to evaluate how effective the tone-aware training is on a strong base before
investing in the smaller SLM tiers. The tone probe auto-sizes to the base's
hidden dim and everything downstream is base-agnostic — no code change to switch
bases. The small SLM `chronopt-research/vietnamese-gpt2-base` (125.6M, 6 GB GPU)
remains the fallback / lightweight tier.

VRAM for Qwen3-4B: **QLoRA 4-bit ≈ 12–16 GB** (`--load-4bit`), **bf16 LoRA ≈ ≥24 GB**
(`--base-dtype bfloat16`). Use the same precision flag on the eval jobs below.

---

## Status at a glance

| Item | State |
|---|---|
| Tone-probe trainer, GPT-2 SLM (`run_train_slm.py`) | ✅ done (Runs 1–3, tone acc → 92.9%) |
| Trainer now supports Qwen3-4B (bf16 / 4-bit QLoRA) | ✅ code done, **needs a run** |
| Tone probe → live compressor bridge (`slm_tone_probe.py`) | ✅ code done, **needs a run** |
| ROUGE-L Vietnamese fix | ✅ done |
| Metric suite (per-task F1/ROUGE-L/needle/BLEU/BERTScore + paired CI) | ✅ done |
| Full VCC-Bench quality run (post-fix) | ⏳ **TODO — run** |
| End-to-end tone-probe A/B (probe vs rule) | ⏳ **TODO — run** |
| Probe control study (LoRA vs frozen-base, selectivity) | ⏳ **TODO — run** |

---

## Datasets

| Purpose | File | Source / license | How to get |
|---|---|---|---|
| **Train** SLM+probe | `vcc_bench_data/training_corpus_v1.json` | UVW-2026 (CC-BY-SA) + Vietnamese poetry (MIT, **gated** — needs HF login) | `python scripts/build_training_corpus.py` (build first, not committed) |
| **Eval** benchmark | `vcc_bench_data/vcc_bench_v1.json` | 243 samples, 5 tasks (Wikipedia CC-BY-SA + legal PD + synthetic MIT) | committed ✅ |
| **Eval** real QA (optional) | `vcc_bench_data/vcc_bench_uit_viquad_qa.json` | UIT-ViQuAD2.0, **eval-only**, license unconfirmed | `python scripts/build_viquad_eval.py` (gitignored) |

Scale used for the paper's best run (Run 3): ~20,000 UVW + ~2,200 poetry ≈ 22,200 docs.
Split is deterministic (seed 42, 90/10); the held-out val split is saved to
`trained_slm/final/val_split.json` at train time.

---

## Jobs to run (in order)

### 1. Train Qwen3-4B + tone probe  ⟵ the real training job (first target)
```bash
# Qwen3-4B, QLoRA 4-bit (~12-16 GB GPU):
python run_train_slm.py --model Qwen/Qwen3-4B --load-4bit \
    --output-dir trained_qwen3 \
    --train-data-path vcc_bench_data/training_corpus_v1.json \
    --epochs 2 --batch-size 1 --max-length 512 --grad-accum 16 --lambda-tone 0.1
#   >=24 GB GPU, no quant:  --base-dtype bfloat16   (drop --load-4bit)

# Fallback / lightweight tier — GPT-2 SLM on a 6 GB GPU:
python run_train_slm.py --output-dir trained_slm \
    --train-data-path vcc_bench_data/training_corpus_v1.json \
    --epochs 2 --batch-size 1 --max-length 128 --grad-accum 8 --lambda-tone 0.1
```
**Outputs** (in `--output-dir`): `final/` (LoRA adapter + tokenizer + `val_split.json`),
`tone_probe.pt`, `tone_probe_meta.json` (NEW — keep it; records base model + dtype,
the inference loader uses it to verify the probe matches the base).

### 2. Evaluate perplexity + tone accuracy  ⟵ the direct "does it work" number
```bash
python evaluate_slm.py --adapter-dir trained_qwen3/final \
    --tone-probe trained_qwen3/tone_probe.pt --load-4bit   # match training precision
python evaluate_slm.py --adapter-dir trained_qwen3/final --no-adapter --load-4bit  # base baseline
```
Report: perplexity (LoRA vs base, same split), tone acc on marked tones, macro-F1,
majority-class baseline. This alone evaluates whether the tone loss works on
Qwen3-4B (paper Table 2). (For the GPT-2 SLM, drop `--load-4bit` and use `trained_slm`.)

### 3. Probe control study (proves the probe learned, not memorized)
```bash
python scripts/train_probe_control.py --mode lora --adapter-dir trained_qwen3/final --load-4bit
python scripts/train_probe_control.py --mode frozen_base --adapter-dir trained_qwen3/final --load-4bit
python scripts/train_probe_control.py --mode lora --control-task --adapter-dir trained_qwen3/final --load-4bit
python scripts/train_probe_control.py --mode frozen_base --control-task --adapter-dir trained_qwen3/final --load-4bit
```
Trains only the probe (cheap, model frozen). Read: `lora − frozen_base` = what
LoRA/λ_tone added; `real − control` = probe selectivity.

### 4. End-to-end tone-probe A/B  ⟵ the headline (paper limitation #2)
```bash
python scripts/verify_tone_probe_e2e.py \
    --generation-model Qwen/Qwen3-4B \
    --scorer-adapter-dir trained_qwen3/final \
    --tone-probe-path   trained_qwen3/tone_probe.pt --scorer-4bit \
    --ratios 2,4 --max-samples 40 --bertscore
# fast structural check, no generation model needed:
python scripts/verify_tone_probe_e2e.py --no-generation --max-samples 20 --scorer-4bit
```
(Qwen3-4B can serve as both scorer and generation model, as above.)
Controlled A/B (one SLM shared): `slm_tone_probe` (trained probe) vs
`slm_tone_probe_rule` (heuristic), same everything else. Reports per-task quality
+ TPR + the **probe−rule delta with 95% bootstrap CI + p-value**. Outputs
`results_tone_probe_e2e/*.json` + `*.md`.

### 5. Full VCC-Bench quality run (paper limitation #1)
```bash
python run_benchmark.py --model Qwen/Qwen3-4B \
    --methods none,random,llmlingua,combined,slm_scorer,slm_tone_probe_rule,slm_tone_probe \
    --scorer-adapter-dir trained_qwen3/final --tone-probe-path trained_qwen3/tone_probe.pt \
    --ratios 2,4,8
```
Full 243 samples × {2,4,8}× with the corrected ROUGE-L. This is the number that
was invalid before the diacritic fix and must be re-run. (run_benchmark loads the
scorer in fp32 by default; for a 4B scorer here, prefer the e2e script in job 4,
or run this on a larger GPU.)

---

## Hardware / notes
- Training (job 1), Qwen3-4B: **QLoRA 4-bit ≈ 12–16 GB**, bf16 LoRA ≈ ≥24 GB.
  GPT-2 SLM fallback: 6 GB. Keep the SAME precision flag (`--load-4bit` /
  `--base-dtype`) on jobs 2–4 so the adapter loads onto a matching base.
- Jobs 2–3 are cheap (probe/eval only, base frozen). Jobs 4–5 also run a
  generation model — Qwen3-4B can be both scorer and generator.
- Windows/CUDA: models load plain FP16 then `.to(cuda)` — **do not** add
  `device_map` (segfaults here; see `docs/benchmark.md`).
- Determinism: seed 42 everywhere. Log optimizer-step count + peak VRAM (Run 3
  didn't — instrumentation gap to close).
- `build_training_corpus.py` overwrites `training_corpus_v1.json`; if you retrain,
  the old `val_split.json` no longer matches — always evaluate against the
  `val_split.json` saved by *that* run.

## Want-to-do (optional, later)
- Calibrate blend weights `w_ppl/w_tone/w_morph` (`vncompress/calibration/weight_search.py`)
  instead of fixed 0.4/0.3/0.3 — the paper notes combined < tone-only under equal weights.
- Grow under-represented tasks (needle=9, agent=8) in VCC-Bench.
- Real-QA eval on UIT-ViQuAD2.0 (job 5 with `--data-path vcc_bench_data/vcc_bench_uit_viquad_qa.json`).

See `docs/tone_probe_bridge.md` for the full design of jobs 1/4.
