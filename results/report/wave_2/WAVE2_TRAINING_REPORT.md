# vncompress wave 2 — 2026-09-17

Full `run_pipeline.sh` recipe (slm -> probe -> encoder -> validate) run on
`pod-test` (stg cluster, llms-dev namespace, 1x H100 80GB, node wknqc),
against local checkpoints in `/mnt/hps/fp16_models/` and the vncompress-vi-v2
corpus already checked into the repo (`data/vncompress_vi_v2/{corpus,qa}.jsonl`).
No wandb -- everything below is raw stdout captured to `logs/*.log`.

A full copy of the code snapshot, logs, and model checkpoints from this run
lives on the cluster's shared storage at
`/mnt/hps/anhm-paper/vncompress-wave2-2026-09-17/` (see Layout below).

## Base models used (from `/mnt/hps/fp16_models/`)

| Role | Model |
|---|---|
| SLM base | `chronopt-research/vietnamese-gpt2-base` (125M) |
| Encoder | `vinai/phobert-base` (135M) |
| Teacher (E6 distillation) | `Qwen/Qwen2.5-0.5B-Instruct` |

## Results

### 1. `slm` — tone-aware LoRA adapter (`results/slm/final/`)

3 epochs, batch=64 (bumped from the pipeline default 8 -- see perf notes),
full corpus (56,729 train texts, 3,220 documents, non-Vietnamese and
other-split rows dropped). 2,394 optimizer steps, 7 min wall-clock.

LM loss plateaued ~4.5-4.7 (expected for a 3-epoch LoRA touching 0.94% of
params); tone auxiliary loss converged 0.22 -> 0.015.

**`train.py --mode slm --validate` (held-out split, 5,673 texts / 792,348
scored tokens):**

| Metric | Value |
|---|---|
| LM validation loss (NLL) | 4.4585 |
| Perplexity | 86.36 |
| Tone accuracy, all tokens | 96.52% |
| Tone accuracy, marked tones only | 94.85% |
| Majority-class baseline (always "ngang") | 43.48% |
| Training-free token-id lookup ceiling | 100.00% |
| Macro-F1 (marked tones only) | 0.9482 |

Per-tone F1 all >=0.92; weakest is "ngã" (F1 0.918, recall 0.880 — this tone
has the least support, 27,147/797k tokens, and the confusion matrix shows it
mostly gets confused with "sắc"/"nặng", a known perceptually-close pair).

The "marked tones only" number is the one that matters (per the training-free
ceiling above being trivially 100% otherwise) -- 94.85% says the LoRA-adapted
hidden states still carry almost all the deterministic tone-by-token-id
signal after 3 epochs of LM fine-tuning.

### 2. `probe` (E4) — query-relevance probe (`results/relevance/`)

3 epochs, batch=64, full `qa.jsonl` train split: 5,482 span-labelled samples
over 131 documents, 258 optimizer steps, ~1 min.

| Metric | Value |
|---|---|
| Accuracy | 71.4% |
| Precision | 5.6% |
| Recall | 63.4% |
| F1 | 0.103 |
| Positive support | 6,567 / 252,736 scored tokens (2.6%) |

Class imbalance is severe (44:1 weighting applied); F1 is low. This is the
metric `run_pipeline.sh` itself flags as the one to trust (accuracy alone
would look fine even for a "predict irrelevant always" probe: ~97%). Reading
this as "the probe learned *something*" (63% recall, well above the 2.6%
base rate) rather than "the probe is production-ready" -- more
epochs/data/threshold tuning would likely help; not attempted here.

### 3. `encoder` (E6) — distilled keep/drop classifier (`results/encoder_compressor_full/`)

ratio=4, 2 epochs, batch=64, **full corpus** (56,729 texts, all of them --
not the 4,000-text scoped run kept alongside it at `results/encoder_compressor/`
from the perf-debugging pass). 1,774 optimizer steps.

Training loss: 0.507 (step 20) -> 0.28-0.30 (final steps of epoch 2). No
held-out eval was run for this stage (the training script doesn't have a
validation split option) -- loss curve only.

## Bugs found and fixed (all applied to `vncompress/` — code changes, not
just this run's flags)

1. **Tone-label O(total tokens) recompute -> O(vocab actually seen) cache**
   (`vncompress/training.py`, `ToneTrainingDataset`/`VietnameseToneDataset`).
   `decode([tid]) + get_dominant_tone` is a pure function of `tid` (the
   codebase's own `_tone_lookup_baseline` already relied on this fact) but
   was being recomputed on every occurrence across the whole corpus.
   Memoizing by token id collapsed a stalled multi-minute dataset build into
   seconds. No behavior change -- verified by the existing baseline function
   using the identical cache pattern.

2. **`return_offsets_mapping` KeyError with PhoBERT**
   (`scripts/train_encoder_compressor.py`, `vncompress/encoder_compression.py`,
   `vncompress/compression.py`). PhoBERT's `config.json` pins
   `tokenizer_class: PhobertTokenizer` (slow), so `use_fast=True` is silently
   ignored; asking that tokenizer for offsets doesn't raise, it just omits
   the key. `encoder_compression.py` already had a decode-and-find fallback
   for this at inference time; `train_encoder_compressor.py` didn't. Extracted
   the shared fix as `encoder_token_offsets()` in `vncompress/compression.py`,
   used by both. (An earlier attempt to fix this by building a fast tokenizer
   from the repo's `tokenizer.json` was reverted: that file's vocab is larger
   than the checkpoint's embedding table -- 66,119 vs 64,001 -- and produced
   out-of-range ids for ~0.8% of real corpus texts, crashing with a CUDA
   `index out of bounds` assert deep in training.)

3. **Unbatched per-text teacher scoring in E6 labeling** (`vncompress/compression.py`,
   `scripts/train_encoder_compressor.py`). `DEFAULT_PPL_WINDOW=2048` means
   virtually every corpus text needs exactly one teacher forward pass, but
   the labeling loop was calling the teacher one text at a time (batch=1;
   ~30% GPU util / ~8GB VRAM on an 80GB H100). Added
   `sliding_window_perplexity_batch()`: right-pads many sequences into one
   forward call (numerically verified against the sequential version --
   causal attention makes right-padding a no-op for real tokens' logits;
   max abs diff observed was 0.023, consistent with ordinary fp16
   batch-size-dependent rounding, not a logic difference). Internally
   re-chunks by a token budget (batch_size * max_len <= 8192) after sorting
   by length, because a first pass at a naive fixed batch=64 hit a 28 GiB
   single-allocation CUDA OOM once a long-ish text padded an otherwise
   ordinary batch (Qwen's ~152k vocab makes log_softmax the dominant memory
   cost, not sequence length alone). Net effect: full-corpus E6 labeling
   dropped from an extrapolated hours-long run to ~9 minutes; GPU util rose
   from ~30% to ~96-99%.

## Layout (on cluster HPS, not this repo)

```
/mnt/hps/anhm-paper/vncompress-wave2-2026-09-17/
├── repo/                        # code snapshot (with all 3 fixes applied)
├── logs/                        # raw stdout, no wandb
│   ├── slm.log
│   ├── probe.log
│   ├── encoder.log              # 4,000-text scoped run (perf debugging)
│   ├── encoder_full.log         # canonical full-corpus run
│   └── validate.log
└── results/
    ├── slm/final/                    # LoRA adapter + tokenizer + val_split.json
    ├── slm/tone_probe.pt
    ├── relevance/relevance_probe.pt
    ├── encoder_compressor/           # 4,000-text scoped (kept for reference)
    └── encoder_compressor_full/      # canonical full-corpus checkpoint
```
