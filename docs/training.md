# Training guide: LACC model training, SLM/tone-probe training, and the inference bridge

Both training pipelines live in [`vncompress/training.py`](../vncompress/training.py) and run
through the single `train.py` entrypoint. This is **not** the VCC-Bench evaluation protocol (that's
[`docs/benchmark.md`](benchmark.md)) — it's how the models that feed LACC's perplexity/tone-probe
signals get produced and validated.

```
train.py --mode lacc   ->  models/lacc/final/  (LoRA adapter + tokenizer)
                        ->  models/lacc/tone_probe.pt

scripts/build_training_corpus.py  ->  data/benchmark/training_corpus_v1.json
                                            |
                                            v
train.py --mode slm --train-data-path ...  ->  models/slm/final/  (LoRA adapter + tokenizer + val_split.json)
                                            ->  models/slm/tone_probe.pt
                                            |
                                            v
train.py --mode slm --validate             ->  NLL / Perplexity / Tone accuracy
train.py --mode slm --validate --no-adapter ->  baseline (base model, no LoRA) for comparison
```

## 1. LACC model training (`train.py --mode lacc`)

Fine-tunes a (generation-sized) causal LM with LoRA + the phonological consistency auxiliary loss,
so its own hidden states can later serve as `LACCCompressor`'s same-tokenizer tone-probe signal
(`tone_source='model'`, no `scorer` set — see `vncompress/compression.py`).

```bash
python train.py --mode lacc --model Qwen/Qwen2.5-0.5B-Instruct --quick
python train.py --mode lacc --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3 --lambda-tone 0.1
python train.py --mode lacc --model Qwen/Qwen2.5-1.5B-Instruct --use-qlora
```

Runs on CPU or GPU (`--device`); GPU strongly recommended. Output: `models/lacc/final/` (LoRA
adapter + tokenizer) and `models/lacc/tone_probe.pt`.

## 2. SLM / tone-probe training (`train.py --mode slm`)

LoRA fine-tunes a small Vietnamese causal LM (default `chronopt-research/vietnamese-gpt2-base`,
137M) on low-VRAM GPUs, producing the `LACCScorer` this repo's `lightweight` and `full` hardware
tiers pair with a generation model via `vncompress.models.load_scorer()`. **Requires an NVIDIA CUDA
GPU** — CPU training is intentionally disabled for this path.

### 2.1 Build/extend the training corpus (optional but recommended)

The bundled fallback (`data/benchmark/wikipedia_vi_raw.json`, 393 paragraphs) is too small — the
tone probe learns close to random on it (~20% accuracy on marked tones; see 4.1 below).
`scripts/build_training_corpus.py` builds a much larger corpus from two Hugging Face sources (see
[`data/benchmark/PROVENANCE.md`](../data/benchmark/PROVENANCE.md) for full license/access details):

| Source | Role | License | Access |
|---|---|---|---|
| `undertheseanlp/UVW-2026` | bulk (~90%) | CC BY-SA 4.0 | public |
| `bigscience-data/roots_vi_vietnamese_poetry` | augmentation (~10%), boosts tone density/variety | MIT | gated — click "Agree and access repository" on the dataset page first |

```bash
python scripts/build_training_corpus.py                 # 5,000 UVW paragraphs + ~10% poetry
python scripts/build_training_corpus.py --uvw-n 20000    # ~4x corpus; consider fewer --epochs to compensate
```

Key flags: `--uvw-n` (paragraph count), `--poetry-ratio`, `--min-quality-score`, `--skip-poetry`
(if you don't have gated access yet), `--seed`, `--output`. Re-running **overwrites**
`training_corpus_v1.json` (rename the old file first if you want to keep it for comparison).
Committing the corpus is optional (licenses allow it); if you do,
`python scripts/checksum_datasets.py --write` and commit the updated `CHECKSUMS.json` in the same
commit, or CI's checksum-verify step will fail.

### 2.2 Train

```bash
python train.py --mode slm --train-data-path data/benchmark/training_corpus_v1.json
python train.py --mode slm --batch-size 1 --max-length 128 --grad-accum 8   # ~6GB GPU
# Larger base (e.g. Qwen3-4B), 4-bit QLoRA so it fits a single 12-16GB GPU:
python train.py --mode slm --model Qwen/Qwen3-4B --load-4bit \
    --train-data-path data/benchmark/training_corpus_v1.json \
    --epochs 2 --batch-size 1 --max-length 512 --grad-accum 16
```

Step-count estimate: `train_n = round(len(dataset)*0.9)`,
`updates_per_epoch = ceil(ceil(train_n/batch_size)/grad_accum)`,
`planned_steps = updates_per_epoch * epochs` (unless `--max-steps > 0`). Logs every 10 steps
(`step=... lm=... tone=... vram=...`).

Output: `models/slm/final/` (LoRA adapter + tokenizer + `val_split.json`, the exact held-out split
used at training time) and `models/slm/tone_probe.pt` (+ `tone_probe_meta.json`, recording the base
model so the scorer loader can catch a probe paired with the wrong base).

**Embedding/tokenizer mismatch**: some community checkpoints (e.g.
`chronopt-research/vietnamese-gpt2-base`) declare more tokenizer ids than the checkpoint has
embedding rows, which crashes with `CUDA error: srcIndex < srcSelectDimSize` on the very first
padded batch. `vncompress.models.resize_embeddings_if_needed()` fixes this (a no-op if sizes already
match) and is called by both training and validation — if you switch base models and hit this
error again, check `len(tokenizer)` vs `model.get_input_embeddings().weight.shape[0]`.

### 2.3 Validate (`train.py --mode slm --validate`)

```bash
python train.py --mode slm --validate --adapter-dir models/slm/final --tone-probe models/slm/tone_probe.pt
# Baseline: raw base model, no LoRA, same held-out split
python train.py --mode slm --validate --adapter-dir models/slm/final --no-adapter
```

If `models/slm/final/val_split.json` exists (recent training runs save it), validation
automatically uses that exact held-out split — no need to pass `--train-data-path`/`--max-length`.
Without it, you must pass the *same* values used at training time, or the split won't match (and
may leak train data into validation).

Output: validation text count, LM NLL/perplexity, tone accuracy (all tokens / marked tones only),
majority-class baseline, macro-F1 + confusion matrix, and the training-free lookup ceiling (below).

### 2.4 Statistical comparison across runs

```bash
python train.py --mode slm --validate --adapter-dir models/slm/final --no-adapter \
    --dump-per-sample results/training/base.json
python train.py --mode slm --validate --adapter-dir models/slm/final \
    --tone-probe models/slm/tone_probe.pt --dump-per-sample results/training/lora.json
python scripts/compare_slm_runs.py results/training/base.json results/training/lora.json
```

Paired (same texts, not independent arms) bootstrap 95% CI on delta-NLL and the perplexity ratio,
Wilcoxon signed-rank, and per-text win rate. Refuses to compare two dumps with different
`split_fingerprint`s (guards against a `val_split.json` silently overwritten between runs).

### 2.5 Control: does LoRA/`--lambda-tone` actually add anything? (`scripts/train_probe_control.py`)

```bash
python scripts/train_probe_control.py --mode frozen_base --out results/training/probe.jsonl
python scripts/train_probe_control.py --mode lora        --out results/training/probe.jsonl
python scripts/train_probe_control.py --mode frozen_base --control-task --out results/training/probe.jsonl
python scripts/train_probe_control.py --mode lora        --control-task --out results/training/probe.jsonl
```

| Delta | Meaning |
|---|---|
| `lora - frozen_base` | how much LoRA/`--lambda-tone` actually adds to the representation. ~0 means the "tone-aware training" claim doesn't hold. |
| `real - control_task` | probe **selectivity** (Hewitt & Liang 2019). Near 0 means the probe is memorizing token identity, not reading structure from the representation. |

## 3. Reading SLM validation numbers correctly

### 3.1 The tone-probe ceiling is 100%, and it's free

`VietnameseToneDataset` labels every token by `analyzer.get_dominant_tone(tokenizer.decode([id]))`
— a deterministic function of the token id alone, no context. So a training-free token-id -> tone
**lookup table** built straight from the tokenizer scores 100.00% on both "all tokens" and "marked
tones only". A trained probe scoring below 100% is losing information that was free; report both
numbers together, and pair with the frozen-base control (2.5) — otherwise a headline probe accuracy
number doesn't mean "the model learned Vietnamese tone", only "some tone information is linearly
readable from its hidden states" (a probing question, Hewitt & Liang 2019, not a prediction one).

### 3.2 Always read "marked tones only", not "all tokens"

*ngang* (no diacritic) is the overwhelming majority class (~46% of tokens), so "all tokens" accuracy
is inflated by trivially guessing ngang. "marked tones only" (the 5 diacritic tones) is what
actually reflects learned tone signal — random guessing among 5 classes is ~20%. Class distribution
is heavily imbalanced (sắc 16.7%, huyền 13.8%, nặng 12.8%, hỏi 7.4%, **ngã 3.1%**), so also check
**macro-F1** and the confusion matrix, not just accuracy: a model can ignore `ngã` almost entirely
and lose only ~3% accuracy while its F1 collapses to ~0 (and hỏi<->ngã is the classic confusable
pair in Vietnamese).

### 3.3 Perplexity: only compare within the same validation split, and always with a baseline

Comparing perplexity across different training corpora is meaningless (a harder/more diverse corpus
raises perplexity regardless of model quality). Always run `--no-adapter` on the *same*
`val_split.json` for a fair baseline, and confirm `Validation texts: <N>` matches between the two
runs before comparing numbers.

## 4. Measuring the SLM's real effect on the compression pipeline

Training/validating the SLM in isolation says nothing about whether it improves compression. Wire
it in via `benchmark.py`'s `--scorer-adapter-dir` (and `--tone-probe-path` for the trained-tone-probe
signal):

```bash
python benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --methods none,lacc \
    --scorer-adapter-dir models/slm/final \
    --output-dir results/slm-impact-v1
```

Compare (all via `vncompress.compression.LACCCompressor`, only the `scorer`/`tone_source` kwargs
differ):
- `lacc` with `scorer=<trained SLM>` vs. `lacc` with no scorer (rule-based, 0 VRAM) — does the SLM
  beat the pure heuristic?
- `lacc` with `use_adapter=True` vs. `use_adapter=False` on the same scorer — does fine-tuning
  itself contribute, independent of the base model?
- `lacc` with `tone_source='model'` vs. `tone_source='rule'` — does the trained tone probe add
  anything over the dictionary heuristic?
- every arm vs. `none` — how much quality is traded for how much compression.

**Tokenizer mismatch is handled automatically**: `benchmark.py` passes `LACCCompressor` token ids
from the *generation* model's tokenizer, while the scorer is a different model with its own
tokenizer (e.g. GPT-2 ~50k vocab vs. Qwen ~152k). `vncompress.compression.LACCScorer` maps scores
back through character offsets on the decoded text rather than assuming shared ids — see its
docstring. Verified: correct realized ratios (2x/4x/8x) with two fully mismatched tokenizers.

**Tokenizer choice affects Tone Preservation Rate directly**: a non-Vietnamese-aware tokenizer
under-represents tone-bearing tokens (measured: an English `gpt2` tokenizer put only 2.5% of tokens
in the tone-bearing category vs. 60.5% for `vietnamese-gpt2-base`'s tokenizer on the same sentence,
on top of inflating token count 3.7x). TPR trivially reports 1.0 when there are ~no tone-bearing
tokens to lose — always report the tone-bearing token ratio alongside TPR.

## 5. Filing a training/eval report

Every training run's results directory (`results/training/<date>_<mode>[-<short-description>]/`,
see [README.md's naming convention](../README.md#quy-tắc-đặt-tên-kết-quả-training)) should include
a short `README.md` with:

- who ran it, when, git commit, GPU, and the purpose of the run
- the exact `train.py --mode slm ...` command and resolved config
- dataset used (path, sample count, checksum if from `data/benchmark/`)
- training log highlights (final lm/tone loss, VRAM, step count)
- validation numbers (NLL, perplexity, tone accuracy all/marked, macro-F1, baseline comparison)
- notes: did it improve over the previous run, anything anomalous (OOM, loss not decreasing, ...)

See `results/training/reports/2026-08-20_slm-scaleup-training.md` for a worked example (a
scale-up training run compared against a smaller, unreliable earlier attempt).

## Related files

- [`scripts/build_training_corpus.py`](../scripts/build_training_corpus.py) — build the SLM training corpus
- [`scripts/compare_slm_runs.py`](../scripts/compare_slm_runs.py) — paired significance test between two eval dumps
- [`scripts/train_probe_control.py`](../scripts/train_probe_control.py) — frozen-base control + probe selectivity
- [`vncompress/training.py`](../vncompress/training.py) — both training pipelines + `validate_slm()`
- [`vncompress/models.py`](../vncompress/models.py) — `load_scorer()`, `resize_embeddings_if_needed()`
- [`vncompress/compression.py`](../vncompress/compression.py) — `LACCCompressor`, `LACCScorer`
- [`data/benchmark/PROVENANCE.md`](../data/benchmark/PROVENANCE.md) — dataset source/license/access details
- [`docs/benchmark.md`](benchmark.md) — VCC-Bench evaluation protocol (the compression algorithm, not the SLM)
