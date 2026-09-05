# VCC-Bench Evaluation Protocol

Ties together the unified experiment config ([`vncompress/config.py`](../vncompress/config.py)),
dataset provenance ([`data/benchmark/PROVENANCE.md`](../data/benchmark/PROVENANCE.md)), and the
Tone Preservation Rate metric ([`docs/tone_preservation_rate.md`](tone_preservation_rate.md)) into
one place: what to run, with what fixed settings, and what the output means.

## Dataset version

Every run against the "official" VCC-Bench evaluates against
`data/benchmark/vcc_bench_v1.json` — version `1.0.0`, snapshot date `2026-07-06`,
243 samples across 5 tasks. See [`data/benchmark/PROVENANCE.md`](../data/benchmark/PROVENANCE.md)
for full source/license/regeneration details and
[`data/benchmark/CHECKSUMS.json`](../data/benchmark/CHECKSUMS.json) for the exact file hashes a
result should be compared against (`python scripts/checksum_datasets.py` to verify).

Results are only directly comparable across runs that used the **same** dataset checksum and the
**same** git commit — both are captured automatically per-run (see "Run metadata" below).

## Fixed split, ratios, and seed

VCC-Bench v1 is a single fixed evaluation set (`ExperimentConfig.split` is always `"full"`). The
canonical ratios for the main table are **2.0, 4.0, 8.0** (`ExperimentConfig.compression_ratios`
default); `--quick` shrinks this to `[2.0]` for fast iteration — quick runs are for smoke-testing a
change, not for a paper table. `ExperimentConfig.seed` defaults to `42`; `benchmark.py`'s `main()`
calls `vncompress.config.set_seed(seed)` before anything stochastic runs (Python `random`, used by
`RandomCompressor`; NumPy; PyTorch CPU + CUDA). Same seed + config + dataset checksum + code commit
reproduces identical `RandomCompressor`-dependent numbers.

## Model, generation, and evaluation parameters

`ExperimentConfig.model` / `.tokenizer` (`.tokenizer` defaults to `.model`), `.max_new_tokens` /
`.temperature` / `.do_sample` (defaults `256` / `0.0` / `False`, i.e. greedy decoding, so quality
numbers aren't confounded by sampling variance).

## Run metadata: config + environment snapshot

`benchmark.py` writes two files into `--output-dir` **before** the run starts (via
`vncompress.config.save_run_metadata`):

- **`config.json`** — the fully-resolved `ExperimentConfig` actually used (`--config` file + CLI
  overrides; see `vncompress/config.py`'s module docstring for precedence).
- **`environment.json`** — git commit (+ dirty flag), UTC timestamp, Python version, and installed
  versions of the key dependencies (torch, transformers, accelerate, peft, numpy, scipy, datasets,
  evaluate, sentencepiece, rouge_score, sacrebleu, bert-score, underthesea).

Any `results/` directory can be traced back to exactly what produced it by reading these two files.

## Result JSON schema

`benchmark.py` writes `vcc_bench_results.json` into `--output-dir`, shaped as:

```jsonc
{
  "<method_name>": {                    // e.g. "none", "llmlingua", "lacc"
    "<task_name>": {                    // e.g. "long_document_qa"
      "ratio_<R>": {                    // e.g. "ratio_4.0" -- one per compression_ratios entry
        "mean_compression_ratio": 4.02,
        "mean_token_savings_pct": 75.1,
        "mean_processing_time_ms": 12.3,
        "mean_rouge_l_f1": 0.61,
        "mean_bleu": 0.34,
        "exact_match_rate": 0.10,
        "mean_quality_score": 0.55,
        "mean_efficiency_score": 0.80,
        "num_samples": 40,
        "mean_tone_preservation_rate": 0.88   // only if the method reports TPR -- see docs/tone_preservation_rate.md
      }
    }
  },
  "summary": { "...": "per-method aggregate across all tasks/ratios, see VCCBench._compute_summary()" }
}
```

Per-sample records (one `CompressionMetrics.to_dict()` per sample, see `vncompress/evaluation.py`)
are additionally written to `<method>_<task>_ratio<R>.json` when `VCCBenchConfig.save_predictions`
is `True` (the default). `config.json`/`environment.json` live alongside these files, so a results
directory is self-describing.

## Baseline vs. proposed vs. ablation

See [`vncompress/evaluation.py`](../vncompress/evaluation.py)'s `categorize()` /
`REGISTRY_METHOD_CATEGORY` / `ABLATION_ARM_CATEGORY` for the canonical classification of every
`vncompress.compression.METHODS` key (and the ablation arms below) into `baseline` / `proposed` /
`ablation`, and `scripts/summarize_results.py` for turning a `results/` directory into a
method x ratio x task comparison table that keeps the three categories visually distinct.

## Ablation: isolating LACC's signals

`LACCCompressor`'s three signals (perplexity, tone, morphology) are just config, so ablation needs
no separate entrypoint — `benchmark.py --ablation` runs four arms against the same samples:

| Arm | Config | Category |
|---|---|---|
| `ppl_only` | `use_tone=False, use_morphology=False` | ablation |
| `tone_only` | `use_perplexity=False, use_morphology=False` | ablation |
| `morph_only` | `use_perplexity=False, use_tone=False` | ablation |
| `lacc` | all three signals on | ablation (the full method, as a comparison point for the isolated arms) |

```bash
python benchmark.py --ablation --model Qwen/Qwen2.5-7B-Instruct --ratios 2,4,8
```

Only the token-selection signal changes between arms; downstream generation always uses the same
model, so ROUGE-L/BLEU/EM differences are attributable to the compression signal alone. Output goes
to `results_ablation/` by default (`ablation_results.json` + `ablation_report.md`, same config +
environment snapshot as a normal run).

## Quick/demo vs. full evaluation

- **`--quick`**: full `VCCBench.evaluate()` pipeline, but a single compression ratio and (when
  falling back to demo samples) fewer samples. Fast enough for iterating on a code change; not a
  substitute for a full run when reporting numbers.
- **`--demo`**: bypasses `VCCBench` entirely — runs every method on one hardcoded sample and prints
  detailed per-token output. For eyeballing what a compressor does to specific text, not for
  producing comparable metrics.
- **Full run** (neither flag): all configured methods x all 3 ratios x all 5 tasks x all 243
  samples. This is what should back any reported table/figure.

## Reproducing a run

```bash
# 1. Full benchmark with the example config (edit model/device as needed)
python benchmark.py --config configs/benchmark.json

# 2. Verify the dataset you're evaluating against hasn't drifted
python scripts/checksum_datasets.py

# 3. Compare results across runs/commits
python scripts/summarize_results.py results/qwen2.5-7b-vcc-bench-v1
```

Given the same `config.json` (dataset path + checksum, model, seed, ratios, generation params) and
the same `git_commit` in `environment.json`, a re-run reproduces the same
`mean_compression_ratio`/`mean_token_savings_pct` numbers exactly (deterministic compression), and
the same `RandomCompressor`/sampling-dependent numbers thanks to `set_seed()`. Quality metrics
(ROUGE-L, BLEU, BERTScore) are also deterministic under the default greedy decoding, modulo any
floating-point non-determinism in the underlying model/hardware kernels.

## Known hardware pitfall: `device_map` segfault on single-GPU Windows/CUDA

On a Windows/single-GPU CUDA machine, `AutoModelForCausalLM.from_pretrained(..., device_map=...)`
(any non-None value, including `{'': 0}`) can trigger an access violation inside
`transformers`/`accelerate` (`caching_allocator_warmup()` -> `torch.cuda.mem_get_info()`), not a
catchable Python exception. Every unquantized model load in this repo (`vncompress/models.py`'s
`load_model()`) therefore loads plain and moves with `.to(device)` afterward, which avoids the
crash entirely. Quantized (4-bit/8-bit) loads require a device_map and remain at some residual risk
on this specific hardware/OS combination — not exercised by the default fp16 benchmark config.

## Vietnamese-aware text metrics (ROUGE-L, token-F1)

`rouge_score`'s default tokenizer replaces every character outside `[a-z0-9]` with a space, which
silently collapses `'bàn'`/`'bán'`/`'bạn'` into the same tokens (every Vietnamese tone mark lives in
Latin Extended Additional). `vncompress/evaluation.py`'s `VietnameseRougeTokenizer` fixes this by
tokenizing to whitespace-separated syllables with punctuation stripped but tone marks intact — the
same units `compute_token_f1`'s `_normalize_answer` counts. Any `results/` produced before this fix
existed is not comparable to current numbers (check `environment.json`'s `git_commit`).
