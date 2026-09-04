# Wiring the trained tone probe into the live compressor

This document describes the training→inference bridge for LACC's model-based
tone signal (paper Sect. 3.4) and how to reproduce its end-to-end effect. It
closes the gap the paper lists as limitation #2: *"the Vietnamese-native SLM
scorer is trained and probed but not yet substituted into the live LACC
pipeline, so it has no measured end-to-end quality effect yet."*

## What was missing

`run_train_slm.py` trains a Vietnamese SLM (`chronopt-research/vietnamese-gpt2-base`)
with LoRA plus the auxiliary **Phonological Consistency Loss**. Its 7-way tone
classifier is saved to `trained_slm/tone_probe.pt`. The paper defines the
inference-time reuse of that same classifier as a *tone probe*:

> S_tone-model(t_i) = 1.0 + max_k softmax(MLP(h_i))_k

where `h_i` is the SLM's last-layer hidden state at token `i`. But nothing in
the compressor pipeline ran it end-to-end:

- `SLMScorerCompressor` (`slm_scorer`) used the SLM only for **perplexity**; its
  tone term was the dictionary heuristic, not the probe.
- `ToneAwareCompressor(use_model_tone=True)` *does* call `score_importance`, but
  on the **generation** model's hidden states (e.g. Qwen 7B, hidden size 3584),
  while the probe was trained on the **SLM**'s hidden states (GPT-2, hidden size
  768). The dimensions do not match, and even if they did the signal would be
  meaningless — a probe read off a model it never saw. That path was also never
  reachable from `run_benchmark.py` (the registry never exposed it).

## The bridge

`vncompress/compressors/slm_tone_probe.py` adds `SLMToneProbeScorer` +
`SLMToneProbeCompressor`:

1. Load the SLM (base + LoRA adapter) **and** its `tone_probe.pt` together —
   the probe only means anything on the model it was trained with. A
   `tone_probe_meta.json` written at training time is checked to catch a probe
   paired with the wrong base model.
2. Run the SLM once over the text. In a **single sliding-window pass** it
   produces, per SLM token, both the perplexity signal `-log P(t_i|t_<i)` and
   the model tone signal `1 + max softmax(probe(h_i))`.
3. Map both back onto the **generation** model's tokens through character
   offsets (the two models tokenize differently), exactly as `SLMScorerCompressor`
   already does for perplexity.
4. Blend `w_ppl·S_ppl + w_tone·S_tone + w_morph·S_morph` and select with the
   shared boundary-aware selector.

Because selection uses a **hard token budget** (`BaseCompressor.select_with_boundary`),
realized compression ratio tracks the target even on long contexts — unlike the
July pilot, whose model-based tier under-compressed ≥32K-token needle contexts
(paper Table 3 footnote).

## Registry methods

| Method | Tone signal | Category | Purpose |
|--------|-------------|----------|---------|
| `slm_tone_probe` | trained probe | proposed | the model-based tone tier |
| `slm_tone_probe_rule` | dictionary heuristic | ablation | controlled A/B: same SLM, same perplexity/morphology/selection, only the tone term differs |

Comparing the two isolates exactly what the probe adds, because everything else
is held fixed. `slm_scorer` (perplexity + heuristic tone) remains available as
the lightweight tier.

## Running the end-to-end verification

No training is run here; the adapter + probe from `run_train_slm.py` must exist.

```bash
# Full A/B with generation (needs a generation model + GPU):
python scripts/verify_tone_probe_e2e.py \
    --generation-model Qwen/Qwen2.5-0.5B-Instruct \
    --scorer-adapter-dir trained_slm/final \
    --tone-probe-path   trained_slm/tone_probe.pt \
    --ratios 2,4 --max-samples 40 --output-dir results_tone_probe_e2e

# Fast structural check (compression ratio + TPR only, no generation):
python scripts/verify_tone_probe_e2e.py --no-generation --max-samples 20
```

It loads the SLM **once** and shares it between the model-tone and rule-tone
arms. Outputs `tone_probe_e2e_results.json` + `tone_probe_e2e_report.md` with a
`config.json`/`environment.json` snapshot.

### Metrics reported

The set mirrors what prompt/context-compression papers report — LLMLingua /
LongLLMLingua, LongBench, the NAACL 2025 prompt-compression survey, and recent
empirical studies — so the numbers are directly comparable to prior work:

- **Task-appropriate quality, per task** (LongBench convention): token-level
  **F1** for QA / agent tasks, **ROUGE-L** for summarization / multi-turn /
  cross-lingual, and **needle-retrieval recall** (RULER-style) for
  needle-in-haystack. **BLEU**, **Exact Match**, and optional multilingual
  **BERTScore** (`--bertscore`) are recorded for every task. All text metrics
  use the tone-preserving tokenization from `evaluation/metrics.py` (post
  ROUGE-L fix), so tone marks are never silently collapsed.
- **Tone Preservation Rate** — the Vietnamese-specific metric (paper P1),
  computed identically for every arm so the table compares like with like.
- **Compression**: realized ratio + token savings (confirms the target is met).
- **Efficiency**: compression overhead vs generation latency, and speedup vs the
  uncompressed baseline — the throughput dimension compression papers headline.
- **Performance retention**: primary quality kept relative to no compression
  (e.g. "retains 92% of uncompressed F1 at 4×").
- **Statistical rigor**: the headline `slm_tone_probe − slm_tone_probe_rule`
  delta is reported with a **paired 95% bootstrap CI**, a **p-value**, and a
  **per-sample win rate** (`evaluation/significance.py`), so a positive delta is
  distinguishable from noise rather than a bare point estimate. Pairing is on
  the same samples, matching `scripts/compare_slm_runs.py`'s paired design for
  the SLM perplexity comparison.

## Or through the full benchmark

```bash
python run_benchmark.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --methods none,random,slm_scorer,slm_tone_probe_rule,slm_tone_probe \
    --scorer-adapter-dir trained_slm/final \
    --tone-probe-path   trained_slm/tone_probe.pt \
    --ratios 2,4,8
```

## Caveats

- `use_adapter=False` on the scorer loads the un-adapted base model; the probe
  was trained jointly with the LoRA weights, so that combination is a diagnostic
  only, not a meaningful readout — interpret it alongside
  `scripts/train_probe_control.py` (frozen-base vs LoRA).
- The probe's ceiling is a training-free token-id→tone lookup (`evaluate_slm.py`),
  because the label is a deterministic function of the token id. The end-to-end
  question this bridge answers is different: whether feeding that learned signal
  into selection improves downstream answer quality over the heuristic tone term.
