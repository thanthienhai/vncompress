# Tone Preservation Rate (TPR)

Resolves P1 issue "research: xác thực và chuẩn hóa Tone Preservation Rate".

TPR is VNCompress's headline Vietnamese-specific quality metric: of the
tokens that carried tonal information in the original (uncompressed)
sequence, what fraction survived compression? Implemented once, canonically,
in [`vncompress/linguistics.py`](../vncompress/linguistics.py) as
`compute_tone_preservation_rate()`; `LACCCompressor.compress()` calls it
rather than recomputing the formula.

## Definition

For an original token sequence `t_0 .. t_{n-1}` with per-token tone analysis
`tone_infos = analyzer.analyze_tokens(tokens)` (see `TokenToneInfo`), and a
set `retained` of indices that survive compression:

```
tone_bearing = { i : tone_infos[i].tones_present is non-empty }

TPR = | tone_bearing ∩ retained | / | tone_bearing |
```

- **Tone-bearing token**: a decoded token string that contains at least one
  character carrying a non-*ngang* (level) tone mark — huyền, sắc, hỏi, ngã,
  or nặng. Determined character-by-character via `char_to_tone` lookup in
  `analyze_token()`; a token needs only *one* such character to count.
- **Numerator**: count of tone-bearing original positions whose index is
  also in `retained` — an exact index-based check (we know precisely which
  original positions survive), not a re-scan of decoded compressed text for
  tone characters, which could be fooled by coincidental character overlap.
- **Denominator**: count of tone-bearing original positions, full stop —
  independent of what the compressor kept.

TPR is always in `[0.0, 1.0]`.

## Edge cases

| Case | Behavior | Rationale |
|---|---|---|
| No tone-bearing tokens at all (denominator = 0) — e.g. all-*ngang* Vietnamese, non-Vietnamese text, pure punctuation/digits | `TPR = 1.0` | Vacuous truth: nothing tonal existed to lose, so nothing was lost. Prevents a `0/0` error and avoids penalizing text that was never tone-sensitive. |
| *ngang* (level tone) token, e.g. `"ma"`, `"xin"` | Excluded from **both** numerator and denominator | *ngang* carries no diacritic and is the tokenizer's/language's default tone; dropping such a token is an ordinary compression decision, not a tone-preservation failure. |
| Token retained but its neighbor (needed for tone *contrast*) was dropped | Does not affect TPR | TPR only measures per-token tone-mark survival by index, not the `preservation_weight`/contrast scoring used to *select* tokens. A token's own diacritic is unaffected by what happens to its neighbors. |
| Multi-syllable / multi-tone-mark token (rare with word-level splits, more common with certain subword vocabularies) | Counted once, as tone-bearing if *any* character in the token carries a tone mark | TPR operates at token granularity because that's the unit a compressor actually drops — see "Granularity" below. |

## Granularity: token-level, not syllable/word-level

TPR is computed over decoded **tokenizer output**, not over linguistic
syllables or words. In the common case a Vietnamese syllable maps to one
token and these coincide, but if a tokenizer's BPE/subword vocabulary ever
splits a syllable in a way that separates a base vowel from its diacritic
across two token IDs, each piece is scored independently as its own
`tones_present` lookup. This is intentional, not an approximation to fix:
compression operates on token IDs, so TPR measures exactly what the
compressor could see and choose to drop. A syllable-level TPR would answer a
different (also valid, but not currently implemented) question — "did whole
Vietnamese syllables survive intact" — which would require realigning
tokens back to syllable boundaries.

## Baseline: detecting metric inflation

A raw TPR number can look good for the wrong reason: text with very few
tone-bearing tokens to begin with will report a high TPR under almost any
compressor, tone-aware or not. `majority_tone_baseline_rate()` (same module)
gives the floor to compare against:

```
majority_tone_baseline_rate = |{non-tone-bearing tokens}| / |all tokens|
```

This is the "majority-class" rate for the binary tone-bearing / not label —
i.e. what a compressor that *only* ever dropped non-tone-bearing tokens
(and never touched a tone-bearing one) would trivially achieve for overall
token retention, used here as a sanity floor rather than a real strategy.
The other natural reference point is the `NoCompressor` baseline: since it
retains every index, `compute_tone_preservation_rate` on its output is
always exactly `1.0` by construction — any proposed method's TPR should be
read relative to that ceiling, not in isolation. See
`tests/test_linguistics.py`'s tone-preservation-rate tests for the concrete
comparison.

## Where TPR is reported

- `LACCCompressor.compress()` writes `metadata['tone_preservation_rate']` on
  every `CompressionResult` (when `use_tone=True`).
- `vncompress/evaluation.py`'s `VCCBench.evaluate()` copies that value onto
  `CompressionMetrics.tone_preservation_rate` when present
  (`'tone_preservation_rate' in result.metadata`), and `_aggregate_metrics()`
  reports `mean_tone_preservation_rate` per method/ratio in the benchmark
  summary — see [`docs/benchmark.md`](benchmark.md) for the result schema.
- Baselines that don't compute a tone signal (`none`, `random`, `llmlingua`,
  `snapkv`, `selective`) simply have no `tone_preservation_rate` key in their
  metadata, so it's omitted from aggregation rather than reported as a
  misleading `0.0` or `1.0`.

## Limitations

- Token-level, not syllable-level (see "Granularity" above) — comparisons
  across tokenizers with different vocabularies are not apples-to-apples
  unless the tokenizer is held fixed within an experiment (see
  `ExperimentConfig` in `vncompress/config.py`).
- Binary tone-bearing / not-tone-bearing signal only — does not distinguish
  "lost a *huyền* mark" from "lost a *nặng* mark", nor weight by how
  semantically load-bearing that particular tone contrast is in context
  (that finer-grained signal is what `preservation_weight` /
  `f_contrast` are for during *selection*; TPR is a coarser, easier-to-audit
  post-hoc retention metric).
- Says nothing about whether the *surrounding* words needed to make sense of
  a preserved tone-bearing token also survived — pair with ROUGE-L/BLEU/exact-match
  (also computed by `VCCBench`) rather than reading TPR alone as an
  end-to-end quality signal.
