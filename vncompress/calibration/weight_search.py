"""
Weight Calibration Harness (LACC improvement B1)
=================================================
The LACC blend weights — w_ppl/w_tone/w_morph (`ScoreWeights`), the tone
formula's alpha/beta/gamma, the tone/morph blend `tone_weight`, and the
per-class morphology multipliers f_func/f_content/f_redup/f_compound — were
all fixed, hand-picked constants. This module replaces "pick numbers that
feel right" with a small, dependency-light search over a validation set of
Vietnamese texts.

Why a proxy objective instead of full VCC-Bench generation:
  Running the 7B generation model for every candidate parameter set is far
  too expensive for a search loop (hundreds of compress+generate calls).
  Instead we optimize a self-supervised proxy that only needs the tokenizer
  and the (cheap, CPU-only) tone/morphology analyzers:

    quality    = 0.5 * tone_preservation_rate + 0.5 * content_word_retention
    efficiency = token_savings_pct / 100
    objective  = harmonized_score = 2 * quality * efficiency / (quality + efficiency)

  This mirrors the `harmonized_score` already used by VCCBench._compute_summary,
  so a set of weights that scores well here is directly comparable to the
  final VCC-Bench numbers. Once a short-list of candidates is found this way,
  it is still worth confirming the winner with a full VCC-Bench run.

Usage:
    from vncompress.calibration import (
        CalibrationObjective, coordinate_ascent, calibrate_combined_compressor,
    )

    best_params, history = calibrate_combined_compressor(
        tokenizer, model, texts, config, device='cuda',
    )
    compressor = CombinedCompressor(tokenizer, model, config, device='cuda', **best_params)
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..tone_aware.vietnamese_tones import VietnameseToneAnalyzer, get_tone_analyzer
from ..morphology.merge_policy import MorphologyAnalyzer, get_morphology_analyzer, WordClass


# ============================================================================
# Default search spaces (centered on the previous hand-picked defaults)
# ============================================================================

DEFAULT_PARAM_CANDIDATES: Dict[str, List[float]] = {
    'alpha': [0.3, 0.5, 0.7],
    'beta': [0.2, 0.3, 0.4],
    'gamma': [0.2, 0.4, 0.6],
    'tone_weight': [0.3, 0.5, 0.7],
    'f_func': [0.3, 0.4, 0.5],
    'f_content': [1.0, 1.2, 1.5],
    'f_redup': [0.4, 0.6, 0.8],
    'f_compound': [1.2, 1.5, 2.0],
}

DEFAULT_SCORE_WEIGHT_CANDIDATES: Dict[str, List[float]] = {
    'perplexity': [0.2, 0.4, 0.6],
    'tone': [0.2, 0.3, 0.4],
    'morphology': [0.2, 0.3, 0.4],
}


def _default_initial(param_candidates: Dict[str, List[float]]) -> Dict[str, float]:
    """Middle value of each candidate list — matches the old hand-picked defaults."""
    return {name: values[len(values) // 2] for name, values in param_candidates.items()}


# ============================================================================
# Result normalization — different compressors return different shapes
# ============================================================================

@dataclass
class SampleResult:
    """Normalized view of a single compress() call, regardless of compressor type."""
    compressed_ids: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_compress_output(raw: Any) -> SampleResult:
    """
    Compressors in this codebase return one of:
      - CompressionResult / NoModelResult (object with .compressed_ids/.metadata)
      - (compressed_ids, stats_dict) tuple (e.g. EnhancedCompressor.compress)
    """
    if isinstance(raw, tuple) and len(raw) == 2:
        compressed_ids, stats = raw
        return SampleResult(compressed_ids=list(compressed_ids), metadata=dict(stats or {}))
    compressed_ids = getattr(raw, 'compressed_ids')
    metadata = getattr(raw, 'metadata', {}) or {}
    return SampleResult(compressed_ids=list(compressed_ids), metadata=dict(metadata))


def _compress(compressor: Any, input_ids: List[int], target_ratio: float) -> SampleResult:
    """Call .compress() the right way for whichever compressor convention this is."""
    if hasattr(compressor, 'config'):
        compressor.config.target_ratio = target_ratio
        raw = compressor.compress(input_ids)
    else:
        raw = compressor.compress(input_ids, target_ratio=target_ratio)
    return _normalize_compress_output(raw)


# ============================================================================
# Proxy quality/efficiency scoring
# ============================================================================

def _decode_tokens(tokenizer, token_ids: List[int]) -> List[str]:
    tokens = []
    for tid in token_ids:
        t = tokenizer.decode([tid])
        t = t.replace('▁', ' ').replace('Ġ', ' ').strip()
        tokens.append(t)
    return tokens


def _approx_tone_preservation_rate(
    orig_tokens: List[str],
    comp_tokens: List[str],
    tone_analyzer: VietnameseToneAnalyzer,
) -> float:
    """
    Fallback used when a compressor doesn't report an exact, index-based
    tone_preservation_rate in its metadata (e.g. EnhancedCompressor).

    Multiset-based approximation: how many tone-bearing tokens from the
    original survive (by token identity, not position) in the compressed
    output. Exact for compressors that already track retained indices —
    those should just put the real rate in metadata instead of relying
    on this.
    """
    orig_infos = tone_analyzer.analyze_tokens(orig_tokens)
    tone_bearing_orig = [t for t, info in zip(orig_tokens, orig_infos) if info.tones_present]
    if not tone_bearing_orig:
        return 1.0

    comp_infos = tone_analyzer.analyze_tokens(comp_tokens)
    remaining = Counter(t for t, info in zip(comp_tokens, comp_infos) if info.tones_present)

    preserved = 0
    for t in tone_bearing_orig:
        if remaining.get(t, 0) > 0:
            preserved += 1
            remaining[t] -= 1
    return preserved / len(tone_bearing_orig)


def _content_word_retention(
    orig_tokens: List[str],
    comp_tokens: List[str],
    morph_analyzer: MorphologyAnalyzer,
) -> float:
    """Fraction of original content-word count still present after compression."""
    orig_content = sum(
        1 for info in morph_analyzer.classify_batch(orig_tokens)
        if info.word_class == WordClass.CONTENT
    )
    if orig_content == 0:
        return 1.0
    comp_content = sum(
        1 for info in morph_analyzer.classify_batch(comp_tokens)
        if info.word_class == WordClass.CONTENT
    )
    return min(1.0, comp_content / orig_content)


class CalibrationObjective:
    """
    Scores a compressor factory (params -> compressor) against a fixed
    validation set of texts, using the proxy quality/efficiency objective
    described in this module's docstring.
    """

    def __init__(
        self,
        tokenizer,
        texts: Sequence[str],
        tone_analyzer: Optional[VietnameseToneAnalyzer] = None,
        morph_analyzer: Optional[MorphologyAnalyzer] = None,
        target_ratio: float = 4.0,
    ):
        if not texts:
            raise ValueError("CalibrationObjective needs at least one validation text")
        self.tokenizer = tokenizer
        self.tone_analyzer = tone_analyzer or get_tone_analyzer()
        self.morph_analyzer = morph_analyzer or get_morphology_analyzer()
        self.target_ratio = target_ratio
        self._input_ids_cache = [
            tokenizer.encode(t, add_special_tokens=False) for t in texts
        ]

    def _score_one(self, input_ids: List[int], result: SampleResult) -> Dict[str, float]:
        n = len(input_ids)
        m = len(result.compressed_ids)
        token_savings_pct = ((n - m) / n) * 100 if n else 0.0
        efficiency = max(0.0, token_savings_pct / 100.0)

        orig_tokens = _decode_tokens(self.tokenizer, input_ids)
        comp_tokens = _decode_tokens(self.tokenizer, result.compressed_ids)

        if 'tone_preservation_rate' in result.metadata:
            tone_rate = result.metadata['tone_preservation_rate']
        else:
            tone_rate = _approx_tone_preservation_rate(
                orig_tokens, comp_tokens, self.tone_analyzer
            )

        content_retention = _content_word_retention(
            orig_tokens, comp_tokens, self.morph_analyzer
        )

        quality = 0.5 * tone_rate + 0.5 * content_retention
        denom = quality + efficiency
        harmonized = (2 * quality * efficiency / denom) if denom > 1e-8 else 0.0

        return {
            'quality': quality,
            'efficiency': efficiency,
            'harmonized_score': harmonized,
            'tone_preservation_rate': tone_rate,
            'content_word_retention': content_retention,
        }

    def evaluate(self, compressor_factory: Callable[[Dict[str, float]], Any],
                 params: Dict[str, float]) -> Dict[str, float]:
        """Build a compressor from `params` and score it on the whole validation set."""
        compressor = compressor_factory(params)
        per_sample = []
        for input_ids in self._input_ids_cache:
            result = _compress(compressor, input_ids, self.target_ratio)
            per_sample.append(self._score_one(input_ids, result))

        agg = {}
        for key in ('quality', 'efficiency', 'harmonized_score',
                    'tone_preservation_rate', 'content_word_retention'):
            values = [s[key] for s in per_sample]
            agg[key] = sum(values) / len(values) if values else 0.0
        agg['n_samples'] = len(per_sample)
        return agg


# ============================================================================
# Search strategies
# ============================================================================

def grid_search(
    objective: CalibrationObjective,
    compressor_factory: Callable[[Dict[str, float]], Any],
    param_candidates: Dict[str, List[float]],
    max_combinations: int = 200,
    seed: int = 0,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Exhaustive Cartesian-product search. Only practical for a handful of
    parameters at a time (dimensionality explodes fast) — for the full
    B1 parameter set, prefer `coordinate_ascent`.

    If the full grid exceeds `max_combinations`, a random subsample of that
    size is evaluated instead (deterministic given `seed`), rather than
    silently taking forever.
    """
    names = list(param_candidates.keys())
    all_combos = list(itertools.product(*(param_candidates[n] for n in names)))

    if len(all_combos) > max_combinations:
        rng = random.Random(seed)
        all_combos = rng.sample(all_combos, max_combinations)

    history: List[Dict[str, Any]] = []
    best_params: Optional[Dict[str, float]] = None
    best_score = float('-inf')

    for combo in all_combos:
        params = dict(zip(names, combo))
        metrics = objective.evaluate(compressor_factory, params)
        history.append({'params': dict(params), 'metrics': metrics})
        if metrics['harmonized_score'] > best_score:
            best_score = metrics['harmonized_score']
            best_params = params

    if best_params is None:
        best_params = _default_initial(param_candidates)
    return best_params, history


def coordinate_ascent(
    objective: CalibrationObjective,
    compressor_factory: Callable[[Dict[str, float]], Any],
    param_candidates: Dict[str, List[float]],
    initial_params: Optional[Dict[str, float]] = None,
    rounds: int = 2,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Cheap alternative to grid_search: sweep one parameter at a time (holding
    all others fixed at the current best), keep the best value, move to the
    next parameter. Repeat for `rounds` full passes over all parameters.

    Cost: O(rounds * sum(len(candidates))) evaluations instead of the
    Cartesian product — e.g. ~48 evaluations for the 8-parameter B1 space
    with 2 rounds, vs 3^8 ≈ 6561 for a full grid.
    """
    current = dict(initial_params or _default_initial(param_candidates))
    history: List[Dict[str, Any]] = []

    best_metrics = objective.evaluate(compressor_factory, current)
    history.append({'params': dict(current), 'metrics': best_metrics})
    best_score = best_metrics['harmonized_score']

    for _ in range(rounds):
        improved = False
        for name, candidates in param_candidates.items():
            for value in candidates:
                if value == current[name]:
                    continue
                trial = dict(current)
                trial[name] = value
                metrics = objective.evaluate(compressor_factory, trial)
                history.append({'params': dict(trial), 'metrics': metrics})
                if metrics['harmonized_score'] > best_score:
                    best_score = metrics['harmonized_score']
                    current = trial
                    improved = True
        if not improved:
            break

    return current, history


# ============================================================================
# Convenience wrappers for the two concrete parameter groups (B1)
# ============================================================================

def calibrate_combined_compressor(
    tokenizer,
    model,
    texts: Sequence[str],
    config,
    device: str = 'cuda',
    param_candidates: Optional[Dict[str, List[float]]] = None,
    initial_params: Optional[Dict[str, float]] = None,
    rounds: int = 2,
    target_ratio: float = 4.0,
    strategy: str = 'coordinate_ascent',
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Calibrate CombinedCompressor's alpha/beta/gamma/tone_weight and
    per-class morphology multipliers against a validation text set.

    Requires torch + the model this compressor scores tokens with (this
    function only imports CombinedCompressor lazily, so importing this
    module itself has no torch dependency).
    """
    from ..compressors.tone_aware import CombinedCompressor

    param_candidates = param_candidates or DEFAULT_PARAM_CANDIDATES
    objective = CalibrationObjective(tokenizer, texts, target_ratio=target_ratio)

    def factory(params: Dict[str, float]):
        return CombinedCompressor(
            tokenizer, model, config, device,
            alpha=params['alpha'], beta=params['beta'], gamma=params['gamma'],
            tone_weight=params['tone_weight'],
            f_func=params['f_func'], f_content=params['f_content'],
            f_redup=params['f_redup'], f_compound=params['f_compound'],
        )

    if strategy == 'grid_search':
        return grid_search(objective, factory, param_candidates)
    return coordinate_ascent(objective, factory, param_candidates, initial_params, rounds)


def calibrate_score_weights(
    tokenizer,
    scorer,
    texts: Sequence[str],
    weight_candidates: Optional[Dict[str, List[float]]] = None,
    rounds: int = 2,
    target_ratio: float = 4.0,
    strategy: str = 'coordinate_ascent',
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Calibrate EnhancedCompressor's ScoreWeights (w_ppl/w_tone/w_morph)
    against a validation text set, using a fixed TinyModelScorer.

    `weight_candidates` values do not need to sum to 1.0 per-combo —
    ScoreWeights normalizes them internally.
    """
    from ..compressors.external_scorer import EnhancedCompressor, ScoreWeights

    weight_candidates = weight_candidates or DEFAULT_SCORE_WEIGHT_CANDIDATES
    objective = CalibrationObjective(tokenizer, texts, target_ratio=target_ratio)

    def factory(params: Dict[str, float]):
        weights = ScoreWeights(
            perplexity=params['perplexity'],
            tone=params['tone'],
            morphology=params['morphology'],
        )
        return EnhancedCompressor(tokenizer, scorer, weights=weights)

    if strategy == 'grid_search':
        return grid_search(objective, factory, weight_candidates)
    return coordinate_ascent(objective, factory, weight_candidates, rounds=rounds)
