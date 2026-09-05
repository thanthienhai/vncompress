"""
Evaluation — VCC-Bench, metrics, significance testing, method taxonomy.
=========================================================================
Everything downstream of "compressed context" lives in this one file:
CompressionMetrics, ROUGE-L/BLEU/BERTScore/Exact-Match/token-F1/needle-recall,
the VCC-Bench task runner, paired bootstrap significance testing, and the
baseline/proposed/ablation method taxonomy used by result-summarizing scripts.

VCC-Bench tasks: Long-Document QA, Multi-turn Conversation, Needle-in-Haystack,
Agent Tool-Calling, Cross-lingual Compression.

Vietnamese-specific metrics preserve tone marks throughout: the default
`rouge_score`/token-overlap tokenizers strip everything outside [a-z0-9],
which silently collapses 'bàn'/'bán'/'bạn' into the same string -- see
VietnameseRougeTokenizer / _normalize_answer below.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import torch
except ImportError:
    torch = None


# ============================================================================
# Metrics
# ============================================================================


@dataclass
class CompressionMetrics:
    """Per-sample compression + quality metrics."""

    compression_ratio: float = 1.0
    token_savings_pct: float = 0.0
    processing_time_ms: float = 0.0

    rouge_l_f1: Optional[float] = None
    rouge_l_precision: Optional[float] = None
    rouge_l_recall: Optional[float] = None
    bleu_score: Optional[float] = None
    bert_score_f1: Optional[float] = None
    exact_match: bool = False
    token_f1: Optional[float] = None

    tone_preservation_rate: Optional[float] = None
    function_word_keep_ratio: Optional[float] = None
    content_word_keep_ratio: Optional[float] = None

    prefill_time_ms: Optional[float] = None
    decode_time_ms: Optional[float] = None
    memory_saved_bytes: int = 0

    quality_score: float = 0.0
    efficiency_score: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        meta = d.pop('metadata')
        d.update(meta)
        return d


class VietnameseRougeTokenizer:
    """Tokenizer for `rouge_score` that does not destroy Vietnamese.

    `rouge_score`'s default tokenizer lowercases and replaces every character
    outside [a-z0-9] with a space. Every Vietnamese tone-marked vowel lives in
    Latin Extended Additional (U+1EA0-U+1EF9) and is therefore deleted, which
    silently collapses distinct words ('bàn'/'bán'/'bạn' -> the same tokens).
    For a project whose entire claim is tone-aware compression, scoring with a
    metric that cannot see tone marks would invalidate the numbers it produces.

    Tokenizes to whitespace-separated syllables (punctuation stripped),
    matching `compute_token_f1` so both metrics count the same units.
    """

    def tokenize(self, text: str) -> List[str]:
        return _normalize_answer(text)


def _mean_or_none(values):
    """Mean of non-None values, or None if there are none -- np.mean([]) is
    NaN, which serializes to invalid JSON (`NaN` is not a JSON token)."""
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def compute_rouge_l(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """ROUGE-L with tone-preserving syllable tokenization (see
    VietnameseRougeTokenizer). Falls back to character-overlap if
    `rouge_score` isn't installed."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=VietnameseRougeTokenizer())
        scores = {'rougeL_f1': [], 'rougeL_precision': [], 'rougeL_recall': []}
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)['rougeL']
            scores['rougeL_f1'].append(result.fmeasure)
            scores['rougeL_precision'].append(result.precision)
            scores['rougeL_recall'].append(result.recall)
        return {k: np.mean(v) for k, v in scores.items()}
    except ImportError:
        f1s, ps, rs = [], [], []
        for pred, ref in zip(predictions, references):
            pred_chars, ref_chars = set(pred), set(ref)
            if not pred_chars or not ref_chars:
                f1s.append(0.0); ps.append(0.0); rs.append(0.0)
                continue
            overlap = pred_chars & ref_chars
            p = len(overlap) / len(pred_chars)
            r = len(overlap) / len(ref_chars)
            f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
            ps.append(p); rs.append(r)
        return {'rougeL_f1': np.mean(f1s), 'rougeL_precision': np.mean(ps), 'rougeL_recall': np.mean(rs)}


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """BLEU, normalized to [0, 1]. Returns 0.0 if `sacrebleu` isn't installed."""
    try:
        from sacrebleu import corpus_bleu

        return corpus_bleu(predictions, [[r] for r in references]).score / 100.0
    except ImportError:
        return 0.0


def compute_bert_score(predictions: List[str], references: List[str], model_name: str = 'bert-base-multilingual-cased') -> float:
    """BERTScore F1. Returns 0.0 if `bert_score` isn't installed."""
    try:
        from bert_score import score

        _, _, f1 = score(predictions, references, model_type=model_name, verbose=False)
        return f1.mean().item()
    except ImportError:
        return 0.0


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
    return matches / len(predictions) if predictions else 0.0


def _normalize_answer(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace (SQuAD-style),
    keeping tone marks intact -- syllable-level overlap for Vietnamese."""
    import re
    import unicodedata

    text = unicodedata.normalize('NFC', text).lower()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return text.split()


def compute_needle_recall(predictions: List[str], references: List[str]) -> float:
    """Mean needle-retrieval recall (LongBench/RULER style): fraction of the
    reference's syllables recovered in the prediction. Deliberately recall,
    not F1 -- surrounding filler in the answer shouldn't be penalized, only
    whether the planted "needle" survived."""
    if not predictions:
        return 0.0
    scores = []
    for pred, ref in zip(predictions, references):
        p_tokens, r_tokens = _normalize_answer(pred), _normalize_answer(ref)
        if not r_tokens:
            scores.append(float(not p_tokens))
            continue
        common = Counter(p_tokens) & Counter(r_tokens)
        scores.append(sum(common.values()) / len(r_tokens))
    return float(np.mean(scores))


def compute_token_f1(predictions: List[str], references: List[str]) -> float:
    """Mean SQuAD-style token-overlap F1 -- the standard companion metric to
    exact match for generative/extractive QA answers."""
    if not predictions:
        return 0.0
    scores = []
    for pred, ref in zip(predictions, references):
        p_tokens, r_tokens = _normalize_answer(pred), _normalize_answer(ref)
        if not p_tokens or not r_tokens:
            scores.append(float(p_tokens == r_tokens))
            continue
        common = Counter(p_tokens) & Counter(r_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            scores.append(0.0)
            continue
        precision, recall = overlap / len(p_tokens), overlap / len(r_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


# ============================================================================
# VCC-Bench
# ============================================================================


@dataclass
class VCCBenchConfig:
    tasks: List[str] = field(default_factory=lambda: [
        'long_document_qa', 'multi_turn_conversation', 'needle_in_haystack',
        'agent_tool_calling', 'cross_lingual',
    ])
    methods: List[str] = field(default_factory=lambda: ['none', 'random', 'llmlingua', 'lacc'])
    compression_ratios: List[float] = field(default_factory=lambda: [2.0, 4.0, 8.0])
    max_new_tokens: int = 256
    temperature: float = 0.0
    do_sample: bool = False
    output_dir: str = './results'
    save_predictions: bool = True
    device: str = 'cuda'


@dataclass
class VCCBenchSample:
    task: str
    context: str
    query: str
    reference_answer: str
    context_length: int  # in tokens
    metadata: Dict[str, Any] = field(default_factory=dict)


class VCCBench:
    """Vietnamese Context Compression Benchmark -- evaluates compression
    methods across multiple tasks with Vietnamese-specific metrics."""

    def __init__(self, config: Optional[VCCBenchConfig] = None):
        self.config = config or VCCBenchConfig()
        self.samples: Dict[str, List[VCCBenchSample]] = defaultdict(list)
        os.makedirs(self.config.output_dir, exist_ok=True)

    def add_samples(self, samples: List[VCCBenchSample]):
        for sample in samples:
            self.samples[sample.task].append(sample)

    def add_sample(self, sample: VCCBenchSample):
        self.samples[sample.task].append(sample)

    @classmethod
    def load_from_json(cls, json_path: str, config: Optional[VCCBenchConfig] = None) -> 'VCCBench':
        """Load a VCC-Bench dataset: {"metadata": {...}, "samples": [...]}."""
        bench = cls(config)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        samples_raw = data.get('samples', [])
        if not samples_raw:
            print(f"[WARN] No samples found in {json_path}")
            return bench

        samples = [
            VCCBenchSample(
                task=raw.get('task', 'unknown'),
                context=raw.get('context', ''),
                query=raw.get('query', ''),
                reference_answer=raw.get('reference_answer', ''),
                context_length=raw.get('char_length', len(raw.get('context', ''))),
                metadata={k: v for k, v in raw.items()
                          if k not in ('task', 'context', 'query', 'reference_answer', 'char_length')},
            )
            for raw in samples_raw
        ]
        bench.add_samples(samples)

        meta = data.get('metadata', {})
        print(f"Loaded VCC-Bench: {meta.get('name', 'unknown')} v{meta.get('version', '?')}")
        task_counts: Dict[str, int] = {}
        for s in samples:
            task_counts[s.task] = task_counts.get(s.task, 0) + 1
        for task, count in sorted(task_counts.items()):
            print(f"  {task}: {count} samples")
        print(f"  Total: {len(samples)} samples")
        return bench

    @property
    def total_samples(self) -> int:
        return sum(len(v) for v in self.samples.values())

    def evaluate(self, compressor_fn: Callable[[str], Any], model, tokenizer, generation_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """Run every configured method on every configured task/ratio."""
        all_results: Dict[str, Any] = {}
        for method_name in self.config.methods:
            print(f"\n{'=' * 60}\nEvaluating: {method_name}\n{'=' * 60}")
            compressor = compressor_fn(method_name)
            method_results: Dict[str, Any] = {}
            for task_name, samples in self.samples.items():
                if task_name not in self.config.tasks or not samples:
                    continue
                task_results = {}
                for ratio in self.config.compression_ratios:
                    print(f"  Task: {task_name}, Ratio: {ratio}x")
                    metrics_list = self._evaluate_task(compressor, model, tokenizer, samples, ratio, task_name, generation_fn)
                    task_results[f'ratio_{ratio}'] = self._aggregate_metrics(metrics_list)
                    if self.config.save_predictions:
                        self._save_results(method_name, task_name, ratio, metrics_list)
                method_results[task_name] = task_results
            all_results[method_name] = method_results
        all_results['summary'] = self._compute_summary(all_results)
        return all_results

    def _evaluate_task(self, compressor, model, tokenizer, samples, ratio, task_name, generation_fn) -> List[CompressionMetrics]:
        metrics_list = []
        compressor.config.target_ratio = ratio
        for sample in tqdm(samples, desc=f"  {task_name} @ {ratio}x"):
            input_ids = tokenizer.encode(sample.context, add_special_tokens=False)
            start_time = time.time()
            # `query`: without it, LACC's query-relevance boost and
            # SelectiveContext's embedding path never run in any benchmark.
            result = compressor.compress(input_ids, query=sample.query)
            comp_time = (time.time() - start_time) * 1000

            metric = CompressionMetrics(
                compression_ratio=result.compression_ratio,
                token_savings_pct=result.token_savings_pct,
                processing_time_ms=comp_time,
            )
            if 'tone_preservation_rate' in result.metadata:
                metric.tone_preservation_rate = result.metadata['tone_preservation_rate']

            output = (generation_fn or self._default_generate)(
                model, tokenizer, compressed_ids=result.compressed_ids, query=sample.query,
                max_new_tokens=self.config.max_new_tokens, temperature=self.config.temperature,
            ) if generation_fn else self._default_generate(model, tokenizer, result.compressed_ids, sample.query)

            if output:
                rouge = compute_rouge_l([output], [sample.reference_answer])
                metric.rouge_l_f1 = rouge['rougeL_f1']
                metric.rouge_l_precision = rouge['rougeL_precision']
                metric.rouge_l_recall = rouge['rougeL_recall']
                metric.bleu_score = compute_bleu([output], [sample.reference_answer])
                metric.exact_match = output.strip().lower() == sample.reference_answer.strip().lower()
                metric.token_f1 = compute_token_f1([output], [sample.reference_answer])

            metric.quality_score = (
                (metric.rouge_l_f1 or 0) * 0.4 + (metric.bleu_score or 0) * 0.2 + float(metric.exact_match) * 0.4
            )
            metric.efficiency_score = metric.token_savings_pct / 100.0
            metrics_list.append(metric)
        return metrics_list

    def _default_generate(self, model, tokenizer, compressed_ids: List[int], query: str) -> Optional[str]:
        try:
            query_ids = tokenizer.encode(query, add_special_tokens=False)
            full_input = compressed_ids + query_ids
            input_tensor = torch.tensor([full_input]).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    input_tensor, max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature, do_sample=self.config.do_sample,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            return tokenizer.decode(outputs[0][len(full_input):], skip_special_tokens=True)
        except Exception:
            return None

    def _aggregate_metrics(self, metrics_list: List[CompressionMetrics]) -> Dict[str, float]:
        if not metrics_list:
            return {}
        agg = {
            'mean_compression_ratio': np.mean([m.compression_ratio for m in metrics_list]),
            'mean_token_savings_pct': np.mean([m.token_savings_pct for m in metrics_list]),
            'mean_processing_time_ms': np.mean([m.processing_time_ms for m in metrics_list]),
            'mean_rouge_l_f1': _mean_or_none([m.rouge_l_f1 for m in metrics_list]),
            'mean_bleu': _mean_or_none([m.bleu_score for m in metrics_list]),
            'exact_match_rate': np.mean([float(m.exact_match) for m in metrics_list]),
            'mean_token_f1': _mean_or_none([m.token_f1 for m in metrics_list]),
            'num_generated': sum(1 for m in metrics_list if m.rouge_l_f1 is not None),
            'mean_quality_score': np.mean([m.quality_score for m in metrics_list]),
            'mean_efficiency_score': np.mean([m.efficiency_score for m in metrics_list]),
            'num_samples': len(metrics_list),
        }
        tone_rates = [m.tone_preservation_rate for m in metrics_list if m.tone_preservation_rate is not None]
        if tone_rates:
            agg['mean_tone_preservation_rate'] = np.mean(tone_rates)
        return agg

    def _compute_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Sample-count-weighted average quality/efficiency per method, and
        their harmonic mean (clamped to >= 0 -- token_savings_pct can go
        negative, which would otherwise blow up 2*q*e/(q+e))."""
        summary = {}
        for method_name, method_results in all_results.items():
            if method_name == 'summary':
                continue
            q_sum = e_sum = n_sum = 0.0
            for task_results in method_results.values():
                for metrics in task_results.values():
                    n = metrics.get('num_samples', 0) or 0
                    if not n:
                        continue
                    n_sum += n
                    q_sum += (metrics.get('mean_quality_score') or 0.0) * n
                    e_sum += (metrics.get('mean_efficiency_score') or 0.0) * n
            mq = max(q_sum / n_sum if n_sum else 0.0, 0.0)
            me = max(e_sum / n_sum if n_sum else 0.0, 0.0)
            summary[method_name] = {
                'avg_quality': mq, 'avg_efficiency': me, 'total_samples': int(n_sum),
                'harmonized_score': (2 * mq * me / (mq + me)) if (mq + me) > 0 else 0.0,
            }
        return summary

    def _save_results(self, method_name: str, task_name: str, ratio: float, metrics_list: List[CompressionMetrics]):
        path = os.path.join(self.config.output_dir, f"{method_name}_{task_name}_ratio{ratio:.1f}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([m.to_dict() for m in metrics_list], f, indent=2, ensure_ascii=False)

    def print_summary(self, results: Dict[str, Any]):
        summary = results.get('summary', {})
        print("\n" + "=" * 80 + "\nVCC-BENCH SUMMARY\n" + "=" * 80)
        print(f"{'Method':<25} {'Quality':>10} {'Efficiency':>10} {'Harmonized':>10}")
        print("-" * 60)
        for method, scores in sorted(summary.items(), key=lambda x: x[1].get('harmonized_score', 0), reverse=True):
            print(f"{method:<25} {scores.get('avg_quality', 0):>10.3f} "
                  f"{scores.get('avg_efficiency', 0):>10.3f} {scores.get('harmonized_score', 0):>10.3f}")
        print("=" * 80 + "\nHarmonized score = 2 x Q x E / (Q + E)  [higher is better]")

    def generate_report(self, results: Dict[str, Any]) -> str:
        summary = results.get('summary', {})
        lines = [
            "# VCC-Bench Evaluation Report\n",
            f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
            f"Samples: {self.total_samples}",
            f"Tasks: {', '.join(self.config.tasks)}",
            f"Methods: {', '.join(self.config.methods)}",
            f"Ratios: {', '.join(f'{r}x' for r in self.config.compression_ratios)}",
            "", "## Overall Results\n",
            "| Method | Quality | Efficiency | Harmonized |",
            "|--------|---------|------------|------------|",
        ]
        for method, scores in sorted(summary.items(), key=lambda x: x[1].get('harmonized_score', 0), reverse=True):
            lines.append(f"| {method} | {scores.get('avg_quality', 0):.3f} | "
                         f"{scores.get('avg_efficiency', 0):.3f} | {scores.get('harmonized_score', 0):.3f} |")
        return '\n'.join(lines)


def evaluate_compression(
    compressor, model, tokenizer, input_text: str, query: str, reference: str,
    ratio: float = 4.0, max_new_tokens: int = 256,
) -> CompressionMetrics:
    """Quick single-sample evaluation of a compression method (used by
    `benchmark.py --demo`)."""
    compressor.config.target_ratio = ratio
    start = time.time()
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    result = compressor.compress(input_ids, query=query)
    comp_time = (time.time() - start) * 1000

    metric = CompressionMetrics(
        compression_ratio=result.compression_ratio, token_savings_pct=result.token_savings_pct,
        processing_time_ms=comp_time,
    )
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    full_input = result.compressed_ids + query_ids
    input_tensor = torch.tensor([full_input]).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_tensor, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    output_text = tokenizer.decode(outputs[0][len(full_input):], skip_special_tokens=True)

    rouge = compute_rouge_l([output_text], [reference])
    metric.rouge_l_f1 = rouge['rougeL_f1']
    metric.bleu_score = compute_bleu([output_text], [reference])
    metric.exact_match = output_text.strip().lower() == reference.strip().lower()
    metric.quality_score = (metric.rouge_l_f1 or 0) * 0.4 + (metric.bleu_score or 0) * 0.2 + float(metric.exact_match) * 0.4
    metric.efficiency_score = metric.token_savings_pct / 100.0
    return metric


# ============================================================================
# Paired significance testing
# ============================================================================


@dataclass
class PairedComparison:
    """Result of comparing arm A (treatment) vs arm B (control) on paired data."""

    n: int
    mean_a: float
    mean_b: float
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    win_rate: float
    significant: bool

    def to_dict(self) -> Dict:
        return asdict(self)


def paired_bootstrap_delta(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10000, ci: float = 0.95, seed: int = 42,
) -> Optional[PairedComparison]:
    """Bootstrap the paired mean difference mean(a - b). Resamples SAMPLE
    INDICES (not the two arms independently) -- controls for per-sample
    difficulty, far more powerful than an unpaired test. Returns None if
    fewer than 2 usable (non-None) pairs remain."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    av = np.array([p[0] for p in pairs], dtype=float)
    bv = np.array([p[1] for p in pairs], dtype=float)
    diff = av - bv
    n = len(diff)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)

    lo_q = (1 - ci) / 2
    ci_low, ci_high = np.quantile(boot_means, [lo_q, 1 - lo_q])
    p_value = min(1.0, 2 * min(float((boot_means <= 0).mean()), float((boot_means >= 0).mean())))

    return PairedComparison(
        n=n, mean_a=float(av.mean()), mean_b=float(bv.mean()), mean_delta=float(diff.mean()),
        ci_low=float(ci_low), ci_high=float(ci_high), p_value=p_value,
        win_rate=float((diff >= 0).mean()), significant=bool(ci_low > 0 or ci_high < 0),
    )


# ============================================================================
# Method taxonomy (baseline / proposed / ablation)
# ============================================================================
#
# Two namespaces, since benchmark.py and its ablation mode name methods
# differently:
#   - "registry": vncompress.compression.METHODS keys (--methods flag).
#   - "ablation": the isolated-signal arms (ppl_only/tone_only/morph_only/lacc).


class MethodCategory(str, Enum):
    BASELINE = "baseline"
    PROPOSED = "proposed"
    ABLATION = "ablation"


REGISTRY_METHOD_CATEGORY: Dict[str, MethodCategory] = {
    "none": MethodCategory.BASELINE,
    "random": MethodCategory.BASELINE,
    "llmlingua": MethodCategory.BASELINE,
    "snapkv": MethodCategory.BASELINE,
    "selective": MethodCategory.BASELINE,
    "lacc": MethodCategory.PROPOSED,
}

ABLATION_ARM_CATEGORY: Dict[str, MethodCategory] = {
    "ppl_only": MethodCategory.ABLATION,
    "tone_only": MethodCategory.ABLATION,
    "morph_only": MethodCategory.ABLATION,
    "lacc": MethodCategory.ABLATION,
}


def categorize(method_name: str, context: str = "registry") -> MethodCategory:
    """Look up a method's category. `context='registry'` (benchmark.py's main
    results table) or `'ablation'` (its ablation arms) -- disambiguates
    'lacc', which is `proposed` in the registry table and `ablation` in the
    ablation table (there it plays the role of "the full method as a point of
    comparison for the isolated single-signal arms").

    Raises ValueError for an unclassified method name -- a mislabeled
    proposed method reported as a baseline (or vice versa) is worse than a
    loud failure.
    """
    if context not in ("registry", "ablation"):
        raise ValueError(f"Unknown context: {context!r}. Expected 'registry' or 'ablation'.")
    table = REGISTRY_METHOD_CATEGORY if context == "registry" else ABLATION_ARM_CATEGORY
    if method_name not in table:
        raise ValueError(
            f"Method {method_name!r} is not classified in evaluation.py's method taxonomy "
            f"(context={context!r}). Add it to REGISTRY_METHOD_CATEGORY or ABLATION_ARM_CATEGORY "
            "before including it in a results table."
        )
    return table[method_name]
