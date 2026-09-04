"""
Evaluation Metrics & VCC-Bench
==============================
Vietnamese Context Compression Benchmark (VCC-Bench) and evaluation framework.

Metrics:
  1. Compression Ratio (CR) = N_original / N_compressed
  2. Token Savings % = (N_original - N_compressed) / N_original × 100
  3. ROUGE-L (F1, Precision, Recall) — for summarization tasks
  4. BLEU — for generation tasks
  5. Exact Match (EM) — for QA tasks
  6. BERTScore — semantic similarity
  7. Needle Retrieval Accuracy — for needle-in-haystack
  8. Tone Preservation Rate (TPR) — novel metric for Vietnamese
  9. Function Word Compression Ratio — morphology-specific
  10. Perplexity Change (ΔPPL) — quality impact

VCC-Bench Tasks:
  - Task 1: Long-Document QA (Vietnamese legal texts, news)
  - Task 2: Multi-turn Conversation Summarization
  - Task 3: Needle-in-Haystack (Vietnamese version)
  - Task 4: Agent Tool-Calling (Vietnamese)
  - Task 5: Cross-lingual Compression Comparison

Reference:
  - LongBench (2023): English long-context benchmark
  - RULER (2024): Needle-in-haystack benchmark
  - arxiv:2606.03618 "Cross-Lingual Token Arbitrage"
"""

from __future__ import annotations
import time
import json
import os
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


# ============================================================================
# Metrics
# ============================================================================

@dataclass
class CompressionMetrics:
    """Collection of compression evaluation metrics."""
    compression_ratio: float = 1.0
    token_savings_pct: float = 0.0
    processing_time_ms: float = 0.0
    
    # Quality metrics (filled after generation comparison)
    rouge_l_f1: Optional[float] = None
    rouge_l_precision: Optional[float] = None
    rouge_l_recall: Optional[float] = None
    bleu_score: Optional[float] = None
    bert_score_f1: Optional[float] = None
    exact_match: bool = False
    token_f1: Optional[float] = None
    
    # Vietnamese-specific metrics
    tone_preservation_rate: Optional[float] = None
    function_word_keep_ratio: Optional[float] = None
    content_word_keep_ratio: Optional[float] = None
    
    # Efficiency
    prefill_time_ms: Optional[float] = None
    decode_time_ms: Optional[float] = None
    memory_saved_bytes: int = 0
    
    # Combined score (weighted)
    quality_score: float = 0.0
    efficiency_score: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            'compression_ratio': self.compression_ratio,
            'token_savings_pct': self.token_savings_pct,
            'processing_time_ms': self.processing_time_ms,
            'rouge_l_f1': self.rouge_l_f1,
            'rouge_l_precision': self.rouge_l_precision,
            'rouge_l_recall': self.rouge_l_recall,
            'bleu_score': self.bleu_score,
            'bert_score_f1': self.bert_score_f1,
            'exact_match': self.exact_match,
            'token_f1': self.token_f1,
            'tone_preservation_rate': self.tone_preservation_rate,
            'function_word_keep_ratio': self.function_word_keep_ratio,
            'content_word_keep_ratio': self.content_word_keep_ratio,
            'quality_score': self.quality_score,
            'efficiency_score': self.efficiency_score,
            **self.metadata,
        }


class VietnameseRougeTokenizer:
    """Tokenizer for `rouge_score` that does not destroy Vietnamese.

    `rouge_score`'s default tokenizer lowercases and replaces every character
    outside [a-z0-9] with a space. Every Vietnamese vowel carrying a tone mark
    lives in Latin Extended Additional (U+1EA0-U+1EF9) and is therefore
    *deleted*, which silently collapses distinct words:

        'bàn' -> ['b', 'n']      'bán' -> ['b', 'n']      'bạn' -> ['b', 'n']
        'Hà Nội là thủ đô' -> ['h', 'n', 'i', 'l', 'th']

    So ROUGE-L scored "bạn của tôi" against "bàn của tôi" as a *perfect* 1.0.
    For a project whose entire claim is tone-aware compression, scoring with a
    metric that cannot see tone marks invalidates the numbers it produces.

    Tokenizes to whitespace-separated syllables (with punctuation stripped)
    rather than running a word segmenter. That matches `compute_token_f1`, so
    the two metrics count the same units, and it sidesteps the open dispute
    over whether Vietnamese NLP should segment syllables into words at all
    (cf. VnCoreNLP/PhoBERT vs. Nguyen et al., AAAI 2025). It is also
    deterministic and needs no extra dependency.

    Precedent: LongBench does exactly this for Chinese -- its `rouge_zh_score`
    runs jieba before scoring instead of using the default tokenizer.
    """

    def tokenize(self, text: str) -> List[str]:
        return _normalize_answer(text)


def _mean_or_none(values):
    """Mean of the non-None values, or None if there are none.

    np.mean([]) is NaN, and NaN serialises to a bare `NaN` token that is not
    valid JSON -- which silently corrupted vcc_bench_results.json whenever
    every generation in a cell failed.
    """
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def compute_rouge_l(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-L scores.

    Uses syllable-level tokenization that preserves Vietnamese tone marks --
    see VietnameseRougeTokenizer for why the library default cannot be used.
    Falls back to character-level if rouge_score is unavailable.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False,
                                          tokenizer=VietnameseRougeTokenizer())
        
        scores = {'rougeL_f1': [], 'rougeL_precision': [], 'rougeL_recall': []}
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            scores['rougeL_f1'].append(result['rougeL'].fmeasure)
            scores['rougeL_precision'].append(result['rougeL'].precision)
            scores['rougeL_recall'].append(result['rougeL'].recall)
        
        return {
            'rougeL_f1': np.mean(scores['rougeL_f1']),
            'rougeL_precision': np.mean(scores['rougeL_precision']),
            'rougeL_recall': np.mean(scores['rougeL_recall']),
        }
    except ImportError:
        # Simple fallback: character-level overlap
        f1s, ps, rs = [], [], []
        for pred, ref in zip(predictions, references):
            pred_chars = set(pred)
            ref_chars = set(ref)
            if not pred_chars or not ref_chars:
                f1s.append(0.0); ps.append(0.0); rs.append(0.0)
                continue
            overlap = pred_chars & ref_chars
            p = len(overlap) / len(pred_chars) if pred_chars else 0
            r = len(overlap) / len(ref_chars) if ref_chars else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1s.append(f1); ps.append(p); rs.append(r)
        return {
            'rougeL_f1': np.mean(f1s),
            'rougeL_precision': np.mean(ps),
            'rougeL_recall': np.mean(rs),
        }


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score."""
    try:
        from sacrebleu import corpus_bleu
        # sacrebleu expects list of references per prediction
        refs = [[r] for r in references]
        bleu = corpus_bleu(predictions, refs)
        return bleu.score / 100.0  # Normalize to 0-1
    except ImportError:
        return 0.0


def compute_bert_score(
    predictions: List[str],
    references: List[str],
    model_name: str = 'bert-base-multilingual-cased',
) -> float:
    """Compute BERTScore F1."""
    try:
        from bert_score import score
        P, R, F1 = score(predictions, references, model_type=model_name, verbose=False)
        return F1.mean().item()
    except ImportError:
        return 0.0


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Compute exact match rate."""
    matches = sum(1 for p, r in zip(predictions, references)
                  if p.strip().lower() == r.strip().lower())
    return matches / len(predictions) if predictions else 0.0


def _normalize_answer(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace (SQuAD-style).

    Vietnamese is whitespace-tokenized into syllables rather than words, so
    this is syllable-level overlap. That is the same unit the compressor
    drops, and it avoids depending on a word segmenter at eval time.
    """
    import re
    import unicodedata

    text = unicodedata.normalize('NFC', text).lower()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return text.split()


def compute_needle_recall(predictions: List[str], references: List[str]) -> float:
    """Mean needle-retrieval recall for needle-in-haystack / synthetic tasks.

    LongBench and RULER score their synthetic retrieval tasks by whether the
    planted answer (the "needle") is recovered, not by ROUGE/BLEU overlap with
    a full sentence -- a compressor that keeps the needle but phrases the answer
    differently should still score 1.0. Here recall = fraction of the
    reference's syllables that appear in the prediction (so a fully recovered
    needle scores 1.0, a partial one scores in between), using the same
    tone-preserving normalization as ROUGE-L / token-F1.

    This is deliberately recall, not F1: for retrieval, surrounding filler in
    the answer should not be penalized -- only whether the needle survived.
    """
    if not predictions:
        return 0.0
    scores = []
    for pred, ref in zip(predictions, references):
        p_tokens, r_tokens = _normalize_answer(pred), _normalize_answer(ref)
        if not r_tokens:
            scores.append(float(not p_tokens))  # empty reference: match iff empty pred
            continue
        common = Counter(p_tokens) & Counter(r_tokens)
        scores.append(sum(common.values()) / len(r_tokens))
    return float(np.mean(scores))


def compute_token_f1(predictions: List[str], references: List[str]) -> float:
    """Mean SQuAD-style token-overlap F1.

    Exact match alone is far too harsh for generative answers -- a model that
    answers "Thủ tướng Phạm Văn Đồng" against the reference "Phạm Văn Đồng"
    scores 0 EM but should not score 0. Token F1 is the standard companion
    metric for extractive QA and is what compression papers report alongside
    EM when measuring how much a compressed context degrades answers.
    """
    if not predictions:
        return 0.0
    scores = []
    for pred, ref in zip(predictions, references):
        p_tokens, r_tokens = _normalize_answer(pred), _normalize_answer(ref)
        if not p_tokens or not r_tokens:
            # Both empty counts as agreement; one empty counts as total miss.
            scores.append(float(p_tokens == r_tokens))
            continue
        common = Counter(p_tokens) & Counter(r_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            scores.append(0.0)
            continue
        precision = overlap / len(p_tokens)
        recall = overlap / len(r_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


# ============================================================================
# VCC-Bench
# ============================================================================

@dataclass
class VCCBenchConfig:
    """Configuration for VCC-Bench evaluation."""
    # Tasks to evaluate
    tasks: List[str] = field(default_factory=lambda: [
        'long_document_qa',
        'multi_turn_conversation',
        'needle_in_haystack',
        'agent_tool_calling',
        'cross_lingual',
    ])
    
    # Compression methods to test
    methods: List[str] = field(default_factory=lambda: [
        'none',
        'random',
        'llmlingua',
        'snapkv',
        'tone_aware',
        'morphology_aware',
        'combined',
    ])
    
    # Compression ratios to test
    compression_ratios: List[float] = field(default_factory=lambda: [2.0, 4.0, 8.0])
    
    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.0
    do_sample: bool = False
    
    # Output
    output_dir: str = './results'
    save_predictions: bool = True
    
    # Device
    device: str = 'cuda'


@dataclass
class VCCBenchSample:
    """A single benchmark sample."""
    task: str
    context: str
    query: str
    reference_answer: str
    context_length: int  # In tokens
    metadata: Dict[str, Any] = field(default_factory=dict)


class VCCBench:
    """
    Vietnamese Context Compression Benchmark.
    
    Evaluates compression methods across multiple tasks with
    Vietnamese-specific metrics (tone preservation, morphology).
    
    Usage:
        bench = VCCBench(config)
        bench.add_samples(samples)
        results = bench.evaluate(create_compressor_fn)
        bench.print_summary(results)
    """
    
    def __init__(self, config: Optional[VCCBenchConfig] = None):
        self.config = config or VCCBenchConfig()
        self.samples: Dict[str, List[VCCBenchSample]] = defaultdict(list)
        
        # Ensure output dir exists
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def add_samples(self, samples: List[VCCBenchSample]):
        """Add benchmark samples."""
        for sample in samples:
            self.samples[sample.task].append(sample)
    
    def add_sample(self, sample: VCCBenchSample):
        """Add a single sample."""
        self.samples[sample.task].append(sample)

    @classmethod
    def load_from_json(cls, json_path: str, config: Optional[VCCBenchConfig] = None) -> 'VCCBench':
        """
        Load VCC-Bench dataset from a JSON file.

        The JSON file should contain:
          - metadata: dict with name, version, date, tasks, total_samples
          - samples: list of dicts with task, context, query, reference_answer, char_length

        Args:
            json_path: Path to the VCC-Bench JSON file.
            config: Optional VCCBenchConfig.

        Returns:
            VCCBench instance with loaded samples.
        """
        bench = cls(config)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples_raw = data.get('samples', [])
        if not samples_raw:
            print(f"[WARN] No samples found in {json_path}")
            return bench

        samples = []
        for raw in samples_raw:
            sample = VCCBenchSample(
                task=raw.get('task', 'unknown'),
                context=raw.get('context', ''),
                query=raw.get('query', ''),
                reference_answer=raw.get('reference_answer', ''),
                context_length=raw.get('char_length', len(raw.get('context', ''))),
                metadata={
                    k: v for k, v in raw.items()
                    if k not in ('task', 'context', 'query', 'reference_answer', 'char_length')
                },
            )
            samples.append(sample)

        bench.add_samples(samples)

        meta = data.get('metadata', {})
        print(f"Loaded VCC-Bench: {meta.get('name', 'unknown')} v{meta.get('version', '?')}")
        print(f"  Date: {meta.get('date', '?')}, License: {meta.get('license', '?')}")
        task_counts = {}
        for s in samples:
            task_counts[s.task] = task_counts.get(s.task, 0) + 1
        for task, count in sorted(task_counts.items()):
            print(f"  {task}: {count} samples")
        print(f"  Total: {len(samples)} samples")

        return bench
    
    @property
    def total_samples(self) -> int:
        return sum(len(v) for v in self.samples.values())
    
    def evaluate(
        self,
        compressor_fn: Callable[[str], Any],  # method_name → compressor
        model,
        tokenizer,
        generation_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate all compression methods on all tasks.
        
        Args:
            compressor_fn: Function that takes method name and returns compressor
            model: HuggingFace model for generation
            tokenizer: HuggingFace tokenizer
            generation_fn: Optional custom generation function
        
        Returns:
            Dict with results per method per task per compression ratio
        """
        all_results = {}
        
        for method_name in self.config.methods:
            print(f"\n{'='*60}")
            print(f"Evaluating: {method_name}")
            print(f"{'='*60}")
            
            compressor = compressor_fn(method_name)
            method_results = {}
            
            for task_name, samples in self.samples.items():
                if task_name not in self.config.tasks:
                    continue
                
                if not samples:
                    continue
                
                task_results = {}
                for ratio in self.config.compression_ratios:
                    print(f"  Task: {task_name}, Ratio: {ratio}x")
                    
                    metrics_list = self._evaluate_task(
                        compressor, model, tokenizer,
                        samples, ratio, task_name,
                        generation_fn,
                    )
                    
                    # Aggregate metrics
                    agg = self._aggregate_metrics(metrics_list)
                    task_results[f'ratio_{ratio}'] = agg
                    
                    # Save predictions
                    if self.config.save_predictions:
                        self._save_results(
                            method_name, task_name, ratio,
                            metrics_list
                        )
                
                method_results[task_name] = task_results
            
            all_results[method_name] = method_results
        
        # Compute summary
        summary = self._compute_summary(all_results)
        all_results['summary'] = summary
        
        return all_results
    
    def _evaluate_task(
        self,
        compressor,
        model,
        tokenizer,
        samples: List[VCCBenchSample],
        ratio: float,
        task_name: str,
        generation_fn: Optional[Callable] = None,
    ) -> List[CompressionMetrics]:
        """Evaluate a single task with a single compression method and ratio."""
        metrics_list = []
        
        # Set compression ratio
        compressor.config.target_ratio = ratio
        
        for sample in tqdm(samples, desc=f"  {task_name} @ {ratio}x"):
            # Encode context
            input_ids = tokenizer.encode(sample.context, add_special_tokens=False)
            
            # Compress
            start_time = time.time()
            # Pass the query: without it CombinedCompressor's query-aware
            # boost (LACC C2) and SelectiveContext's embedding path never ran
            # in ANY benchmark -- metadata['query_applied'] was always False.
            # Compressors that don't use it absorb it via **kwargs.
            result = compressor.compress(input_ids, query=sample.query)
            comp_time = (time.time() - start_time) * 1000
            
            metric = CompressionMetrics(
                compression_ratio=result.compression_ratio,
                token_savings_pct=result.token_savings_pct,
                processing_time_ms=comp_time,
            )
            
            # Extract compressor-specific metrics
            if 'tone_preservation_rate' in result.metadata:
                metric.tone_preservation_rate = result.metadata['tone_preservation_rate']
            
            # Generate with compressed context
            if generation_fn:
                output = generation_fn(
                    model, tokenizer,
                    compressed_ids=result.compressed_ids,
                    query=sample.query,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )
            else:
                output = self._default_generate(
                    model, tokenizer,
                    result.compressed_ids, sample.query,
                )
            
            # Compute quality metrics
            if output:
                rouge = compute_rouge_l([output], [sample.reference_answer])
                metric.rouge_l_f1 = rouge['rougeL_f1']
                metric.rouge_l_precision = rouge['rougeL_precision']
                metric.rouge_l_recall = rouge['rougeL_recall']
                
                metric.bleu_score = compute_bleu([output], [sample.reference_answer])
                metric.exact_match = output.strip().lower() == sample.reference_answer.strip().lower()
                metric.token_f1 = compute_token_f1([output], [sample.reference_answer])

            # Compute quality score (combination of metrics)
            metric.quality_score = (
                (metric.rouge_l_f1 or 0) * 0.4 +
                (metric.bleu_score or 0) * 0.2 +
                (float(metric.exact_match)) * 0.4
            )
            
            # Efficiency score
            metric.efficiency_score = metric.token_savings_pct / 100.0
            
            metrics_list.append(metric)
        
        return metrics_list
    
    def _default_generate(
        self,
        model,
        tokenizer,
        compressed_ids: List[int],
        query: str,
    ) -> Optional[str]:
        """Default generation using compressed context + query."""
        try:
            # Build prompt: compressed context + query
            query_ids = tokenizer.encode(query, add_special_tokens=False)
            full_input = compressed_ids + query_ids
            
            input_tensor = torch.tensor([full_input]).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_tensor,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            
            # Decode only new tokens
            generated = outputs[0][len(full_input):]
            return tokenizer.decode(generated, skip_special_tokens=True)
        except Exception:
            return None
    
    def _aggregate_metrics(
        self,
        metrics_list: List[CompressionMetrics],
    ) -> Dict[str, float]:
        """Aggregate metrics across samples."""
        if not metrics_list:
            return {}
        
        agg = {
            'mean_compression_ratio': np.mean([m.compression_ratio for m in metrics_list]),
            'mean_token_savings_pct': np.mean([m.token_savings_pct for m in metrics_list]),
            'mean_processing_time_ms': np.mean([m.processing_time_ms for m in metrics_list]),
            # `if xs else None`: when every generation failed these lists are
            # empty and np.mean([]) returns NaN, which json.dump writes as a
            # bare `NaN` token -- not valid JSON per RFC 8259, so strict
            # parsers reject vcc_bench_results.json.
            'mean_rouge_l_f1': _mean_or_none([m.rouge_l_f1 for m in metrics_list]),
            'mean_bleu': _mean_or_none([m.bleu_score for m in metrics_list]),
            'exact_match_rate': np.mean([float(m.exact_match) for m in metrics_list]),
            'mean_token_f1': _mean_or_none([m.token_f1 for m in metrics_list]),
            # Quality metrics above skip failed generations; quality_score and
            # exact_match_rate count them as 0. Report the denominator so the
            # two are not read as sharing one.
            'num_generated': sum(1 for m in metrics_list if m.rouge_l_f1 is not None),
            'mean_quality_score': np.mean([m.quality_score for m in metrics_list]),
            'mean_efficiency_score': np.mean([m.efficiency_score for m in metrics_list]),
            'num_samples': len(metrics_list),
        }
        
        # Vietnamese-specific
        tone_rates = [m.tone_preservation_rate for m in metrics_list if m.tone_preservation_rate is not None]
        if tone_rates:
            agg['mean_tone_preservation_rate'] = np.mean(tone_rates)
        
        return agg
    
    def _compute_summary(
        self,
        all_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute summary comparison across all methods."""
        summary = {}
        
        for method_name, method_results in all_results.items():
            if method_name == 'summary':
                continue
            
            # Average across all tasks and ratios
            # Weight each (task, ratio) cell by its sample count. Plain
            # np.mean over cells gave the 8-sample agent_tool_calling task the
            # same weight as the 160-sample long_document_qa task, so a method
            # strong on the smallest task climbed the ranking this table is
            # used to produce.
            q_sum = e_sum = n_sum = 0.0
            for task_name, task_results in method_results.items():
                for ratio_key, metrics in task_results.items():
                    n = metrics.get('num_samples', 0) or 0
                    if not n:
                        continue
                    n_sum += n
                    q_sum += (metrics.get('mean_quality_score') or 0.0) * n
                    e_sum += (metrics.get('mean_efficiency_score') or 0.0) * n

            mq = q_sum / n_sum if n_sum else 0.0
            me = e_sum / n_sum if n_sum else 0.0
            # Clamp before the harmonic mean: token_savings_pct goes negative
            # when a compressor returns more tokens than it received, and
            # 2*q*e/(q+e) then explodes -- measured -18,000,000 for q=0.3,
            # e=-0.3, which sorts to the top of a table labelled
            # "higher is better".
            mq, me = max(mq, 0.0), max(me, 0.0)
            summary[method_name] = {
                'avg_quality': mq,
                'avg_efficiency': me,
                'total_samples': int(n_sum),
                'harmonized_score': (2 * mq * me / (mq + me)) if (mq + me) > 0 else 0.0,
            }
        
        return summary
    
    def _save_results(
        self,
        method_name: str,
        task_name: str,
        ratio: float,
        metrics_list: List[CompressionMetrics],
    ):
        """Save detailed results to disk."""
        output_path = os.path.join(
            self.config.output_dir,
            f"{method_name}_{task_name}_ratio{ratio:.1f}.json"
        )
        
        results = [m.to_dict() for m in metrics_list]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary table."""
        summary = results.get('summary', {})
        
        print("\n" + "="*80)
        print("VCC-BENCH SUMMARY")
        print("="*80)
        print(f"{'Method':<25} {'Quality':>10} {'Efficiency':>10} {'Harmonized':>10}")
        print("-"*60)
        
        for method, scores in sorted(
            summary.items(),
            key=lambda x: x[1].get('harmonized_score', 0),
            reverse=True
        ):
            q = scores.get('avg_quality', 0)
            e = scores.get('avg_efficiency', 0)
            h = scores.get('harmonized_score', 0)
            print(f"{method:<25} {q:>10.3f} {e:>10.3f} {h:>10.3f}")
        
        print("="*80)
        print("Harmonized score = 2 × Q × E / (Q + E)  [higher is better]")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a markdown report from results."""
        summary = results.get('summary', {})
        
        lines = [
            "# VCC-Bench Evaluation Report\n",
            f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
            f"Samples: {self.total_samples}",
            f"Tasks: {', '.join(self.config.tasks)}",
            f"Methods: {', '.join(self.config.methods)}",
            f"Ratios: {', '.join(f'{r}x' for r in self.config.compression_ratios)}",
            "",
            "## Overall Results\n",
            "| Method | Quality | Efficiency | Harmonized |",
            "|--------|---------|------------|------------|",
        ]
        
        for method, scores in sorted(
            summary.items(),
            key=lambda x: x[1].get('harmonized_score', 0),
            reverse=True
        ):
            q = scores.get('avg_quality', 0)
            e = scores.get('avg_efficiency', 0)
            h = scores.get('harmonized_score', 0)
            lines.append(f"| {method} | {q:.3f} | {e:.3f} | {h:.3f} |")
        
        return '\n'.join(lines)


# ============================================================================
# Convenience function
# ============================================================================

def evaluate_compression(
    compressor,
    model,
    tokenizer,
    input_text: str,
    query: str,
    reference: str,
    ratio: float = 4.0,
    max_new_tokens: int = 256,
) -> CompressionMetrics:
    """
    Quick single-sample evaluation of a compression method.
    
    Args:
        compressor: BaseCompressor instance
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        input_text: Full input text
        query: Query/prompt
        reference: Reference answer
        ratio: Target compression ratio
        max_new_tokens: Max tokens to generate
    
    Returns:
        CompressionMetrics with all computed scores
    """
    compressor.config.target_ratio = ratio
    
    # Compress
    start = time.time()
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    result = compressor.compress(input_ids)
    comp_time = (time.time() - start) * 1000
    
    metric = CompressionMetrics(
        compression_ratio=result.compression_ratio,
        token_savings_pct=result.token_savings_pct,
        processing_time_ms=comp_time,
    )
    
    # Generate
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    full_input = result.compressed_ids + query_ids
    
    input_tensor = torch.tensor([full_input]).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    generated = outputs[0][len(full_input):]
    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    
    # Compute quality
    rouge = compute_rouge_l([output_text], [reference])
    metric.rouge_l_f1 = rouge['rougeL_f1']
    metric.bleu_score = compute_bleu([output_text], [reference])
    metric.exact_match = output_text.strip().lower() == reference.strip().lower()
    
    metric.quality_score = (
        (metric.rouge_l_f1 or 0) * 0.4 +
        (metric.bleu_score or 0) * 0.2 +
        (float(metric.exact_match)) * 0.4
    )
    metric.efficiency_score = metric.token_savings_pct / 100.0
    
    return metric
