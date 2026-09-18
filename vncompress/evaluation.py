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
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from statistics import NormalDist
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
    # Needle-retrieval recall -- only filled for the needle_in_haystack task
    # (see compute_needle_recall); None elsewhere so it never dilutes a mean.
    needle_recall: Optional[float] = None

    tone_preservation_rate: Optional[float] = None
    function_word_keep_ratio: Optional[float] = None
    content_word_keep_ratio: Optional[float] = None

    # Evidence/attribution metrics (docs/lacc_coling2027_tasklist.md G0). None
    # when they could not be computed (no identifiable evidence sentence, or
    # no NLI scorer configured) -- never 0.0, which would silently dilute the
    # mean of samples that were never actually scored.
    evidence_recall: Optional[float] = None
    source_span_recoverability: Optional[float] = None
    # Deterministic (no-NLI) counterpart to source_span_recoverability: the
    # fraction of compressed tokens that occur verbatim in the ORIGINAL context
    # (== 1 - LLMLingua-2's Variation Rate, arXiv:2403.12968 Eq. VR). For a
    # truly extractive compressor this is ~1.0; abstractive / translate-then-
    # compress arms score below 1.0 because they introduce tokens absent from
    # the source. Reported as the "extractive doesn't launder attribution"
    # control (cf. the emitted-grounded gap, arXiv:2609.14245 sec 3.2).
    span_recoverability_hard: Optional[float] = None
    unsupported_claim_rate: Optional[float] = None

    prefill_time_ms: Optional[float] = None
    decode_time_ms: Optional[float] = None
    # Wall-clock of the generation call alone (processing_time_ms is compression
    # alone) and CUDA peak allocation across compress+generate -- docs/eval_sweep_gate1.md
    # SS4 rule 4 requires re-measuring cost after the 2048/256 perplexity window.
    generation_time_ms: Optional[float] = None
    peak_vram_bytes: Optional[int] = None
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
# Evidence / attribution metrics (docs/lacc_coling2027_tasklist.md G0)
# ============================================================================
#
# The gold evidence sentence is "the sentence that contains the answer". In
# VCC-Bench v2 the reference_answer is a verbatim substring of the context for
# 100% of QA rows (the source qa.jsonl stores exact [start,end] char offsets;
# measured 6000/6000), so the FIRST choice is exact: the sentence that contains
# reference_answer as a normalized substring -- that is ground truth, not an
# approximation. Only when no sentence contains it verbatim (e.g. an answer
# stitched across a sentence boundary) do we fall back to the highest
# token-recall sentence, which stays a diagnostic. See docs/agent_research_findings.md.


def find_evidence_sentence(tokenizer, input_ids: Sequence[int], reference_answer: str) -> str:
    """Answer-bearing sentence from the ORIGINAL (uncompressed) context, over the
    same sentence spans (compression.sentence_spans) that E5/lacc_sentence select
    over. Prefers the sentence containing reference_answer verbatim (exact gold);
    otherwise the sentence with highest token overlap. Returns '' if
    reference_answer has no content tokens or no sentence overlaps it at all.
    """
    from .compression import sentence_spans

    ans_tokens = set(_normalize_answer(reference_answer))
    if not ans_tokens:
        return ''
    spans = sentence_spans(tokenizer, list(input_ids))
    decoded = [tokenizer.decode(input_ids[start:end], skip_special_tokens=True)
               for start, end in spans]
    # Exact gold: shortest sentence that contains the answer verbatim (NFC +
    # lowercase + whitespace-collapsed), which pins the evidence to one span
    # rather than the longest sentence that happens to include it.
    ans_norm = ' '.join(_normalize_answer(reference_answer))
    verbatim = [s for s in decoded if ans_norm and ans_norm in ' '.join(_normalize_answer(s))]
    if verbatim:
        return min(verbatim, key=len)
    best_sent, best_overlap = '', 0
    for sent_text in decoded:
        overlap = len(set(_normalize_answer(sent_text)) & ans_tokens)
        if overlap > best_overlap:
            best_overlap, best_sent = overlap, sent_text
    return best_sent


def compute_evidence_recall(evidence_sentence: str, compressed_text: str) -> Optional[float]:
    """Fraction of the answer-bearing sentence's tokens still present (as a
    multiset, matching compute_needle_recall) in the compressed context --
    did compression keep the evidence, not just some tokens. None if no
    evidence sentence could be identified for this sample."""
    ev_tokens = _normalize_answer(evidence_sentence)
    if not ev_tokens:
        return None
    ev_counts = Counter(ev_tokens)
    comp_counts = Counter(_normalize_answer(compressed_text))
    common = ev_counts & comp_counts
    return sum(common.values()) / sum(ev_counts.values())


_PHOBERT_TOKENIZER_CACHE: Dict[str, Any] = {}


def get_phobert_tokenizer(model_name: str = 'vinai/phobert-base'):
    """Lazy-load + process-cache a second tokenizer for dual-tokenizer
    achieved-rate reporting (docs/lacc_coling2027_tasklist.md G2): the
    generation LLM's tokenizer and PhoBERT's disagree on token counts, so a
    single achieved-ratio number hides which one a reader is actually
    getting. Cheap (vocab-only, no model weights) so this defaults on, unlike
    get_nli_scorer. Returns None (achieved-ratio stays absent, not wrong) if
    transformers/the tokenizer can't be loaded."""
    if model_name in _PHOBERT_TOKENIZER_CACHE:
        return _PHOBERT_TOKENIZER_CACHE[model_name]
    tok = None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        print(f"[WARN] Second tokenizer unavailable ({model_name}): {exc}. "
              "phobert_achieved_ratio will be absent from metadata.")
    _PHOBERT_TOKENIZER_CACHE[model_name] = tok
    return tok


_NLI_PIPELINE_CACHE: Dict[str, Any] = {}


def get_nli_scorer(model_name: str = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'):
    """Lazy-load + process-cache a multilingual NLI text-pair classifier for
    source_span_recoverability / unsupported_claim_rate.

    Optional diagnostics, not a required part of a benchmark run: returns None
    (metrics stay None, never silently 0.0) if `transformers` or the model
    can't be loaded, e.g. no network/no GPU on a dev machine -- the caller
    decides whether that's fatal.
    """
    if model_name in _NLI_PIPELINE_CACHE:
        return _NLI_PIPELINE_CACHE[model_name]
    clf = None
    try:
        from transformers import pipeline

        clf = pipeline(
            'text-classification', model=model_name, top_k=None,
            device=0 if _cuda_available() else -1,
        )
    except Exception as exc:
        print(f"[WARN] NLI scorer unavailable ({model_name}): {exc}. "
              "source_span_recoverability / unsupported_claim_rate will be None.")
    _NLI_PIPELINE_CACHE[model_name] = clf
    return clf


def _nli_entailment_prob(nli, premise: str, hypothesis: str) -> Optional[float]:
    """P(entailment) for premise -> hypothesis from a text-classification NLI
    pipeline. None if the scorer, premise, or hypothesis is unusable."""
    if nli is None or not premise.strip() or not hypothesis.strip():
        return None
    try:
        scores = nli({'text': premise, 'text_pair': hypothesis}, truncation=True)
        if scores and isinstance(scores[0], list):
            scores = scores[0]
        for entry in scores:
            if 'entail' in entry['label'].lower():
                return float(entry['score'])
        return None
    except Exception:
        return None


def compute_source_span_recoverability(nli, compressed_text: str, evidence_sentence: str) -> Optional[float]:
    """P(compressed context entails the original evidence sentence): can the
    answer-bearing span be recovered/inferred from what compression kept.
    None without an NLI scorer or an identified evidence sentence."""
    return _nli_entailment_prob(nli, premise=compressed_text, hypothesis=evidence_sentence)


def compute_span_recoverability_hard(original_text: str, compressed_text: str) -> Optional[float]:
    """Deterministic, NLI-free span recoverability: the fraction of compressed
    tokens that occur (as a multiset) in the ORIGINAL context after Unicode-NFC
    + lowercase + whitespace normalization.

    This is the complement of LLMLingua-2's Variation Rate
    (VR = fraction of compressed words absent from the source; arXiv:2403.12968,
    sec. data filtering), so span_recoverability_hard = 1 - VR. A genuinely
    extractive compressor keeps verbatim spans and scores ~1.0; abstractive or
    translate-then-compress arms introduce tokens absent from the source and
    score below 1.0. Reported as a control that our extractive compressor does
    not launder attribution (contrast the emitted-grounded gap of
    arXiv:2609.14245, sec. 3.2), and unlike source_span_recoverability it needs
    no NLI model, so it carries none of that evaluator's circularity risk.

    None when the compressed text has no scorable tokens."""
    comp_counts = Counter(_normalize_answer(compressed_text))
    if not comp_counts:
        return None
    orig_counts = Counter(_normalize_answer(original_text))
    present = comp_counts & orig_counts  # multiset intersection
    return sum(present.values()) / sum(comp_counts.values())


_CLAIM_SPLIT_RE = re.compile(r'(?<=[.!?\n])\s+')


def compute_unsupported_claim_rate(nli, compressed_text: str, output_text: str) -> Optional[float]:
    """Fraction of the generated answer's sentence-level claims NOT entailed
    (P(entailment) < 0.5) by the compressed context -- an attribution/
    hallucination proxy: does the answer say things the surviving context
    doesn't support. None without an NLI scorer or a non-empty output."""
    if nli is None or not output_text.strip():
        return None
    claims = [c.strip() for c in _CLAIM_SPLIT_RE.split(output_text) if c.strip()]
    if not claims:
        return None
    scored = unsupported = 0
    for claim in claims:
        p = _nli_entailment_prob(nli, premise=compressed_text, hypothesis=claim)
        if p is None:
            continue
        scored += 1
        if p < 0.5:
            unsupported += 1
    return (unsupported / scored) if scored else None


# ============================================================================
# Run-level instrumentation (cost + tone diagnostics for every arm)
# ============================================================================


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def _reset_peak_vram() -> None:
    if _cuda_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_vram() -> Optional[int]:
    """Peak CUDA allocation since the last reset, or None off GPU.

    docs/eval_sweep_gate1.md SS4 rule 4: the wave-1 cost table predates the
    2048/256 perplexity window and cannot be reused, so every run has to carry
    its own cost measurement rather than citing the old one.
    """
    return int(torch.cuda.max_memory_allocated()) if _cuda_available() else None


def _resolve_tone_preservation_rate(result, tokenizer, input_ids: List[int]) -> Optional[float]:
    """Tone preservation rate for ANY arm, not just LACC.

    LACC reports an exact index-based rate in its own metadata. Every other arm
    reported nothing, which made the TPR-vs-token-F1 correlation in
    docs/eval_sweep_gate1.md SS5 impossible to compute across arms -- the
    comparison needs `llmlingua` and `random` to have a TPR too. Fall back to
    the surface approximation over decoded tokens for those.
    """
    if 'tone_preservation_rate' in result.metadata:
        return result.metadata['tone_preservation_rate']
    try:
        from .compression import approx_tone_preservation_rate, token_surface_strings
        from .linguistics import get_tone_analyzer

        # token_surface_strings, not a per-id decode loop: on a 30k-character
        # needle context the latter costs more than the compression it is
        # annotating, across 12 arms x 3 ratios x 414 samples.
        orig = token_surface_strings(tokenizer, input_ids)
        comp = token_surface_strings(tokenizer, result.compressed_ids)
        return approx_tone_preservation_rate(orig, comp, get_tone_analyzer())
    except Exception:
        return None


# The instruction the generator sees alongside the compressed context. Bare
# `context + query` concatenation (the original behaviour) leaves an
# instruction-tuned model free to continue the text instead of answering, which
# makes exact-match and token-F1 measure formatting luck rather than whether the
# answer survived compression. Answer-only, grounded, Vietnamese.
ANSWER_INSTRUCTION_VI = (
    "Dựa CHỈ vào ngữ cảnh dưới đây, trả lời câu hỏi thật ngắn gọn. "
    "Chỉ đưa ra đáp án, không giải thích, không lặp lại câu hỏi. "
    "Nếu ngữ cảnh không chứa đáp án, trả lời: KHÔNG CÓ THÔNG TIN."
)


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
    # 1 = one generate() call per sample, exactly as every run before
    # 2026-09-18. Raise it for a full sweep: 5 tasks x 14 arms x 3 ratios is
    # ~23k generations, which one at a time on a 7B reader costs days.
    generation_batch_size: int = 1
    temperature: float = 0.0
    do_sample: bool = False
    output_dir: str = './results'
    save_predictions: bool = True
    device: str = 'cuda'
    # 'chat' wraps the compressed context + query in the generator's own chat
    # template with an explicit Vietnamese answer-only instruction; 'raw' is the
    # original bare `compressed_ids + query_ids` concatenation, kept so wave-1
    # numbers stay reproducible. See docs/eval_sweep_gate1.md SS6.
    prompt_style: str = 'chat'
    # Stamped into every per-sample record and into the result filename, so
    # multi-seed arms (the `random` floor, SS4 rule 3) never overwrite each other.
    seed: int = 42
    # HF text-classification NLI model id for source_span_recoverability /
    # unsupported_claim_rate (docs/lacc_coling2027_tasklist.md G0). None
    # (default) skips both metrics entirely -- no model load, no per-sample
    # cost -- since they are optional diagnostics, not required for a run.
    nli_model: Optional[str] = None
    # Second tokenizer for dual-tokenizer achieved-rate reporting
    # (docs/lacc_coling2027_tasklist.md G2: CR/TS on the generation
    # tokenizer alone hides how much a PhoBERT-tokenized reader -- e.g. the
    # 'encoder' arm's own tokenizer -- would see. None skips it.
    phobert_tokenizer: Optional[str] = 'vinai/phobert-base'


# Dataset fields copied verbatim into every per-sample metric record. Without
# these, a finished GPU sweep cannot answer the question the sweep exists to
# answer -- docs/eval_sweep_gate1.md SS5 splits needle results by `needle_group`,
# and SS5's secondary analyses need `insert_position` / `qa_subset`. Add keys
# here rather than post-hoc joining results back onto the dataset by row order.
ANALYSIS_METADATA_KEYS = (
    'sample_id',
    'source', 'domain', 'title',
    'needle_group', 'insert_position', 'needle_category', 'needle_has_diacritic',
    'haystack_id', 'synthetic_pii',
    'qa_subset', 'law_id', 'chapter',
    'cross_config', 'num_turns', 'scenario',
)


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
        # Loaded once for the whole sweep, not per-sample/per-arm -- a
        # cross-encoder load is seconds, not something to pay 11 arms x 3
        # ratios x 414 samples times.
        nli = get_nli_scorer(self.config.nli_model) if self.config.nli_model else None
        phobert_tok = get_phobert_tokenizer(self.config.phobert_tokenizer) if self.config.phobert_tokenizer else None
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
                    metrics_list = self._evaluate_task(
                        compressor, model, tokenizer, samples, ratio, task_name, generation_fn,
                        method_name=method_name, nli=nli, phobert_tok=phobert_tok,
                    )
                    task_results[f'ratio_{ratio}'] = self._aggregate_metrics(metrics_list)
                    self._warn_on_generation_errors(method_name, task_name, ratio, metrics_list)
                    if self.config.save_predictions:
                        self._save_results(method_name, task_name, ratio, metrics_list)
                method_results[task_name] = task_results
            all_results[method_name] = method_results
        all_results['summary'] = self._compute_summary(all_results)
        return all_results

    @staticmethod
    def _warn_on_generation_errors(method_name, task_name, ratio, metrics_list) -> None:
        """Say it out loud, at the point it happens.

        Failed generations produce empty predictions, which score 0 -- visually
        identical to an arm that genuinely lost the answer. Anyone reading a
        results table has no way to tell the two apart, so the run itself has to.
        """
        errors = [m.metadata.get('generation_error') for m in metrics_list if m.metadata.get('generation_error')]
        if not errors:
            return
        kinds = Counter(e.split(':', 1)[0] for e in errors)
        print(f"    [WARN] {len(errors)}/{len(metrics_list)} generations FAILED for "
              f"{method_name}/{task_name}@{ratio}x -- {dict(kinds)}. "
              f"First: {errors[0][:200]}")
        print("           These score 0 and are NOT evidence about the compressor. Fix before reporting.")

    def _evaluate_task(self, compressor, model, tokenizer, samples, ratio, task_name, generation_fn,
                       method_name: str = 'unknown', nli=None, phobert_tok=None) -> List[CompressionMetrics]:
        """Compress every sample, then generate, then score.

        Generation is a separate phase so it can be batched (see
        `_run_generation`): a full 5-task sweep is ~23k generations, and
        decoding them one at a time on a 7B reader is the difference between a
        run that fits the schedule and one that does not. Compression stays
        per-sample -- it is the thing being measured.
        """
        metrics_list = []
        pending = []
        compressor.config.target_ratio = ratio
        for sample in tqdm(samples, desc=f"  {task_name} @ {ratio}x [compress]"):
            input_ids = tokenizer.encode(sample.context, add_special_tokens=False)
            _reset_peak_vram()
            start_time = time.time()
            # `query`: without it, LACC's query-relevance boost and
            # SelectiveContext's embedding path never run in any benchmark.
            # `task`: lets LACC's task-gated tone signal (E8) know whether this
            # is a surface task. Compressors that don't accept it ignore it via **kwargs.
            result = compressor.compress(input_ids, query=sample.query, task=task_name)
            comp_time = (time.time() - start_time) * 1000

            metric = CompressionMetrics(
                compression_ratio=result.compression_ratio,
                token_savings_pct=result.token_savings_pct,
                processing_time_ms=comp_time,
            )
            metric.tone_preservation_rate = _resolve_tone_preservation_rate(
                result, tokenizer, input_ids,
            )

            # Evidence/attribution (docs/lacc_coling2027_tasklist.md G0).
            # evidence_recall needs no NLI -- cheap, always computed.
            # source_span_recoverability/unsupported_claim_rate need `nli`
            # (config.nli_model); stay None on every sample when it's unset.
            compressed_text = tokenizer.decode(result.compressed_ids, skip_special_tokens=True)
            original_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            evidence_sentence = find_evidence_sentence(tokenizer, input_ids, sample.reference_answer)
            metric.evidence_recall = compute_evidence_recall(evidence_sentence, compressed_text)
            metric.source_span_recoverability = compute_source_span_recoverability(
                nli, compressed_text, evidence_sentence,
            )
            # Deterministic control (no NLI): does compression keep verbatim
            # source spans, or introduce new tokens (abstractive laundering)?
            metric.span_recoverability_hard = compute_span_recoverability_hard(
                original_text, compressed_text,
            )

            # Dual-tokenizer achieved-rate (G2): the generation tokenizer's
            # achieved ratio is already `result.compression_ratio` /
            # `original_tokens` / `compressed_tokens` below; this is the SAME
            # compressed text re-tokenized with PhoBERT, since the two
            # disagree on token counts and a single number hides which
            # tokenizer a reader is actually getting.
            phobert_orig_n = phobert_comp_n = phobert_achieved = None
            if phobert_tok is not None:
                phobert_orig_n = len(phobert_tok.encode(original_text, add_special_tokens=False))
                phobert_comp_n = len(phobert_tok.encode(compressed_text, add_special_tokens=False))
                phobert_achieved = (phobert_orig_n / phobert_comp_n) if phobert_comp_n else None

            # Compression-phase peak only. Generation VRAM is now shared across
            # a batch and is identical across arms anyway; the arm-specific cost
            # is what the cost table is about.
            metric.peak_vram_bytes = _peak_vram()

            # Per-sample provenance. `compression_ratio` above is the REALIZED
            # ratio; `requested_ratio` is what was configured. Reporting at the
            # configured ratio compares arms at different operating points --
            # docs/eval_sweep_gate1.md SS4 rule 1 -- so both must survive to the
            # analysis stage, along with the prediction text (re-scoring a run
            # with a new metric must not require re-running the GPU sweep).
            metric.metadata.update({
                'method': method_name,
                'task': task_name,
                # Pairs a row with the same benchmark sample scored by another
                # arm: a paired bootstrap across arms is only valid if the rows
                # line up, and index alignment breaks silently the moment one
                # arm skips a sample.
                'sample_id': sample.metadata.get('sample_id'),
                'requested_ratio': float(ratio),
                'seed': self.config.seed,
                'original_tokens': len(input_ids),
                'compressed_tokens': len(result.compressed_ids),
                'phobert_original_tokens': phobert_orig_n,
                'phobert_compressed_tokens': phobert_comp_n,
                'phobert_achieved_ratio': phobert_achieved,
                'reference_answer': sample.reference_answer,
                'query': sample.query,
            })
            for key in ANALYSIS_METADATA_KEYS:
                if key in sample.metadata:
                    metric.metadata[key] = sample.metadata[key]
            # Whatever the compressor itself reported (budget_used,
            # translation_time_ms, num_sentences_kept, ...). Prefixed so an arm
            # can never shadow a provenance key, and kept because these are the
            # diagnostics that explain WHY an arm landed where it did -- and the
            # cost table owes translation time to the translate-then-compress arm.
            for key, value in result.metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    metric.metadata[f'arm_{key}'] = value
            pending.append((metric, sample, result, compressed_text))
            metrics_list.append(metric)

        outputs, gen_errors, gen_times, batch_size = self._run_generation(
            model, tokenizer, pending, generation_fn, task_name, ratio,
        )

        for (metric, sample, _result, compressed_text), output, gen_error, gen_ms in zip(
                pending, outputs, gen_errors, gen_times):
            metric.generation_time_ms = gen_ms

            if output:
                rouge = compute_rouge_l([output], [sample.reference_answer])
                metric.rouge_l_f1 = rouge['rougeL_f1']
                metric.rouge_l_precision = rouge['rougeL_precision']
                metric.rouge_l_recall = rouge['rougeL_recall']
                metric.bleu_score = compute_bleu([output], [sample.reference_answer])
                metric.exact_match = output.strip().lower() == sample.reference_answer.strip().lower()
                metric.token_f1 = compute_token_f1([output], [sample.reference_answer])
                if task_name == 'needle_in_haystack':
                    metric.needle_recall = compute_needle_recall([output], [sample.reference_answer])
                metric.unsupported_claim_rate = compute_unsupported_claim_rate(nli, compressed_text, output)

            metric.quality_score = (
                (metric.rouge_l_f1 or 0) * 0.4 + (metric.bleu_score or 0) * 0.2 + float(metric.exact_match) * 0.4
            )
            metric.efficiency_score = metric.token_savings_pct / 100.0
            metric.metadata.update({
                'prediction': output or '',
                'generation_error': gen_error,
                # generation_time_ms is the batch's wall time divided by the
                # batch, not a single-sample latency: any per-sample generation
                # latency claim has to come from a --gen-batch-size 1 run.
                'generation_batch_size': batch_size,
            })
        return metrics_list

    def _run_generation(self, model, tokenizer, pending, generation_fn, task_name, ratio):
        """Generate an answer for every pending sample.

        Returns (outputs, errors, per-sample ms, batch_size). Batched whenever
        config.generation_batch_size > 1 and no custom generation_fn is given --
        a custom hook takes one sample at a time and cannot be batched behind
        its back. Prompts are grouped longest-first so each batch pads to a
        similar width, and a batch that raises (OOM, above all) is retried one
        sample at a time: a single bad sample must not score its seven
        neighbours as failed generations.
        """
        n = len(pending)
        outputs: List[Optional[str]] = [None] * n
        errors: List[Optional[str]] = [None] * n
        times: List[float] = [0.0] * n
        batch_size = 1 if generation_fn else max(1, int(getattr(self.config, 'generation_batch_size', 1)))

        def generate_one(i):
            metric, sample, result, _ct = pending[i]
            start = time.time()
            if generation_fn:
                out = generation_fn(
                    model, tokenizer, compressed_ids=result.compressed_ids, query=sample.query,
                    max_new_tokens=self.config.max_new_tokens, temperature=self.config.temperature,
                )
                err = None
            else:
                out = self._default_generate(model, tokenizer, result.compressed_ids, sample.query)
                err = getattr(self, '_last_generation_error', None)
            outputs[i], errors[i], times[i] = out, err, (time.time() - start) * 1000

        if batch_size == 1:
            for i in tqdm(range(n), desc=f"  {task_name} @ {ratio}x [generate]"):
                generate_one(i)
            return outputs, errors, times, batch_size

        prompts = [self._build_prompt_ids(tokenizer, r.compressed_ids, s.query)
                   for _m, s, r, _ct in pending]
        order = sorted(range(n), key=lambda i: -len(prompts[i]))
        chunks = [order[i:i + batch_size] for i in range(0, n, batch_size)]
        for chunk in tqdm(chunks, desc=f"  {task_name} @ {ratio}x [generate x{batch_size}]"):
            start = time.time()
            texts, batch_error = self._generate_batch(model, tokenizer, [prompts[i] for i in chunk])
            elapsed = (time.time() - start) * 1000
            if texts is None:
                print(f"    [WARN] batched generation failed ({batch_error}); "
                      f"retrying {len(chunk)} sample(s) one at a time.")
                for i in chunk:
                    generate_one(i)
                continue
            for j, i in enumerate(chunk):
                outputs[i], times[i] = texts[j], elapsed / len(chunk)
        return outputs, errors, times, batch_size

    def _generate_batch(self, model, tokenizer, prompts: List[List[int]]):
        """Greedy-decode a batch of prompts. -> (texts, None) or (None, error).

        LEFT-padded: a decoder-only model padded on the right would attend to
        pad tokens as if they were context, and the generated span would start
        at a different offset in every row. With left padding every row's new
        tokens begin at exactly `width`, so one slice is correct for all.
        """
        try:
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            width = max(len(p) for p in prompts)
            input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
            attention = torch.zeros((len(prompts), width), dtype=torch.long)
            for row, prompt in enumerate(prompts):
                input_ids[row, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long)
                attention[row, width - len(prompt):] = 1
            input_ids = input_ids.to(model.device)
            attention = attention.to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids, attention_mask=attention,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature, do_sample=self.config.do_sample,
                    pad_token_id=pad_id,
                )
            return [tokenizer.decode(row[width:], skip_special_tokens=True) for row in out], None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _build_prompt_ids(self, tokenizer, compressed_ids: List[int], query: str) -> List[int]:
        """Token ids for the generation call.

        'raw' reproduces the original bare concatenation; 'chat' (default) wraps
        the same compressed context in the generator's own chat template with
        ANSWER_INSTRUCTION_VI. Falls back to 'raw' when the tokenizer has no
        chat template, so non-instruct generators still run.
        """
        if self.config.prompt_style == 'chat' and getattr(tokenizer, 'chat_template', None):
            context = tokenizer.decode(compressed_ids, skip_special_tokens=True)
            messages = [{
                'role': 'user',
                'content': f"{ANSWER_INSTRUCTION_VI}\n\n### Ngữ cảnh:\n{context}\n\n### Câu hỏi:\n{query}\n\n### Đáp án:",
            }]
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            # transformers >=4.49 (confirmed: 5.12.1) returns a BatchEncoding
            # here, not a plain list of ids -- torch.tensor([encoded]) used to
            # silently fail with "Could not infer dtype of tokenizers.Encoding"
            # for EVERY arm, and even if it hadn't, len(encoded) would count
            # dict keys (2), not tokens, corrupting the slice that strips the
            # prompt off the generated output below.
            return encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
        return compressed_ids + tokenizer.encode(query, add_special_tokens=False)

    def _default_generate(self, model, tokenizer, compressed_ids: List[int], query: str) -> Optional[str]:
        self._last_generation_error = None
        try:
            full_input = self._build_prompt_ids(tokenizer, compressed_ids, query)
            input_tensor = torch.tensor([full_input]).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    input_tensor, max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature, do_sample=self.config.do_sample,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            return tokenizer.decode(outputs[0][len(full_input):], skip_special_tokens=True)
        except Exception as exc:
            # Record it. A swallowed OOM/context-overflow used to be
            # indistinguishable from "compression destroyed the answer": both
            # produced an empty prediction and a quality score of 0. A broken
            # run that reads as a bad arm is the one failure mode a gate
            # decision cannot survive.
            self._last_generation_error = f"{type(exc).__name__}: {exc}"
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
            'mean_needle_recall': _mean_or_none([m.needle_recall for m in metrics_list]),
            'mean_generation_time_ms': _mean_or_none([m.generation_time_ms for m in metrics_list]),
            'peak_vram_bytes': max((m.peak_vram_bytes or 0) for m in metrics_list) or None,
            'num_generated': sum(1 for m in metrics_list if m.rouge_l_f1 is not None),
            'num_generation_errors': sum(1 for m in metrics_list if m.metadata.get('generation_error')),
            'mean_quality_score': np.mean([m.quality_score for m in metrics_list]),
            'mean_efficiency_score': np.mean([m.efficiency_score for m in metrics_list]),
            'mean_evidence_recall': _mean_or_none([m.evidence_recall for m in metrics_list]),
            'mean_source_span_recoverability': _mean_or_none([m.source_span_recoverability for m in metrics_list]),
            'mean_span_recoverability_hard': _mean_or_none([m.span_recoverability_hard for m in metrics_list]),
            'mean_unsupported_claim_rate': _mean_or_none([m.unsupported_claim_rate for m in metrics_list]),
            'mean_phobert_achieved_ratio': _mean_or_none(
                [m.metadata.get('phobert_achieved_ratio') for m in metrics_list]
            ),
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
        path = os.path.join(
            self.config.output_dir,
            f"{method_name}_{task_name}_ratio{ratio:.1f}_seed{self.config.seed}.json",
        )
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


@dataclass
class EquivalenceResult:
    """Result of a two-one-sided-tests (TOST) equivalence check on the paired
    mean difference mean(a - b), against a pre-registered SESOI margin."""

    n: int
    mean_delta: float
    sesoi: float
    # 100*(1 - 2*alpha)% CI -- the interval TOST checks against +/-sesoi.
    ci_low: float
    ci_high: float
    p_tost: float           # max of the two one-sided bootstrap p-values
    equivalent: bool        # CI within [-sesoi, +sesoi] at this alpha

    def to_dict(self) -> Dict:
        return asdict(self)


def tost_equivalence(
    a: Sequence[float], b: Sequence[float], sesoi: float,
    alpha: float = 0.05, n_boot: int = 10000, seed: int = 42,
) -> Optional[EquivalenceResult]:
    """Two One-Sided Tests (TOST) equivalence check on the paired mean
    difference mean(a - b), bootstrapped exactly like paired_bootstrap_delta.

    A non-significant paired test (p > 0.05) only fails to find a difference; it
    does NOT establish equivalence -- the trap every negative result must avoid
    (Lakens 2017, "Equivalence Tests: A Practical Primer"). TOST instead asks
    whether the effect is smaller than a pre-registered smallest effect size of
    interest (`sesoi`, e.g. 1 EM/F1 point on VCC-Bench): equivalence holds when
    the 100*(1-2*alpha)% CI of the difference lies entirely within
    [-sesoi, +sesoi]. That licenses the claim "tone-augmented is equivalent to
    language-agnostic selection within +/-SESOI", not merely "no significant
    difference". See docs/agent_research_findings_round2.md.

    sesoi must be > 0. Returns None if fewer than 2 usable (non-None) pairs
    remain, matching paired_bootstrap_delta.
    """
    if sesoi <= 0:
        raise ValueError(f"sesoi must be positive, got {sesoi}")
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    diff = np.array([x - y for x, y in pairs], dtype=float)
    n = len(diff)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)

    # TOST at alpha == the (1 - 2*alpha) CI lying within the bounds.
    lo_q = alpha
    ci_low, ci_high = np.quantile(boot_means, [lo_q, 1 - lo_q])
    # Two one-sided bootstrap p-values: H0_upper: delta >= +sesoi (reject if
    # few resamples reach it); H0_lower: delta <= -sesoi. Equivalence needs both
    # rejected, so the TOST p is the larger (weaker) of the two.
    p_upper = float((boot_means >= sesoi).mean())
    p_lower = float((boot_means <= -sesoi).mean())
    p_tost = max(p_upper, p_lower)

    return EquivalenceResult(
        n=n, mean_delta=float(diff.mean()), sesoi=float(sesoi),
        ci_low=float(ci_low), ci_high=float(ci_high),
        p_tost=p_tost, equivalent=bool(ci_low > -sesoi and ci_high < sesoi),
    )


@dataclass
class PowerResult:
    """Statistical power of a paired comparison to detect an effect of size
    `sesoi`, plus the minimum detectable effect at a target power. Normal
    approximation to the paired-mean test (Card et al. 2020)."""

    n: int
    sd_diff: float
    se: float               # standard error of the paired mean difference
    sesoi: float
    alpha: float
    power: float            # P(detect a true effect of size sesoi), two-sided
    target_power: float
    mde: float              # smallest effect detectable at target_power
    underpowered: bool      # power < target_power to see the SESOI

    def to_dict(self) -> Dict:
        return asdict(self)


def paired_power_analysis(
    a: Sequence[float], b: Sequence[float], sesoi: float,
    alpha: float = 0.05, target_power: float = 0.80,
) -> Optional[PowerResult]:
    """Power and minimum-detectable-effect for the paired difference a - b,
    using the observed per-sample variance (normal approximation to the paired
    t-test; Card et al. 2020, "With Little Power Comes Great Responsibility").

    A negative result must show the test COULD have detected an effect of the
    smallest size worth caring about (`sesoi`) had one existed; otherwise "no
    difference" may just be too few samples. `power` is that probability at the
    current n; `mde` is the smallest true effect this n could detect at
    `target_power` -- report it so an underpowered subset (e.g. E8's n=18) is
    called out rather than over-claimed as equivalence.

    sesoi must be > 0. Returns None if fewer than 2 usable (non-None) pairs
    remain or the differences have zero variance (power is then undefined here).
    """
    if sesoi <= 0:
        raise ValueError(f"sesoi must be positive, got {sesoi}")
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    diff = np.array([x - y for x, y in pairs], dtype=float)
    n = len(diff)
    sd = float(diff.std(ddof=1))
    if sd == 0.0:
        return None
    se = sd / np.sqrt(n)

    nd = NormalDist()
    z_alpha = nd.inv_cdf(1 - alpha / 2)
    ncp = sesoi / se  # non-centrality: how many SEs the SESOI is
    power = nd.cdf(ncp - z_alpha) + nd.cdf(-ncp - z_alpha)
    z_power = nd.inv_cdf(target_power)
    mde = (z_alpha + z_power) * se

    return PowerResult(
        n=n, sd_diff=sd, se=float(se), sesoi=float(sesoi), alpha=float(alpha),
        power=float(power), target_power=float(target_power), mde=float(mde),
        underpowered=bool(power < target_power),
    )


# ============================================================================
# Method taxonomy (baseline / proposed / ablation)
# ============================================================================
#
# Two namespaces, since benchmark.py and its ablation mode name methods
# differently:
#   - "registry": vncompress.compression.METHODS keys (--methods flag).
#   - "ablation": the isolated-signal arms (ppl_only/tone_only/morph_only/lacc).


RANDOM_SEED_ARM_RE = re.compile(r'random_s\d+')


class MethodCategory(str, Enum):
    BASELINE = "baseline"
    PROPOSED = "proposed"
    ABLATION = "ablation"


REGISTRY_METHOD_CATEGORY: Dict[str, MethodCategory] = {
    "none": MethodCategory.BASELINE,
    "random": MethodCategory.BASELINE,
    "llmlingua": MethodCategory.BASELINE,
    "llmlingua_contrastive": MethodCategory.BASELINE,  # LongLLMLingua (wave-2 E1/E11)
    "snapkv": MethodCategory.BASELINE,
    "selective": MethodCategory.BASELINE,
    "encoder": MethodCategory.BASELINE,                # LLMLingua-2 / PhoBERT-style (wave-2 E6/E11)
    # Gate-1 baselines (docs/eval_sweep_gate1.md SS3) -- the two arms that, if
    # they win, change what the paper claims.
    "sentence_retrieval": MethodCategory.BASELINE,
    "translate_then_compress": MethodCategory.BASELINE,
    "translate_then_compress_llmlingua2": MethodCategory.BASELINE,
    "lacc": MethodCategory.PROPOSED,
    # wave-2 LACC arms (E1/E2/E5/E7): proposed variants of the LACC method
    "lacc_ppl_contrastive": MethodCategory.PROPOSED,
    "lacc_ppl_morph": MethodCategory.PROPOSED,
    "lacc_cx_morph": MethodCategory.PROPOSED,  # E1+E2: query-conditioned ppl x morph
    "lacc_sentence": MethodCategory.PROPOSED,
    "lacc_classprop": MethodCategory.PROPOSED,
    "lacc_tone_gated": MethodCategory.PROPOSED,
    "lacc_tone": MethodCategory.PROPOSED,  # A9 (docs/eval_sweep_gate1.md SS2): wave-1 re-check arm
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
    # `random_s1`/`random_s2`/... are the same baseline run under different
    # seeds (docs/eval_sweep_gate1.md SS4 rule 3: one seed of random is a
    # sample, not a floor), so they inherit `random`'s category.
    if context == "registry" and RANDOM_SEED_ARM_RE.fullmatch(method_name):
        return MethodCategory.BASELINE
    if method_name not in table:
        raise ValueError(
            f"Method {method_name!r} is not classified in evaluation.py's method taxonomy "
            f"(context={context!r}). Add it to REGISTRY_METHOD_CATEGORY or ABLATION_ARM_CATEGORY "
            "before including it in a results table."
        )
    return table[method_name]
