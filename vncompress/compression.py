"""
LACC — Language-Aware Context Compression. The core algorithm.
=================================================================
Everything from "input tokens" to "compressed tokens" lives in this one file:
CompressionResult/CompressionConfig, the shared token-budget/selection logic,
baseline compressors (prior art, for comparison), and LACCCompressor -- the
proposed method.

The single most important architectural rule in this file (see docs/training.md
"Method" section): tone, morphology, perplexity and the trained tone probe are
NOT separate compressors. They are *signals*, and LACCCompressor is the one
algorithm that combines them:

    S(t) = w_ppl * S_ppl(t) + w_tone * S_tone(t) + w_morph * S_morph(t)

each term normalized to [0, 1], weights renormalized to sum to 1 over
whichever signals are enabled. Ablation is just config:

    LACCCompressor(tokenizer, model, use_perplexity=True,
                   use_tone=True, use_morphology=False)   # ppl + tone only

This replaces what used to be six separate classes (ToneAwareCompressor,
MorphologyAwareCompressor, CombinedCompressor, EnhancedCompressor,
SLMScorerCompressor, SLMToneProbeCompressor) -- see git history for the
pre-refactor implementations if you need to compare exact old-vs-new numbers.
One behavioral note from that unification: the old classes combined signals
*multiplicatively* (base_score * tone_multiplier * morph_multiplier); LACC
combines them *additively* (a weighted sum of three independently-normalized
signals), matching the formula already documented in README.md. This is a
single, consistent combination rule across every signal and every hardware
tier (0 VRAM / lightweight scorer / full generation model), instead of two
different blending rules depending on which class you picked.

Hardware tiers, all through the same LACCCompressor:
  no_model     (0 GB) : model=None, scorer=None      -> rule-based tone/morph only
  lightweight  (~0.3GB): scorer=<tiny SLM>            -> + perplexity/tone-probe from a small model
  full         (~5GB+) : model=<real generation model> -> + perplexity/tone-probe from the model itself
"""

from __future__ import annotations

import itertools
import random
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from .linguistics import (
    MorphologyAnalyzer,
    MorphologyConfig,
    VietnameseToneAnalyzer,
    compute_tone_preservation_rate,
    get_morphology_analyzer,
    get_tone_analyzer,
)

# ============================================================================
# Result / config
# ============================================================================


@dataclass
class CompressionResult:
    """The one result type every compressor returns. Method-specific extras
    (weights used, tone_source, quality-gate stats, ...) go in `metadata`
    rather than growing a new result subclass."""

    compressed_ids: List[int]
    compressed_text: str

    original_length: int
    compressed_length: int

    compression_ratio: float          # original_length / compressed_length
    token_savings_pct: float          # (original - compressed) / original * 100

    method_name: str
    processing_time_ms: float

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionConfig:
    """Base configuration shared by every compressor."""

    target_ratio: float = 4.0
    keep_special_tokens: bool = True
    keep_boundary_tokens: int = 2
    min_compressed_length: int = 1
    max_compressed_length: int = 32768

    language: str = 'vi'
    detect_language: bool = True
    verbose: bool = False


class BaseCompressor(ABC):
    """Abstract base for all compression methods. Subclasses implement
    `compress()` and `get_name()`; everything else (budget math, boundary-aware
    selection, result building) is shared here so it cannot silently diverge
    between methods -- see target_length()/select_with_boundary() docstrings
    for the two bugs a previously-duplicated version of this logic had."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Optional[PreTrainedModel] = None,
        config: Optional[CompressionConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config or CompressionConfig()

    @abstractmethod
    def compress(self, input_ids: List[int], **kwargs) -> CompressionResult:
        """Compress a sequence of token IDs, returning a CompressionResult."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this compression method."""

    def compress_text(self, text: str, **kwargs) -> CompressionResult:
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self.compress(input_ids, **kwargs)

    def _compute_compression_ratio(self, original_length: int, compressed_length: int) -> Tuple[float, float]:
        ratio = original_length / max(compressed_length, 1)
        savings = ((original_length - compressed_length) / max(original_length, 1)) * 100
        return ratio, savings

    def _build_result(
        self, compressed_ids: List[int], original_length: int, processing_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CompressionResult:
        comp_len = len(compressed_ids)
        ratio, savings = self._compute_compression_ratio(original_length, comp_len)
        try:
            compressed_text = self.tokenizer.decode(compressed_ids, skip_special_tokens=True)
        except Exception:
            compressed_text = '[decode error]'
        return CompressionResult(
            compressed_ids=compressed_ids, compressed_text=compressed_text,
            compression_ratio=ratio, token_savings_pct=savings,
            original_length=original_length, compressed_length=comp_len,
            method_name=self.get_name(), processing_time_ms=processing_time_ms,
            metadata=metadata or {},
        )

    def target_length(self, n: int) -> int:
        """Tokens to keep for `n` input tokens at the configured ratio. A
        single definition on purpose -- every compressor used to floor at a
        different value (1 vs 2*keep_boundary_tokens), so at the same
        configured ratio they compressed to different actual lengths and the
        benchmark compared them at different operating points."""
        ratio = self.config.target_ratio
        if ratio <= 0:
            raise ValueError(f"target_ratio must be > 0 (ratio = original/compressed); got {ratio}")
        return max(int(n / ratio), self.config.min_compressed_length)

    def select_with_boundary(self, scores: Sequence[float], n: int) -> List[int]:
        """Indices to keep: the `k` boundary tokens on each side plus the
        top-scoring middle, in original order. `scores` has length `n`,
        higher = keep. Replaces a block once copy-pasted into seven
        compressors with a bug: when the budget left no room for middle
        tokens, the fallback returned *every* index -- so asking for MORE
        compression produced NONE."""
        k = max(0, min(self.config.keep_boundary_tokens, n // 2))
        target_len = min(self.target_length(n), n)
        mid_start, mid_end = k, max(k, n - k)
        mid_budget = max(0, min(target_len - 2 * k, mid_end - mid_start))
        chosen: List[int] = []
        if mid_budget > 0:
            mid = list(range(mid_start, mid_end))
            chosen = sorted(sorted(mid, key=lambda i: scores[i], reverse=True)[:mid_budget])
        return sorted(set(range(k)) | set(chosen) | set(range(mid_end, n)))

    def validate_input(self, input_ids: List[int]) -> bool:
        if not input_ids:
            raise ValueError("Empty input sequence")
        return len(input_ids) >= self.config.min_compressed_length

    def __repr__(self) -> str:
        return f"{self.get_name()}(config={self.config})"


class NoCompressor(BaseCompressor):
    """Identity compressor -- returns input unchanged (baseline)."""

    def compress(self, input_ids: List[int], **kwargs) -> CompressionResult:
        start = time.time()
        self.validate_input(input_ids)
        return self._build_result(list(input_ids), len(input_ids), (time.time() - start) * 1000)

    def get_name(self) -> str:
        return "NoCompression"


class RandomCompressor(BaseCompressor):
    """Random token dropout -- simple baseline. Uses its own RNG (not the
    global `random` module) so the benchmark's baseline doesn't shift because
    unrelated code called random.seed()."""

    def __init__(self, tokenizer, model=None, config=None, seed: int = 42):
        super().__init__(tokenizer, model, config)
        self._rng = random.Random(seed)

    def compress(self, input_ids: List[int], **kwargs) -> CompressionResult:
        start = time.time()
        if not self.validate_input(input_ids):
            return self._build_result(list(input_ids), len(input_ids), 0.0)
        n = len(input_ids)
        scores = [self._rng.random() for _ in range(n)]
        keep_indices = self.select_with_boundary(scores, n)
        compressed = [input_ids[i] for i in keep_indices]
        return self._build_result(compressed, n, (time.time() - start) * 1000)

    def get_name(self) -> str:
        return "RandomBaseline"


# ============================================================================
# Shared perplexity scoring (used by LLMLinguaCompressor and by LACC's
# same-tokenizer perplexity signal)
# ============================================================================


def _normalize(scores: torch.Tensor) -> torch.Tensor:
    """Min-max normalize to [0, 1]; a constant input maps to a neutral 0.5."""
    if scores.numel() == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi > lo:
        return (scores - lo) / (hi - lo)
    return torch.full_like(scores, 0.5)


@torch.no_grad()
def sliding_window_perplexity(
    model, input_ids: Sequence[int], window_size: int = 512, stride: Optional[int] = None,
) -> torch.Tensor:
    """importance(t_i) = -log P(t_i | context), scored over overlapping
    sliding windows of at most `window_size` tokens so tokens after the first
    window keep real left context instead of being rescored with empty
    context at every window boundary. Each window only contributes scores for
    its trailing (non-overlapping) positions -- the standard fixed-length
    perplexity scheme. Returns a length-n tensor, NOT yet normalized."""
    n = len(input_ids)
    if n == 0:
        return torch.zeros(0)
    device = next(model.parameters()).device
    if stride is None:
        stride = max(1, window_size // 2)
    stride = max(1, min(stride, window_size - 1)) if window_size > 1 else 1

    scores = torch.zeros(n)
    begin, prev_end = 0, 0
    while begin < max(n - 1, 1):
        end = min(begin + window_size, n)
        chunk = list(input_ids[begin:end])
        if len(chunk) < 2:
            break
        input_tensor = torch.tensor([chunk], device=device)
        logits = model(input_tensor).logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_log_probs = log_probs.gather(-1, input_tensor[:, 1:].unsqueeze(-1)).squeeze(-1)
        importance = (-token_log_probs[0]).cpu()

        # Position 0 has no predecessor; approximate it with position 1's score.
        if begin == 0 and importance.numel() > 0:
            scores[0] = importance[0]
        target_start = 1 if begin == 0 else prev_end
        for k in range(importance.numel()):
            pos = begin + k + 1
            if target_start <= pos < n:
                scores[pos] = importance[k]

        prev_end = end
        if end >= n:
            break
        begin += stride
    return scores


def _decode_tokens(tokenizer, input_ids: Sequence[int]) -> List[str]:
    tokens = []
    for tid in input_ids:
        t = tokenizer.decode([tid]).replace('▁', ' ').replace('Ġ', ' ').strip()
        tokens.append(t)
    return tokens


def _token_spans(tokenizer, input_ids: Sequence[int]) -> Tuple[str, List[Tuple[int, int]], List[str]]:
    """Rebuild the decoded text and each token's character span in it."""
    pieces, spans, cursor = [], [], 0
    for tid in input_ids:
        piece = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        pieces.append(piece)
        spans.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return ''.join(pieces), spans, pieces


def _pool_char_to_token(char_scores: torch.Tensor, spans: List[Tuple[int, int]], n: int, fill_neutral: bool) -> torch.Tensor:
    """Average per-character scores onto each token's span."""
    out = torch.zeros(n)
    for i, (s, e) in enumerate(spans):
        if e > s:
            window = char_scores[s:e]
            valid = window[~torch.isnan(window)]
            if valid.numel():
                out[i] = valid.mean()
    if fill_neutral:
        # Whitespace-only / unmapped tokens stay 0; treat as neutral (the
        # median) rather than maximally droppable.
        unmapped = out == 0
        if unmapped.any() and (~unmapped).any():
            out[unmapped] = out[~unmapped].median()
    return out


# ============================================================================
# Baseline compressors (prior art, kept for comparison)
# ============================================================================


class LLMLinguaCompressor(BaseCompressor):
    """LLMLingua-style prompt compression (Jiang et al., EMNLP 2023,
    arxiv:2310.05736): coarse sentence-level filtering, then token-level
    perplexity-based scoring via `small_model` (defaults to `model`)."""

    def __init__(
        self, tokenizer, model=None, small_model=None, config: Optional[CompressionConfig] = None, device: str = 'cuda',
    ):
        super().__init__(tokenizer, model, config)
        self.small_model = small_model or model
        self.device = device
        if self.small_model:
            self.small_model.eval()

    def get_name(self) -> str:
        return "LLMLingua"

    def _compute_token_importance(self, input_ids: List[int]) -> torch.Tensor:
        if self.small_model is None:
            raise RuntimeError("No model available for perplexity computation")
        return _normalize(sliding_window_perplexity(self.small_model, input_ids))

    def _sentence_level_filter(self, input_ids: List[int], budget: int) -> List[int]:
        """Coarse sentence-level filtering: if the input has 4+ sentences,
        drop the least-important ones (mean token importance) before
        token-level compression."""
        sentence_end_tokens = set()
        for token_str in ['.', '!', '?', '\n', '。', '！', '？']:
            sentence_end_tokens.update(self.tokenizer.encode(token_str, add_special_tokens=False))

        sentences, current = [], []
        for tid in input_ids:
            current.append(tid)
            if tid in sentence_end_tokens or len(current) >= 200:
                sentences.append(current)
                current = []
        if current:
            sentences.append(current)
        if len(sentences) < 4:
            return input_ids

        all_ids_flat = [tid for sent in sentences for tid in sent]
        importance = self._compute_token_importance(all_ids_flat)

        offset = 0
        sent_importance = []
        for sent in sentences:
            sent_importance.append(importance[offset:offset + len(sent)].mean().item())
            offset += len(sent)

        sent_count = min(len(sentences), max(3, budget // 50))
        if sent_count >= len(sentences):
            return input_ids
        ranked = sorted(enumerate(sent_importance), key=lambda x: x[1], reverse=True)
        selected = sorted(i for i, _ in ranked[:sent_count])
        result = []
        for i in selected:
            result.extend(sentences[i])
        return result

    def compress(self, input_ids: List[int], **kwargs) -> CompressionResult:
        start = time.time()
        n = len(input_ids)
        if not self.validate_input(input_ids):
            return self._build_result(list(input_ids), n, (time.time() - start) * 1000)

        target_len = max(int(n / self.config.target_ratio), self.config.min_compressed_length)
        filtered_ids = self._sentence_level_filter(input_ids, target_len)

        if len(filtered_ids) > target_len * 1.5:
            importance = self._compute_token_importance(filtered_ids)
            k = self.config.keep_boundary_tokens
            mid_start, mid_end = k, len(filtered_ids) - k
            if mid_start < mid_end:
                mid_importance = importance[mid_start:mid_end]
                mid_budget = max(0, target_len - 2 * k)
                if 0 < mid_budget < len(mid_importance):
                    _, top_indices = torch.topk(mid_importance, mid_budget)
                    mid_kept = [filtered_ids[mid_start + i] for i in sorted(top_indices.tolist())]
                else:
                    mid_kept = filtered_ids[mid_start:mid_end]
                compressed = filtered_ids[:k] + mid_kept + filtered_ids[-k:]
            else:
                compressed = filtered_ids
        else:
            compressed = filtered_ids

        return self._build_result(
            compressed[:self.config.max_compressed_length], n, (time.time() - start) * 1000,
            metadata={'sentence_filtered': len(filtered_ids) < n},
        )


class SnapKVCompressor(BaseCompressor):
    """SnapKV-style KV-cache compression (Li et al. 2024, arxiv:2404.14469):
    identifies important tokens from attention patterns in an observation
    window at the end of the prompt. Training-free, model-agnostic. Also
    supports H2O ('heavy hitter' cumulative attention) and StreamingLLM
    (attention-sink + recency) selection modes."""

    def __init__(
        self, tokenizer, model=None, config: Optional[CompressionConfig] = None, device: str = 'cuda',
        window_size: int = 32, kernel_size: int = 5, max_capacity_prompt: int = 512,
        pooling: str = 'maxpool', budget_mode: str = 'uniform', mode: str = 'snapkv',
    ):
        super().__init__(tokenizer, model, config)
        self.device = device
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.max_capacity_prompt = max_capacity_prompt
        self.pooling = pooling
        self.budget_mode = budget_mode
        self.mode = mode

    def get_name(self) -> str:
        return f"SnapKV-{self.mode}"

    def _compute_attention_importance(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("SnapKV requires a model for attention computation")
        n = input_ids.shape[1]
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, use_cache=True)
        all_attentions = outputs.attentions
        if all_attentions is None:
            raise RuntimeError("Model did not return attention weights. Set output_attentions=True.")

        attn_stack = torch.stack(all_attentions, dim=0)  # [L, B, H, S, S]
        num_layers_to_use = max(1, len(all_attentions) // 4)
        recent_attns = attn_stack[-num_layers_to_use:]
        window_start = max(0, n - self.window_size)
        window_attn = recent_attns[:, :, :, window_start:, :]

        importance = window_attn.mean(dim=0).sum(dim=2).mean(dim=0)  # [H, S]
        if self.pooling in ('maxpool', 'avgpool') and self.kernel_size > 1:
            pool = F.max_pool1d if self.pooling == 'maxpool' else F.avg_pool1d
            imp = pool(importance.unsqueeze(0), kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
            importance = imp.squeeze(0)
        for h in range(importance.shape[0]):
            importance[h] = _normalize(importance[h])
        return importance  # [H, S]

    def _h2o_importance(self, input_ids, attention_mask=None) -> torch.Tensor:
        return self._compute_attention_importance(input_ids, attention_mask).mean(dim=0).unsqueeze(0)

    def _streamingllm_importance(self, n_tokens: int) -> torch.Tensor:
        importance = torch.zeros(1, n_tokens)
        importance[0, :4] = 1.0
        window = min(self.window_size, n_tokens - 4)
        if window > 0:
            importance[0, -window:] = 0.8
        mid_start, mid_end = 4, n_tokens - window
        if mid_start < mid_end:
            importance[0, mid_start:mid_end] = 0.1
        return importance

    def compress(self, input_ids: List[int], **kwargs) -> CompressionResult:
        start = time.time()
        n = len(input_ids)
        if not self.validate_input(input_ids):
            return self._build_result(list(input_ids), n, (time.time() - start) * 1000)

        input_tensor = torch.tensor([input_ids]).to(self.device)
        if self.mode == 'h2o':
            importance = self._h2o_importance(input_tensor)
        elif self.mode == 'streamingllm':
            importance = self._streamingllm_importance(n)
        else:
            importance = self._compute_attention_importance(input_tensor)
        num_heads = importance.shape[0]

        budget = min(self.max_capacity_prompt, max(int(n / self.config.target_ratio), self.config.min_compressed_length))
        kv_mask = torch.zeros(num_heads, n, dtype=torch.bool)
        k = self.config.keep_boundary_tokens
        for head in range(num_heads):
            head_imp = importance[head]
            mid_imp = head_imp[k:n - k] if n > 2 * k else head_imp
            mid_budget = max(0, budget - 2 * k)
            if 0 < mid_budget < len(mid_imp):
                _, top_indices = torch.topk(mid_imp, mid_budget)
                for idx in top_indices:
                    kv_mask[head, k + idx.item()] = True
            elif len(mid_imp) > 0:
                kv_mask[head, k:n - k] = True
            for i in range(min(k, n)):
                kv_mask[head, i] = True
            for i in range(max(0, n - k), n):
                kv_mask[head, i] = True

        keep_indices = kv_mask.any(dim=0).nonzero(as_tuple=True)[0].tolist()
        compressed_ids = [input_ids[i] for i in keep_indices]

        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        kv_memory_saved = (n - len(compressed_ids)) * (2 * num_heads * head_dim * 2)

        return self._build_result(
            compressed_ids, n, (time.time() - start) * 1000,
            metadata={'mode': self.mode, 'num_heads': num_heads, 'kv_memory_saved_bytes': kv_memory_saved},
        )


class SelectiveContextCompressor(BaseCompressor):
    """Keep tokens whose embedding is most similar to a query embedding
    (cosine similarity). Falls back to RandomCompressor without a
    model+query."""

    def __init__(self, tokenizer, model=None, config: Optional[CompressionConfig] = None, device: str = 'cuda'):
        super().__init__(tokenizer, model, config)
        self.device = device

    def get_name(self) -> str:
        return "SelectiveContext"

    def compress(self, input_ids: List[int], query_ids: Optional[List[int]] = None, **kwargs) -> CompressionResult:
        start = time.time()
        n = len(input_ids)
        if not self.validate_input(input_ids):
            return self._build_result(list(input_ids), n, (time.time() - start) * 1000)

        if self.model is None or not query_ids:
            return RandomCompressor(self.tokenizer, config=self.config).compress(input_ids)

        target_len = max(int(n / self.config.target_ratio), self.config.min_compressed_length)
        input_t = torch.tensor([input_ids]).to(self.device)
        query_t = torch.tensor([query_ids]).to(self.device)
        with torch.no_grad():
            input_emb = self.model.get_input_embeddings()(input_t).squeeze(0)
            query_vec = self.model.get_input_embeddings()(query_t).mean(dim=1)
            similarities = torch.mm(F.normalize(input_emb, dim=-1), F.normalize(query_vec, dim=-1).T).squeeze()

        k = self.config.keep_boundary_tokens
        mid_sim = similarities[k:n - k] if n > 2 * k else similarities
        mid_budget = max(0, target_len - 2 * k)
        if 0 < mid_budget < len(mid_sim):
            _, top_indices = torch.topk(mid_sim, mid_budget)
            compressed = input_ids[:k] + [input_ids[k + i] for i in sorted(top_indices.tolist())] + input_ids[n - k:]
        else:
            compressed = list(input_ids)

        return self._build_result(compressed, n, (time.time() - start) * 1000)


# ============================================================================
# Semantic Quality Gate (post-hoc safety net) & query-relevance boost
# ============================================================================


class SemanticQualityGate:
    """Restores highest-scoring dropped tokens until embedding similarity
    between the original and compressed sequence is >= `threshold`, or until
    `max_restore_fraction` of the dropped tokens have been restored --
    whichever comes first. Turns "compress to a fixed ratio no matter what"
    into "compress to a fixed ratio, unless that damages meaning too much".

    `similarity_fn(orig_ids, compressed_ids) -> float` is generic (no torch
    dependency here); `make_similarity_fn_from_model` below builds one from
    an actual model's hidden states.
    """

    def __init__(
        self, similarity_fn: Callable[[List[int], List[int]], float],
        threshold: float = 0.85, max_restore_fraction: float = 0.5, batch_fraction: float = 0.1,
    ):
        self.similarity_fn = similarity_fn
        self.threshold = threshold
        self.max_restore_fraction = max_restore_fraction
        self.batch_fraction = batch_fraction

    def apply(self, input_ids: List[int], retained_indices: Sequence[int], scores: Sequence[float]) -> Tuple[List[int], Dict]:
        n = len(input_ids)
        retained_set = set(retained_indices)
        dropped = sorted(i for i in range(n) if i not in retained_set)
        compressed_ids = [input_ids[i] for i in sorted(retained_set)]
        info: Dict = {'gate_fired': False, 'threshold': self.threshold, 'n_restored': 0, 'retained_indices': sorted(retained_set)}

        if not dropped:
            similarity = self.similarity_fn(input_ids, compressed_ids)
            info['initial_similarity'] = info['final_similarity'] = similarity
            return compressed_ids, info

        similarity = self.similarity_fn(input_ids, compressed_ids)
        info['initial_similarity'] = info['final_similarity'] = similarity
        if similarity >= self.threshold:
            return compressed_ids, info

        info['gate_fired'] = True
        dropped_by_score = sorted(dropped, key=lambda i: scores[i], reverse=True)
        max_restore = max(1, round(len(dropped) * self.max_restore_fraction))
        batch_size = max(1, round(len(dropped) * self.batch_fraction))

        restored, pos = 0, 0
        while similarity < self.threshold and restored < max_restore and pos < len(dropped_by_score):
            batch = dropped_by_score[pos:pos + min(batch_size, max_restore - restored)]
            if not batch:
                break
            retained_set.update(batch)
            pos += len(batch)
            restored += len(batch)
            compressed_ids = [input_ids[i] for i in sorted(retained_set)]
            similarity = self.similarity_fn(input_ids, compressed_ids)

        info['final_similarity'] = similarity
        info['n_restored'] = restored
        info['retained_indices'] = sorted(retained_set)
        return compressed_ids, info


def embed_ids_mean_pooled(model, input_ids: List[int], device=None) -> torch.Tensor:
    """Mean-pool a causal LM's last hidden state as a cheap sentence
    embedding -- avoids needing a separate embedding model."""
    if not input_ids:
        return torch.zeros(getattr(model.config, 'hidden_size', 1))
    dev = device or next(model.parameters()).device
    with torch.no_grad():
        outputs = model(torch.tensor([input_ids], device=dev), output_hidden_states=True)
    return outputs.hidden_states[-1][0].mean(dim=0).float().cpu()


def cosine_similarity(u: torch.Tensor, v: torch.Tensor) -> float:
    u, v = u.flatten(), v.flatten()
    denom = float(u.norm() * v.norm())
    return 1.0 if denom < 1e-8 else float(torch.dot(u, v) / denom)


def make_similarity_fn_from_model(model, device: Optional[str] = None) -> Callable[[List[int], List[int]], float]:
    """Build a (orig_ids, compressed_ids) -> cosine_similarity callable
    backed by a real model's hidden states, for SemanticQualityGate."""

    def _similarity_fn(orig_ids: List[int], comp_ids: List[int]) -> float:
        return cosine_similarity(
            embed_ids_mean_pooled(model, orig_ids, device),
            embed_ids_mean_pooled(model, comp_ids, device),
        )

    return _similarity_fn


def compute_query_relevance_weights(
    tokens: List[str], query: Optional[str], boost: float = 1.5, min_token_len: int = 2,
) -> List[float]:
    """Per-token multiplier for the optional downstream `query` signal: a
    token appearing as a whole word in `query` (case-insensitive, at least
    `min_token_len` characters) gets `boost`, everything else gets 1.0.
    Lexical, not embedding-based -- works with any tokenizer, no model needed."""
    if not query:
        return [1.0] * len(tokens)
    import re as _re

    query_words = {w.lower() for w in _re.findall(r"\w+", query, flags=_re.UNICODE)}
    if not query_words:
        return [1.0] * len(tokens)
    return [boost if len(t := tok.strip().lower()) >= min_token_len and t in query_words else 1.0 for tok in tokens]


# ============================================================================
# LACC: signal blend weights + external-scorer wrapper + the core compressor
# ============================================================================


@dataclass
class ScoreWeights:
    """Blend weights for LACC's three signals. Automatically renormalized to
    sum to 1.0 in __post_init__ -- callers (including LACCCompressor's
    ablation config) can pass raw, un-normalized weights."""

    perplexity: float = 0.40
    tone: float = 0.30
    morphology: float = 0.30

    def __post_init__(self):
        total = self.perplexity + self.tone + self.morphology
        if total <= 0:
            raise ValueError("ScoreWeights: at least one of perplexity/tone/morphology must be > 0")
        if total != 1.0:
            self.perplexity /= total
            self.tone /= total
            self.morphology /= total

    def to_dict(self) -> Dict[str, float]:
        return {'perplexity': self.perplexity, 'tone': self.tone, 'morphology': self.morphology}


class LACCScorer:
    """Wraps a (possibly different-tokenizer) SLM as the source of LACC's
    perplexity and/or trained-tone-probe signals -- built by
    `models.load_scorer()`.

    Works on raw text rather than caller-supplied token ids, precisely so it
    can be paired with a generation model that tokenizes differently (e.g. a
    small Vietnamese GPT-2 scorer alongside a Qwen generation model):
    `char_signals()` returns per-CHARACTER scores, which LACCCompressor then
    pools back onto its own tokenizer's token spans.
    """

    def __init__(self, model, tokenizer, tone_probe=None, window_size: int = 512, stride: Optional[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.tone_probe = tone_probe  # linguistics.PhonologicalConsistencyLoss, or None
        self.window_size = window_size
        self.stride = stride if stride is not None else max(1, window_size // 2)

    @torch.no_grad()
    def _token_signals(self, ids: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-SLM-token (perplexity, model-tone) computed in a single
        sliding-window pass -- tone defaults to neutral 1.0 where no probe is set."""
        n = len(ids)
        ppl, tone = torch.zeros(n), torch.ones(n)
        if n == 0:
            return ppl, tone
        device = next(self.model.parameters()).device
        window = self.window_size
        stride = max(1, min(self.stride, window - 1)) if window > 1 else 1

        begin, prev_end = 0, 0
        while begin < max(n - 1, 1):
            end = min(begin + window, n)
            chunk = list(ids[begin:end])
            if len(chunk) < 2:
                break
            tensor = torch.tensor([chunk], device=device)
            out = self.model(tensor, output_hidden_states=self.tone_probe is not None)
            log_probs = F.log_softmax(out.logits.float()[:, :-1, :], dim=-1)
            token_lp = log_probs.gather(-1, tensor[:, 1:].unsqueeze(-1)).squeeze(-1)
            importance = (-token_lp[0]).cpu()
            if begin == 0 and importance.numel() > 0:
                ppl[0] = importance[0]
            ppl_start = 1 if begin == 0 else prev_end
            for k in range(importance.numel()):
                pos = begin + k + 1
                if ppl_start <= pos < n:
                    ppl[pos] = importance[k]

            if self.tone_probe is not None:
                h = out.hidden_states[-1].to(self.tone_probe.tone_classifier[0].weight.dtype)
                tone_imp = self.tone_probe.score_importance(h)[0].cpu()
                tone_start = 0 if begin == 0 else prev_end
                for j in range(tone_imp.numel()):
                    pos = begin + j
                    if tone_start <= pos < n:
                        tone[pos] = tone_imp[j]

            prev_end = end
            if end >= n:
                break
            begin += stride
        return ppl, tone

    def _offsets(self, text: str):
        """(input_ids, char offsets) for this scorer's own tokenization."""
        try:
            enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
            return enc['input_ids'], enc['offset_mapping']
        except (TypeError, NotImplementedError, ValueError):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            offsets, cursor = [], 0
            for tid in ids:
                piece = self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                if not piece:
                    offsets.append((cursor, cursor))
                    continue
                found = text.find(piece, cursor)
                if found < 0:
                    offsets.append((cursor, min(cursor + len(piece), len(text))))
                    cursor = min(cursor + len(piece), len(text))
                else:
                    offsets.append((found, found + len(piece)))
                    cursor = found + len(piece)
            return ids, offsets

    def char_signals(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-character (perplexity, model-tone); NaN where no scorer token
        covers the character, so the caller can distinguish that from a real 0."""
        ppl_char = torch.full((len(text),), float('nan'))
        tone_char = torch.full((len(text),), float('nan'))
        if not text:
            return ppl_char, tone_char
        ids, offsets = self._offsets(text)
        if not ids:
            return ppl_char, tone_char
        ppl, tone = self._token_signals(ids)
        for (start, end), pv, tv in zip(offsets, ppl.tolist(), tone.tolist()):
            if end > start:
                ppl_char[start:end] = pv
                tone_char[start:end] = tv
        return ppl_char, tone_char

    def to(self, device: str) -> "LACCScorer":
        self.model.to(device)
        return self


class LACCCompressor(BaseCompressor):
    """Language-Aware Context Compression -- the proposed method.

    S(t) = w_ppl * S_ppl(t) + w_tone * S_tone(t) + w_morph * S_morph(t)

    Args:
        use_perplexity/use_tone/use_morphology: enable/disable a signal.
            Ablation is just config -- e.g. `use_morphology=False` for the
            "ppl + tone" arm. Disabled signals get weight 0 (renormalized
            among the enabled ones) and are not computed.
        scorer: an optional LACCScorer (see models.load_scorer()) supplying
            perplexity/tone from a *different* model+tokenizer than `model`
            (the lightweight hardware tier: a small SLM scores tokens while
            `model` -- or no model at all -- handles generation). When unset,
            the perplexity/model-tone signals (if enabled) come from `model`
            directly, using this compressor's own tokenizer.
        tone_source: 'rule' (dictionary/phonology heuristic, always
            available) or 'model' (the trained tone probe -- via `scorer`, or
            via `tone_probe` scored on `model`'s own hidden states).
        tone_probe: a linguistics.PhonologicalConsistencyLoss trained on
            `model`'s own hidden states (only meaningful when `scorer` is
            unset -- a probe only means anything on the model it was trained
            with).
        merge_redup_pairs: soften the morphology score of a reduplicative
            pair's redundant second syllable (see linguistics.MorphologyAnalyzer
            .find_reduplicative_pairs).
        weights: base ScoreWeights before disabled signals are zeroed out.
        quality_gate: optional SemanticQualityGate (restores tokens if
            compression drops semantic similarity too far).

    Examples:
        LACCCompressor(tok, model)                                    # ppl + tone + morph (full)
        LACCCompressor(tok, None, use_perplexity=False)                # tone + morph, 0 VRAM
        LACCCompressor(tok, model, use_tone=False, use_morphology=False)  # ppl only
        LACCCompressor(tok, model, scorer=slm_scorer, tone_source='model')  # trained tone probe
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Optional[PreTrainedModel] = None,
        config: Optional[CompressionConfig] = None,
        device: str = 'cuda',
        use_perplexity: bool = True,
        use_tone: bool = True,
        use_morphology: bool = True,
        scorer: Optional[LACCScorer] = None,
        tone_source: str = 'rule',
        tone_probe=None,
        alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.4,
        tone_contrast: Optional[Dict[Tuple[str, str], float]] = None,
        tone_window: int = 2,
        f_func: float = 0.4, f_content: float = 1.2, f_redup: float = 0.6,
        f_compound: float = 1.5, f_sino: float = 1.5, f_other: float = 1.0,
        merge_redup_pairs: bool = True,
        weights: Optional[ScoreWeights] = None,
        quality_gate: Optional[SemanticQualityGate] = None,
        query_boost: float = 1.5,
        name: Optional[str] = None,
    ):
        super().__init__(tokenizer, model, config)
        if tone_source not in ('rule', 'model'):
            raise ValueError(f"tone_source must be 'rule' or 'model'; got {tone_source!r}")

        self.device = device
        self.use_perplexity = use_perplexity
        self.use_tone = use_tone
        self.use_morphology = use_morphology
        self.scorer = scorer
        self.tone_source = tone_source
        self._tone_probe = tone_probe
        self.tone_window = tone_window
        self.merge_redup_pairs = merge_redup_pairs
        self.quality_gate = quality_gate
        self.default_query_boost = query_boost
        self._name = name

        self.tone_analyzer: VietnameseToneAnalyzer = get_tone_analyzer(alpha=alpha, beta=beta, gamma=gamma, tone_contrast=tone_contrast)
        self.morph_analyzer: MorphologyAnalyzer = get_morphology_analyzer()
        self.morph_config = MorphologyConfig(
            f_func=f_func, f_content=f_content, f_redup=f_redup,
            f_compound=f_compound, f_sino=f_sino, f_other=f_other,
        )

        base = weights or ScoreWeights(perplexity=0.4, tone=0.3, morphology=0.3)
        raw = {
            'perplexity': base.perplexity if use_perplexity else 0.0,
            'tone': base.tone if use_tone else 0.0,
            'morphology': base.morphology if use_morphology else 0.0,
        }
        if not any(raw.values()):
            raise ValueError("LACCCompressor needs at least one of use_perplexity/use_tone/use_morphology enabled")
        self.weights = ScoreWeights(**raw)  # renormalizes to sum 1.0

    def get_name(self) -> str:
        if self._name:
            return self._name
        parts = []
        if self.use_perplexity:
            parts.append('ppl' + ('-slm' if self.scorer else ''))
        if self.use_tone:
            parts.append('tone' + ('-probe' if self.tone_source == 'model' else ''))
        if self.use_morphology:
            parts.append('morph')
        return 'LACC[' + '+'.join(parts) + ']'

    def _compute_model_tone_weights(self, input_ids: List[int]) -> torch.Tensor:
        """Trained tone probe scored on `model`'s own hidden states (same
        tokenizer as `self.tokenizer`) -- only meaningful when the probe was
        trained on this exact model."""
        input_t = torch.tensor([input_ids], device=next(self.model.parameters()).device)
        with torch.no_grad():
            hidden_states = self.model(input_t, output_hidden_states=True).hidden_states[-1]
        self._tone_probe.to(hidden_states.device)
        return self._tone_probe.score_importance(hidden_states).squeeze(0).cpu()

    def compress(self, input_ids: List[int], query: Optional[str] = None, query_boost: Optional[float] = None, **kwargs) -> CompressionResult:
        start = time.time()
        n = len(input_ids)
        if not self.validate_input(input_ids):
            return self._build_result(list(input_ids), n, (time.time() - start) * 1000)

        tokens = _decode_tokens(self.tokenizer, input_ids)

        # --- perplexity signal ------------------------------------------------
        if self.use_perplexity and self.scorer is not None:
            text, spans, _ = _token_spans(self.tokenizer, input_ids)
            ppl_char, _ = self.scorer.char_signals(text)
            ppl_scores = _pool_char_to_token(ppl_char, spans, n, fill_neutral=True)
        elif self.use_perplexity and self.model is not None:
            ppl_scores = sliding_window_perplexity(self.model, input_ids)
        else:
            ppl_scores = torch.full((n,), 0.5)

        # --- tone signal --------------------------------------------------------
        tone_infos = self.tone_analyzer.analyze_tokens(tokens, window_size=self.tone_window)
        rule_tone_scores = torch.tensor([max(0.5, min(3.0, info.preservation_weight)) for info in tone_infos]) \
            if tone_infos else torch.ones(n)
        if not self.use_tone:
            tone_scores = torch.ones(n)
        elif self.tone_source == 'model' and self.scorer is not None and self.scorer.tone_probe is not None:
            text, spans, _ = _token_spans(self.tokenizer, input_ids)
            _, tone_char = self.scorer.char_signals(text)
            tone_scores = _pool_char_to_token(tone_char, spans, n, fill_neutral=False)
            # A token the scorer never covered falls back to the rule weight,
            # rather than being (wrongly) treated as tone-neutral.
            uncovered = tone_scores == 0
            if uncovered.any():
                tone_scores[uncovered] = rule_tone_scores[uncovered]
        elif self.tone_source == 'model' and self._tone_probe is not None and self.model is not None:
            tone_scores = self._compute_model_tone_weights(input_ids)
        else:
            tone_scores = rule_tone_scores

        # --- morphology signal ---------------------------------------------------
        word_infos = self.morph_analyzer.classify_batch(tokens)
        n_redup_pairs = 0
        if self.use_morphology:
            morph_scores = torch.tensor([
                self.morph_analyzer.get_preservation_multiplier(info, self.morph_config) for info in word_infos
            ])
            if self.merge_redup_pairs:
                pairs = self.morph_analyzer.find_reduplicative_pairs([w.token for w in word_infos], config=self.morph_config)
                n_redup_pairs = len(pairs)
                for _, right, _ in pairs:
                    if right < len(morph_scores):
                        morph_scores[right] *= 0.3  # redundant second syllable of a pair
        else:
            morph_scores = torch.ones(n)

        combined = (
            self.weights.perplexity * _normalize(ppl_scores)
            + self.weights.tone * _normalize(tone_scores)
            + self.weights.morphology * _normalize(morph_scores)
        )

        if query:
            qw = torch.tensor(compute_query_relevance_weights(tokens, query, boost=query_boost or self.default_query_boost))
            combined = combined * qw

        retained_indices = self.select_with_boundary(combined.tolist(), n)

        gate_info: Dict = {}
        if self.quality_gate is not None:
            compressed, gate_info = self.quality_gate.apply(input_ids, retained_indices, combined.tolist())
            retained_set = set(gate_info['retained_indices'])
        else:
            compressed = [input_ids[i] for i in retained_indices]
            retained_set = set(retained_indices)

        metadata = {
            'signals': {'perplexity': self.use_perplexity, 'tone': self.use_tone, 'morphology': self.use_morphology},
            'weights': self.weights.to_dict(),
            'tone_source': self.tone_source if self.use_tone else None,
            'tone_preservation_rate': compute_tone_preservation_rate(tone_infos, retained_set),
            'query_applied': bool(query),
            'redup_pairs_found': n_redup_pairs,
        }
        if gate_info:
            metadata['quality_gate'] = gate_info

        return self._build_result(compressed, n, (time.time() - start) * 1000, metadata)


# ============================================================================
# Method registry
# ============================================================================

METHODS: Dict[str, type] = {
    'none': NoCompressor,
    'random': RandomCompressor,
    'llmlingua': LLMLinguaCompressor,
    'snapkv': SnapKVCompressor,
    'selective': SelectiveContextCompressor,
    'lacc': LACCCompressor,
}


def create_compressor(
    method: str, tokenizer, model=None, config: Optional[CompressionConfig] = None, device: str = 'cuda', **kwargs,
) -> BaseCompressor:
    """Build a compressor by name. `kwargs` are forwarded to the class --
    for 'lacc' that includes use_perplexity/use_tone/use_morphology/scorer/
    tone_source/... (see LACCCompressor)."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method!r}. Available: {list(METHODS)}")
    cls = METHODS[method]
    if method in ('none', 'random'):
        return cls(tokenizer, model=model, config=config)
    return cls(tokenizer, model=model, config=config, device=device, **kwargs)


# ============================================================================
# Weight calibration (LACC's blend weights / tone & morphology parameters)
# ============================================================================
#
# w_ppl/w_tone/w_morph, the tone formula's alpha/beta/gamma, and the per-class
# morphology multipliers were originally hand-picked constants. This search
# replaces "pick numbers that feel right" with a small, dependency-light
# search over a validation set of Vietnamese texts, using a self-supervised
# proxy objective (running the full generation model for every candidate
# parameter set would be far too expensive for a search loop):
#
#   quality    = 0.5 * tone_preservation_rate + 0.5 * content_word_retention
#   efficiency = token_savings_pct / 100
#   objective  = harmonized_score = 2 * quality * efficiency / (quality + efficiency)
#
# This mirrors the harmonized_score VCCBench reports, so a winning parameter
# set here is directly comparable to VCC-Bench numbers -- though it's still
# worth confirming the winner with a full benchmark run.

DEFAULT_PARAM_CANDIDATES: Dict[str, List[float]] = {
    'alpha': [0.3, 0.5, 0.7],
    'beta': [0.2, 0.3, 0.4],
    'gamma': [0.2, 0.4, 0.6],
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
    return {name: values[len(values) // 2] for name, values in param_candidates.items()}


@dataclass
class _SampleResult:
    compressed_ids: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _content_word_retention(orig_tokens: List[str], comp_tokens: List[str], morph_analyzer: MorphologyAnalyzer) -> float:
    from .linguistics import WordClass

    orig_content = sum(1 for info in morph_analyzer.classify_batch(orig_tokens) if info.word_class == WordClass.CONTENT)
    if orig_content == 0:
        return 1.0
    comp_content = sum(1 for info in morph_analyzer.classify_batch(comp_tokens) if info.word_class == WordClass.CONTENT)
    return min(1.0, comp_content / orig_content)


def _approx_tone_preservation_rate(orig_tokens: List[str], comp_tokens: List[str], tone_analyzer: VietnameseToneAnalyzer) -> float:
    """Multiset-based fallback when a result doesn't carry an exact,
    index-based tone_preservation_rate in its metadata."""
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


class CalibrationObjective:
    """Scores a compressor factory (params -> compressor) against a fixed
    validation set of texts using the proxy quality/efficiency objective
    described above."""

    def __init__(
        self, tokenizer, texts: Sequence[str],
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
        self._input_ids_cache = [tokenizer.encode(t, add_special_tokens=False) for t in texts]

    def _score_one(self, input_ids: List[int], result: _SampleResult) -> Dict[str, float]:
        n, m = len(input_ids), len(result.compressed_ids)
        efficiency = max(0.0, ((n - m) / n * 100 if n else 0.0) / 100.0)
        orig_tokens = _decode_tokens(self.tokenizer, input_ids)
        comp_tokens = _decode_tokens(self.tokenizer, result.compressed_ids)

        tone_rate = result.metadata.get(
            'tone_preservation_rate',
            _approx_tone_preservation_rate(orig_tokens, comp_tokens, self.tone_analyzer),
        )
        content_retention = _content_word_retention(orig_tokens, comp_tokens, self.morph_analyzer)
        quality = 0.5 * tone_rate + 0.5 * content_retention
        denom = quality + efficiency
        harmonized = (2 * quality * efficiency / denom) if denom > 1e-8 else 0.0
        return {
            'quality': quality, 'efficiency': efficiency, 'harmonized_score': harmonized,
            'tone_preservation_rate': tone_rate, 'content_word_retention': content_retention,
        }

    def evaluate(self, compressor_factory: Callable[[Dict[str, float]], Any], params: Dict[str, float]) -> Dict[str, float]:
        compressor = compressor_factory(params)
        per_sample = []
        for input_ids in self._input_ids_cache:
            compressor.config.target_ratio = self.target_ratio
            raw = compressor.compress(input_ids)
            result = _SampleResult(compressed_ids=list(raw.compressed_ids), metadata=dict(raw.metadata or {}))
            per_sample.append(self._score_one(input_ids, result))
        agg = {}
        for key in ('quality', 'efficiency', 'harmonized_score', 'tone_preservation_rate', 'content_word_retention'):
            values = [s[key] for s in per_sample]
            agg[key] = sum(values) / len(values) if values else 0.0
        agg['n_samples'] = len(per_sample)
        return agg


def grid_search(
    objective: CalibrationObjective, compressor_factory: Callable[[Dict[str, float]], Any],
    param_candidates: Dict[str, List[float]], max_combinations: int = 200, seed: int = 0,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Exhaustive Cartesian-product search (subsampled to `max_combinations`
    if the full grid is larger). Prefer coordinate_ascent for many parameters."""
    names = list(param_candidates.keys())
    all_combos = list(itertools.product(*(param_candidates[n] for n in names)))
    if len(all_combos) > max_combinations:
        all_combos = random.Random(seed).sample(all_combos, max_combinations)

    history: List[Dict[str, Any]] = []
    best_params, best_score = None, float('-inf')
    for combo in all_combos:
        params = dict(zip(names, combo))
        metrics = objective.evaluate(compressor_factory, params)
        history.append({'params': dict(params), 'metrics': metrics})
        if metrics['harmonized_score'] > best_score:
            best_score, best_params = metrics['harmonized_score'], params
    return best_params or _default_initial(param_candidates), history


def coordinate_ascent(
    objective: CalibrationObjective, compressor_factory: Callable[[Dict[str, float]], Any],
    param_candidates: Dict[str, List[float]], initial_params: Optional[Dict[str, float]] = None, rounds: int = 2,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Sweep one parameter at a time holding others fixed at the current
    best. O(rounds * sum(len(candidates))) evaluations instead of the full
    Cartesian product (e.g. ~48 vs 3^7 for the 7-parameter default space)."""
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
                    best_score, current, improved = metrics['harmonized_score'], trial, True
        if not improved:
            break
    return current, history


def calibrate_lacc_compressor(
    tokenizer, model, texts: Sequence[str], config: CompressionConfig, device: str = 'cuda',
    param_candidates: Optional[Dict[str, List[float]]] = None,
    initial_params: Optional[Dict[str, float]] = None, rounds: int = 2,
    target_ratio: float = 4.0, strategy: str = 'coordinate_ascent',
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Calibrate LACCCompressor's tone alpha/beta/gamma and per-class
    morphology multipliers against a validation text set."""
    param_candidates = param_candidates or DEFAULT_PARAM_CANDIDATES
    objective = CalibrationObjective(tokenizer, texts, target_ratio=target_ratio)

    def factory(params: Dict[str, float]):
        return LACCCompressor(
            tokenizer, model, config, device,
            alpha=params['alpha'], beta=params['beta'], gamma=params['gamma'],
            f_func=params['f_func'], f_content=params['f_content'],
            f_redup=params['f_redup'], f_compound=params['f_compound'],
        )

    if strategy == 'grid_search':
        return grid_search(objective, factory, param_candidates)
    return coordinate_ascent(objective, factory, param_candidates, initial_params, rounds)


def calibrate_score_weights(
    tokenizer, model, texts: Sequence[str], config: CompressionConfig, device: str = 'cuda',
    weight_candidates: Optional[Dict[str, List[float]]] = None, rounds: int = 2,
    target_ratio: float = 4.0, strategy: str = 'coordinate_ascent',
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Calibrate LACCCompressor's ScoreWeights (w_ppl/w_tone/w_morph) against
    a validation text set."""
    weight_candidates = weight_candidates or DEFAULT_SCORE_WEIGHT_CANDIDATES
    objective = CalibrationObjective(tokenizer, texts, target_ratio=target_ratio)

    def factory(params: Dict[str, float]):
        weights = ScoreWeights(perplexity=params['perplexity'], tone=params['tone'], morphology=params['morphology'])
        return LACCCompressor(tokenizer, model, config, device, weights=weights)

    if strategy == 'grid_search':
        return grid_search(objective, factory, weight_candidates)
    return coordinate_ascent(objective, factory, weight_candidates, rounds=rounds)
