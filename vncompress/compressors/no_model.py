"""
No-Model Compressors — CPU-only compression (no GPU needed)
=============================================================

These compressors work WITHOUT any neural model — they only use:
  1. Tone analysis (VietnameseToneAnalyzer) — dictionary-based
  2. Morphology analysis (MorphologyAnalyzer) — dictionary-based  
  3. Simple heuristics (random, boundary, word-length)

Why no-model scoring works (and when you might want more):
  - Vietnamese has FIXED tone marks (à, á, ả, ã, ạ) → deterministic weight
  - Vietnamese has known function words (đã, sẽ, của...) → deterministic weight
  - These are formula-based, not learned — they use static linguistic knowledge
  - Quality is GOOD for tone/morphology preservation tasks

For BETTER quality, use an external tiny model for perplexity scoring:
  >>> from vncompress.compressors.external_scorer import create_tiny_scorer, EnhancedCompressor
  >>> scorer = create_tiny_scorer('smollm2-135m')  # 135M params, 0.3GB VRAM (INT4)
  >>> comp = EnhancedCompressor(tokenizer, scorer)
  >>> compressed_ids, stats = comp.compress(input_ids, target_ratio=4.0)
  
This combines:
  - Perplexity scoring (neural, from external model) → 40% weight
  - Tone preservation (linguistic) → 30% weight
  - Morphology preservation (linguistic) → 30% weight

Why this matters for limited hardware:
  - T4/P100 have only 16GB VRAM
  - Loading a 7B model in INT4 uses ~5GB, leaving limited room
  - No-model compression uses 0MB VRAM → all GPU memory for generation
  - Compression quality is still decent (tone/morphology signal is strong)

Use case:
  1. Run no-model compression on CPU (free)
  2. Load main model in GPU ONLY for generation
  3. Feed compressed context → model → output

VRAM saved: ~5GB (the INT4 model needed for scoring)
"""

import random
import math
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..tone_aware.vietnamese_tones import (
    VietnameseToneAnalyzer,
    get_tone_analyzer,
    TokenToneInfo,
)
from ..morphology.merge_policy import (
    MorphologyAnalyzer,
    get_morphology_analyzer,
    WordClass,
    WordInfo,
)


@dataclass
class NoModelResult:
    """Result from no-model compression (lightweight, no torch dependency)."""
    compressed_ids: List[int]
    original_length: int
    compressed_length: int
    compression_ratio: float
    token_savings_pct: float
    processing_time_ms: float
    metadata: dict


class NoModelCompressor:
    """
    Base class for compressors that don't need a neural model.
    
    All computation is on CPU, using only:
      - String operations
      - Tone analysis (dictionary-based)
      - Morphology analysis (dictionary-based)
    
    No torch, no GPU, no VRAM usage.
    """
    
    def __init__(
        self,
        tokenizer,
        tone_analyzer: Optional[VietnameseToneAnalyzer] = None,
        morph_analyzer: Optional[MorphologyAnalyzer] = None,
    ):
        self.tokenizer = tokenizer
        self.tone_analyzer = tone_analyzer or get_tone_analyzer()
        self.morph_analyzer = morph_analyzer or get_morphology_analyzer()
    
    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """Decode token IDs to clean strings."""
        tokens = []
        for tid in token_ids:
            t = self.tokenizer.decode([tid])
            t = t.replace('\u2581', ' ').replace('Ġ', ' ').strip()
            tokens.append(t)
        return tokens
    
    def _compute_ratio(self, orig: int, compressed: int) -> Tuple[float, float]:
        cr = orig / max(compressed, 1)
        ts = ((orig - compressed) / max(orig, 1)) * 100
        return cr, ts
    
    def _build_result(
        self, compressed_ids: List[int], orig_len: int, elapsed_ms: float,
        metadata: dict = None,
    ) -> NoModelResult:
        cr, ts = self._compute_ratio(orig_len, len(compressed_ids))
        return NoModelResult(
            compressed_ids=compressed_ids,
            original_length=orig_len,
            compressed_length=len(compressed_ids),
            compression_ratio=cr,
            token_savings_pct=ts,
            processing_time_ms=elapsed_ms,
            metadata=metadata or {},
        )
    
    def compress(self, input_ids: List[int], target_ratio: float = 4.0) -> NoModelResult:
        raise NotImplementedError


class NoModelToneCompressor(NoModelCompressor):
    """
    Tone-only compression — no model needed.
    
    Score = tone_preservation_weight(t) for each token.
    Keep tokens with highest tone importance.
    
    This preserves characters with diacritics/tone marks,
    which are critical for Vietnamese meaning.
    
    No GPU needed — runs on CPU in milliseconds.
    """
    
    def compress(
        self,
        input_ids: List[int],
        target_ratio: float = 4.0,
        keep_boundary: int = 2,
    ) -> NoModelResult:
        start = time.time()
        n = len(input_ids)
        target_len = max(int(n / target_ratio), 2 * keep_boundary)
        
        tokens = self.decode_tokens(input_ids)
        tone_infos = self.tone_analyzer.analyze_tokens(tokens)
        
        # Score = preservation_weight
        scores = [info.preservation_weight for info in tone_infos]
        
        # Always keep boundaries
        k = keep_boundary
        mid_indices = list(range(k, n - k)) if n > 2 * k else []
        mid_scores = [(i, scores[i]) for i in mid_indices]
        
        # Sort by score descending
        mid_scores.sort(key=lambda x: x[1], reverse=True)
        mid_budget = max(0, target_len - 2 * k)
        
        keep_indices = set(range(min(k, n)))
        keep_indices.update(range(max(k, n - k), n))
        
        for i, (idx, _) in enumerate(mid_scores):
            if i >= mid_budget:
                break
            keep_indices.add(idx)
        
        compressed = [input_ids[i] for i in sorted(keep_indices) if i < n]
        elapsed = (time.time() - start) * 1000
        
        return self._build_result(compressed, n, elapsed, {
            'method': 'no_model_tone',
            'mean_tone_weight': sum(scores) / len(scores) if scores else 0,
        })


class NoModelMorphCompressor(NoModelCompressor):
    """
    Morphology-only compression — no model needed.
    
    Strategy:
      - Function words: keep 30%  (aggressive compression)
      - Content words:  keep 85%  (preserve meaning)
      - Reduplicative:  keep 50%  (merge pairs)
      - Compounds:      keep 95%  (preserve as unit)
    
    No GPU needed — runs on CPU in milliseconds.
    """
    
    def compress(
        self,
        input_ids: List[int],
        target_ratio: float = 4.0,
        keep_boundary: int = 2,
    ) -> NoModelResult:
        start = time.time()
        n = len(input_ids)
        target_len = max(int(n / target_ratio), 2 * keep_boundary)
        
        tokens = self.decode_tokens(input_ids)
        word_infos = self.morph_analyzer.classify_batch(tokens)
        
        # Class-aware keep ratios
        keep_ratios = {
            WordClass.FUNC.value: 0.30,
            WordClass.CONTENT.value: 0.85,
            WordClass.REDUP.value: 0.50,
            WordClass.COMPOUND.value: 0.95,
            WordClass.OTHER.value: 0.50,
        }
        
        # Group by class
        k = keep_boundary
        mid_indices = list(range(k, n - k)) if n > 2 * k else []
        
        class_buckets: Dict[str, List[int]] = {}
        for i in mid_indices:
            cls = word_infos[i].word_class.value if i < len(word_infos) else 'other'
            if cls not in class_buckets:
                class_buckets[cls] = []
            class_buckets[cls].append(i)
        
        # Select per class (random within class, but proportional to keep ratio)
        keep_indices = set(range(min(k, n)))
        keep_indices.update(range(max(k, n - k), n))
        
        for cls, indices in class_buckets.items():
            ratio = keep_ratios.get(cls, 0.5)
            keep_n = max(0, int(len(indices) * ratio))
            selected = random.sample(indices, min(keep_n, len(indices)))
            keep_indices.update(selected)
        
        compressed = [input_ids[i] for i in sorted(keep_indices) if i < n]
        elapsed = (time.time() - start) * 1000
        
        # Class distribution stats
        class_counts = {}
        for info in word_infos:
            cls = info.word_class.value
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        return self._build_result(compressed, n, elapsed, {
            'method': 'no_model_morph',
            'class_distribution': class_counts,
        })


class NoModelCombinedCompressor(NoModelCompressor):
    """
    Combined tone + morphology — no model needed.
    
    Score = tone_weight × morphology_multiplier
    
    Best of both worlds without any GPU usage.
    """
    
    def compress(
        self,
        input_ids: List[int],
        target_ratio: float = 4.0,
        keep_boundary: int = 2,
        tone_weight: float = 0.5,
    ) -> NoModelResult:
        start = time.time()
        n = len(input_ids)
        target_len = max(int(n / target_ratio), 2 * keep_boundary)
        
        tokens = self.decode_tokens(input_ids)
        
        # Tone analysis
        tone_infos = self.tone_analyzer.analyze_tokens(tokens)
        tone_scores = [info.preservation_weight for info in tone_infos]
        
        # Morphology analysis
        word_infos = self.morph_analyzer.classify_batch(tokens)
        
        # Morph multipliers
        morph_mult = {
            WordClass.FUNC: 0.4,
            WordClass.CONTENT: 1.2,
            WordClass.REDUP: 0.6,
            WordClass.COMPOUND: 1.5,
            WordClass.OTHER: 1.0,
        }
        morph_scores = [
            morph_mult.get(info.word_class, 1.0)
            for info in word_infos
        ]
        
        # Combined: weighted blend
        combined = [
            tone_weight * t + (1 - tone_weight) * m
            for t, m in zip(tone_scores, morph_scores)
        ]
        
        # Select top
        k = keep_boundary
        mid_indices = list(range(k, n - k)) if n > 2 * k else []
        mid_scores = [(i, combined[i]) for i in mid_indices]
        mid_scores.sort(key=lambda x: x[1], reverse=True)
        mid_budget = max(0, target_len - 2 * k)
        
        keep_indices = set(range(min(k, n)))
        keep_indices.update(range(max(k, n - k), n))
        
        for i, (idx, _) in enumerate(mid_scores):
            if i >= mid_budget:
                break
            keep_indices.add(idx)
        
        compressed = [input_ids[i] for i in sorted(keep_indices) if i < n]
        elapsed = (time.time() - start) * 1000
        
        return self._build_result(compressed, n, elapsed, {
            'method': 'no_model_combined',
            'tone_weight': tone_weight,
            'mean_tone_score': sum(tone_scores) / len(tone_scores) if tone_scores else 0,
            'mean_morph_score': sum(morph_scores) / len(morph_scores) if morph_scores else 0,
        })


class NoModelBaselineCompressor(NoModelCompressor):
    """Simple baselines that don't need a model."""
    
    MODES = ['first', 'random', 'every_nth', 'word_length']
    
    def __init__(self, tokenizer, mode: str = 'random', **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.mode = mode
    
    def compress(
        self,
        input_ids: List[int],
        target_ratio: float = 4.0,
        keep_boundary: int = 2,
    ) -> NoModelResult:
        start = time.time()
        n = len(input_ids)
        target_len = max(int(n / target_ratio), 2 * keep_boundary)
        
        k = keep_boundary
        mid_indices = list(range(k, n - k)) if n > 2 * k else []
        mid_budget = max(0, target_len - 2 * k)
        
        if self.mode == 'first':
            # Keep first N tokens
            selected_mid = mid_indices[:mid_budget]
        elif self.mode == 'every_nth':
            # Keep every Nth token
            step = max(1, len(mid_indices) // max(mid_budget, 1))
            selected_mid = mid_indices[::step][:mid_budget]
        elif self.mode == 'word_length':
            # Keep longest tokens (more information per token)
            tokens = self.decode_tokens(input_ids)
            mid_tokens = [(i, len(tokens[i])) for i in mid_indices]
            mid_tokens.sort(key=lambda x: x[1], reverse=True)
            selected_mid = [i for i, _ in mid_tokens[:mid_budget]]
        else:  # random
            selected_mid = random.sample(
                mid_indices,
                min(mid_budget, len(mid_indices))
            )
        
        keep_indices = set(range(min(k, n)))
        keep_indices.update(range(max(k, n - k), n))
        keep_indices.update(selected_mid)
        
        compressed = [input_ids[i] for i in sorted(keep_indices) if i < n]
        elapsed = (time.time() - start) * 1000
        
        return self._build_result(compressed, n, elapsed, {
            'method': f'no_model_{self.mode}',
        })


# ============================================================================
# Quick evaluation helper (no GPU needed)
# ============================================================================

def evaluate_no_model(
    text: str,
    tokenizer,
    target_ratio: float = 4.0,
) -> Dict[str, NoModelResult]:
    """
    Run all no-model compressors on a single text.
    
    Returns dict mapping method_name → NoModelResult.
    No GPU needed at all.
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    
    compressors = {
        'baseline_first': NoModelBaselineCompressor(tokenizer, mode='first'),
        'baseline_random': NoModelBaselineCompressor(tokenizer, mode='random'),
        'baseline_longest': NoModelBaselineCompressor(tokenizer, mode='word_length'),
        'tone_only': NoModelToneCompressor(tokenizer),
        'morph_only': NoModelMorphCompressor(tokenizer),
        'tone_morph_combined': NoModelCombinedCompressor(tokenizer),
    }
    
    results = {}
    for name, comp in compressors.items():
        result = comp.compress(input_ids, target_ratio)
        results[name] = result
        print(f"  {name:25s}: {result.original_length:>5d} → "
              f"{result.compressed_length:>5d} tokens "
              f"({result.compression_ratio:.1f}x, {result.token_savings_pct:.0f}% saved) "
              f"in {result.processing_time_ms:.1f}ms")
    
    return results
