"""
Tone-Aware & Morphology-Aware Compressors
=========================================
Novel compression methods designed for Vietnamese and other low-resource
tonal/isolating languages.

These are the main contributions of this research project.

Papers this extends:
  - LLMLingua (EMNLP 2023, arxiv:2310.05736)
  - SnapKV (2024, arxiv:2404.14469)
  - Cross-Lingual Token Arbitrage (2026, arxiv:2606.03618)
  - Equity with Efficiency (2026, arxiv:2606.15044)

---

TONE-AWARE COMPRESSION
=======================

Key Formula:
  S_tone(t) = S_base(t) × w_tone(t) × f_contrast(t, neighbors)

where:
  S_base(t)       = base compression score (perplexity or attention-based)
  w_tone(t)       = tone preservation weight (see vietnamese_tones.py)
  f_contrast(t,N) = tonal contrast factor with neighbor tokens

Why Tone-Aware matters for Vietnamese:
  - 6 contrastive tones: ma ≠ má ≠ mà ≠ mả ≠ mã ≠ mạ
  - Token-level compression may delete diacritic-carrying characters
  - Tone changes can completely alter word meaning
  - Current compression methods are tone-blind

---

MORPHOLOGY-AWARE COMPRESSION
=============================

Key Formula:
  S_morph(t) = S_base(t) × f_class(c(t))

where:
  f_class(FUNC)     = 0.3-0.5   (function words → compress aggressively)
  f_class(CONTENT)  = 1.0-1.5   (content words → preserve)
  f_class(REDUP)    = 0.4-0.6   (reduplicative → moderate merge)
  f_class(COMPOUND) = 1.2-2.0   (compounds → preserve strongly)
  c(t)             = word class of token t

Why Morphology-Aware matters for Vietnamese:
  - ~30-40% of tokens are function words (đã, sẽ, của, những...)
  - Function words carry low semantic content → safe to compress
  - Content words carry meaning → must preserve
  - Reduplicative words have redundancy → can merge
  - Compound words should not be split during compression

---

COMBINED SCORE:
  S_combined(t) = S_base(t) × w_tone(t) × f_contrast(t,N) × f_class(c(t))

This provides comprehensive language-aware compression for Vietnamese.
"""

import torch
import time
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedTokenizer, PreTrainedModel

from .base import BaseCompressor, CompressionResult, CompressionConfig
from ..tone_aware.vietnamese_tones import (
    VietnameseToneAnalyzer,
    get_tone_analyzer,
    is_vietnamese,
)
from ..morphology.merge_policy import (
    MorphologyAnalyzer,
    MorphologyConfig,
    get_morphology_analyzer,
    WordClass,
    WordInfo,
)


class ToneAwareCompressor(BaseCompressor):
    """
    Tone-Aware Context Compression (Novel Contribution #1).
    
    Extends any base compressor with tone-aware token scoring.
    
    Designed for Vietnamese and extensible to other tonal languages
    (Chinese, Thai, Yoruba, etc.).
    
    Works as a "wrapper" around existing compressors:
      base_compressor → compute base scores → apply tone weights → select tokens
    
    Or as a standalone compressor using LLMLingua-style perplexity scoring
    as the base score.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Optional[PreTrainedModel] = None,
        config: Optional[CompressionConfig] = None,
        device: str = 'cuda',
        # Tone-aware parameters
        alpha: float = 0.5,           # Base tone importance
        beta: float = 0.3,           # Tone variety bonus
        gamma: float = 0.4,          # Contrast amplification
        tone_window: int = 2,        # Context window for contrast
        # Base scoring
        base_method: str = 'llmlingua',  # 'llmlingua', 'snapkv', 'selective'
        # Options
        auto_detect_language: bool = True,
        fallback_to_base: bool = True,  # If not Vietnamese, use base method
    ):
        super().__init__(tokenizer, model, config)
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tone_window = tone_window
        self.base_method = base_method
        self.auto_detect_language = auto_detect_language
        self.fallback_to_base = fallback_to_base
        
        # Initialize tone analyzer with our parameters
        self.tone_analyzer = get_tone_analyzer(
            alpha=alpha, beta=beta, gamma=gamma
        )
        
        # Determine if tone-aware mode should be active
        self._tone_active = True
    
    def get_name(self) -> str:
        return f"ToneAware-{self.base_method}"
    
    def _compute_base_scores(
        self,
        input_ids: List[int],
    ) -> torch.Tensor:
        """
        Compute base compression scores using the selected method.
        
        Methods:
          - llmlingua: Perplexity-based importance (LLMLingua-style)
          - uniform: All tokens equal weight (tone-only mode)
          - random: Random baseline
        """
        n = len(input_ids)
        
        if self.base_method == 'uniform' or self.model is None:
            # Uniform: tone-only scoring
            return torch.ones(n)
        
        elif self.base_method == 'llmlingua':
            # Use LLMLingua's perplexity-based scoring
            from .llmlingua import LLMLinguaCompressor
            llmlingua = LLMLinguaCompressor(
                self.tokenizer, self.model,
                small_model=self.model,
                config=self.config,
                device=self.device,
            )
            try:
                scores = llmlingua._compute_token_importance(input_ids)
                return scores
            except Exception:
                # Fallback to uniform
                return torch.ones(n)
        
        elif self.base_method == 'snapkv':
            # Use SnapKV-style attention-based scoring
            from .snapkv import SnapKVCompressor
            snapkv = SnapKVCompressor(
                self.tokenizer, self.model,
                config=self.config,
                device=self.device,
            )
            try:
                input_t = torch.tensor([input_ids]).to(self.device)
                importance = snapkv._compute_attention_importance(input_t)
                # Aggregate across heads
                return importance.mean(dim=0).cpu()  # [S]
            except Exception:
                return torch.ones(n)
        
        else:
            return torch.ones(n)
    
    def _detect_language(self, input_ids: List[int]) -> str:
        """
        Detect if input is Vietnamese.
        
        Returns 'vi' if Vietnamese, 'other' otherwise.
        """
        text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        if is_vietnamese(text[:500]):  # Check first 500 chars
            return 'vi'
        return 'other'
    
    def _decode_tokens(self, token_ids: List[int]) -> List[str]:
        """Decode token IDs to readable strings for tone analysis."""
        tokens = []
        for tid in token_ids:
            token_str = self.tokenizer.decode([tid])
            token_str = token_str.replace('\u2581', ' ').replace('Ġ', ' ').strip()
            tokens.append(token_str)
        return tokens
    
    def compress(
        self,
        input_ids: List[int],
        **kwargs,
    ) -> CompressionResult:
        """
        Tone-aware compression.
        
        Steps:
          1. Detect language (auto or manual)
          2. If not Vietnamese and fallback enabled → use base method
          3. Compute base scores (perplexity or attention)
          4. Compute tone weights for each token
          5. Compute contrast factors with neighbors
          6. Apply combined scoring: S = S_base × w_tone × f_contrast
          7. Select top-k tokens
        """
        start = time.time()
        n = len(input_ids)
        
        if not self.validate_input(input_ids):
            elapsed = (time.time() - start) * 1000
            return self._build_result(list(input_ids), n, elapsed)
        
        # Auto-detect language
        if self.auto_detect_language:
            lang = self._detect_language(input_ids)
            if lang != 'vi' and self.fallback_to_base:
                # Non-Vietnamese: use base method directly
                from .llmlingua import LLMLinguaCompressor
                base = LLMLinguaCompressor(
                    self.tokenizer, self.model,
                    config=self.config, device=self.device,
                )
                return base.compress(input_ids)
        
        target_len = max(int(n / self.config.target_ratio), self.config.min_compressed_length)
        
        # Step 1: Compute base scores
        base_scores = self._compute_base_scores(input_ids)
        
        # Step 2: Decode tokens for tone analysis
        tokens = self._decode_tokens(input_ids)
        
        # Step 3: Compute tone-aware scores
        tone_infos = self.tone_analyzer.analyze_tokens(
            tokens, window_size=self.tone_window
        )
        
        # Build combined score
        combined_scores = base_scores.clone()
        for i, info in enumerate(tone_infos):
            # Tone weight × contrast factor
            tone_multiplier = info.preservation_weight
            # Clamp to reasonable range
            tone_multiplier = max(0.5, min(3.0, tone_multiplier))
            combined_scores[i] *= tone_multiplier
        
        # Step 4: Select tokens
        k = self.config.keep_boundary_tokens
        mid_scores = combined_scores[k:n - k] if n > 2 * k else combined_scores
        mid_budget = max(0, target_len - 2 * k)
        
        if mid_budget > 0 and mid_budget < len(mid_scores):
            _, top_indices = torch.topk(mid_scores, mid_budget)
            top_indices = sorted(top_indices.tolist())
            compressed = (
                input_ids[:k] +
                [input_ids[k + i] for i in top_indices] +
                input_ids[n - k:]
            )
        else:
            compressed = list(input_ids)
        
        elapsed = (time.time() - start) * 1000
        
        # Compute tone preservation metrics
        original_tones = sum(1 for info in tone_infos if info.tones_present)
        # Count preserved tones (approximate: check compressed tokens)
        preserved_tones = 0
        compressed_tokens = self._decode_tokens(compressed)
        for ct in compressed_tokens:
            preserved_tones += len(self.tone_analyzer.get_tone_sequence(ct))
            for t in self.tone_analyzer.get_tone_sequence(ct):
                if t > 0:  # non-ngang
                    preserved_tones += 1
        
        tone_preservation_rate = preserved_tones / max(original_tones, 1)
        
        return self._build_result(
            compressed_ids=compressed,
            original_length=n,
            processing_time_ms=elapsed,
            metadata={
                'language': self._detect_language(input_ids) if self.auto_detect_language else 'unknown',
                'tone_active': self._tone_active,
                'tone_preservation_rate': tone_preservation_rate,
                'avg_tone_weight': sum(info.preservation_weight for info in tone_infos) / max(len(tone_infos), 1),
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
            },
        )


class MorphologyAwareCompressor(BaseCompressor):
    """
    Morphology-Aware Context Compression (Novel Contribution #2).
    
    Leverages Vietnamese word morphology for smarter compression:
      - Function words (hư từ): compress aggressively
      - Content words (thực từ): preserve carefully
      - Reduplicative words (từ láy): merge with partner
      - Compound words (từ ghép): preserve as unit
    
    Works standalone using LLMLingua-style scoring with morphology multipliers.
    
    Mathematical basis:
      S(t) = S_base(t) × f_class(c(t))
      
    where c(t) is the word class and f_class provides class-specific
    preservation factors.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Optional[PreTrainedModel] = None,
        config: Optional[CompressionConfig] = None,
        device: str = 'cuda',
        # Morphology parameters
        f_func: float = 0.4,        # Function word preservation factor
        f_content: float = 1.2,     # Content word preservation factor
        f_redup: float = 0.6,       # Reduplicative preservation factor
        f_compound: float = 1.5,    # Compound word preservation factor
        f_other: float = 1.0,       # Unknown word preservation factor
        # Merge behavior
        merge_redup_pairs: bool = True,   # Merge từ láy into single token
        split_compounds: bool = False,    # Never split compound words
        # POS tagger
        use_pos_tagger: bool = False,
    ):
        super().__init__(tokenizer, model, config)
        self.device = device
        
        # Build morphology config
        self.morph_config = MorphologyConfig(
            f_func=f_func,
            f_content=f_content,
            f_redup=f_redup,
            f_compound=f_compound,
            f_other=f_other,
        )
        
        self.merge_redup_pairs = merge_redup_pairs
        self.split_compounds = split_compounds
        
        # Initialize morphology analyzer
        self.morph_analyzer = get_morphology_analyzer(
            use_pos_tagger=use_pos_tagger
        )
    
    def get_name(self) -> str:
        return "MorphologyAware"
    
    def _decode_tokens(self, token_ids: List[int]) -> List[str]:
        """Decode to clean token strings."""
        tokens = []
        for tid in token_ids:
            token_str = self.tokenizer.decode([tid])
            token_str = token_str.replace('\u2581', ' ').replace('Ġ', ' ').strip()
            tokens.append(token_str)
        return tokens
    
    def _compute_class_distribution(
        self,
        word_infos: List[WordInfo],
    ) -> Dict[str, int]:
        """Count tokens per word class."""
        dist = {c.value: 0 for c in WordClass}
        for info in word_infos:
            dist[info.word_class.value] += 1
        return dist
    
    def compress(
        self,
        input_ids: List[int],
        **kwargs,
    ) -> CompressionResult:
        """
        Morphology-aware compression.
        
        Steps:
          1. Decode tokens and classify each by word class
          2. Find reduplicative pairs (if merge enabled)
          3. Compute base scores
          4. Apply morphology multipliers
          5. Select tokens with class-aware budget
        """
        start = time.time()
        n = len(input_ids)
        
        if not self.validate_input(input_ids):
            elapsed = (time.time() - start) * 1000
            return self._build_result(list(input_ids), n, elapsed)
        
        target_len = max(int(n / self.config.target_ratio), self.config.min_compressed_length)
        
        # Step 1: Decode and classify
        tokens = self._decode_tokens(input_ids)
        word_infos = self.morph_analyzer.classify_batch(tokens)
        
        # Step 2: Find reduplicative pairs
        redup_pairs = []
        if self.merge_redup_pairs:
            redup_tokens = [w.token for w in word_infos]
            redup_pairs = self.morph_analyzer.find_reduplicative_pairs(redup_tokens)
        
        # Step 3: Compute base scores (LLMLingua-style)
        base_scores = torch.ones(n)
        if self.model is not None:
            try:
                from .llmlingua import LLMLinguaCompressor
                llm = LLMLinguaCompressor(
                    self.tokenizer, self.model,
                    small_model=self.model,
                    config=self.config,
                    device=self.device,
                )
                base_scores = llm._compute_token_importance(input_ids)
            except Exception:
                pass
        
        # Step 4: Apply morphology multipliers
        multipliers = torch.ones(n)
        for i, info in enumerate(word_infos):
            mult = self.morph_analyzer.get_preservation_multiplier(
                info, self.morph_config
            )
            multipliers[i] = mult
        
        # Handle reduplicative pairs: merge partner into first word
        redup_merged = set()
        for left, right in redup_pairs:
            # Mark right partner as "merged" (low importance → will be removed)
            if right < len(multipliers):
                multipliers[right] = 0.1  # Very low → drop
                redup_merged.add(right)
            # Boost left partner (it now represents both)
            if left < len(multipliers):
                multipliers[left] = min(multipliers[left] * 1.5, 3.0)
        
        # Step 5: Apply multipliers to scores
        combined_scores = base_scores * multipliers
        
        # Step 6: Class-aware budget allocation
        # Count tokens per class
        class_dist = self._compute_class_distribution(word_infos)
        
        # Always keep boundaries
        k = self.config.keep_boundary_tokens
        mid_indices = list(range(k, n - k)) if n > 2 * k else []
        
        # Separate mid-tokens by class
        class_indices: Dict[str, List[int]] = {c.value: [] for c in WordClass}
        for i in mid_indices:
            if i < len(word_infos):
                cls = word_infos[i].word_class.value
                class_indices[cls].append(i)
        
        # Budget allocation per class (proportional to original distribution
        # but adjusted by class importance)
        total_mid_budget = max(0, target_len - 2 * k)
        
        # Weight = count × keep_ratio for each class
        keep_ratios = {
            'function': self.morph_config.r_func,
            'content': self.morph_config.r_content,
            'reduplicative': self.morph_config.r_redup,
            'compound': self.morph_config.r_compound,
            'other': self.morph_config.r_other,
        }
        
        total_weight = 0.0
        for cls, indices in class_indices.items():
            total_weight += len(indices) * keep_ratios.get(cls, 0.5)
        
        # Allocate budget proportionally
        selected_mid = []
        for cls, indices in class_indices.items():
            if not indices:
                continue
            
            weight = len(indices) * keep_ratios.get(cls, 0.5)
            cls_budget = max(0, min(len(indices),
                                   int(total_mid_budget * weight / max(total_weight, 1))))
            
            if cls_budget > 0:
                # Get scores for this class
                cls_scores = combined_scores[indices]
                if cls_budget < len(cls_scores):
                    _, top_local = torch.topk(cls_scores, cls_budget)
                    selected = [indices[i] for i in top_local.tolist()]
                else:
                    selected = indices
                selected_mid.extend(selected)
        
        # Build compressed sequence (keep original order)
        selected_mid = sorted(set(selected_mid) - redup_merged)
        compressed = (
            input_ids[:k] +
            [input_ids[i] for i in selected_mid] +
            input_ids[n - k:]
        )
        
        elapsed = (time.time() - start) * 1000
        
        class_dist = self._compute_class_distribution(word_infos)
        compressed_infos = self.morph_analyzer.classify_batch(
            self._decode_tokens(compressed)
        )
        compressed_dist = self._compute_class_distribution(compressed_infos)
        
        return self._build_result(
            compressed_ids=compressed,
            original_length=n,
            processing_time_ms=elapsed,
            metadata={
                'original_class_distribution': class_dist,
                'compressed_class_distribution': compressed_dist,
                'redup_pairs_found': len(redup_pairs),
                'redup_pairs_merged': len(redup_merged),
                'f_func': self.morph_config.f_func,
                'f_content': self.morph_config.f_content,
            },
        )


class CombinedCompressor(BaseCompressor):
    """
    Combined Tone-Aware + Morphology-Aware Compressor (Novel Contribution #3).
    
    Applies both tone and morphology signals together:
    
      S(t) = S_base(t) × w_tone(t) × f_contrast(t,N) × f_class(c(t))
    
    This is the most comprehensive language-aware compressor for Vietnamese,
    combining tonal information preservation with morphological structure awareness.
    
    Expected to achieve the best quality-compression trade-off for Vietnamese texts.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Optional[PreTrainedModel] = None,
        config: Optional[CompressionConfig] = None,
        device: str = 'cuda',
        # Tone params
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.4,
        tone_window: int = 2,
        # Morphology params
        f_func: float = 0.4,
        f_content: float = 1.2,
        f_redup: float = 0.6,
        f_compound: float = 1.5,
        # Options
        auto_detect: bool = True,
        tone_weight: float = 0.5,      # Blend: tone vs morphology importance
    ):
        super().__init__(tokenizer, model, config)
        self.device = device
        
        # Create sub-compressors
        self.tone_comp = ToneAwareCompressor(
            tokenizer, model, config, device,
            alpha=alpha, beta=beta, gamma=gamma,
            tone_window=tone_window,
            base_method='llmlingua',
        )
        self.morph_comp = MorphologyAwareCompressor(
            tokenizer, model, config, device,
            f_func=f_func, f_content=f_content,
            f_redup=f_redup, f_compound=f_compound,
        )
        
        self.tone_weight = tone_weight
        self.auto_detect = auto_detect
    
    def get_name(self) -> str:
        return "Combined-ToneMorph"
    
    def compress(
        self,
        input_ids: List[int],
        **kwargs,
    ) -> CompressionResult:
        """
        Combined compression using both tone and morphology signals.
        
        Steps:
          1. Run tone-aware compression to get tone-weighted scores
          2. Run morphology-aware compression to get morphology-weighted scores  
          3. Blend both score sets: S = w_t × S_tone + (1-w_t) × S_morph
          4. Select top-k tokens
        """
        start = time.time()
        n = len(input_ids)
        
        if not self.validate_input(input_ids):
            elapsed = (time.time() - start) * 1000
            return self._build_result(list(input_ids), n, elapsed)
        
        target_len = max(int(n / self.config.target_ratio), self.config.min_compressed_length)
        
        # Get scores from both methods
        base_scores = self.tone_comp._compute_base_scores(input_ids)
        
        # Decode tokens once
        tokens = []
        for tid in input_ids:
            token_str = self.tokenizer.decode([tid])
            token_str = token_str.replace('\u2581', ' ').replace('Ġ', ' ').strip()
            tokens.append(token_str)
        
        # Tone analysis
        tone_infos = self.tone_comp.tone_analyzer.analyze_tokens(tokens)
        tone_weights = torch.tensor([info.preservation_weight for info in tone_infos])
        tone_weights = torch.clamp(tone_weights, 0.5, 3.0)
        
        # Morphology analysis
        word_infos = self.morph_comp.morph_analyzer.classify_batch(tokens)
        morph_weights = torch.tensor([
            self.morph_comp.morph_analyzer.get_preservation_multiplier(
                info, self.morph_comp.morph_config
            ) for info in word_infos
        ])
        
        # Blend scores
        wt = self.tone_weight
        combined_weights = wt * tone_weights + (1 - wt) * morph_weights
        combined_scores = base_scores * combined_weights
        
        # Select tokens
        k = self.config.keep_boundary_tokens
        mid_scores = combined_scores[k:n - k] if n > 2 * k else combined_scores
        mid_budget = max(0, target_len - 2 * k)
        
        if mid_budget > 0 and mid_budget < len(mid_scores):
            _, top_indices = torch.topk(mid_scores, mid_budget)
            top_indices = sorted(top_indices.tolist())
            compressed = (
                input_ids[:k] +
                [input_ids[k + i] for i in top_indices] +
                input_ids[n - k:]
            )
        else:
            compressed = list(input_ids)
        
        elapsed = (time.time() - start) * 1000
        
        return self._build_result(
            compressed_ids=compressed,
            original_length=n,
            processing_time_ms=elapsed,
            metadata={
                'tone_weight': wt,
                'morph_weight': 1 - wt,
                'mean_tone_multiplier': tone_weights.mean().item(),
                'mean_morph_multiplier': morph_weights.mean().item(),
                'method': 'combined_tone_morph',
            },
        )
