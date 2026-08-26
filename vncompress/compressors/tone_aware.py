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
  S_combined(t) = S_base(t) × [w_t × w_tone(t)·f_contrast(t,N) + (1 - w_t) × f_class(c(t))]

  i.e. a weighted linear blend of the tone-aware and morphology-aware
  multipliers (weight w_t = tone_weight), not a product of all four
  factors — see CombinedCompressor.compress() for the exact computation.

This provides comprehensive language-aware compression for Vietnamese.
"""

import torch
import time
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedTokenizer, PreTrainedModel

from .base import BaseCompressor, CompressionResult, CompressionConfig
from ..tone_aware.vietnamese_tones import (
    get_tone_analyzer,
    is_vietnamese,
    compute_tone_preservation_rate,
)
from ..tone_aware.tone_scoring import (
    PhonologicalConsistencyLoss,
)
from ..morphology.merge_policy import (
    MorphologyConfig,
    get_morphology_analyzer,
    WordClass,
    WordInfo,
)
from .quality_gate import SemanticQualityGate
from .query_aware import compute_query_relevance_weights


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
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.4,
        tone_contrast: Optional[Dict[Tuple[str, str], float]] = None,
        tone_window: int = 2,
        base_method: str = 'llmlingua',
        auto_detect_language: bool = True,
        fallback_to_base: bool = True,
        use_model_tone: bool = False,
        tone_probe_path: Optional[str] = None,
        quality_gate: Optional[SemanticQualityGate] = None,
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
        self.use_model_tone = use_model_tone
        self.quality_gate = quality_gate

        # tone_contrast: optional data-driven matrix from
        # estimate_tone_contrast_matrix(), overriding the hand-picked default.
        self.tone_analyzer = get_tone_analyzer(
            alpha=alpha, beta=beta, gamma=gamma, tone_contrast=tone_contrast
        )

        self._tone_probe: Optional[PhonologicalConsistencyLoss] = None
        if model is not None and use_model_tone:
            self._init_tone_probe(model)
        if tone_probe_path and model is not None:
            self._load_tone_probe(tone_probe_path)

        self._tone_active = True

    def _init_tone_probe(self, model: PreTrainedModel):
        hidden_dim = model.config.hidden_size
        self._tone_probe = PhonologicalConsistencyLoss(
            hidden_dim=hidden_dim, lambda_tone=0.0
        )
        self._tone_probe.eval()

    def _load_tone_probe(self, path: str):
        state = torch.load(path, map_location='cpu')
        hidden_dim = state['tone_classifier.0.weight'].shape[1]
        self._tone_probe = PhonologicalConsistencyLoss(
            hidden_dim=hidden_dim, lambda_tone=0.0
        )
        self._tone_probe.load_state_dict(state)
        self._tone_probe.eval()
        self._tone_probe.to(self.device)
        self.use_model_tone = True

    def get_name(self) -> str:
        suffix = "-model" if self.use_model_tone else ""
        return f"ToneAware-{self.base_method}{suffix}"

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

    def _compute_model_tone_weights(
        self,
        input_ids: List[int],
    ) -> torch.Tensor:
        """
        Compute tone importance weights from model hidden states.

        Uses the trained PhonologicalConsistencyLoss classifier as a probe:
        higher tone prediction confidence → better tone encoding →
        higher preservation weight.

        This creates the direct training→inference bridge that was missing:
        the model learns tone during training, and the same classifier
        guides compression decisions at inference time.
        """
        input_t = torch.tensor([input_ids]).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_t, output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]  # [1, S, D]

        assert self._tone_probe is not None
        self._tone_probe.to(self.device)
        tone_weights = self._tone_probe.score_importance(
            hidden_states
        )  # [1, S]

        return tone_weights.squeeze(0).cpu()
    
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
            # `self.model is not None` matters: LLMLingua's sentence filter
            # raises RuntimeError without a model, while this compressor runs
            # fine without one (_compute_base_scores returns ones). Falling
            # back unconditionally turned a working no-model run into a crash
            # on any non-Vietnamese input of >= 4 sentences.
            if lang != 'vi' and self.fallback_to_base and self.model is not None:
                # Non-Vietnamese: use base method directly
                from .llmlingua import LLMLinguaCompressor
                base = LLMLinguaCompressor(
                    self.tokenizer, self.model,
                    config=self.config, device=self.device,
                )
                return base.compress(input_ids)
        

        base_scores = self._compute_base_scores(input_ids)

        # Tone analysis is always computed from the decoded characters (cheap,
        # no model call) so it is available for the tone-preservation metric
        # below regardless of which scoring path (model probe vs. heuristic)
        # supplies the actual selection weights.
        tokens = self._decode_tokens(input_ids)
        tone_infos = self.tone_analyzer.analyze_tokens(
            tokens, window_size=self.tone_window
        )

        if self.use_model_tone and self.model is not None and self._tone_probe is not None:
            tone_weights = self._compute_model_tone_weights(input_ids)
        else:
            tone_weights = torch.tensor(
                [max(0.5, min(3.0, info.preservation_weight)) for info in tone_infos]
            )

        combined_scores = base_scores.clone()
        if len(tone_weights) == len(combined_scores):
            combined_scores *= tone_weights

        # Step 4: Select tokens (shared selector -- see
        # BaseCompressor.select_with_boundary for the two bugs the old inline
        # version had: returning the whole sequence when the budget left no
        # middle room, and offsetting indices by k after falling back to the
        # full score vector for n <= 2k, which raised IndexError.)
        retained_indices = self.select_with_boundary(combined_scores.tolist(), n)

        gate_info: Dict = {}
        if self.quality_gate is not None:
            compressed, gate_info = self.quality_gate.apply(
                input_ids, retained_indices, combined_scores.tolist()
            )
            retained_set = set(gate_info['retained_indices'])
        else:
            compressed = [input_ids[i] for i in retained_indices]
            retained_set = set(retained_indices)

        elapsed = (time.time() - start) * 1000

        # Tone Preservation Rate -- see compute_tone_preservation_rate()
        # (vncompress/tone_aware/vietnamese_tones.py) for the formal
        # definition and docs/tone_preservation_rate.md for edge cases.
        tone_preservation_rate = compute_tone_preservation_rate(tone_infos, retained_set)

        metadata = {
            'language': self._detect_language(input_ids) if self.auto_detect_language else 'unknown',
            'tone_active': self._tone_active,
            'tone_preservation_rate': tone_preservation_rate,
            'avg_tone_weight': tone_weights.mean().item() if len(tone_weights) > 0 else 1.0,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'use_model_tone': self.use_model_tone,
        }
        if gate_info:
            metadata['quality_gate'] = gate_info

        return self._build_result(
            compressed_ids=compressed,
            original_length=n,
            processing_time_ms=elapsed,
            metadata=metadata,
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
            raw_pairs = self.morph_analyzer.find_reduplicative_pairs(
                redup_tokens, config=self.morph_config
            )
            redup_pairs = [(l, r) for l, r, _ in raw_pairs]
        
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
        
        # Build compressed sequence (keep original order). Use an index
        # set rather than concatenating raw slices: for very short
        # sequences (n <= 2*k) input_ids[:k] and input_ids[n - k:] overlap
        # or cover the whole sequence, so naive concatenation would
        # duplicate tokens.
        selected_mid = set(selected_mid) - redup_merged
        retained_indices = (
            set(range(min(k, n))) | selected_mid | set(range(max(k, n - k), n))
        )
        compressed = [input_ids[i] for i in sorted(retained_indices)]
        
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

      S(t) = S_base(t) × [w_t × w_tone(t)·f_contrast(t,N) + (1 - w_t) × f_class(c(t))]

    i.e. a weighted linear blend of the two multipliers (see `tone_weight`),
    not their product — this matches the implementation below.

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
        tone_contrast: Optional[Dict[Tuple[str, str], float]] = None,
        tone_window: int = 2,
        # Morphology params
        f_func: float = 0.4,
        f_content: float = 1.2,
        f_redup: float = 0.6,
        f_compound: float = 1.5,
        # Options
        auto_detect: bool = True,
        tone_weight: float = 0.5,
        use_attention: bool = False,
        attention_weight: float = 0.3,
        quality_gate: Optional[SemanticQualityGate] = None,
    ):
        super().__init__(tokenizer, model, config)
        self.device = device
        self.tone_weight = tone_weight
        self.attention_weight = attention_weight
        self.use_attention = use_attention
        self.auto_detect = auto_detect
        self.quality_gate = quality_gate

        self.tone_comp = ToneAwareCompressor(
            tokenizer, model, config, device,
            alpha=alpha, beta=beta, gamma=gamma,
            tone_contrast=tone_contrast,
            tone_window=tone_window,
            base_method='llmlingua',
        )
        self.morph_comp = MorphologyAwareCompressor(
            tokenizer, model, config, device,
            f_func=f_func, f_content=f_content,
            f_redup=f_redup, f_compound=f_compound,
        )
    
    def get_name(self) -> str:
        return "Combined-ToneMorph"

    def _compute_attention_importance(
        self,
        input_ids: List[int],
        observation_window: int = 64,
    ) -> torch.Tensor:
        """
        Compute token importance from attention patterns (SnapKV-style).

        Uses the last few tokens as an observation window to identify
        which prompt tokens consistently receive high attention across
        heads and layers. High-attention tokens indicate key information.

        Reference: SnapKV (NeurIPS 2024)
        """
        n = len(input_ids)
        input_t = torch.tensor([input_ids]).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_t,
                output_attentions=True,
            )

        attentions = outputs.attentions
        if not attentions:
            return torch.ones(n)

        obs_len = min(observation_window, n)
        stacked = torch.stack(attentions, dim=0)  # [L, B, H, S, S]

        obs_attention = stacked[:, :, :, -obs_len:, :]

        head_importance = obs_attention.mean(dim=3)
        importance = head_importance.mean(dim=(0, 1, 2)).squeeze(0).cpu()

        if importance.numel() > 0 and importance.max() > importance.min():
            importance = (importance - importance.min()) / (
                importance.max() - importance.min() + 1e-8
            )
        importance = 1.0 + importance
        importance = torch.clamp(importance, 0.5, 3.0)

        return importance

    def compress(
        self,
        input_ids: List[int],
        query: Optional[str] = None,
        query_boost: float = 1.5,
        **kwargs,
    ) -> CompressionResult:
        """
        Combined compression using both tone and morphology signals.

        Steps:
          1. Run tone-aware compression to get tone-weighted scores
          2. Run morphology-aware compression to get morphology-weighted scores
          3. Blend both score sets: S = w_t × S_tone + (1-w_t) × S_morph
          4. Boost tokens overlapping `query`, if provided (LACC C2)
          5. Select top-k tokens

        Args:
            query: optional downstream question / tool schema. Tokens whose
                decoded text overlaps with it are boosted by `query_boost`
                before selection, so task-relevant tokens survive compression
                instead of competing purely on tone/morphology/perplexity.
        """
        start = time.time()
        n = len(input_ids)
        
        if not self.validate_input(input_ids):
            elapsed = (time.time() - start) * 1000
            return self._build_result(list(input_ids), n, elapsed)
        
        
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

        if self.use_attention and self.model is not None:
            attn_importance = self._compute_attention_importance(input_ids)
            if len(attn_importance) == len(combined_weights):
                aw = self.attention_weight
                combined_weights = (
                    (1 - aw) * combined_weights + aw * attn_importance
                )

        combined_scores = base_scores * combined_weights

        if query:
            query_weights = torch.tensor(
                compute_query_relevance_weights(tokens, query, boost=query_boost)
            )
            if len(query_weights) == len(combined_scores):
                combined_scores = combined_scores * query_weights

        # Select tokens -- shared selector, see BaseCompressor.
        retained_indices = self.select_with_boundary(combined_scores.tolist(), n)

        gate_info: Dict = {}
        if self.quality_gate is not None:
            compressed, gate_info = self.quality_gate.apply(
                input_ids, retained_indices, combined_scores.tolist()
            )
            retained_set = set(gate_info['retained_indices'])
        else:
            compressed = [input_ids[i] for i in retained_indices]
            retained_set = set(retained_indices)

        elapsed = (time.time() - start) * 1000

        # Same exact, index-based Tone Preservation Rate as ToneAwareCompressor
        # -- see compute_tone_preservation_rate() for the formal definition.
        tone_preservation_rate = compute_tone_preservation_rate(tone_infos, retained_set)

        metadata = {
            'tone_weight': wt,
            'morph_weight': 1 - wt,
            'mean_tone_multiplier': tone_weights.mean().item(),
            'mean_morph_multiplier': morph_weights.mean().item(),
            'tone_preservation_rate': tone_preservation_rate,
            'method': 'combined_tone_morph',
            'query_applied': bool(query),
        }
        if gate_info:
            metadata['quality_gate'] = gate_info

        return self._build_result(
            compressed_ids=compressed,
            original_length=n,
            processing_time_ms=elapsed,
            metadata=metadata,
        )
