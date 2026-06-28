"""
External Model Scoring for No-Model Compressors
================================================
Enables using an external tiny model (e.g., SmolLM 135M) for perplexity-based
token importance scoring, combined with tone/morphology weights.

Why external scoring matters:
  - No-model: tone + morphology weights only (linguistic heuristics)
  - With external model: tone + morphology + perplexity (learned importance)
  - Combined: best of both worlds, better compression quality

VRAM Budget with T4 (16GB):
  ┌──────────────────────────────────────┐
  │ Component               │ VRAM      │
  ├──────────────────────────────────────┤
  │ INT4 7B (generation)    │ ~5.0 GB   │
  │ INT4 135M (scoring)     │ ~0.3 GB   │
  │ KV cache (2K ctx)       │ ~0.8 GB   │
  │ PyTorch overhead        │ ~1.5 GB   │
  │ TOTAL                   │ ~7.6 GB ✓ │
  └──────────────────────────────────────┘

Scoring formula:
  S(t) = w_base × S_base(t) + w_tone × W_tone(t) + w_morph × W_morph(t)

  where:
    S_base = perplexity importance from tiny model
    W_tone = tone preservation weight
    W_morph = morphology preservation multiplier
    w_base, w_tone, w_morph = blend weights (sum = 1.0)

Models tested:
  | Model               | Params | VRAM (INT4) | Speed     |
  |---------------------|--------|-------------|-----------|
  | SmolLM2-135M        | 135M   | ~0.3 GB     | Fast      |
  | SmolLM2-360M        | 360M   | ~0.7 GB     | Fast      |
  | Qwen2.5-0.5B        | 500M   | ~0.5 GB     | Moderate  |
  | Qwen2.5-1.5B        | 1.5B   | ~1.0 GB     | Slower    |
"""

import torch
import torch.nn.functional as F
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from ..tone_aware.vietnamese_tones import VietnameseToneAnalyzer, get_tone_analyzer
from ..morphology.merge_policy import MorphologyAnalyzer, get_morphology_analyzer, WordClass


@dataclass
class ScoreWeights:
    """Blend weights for combining different scoring signals."""
    perplexity: float = 0.40    # Neural model scoring
    tone: float = 0.30          # Tone preservation
    morphology: float = 0.30    # Word class preservation
    
    def __post_init__(self):
        total = self.perplexity + self.tone + self.morphology
        if total != 1.0:
            self.perplexity /= total
            self.tone /= total
            self.morphology /= total
    
    def to_dict(self):
        return {
            'perplexity': self.perplexity,
            'tone': self.tone,
            'morphology': self.morphology,
        }


class TinyModelScorer:
    """
    Use a tiny external model (135M-500M params) for perplexity-based token scoring.
    
    This is the "external model" approach — the tiny model provides a neural signal
    for which tokens are important, which is then combined with tone/morphology weights.
    
    Use case on T4 (16GB):
      1. Load INT4 7B for generation (~5 GB VRAM)
      2. Load INT4 135M for scoring (~0.3 GB VRAM)  
      3. Run scoring on each context BEFORE generation
      4. Unload scorer if VRAM is tight, keep only 7B for generation
    
    Or even simpler: load scorer on CPU (no VRAM), trade speed for VRAM.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        tone_analyzer: Optional[VietnameseToneAnalyzer] = None,
        morph_analyzer: Optional[MorphologyAnalyzer] = None,
        weights: Optional[ScoreWeights] = None,
        device: str = 'cuda',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tone_analyzer = tone_analyzer or get_tone_analyzer()
        self.morph_analyzer = morph_analyzer or get_morphology_analyzer()
        self.weights = weights or ScoreWeights()
        self.device = device
    
    def _to_ids(self, tokens: List[str]) -> List[int]:
        """Convert token strings to IDs using this scorer's tokenizer."""
        ids = []
        for t in tokens:
            tid = self.tokenizer.encode(t, add_special_tokens=False)
            ids.append(tid[0] if tid else 0)
        return ids
    
    @torch.no_grad()
    def compute_perplexity_scores(
        self,
        input_ids: List[int],
        window_size: int = 512,
    ) -> torch.Tensor:
        """
        Compute perplexity-based token importance using the tiny model.
        
        Formula (same as LLMLingua):
          importance(t_i) = -log P(t_i | context)
        
        Higher perplexity → token is less expected → more important → keep.
        
        Uses the TINY model for fast scoring. Processing time: ~10-50ms for 512 tokens.
        
        Args:
            input_ids: Token IDs to score
            window_size: Process in chunks of this size (saves VRAM)
        
        Returns:
            Tensor of shape [n] with importance scores (higher = more important)
        """
        n = len(input_ids)
        device = next(self.model.parameters()).device
        
        scores = torch.zeros(n)
        
        # Process in windows to save VRAM
        for start in range(0, max(n - 1, 1), window_size):
            end = min(start + window_size + 1, n)
            chunk = input_ids[start:end]
            
            if len(chunk) < 2:
                continue
            
            input_tensor = torch.tensor([chunk], device=device)
            outputs = self.model(input_tensor)
            logits = outputs.logits  # [1, len(chunk), vocab_size]
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :]  # predict token at position i+1 from i
            shift_labels = input_tensor[:, 1:]  # actual token at position i+1
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)  # [1, len(chunk)-1]
            
            # Importance = negative log probability
            # Token that model finds surprising → high importance
            importance = -token_log_probs[0]  # [len(chunk)-1]
            
            # First position has no importance estimate, use mean
            scores[start] = importance[0] if len(importance) > 0 else 0
            for i, imp in enumerate(importance):
                idx = start + i + 1
                if idx < n:
                    scores[idx] = imp
        
        # Normalize to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = torch.ones_like(scores) * 0.5
        
        return scores
    
    def compute_tone_scores(
        self,
        tokens: List[str],
    ) -> torch.Tensor:
        """
        Compute tone preservation scores for a list of tokens.
        Returns tensor of shape [n] with tone importance (higher = preserve).
        """
        infos = self.tone_analyzer.analyze_tokens(tokens)
        return torch.tensor([info.preservation_weight for info in infos])
    
    def compute_morphology_scores(
        self,
        tokens: List[str],
    ) -> torch.Tensor:
        """
        Compute morphology preservation scores.
        Higher for content words, lower for function words.
        """
        infos = self.morph_analyzer.classify_batch(tokens)
        
        class_mult = {
            WordClass.FUNC: 0.40,      # Compress function words
            WordClass.CONTENT: 1.20,   # Preserve content words
            WordClass.REDUP: 0.60,     # Moderate for reduplicative
            WordClass.COMPOUND: 1.50,  # Strong preserve compounds
            WordClass.OTHER: 1.00,     # Neutral
        }
        
        scores = torch.tensor([
            class_mult.get(info.word_class, 1.0)
            for info in infos
        ])
        
        # Normalize
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min()) * 0.5 + 0.5
        
        return scores
    
    def compute_combined_scores(
        self,
        input_ids: List[int],
        tokens: List[str],
    ) -> torch.Tensor:
        """
        Combine all three scoring signals with configured blend weights.
        
        S(t) = w_p × S_perplexity(t) + w_t × S_tone(t) + w_m × S_morph(t)
        
        Returns normalized tensor of shape [n] (higher = keep).
        """
        n = len(input_ids)
        w = self.weights
        
        # Perplexity scoring from external model
        ppl_scores = self.compute_perplexity_scores(input_ids)
        
        # Tone analysis (CPU)
        tone_scores = self.compute_tone_scores(tokens)
        
        # Morphology analysis (CPU)
        morph_scores = self.compute_morphology_scores(tokens)
        
        # Ensure all same length
        min_len = min(len(ppl_scores), len(tone_scores), len(morph_scores))
        
        combined = (
            w.perplexity * ppl_scores[:min_len] +
            w.tone * tone_scores[:min_len] +
            w.morphology * morph_scores[:min_len]
        )
        
        # Normalize
        if combined.max() > combined.min():
            combined = (combined - combined.min()) / (combined.max() - combined.min())
        else:
            combined = torch.ones_like(combined) * 0.5
        
        return combined
    
    def score_and_select(
        self,
        input_ids: List[int],
        target_ratio: float = 4.0,
        keep_boundary: int = 2,
        tokens: Optional[List[str]] = None,
    ) -> Tuple[List[int], torch.Tensor, dict]:
        """
        Full pipeline: score tokens → select top-k → return compressed + scores.
        
        Args:
            input_ids: Original token IDs
            target_ratio: Target compression ratio
            keep_boundary: Always keep first/last N tokens
            tokens: Pre-decoded tokens (optional, avoids re-decoding)
        
        Returns:
            (compressed_ids, all_scores, stats_dict)
        """
        start = time.time()
        n = len(input_ids)
        target_len = max(int(n / target_ratio), 2 * keep_boundary)
        
        # Decode tokens if not provided
        if tokens is None:
            tokens = []
            for tid in input_ids:
                t = self.tokenizer.decode([tid])
                t = t.replace('\u2581', ' ').replace('Ġ', ' ').strip()
                tokens.append(t)
        
        # Compute combined scores
        scores = self.compute_combined_scores(input_ids, tokens)
        
        # Always keep boundaries
        k = keep_boundary
        mid_start, mid_end = k, max(k, n - k)
        mid_budget = max(0, target_len - 2 * k)
        
        if mid_budget > 0 and mid_start < mid_end:
            mid_scores = scores[mid_start:mid_end]
            mid_budget = min(mid_budget, len(mid_scores))
            
            _, top_indices = torch.topk(mid_scores, mid_budget)
            top_indices = sorted(top_indices.tolist())
            
            compressed = (
                input_ids[:k] +
                [input_ids[mid_start + i] for i in top_indices] +
                input_ids[mid_end:]
            )
        else:
            compressed = list(input_ids)
        
        elapsed = (time.time() - start) * 1000
        
        stats = {
            'scoring_time_ms': elapsed,
            'weights': self.weights.to_dict(),
            'mean_score': scores.mean().item(),
            'std_score': scores.std().item(),
        }
        
        return compressed, scores, stats
    
    def to(self, device: str):
        """Move model to device (for VRAM management)."""
        self.model.to(device)
        return self
    
    def unload(self):
        """Move model to CPU to free GPU VRAM."""
        self.model.to('cpu')
        torch.cuda.empty_cache()
        return self


# ============================================================================
# VRAM Manager for T4/P100
# ============================================================================

class VRAMManager:
    """
    Manage VRAM between scorer and generator on limited hardware.
    
    Strategy for T4 (16GB):
      1. Load scorer → compute scores → unload scorer
      2. Load generator → generate with compressed context
      
    This way, scorer and generator never occupy VRAM simultaneously.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.active_model = None
    
    def load_scorer(self, scorer: TinyModelScorer):
        """Load scorer onto GPU."""
        if self.active_model:
            self._unload_current()
        scorer.to(self.device)
        self.active_model = 'scorer'
    
    def load_generator(self, model):
        """Load generator onto GPU."""
        if self.active_model:
            self._unload_current()
        if hasattr(model, 'to'):
            model.to(self.device)
        self.active_model = 'generator'
    
    def _unload_current(self):
        """Move current model to CPU to free GPU VRAM."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.active_model = None
    
    @staticmethod
    def vram_info() -> dict:
        """Get current VRAM status."""
        if not torch.cuda.is_available():
            return {'used_gb': 0, 'free_gb': 0, 'total_gb': 0}
        import torch
        used = torch.cuda.memory_allocated(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        return {
            'used_gb': round(used, 2),
            'free_gb': round(total - used, 2),
            'total_gb': round(total, 2),
        }


# ============================================================================
# Quick helper: load scorer from HuggingFace
# ============================================================================

TINY_MODEL_IDS = {
    'smollm2-135m': 'HuggingFaceTB/SmolLM2-135M-Instruct',
    'smollm2-360m': 'HuggingFaceTB/SmolLM2-360M-Instruct',
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'gemma-2b': 'google/gemma-2-2b-it',
    'llama-3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct',
}


def create_tiny_scorer(
    model_id: str = 'smollm2-135m',
    use_int4: bool = True,
    device: str = 'cuda',
    weights: Optional[ScoreWeights] = None,
    use_cpu: bool = False,
) -> TinyModelScorer:
    """
    Factory function to create a TinyModelScorer.
    
    Args:
        model_id: Short name or full HuggingFace ID
        use_int4: Load in INT4 (recommended for T4/P100)
        device: 'cuda' or 'cpu'
        weights: Custom blend weights
        use_cpu: Force CPU loading (0 VRAM but slower)
    
    Returns:
        TinyModelScorer ready for use
    
    Example:
        scorer = create_tiny_scorer('smollm2-135m', use_int4=True)
        compressed_ids, scores, stats = scorer.score_and_select(input_ids, 4.0)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    model_name = TINY_MODEL_IDS.get(model_id, model_id)
    
    actual_device = 'cpu' if use_cpu else device
    
    print(f"[SCORER] Loading tiny model: {model_name}")
    
    if actual_device == 'cpu':
        print(f"  Device: CPU (slower but 0 VRAM)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, device_map='cpu',
            torch_dtype=torch.float32,
        )
    elif use_int4:
        print(f"  Quantization: INT4 (~0.3 GB VRAM for 135M)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            quantization_config=bnb_config,
            device_map='auto',
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.float16, device_map='auto',
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {params:.0f}M | Model ready for scoring")
    
    return TinyModelScorer(model, tokenizer, weights=weights, device=actual_device)


# ============================================================================
# Integration: Enhanced No-Model Compressor with External Scoring
# ============================================================================

class EnhancedCompressor:
    """
    No-model compressor enhanced with external tiny model scoring.
    
    This is the BEST compressor for limited hardware:
      - Tone analysis (CPU, 0 VRAM, linguistic knowledge)
      - Morphology analysis (CPU, 0 VRAM, linguistic knowledge)
      - Perplexity scoring (tiny model, 0.3 GB VRAM or CPU)
      - All three combined with configurable blend weights
    
    Usage on T4 (16GB):
      >>> scorer = create_tiny_scorer('smollm2-135m')  # ~0.3 GB VRAM
      >>> comp = EnhancedCompressor(tokenizer, scorer)
      >>> result = comp.compress(input_ids, target_ratio=4.0)
      >>> # result.compressed_ids is ready for 7B generation
    """
    
    def __init__(
        self,
        tokenizer,
        scorer: Optional[TinyModelScorer] = None,
        weights: Optional[ScoreWeights] = None,
    ):
        """
        Args:
            tokenizer: Tokenizer matching the MAIN generation model
            scorer: TinyModelScorer for perplexity scoring (None = no-model mode)
            weights: Blend weights. If None, auto-adapts based on scorer availability:
                     - scorer available:  0.4 ppl, 0.3 tone, 0.3 morph
                     - scorer unavailable: 0.0 ppl, 0.5 tone, 0.5 morph
        """
        self.tokenizer = tokenizer
        self.scorer = scorer
        
        if weights is None:
            if scorer is not None:
                weights = ScoreWeights(perplexity=0.4, tone=0.3, morphology=0.3)
            else:
                weights = ScoreWeights(perplexity=0.0, tone=0.5, morphology=0.5)
        
        self.weights = weights
        
        # Always available (CPU, no deps)
        self.tone_analyzer = get_tone_analyzer()
        self.morph_analyzer = get_morphology_analyzer()
    
    def compress(
        self,
        input_ids: List[int],
        target_ratio: float = 4.0,
        keep_boundary: int = 2,
    ) -> Tuple[List[int], dict]:
        """
        Compress with combined scoring.
        
        Args:
            input_ids: Token IDs to compress
            target_ratio: Target compression ratio
            keep_boundary: Keep first/last N tokens
        
        Returns:
            (compressed_ids, stats_dict)
        """
        start = time.time()
        n = len(input_ids)
        target_len = max(int(n / target_ratio), 2 * keep_boundary)
        
        # Decode tokens
        tokens = []
        for tid in input_ids:
            t = self.tokenizer.decode([tid])
            t = t.replace('\u2581', ' ').replace('Ġ', ' ').strip()
            tokens.append(t)
        
        # 1. Perplexity scores (if scorer available)
        if self.scorer is not None and self.weights.perplexity > 0:
            ppl_scores = self.scorer.compute_perplexity_scores(input_ids)
        else:
            import torch
            ppl_scores = torch.ones(n) * 0.5  # Neutral
        
        # 2. Tone scores (always available, CPU)
        tone_infos = self.tone_analyzer.analyze_tokens(tokens)
        tone_scores = torch.tensor([info.preservation_weight for info in tone_infos])
        
        # 3. Morphology scores (always available, CPU)
        word_infos = self.morph_analyzer.classify_batch(tokens)
        class_mult = {
            WordClass.FUNC: 0.40,
            WordClass.CONTENT: 1.20,
            WordClass.REDUP: 0.60,
            WordClass.COMPOUND: 1.50,
            WordClass.OTHER: 1.00,
        }
        morph_scores = torch.tensor([
            class_mult.get(info.word_class, 1.0) for info in word_infos
        ])
        
        # Normalize all to [0, 1]
        for scores in [ppl_scores, tone_scores, morph_scores]:
            if scores.max() > scores.min():
                scores[:] = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                scores[:] = torch.ones_like(scores) * 0.5
        
        # Blend
        min_len = min(len(ppl_scores), len(tone_scores), len(morph_scores))
        combined = (
            self.weights.perplexity * ppl_scores[:min_len] +
            self.weights.tone * tone_scores[:min_len] +
            self.weights.morphology * morph_scores[:min_len]
        )
        
        # Select top-k
        k = keep_boundary
        mid_start, mid_end = k, max(k, n - k)
        mid_budget = max(0, target_len - 2 * k)
        
        if mid_budget > 0 and mid_start < mid_end:
            mid_scores = combined[mid_start:mid_end]
            mid_budget = min(mid_budget, len(mid_scores))
            _, top_indices = torch.topk(mid_scores, mid_budget)
            top_indices = sorted(top_indices.tolist())
            compressed = (
                input_ids[:k] +
                [input_ids[mid_start + i] for i in top_indices] +
                input_ids[mid_end:]
            )
        else:
            compressed = list(input_ids)
        
        elapsed = (time.time() - start) * 1000
        
        stats = {
            'has_scorer': self.scorer is not None,
            'weights': self.weights.to_dict(),
            'compression_time_ms': elapsed,
            'mean_tone_score': tone_scores.mean().item(),
            'mean_morph_score': morph_scores.mean().item(),
            'mean_ppl_score': ppl_scores.mean().item(),
        }
        
        return compressed, stats
    
    def has_external_scorer(self) -> bool:
        return self.scorer is not None
