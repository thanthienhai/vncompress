"""
Base Compressor — Abstract interface for all compression methods.

All compressors in vncompress follow this interface, making it easy to:
  1. Add new compression methods
  2. Benchmark comparably
  3. Combine methods (e.g., ToneAware + MorphologyAware)

Compression Ratio:  CR = N_original / N_compressed
Token Savings:      TS = (N_original - N_compressed) / N_original × 100%
"""

import torch
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from transformers import PreTrainedTokenizer, PreTrainedModel


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    # Compressed representation
    compressed_ids: List[int]           # Compressed token IDs
    compressed_text: str                # Decoded compressed text
    compression_ratio: float            # N_original / N_compressed
    token_savings_pct: float            # Percentage of tokens saved
    
    # Metadata
    original_length: int
    compressed_length: int
    method_name: str
    processing_time_ms: float
    
    # Analysis (optional, filled by specific compressors)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # For KV cache compression
    kv_cache_mask: Optional[torch.Tensor] = None  # Shape: [layers, heads, seq_len]
    kv_memory_saved_bytes: int = 0


@dataclass 
class CompressionConfig:
    """Base configuration for all compressors."""
    target_ratio: float = 4.0           # Target compression ratio
    keep_special_tokens: bool = True    # Always keep BOS, EOS, SEP, etc.
    keep_boundary_tokens: int = 2       # Keep first/last N tokens
    min_compressed_length: int = 1      # Minimum length after compression
    max_compressed_length: int = 32768  # Maximum length after compression
    
    # Language-specific
    language: str = 'vi'                # 'vi', 'en', 'auto'
    detect_language: bool = True
    
    # Verbose
    verbose: bool = False


class BaseCompressor(ABC):
    """
    Abstract base class for all compression methods.
    
    To implement a new compressor, subclass and implement:
      - compress(): Core compression logic
      - get_name(): Return method name for logging
    """
    
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
    def compress(
        self,
        input_ids: List[int],
        **kwargs,
    ) -> CompressionResult:
        """
        Compress a sequence of token IDs.
        
        Args:
            input_ids: List of token IDs to compress
            **kwargs: Method-specific parameters
        
        Returns:
            CompressionResult with compressed token IDs and metadata
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this compression method."""
        pass
    
    def compress_text(
        self,
        text: str,
        **kwargs,
    ) -> CompressionResult:
        """
        Compress a text string (convenience wrapper).
        
        Encodes text → compresses → returns result with decoded text.
        """
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self.compress(input_ids, **kwargs)
    
    def _compute_compression_ratio(
        self,
        original_length: int,
        compressed_length: int,
    ) -> Tuple[float, float]:
        """Compute compression ratio and token savings percentage."""
        ratio = original_length / max(compressed_length, 1)
        savings = ((original_length - compressed_length) / max(original_length, 1)) * 100
        return ratio, savings
    
    def _build_result(
        self,
        compressed_ids: List[int],
        original_length: int,
        processing_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CompressionResult:
        """Build a CompressionResult from compressed IDs."""
        comp_len = len(compressed_ids)
        ratio, savings = self._compute_compression_ratio(original_length, comp_len)
        
        # Decode compressed IDs
        try:
            compressed_text = self.tokenizer.decode(
                compressed_ids, skip_special_tokens=True
            )
        except Exception:
            compressed_text = "[decode error]"
        
        return CompressionResult(
            compressed_ids=compressed_ids,
            compressed_text=compressed_text,
            compression_ratio=ratio,
            token_savings_pct=savings,
            original_length=original_length,
            compressed_length=comp_len,
            method_name=self.get_name(),
            processing_time_ms=processing_time_ms,
            metadata=metadata or {},
        )
    
    def target_length(self, n: int) -> int:
        """Number of tokens to keep for `n` input tokens at the configured ratio.

        Single definition on purpose. The compressors previously used two
        incompatible formulas -- some floored at `min_compressed_length` (1),
        others at `2 * keep_boundary_tokens` (4) -- so at the same configured
        ratio they compressed to different lengths and the benchmark was
        comparing them at different operating points. Measured at n=55,
        ratio=16: tone_aware kept 55 tokens while no_model kept 4.
        """
        ratio = self.config.target_ratio
        if ratio <= 0:
            raise ValueError(
                f"target_ratio must be > 0 (ratio = original/compressed); got {ratio}"
            )
        return max(int(n / ratio), self.config.min_compressed_length)

    def select_with_boundary(self, scores, n: int) -> List[int]:
        """Indices to keep: the `k` boundary tokens plus the top-scoring middle.

        `scores` is any sequence of length `n` where higher means keep.

        Replaces a block that was copy-pasted into seven compressors with a
        bug: when the budget left no room for middle tokens (`mid_budget == 0`,
        i.e. `target_len <= 2k`), the `else` branch returned *every* index, so
        asking for more compression produced none at all. Measured at n=55,
        k=2: ratio 8 kept 6 tokens, ratio 16 kept 55 (CR 1.0). Short inputs hit
        it at ordinary ratios too -- n=10 at ratio 4 was never compressed.
        """
        k = max(0, min(self.config.keep_boundary_tokens, n // 2))
        target_len = min(self.target_length(n), n)
        mid_start, mid_end = k, max(k, n - k)
        mid_budget = max(0, min(target_len - 2 * k, mid_end - mid_start))

        chosen: List[int] = []
        if mid_budget > 0:
            mid = list(range(mid_start, mid_end))
            # Rank by score, then restore original order: compressed output
            # must stay a subsequence of the input.
            chosen = sorted(sorted(mid, key=lambda i: scores[i], reverse=True)[:mid_budget])
        # Keeping only the boundary is the correct answer when the budget is
        # exhausted -- never the whole sequence.
        return sorted(set(range(k)) | set(chosen) | set(range(mid_end, n)))

    def validate_input(self, input_ids: List[int]) -> bool:
        """Validate input before compression."""
        if not input_ids:
            raise ValueError("Empty input sequence")
        if len(input_ids) < self.config.min_compressed_length:
            return False  # Too short to compress
        return True
    
    def __repr__(self) -> str:
        return f"{self.get_name()}(config={self.config})"


class NoCompressor(BaseCompressor):
    """Identity compressor — returns input unchanged (baseline)."""
    
    def compress(self, input_ids: List[int], **kwargs) -> CompressionResult:
        start = time.time()
        self.validate_input(input_ids)
        elapsed = (time.time() - start) * 1000
        
        return self._build_result(
            compressed_ids=list(input_ids),
            original_length=len(input_ids),
            processing_time_ms=elapsed,
        )
    
    def get_name(self) -> str:
        return "NoCompression"


class RandomCompressor(BaseCompressor):
    """Random token dropout — simple baseline."""

    def __init__(self, tokenizer, model=None, config=None, seed: int = 42):
        super().__init__(tokenizer, model, config)
        # Own RNG instead of the global `random` module: the benchmark's
        # headline baseline must not change because some unrelated code called
        # random.seed(), and two runs of run_benchmark.py must agree.
        import random as _random
        self._rng = _random.Random(seed)

    def compress(self, input_ids: List[int], **kwargs) -> CompressionResult:
        start = time.time()
        if not self.validate_input(input_ids):
            return self._build_result(list(input_ids), len(input_ids), 0.0)
        
        n = len(input_ids)
        # Random scores + the shared selector. The previous inline version
        # built `range(n - k, n)` without clamping k, so for n < 2k it produced
        # NEGATIVE indices; `if i < n` let them through and Python wrapped them
        # to the end, duplicating and reordering tokens. Measured: n=3, k=5 on
        # "A B C" returned "B C A B C" -- longer than the input, out of order,
        # and CR 0.60 (< 1).
        scores = [self._rng.random() for _ in range(n)]
        keep_indices = self.select_with_boundary(scores, n)

        compressed = [input_ids[i] for i in keep_indices]
        elapsed = (time.time() - start) * 1000

        return self._build_result(compressed, n, elapsed)
    
    def get_name(self) -> str:
        return "RandomBaseline"
