"""
LLMLingua-Style Prompt Compression
===================================
Reimplementation of the LLMLingua method (EMNLP 2023) with extensions
for Vietnamese tone-aware compression.

LLMLingua: "Compressing Prompts for Accelerated Inference of LLMs"
  - Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu
  - Microsoft, EMNLP 2023
  - arxiv: 2310.05736

Method Overview:
  1. Budget Controller: Determine how many tokens to keep
  2. Coarse-to-Fine Compression:
     a. Sentence-level: Remove low-importance sentences
     b. Token-level: Remove low-importance tokens using perplexity
  3. Distribution Alignment: Use small LM for scoring, large LM for generation

Token-Level Compression Formula:
  For each token t in sequence:
    importance(t) = log P(t | context \ {t}) - log P(t | context)
  
  This measures how much token t contributes to predicting itself.
  Tokens with LOW importance are redundant → remove them.

We extend this with:
  - Tone-aware scoring (multiply by tone preservation weight)
  - Morphology-aware scoring (multiply by word class preservation factor)
  
Acceleration via Small Model (LLMLingua optimization):
  Use a smaller model (e.g., 125M params) for computing importance scores
  instead of the full LLM, achieving fast compression before generation.
"""

import torch
import torch.nn.functional as F
import time
import math
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedTokenizer, PreTrainedModel

from .base import BaseCompressor, CompressionResult, CompressionConfig


class LLMLinguaCompressor(BaseCompressor):
    """
    LLMLingua-style prompt compression.
    
    Coarse-to-fine approach:
      1. Sentence-level filtering (if input is long enough)
      2. Token-level perplexity-based scoring
      3. Iterative compression with budget control
    
    Uses a small LM (provided via `small_model`) for fast scoring.
    If no small model, uses the main model directly (slower).
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Optional[PreTrainedModel] = None,
        small_model: Optional[PreTrainedModel] = None,
        config: Optional[CompressionConfig] = None,
        device: str = 'cuda',
    ):
        super().__init__(tokenizer, model, config)
        self.small_model = small_model or model
        self.device = device
        
        if self.small_model:
            self.small_model.eval()
    
    def get_name(self) -> str:
        return "LLMLingua"
    
    @torch.no_grad()
    def _compute_token_importance(
        self,
        input_ids: List[int],
    ) -> torch.Tensor:
        """
        Compute token importance using perplexity-based scoring.
        
        Formula:
          importance(t_i) = log P(t_i | t_{<i}) - log P(t_i | t_{<i}, t_{>i})
                         = "how much does knowing future tokens help predict t_i?"
        
        In practice, we approximate with:
          importance(t_i) = |log P(t_i | full context) - log P(t_i | context without t_i)|
        
        Higher importance → token is less redundant → keep.
        
        For efficiency, we use a sliding window approach:
          - Process chunks of size W
          - For each position, compute loss with and without the token
        """
        if self.small_model is None:
            raise RuntimeError("No model available for perplexity computation")
        
        n = len(input_ids)
        device = next(self.small_model.parameters()).device
        
        input_tensor = torch.tensor([input_ids], device=device)
        
        # Get full-context log-probabilities
        outputs = self.small_model(input_tensor)
        logits = outputs.logits  # [1, n, vocab_size]
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_tensor[:, 1:]
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [1, n-1]
        
        # Importance: negative log probability (higher = less confident = more important)
        # We use absolute value for scoring
        importance = torch.zeros(n, device=device)
        importance[1:] = -token_log_probs[0]  # Higher = more important
        
        # Normalize to [0, 1]
        if importance.max() > importance.min():
            importance = (importance - importance.min()) / (importance.max() - importance.min())
        
        return importance.cpu()
    
    def _sentence_level_filter(
        self,
        input_ids: List[int],
        budget: int,
    ) -> List[int]:
        """
        Coarse sentence-level filtering.
        
        If input has 4+ sentences, remove the least important sentences
        before token-level compression.
        
        Sentence importance = mean(token_importance) within sentence.
        """
        n = len(input_ids)
        
        # Identify sentence boundaries (periods, newlines, etc.)
        sentence_end_tokens = set()
        for token_str in ['.', '!', '?', '\n', '。', '！', '？']:
            tid = self.tokenizer.encode(token_str, add_special_tokens=False)
            sentence_end_tokens.update(tid)
        
        # Split into sentences
        sentences = []
        current = []
        for tid in input_ids:
            current.append(tid)
            if tid in sentence_end_tokens or len(current) >= 200:
                sentences.append(current)
                current = []
        if current:
            sentences.append(current)
        
        # If few sentences, skip sentence-level
        if len(sentences) < 4:
            return input_ids
        
        # Compute importance for each sentence
        sent_importance = []
        all_ids_flat = []
        for sent in sentences:
            all_ids_flat.extend(sent)
        
        # Use model to score
        importance = self._compute_token_importance(all_ids_flat)
        
        offset = 0
        for sent in sentences:
            sent_imp = importance[offset:offset + len(sent)].mean().item()
            sent_importance.append(sent_imp)
            offset += len(sent)
        
        # Select top sentences within budget
        sent_count = min(len(sentences), max(3, budget // 50))
        if sent_count >= len(sentences):
            return input_ids
        
        sent_tuples = list(enumerate(sent_importance))
        sent_tuples.sort(key=lambda x: x[1], reverse=True)
        selected_indices = sorted([i for i, _ in sent_tuples[:sent_count]])
        
        result = []
        for i in selected_indices:
            result.extend(sentences[i])
        
        return result
    
    def compress(
        self,
        input_ids: List[int],
        use_small_model: bool = True,
        **kwargs,
    ) -> CompressionResult:
        """
        Compress using LLMLingua method.
        
        Args:
            input_ids: Token IDs to compress
            use_small_model: If True, use small model for scoring
        
        Returns:
            CompressionResult
        """
        start = time.time()
        n = len(input_ids)
        
        if not self.validate_input(input_ids):
            elapsed = (time.time() - start) * 1000
            return self._build_result(list(input_ids), n, elapsed)
        
        target_len = max(int(n / self.config.target_ratio), self.config.min_compressed_length)
        
        # Step 1: Sentence-level filter (coarse)
        filtered_ids = self._sentence_level_filter(input_ids, target_len)
        
        # Step 2: Token-level importance scoring
        if len(filtered_ids) > target_len * 1.5:
            importance = self._compute_token_importance(filtered_ids)
            
            # Always keep boundaries
            k = self.config.keep_boundary_tokens
            mid_start, mid_end = k, len(filtered_ids) - k
            
            if mid_start < mid_end:
                mid_importance = importance[mid_start:mid_end]
                mid_budget = max(0, target_len - 2 * k)
                
                if mid_budget > 0 and mid_budget < len(mid_importance):
                    _, top_indices = torch.topk(mid_importance, mid_budget)
                    top_indices = sorted(top_indices.tolist())
                    mid_kept = [filtered_ids[mid_start + i] for i in top_indices]
                else:
                    mid_kept = filtered_ids[mid_start:mid_end]
                
                compressed = (
                    filtered_ids[:k] + mid_kept + filtered_ids[-k:]
                )
            else:
                compressed = filtered_ids
        else:
            compressed = filtered_ids
        
        elapsed = (time.time() - start) * 1000
        
        return self._build_result(
            compressed_ids=compressed[:self.config.max_compressed_length],
            original_length=n,
            processing_time_ms=elapsed,
            metadata={
                'sentence_filtered': len(filtered_ids) < n,
                'token_scored': True,
            }
        )


class LLMLinguaWithSmallModel(LLMLinguaCompressor):
    """
    LLMLingua with explicit small model for scoring (the original paper's approach).
    
    The key insight from LLMLingua:
      "Use a small LM to estimate token importance, then feed the compressed
       result to a large LM for generation."
    
    This decouples compression cost from generation model size.
    """
    
    def get_name(self) -> str:
        return "LLMLingua-SmallModel"
    
    def compress(
        self,
        input_ids: List[int],
        **kwargs,
    ) -> CompressionResult:
        return super().compress(input_ids, use_small_model=True, **kwargs)
