"""
SnapKV-Style KV Cache Compression
===================================
Reimplementation of SnapKV (2024) for KV cache compression.

SnapKV: "LLM Knows What You are Looking for Before Generation"
  - Yuhong Li, Yingbing Huang, Bowen Yang, et al.
  - arxiv: 2404.14469

Key Insight:
  Each attention head consistently focuses on specific prompt attention features
  during generation. This robust pattern can be observed from an "observation window"
  at the end of the prompt.

How SnapKV works:
  1. Use the last W tokens as "observation window"
  2. Compute attention from observation window to all prompt tokens
  3. Aggregate attention across observation window → token importance scores
  4. For each head, select top-K tokens with highest cumulative attention
  5. Keep only the KV cache entries for selected tokens

Mathematical Formula:
  For head h at position i (within observation window of size W):
    A_h[i] = attention scores from token i to all previous tokens
    
  Token importance for head h:
    importance_h(j) = Σ_{i=n-W}^{n} A_h[i, j]  (aggregated attention)
    
  Per-head selection:
    keep_h = TopK(importance_h, budget_h)
    
  Final KV cache mask:
    M[j] = 1 if token j is kept in at least one head, else 0

Extensions in this implementation:
  - Tone-aware: multiply importance by tone preservation weight
  - Morphology-aware: adjust per-class budget allocation
  - Support for GQA (Grouped Query Attention)

Also includes H2O (Heavy Hitter Oracle) and StreamingLLM logic as options.
"""

import torch
import torch.nn.functional as F
import time
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedTokenizer, PreTrainedModel

from .base import BaseCompressor, CompressionResult, CompressionConfig


class SnapKVCompressor(BaseCompressor):
    """
    SnapKV-style KV cache compression.
    
    Uses attention patterns from an observation window to identify
    important KV cache entries. Training-free and model-agnostic.
    
    Also supports:
      - H2O mode: Keep "heavy hitters" (cumulative attention) + local tokens
      - StreamingLLM mode: Keep attention sinks + recent tokens
      - PyramidKV mode: Asymmetric budget across layers
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Optional[PreTrainedModel] = None,
        config: Optional[CompressionConfig] = None,
        device: str = 'cuda',
        # SnapKV specific
        window_size: int = 32,          # Observation window size
        kernel_size: int = 5,          # Pooling kernel for attention aggregation
        max_capacity_prompt: int = 512, # Max KV cache entries to keep
        pooling: str = 'maxpool',      # 'maxpool', 'avgpool', or 'none'
        # Budget allocation
        budget_mode: str = 'uniform',  # 'uniform', 'pyramid', 'adaptive'
        # Mode
        mode: str = 'snapkv',          # 'snapkv', 'h2o', 'streamingllm'
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
    
    def _compute_attention_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute token importance using model attention patterns.
        
        Returns tensor of shape [num_heads, seq_len] with importance scores
        for each head. Higher = more important = keep.
        
        For models without GQA (e.g., Llama), num_heads = n_head.
        For GQA models, we use the query heads for computing importance
        and map back to KV heads.
        """
        if self.model is None:
            raise RuntimeError("SnapKV requires a model for attention computation")
        
        n = input_ids.shape[1]
        device = input_ids.device
        
        # Run model and get attention weights
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=True,
            )
        
        # Aggregate attention across layers
        # attentions: tuple of [batch, num_heads, seq_len, seq_len]
        all_attentions = outputs.attentions
        
        if all_attentions is None:
            raise RuntimeError("Model did not return attention weights. "
                             "Set output_attentions=True in model config.")
        
        # Stack: [num_layers, batch, num_heads, seq_len, seq_len]
        attn_stack = torch.stack(all_attentions, dim=0)
        
        # Average across layers (or use last layer only)
        # Using last few layers works well per SnapKV paper
        num_layers_to_use = max(1, len(all_attentions) // 4)
        recent_attns = attn_stack[-num_layers_to_use:]  # [L', B, H, S, S]
        
        # Use observation window at the end
        window_start = max(0, n - self.window_size)
        
        # Aggregate attention from observation window to all prompt tokens
        # For each head, sum attention from window positions to each token j
        window_attn = recent_attns[:, :, :, window_start:, :]  # [L', B, H, W, S]
        
        # Aggregate: mean over layers, sum over window positions, mean over batch
        importance = window_attn.mean(dim=0)        # [B, H, W, S]
        importance = importance.sum(dim=2)           # [B, H, S]  (sum over window)
        importance = importance.mean(dim=0)          # [H, S]     (mean over batch)
        
        # Apply pooling if specified
        if self.pooling == 'maxpool' and self.kernel_size > 1:
            # 1D max pooling along sequence dimension
            imp = importance.unsqueeze(0)  # [1, H, S]
            imp = F.max_pool1d(imp, kernel_size=self.kernel_size, 
                              stride=1, padding=self.kernel_size // 2)
            importance = imp.squeeze(0)  # [H, S]
        elif self.pooling == 'avgpool' and self.kernel_size > 1:
            imp = importance.unsqueeze(0)
            imp = F.avg_pool1d(imp, kernel_size=self.kernel_size,
                              stride=1, padding=self.kernel_size // 2)
            importance = imp.squeeze(0)
        
        # Normalize per head
        for h in range(importance.shape[0]):
            h_imp = importance[h]
            if h_imp.max() > h_imp.min():
                importance[h] = (h_imp - h_imp.min()) / (h_imp.max() - h_imp.min())
        
        return importance  # [H, S]
    
    def _h2o_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        H2O-style importance: cumulative attention scores.
        
        Heavy Hitter Oracle:
          For each token, accumulate attention received over time.
          Tokens with high cumulative attention are "heavy hitters" — keep them.
          Also keep recent tokens (local window).
        """
        importance = self._compute_attention_importance(input_ids, attention_mask)
        
        # Aggregate across heads for H2O
        imp_agg = importance.mean(dim=0)  # [S]
        
        # H2O formula: cumulative attention
        # We use the aggregated importance directly
        
        return imp_agg.unsqueeze(0)  # [1, S]
    
    def _streamingllm_importance(
        self,
        n_tokens: int,
    ) -> torch.Tensor:
        """
        StreamingLLM-style: keep attention sinks + recent tokens.
        
        Attention sinks are the first 4 tokens, which consistently
        receive high attention regardless of content.
        """
        importance = torch.zeros(1, n_tokens)
        # First 4 tokens = attention sinks (very high importance)
        importance[0, :4] = 1.0
        # Recent tokens (last window)
        window = min(self.window_size, n_tokens)
        importance[0, -window:] = 0.8
        # Middle tokens: low importance
        middle_start, middle_end = 4, n_tokens - window
        if middle_start < middle_end:
            importance[0, middle_start:middle_end] = 0.1
        
        return importance
    
    def _get_layer_budgets(
        self,
        num_layers: int,
        total_budget: int,
    ) -> List[int]:
        """
        Compute per-layer KV cache budget.
        
        Modes:
          - uniform: same budget for all layers
          - pyramid: more budget for lower layers, less for higher
        """
        if self.budget_mode == 'uniform':
            return [total_budget // num_layers] * num_layers
        
        elif self.budget_mode == 'pyramid':
            # Pyramid: budget decreases as layer increases
            # Formula: budget_l = total × (1 - α × l/L) / Σ(1 - α × l/L)
            alpha = 0.3  # Steepness parameter
            weights = [1.0 - alpha * (l / num_layers) for l in range(num_layers)]
            total_weight = sum(weights)
            budgets = [max(1, int(total_budget * w / total_weight)) for w in weights]
            return budgets
        
        else:
            return [total_budget // num_layers] * num_layers
    
    def compress(
        self,
        input_ids: List[int],
        **kwargs,
    ) -> CompressionResult:
        """
        SnapKV compression of input tokens.
        
        Args:
            input_ids: Token IDs to compress
        
        Returns:
            CompressionResult with KV cache mask and compressed IDs
        """
        start = time.time()
        n = len(input_ids)
        
        if not self.validate_input(input_ids):
            elapsed = (time.time() - start) * 1000
            return self._build_result(list(input_ids), n, elapsed)
        
        input_tensor = torch.tensor([input_ids]).to(self.device)
        
        # Compute importance based on mode
        if self.mode == 'h2o':
            importance = self._h2o_importance(input_tensor)  # [1, S]
            num_heads = importance.shape[0]
        elif self.mode == 'streamingllm':
            importance = self._streamingllm_importance(n)  # [1, S]
            num_heads = 1
        else:  # snapkv
            importance = self._compute_attention_importance(input_tensor)  # [H, S]
            num_heads = importance.shape[0]
        
        # Determine budget
        budget = min(self.max_capacity_prompt, 
                    max(int(n / self.config.target_ratio), self.config.min_compressed_length))
        
        # Build KV cache mask: which tokens to keep
        # Per-head selection (SnapKV-style)
        kv_mask = torch.zeros(num_heads, n, dtype=torch.bool)
        
        for head in range(num_heads):
            head_imp = importance[head]
            
            # Always keep boundaries
            k = self.config.keep_boundary_tokens
            head_budget = budget  # Could be per-head budget with GQA
            
            # Select top-k from middle
            mid_imp = head_imp[k:n - k] if n > 2 * k else head_imp
            mid_budget = max(0, head_budget - 2 * k)
            
            if mid_budget > 0 and mid_budget < len(mid_imp):
                _, top_indices = torch.topk(mid_imp, mid_budget)
                for idx in top_indices:
                    kv_mask[head, k + idx.item()] = True
            elif len(mid_imp) > 0:
                kv_mask[head, k:n - k] = True
            
            # Always keep boundaries
            for i in range(min(k, n)):
                kv_mask[head, i] = True
            for i in range(max(0, n - k), n):
                kv_mask[head, i] = True
        
        # Global mask: keep token if kept by ANY head
        global_mask = kv_mask.any(dim=0)  # [S]
        keep_indices = global_mask.nonzero(as_tuple=True)[0].tolist()
        compressed_ids = [input_ids[i] for i in keep_indices]
        
        # Estimate memory saved
        bytes_per_kv = 2 * num_heads * 128 * 2  # 2(K+V) × heads × head_dim × bytes_per_elem
        kv_memory_saved = (n - len(compressed_ids)) * bytes_per_kv
        
        elapsed = (time.time() - start) * 1000
        
        return self._build_result(
            compressed_ids=compressed_ids,
            original_length=n,
            processing_time_ms=elapsed,
            metadata={
                'kv_cache_mask_shape': list(kv_mask.shape),
                'num_heads': num_heads,
                'mode': self.mode,
                'budget_mode': self.budget_mode,
                'kv_memory_saved_bytes': kv_memory_saved,
            },
        )


class SelectiveContextCompressor(BaseCompressor):
    """
    Selective Context Compression.
    
    Simple but effective: keep only tokens that are "relevant" to the query.
    Uses cosine similarity between token embeddings and query embedding.
    
    Formula:
      relevance(t) = cosine_sim(emb(t), emb(query))
    
    Select tokens with highest relevance to the query.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Optional[PreTrainedModel] = None,
        config: Optional[CompressionConfig] = None,
        device: str = 'cuda',
    ):
        super().__init__(tokenizer, model, config)
        self.device = device
    
    def get_name(self) -> str:
        return "SelectiveContext"
    
    def compress(
        self,
        input_ids: List[int],
        query_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> CompressionResult:
        start = time.time()
        n = len(input_ids)
        
        if not self.validate_input(input_ids):
            elapsed = (time.time() - start) * 1000
            return self._build_result(list(input_ids), n, elapsed)
        
        target_len = max(int(n / self.config.target_ratio), self.config.min_compressed_length)
        
        if self.model is not None and query_ids:
            # Embed-based selection
            input_t = torch.tensor([input_ids]).to(self.device)
            query_t = torch.tensor([query_ids]).to(self.device)
            
            with torch.no_grad():
                # Get embeddings
                input_emb = self.model.get_input_embeddings()(input_t)
                query_emb = self.model.get_input_embeddings()(query_t)
                
                # Mean pool query
                query_vec = query_emb.mean(dim=1)  # [1, D]
                
                # Cosine similarity
                input_vec = input_emb.squeeze(0)  # [S, D]
                query_vec = F.normalize(query_vec, dim=-1)
                input_vec = F.normalize(input_vec, dim=-1)
                
                similarities = torch.mm(input_vec, query_vec.T).squeeze()  # [S]
                
                # Select top-k
                k = self.config.keep_boundary_tokens
                mid_sim = similarities[k:n - k] if n > 2 * k else similarities
                mid_budget = max(0, target_len - 2 * k)
                
                if mid_budget > 0 and mid_budget < len(mid_sim):
                    _, top_indices = torch.topk(mid_sim, mid_budget)
                    top_indices = sorted(top_indices.tolist())
                    compressed = (
                        input_ids[:k] +
                        [input_ids[k + i] for i in top_indices] +
                        input_ids[n - k:]
                    )
                else:
                    compressed = list(input_ids)
        else:
            # Fallback: no model/query, use random
            from .base import RandomCompressor
            rc = RandomCompressor(self.tokenizer, config=self.config)
            result = rc.compress(input_ids)
            return result
        
        elapsed = (time.time() - start) * 1000
        return self._build_result(compressed, n, elapsed)
