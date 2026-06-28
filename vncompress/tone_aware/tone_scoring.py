"""
Tone-Aware Compression
======================
Novel compression methods that preserve Vietnamese tonal information.

Key Mathematical Formulas:
-------------------------

1. TONE PRESERVATION WEIGHT:

   w_tone(t) = 1.0 + α × ρ(t) × (1 + β × ν(t) / 6)

   where:
     ρ(t) = |{c ∈ t : tone(c) ≠ ngang}| / |t|     (tone density)
     ν(t) = |{unique tones in t}|                   (tone variety)
     α   = base importance (default 0.5)
     β   = variety bonus (default 0.3)

2. CONTRAST FACTOR:

   f_contrast(t) = 1 + γ × (1/|N|) Σ_{n∈N} D_tone(t, n)

   where:
     N  = neighbor tokens in window
     D_tone = tone contrast matrix value
     γ  = contrast amplification (default 0.4)

3. TONE-AWARE SCORE (for token selection):

   S_tone(t) = S_base(t) × w_tone(t) × f_contrast(t)

   where S_base(t) is the base compression score (perplexity, attention, etc.)

4. PHONOLOGICAL CONSISTENCY LOSS (for training):

   L_tone = (1/|T|) Σ_{t∈T} CrossEntropy(pred_tone_seq, true_tone_seq)

   Total loss: L = L_LM + λ × L_tone

5. COMPRESSION RATIO:

   CR = N_original / N_compressed

6. TONE PRESERVATION RATE:

   TPR = |{t ∈ compressed : tone(t) preserved}| / |{t ∈ original : tone(t) ≠ ngang}|

Reference Papers:
  - arxiv:2605.09463 "Beyond Position Bias" (SeCo) — semantic consistency
  - arxiv:2606.15044 "Equity with Efficiency" — tokenizer equity for SEA languages
  - arxiv:2606.03618 "Cross-Lingual Token Arbitrage" — non-English prompt optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedModel

from .vietnamese_tones import (
    VietnameseToneAnalyzer,
    TokenToneInfo,
    TONE_NAME_TO_ID,
    TONE_CONTRAST,
    get_tone_analyzer,
)


@dataclass
class ToneAwareConfig:
    """Configuration for Tone-Aware Compression."""
    alpha: float = 0.5          # Base tone importance weight
    beta: float = 0.3           # Tone variety bonus
    gamma: float = 0.4          # Contrast amplification
    window_size: int = 2        # Context window for contrast computation
    tone_embed_dim: int = 64    # Dimension for tone embeddings
    lambda_tone: float = 0.1    # Weight for phonological consistency loss
    use_tone_embedding: bool = True
    use_contrast: bool = True
    min_preservation_weight: float = 1.0
    max_preservation_weight: float = 3.0


class ToneAwareScorer:
    """
    Compute tone-aware token importance scores for compression.

    Extends base compression methods (LLMLingua, SnapKV) with
    tone-aware scoring to better preserve Vietnamese tonal information.

    Usage:
        scorer = ToneAwareScorer(tokenizer, config)
        base_scores = get_base_compression_scores(tokens)  # from LLMLingua/SnapKV
        tone_scores = scorer.score_tokens(tokens, base_scores)
        keep_indices = select_top_k(tone_scores, budget)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: Optional[ToneAwareConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config or ToneAwareConfig()
        self.tone_analyzer = get_tone_analyzer(
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
        )

    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """Decode individual token IDs to their string representations."""
        tokens = []
        for tid in token_ids:
            token_str = self.tokenizer.decode([tid])
            # Clean up tokenizer artifacts for analysis
            token_str = token_str.replace('▁', ' ').replace('Ġ', ' ').strip()
            tokens.append(token_str)
        return tokens

    def compute_tone_scores(
        self,
        token_ids: List[int],
        base_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute tone-aware scores for a sequence of tokens.

        Args:
            token_ids: List of token IDs
            base_scores: Base compression scores (shape: [seq_len]) from
                         LLMLingua (perplexity-based) or SnapKV (attention-based).
                         If None, returns only tone weights.

        Returns:
            tone_scores: Tensor of shape [seq_len] with tone-aware scores
        """
        tokens = self.decode_tokens(token_ids)
        seq_len = len(tokens)

        # Step 1: Analyze tones for all tokens
        tone_infos = self.tone_analyzer.analyze_tokens(
            tokens, window_size=self.config.window_size
        )

        # Step 2: Build tone weight vector
        tone_weights = torch.ones(seq_len)
        for i, info in enumerate(tone_infos):
            w = info.preservation_weight
            w = max(self.config.min_preservation_weight,
                    min(self.config.max_preservation_weight, w))
            tone_weights[i] = w

        # Step 3: Combine with base scores
        if base_scores is not None:
            # Normalize base scores to [0, 1]
            if base_scores.max() > base_scores.min():
                base_norm = (base_scores - base_scores.min()) / (
                    base_scores.max() - base_scores.min()
                )
            else:
                base_norm = base_scores

            # Tone-aware score = base × tone_weight
            # Higher → more important → keep
            tone_scores = base_norm * tone_weights
        else:
            tone_scores = tone_weights

        return tone_scores

    def select_tokens(
        self,
        token_ids: List[int],
        budget: int,
        base_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Select top-k tokens to keep based on tone-aware scoring.

        Args:
            token_ids: Full list of token IDs
            budget: Number of tokens to keep (compression budget)
            base_scores: Optional base compression scores

        Returns:
            keep_ids: Selected token IDs
            all_scores: Full score tensor (for analysis)
        """
        tone_scores = self.compute_tone_scores(token_ids, base_scores)

        # Always keep first and last tokens (boundary preservation)
        n = len(token_ids)
        if budget >= n:
            return token_ids, tone_scores

        # Select top-(budget-2) from middle tokens + 2 boundaries
        mid_scores = tone_scores[1:-1].clone()
        mid_budget = budget - 2

        if mid_budget <= 0:
            # Budget too small, just keep boundaries
            keep_indices = [0, n - 1] if n > 2 else list(range(n))
        else:
            _, mid_top_indices = torch.topk(mid_scores, min(mid_budget, len(mid_scores)))
            # Convert to original indices (+1 offset for boundary)
            mid_top_indices = mid_top_indices + 1
            keep_indices = sorted([0] + mid_top_indices.tolist() + [n - 1])

        keep_ids = [token_ids[i] for i in keep_indices if i < n]
        return keep_ids, tone_scores


class ToneEmbeddingAugmentation(nn.Module):
    """
    Add tone embeddings to token embeddings for compression-aware training.

    Mathematical Formula:
      e'_t = e_t ⊕ W_tone[tone_id(t)]

    where:
      e_t           = original token embedding (d_model dim)
      W_tone        = learnable tone embedding matrix (7 × tone_embed_dim)
      tone_id(t)    = tone class of token t (0-6)
      ⊕             = concatenation
      e'_t          = augmented embedding (d_model + tone_embed_dim dim)

    This augmented embedding is fed to the compression model so it can
    learn to preserve tonal information.

    For non-tonal tokens or non-Vietnamese text, tone_id = 0 (neutral/no-tone).
    """

    def __init__(
        self,
        d_model: int,
        tone_embed_dim: int = 64,
        num_tones: int = 7,  # 0=no_tone/ngang, 1=huyền, ..., 5=nặng, 6=unknown
    ):
        super().__init__()
        self.d_model = d_model
        self.tone_embed_dim = tone_embed_dim
        self.num_tones = num_tones

        # Learnable tone embeddings
        self.tone_embedding = nn.Embedding(num_tones, tone_embed_dim)
        nn.init.normal_(self.tone_embedding.weight, std=0.02)

        # Projection layer: concat → back to d_model (if needed)
        self.projection = nn.Linear(d_model + tone_embed_dim, d_model, bias=False)
        self.use_projection = False  # Set True to project back to d_model

    def forward(
        self,
        token_embeddings: torch.Tensor,
        tone_ids: torch.Tensor,
        project: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            token_embeddings: [batch, seq_len, d_model]
            tone_ids: [batch, seq_len] integer tone IDs (0-6)

        Returns:
            Augmented embeddings (if project=False: [B, S, d_model + tone_embed_dim])
        """
        tone_embeds = self.tone_embedding(tone_ids)  # [B, S, tone_embed_dim]
        augmented = torch.cat([token_embeddings, tone_embeds], dim=-1)

        if project or self.use_projection:
            augmented = self.projection(augmented)

        return augmented


class PhonologicalConsistencyLoss(nn.Module):
    """
    Auxiliary loss for preserving tonal information during compression training.

    Mathematical Formula:
      L_tone = (1/N) Σ_{i=1}^{N} CE(p_i, y_i)

    where:
      p_i  = predicted tone distribution at position i (from compressed rep)
      y_i  = ground-truth tone class at position i
      N    = number of positions
      CE   = cross-entropy loss

    Combined with LM loss:
      L_total = L_LM + λ × L_tone

    This encourages the model to retain tonal information in compressed
    representations, which is critical for Vietnamese where tone changes
    word meaning.

    Reference: Inspired by phonetic auxiliary objectives in speech processing.
    """

    def __init__(
        self,
        num_tones: int = 7,
        lambda_tone: float = 0.1,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.num_tones = num_tones
        self.lambda_tone = lambda_tone
        self.reduction = reduction

        # Tone classifier head
        self.tone_classifier = nn.Linear(1, num_tones)  # Will be replaced at forward

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, D] compressed representation
        tone_labels: torch.Tensor,      # [B, S] ground-truth tone IDs
        mask: Optional[torch.Tensor] = None,  # [B, S] valid positions
    ) -> torch.Tensor:
        """
        Compute phonological consistency loss.

        Args:
            hidden_states: Hidden states from compressed model
            tone_labels: Ground-truth tone class for each position
            mask: Optional mask for valid positions (1=valid, 0=ignore)

        Returns:
            Scalar loss value
        """
        B, S, D = hidden_states.shape

        # Ensure tone classifier matches hidden dim
        if self.tone_classifier.in_features != D:
            self.tone_classifier = nn.Linear(D, self.num_tones).to(
                hidden_states.device
            )

        logits = self.tone_classifier(hidden_states)  # [B, S, num_tones]

        if mask is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_tones),
                tone_labels.view(-1),
                reduction='none',
            )
            loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-8)
        else:
            loss = F.cross_entropy(
                logits.view(-1, self.num_tones),
                tone_labels.view(-1),
                reduction=self.reduction,
            )

        return self.lambda_tone * loss


class ToneAugmentedTrainer:
    """
    Training helper for tone-aware compression models.

    Wraps the training loop with:
      1. Tone embedding injection
      2. Phonological consistency loss
      3. Tone preservation rate tracking

    Usage:
        trainer = ToneAugmentedTrainer(model, tokenizer, config)
        trainer.train_step(batch)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: ToneAwareConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.tone_analyzer = get_tone_analyzer()
        
        self.tone_augmentation = ToneEmbeddingAugmentation(
            d_model=model.config.hidden_size,
            tone_embed_dim=config.tone_embed_dim,
        )
        self.tone_loss = PhonologicalConsistencyLoss(
            lambda_tone=config.lambda_tone,
        )

    def compute_tone_labels(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token tone IDs from input token IDs.

        Returns tensor of same shape as input_ids with tone class 0-6.
        """
        B, S = input_ids.shape
        tone_labels = torch.zeros(B, S, dtype=torch.long)

        for b in range(B):
            token_ids = input_ids[b].tolist()
            tokens = []
            for tid in token_ids:
                t = self.tokenizer.decode([tid])
                tokens.append(t if len(t) <= 10 else t[:10])
            
            for s in range(S):
                token = tokens[s] if s < len(tokens) else ''
                dominant = self.tone_analyzer.get_dominant_tone(token)
                tone_id = TONE_NAME_TO_ID.get(dominant or 'ngang', 0)
                tone_labels[b, s] = tone_id

        return tone_labels

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Single training step with tone-aware objectives.

        Returns dict of loss values for logging.
        """
        self.model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        lm_loss = outputs.loss

        # Compute tone labels
        tone_labels = self.compute_tone_labels(input_ids)
        tone_labels = tone_labels.to(input_ids.device)

        # Phonological consistency loss on last hidden states
        hidden_states = outputs.hidden_states[-1]  # [B, S, D]
        tone_l = self.tone_loss(hidden_states, tone_labels, attention_mask)

        # Combined loss
        total_loss = lm_loss + tone_l

        total_loss.backward()
        optimizer.step()

        return {
            'lm_loss': lm_loss.item(),
            'tone_loss': tone_l.item(),
            'total_loss': total_loss.item(),
        }

    def compute_tone_preservation_rate(
        self,
        original_ids: torch.Tensor,
        compressed_rep: torch.Tensor,
    ) -> float:
        """
        Measure how well tonal information is preserved after compression.

        Trains a small probe on the compressed representation to predict
        original tone labels. Higher accuracy = better tone preservation.
        """
        # Placeholder: in practice, train a lightweight probe
        # and evaluate on held-out data
        return 0.0
