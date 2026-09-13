"""
Encoder token-classification compression (wave-2 E6 / E11).
============================================================
A cheap "keep/drop" token classifier that replaces the heavy generative
perplexity scorer. This is the LLMLingua-2 recipe (Pan et al., ACL 2024
findings, arxiv:2403.12968): frame compression as *token classification* with a
small bidirectional encoder (BERT-level), rather than one-directional
perplexity from a causal LM.

Why this exists (wave-1 report §8.6, §5): the fine-tuned 4B *generative* scorer
loads ~8 GB of weights merely to rank tokens and still LOSES to an
off-the-shelf 0.5B scorer -- "a better scorer is not a better compressor." A
Vietnamese encoder (PhoBERT / XLM-R, ~135-560M) runs in one bidirectional
forward pass (no sequential sliding-window causal decode) and is the natural
"win on cost" arm. Bidirectional context is also better aligned with the
compression objective than one-directional entropy.

The classifier itself is trained offline by distillation -- see
`scripts/train_encoder_compressor.py`. At inference this module only runs the
trained encoder and keeps the highest-probability tokens.

Registered lazily as the 'encoder' method in compression.METHODS/LAZY_METHODS,
so importing this module (and transformers' encoder stack) only happens when
the 'encoder' arm is actually requested.
"""

from __future__ import annotations

import time
from typing import List, Optional

import torch
import torch.nn.functional as F

from .compression import (
    BaseCompressor,
    CompressionConfig,
    CompressionResult,
    _pool_char_to_token,
    _token_spans,
)


class EncoderClassifierCompressor(BaseCompressor):
    """Keep tokens a bidirectional encoder classifies as "keep" (label
    `keep_label`), LLMLingua-2 style.

    The encoder tokenizes the decoded context with its OWN tokenizer, predicts
    P(keep) per encoder-token, and those probabilities are pooled back onto the
    generation tokenizer's token spans through the same per-character bridge
    LACC uses for a different-tokenizer scorer (`_token_spans` +
    `_pool_char_to_token`). Selection is then the shared boundary-aware top-k.

    Args:
        tokenizer: the generation-model tokenizer (defines the token ids to
            keep/drop and the compressed output).
        model: unused for scoring (kept for the BaseCompressor interface /
            create_compressor signature); scoring is done by the encoder.
        encoder_model: a preloaded `AutoModelForTokenClassification`, or None.
        encoder_tokenizer: a preloaded fast tokenizer for the encoder (needs
            `return_offsets_mapping`), or None.
        encoder_id: HF id to load the encoder+tokenizer from (e.g.
            'vinai/phobert-base', 'xlm-roberta-base') when not preloaded.
        encoder_path: local path to a fine-tuned checkpoint (from
            scripts/train_encoder_compressor.py); takes precedence over encoder_id.
        keep_label: the classifier label id meaning "keep" (default 1).
        max_encoder_len: encoder-token window size for long contexts.
        stride: encoder-token stride between overlapping windows (predictions in
            overlaps are averaged).
    """

    def __init__(
        self,
        tokenizer,
        model=None,
        config: Optional[CompressionConfig] = None,
        device: str = 'cuda',
        encoder_model=None,
        encoder_tokenizer=None,
        encoder_id: Optional[str] = None,
        encoder_path: Optional[str] = None,
        keep_label: int = 1,
        max_encoder_len: int = 512,
        stride: int = 128,
    ):
        super().__init__(tokenizer, model, config)
        self.device = device
        self.keep_label = keep_label
        self.max_encoder_len = max_encoder_len
        self.stride = max(1, min(stride, max_encoder_len - 1)) if max_encoder_len > 1 else 1
        self.encoder_id = encoder_id
        self.encoder_path = encoder_path
        self._encoder = encoder_model
        self._enc_tok = encoder_tokenizer
        # Lazy: an actual model/tokenizer is only loaded on first compress(),
        # so constructing this object (and unit-testing with injected stubs)
        # never triggers a download.

    def get_name(self) -> str:
        tag = self.encoder_path or self.encoder_id or 'encoder'
        return f"EncoderCls[{tag}]"

    # ------------------------------------------------------------------ loading
    def _ensure_encoder(self):
        if self._encoder is not None and self._enc_tok is not None:
            return
        src = self.encoder_path or self.encoder_id
        if src is None:
            raise RuntimeError(
                "EncoderClassifierCompressor has no encoder. Pass encoder_id "
                "(e.g. 'vinai/phobert-base') or encoder_path (a fine-tuned "
                "checkpoint), or train one with scripts/train_encoder_compressor.py."
            )
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        if self._enc_tok is None:
            self._enc_tok = AutoTokenizer.from_pretrained(src, use_fast=True)
        if self._encoder is None:
            self._encoder = AutoModelForTokenClassification.from_pretrained(src)
            if self.device == 'cuda' and torch.cuda.is_available():
                self._encoder = self._encoder.to('cuda')
            self._encoder.eval()

    def _encoder_device(self):
        try:
            return next(self._encoder.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device('cpu')

    # ------------------------------------------------------------------ scoring
    def _encode_offsets(self, text: str):
        """(encoder input_ids, char offsets) with special tokens off so ids and
        offsets align 1:1; falls back to a decode-and-find scan if the tokenizer
        has no fast offset mapping."""
        try:
            enc = self._enc_tok(text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
            return list(enc['input_ids']), list(enc['offset_mapping'])
        except (TypeError, NotImplementedError, ValueError, KeyError):
            ids = self._enc_tok.encode(text, add_special_tokens=False)
            offsets, cursor = [], 0
            for tid in ids:
                piece = self._enc_tok.decode([tid])
                found = text.find(piece, cursor) if piece else -1
                if found < 0:
                    offsets.append((cursor, cursor))
                else:
                    offsets.append((found, found + len(piece)))
                    cursor = found + len(piece)
            return ids, offsets

    @torch.no_grad()
    def _keep_prob_per_char(self, text: str) -> torch.Tensor:
        """Per-character P(keep), NaN where no encoder token covers the char.
        Overlapping windows are averaged."""
        ids, offsets = self._encode_offsets(text)
        char_sum = torch.zeros(len(text))
        char_cnt = torch.zeros(len(text))
        if not ids:
            return torch.full((len(text),), float('nan'))

        device = self._encoder_device()
        n_tok = len(ids)
        begin = 0
        while begin < n_tok:
            end = min(begin + self.max_encoder_len, n_tok)
            window_ids = ids[begin:end]
            tensor = torch.tensor([window_ids], device=device)
            logits = self._encoder(tensor).logits  # [1, S, num_labels]
            probs = F.softmax(logits.float(), dim=-1)[0]  # [S, num_labels]
            label = min(self.keep_label, probs.shape[-1] - 1)
            pkeep = probs[:, label].cpu()
            for k, (s, e) in enumerate(offsets[begin:end]):
                if e > s and k < pkeep.numel():
                    char_sum[s:e] += float(pkeep[k])
                    char_cnt[s:e] += 1.0
            if end >= n_tok:
                break
            begin += self.stride

        char_scores = torch.full((len(text),), float('nan'))
        covered = char_cnt > 0
        char_scores[covered] = char_sum[covered] / char_cnt[covered]
        return char_scores

    # ------------------------------------------------------------------ compress
    def compress(self, input_ids: List[int], query: Optional[str] = None, task: Optional[str] = None, **kwargs) -> CompressionResult:
        start = time.time()
        n = len(input_ids)
        if not self.validate_input(input_ids):
            return self._build_result(list(input_ids), n, (time.time() - start) * 1000)

        self._ensure_encoder()
        text, spans, _ = _token_spans(self.tokenizer, input_ids)
        char_scores = self._keep_prob_per_char(text)
        # fill_neutral: tokens the encoder never covered get the median keep
        # prob (neutral), not treated as maximally droppable.
        token_scores = _pool_char_to_token(char_scores, spans, n, fill_neutral=True)

        keep_indices = self.select_with_boundary(token_scores.tolist(), n)
        compressed = [input_ids[i] for i in keep_indices]

        return self._build_result(
            compressed, n, (time.time() - start) * 1000,
            metadata={
                'encoder_id': self.encoder_path or self.encoder_id,
                'keep_label': self.keep_label,
                'query_applied': False,  # query-agnostic token-classification faithfulness compressor
            },
        )
