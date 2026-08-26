"""LACC `lightweight` mode: a real trained SLM as the external perplexity scorer.

This closes the gap where `run_train_slm.py` produced a fine-tuned Vietnamese
SLM that nothing in the benchmark could actually use: `TinyModelScorer` /
`EnhancedCompressor` in external_scorer.py are not in COMPRESSOR_REGISTRY and
do not implement the BaseCompressor interface, so `run_benchmark.py` had no
path to them. Without this, "the SLM improves compression" is untested.

Two registry entries let the SLM's contribution be isolated:
  - `slm_scorer`      -- the fine-tuned adapter (pass --scorer-adapter-dir)
  - `slm_scorer_base` -- the same architecture WITHOUT the LoRA adapter

Comparing those two against `combined` (tone+morph, no model at all) answers
the question the perplexity/tone numbers cannot: does fine-tuning the scorer
change downstream task quality?

Tokenizer mismatch is handled explicitly. The benchmark hands `compress()`
token ids from the *generation* model's tokenizer (e.g. Qwen, ~152k vocab),
while the scorer is a different model with its own tokenizer (e.g. GPT-2,
~50k vocab). Feeding one model's ids to the other is meaningless and can
index out of range, so scores are aligned through character offsets on the
decoded text instead.
"""

import time
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..morphology.merge_policy import WordClass, get_morphology_analyzer
from ..tone_aware.vietnamese_tones import (
    compute_tone_preservation_rate,
    get_tone_analyzer,
)
from .base import BaseCompressor, CompressionConfig, CompressionResult
from .external_scorer import ScoreWeights

# Matches EnhancedCompressor's multipliers so the morphology signal is the
# same one the no-model compressors use; only the perplexity term differs.
_CLASS_MULTIPLIER = {
    WordClass.FUNC: 0.40,
    WordClass.CONTENT: 1.20,
    WordClass.REDUP: 0.60,
    WordClass.COMPOUND: 1.50,
    WordClass.OTHER: 1.00,
}


def _normalize(scores: torch.Tensor) -> torch.Tensor:
    """Min-max to [0, 1]; constant input maps to a neutral 0.5."""
    if scores.numel() == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi > lo:
        return (scores - lo) / (hi - lo)
    return torch.full_like(scores, 0.5)


class SLMPerplexityScorer:
    """Per-character token importance from a small causal LM.

    Works on raw text rather than caller-supplied ids precisely so it can be
    paired with a generation model that tokenizes differently.
    """

    def __init__(self, model, tokenizer, window_size: int = 512, stride: Optional[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride if stride is not None else max(1, window_size // 2)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_adapter_dir: str,
        use_adapter: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> "SLMPerplexityScorer":
        """Load the scorer, optionally applying a LoRA adapter.

        `use_adapter=False` loads the adapter's *base* model only -- the
        ablation that isolates what fine-tuning contributed.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        is_adapter = False
        base_name = model_name_or_adapter_dir
        try:
            from peft import PeftConfig

            peft_config = PeftConfig.from_pretrained(model_name_or_adapter_dir)
            base_name = peft_config.base_model_name_or_path
            is_adapter = True
        except Exception:
            pass  # a plain HF model id / local model dir

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_adapter_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
        # Same deterministic resize as run_train_slm.py -- adapters are saved
        # without embedding layers, so the loader must reconstruct them the
        # same way the trainer did.
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
        from run_train_slm import resize_embeddings_if_needed
        resize_embeddings_if_needed(model, tokenizer)
        if is_adapter and use_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, model_name_or_adapter_dir)
        model = model.to(device).eval()
        model.config.use_cache = False
        return cls(model, tokenizer, **kwargs)

    @torch.no_grad()
    def _token_importance(self, ids: Sequence[int]) -> torch.Tensor:
        """importance(t_i) = -log P(t_i | context), over sliding windows.

        Same scheme as TinyModelScorer.compute_perplexity_scores: each window
        only claims its trailing (non-overlapping) positions, so tokens past
        the first window keep real left context.
        """
        n = len(ids)
        if n == 0:
            return torch.zeros(0)
        device = next(self.model.parameters()).device
        scores = torch.zeros(n)
        window = self.window_size
        stride = max(1, min(self.stride, window - 1)) if window > 1 else 1

        begin, prev_end = 0, 0
        while begin < max(n - 1, 1):
            end = min(begin + window, n)
            chunk = list(ids[begin:end])
            if len(chunk) < 2:
                break
            tensor = torch.tensor([chunk], device=device)
            logits = self.model(tensor).logits.float()
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            token_lp = log_probs.gather(-1, tensor[:, 1:].unsqueeze(-1)).squeeze(-1)
            importance = (-token_lp[0]).cpu()

            # Position 0 has no predecessor; approximate it with position 1.
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

    def _offsets(self, text: str):
        """(input_ids, char offsets). Falls back to incremental decoding when
        the tokenizer is slow (no native offset mapping)."""
        try:
            enc = self.tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=False,
                truncation=False,
            )
            return enc["input_ids"], enc["offset_mapping"]
        except (TypeError, NotImplementedError, ValueError):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            offsets, cursor = [], 0
            for tid in ids:
                piece = self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                if not piece:
                    offsets.append((cursor, cursor))
                    continue
                found = text.find(piece, cursor)
                if found < 0:  # normalization changed the surface form
                    offsets.append((cursor, min(cursor + len(piece), len(text))))
                    cursor = min(cursor + len(piece), len(text))
                else:
                    offsets.append((found, found + len(piece)))
                    cursor = found + len(piece)
            return ids, offsets

    def char_importance(self, text: str) -> torch.Tensor:
        """Per-character importance, NaN where no token covers the character."""
        char_scores = torch.full((len(text),), float("nan"))
        if not text:
            return char_scores
        ids, offsets = self._offsets(text)
        if not ids:
            return char_scores
        importance = self._token_importance(ids)
        for (start, end), value in zip(offsets, importance.tolist()):
            if end > start:
                char_scores[start:end] = value
        return char_scores


class SLMScorerCompressor(BaseCompressor):
    """LACC lightweight: S(t) = w_p*S_ppl + w_t*S_tone + w_m*S_morph.

    Identical blend to the no-model `combined` compressor except that the
    perplexity term comes from a real SLM instead of being a constant, which
    is exactly the difference being measured.
    """

    def __init__(
        self,
        tokenizer,
        model=None,
        config: Optional[CompressionConfig] = None,
        device: str = "cuda",
        scorer: Optional[SLMPerplexityScorer] = None,
        scorer_adapter_dir: Optional[str] = None,
        use_adapter: bool = True,
        weights: Optional[ScoreWeights] = None,
        name: str = "slm_scorer",
    ):
        super().__init__(tokenizer, model, config)
        self.device = device
        self._name = name
        self.weights = weights or ScoreWeights(perplexity=0.4, tone=0.3, morphology=0.3)
        self.tone_analyzer = get_tone_analyzer()
        self.morph_analyzer = get_morphology_analyzer()
        if scorer is None:
            if not scorer_adapter_dir:
                raise ValueError(
                    f"{name} needs a scorer: pass scorer_adapter_dir (a LoRA adapter "
                    "directory from run_train_slm.py, or a HuggingFace model id). "
                    "From the CLI: run_benchmark.py --scorer-adapter-dir trained_slm/final"
                )
            scorer = SLMPerplexityScorer.from_pretrained(
                scorer_adapter_dir, use_adapter=use_adapter, device=device,
            )
        self.scorer = scorer

    def get_name(self) -> str:
        return self._name

    def compress(self, input_ids: List[int], **kwargs) -> CompressionResult:
        start = time.time()
        self.validate_input(input_ids)
        n = len(input_ids)
        keep = self.config.keep_boundary_tokens
        target_len = max(int(n / max(self.config.target_ratio, 1e-6)), 2 * keep)

        # Rebuild the text and each generation-token's character span, so the
        # scorer's own tokenization can be mapped back onto these positions.
        pieces, spans, cursor = [], [], 0
        for tid in input_ids:
            piece = self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            pieces.append(piece)
            spans.append((cursor, cursor + len(piece)))
            cursor += len(piece)
        text = "".join(pieces)

        char_scores = self.scorer.char_importance(text)
        ppl_scores = torch.zeros(n)
        for i, (start_c, end_c) in enumerate(spans):
            if end_c > start_c:
                window = char_scores[start_c:end_c]
                valid = window[~torch.isnan(window)]
                if valid.numel():
                    ppl_scores[i] = valid.mean()
        # Whitespace-only / unmapped tokens stay 0; treat them as neutral
        # rather than maximally droppable.
        unmapped = ppl_scores == 0
        if unmapped.any() and (~unmapped).any():
            ppl_scores[unmapped] = ppl_scores[~unmapped].median()

        stripped = [p.replace("▁", " ").replace("Ġ", " ").strip() for p in pieces]
        tone_infos = self.tone_analyzer.analyze_tokens(stripped)
        tone_scores = torch.tensor([info.preservation_weight for info in tone_infos], dtype=torch.float)
        morph_scores = torch.tensor(
            [_CLASS_MULTIPLIER.get(info.word_class, 1.0)
             for info in self.morph_analyzer.classify_batch(stripped)],
            dtype=torch.float,
        )

        combined = (
            self.weights.perplexity * _normalize(ppl_scores)
            + self.weights.tone * _normalize(tone_scores)
            + self.weights.morphology * _normalize(morph_scores)
        )

        mid_start, mid_end = keep, max(keep, n - keep)
        mid_budget = max(0, target_len - 2 * keep)
        if mid_budget > 0 and mid_start < mid_end:
            mid_scores = combined[mid_start:mid_end]
            mid_budget = min(mid_budget, mid_scores.numel())
            top = sorted(torch.topk(mid_scores, mid_budget).indices.tolist())
            retained = list(range(keep)) + [mid_start + i for i in top] + list(range(mid_end, n))
        else:
            retained = list(range(n))

        compressed = [input_ids[i] for i in retained]
        metadata: Dict[str, Any] = {
            "weights": self.weights.to_dict(),
            "scorer_uses_adapter": self._name != "slm_scorer_base",
            "tone_preservation_rate": compute_tone_preservation_rate(tone_infos, retained),
            "mean_ppl_score": float(ppl_scores.mean()),
        }
        return self._build_result(compressed, n, (time.time() - start) * 1000, metadata)
