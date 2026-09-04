"""LACC full/INT4 tier: the *trained tone probe* as the live tone signal.

This is the training->inference bridge the paper (Sect. 3.4) describes but that
nothing in the pipeline previously closed. `run_train_slm.py` trains a
Vietnamese SLM (`chronopt-research/vietnamese-gpt2-base`) with LoRA plus the
auxiliary Phonological Consistency Loss, whose 7-way tone classifier is saved
as `trained_slm/tone_probe.pt`. At inference the *same* classifier is reused as
a tone probe:

    S_tone-model(t_i) = 1.0 + max_k softmax(MLP(h_i))_k          (Eq. 3, paper)

where h_i is the SLM's last-layer hidden state at token i. Tokens the model is
tonally confident about are scored as more important to keep.

Why a NEW compressor instead of `ToneAwareCompressor(use_model_tone=True)`:
that path runs the probe on the *generation* model's hidden states (e.g. Qwen
7B, hidden_size 3584), but the probe was trained on the SLM's hidden states
(GPT-2, hidden_size 768). The dimensions do not even match, and if they did the
signal would be meaningless -- a probe read off a model it never saw. The probe
only means anything on the model it was trained with, so this compressor loads
that SLM, and -- exactly like `SLMScorerCompressor` -- maps the SLM-token
signals back onto the generation model's tokens through character offsets,
because the two models tokenize differently.

Registry entries (see compressors/__init__.py):
  - `slm_tone_probe`       -- tone signal from the trained probe (PROPOSED)
  - `slm_tone_probe_rule`  -- identical pipeline, tone from the dictionary
                              heuristic instead of the probe (ABLATION). This
                              is the controlled A/B that isolates what the
                              trained probe adds: same SLM, same perplexity
                              signal, same morphology signal, same selection --
                              only the tone term differs.
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..morphology.merge_policy import WordClass, get_morphology_analyzer
from ..tone_aware.vietnamese_tones import (
    compute_tone_preservation_rate,
    get_tone_analyzer,
)
from .base import BaseCompressor, CompressionConfig, CompressionResult
from .external_scorer import ScoreWeights

# Same multipliers as the no-model / slm_scorer compressors, so the morphology
# signal is identical across the whole LACC family and only the tone/ppl terms
# distinguish the tiers.
_CLASS_MULTIPLIER = {
    WordClass.FUNC: 0.40,
    WordClass.CONTENT: 1.20,
    WordClass.REDUP: 0.60,
    WordClass.COMPOUND: 1.50,
    WordClass.SINO: 1.50,
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


class SLMToneProbeScorer:
    """Runs the fine-tuned SLM once per text to produce, per SLM-token:

      * a perplexity signal   -log P(t_i | t_<i)               (LLMLingua-style)
      * a model tone signal    1 + max softmax(probe(h_i))      (Eq. 3, paper)

    Both are computed in a single sliding-window forward pass. Works on raw
    text (not caller ids) precisely so it can be paired with a generation model
    that tokenizes differently.
    """

    def __init__(
        self,
        model,
        tokenizer,
        probe,
        window_size: int = 512,
        stride: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.probe = probe  # PhonologicalConsistencyLoss (or None -> no tone)
        self.window_size = window_size
        self.stride = stride if stride is not None else max(1, window_size // 2)

    @classmethod
    def from_pretrained(
        cls,
        adapter_dir: str,
        tone_probe_path: str,
        use_adapter: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        load_4bit: bool = False,
        **kwargs,
    ) -> "SLMToneProbeScorer":
        """Load the SLM (base + optional LoRA adapter) and its trained tone probe.

        The probe was trained jointly with the LoRA adapter, so the meaningful
        configuration is `use_adapter=True`. `use_adapter=False` is offered only
        as a diagnostic (probe read off the un-adapted base) and the caller is
        responsible for interpreting it -- see docs/tone_probe_bridge.md.
        """
        import os
        import sys as _sys

        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..tone_aware import PhonologicalConsistencyLoss

        is_adapter = False
        base_name = adapter_dir
        try:
            from peft import PeftConfig

            peft_config = PeftConfig.from_pretrained(adapter_dir)
            base_name = peft_config.base_model_name_or_path
            is_adapter = True
        except Exception:
            pass  # a plain HF model id / local model dir

        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # A few-B base (Qwen3-4B) scorer loads in 4-bit NF4 or bfloat16 so it
        # fits alongside the generation model; a small SLM stays in float32.
        if load_4bit:
            from transformers import BitsAndBytesConfig

            dtype = torch.bfloat16  # compute dtype for the probe/hidden states
            model = AutoModelForCausalLM.from_pretrained(
                base_name,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
                device_map={"": 0})
        else:
            model = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)

        # Same deterministic resize as run_train_slm.py -- adapters are saved
        # without embedding layers (save_embedding_layers=False), so the loader
        # must reconstruct the extra zeroed rows exactly the way the trainer did.
        _sys.path.insert(
            0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        from run_train_slm import resize_embeddings_if_needed

        resize_embeddings_if_needed(model, tokenizer)
        if is_adapter and use_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
        # A 4-bit model is already placed by device_map and cannot be .to()'d.
        model = (model if load_4bit else model.to(device)).eval()
        model.config.use_cache = False

        # If a training-time meta file sits next to the probe, use it to catch a
        # probe paired with the wrong base model early (a clearer failure than a
        # silent dim match on two same-width but different models).
        meta_path = os.path.join(os.path.dirname(os.path.abspath(tone_probe_path)),
                                 "tone_probe_meta.json")
        if os.path.exists(meta_path):
            import json as _json

            with open(meta_path, encoding="utf-8") as _f:
                meta = _json.load(_f)
            if meta.get("base_model") and base_name and meta["base_model"] != base_name:
                raise ValueError(
                    f"tone_probe_meta.json says this probe was trained on "
                    f"{meta['base_model']!r}, but --scorer-adapter-dir resolves to "
                    f"base model {base_name!r}. Pair the probe with its own adapter."
                )

        probe = PhonologicalConsistencyLoss(
            hidden_dim=model.config.hidden_size, lambda_tone=0.0
        )
        state = torch.load(tone_probe_path, map_location="cpu", weights_only=True)
        probe_dim = state["tone_classifier.0.weight"].shape[1]
        if probe_dim != model.config.hidden_size:
            raise ValueError(
                f"Tone probe hidden dim ({probe_dim}) does not match the SLM's "
                f"hidden size ({model.config.hidden_size}). This probe was trained "
                f"on a different base model -- pass the tone_probe.pt that belongs "
                f"to {base_name}."
            )
        probe.load_state_dict(state)
        probe = probe.to(device=device, dtype=dtype).eval()
        return cls(model, tokenizer, probe, **kwargs)

    @torch.no_grad()
    def _token_signals(self, ids: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-token (perplexity, model-tone) over sliding windows.

        Each window only claims its trailing (non-overlapping) positions, so a
        token past the first window keeps real left context for both its
        perplexity and its hidden-state representation -- the same accounting
        SLMPerplexityScorer / TinyModelScorer use for perplexity, extended to
        cover the tone probe's hidden states in the same pass.
        """
        n = len(ids)
        ppl = torch.zeros(n)
        tone = torch.ones(n)  # neutral 1.0 where the probe is disabled/uncovered
        if n == 0:
            return ppl, tone

        device = next(self.model.parameters()).device
        window = self.window_size
        stride = max(1, min(self.stride, window - 1)) if window > 1 else 1

        begin, prev_end = 0, 0
        while begin < max(n - 1, 1):
            end = min(begin + window, n)
            chunk = list(ids[begin:end])
            if len(chunk) < 2:
                break
            tensor = torch.tensor([chunk], device=device)
            out = self.model(tensor, output_hidden_states=self.probe is not None)
            logits = out.logits.float()

            # --- perplexity: importance[k] scores predicting abs pos begin+k+1
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            token_lp = log_probs.gather(-1, tensor[:, 1:].unsqueeze(-1)).squeeze(-1)
            importance = (-token_lp[0]).cpu()
            if begin == 0 and importance.numel() > 0:
                ppl[0] = importance[0]  # pos 0 has no predecessor; approximate
            ppl_start = 1 if begin == 0 else prev_end
            for k in range(importance.numel()):
                pos = begin + k + 1
                if ppl_start <= pos < n:
                    ppl[pos] = importance[k]

            # --- model tone: probe(h_j) at hidden local index j = abs pos begin+j
            if self.probe is not None:
                h = out.hidden_states[-1].to(self.probe.tone_classifier[0].weight.dtype)
                tone_imp = self.probe.score_importance(h)[0].cpu()  # [L]
                tone_start = 0 if begin == 0 else prev_end
                for j in range(tone_imp.numel()):
                    pos = begin + j
                    if tone_start <= pos < n:
                        tone[pos] = tone_imp[j]

            prev_end = end
            if end >= n:
                break
            begin += stride
        return ppl, tone

    def _offsets(self, text: str):
        """(input_ids, char offsets) for the SLM's own tokenization.

        Falls back to incremental decoding when the tokenizer has no native
        offset mapping -- identical strategy to SLMPerplexityScorer._offsets.
        """
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
                if found < 0:
                    offsets.append((cursor, min(cursor + len(piece), len(text))))
                    cursor = min(cursor + len(piece), len(text))
                else:
                    offsets.append((found, found + len(piece)))
                    cursor = found + len(piece)
            return ids, offsets

    def char_signals(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-character (perplexity, model-tone). NaN where no SLM token covers
        the character, so the caller can distinguish 'uncovered' from a real 0.
        """
        ppl_char = torch.full((len(text),), float("nan"))
        tone_char = torch.full((len(text),), float("nan"))
        if not text:
            return ppl_char, tone_char
        ids, offsets = self._offsets(text)
        if not ids:
            return ppl_char, tone_char
        ppl, tone = self._token_signals(ids)
        for (start, end), pv, tv in zip(offsets, ppl.tolist(), tone.tolist()):
            if end > start:
                ppl_char[start:end] = pv
                tone_char[start:end] = tv
        return ppl_char, tone_char


class SLMToneProbeCompressor(BaseCompressor):
    """LACC full/INT4 tier compressor driven by the trained tone probe.

        S(t) = w_ppl * S_ppl + w_tone * S_tone + w_morph * S_morph

    identical in structure to `SLMScorerCompressor`, with one difference that is
    the whole point: S_tone comes from the trained model probe
    (`tone_source='model'`) rather than the dictionary heuristic. Setting
    `tone_source='rule'` recovers the heuristic tone term while keeping every
    other signal fixed -- the controlled ablation for measuring the probe's
    contribution end-to-end.
    """

    def __init__(
        self,
        tokenizer,
        model=None,
        config: Optional[CompressionConfig] = None,
        device: str = "cuda",
        scorer: Optional[SLMToneProbeScorer] = None,
        scorer_adapter_dir: Optional[str] = None,
        tone_probe_path: Optional[str] = None,
        use_adapter: bool = True,
        tone_source: str = "model",
        weights: Optional[ScoreWeights] = None,
        name: str = "slm_tone_probe",
    ):
        super().__init__(tokenizer, model, config)
        self.device = device
        self._name = name
        if tone_source not in ("model", "rule"):
            raise ValueError(f"tone_source must be 'model' or 'rule'; got {tone_source!r}")
        self.tone_source = tone_source
        self.weights = weights or ScoreWeights(perplexity=0.4, tone=0.3, morphology=0.3)
        self.tone_analyzer = get_tone_analyzer()
        self.morph_analyzer = get_morphology_analyzer()

        if scorer is None:
            if not scorer_adapter_dir or not tone_probe_path:
                raise ValueError(
                    f"{name} needs the trained SLM AND its tone probe: pass "
                    "scorer_adapter_dir (a LoRA adapter dir from run_train_slm.py, "
                    "e.g. trained_slm/final) and tone_probe_path (e.g. "
                    "trained_slm/tone_probe.pt). From the CLI: run_benchmark.py "
                    "--scorer-adapter-dir trained_slm/final --tone-probe-path "
                    "trained_slm/tone_probe.pt"
                )
            scorer = SLMToneProbeScorer.from_pretrained(
                scorer_adapter_dir, tone_probe_path,
                use_adapter=use_adapter, device=device,
            )
        self.scorer = scorer

    def get_name(self) -> str:
        return self._name

    def _token_spans(self, input_ids: List[int]) -> Tuple[str, List[Tuple[int, int]], List[str]]:
        """Rebuild the text and each generation-token's character span."""
        pieces, spans, cursor = [], [], 0
        for tid in input_ids:
            piece = self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            pieces.append(piece)
            spans.append((cursor, cursor + len(piece)))
            cursor += len(piece)
        return "".join(pieces), spans, pieces

    def _pool_char_to_token(
        self, char_scores: torch.Tensor, spans: List[Tuple[int, int]], n: int, fill_neutral: bool
    ) -> torch.Tensor:
        """Average per-character scores onto each generation token's span."""
        out = torch.zeros(n)
        for i, (start_c, end_c) in enumerate(spans):
            if end_c > start_c:
                window = char_scores[start_c:end_c]
                valid = window[~torch.isnan(window)]
                if valid.numel():
                    out[i] = valid.mean()
        if fill_neutral:
            # Whitespace-only / unmapped tokens stay 0; treat them as neutral
            # (the median) rather than maximally droppable.
            unmapped = out == 0
            if unmapped.any() and (~unmapped).any():
                out[unmapped] = out[~unmapped].median()
        return out

    def compress(self, input_ids: List[int], **kwargs) -> CompressionResult:
        start = time.time()
        self.validate_input(input_ids)
        n = len(input_ids)

        text, spans, pieces = self._token_spans(input_ids)
        ppl_char, tone_char = self.scorer.char_signals(text)

        ppl_scores = self._pool_char_to_token(ppl_char, spans, n, fill_neutral=True)

        stripped = [p.replace("▁", " ").replace("Ġ", " ").strip() for p in pieces]
        tone_infos = self.tone_analyzer.analyze_tokens(stripped)

        if self.tone_source == "model":
            # Model probe supplies the tone term. Uncovered tokens fall back to
            # the rule weight so a token the SLM never saw is not treated as
            # tone-neutral by accident.
            tone_scores = self._pool_char_to_token(tone_char, spans, n, fill_neutral=False)
            uncovered = tone_scores == 0
            if uncovered.any():
                rule = torch.tensor(
                    [info.preservation_weight for info in tone_infos], dtype=torch.float
                )
                tone_scores[uncovered] = rule[uncovered]
        else:  # 'rule' -- the controlled ablation
            tone_scores = torch.tensor(
                [info.preservation_weight for info in tone_infos], dtype=torch.float
            )

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

        # Shared boundary-aware selector: keep the k boundary tokens each side
        # plus the top-scoring middle, in original order (see BaseCompressor).
        retained = self.select_with_boundary(combined.tolist(), n)
        compressed = [input_ids[i] for i in retained]

        metadata: Dict[str, Any] = {
            "weights": self.weights.to_dict(),
            "tone_source": self.tone_source,
            "scorer_uses_adapter": self._name != "slm_tone_probe_base",
            "tone_preservation_rate": compute_tone_preservation_rate(tone_infos, set(retained)),
            "mean_ppl_score": float(ppl_scores.mean()),
            "mean_model_tone_score": float(tone_scores.mean()),
        }
        return self._build_result(compressed, n, (time.time() - start) * 1000, metadata)
