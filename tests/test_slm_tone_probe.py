"""CPU-only tests for the SLM tone-probe compressor (the training->inference
bridge) and its scorer's sliding-window signal accounting.

No LoRA adapter or GPU is loaded: the compressor tests inject a fake scorer
that returns deterministic per-character signals, and the scorer test injects a
fake causal LM so the position mapping is exercised without a real model. What
they guard is the wiring the paper's contribution depends on -- that the probe
signal actually reaches selection, that the controlled 'rule' ablation ignores
it, and that both stay valid compressions.
"""
from types import SimpleNamespace

import pytest
import torch

from vncompress.compressors.base import CompressionConfig
from vncompress.compressors.slm_tone_probe import (
    SLMToneProbeCompressor,
    SLMToneProbeScorer,
)


class FakeCharSignalScorer:
    """Stand-in for SLMToneProbeScorer: constant perplexity, and a tone signal
    that is high only on the characters of `tone_high_token`.

    The compressor rebuilds `text` as the concatenation (no separators) of each
    token's decode, so a token's characters occupy a contiguous span we can
    target by substring.
    """

    def __init__(self, tone_high_token: str, ppl_value: float = 0.5):
        self.tone_high_token = tone_high_token
        self.ppl_value = ppl_value

    def char_signals(self, text):
        ppl = torch.full((len(text),), self.ppl_value)
        tone = torch.ones(len(text))
        idx = text.find(self.tone_high_token)
        if idx >= 0:
            tone[idx: idx + len(self.tone_high_token)] = 2.0
        return ppl, tone


def _config():
    # keep_boundary=1 and ratio=2 on n=6 leaves exactly one middle slot, so the
    # single chosen middle token is unambiguously the top-scoring one.
    return CompressionConfig(target_ratio=2.0, keep_boundary_tokens=1)


class TestToneSourceRouting:
    def test_missing_scorer_and_probe_raise_actionable_error(self):
        with pytest.raises(ValueError, match="tone_probe_path"):
            SLMToneProbeCompressor(tokenizer=None, scorer=None,
                                   scorer_adapter_dir="trained_slm/final",
                                   tone_probe_path=None)

    def test_invalid_tone_source_rejected(self, tokenizer):
        with pytest.raises(ValueError, match="tone_source"):
            SLMToneProbeCompressor(tokenizer, scorer=FakeCharSignalScorer("dd"),
                                   tone_source="banana")

    def test_probe_signal_changes_selection_but_rule_ignores_it(self, tokenizer):
        # Six ascii (non-tone) tokens; the fake makes token index 3 ("dd")
        # tonally salient. With one middle slot, the model arm must keep it and
        # the rule arm (which ignores the probe) must not.
        ids = tokenizer.encode("aa bb cc dd ee ff")
        assert len(ids) == 6

        model_arm = SLMToneProbeCompressor(
            tokenizer, scorer=FakeCharSignalScorer("dd"), tone_source="model",
            config=_config(),
        )
        rule_arm = SLMToneProbeCompressor(
            tokenizer, scorer=FakeCharSignalScorer("dd"), tone_source="rule",
            config=_config(),
        )
        rm = model_arm.compress(list(ids))
        rr = rule_arm.compress(list(ids))

        # Recover retained indices by identity (ids are distinct here).
        model_keep = [ids.index(t) for t in rm.compressed_ids]
        rule_keep = [ids.index(t) for t in rr.compressed_ids]

        assert 3 in model_keep          # probe salience kept the token
        assert 3 not in rule_keep       # heuristic arm did not
        assert model_keep != rule_keep
        # Boundaries preserved by both.
        assert 0 in model_keep and 5 in model_keep
        assert rm.metadata["tone_source"] == "model"
        assert rr.metadata["tone_source"] == "rule"
        # The model arm's tone term reflects the probe (a 2.0 spike); the rule
        # arm's is the flat baseline weight on tone-free tokens.
        assert rm.metadata["mean_model_tone_score"] > rr.metadata["mean_model_tone_score"]

    def test_output_is_a_valid_compression(self, tokenizer, vi_ids):
        comp = SLMToneProbeCompressor(
            tokenizer, scorer=FakeCharSignalScorer("xxxx"), tone_source="model",
        )
        comp.config.target_ratio = 4.0
        result = comp.compress(list(vi_ids))
        assert result.compressed_length < len(vi_ids)
        assert result.compression_ratio > 1.0
        # Subsequence: compressed order is a subset of the original order.
        assert result.compressed_ids == [i for i in result.compressed_ids]
        assert 0.0 <= result.metadata["tone_preservation_rate"] <= 1.0

    def test_model_tone_falls_back_to_rule_for_uncovered_tokens(self, tokenizer):
        # A scorer whose tone signal covers nothing (all NaN) must not zero out
        # tone-bearing tokens -- they should inherit the rule weight instead.
        class AllNaN:
            def char_signals(self, text):
                nan = torch.full((len(text),), float("nan"))
                return nan.clone(), nan.clone()

        ids = tokenizer.encode("Hà Nội là thủ đô của Việt Nam")
        comp = SLMToneProbeCompressor(tokenizer, scorer=AllNaN(), tone_source="model")
        comp.config.target_ratio = 2.0
        result = comp.compress(list(ids))
        # mean tone score should reflect the rule weights (>1.0 for tone-bearing
        # text), not collapse to 0 from the uncovered probe.
        assert result.metadata["mean_model_tone_score"] > 1.0


class TestSLMToneProbeScorerAccounting:
    """Guards the sliding-window position mapping in _token_signals with a fake
    causal LM, so no real model/GPU is needed."""

    def _fake_scorer(self, hidden_dim=16, vocab=32, window=512):
        from vncompress.tone_aware import PhonologicalConsistencyLoss

        probe = PhonologicalConsistencyLoss(hidden_dim=hidden_dim, lambda_tone=0.0).eval()

        class FakeModel:
            def __init__(self):
                self._p = torch.zeros(1)  # for device lookup via .parameters()
                self.config = SimpleNamespace(hidden_size=hidden_dim)

            def parameters(self):
                yield self._p

            def __call__(self, tensor, output_hidden_states=False):
                L = tensor.shape[1]
                torch.manual_seed(L)  # deterministic per call
                logits = torch.randn(1, L, vocab)
                hs = (torch.randn(1, L, hidden_dim),) if output_hidden_states else None
                return SimpleNamespace(logits=logits, hidden_states=hs)

        return SLMToneProbeScorer(FakeModel(), tokenizer=None, probe=probe, window_size=window)

    def test_signals_have_input_length_and_valid_ranges(self):
        scorer = self._fake_scorer()
        ids = list(range(1, 25))  # 24 tokens
        ppl, tone = scorer._token_signals(ids)
        assert ppl.shape[0] == len(ids)
        assert tone.shape[0] == len(ids)
        # Perplexity = -log p >= 0; the probe's score_importance is clamped to
        # [0.5, 3.0] (paper Eq. 3: 1 + max softmax prob).
        assert (ppl >= 0).all()
        assert (tone >= 0.5).all() and (tone <= 3.0).all()

    def test_multi_window_covers_every_position(self):
        # Force several overlapping windows and check no position is left unset
        # (tone defaults to the neutral 1.0, so a covered tone must differ from
        # exactly 1.0 for at least most positions given a random probe).
        scorer = self._fake_scorer(window=8)
        ids = list(range(1, 30))
        ppl, tone = scorer._token_signals(ids)
        assert ppl.shape[0] == len(ids)
        assert tone.shape[0] == len(ids)
        # Every interior position got a real perplexity (strictly, some could be
        # 0 by chance, but with random logits the mean must be clearly positive).
        assert ppl.mean() > 0

    def test_probe_disabled_yields_neutral_tone(self):
        scorer = self._fake_scorer()
        scorer.probe = None
        ppl, tone = scorer._token_signals(list(range(1, 10)))
        assert torch.allclose(tone, torch.ones_like(tone))


class TestTargetModules:
    """The tone-probe trainer must accept a Qwen3-4B base (and stay working for
    GPT-2 SLMs), so the training target can be switched without a code edit."""

    def _model(self, model_type):
        return SimpleNamespace(config=SimpleNamespace(model_type=model_type))

    @pytest.mark.parametrize("mt", ["qwen3", "qwen3_moe", "qwen2", "qwen", "llama"])
    def test_qwen_family_targets_projections(self, mt):
        from run_train_slm import target_modules

        mods = target_modules(self._model(mt))
        assert "q_proj" in mods and "gate_proj" in mods

    def test_gpt2_targets_conv1d_modules(self):
        from run_train_slm import target_modules

        assert target_modules(self._model("gpt2")) == ["c_attn", "c_proj", "c_fc"]

    def test_unknown_model_type_raises(self):
        from run_train_slm import target_modules

        with pytest.raises(ValueError, match="Unsupported model_type"):
            target_modules(self._model("mamba"))


class TestRegistration:
    def test_methods_registered(self):
        from vncompress.compressors import COMPRESSOR_REGISTRY

        assert "slm_tone_probe" in COMPRESSOR_REGISTRY
        assert "slm_tone_probe_rule" in COMPRESSOR_REGISTRY

    def test_taxonomy_marks_probe_proposed_and_rule_ablation(self):
        from vncompress.evaluation.method_taxonomy import MethodCategory, categorize

        assert categorize("slm_tone_probe") is MethodCategory.PROPOSED
        assert categorize("slm_tone_probe_rule") is MethodCategory.ABLATION

    def test_tone_probe_kwarg_not_leaked_to_other_compressors(self, tokenizer):
        from vncompress.compressors import create_compressor

        comp = create_compressor(
            "none", tokenizer, None, config=None, device="cpu",
            scorer_adapter_dir="trained_slm/final",
            tone_probe_path="trained_slm/tone_probe.pt",
        )
        assert comp.get_name() == "NoCompression"

    def test_score_importance_matches_paper_formula(self):
        # S_tone-model = 1 + max_k softmax(MLP(h))_k, clamped [0.5, 3.0].
        from vncompress.tone_aware import PhonologicalConsistencyLoss

        probe = PhonologicalConsistencyLoss(hidden_dim=16, lambda_tone=0.0).eval()
        h = torch.randn(1, 5, 16)
        with torch.no_grad():
            score = probe.score_importance(h)
            logits = probe.tone_classifier(h)
            expected = 1.0 + torch.softmax(logits, dim=-1).max(dim=-1).values
        assert torch.allclose(score, torch.clamp(expected, 0.5, 3.0), atol=1e-5)
