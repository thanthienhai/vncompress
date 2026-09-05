"""Tests for vncompress/compression.py: CompressionResult/Config,
BaseCompressor, NoCompressor, RandomCompressor, LACCCompressor (the no-model
and fake-scorer paths), SemanticQualityGate, compute_query_relevance_weights,
and the METHODS registry / create_compressor() factory.

LACCCompressor is exercised with model=None (rule-based tone/morphology
only, use_perplexity=False) or a fake scorer (no torch model download, no
GPU) -- the same CPU-only, no-model tier the `no_model` hardware tier in
docs uses.
"""
import json

import pytest
import torch

from vncompress.compression import (
    METHODS,
    BaseCompressor,
    CompressionConfig,
    CompressionResult,
    LACCCompressor,
    LACCScorer,
    NoCompressor,
    RandomCompressor,
    SemanticQualityGate,
    compute_query_relevance_weights,
    create_compressor,
)


def _is_subsequence(sub, full):
    """True if `sub` occurs as an (order-preserving) subsequence of `full`."""
    it = iter(full)
    return all(item in it for item in sub)


def _lacc_no_model(tokenizer, **kwargs):
    """The 0-VRAM LACC tier: no model, no scorer -- pure rule-based signals."""
    kwargs.setdefault("use_perplexity", False)
    return LACCCompressor(tokenizer, model=None, device="cpu", **kwargs)


# ============================================================================
# CompressionResult / CompressionConfig / BaseCompressor
# ============================================================================


def test_compression_config_defaults():
    config = CompressionConfig()
    assert config.target_ratio == 4.0
    assert config.keep_special_tokens is True
    assert config.keep_boundary_tokens == 2
    assert config.min_compressed_length == 1
    assert config.language == "vi"


def test_compression_result_is_a_dataclass_with_expected_fields():
    result = CompressionResult(
        compressed_ids=[1, 2, 3], compressed_text="a b c",
        compression_ratio=2.0, token_savings_pct=50.0,
        original_length=6, compressed_length=3,
        method_name="test", processing_time_ms=1.0,
    )
    assert result.compressed_ids == [1, 2, 3]
    assert result.metadata == {}


def test_base_compressor_cannot_be_instantiated_directly(tokenizer):
    with pytest.raises(TypeError):
        BaseCompressor(tokenizer)


class TestNoCompressor:
    def test_returns_input_unchanged(self, tokenizer, vi_ids):
        result = NoCompressor(tokenizer).compress(vi_ids)
        assert result.compressed_ids == list(vi_ids)
        assert result.compression_ratio == 1.0
        assert result.token_savings_pct == 0.0

    def test_get_name(self, tokenizer):
        assert NoCompressor(tokenizer).get_name() == "NoCompression"

    def test_empty_input_raises(self, tokenizer):
        with pytest.raises(ValueError):
            NoCompressor(tokenizer).compress([])

    def test_compress_text_roundtrips_through_tokenizer(self, tokenizer):
        result = NoCompressor(tokenizer).compress_text("xin chào các bạn")
        assert result.compressed_text == "xin chào các bạn"

    def test_repr_includes_method_name(self, tokenizer):
        assert "NoCompression" in repr(NoCompressor(tokenizer))


class TestRandomCompressor:
    def test_get_name(self, tokenizer):
        assert RandomCompressor(tokenizer).get_name() == "RandomBaseline"

    def test_empty_input_raises(self, tokenizer):
        with pytest.raises(ValueError):
            RandomCompressor(tokenizer).compress([])

    def test_boundary_tokens_are_always_kept(self, tokenizer, vi_ids):
        config = CompressionConfig(target_ratio=4.0, keep_boundary_tokens=2)
        result = RandomCompressor(tokenizer, config=config).compress(vi_ids)
        assert result.compressed_ids[:2] == vi_ids[:2]
        assert result.compressed_ids[-2:] == vi_ids[-2:]

    def test_token_order_is_preserved(self, tokenizer, vi_ids):
        result = RandomCompressor(tokenizer).compress(vi_ids)
        assert _is_subsequence(result.compressed_ids, vi_ids)

    def test_compression_ratio_is_approximately_respected(self, tokenizer, vi_ids):
        config = CompressionConfig(target_ratio=2.0)
        result = RandomCompressor(tokenizer, config=config).compress(vi_ids)
        assert result.compressed_length < len(vi_ids)
        assert result.compressed_length == pytest.approx(len(vi_ids) / 2, abs=3)

    def test_is_reproducible_across_instances(self, tokenizer, vi_ids):
        # Own RNG, not the global `random` module -- an unrelated
        # random.seed() call elsewhere must not change this baseline.
        outs = []
        for _ in range(2):
            comp = RandomCompressor(tokenizer, config=CompressionConfig(target_ratio=4.0))
            outs.append(comp.compress(list(vi_ids)).compressed_ids)
        assert outs[0] == outs[1]

    @pytest.mark.parametrize("ratio", [1.0, 1.5, 8.0, 100.0])
    def test_extreme_target_ratios_do_not_crash(self, tokenizer, vi_ids, ratio):
        config = CompressionConfig(target_ratio=ratio)
        result = RandomCompressor(tokenizer, config=config).compress(vi_ids)
        assert 0 < result.compressed_length <= len(vi_ids)


class TestSelectWithBoundary:
    def test_output_never_longer_than_input(self, tokenizer, vi_ids):
        # Was: unclamped `range(n-k, n)` produced negative indices that
        # Python wrapped to the end, duplicating and reordering tokens.
        for n_small, k in ((1, 2), (2, 2), (3, 5)):
            comp = RandomCompressor(tokenizer, config=CompressionConfig(target_ratio=4.0, keep_boundary_tokens=k))
            sub = list(vi_ids[:n_small])
            out = comp.compress(sub).compressed_ids
            assert len(out) <= len(sub)
            indices = comp.select_with_boundary([0.0] * n_small, n_small)
            assert indices == sorted(set(indices))
            assert all(0 <= i < n_small for i in indices)

    def test_high_ratio_still_compresses(self, tokenizer, vi_ids):
        # Was: mid_budget==0 took a branch returning every index, so asking
        # for MORE compression returned NONE (CR 1.0 at ratio 16+).
        for ratio in (8.0, 16.0, 32.0):
            comp = _lacc_no_model(tokenizer, config=CompressionConfig(target_ratio=ratio))
            result = comp.compress(list(vi_ids))
            assert result.compressed_length < len(vi_ids)
            assert result.compression_ratio > 1.0


# ============================================================================
# LACCCompressor -- no-model (0 VRAM) tier
# ============================================================================

SIGNAL_COMBOS = [
    dict(use_tone=True, use_morphology=True),
    dict(use_tone=True, use_morphology=False),
    dict(use_tone=False, use_morphology=True),
]


@pytest.mark.parametrize("kwargs", SIGNAL_COMBOS)
def test_no_model_does_not_crash(kwargs, tokenizer, vi_ids):
    comp = _lacc_no_model(tokenizer, **kwargs)
    result = comp.compress(vi_ids)
    assert 0 < result.compressed_length <= result.original_length


@pytest.mark.parametrize("kwargs", SIGNAL_COMBOS)
def test_no_model_token_order_is_preserved(kwargs, tokenizer, vi_ids):
    comp = _lacc_no_model(tokenizer, **kwargs)
    result = comp.compress(vi_ids)
    assert _is_subsequence(result.compressed_ids, vi_ids)


@pytest.mark.parametrize("kwargs", SIGNAL_COMBOS)
def test_no_model_boundary_tokens_kept(kwargs, tokenizer, vi_ids):
    config = CompressionConfig(target_ratio=4.0, keep_boundary_tokens=2)
    comp = _lacc_no_model(tokenizer, config=config, **kwargs)
    result = comp.compress(vi_ids)
    assert result.compressed_ids[:2] == vi_ids[:2]
    assert result.compressed_ids[-2:] == vi_ids[-2:]


@pytest.mark.parametrize("kwargs", SIGNAL_COMBOS)
def test_no_model_very_short_sequence_does_not_crash(kwargs, tokenizer):
    comp = _lacc_no_model(tokenizer, **kwargs)
    ids = tokenizer.encode("một hai")
    result = comp.compress(ids)
    assert result.compressed_length <= len(ids)


def test_disabling_every_signal_raises():
    class _Tok:
        def encode(self, *a, **k):
            return []

        def decode(self, *a, **k):
            return ""

    with pytest.raises(ValueError):
        LACCCompressor(_Tok(), model=None, use_perplexity=False, use_tone=False, use_morphology=False)


def test_get_name_reflects_enabled_signals(tokenizer):
    name = _lacc_no_model(tokenizer, use_tone=True, use_morphology=False).get_name()
    assert "tone" in name and "morph" not in name


def test_tone_preservation_rate_reported_when_tone_enabled(tokenizer, vi_ids):
    result = _lacc_no_model(tokenizer, use_tone=True).compress(vi_ids)
    assert 0.0 <= result.metadata["tone_preservation_rate"] <= 1.0


def test_tone_preservation_rate_absent_when_tone_disabled(tokenizer, vi_ids):
    result = _lacc_no_model(tokenizer, use_tone=False, use_morphology=True).compress(vi_ids)
    assert result.metadata["tone_source"] is None


def test_query_boost_keeps_query_relevant_token(tokenizer):
    # "vào" is a low-priority function word that loses out to content words
    # under normal scoring, so it's a clean signal that the query boost is
    # actually reaching token selection: present only once boosted.
    text = "chúng tôi đi học ở trường đại học rất xa vào buổi sáng hôm nay trời đẹp"
    ids = tokenizer.encode(text)
    config = CompressionConfig(target_ratio=4.0, keep_boundary_tokens=1)
    comp = _lacc_no_model(tokenizer, config=config)

    vao_id = tokenizer.encode("vào")[0]
    without_query = comp.compress(list(ids))
    with_query = comp.compress(list(ids), query="vào", query_boost=5.0)

    assert vao_id not in without_query.compressed_ids
    assert vao_id in with_query.compressed_ids


def test_non_vietnamese_text_does_not_crash(tokenizer, en_text):
    ids = tokenizer.encode(en_text)
    result = _lacc_no_model(tokenizer).compress(ids)
    assert result.compressed_length <= len(ids)


class TestVietnameseLinguisticsRegressions:
    """Guards specific to how LACCCompressor consumes linguistics.py."""

    @pytest.mark.parametrize("word", ["vàng", "tình", "năng", "thái", "linh"])
    def test_standalone_second_syllables_dont_get_merged_away(self, word, tokenizer):
        # Was: classified REDUP purely for being some reduplication's 2nd
        # syllable, and compressed away even as a standalone content word.
        text = f"chúng tôi có rất nhiều {word} ở đây hôm nay và ngày mai"
        ids = tokenizer.encode(text)
        target_id = tokenizer.encode(word)[0]
        comp = _lacc_no_model(tokenizer, config=CompressionConfig(target_ratio=2.0))
        result = comp.compress(list(ids))
        assert target_id in result.compressed_ids


# ============================================================================
# LACCCompressor -- fake external scorer (LACCScorer wiring, no GPU)
# ============================================================================


class FakeCharSignalScorer(LACCScorer):
    """Stand-in for LACCScorer: constant perplexity, and a tone signal that
    is high only on the characters of `tone_high_token`. The compressor
    rebuilds `text` as the concatenation (no separators) of each token's
    decode, so a token's characters occupy a contiguous span targetable by
    substring."""

    def __init__(self, tone_high_token: str, ppl_value: float = 0.5):
        self.tone_high_token = tone_high_token
        self.ppl_value = ppl_value
        self.tone_probe = object()  # truthy: signals "this scorer has a tone probe"

    def char_signals(self, text):
        ppl = torch.full((len(text),), self.ppl_value)
        tone = torch.ones(len(text))
        idx = text.find(self.tone_high_token)
        if idx >= 0:
            tone[idx: idx + len(self.tone_high_token)] = 2.0
        return ppl, tone


def _scorer_config():
    # keep_boundary=1 and ratio=2 on n=6 leaves exactly one middle slot, so
    # the single chosen middle token is unambiguously the top-scoring one.
    return CompressionConfig(target_ratio=2.0, keep_boundary_tokens=1)


class TestLACCScoreWiring:
    def test_probe_signal_changes_selection_but_rule_ignores_it(self, tokenizer):
        # Six ascii (non-tone) tokens; the fake makes token index 3 ("dd")
        # tonally salient. With one middle slot, the model arm must keep it
        # and the rule arm (which ignores the probe) must not.
        ids = tokenizer.encode("aa bb cc dd ee ff")
        assert len(ids) == 6

        model_arm = LACCCompressor(
            tokenizer, model=None, scorer=FakeCharSignalScorer("dd"),
            tone_source="model", config=_scorer_config(),
        )
        rule_arm = LACCCompressor(
            tokenizer, model=None, scorer=FakeCharSignalScorer("dd"),
            tone_source="rule", config=_scorer_config(),
        )
        rm = model_arm.compress(list(ids))
        rr = rule_arm.compress(list(ids))

        model_keep = [ids.index(t) for t in rm.compressed_ids]
        rule_keep = [ids.index(t) for t in rr.compressed_ids]

        assert 3 in model_keep
        assert 3 not in rule_keep
        assert model_keep != rule_keep
        assert 0 in model_keep and 5 in model_keep
        assert rm.metadata["tone_source"] == "model"
        assert rr.metadata["tone_source"] == "rule"

    def test_output_is_a_valid_compression(self, tokenizer, vi_ids):
        comp = LACCCompressor(
            tokenizer, model=None, scorer=FakeCharSignalScorer("xxxx"), tone_source="model",
            config=CompressionConfig(target_ratio=4.0),
        )
        result = comp.compress(list(vi_ids))
        assert result.compressed_length < len(vi_ids)
        assert result.compression_ratio > 1.0
        assert 0.0 <= result.metadata["tone_preservation_rate"] <= 1.0

    def test_model_tone_falls_back_to_rule_for_uncovered_tokens(self, tokenizer):
        # A scorer whose tone signal covers nothing (all NaN) must not zero
        # out tone-bearing tokens -- they should inherit the rule weight.
        class AllNaN(LACCScorer):
            def __init__(self):
                self.tone_probe = object()

            def char_signals(self, text):
                nan = torch.full((len(text),), float("nan"))
                return nan.clone(), nan.clone()

        ids = tokenizer.encode("Hà Nội là thủ đô của Việt Nam")
        comp = LACCCompressor(
            tokenizer, model=None, scorer=AllNaN(), tone_source="model",
            config=CompressionConfig(target_ratio=2.0),
        )
        result = comp.compress(list(ids))
        # The scorer's mean isn't reported directly, but a tone-bearing text
        # with a fully-uncovered probe must still get real (non-zero) rule
        # weights, reflected in a non-trivial TPR rather than a degenerate one.
        assert 0.0 < result.metadata["tone_preservation_rate"] <= 1.0


# ============================================================================
# SemanticQualityGate
# ============================================================================


def _fixed_similarity(value):
    return lambda orig_ids, compressed_ids: value


def _coverage_similarity(orig_ids, compressed_ids):
    """Similarity proportional to how much of the original sequence
    survived -- lets restoring tokens actually raise the score."""
    return 1.0 if not orig_ids else len(compressed_ids) / len(orig_ids)


class TestSemanticQualityGate:
    def test_does_not_fire_when_similarity_already_above_threshold(self):
        gate = SemanticQualityGate(_fixed_similarity(0.95), threshold=0.85)
        input_ids = list(range(10))
        retained = [0, 1, 8, 9]
        compressed, info = gate.apply(input_ids, retained, scores=[1.0] * 10)
        assert info["gate_fired"] is False
        assert compressed == [input_ids[i] for i in retained]

    def test_fires_and_restores_when_similarity_below_threshold(self):
        input_ids = list(range(20))
        retained = [0, 19]
        scores = list(range(20))
        gate = SemanticQualityGate(_coverage_similarity, threshold=0.85, max_restore_fraction=1.0, batch_fraction=0.2)
        compressed, info = gate.apply(input_ids, retained, scores)
        assert info["gate_fired"] is True
        assert info["n_restored"] > 0
        assert info["final_similarity"] >= info["initial_similarity"]
        assert len(compressed) > len(retained)
        assert compressed == sorted(compressed)

    def test_respects_max_restore_fraction_cap(self):
        input_ids = list(range(20))
        retained = [0, 19]
        dropped_count = len(input_ids) - len(retained)
        gate = SemanticQualityGate(_fixed_similarity(0.0), threshold=0.85, max_restore_fraction=0.25, batch_fraction=0.5)
        _, info = gate.apply(input_ids, retained, scores=[1.0] * len(input_ids))
        assert info["n_restored"] <= max(1, round(dropped_count * 0.25))

    def test_restores_highest_scoring_tokens_first(self):
        input_ids = list(range(10))
        retained = [0, 9]
        scores = [0] * 10
        scores[5] = 100
        gate = SemanticQualityGate(_coverage_similarity, threshold=0.31, max_restore_fraction=1.0, batch_fraction=0.125)
        _, info = gate.apply(input_ids, retained, scores)
        assert 5 in info["retained_indices"]

    def test_no_dropped_tokens_is_a_no_op(self):
        gate = SemanticQualityGate(_fixed_similarity(0.5), threshold=0.85)
        input_ids = [1, 2, 3]
        compressed, info = gate.apply(input_ids, [0, 1, 2], scores=[1.0, 1.0, 1.0])
        assert compressed == input_ids
        assert info["gate_fired"] is False


# ============================================================================
# compute_query_relevance_weights
# ============================================================================


class TestQueryRelevanceWeights:
    def test_no_query_returns_all_ones(self):
        assert compute_query_relevance_weights(["xin", "chào", "bạn"], query=None) == [1.0, 1.0, 1.0]

    def test_empty_query_returns_all_ones(self):
        assert compute_query_relevance_weights(["xin", "chào", "bạn"], query="") == [1.0, 1.0, 1.0]

    def test_matching_token_gets_boosted(self):
        tokens = ["giá", "vé", "máy", "bay", "hôm", "nay"]
        weights = compute_query_relevance_weights(tokens, query="giá vé máy bay bao nhiêu?", boost=2.0)
        assert weights[0] == 2.0
        assert weights[4] == 1.0

    def test_matching_is_case_insensitive(self):
        weights = compute_query_relevance_weights(["Hanoi"], query="what is the weather in hanoi", boost=1.5)
        assert weights[0] == 1.5

    def test_short_tokens_never_match(self):
        weights = compute_query_relevance_weights(["đi"], query="đi đâu vậy", boost=1.5, min_token_len=3)
        assert weights[0] == 1.0

    def test_empty_tokens_returns_empty_list(self):
        assert compute_query_relevance_weights([], query="anything") == []


# ============================================================================
# METHODS registry / create_compressor()
# ============================================================================


def test_registry_lists_expected_methods():
    assert set(METHODS) == {"none", "random", "llmlingua", "snapkv", "selective", "lacc"}


def test_create_compressor_none(tokenizer):
    assert isinstance(create_compressor("none", tokenizer), NoCompressor)


def test_create_compressor_random(tokenizer):
    assert isinstance(create_compressor("random", tokenizer), RandomCompressor)


def test_create_compressor_lacc_forwards_kwargs(tokenizer):
    comp = create_compressor("lacc", tokenizer, model=None, device="cpu", use_perplexity=False, use_morphology=False)
    assert isinstance(comp, LACCCompressor)
    assert comp.use_morphology is False


def test_create_compressor_unknown_method_raises(tokenizer):
    with pytest.raises(ValueError):
        create_compressor("does-not-exist", tokenizer)


def test_none_and_random_are_usable_end_to_end(tokenizer, vi_ids):
    for method in ("none", "random"):
        result = create_compressor(method, tokenizer).compress(vi_ids)
        assert result.compressed_length <= len(vi_ids)


# ============================================================================
# JSON-safety of aggregate metadata (guards json.dumps(..., allow_nan=False))
# ============================================================================


def test_lacc_result_metadata_is_json_serializable(tokenizer, vi_ids):
    result = _lacc_no_model(tokenizer).compress(vi_ids)
    json.dumps(result.metadata, allow_nan=False)
