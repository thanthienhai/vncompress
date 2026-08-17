"""Tests for the main research compressors (ToneAwareCompressor,
MorphologyAwareCompressor, CombinedCompressor) in
vncompress/compressors/tone_aware.py.

These run with model=None on device='cpu': all three fall back to
uniform/ones base scores when no model is supplied (see
`_compute_base_scores` / the `if self.model is not None` guards in
compress()), so the tone/morphology scoring logic can be exercised fully
without downloading or loading any neural network.
"""
import pytest

from vncompress.compressors.base import CompressionConfig
from vncompress.compressors.tone_aware import (
    CombinedCompressor,
    MorphologyAwareCompressor,
    ToneAwareCompressor,
)

COMPRESSOR_CLASSES = [ToneAwareCompressor, MorphologyAwareCompressor, CombinedCompressor]


def _is_subsequence(sub, full):
    it = iter(full)
    return all(item in it for item in sub)


@pytest.mark.parametrize("cls", COMPRESSOR_CLASSES)
def test_compress_no_model_cpu_does_not_crash(cls, tokenizer, vi_ids):
    comp = cls(tokenizer, model=None, device="cpu")
    result = comp.compress(vi_ids)
    assert result.compressed_length <= result.original_length
    assert result.compressed_length > 0


@pytest.mark.parametrize("cls", COMPRESSOR_CLASSES)
def test_token_order_is_preserved(cls, tokenizer, vi_ids):
    comp = cls(tokenizer, model=None, device="cpu")
    result = comp.compress(vi_ids)
    assert _is_subsequence(result.compressed_ids, vi_ids)


@pytest.mark.parametrize("cls", COMPRESSOR_CLASSES)
def test_boundary_tokens_are_kept(cls, tokenizer, vi_ids):
    config = CompressionConfig(target_ratio=4.0, keep_boundary_tokens=2)
    comp = cls(tokenizer, model=None, device="cpu", config=config)
    result = comp.compress(vi_ids)
    assert result.compressed_ids[:2] == vi_ids[:2]
    assert result.compressed_ids[-2:] == vi_ids[-2:]


@pytest.mark.parametrize("cls", COMPRESSOR_CLASSES)
def test_decode_does_not_crash(cls, tokenizer, vi_ids):
    comp = cls(tokenizer, model=None, device="cpu")
    result = comp.compress(vi_ids)
    assert isinstance(result.compressed_text, str)


@pytest.mark.parametrize("cls", COMPRESSOR_CLASSES)
def test_very_short_sequence_does_not_crash(cls, tokenizer):
    comp = cls(tokenizer, model=None, device="cpu")
    ids = tokenizer.encode("một hai")
    result = comp.compress(ids)
    assert result.compressed_length <= len(ids)


@pytest.mark.parametrize("cls", COMPRESSOR_CLASSES)
@pytest.mark.parametrize("ratio", [1.5, 8.0])
def test_edge_case_ratios_do_not_crash(cls, tokenizer, vi_ids, ratio):
    config = CompressionConfig(target_ratio=ratio)
    comp = cls(tokenizer, model=None, device="cpu", config=config)
    result = comp.compress(vi_ids)
    assert 0 < result.compressed_length <= len(vi_ids)


def test_tone_aware_get_name():
    from vncompress.compressors.base import CompressionConfig as _  # noqa: F401
    assert ToneAwareCompressor.__name__ == "ToneAwareCompressor"


def test_tone_aware_reports_tone_preservation_rate(tokenizer, vi_ids):
    comp = ToneAwareCompressor(tokenizer, model=None, device="cpu")
    result = comp.compress(vi_ids)
    assert 0.0 <= result.metadata["tone_preservation_rate"] <= 1.0


def test_morphology_aware_reports_class_distribution(tokenizer, vi_ids):
    comp = MorphologyAwareCompressor(tokenizer, model=None, device="cpu")
    result = comp.compress(vi_ids)
    assert "original_class_distribution" in result.metadata
    assert "compressed_class_distribution" in result.metadata


def test_combined_query_boost_keeps_query_relevant_token(tokenizer):
    # "vào" is a low-priority function word (small morphology multiplier,
    # no tone mark) that loses out to content words under normal scoring,
    # so it's a clean signal that the query boost in compute_query_relevance_weights
    # is actually reaching token selection: present only once boosted.
    text = "chúng tôi đi học ở trường đại học rất xa vào buổi sáng hôm nay trời đẹp"
    ids = tokenizer.encode(text)
    config = CompressionConfig(target_ratio=4.0, keep_boundary_tokens=1)
    comp = CombinedCompressor(tokenizer, model=None, device="cpu", config=config)

    vao_id = tokenizer.encode("vào")[0]
    without_query = comp.compress(list(ids))
    with_query = comp.compress(list(ids), query="vào", query_boost=5.0)

    assert vao_id not in without_query.compressed_ids
    assert vao_id in with_query.compressed_ids


def test_non_vietnamese_text_without_language_fallback_does_not_crash(tokenizer, en_text):
    # auto_detect_language=False skips the "fall back to LLMLingua" branch,
    # which requires a real model and isn't usable with model=None -- that
    # fallback path is exercised only when a model is available.
    ids = tokenizer.encode(en_text)
    comp = ToneAwareCompressor(tokenizer, model=None, device="cpu", auto_detect_language=False)
    result = comp.compress(ids)
    assert result.compressed_length <= len(ids)
