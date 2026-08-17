"""Tests for BaseCompressor, CompressionResult, CompressionConfig,
NoCompressor and RandomCompressor (vncompress/compressors/base.py)."""
import pytest

from vncompress.compressors.base import (
    BaseCompressor,
    CompressionConfig,
    CompressionResult,
    NoCompressor,
    RandomCompressor,
)


def _is_subsequence(sub, full):
    """True if `sub` occurs as an (order-preserving) subsequence of `full`."""
    it = iter(full)
    return all(item in it for item in sub)


def test_compression_config_defaults():
    config = CompressionConfig()
    assert config.target_ratio == 4.0
    assert config.keep_special_tokens is True
    assert config.keep_boundary_tokens == 2
    assert config.min_compressed_length == 1
    assert config.language == "vi"


def test_compression_result_is_a_dataclass_with_expected_fields():
    result = CompressionResult(
        compressed_ids=[1, 2, 3],
        compressed_text="a b c",
        compression_ratio=2.0,
        token_savings_pct=50.0,
        original_length=6,
        compressed_length=3,
        method_name="test",
        processing_time_ms=1.0,
    )
    assert result.compressed_ids == [1, 2, 3]
    assert result.metadata == {}
    assert result.kv_cache_mask is None
    assert result.kv_memory_saved_bytes == 0


class TestNoCompressor:
    def test_returns_input_unchanged(self, tokenizer, vi_ids):
        comp = NoCompressor(tokenizer)
        result = comp.compress(vi_ids)
        assert result.compressed_ids == list(vi_ids)
        assert result.compression_ratio == 1.0
        assert result.token_savings_pct == 0.0
        assert result.original_length == len(vi_ids)
        assert result.compressed_length == len(vi_ids)

    def test_get_name(self, tokenizer):
        assert NoCompressor(tokenizer).get_name() == "NoCompression"

    def test_empty_input_raises(self, tokenizer):
        comp = NoCompressor(tokenizer)
        with pytest.raises(ValueError):
            comp.compress([])

    def test_compress_text_roundtrips_through_tokenizer(self, tokenizer):
        comp = NoCompressor(tokenizer)
        result = comp.compress_text("xin chào các bạn")
        assert result.compressed_text == "xin chào các bạn"

    def test_decode_does_not_crash_on_unknown_ids(self, tokenizer):
        comp = NoCompressor(tokenizer)
        # IDs the mock tokenizer never registered via encode()
        result = comp.compress([999, 1000])
        assert isinstance(result.compressed_text, str)

    def test_repr_includes_method_name(self, tokenizer):
        comp = NoCompressor(tokenizer)
        assert "NoCompression" in repr(comp)


class TestRandomCompressor:
    def test_get_name(self, tokenizer):
        assert RandomCompressor(tokenizer).get_name() == "RandomBaseline"

    def test_empty_input_raises(self, tokenizer):
        comp = RandomCompressor(tokenizer)
        with pytest.raises(ValueError):
            comp.compress([])

    def test_boundary_tokens_are_always_kept(self, tokenizer, vi_ids):
        config = CompressionConfig(target_ratio=4.0, keep_boundary_tokens=2)
        comp = RandomCompressor(tokenizer, config=config)
        result = comp.compress(vi_ids)
        assert result.compressed_ids[:2] == vi_ids[:2]
        assert result.compressed_ids[-2:] == vi_ids[-2:]

    def test_token_order_is_preserved(self, tokenizer, vi_ids):
        comp = RandomCompressor(tokenizer)
        result = comp.compress(vi_ids)
        assert _is_subsequence(result.compressed_ids, vi_ids)

    def test_compression_ratio_is_approximately_respected(self, tokenizer, vi_ids):
        config = CompressionConfig(target_ratio=2.0)
        comp = RandomCompressor(tokenizer, config=config)
        result = comp.compress(vi_ids)
        # Randomized selection; assert it's in the right ballpark rather than exact.
        assert result.compressed_length < len(vi_ids)
        assert result.compressed_length == pytest.approx(len(vi_ids) / 2, abs=3)

    def test_very_short_input_does_not_crash(self, tokenizer):
        comp = RandomCompressor(tokenizer)
        ids = tokenizer.encode("một hai")
        result = comp.compress(ids)
        assert result.compressed_length <= len(ids)

    @pytest.mark.parametrize("ratio", [1.0, 1.5, 8.0, 100.0])
    def test_extreme_target_ratios_do_not_crash(self, tokenizer, vi_ids, ratio):
        config = CompressionConfig(target_ratio=ratio)
        comp = RandomCompressor(tokenizer, config=config)
        result = comp.compress(vi_ids)
        assert 0 < result.compressed_length <= len(vi_ids)


def test_base_compressor_cannot_be_instantiated_directly(tokenizer):
    with pytest.raises(TypeError):
        BaseCompressor(tokenizer)
