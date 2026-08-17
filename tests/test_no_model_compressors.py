"""Tests for the CPU-only compressors in vncompress/compressors/no_model.py."""
import pytest

from vncompress.compressors.no_model import (
    NoModelBaselineCompressor,
    NoModelCombinedCompressor,
    NoModelCompressor,
    NoModelMorphCompressor,
    NoModelResult,
    NoModelToneCompressor,
    evaluate_no_model,
)


def _is_subsequence(sub, full):
    it = iter(full)
    return all(item in it for item in sub)


COMPRESSOR_FACTORIES = {
    "tone": lambda tok: NoModelToneCompressor(tok),
    "morph": lambda tok: NoModelMorphCompressor(tok),
    "combined": lambda tok: NoModelCombinedCompressor(tok),
    "baseline_first": lambda tok: NoModelBaselineCompressor(tok, mode="first"),
    "baseline_random": lambda tok: NoModelBaselineCompressor(tok, mode="random"),
    "baseline_every_nth": lambda tok: NoModelBaselineCompressor(tok, mode="every_nth"),
    "baseline_word_length": lambda tok: NoModelBaselineCompressor(tok, mode="word_length"),
}


@pytest.mark.parametrize("name", COMPRESSOR_FACTORIES)
def test_compress_returns_no_model_result(name, tokenizer, vi_ids):
    comp = COMPRESSOR_FACTORIES[name](tokenizer)
    result = comp.compress(vi_ids, target_ratio=4.0)
    assert isinstance(result, NoModelResult)
    assert result.original_length == len(vi_ids)
    assert result.compressed_length == len(result.compressed_ids)


@pytest.mark.parametrize("name", COMPRESSOR_FACTORIES)
def test_token_order_is_preserved(name, tokenizer, vi_ids):
    comp = COMPRESSOR_FACTORIES[name](tokenizer)
    result = comp.compress(vi_ids, target_ratio=4.0)
    assert _is_subsequence(result.compressed_ids, vi_ids)


@pytest.mark.parametrize("name", COMPRESSOR_FACTORIES)
def test_boundary_tokens_kept(name, tokenizer, vi_ids):
    comp = COMPRESSOR_FACTORIES[name](tokenizer)
    result = comp.compress(vi_ids, target_ratio=4.0, keep_boundary=2)
    assert result.compressed_ids[0] == vi_ids[0]
    assert result.compressed_ids[-1] == vi_ids[-1]


@pytest.mark.parametrize("name", COMPRESSOR_FACTORIES)
def test_compression_ratio_in_right_ballpark(name, tokenizer, vi_ids):
    comp = COMPRESSOR_FACTORIES[name](tokenizer)
    result = comp.compress(vi_ids, target_ratio=4.0)
    target_len = len(vi_ids) / 4.0
    assert result.compressed_length <= len(vi_ids)
    assert result.compressed_length == pytest.approx(target_len, abs=max(4, int(target_len * 0.5)))


@pytest.mark.parametrize("name", COMPRESSOR_FACTORIES)
def test_empty_input_does_not_crash(name, tokenizer):
    comp = COMPRESSOR_FACTORIES[name](tokenizer)
    result = comp.compress([], target_ratio=4.0)
    assert result.compressed_ids == []
    assert result.original_length == 0


@pytest.mark.parametrize("name", COMPRESSOR_FACTORIES)
def test_very_short_input_does_not_crash(name, tokenizer):
    comp = COMPRESSOR_FACTORIES[name](tokenizer)
    ids = tokenizer.encode("một hai")
    result = comp.compress(ids, target_ratio=4.0)
    assert result.compressed_length <= len(ids)


@pytest.mark.parametrize("name", COMPRESSOR_FACTORIES)
def test_large_target_ratio_still_keeps_boundary(name, tokenizer, vi_ids):
    comp = COMPRESSOR_FACTORIES[name](tokenizer)
    result = comp.compress(vi_ids, target_ratio=1000.0, keep_boundary=2)
    assert result.compressed_ids[0] == vi_ids[0]
    assert result.compressed_ids[-1] == vi_ids[-1]
    assert result.compressed_length >= 1


def test_base_class_compress_raises_not_implemented(tokenizer):
    comp = NoModelCompressor(tokenizer)
    with pytest.raises(NotImplementedError):
        comp.compress([1, 2, 3])


def test_tone_compressor_metadata_has_mean_tone_weight(tokenizer, vi_ids):
    result = NoModelToneCompressor(tokenizer).compress(vi_ids, target_ratio=4.0)
    assert "mean_tone_weight" in result.metadata


def test_morph_compressor_metadata_has_class_distribution(tokenizer, vi_ids):
    result = NoModelMorphCompressor(tokenizer).compress(vi_ids, target_ratio=4.0)
    assert "class_distribution" in result.metadata


def test_baseline_word_length_prefers_longer_tokens(tokenizer):
    text = "a bb ccc dddd eeeee ffffff ggggggg hhhhhhhh"
    ids = tokenizer.encode(text)
    comp = NoModelBaselineCompressor(tokenizer, mode="word_length")
    result = comp.compress(ids, target_ratio=2.0, keep_boundary=1)
    kept_words = [tokenizer.decode([i]) for i in result.compressed_ids]
    assert "hhhhhhhh" in kept_words or "ggggggg" in kept_words


def test_evaluate_no_model_runs_all_methods(tokenizer, vi_text, capsys):
    results = evaluate_no_model(vi_text, tokenizer, target_ratio=4.0)
    expected = {
        "baseline_first", "baseline_random", "baseline_longest",
        "tone_only", "morph_only", "tone_morph_combined",
    }
    assert set(results.keys()) == expected
    for result in results.values():
        assert isinstance(result, NoModelResult)
        assert result.compressed_length <= result.original_length
