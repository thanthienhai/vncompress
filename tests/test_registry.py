"""Tests for the compressor registry / factory in
vncompress/compressors/__init__.py.

Only exercises 'none' and 'random', which need nothing but a tokenizer --
the other registry entries (llmlingua, snapkv, tone_aware, ...) require a
real causal LM and are out of scope for a CPU-only, no-GPU test suite.
"""
import pytest

from vncompress import compressors


def test_registry_lists_expected_methods():
    expected = {
        "none", "random", "llmlingua", "llmlingua_small",
        "snapkv", "selective", "tone_aware", "morphology_aware", "combined",
    }
    assert expected.issubset(set(compressors.COMPRESSOR_REGISTRY.keys()))


def test_create_compressor_none(tokenizer):
    comp = compressors.create_compressor("none", tokenizer)
    assert isinstance(comp, compressors.NoCompressor)


def test_create_compressor_random(tokenizer):
    comp = compressors.create_compressor("random", tokenizer)
    assert isinstance(comp, compressors.RandomCompressor)


def test_create_compressor_unknown_method_raises(tokenizer):
    with pytest.raises(ValueError):
        compressors.create_compressor("does-not-exist", tokenizer)


def test_none_and_random_are_usable_end_to_end(tokenizer, vi_ids):
    for method in ("none", "random"):
        comp = compressors.create_compressor(method, tokenizer)
        result = comp.compress(vi_ids)
        assert result.compressed_length <= len(vi_ids)
