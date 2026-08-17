"""Tests for vncompress/evaluation/method_taxonomy.py, resolving P1 issue
"eval: tach baseline va proposed method trong benchmark"."""
import pytest

from vncompress.compressors import COMPRESSOR_REGISTRY
from vncompress.evaluation.method_taxonomy import (
    ABLATION_ARM_CATEGORY,
    REGISTRY_METHOD_CATEGORY,
    MethodCategory,
    categorize,
)


def test_every_compressor_registry_key_is_classified():
    missing = set(COMPRESSOR_REGISTRY) - set(REGISTRY_METHOD_CATEGORY)
    assert not missing, f"COMPRESSOR_REGISTRY methods missing from taxonomy: {missing}"


@pytest.mark.parametrize("name", ["none", "random", "llmlingua", "llmlingua_small", "snapkv", "selective"])
def test_prior_art_methods_are_baseline(name):
    assert categorize(name, context="registry") == MethodCategory.BASELINE


@pytest.mark.parametrize("name", ["tone_aware", "morphology_aware", "combined"])
def test_lacc_contributions_are_proposed_in_registry_context(name):
    assert categorize(name, context="registry") == MethodCategory.PROPOSED


@pytest.mark.parametrize("name", ["ppl_only", "tone_only", "morph_only", "combined"])
def test_ablation_arms_are_ablation_in_ablation_context(name):
    assert categorize(name, context="ablation") == MethodCategory.ABLATION


def test_combined_is_proposed_in_registry_but_ablation_in_ablation_context():
    # Same method name, different table -> different category, by design
    # (see method_taxonomy.py module docstring).
    assert categorize("combined", context="registry") == MethodCategory.PROPOSED
    assert categorize("combined", context="ablation") == MethodCategory.ABLATION


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        categorize("not_a_real_method", context="registry")


def test_invalid_context_raises():
    with pytest.raises(ValueError):
        categorize("none", context="not_a_real_context")


def test_ablation_arm_category_only_contains_ablation():
    assert set(ABLATION_ARM_CATEGORY.values()) == {MethodCategory.ABLATION}
