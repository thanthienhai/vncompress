"""
Baseline / proposed / ablation taxonomy for VCC-Bench methods.
================================================================
Resolves P1 issue "eval: tach baseline va proposed method trong benchmark":
without an explicit classification, a results table is just a flat list of
method names and it's easy to accidentally compare a baseline against an
ablation arm as if they were peers. This module is the single source of
truth for which category each method name belongs to, used by
scripts/summarize_results.py and (optionally) any script printing a
results table.

Two separate namespaces, because run_benchmark.py and run_ablation.py
name their methods differently and are evaluated as independent tables:

  - "registry" methods: keys of vncompress.compressors.COMPRESSOR_REGISTRY,
    the --methods argument to run_benchmark.py. Split into:
      * baseline: methods with no Vietnamese-specific signal at all
        (none, random) or a general-purpose signal not designed for
        Vietnamese (llmlingua*, snapkv, selective) -- the prior art this
        project compares against.
      * proposed: this project's contributions (tone_aware,
        morphology_aware, combined).

  - "ablation" arms: ABLATION_METHODS in run_ablation.py, which isolates
    each scoring signal (perplexity / tone / morphology) one at a time
    plus the full combined method, to show each signal's incremental
    contribution. All four arms are labeled `ablation` here, including
    'combined' -- in the ablation *table* it plays the role of "the full
    method as a point of comparison for the isolated arms", a different
    question from "is this a baseline or our contribution" (its answer to
    that question, when reported in the main registry table, is
    `proposed` -- see REGISTRY_METHOD_CATEGORY).
"""
from enum import Enum
from typing import Dict


class MethodCategory(str, Enum):
    BASELINE = "baseline"
    PROPOSED = "proposed"
    ABLATION = "ablation"


REGISTRY_METHOD_CATEGORY: Dict[str, MethodCategory] = {
    "none": MethodCategory.BASELINE,
    "random": MethodCategory.BASELINE,
    "llmlingua": MethodCategory.BASELINE,
    "llmlingua_small": MethodCategory.BASELINE,
    "snapkv": MethodCategory.BASELINE,
    "selective": MethodCategory.BASELINE,
    "tone_aware": MethodCategory.PROPOSED,
    "morphology_aware": MethodCategory.PROPOSED,
    "combined": MethodCategory.PROPOSED,
}

ABLATION_ARM_CATEGORY: Dict[str, MethodCategory] = {
    "ppl_only": MethodCategory.ABLATION,
    "tone_only": MethodCategory.ABLATION,
    "morph_only": MethodCategory.ABLATION,
    "combined": MethodCategory.ABLATION,
}


def categorize(method_name: str, context: str = "registry") -> MethodCategory:
    """Look up a method's category.

    Args:
        method_name: a COMPRESSOR_REGISTRY key (context='registry') or an
            ABLATION_METHODS arm name (context='ablation').
        context: 'registry' (default, run_benchmark.py results) or
            'ablation' (run_ablation.py results) -- disambiguates
            'combined', which is `proposed` in the registry table and
            `ablation` in the ablation table (see module docstring).

    Returns:
        MethodCategory. Unknown method names outside both known tables
        (e.g. a new compressor added to the registry but not yet
        classified here) raise ValueError rather than silently defaulting
        to a category -- a mislabeled proposed method reported as a
        baseline (or vice versa) is worse than a loud failure.
    """
    table = REGISTRY_METHOD_CATEGORY if context == "registry" else ABLATION_ARM_CATEGORY
    if context not in ("registry", "ablation"):
        raise ValueError(f"Unknown context: {context!r}. Expected 'registry' or 'ablation'.")
    if method_name not in table:
        raise ValueError(
            f"Method {method_name!r} is not classified in method_taxonomy.py "
            f"(context={context!r}). Add it to REGISTRY_METHOD_CATEGORY or "
            f"ABLATION_ARM_CATEGORY before including it in a results table."
        )
    return table[method_name]
