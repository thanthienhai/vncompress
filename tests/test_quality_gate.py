"""Tests for SemanticQualityGate (vncompress/compressors/quality_gate.py).

SemanticQualityGate has no torch dependency -- it takes any
`similarity_fn(orig_ids, compressed_ids) -> float` callable, so it can be
tested with a plain-Python stand-in instead of a real embedding model.
"""
from vncompress.compressors.quality_gate import SemanticQualityGate


def make_similarity_fn(fixed_value):
    """Similarity function that ignores its inputs and always returns a
    fixed score -- used to test gate firing/not-firing deterministically."""
    return lambda orig_ids, compressed_ids: fixed_value


def coverage_similarity_fn(orig_ids, compressed_ids):
    """Similarity proportional to how much of the original sequence survived
    -- lets restoring tokens actually raise the similarity score, so the
    gate's restore loop has something real to converge on."""
    if not orig_ids:
        return 1.0
    return len(compressed_ids) / len(orig_ids)


def test_gate_does_not_fire_when_similarity_already_above_threshold():
    gate = SemanticQualityGate(make_similarity_fn(0.95), threshold=0.85)
    input_ids = list(range(10))
    retained = [0, 1, 8, 9]
    compressed, info = gate.apply(input_ids, retained, scores=[1.0] * 10)
    assert info["gate_fired"] is False
    assert compressed == [input_ids[i] for i in retained]
    assert info["n_restored"] == 0


def test_gate_fires_and_restores_when_similarity_below_threshold():
    input_ids = list(range(20))
    retained = [0, 19]  # aggressive compression -> low coverage similarity
    scores = list(range(20))  # higher index == higher score

    gate = SemanticQualityGate(
        coverage_similarity_fn, threshold=0.85, max_restore_fraction=1.0, batch_fraction=0.2,
    )
    compressed, info = gate.apply(input_ids, retained, scores)

    assert info["gate_fired"] is True
    assert info["n_restored"] > 0
    assert info["final_similarity"] >= info["initial_similarity"]
    assert len(compressed) > len(retained)
    assert compressed == sorted(compressed)


def test_gate_respects_max_restore_fraction_cap():
    input_ids = list(range(20))
    retained = [0, 19]
    dropped_count = len(input_ids) - len(retained)
    scores = [1.0] * len(input_ids)

    # Similarity that can never reach threshold, so the gate always
    # restores up to (but not past) its cap.
    gate = SemanticQualityGate(
        make_similarity_fn(0.0), threshold=0.85, max_restore_fraction=0.25, batch_fraction=0.5,
    )
    _, info = gate.apply(input_ids, retained, scores)

    max_allowed = max(1, int(round(dropped_count * 0.25)))
    assert info["n_restored"] <= max_allowed


def test_gate_restores_highest_scoring_tokens_first():
    input_ids = list(range(10))
    retained = [0, 9]
    # Token 5 has the highest score among dropped tokens -> should be the
    # first one restored when the gate needs exactly one restoration.
    scores = [0] * 10
    scores[5] = 100

    gate = SemanticQualityGate(
        coverage_similarity_fn, threshold=0.31, max_restore_fraction=1.0, batch_fraction=0.125,
    )
    compressed, info = gate.apply(input_ids, retained, scores)
    assert 5 in info["retained_indices"]


def test_gate_with_no_dropped_tokens_is_a_no_op():
    gate = SemanticQualityGate(make_similarity_fn(0.5), threshold=0.85)
    input_ids = [1, 2, 3]
    retained = [0, 1, 2]
    compressed, info = gate.apply(input_ids, retained, scores=[1.0, 1.0, 1.0])
    assert compressed == input_ids
    assert info["gate_fired"] is False
