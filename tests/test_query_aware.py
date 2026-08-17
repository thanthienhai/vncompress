"""Tests for compute_query_relevance_weights (vncompress/compressors/query_aware.py)."""
from vncompress.compressors.query_aware import compute_query_relevance_weights


def test_no_query_returns_all_ones():
    tokens = ["xin", "chào", "bạn"]
    weights = compute_query_relevance_weights(tokens, query=None)
    assert weights == [1.0] * len(tokens)


def test_empty_query_returns_all_ones():
    tokens = ["xin", "chào", "bạn"]
    weights = compute_query_relevance_weights(tokens, query="")
    assert weights == [1.0] * len(tokens)


def test_matching_token_gets_boosted():
    tokens = ["giá", "vé", "máy", "bay", "hôm", "nay"]
    weights = compute_query_relevance_weights(
        tokens, query="giá vé máy bay bao nhiêu?", boost=2.0
    )
    assert weights[0] == 2.0  # "giá"
    assert weights[4] == 1.0  # "hôm" not in query


def test_matching_is_case_insensitive():
    weights = compute_query_relevance_weights(
        ["Hanoi"], query="what is the weather in hanoi", boost=1.5
    )
    assert weights[0] == 1.5


def test_short_tokens_never_match():
    weights = compute_query_relevance_weights(
        ["đi"], query="đi đâu vậy", boost=1.5, min_token_len=3
    )
    assert weights[0] == 1.0


def test_output_length_matches_input_length():
    tokens = ["a", "b", "c", "d"]
    weights = compute_query_relevance_weights(tokens, query="a c")
    assert len(weights) == len(tokens)


def test_empty_tokens_returns_empty_list():
    assert compute_query_relevance_weights([], query="anything") == []
