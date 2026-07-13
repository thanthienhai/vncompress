"""
Task/Query-Aware Compression Boost (LACC improvement C2)
==========================================================
Compression so far picks tokens using signals intrinsic to the input text
(perplexity, tone, morphology) — nothing about *why* the context is being
compressed. But a downstream RAG query or tool-calling schema often names
exactly the entities/keywords that must survive compression. This module
computes a cheap per-token boost from lexical overlap with a `query` string
(e.g. a user question, or a serialized tool schema), so query-relevant
tokens are less likely to be dropped — without hard-coding a keep_ratio
override for them (they still compete on score with everything else, just
from a higher starting point).

Deliberately not embedding-based: works with any tokenizer, doesn't need a
model, and is easy to reason about and tune (`query_boost`).
"""

import re
from typing import List, Optional


def compute_query_relevance_weights(
    tokens: List[str],
    query: Optional[str],
    boost: float = 1.5,
    min_token_len: int = 2,
) -> List[float]:
    """
    Args:
        tokens: decoded token strings (whitespace-stripped, one per input id).
        query: the downstream question / tool schema text tokens are scored
            against. None or empty -> no-op (all weights 1.0).
        boost: multiplier applied to a token that appears as a whole word in
            `query` (case-insensitive). Tokens shorter than `min_token_len`
            never match, since 1-2 char overlaps (e.g. "la", "di") are almost
            always coincidental rather than query-relevant.
        min_token_len: minimum decoded-token length (in characters) required
            before a match can boost it.

    Returns:
        List of per-token multipliers, same length as `tokens`: `boost` for
        a query-overlapping token, `1.0` otherwise.
    """
    if not query:
        return [1.0] * len(tokens)

    query_words = set(w.lower() for w in re.findall(r"\w+", query, flags=re.UNICODE))
    if not query_words:
        return [1.0] * len(tokens)

    weights = []
    for tok in tokens:
        t = tok.strip().lower()
        if len(t) >= min_token_len and t in query_words:
            weights.append(boost)
        else:
            weights.append(1.0)
    return weights
