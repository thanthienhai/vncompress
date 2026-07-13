"""
Semantic Quality Gate (LACC improvement C1)
============================================
Compression selects tokens using a linguistic/perplexity score, but nothing
so far checks whether the *result* still means roughly the same thing as
the original. This module adds a post-hoc safety net: embed the original
and compressed token sequences, compare them with cosine similarity, and if
they've drifted too far apart, restore the highest-scoring dropped tokens
until similarity recovers (or a restore budget runs out).

This turns "compress to a fixed ratio no matter what" into "compress to a
fixed ratio, unless that damages meaning too much, in which case spend a
few more tokens to fix it."

`SemanticQualityGate` itself has no torch dependency — it operates on a
generic `similarity_fn(orig_ids, compressed_ids) -> float` callable, so it
can be tested with a plain-Python similarity function. `make_similarity_fn_from_model`
is the torch-dependent adapter that builds such a callable from an actual
causal LM (mean-pooled last hidden state, since that avoids needing a
separate embedding model).
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple


class SemanticQualityGate:
    """
    Post-compression safety net: restore highest-scoring dropped tokens
    until embedding similarity between original and compressed context is
    >= `threshold`, or until `max_restore_fraction` of the dropped tokens
    have been restored (whichever comes first — this caps how much of the
    compression benefit the gate is allowed to give back).
    """

    def __init__(
        self,
        similarity_fn: Callable[[List[int], List[int]], float],
        threshold: float = 0.85,
        max_restore_fraction: float = 0.5,
        batch_fraction: float = 0.1,
    ):
        """
        Args:
            similarity_fn: (original_ids, compressed_ids) -> similarity in
                roughly [0, 1] (or [-1, 1] for raw cosine — only the
                comparison against `threshold` matters, so use whatever
                scale your embedding source produces).
            threshold: minimum acceptable similarity. Below this, the gate
                restores tokens.
            max_restore_fraction: hard cap on how many of the dropped
                tokens can be restored, as a fraction of the drop count.
                Prevents the gate from silently undoing most of the
                compression when similarity can't be recovered cheaply.
            batch_fraction: how many dropped tokens to restore per
                iteration (as a fraction of the drop count), so we don't
                need one similarity computation per single token.
        """
        self.similarity_fn = similarity_fn
        self.threshold = threshold
        self.max_restore_fraction = max_restore_fraction
        self.batch_fraction = batch_fraction

    def apply(
        self,
        input_ids: List[int],
        retained_indices: Sequence[int],
        scores: Sequence[float],
    ) -> Tuple[List[int], Dict]:
        """
        Args:
            input_ids: full original token IDs.
            retained_indices: indices (into input_ids) kept by the
                compressor's top-k selection.
            scores: per-token importance score for every index in
                input_ids — used to decide which dropped tokens are
                worth restoring first (highest score restored first).

        Returns:
            (final_ids, info): final_ids is retained_indices unchanged if
            the gate didn't fire, or the ids after restoring tokens.
            info reports whether the gate fired and how much it restored.
        """
        n = len(input_ids)
        retained_set = set(retained_indices)
        dropped = sorted(i for i in range(n) if i not in retained_set)

        compressed_ids = [input_ids[i] for i in sorted(retained_set)]
        info: Dict = {
            'gate_fired': False,
            'threshold': self.threshold,
            'n_restored': 0,
            'retained_indices': sorted(retained_set),
        }

        if not dropped:
            similarity = self.similarity_fn(input_ids, compressed_ids)
            info['initial_similarity'] = similarity
            info['final_similarity'] = similarity
            return compressed_ids, info

        similarity = self.similarity_fn(input_ids, compressed_ids)
        info['initial_similarity'] = similarity
        info['final_similarity'] = similarity

        if similarity >= self.threshold:
            return compressed_ids, info

        info['gate_fired'] = True

        dropped_by_score = sorted(dropped, key=lambda i: scores[i], reverse=True)
        max_restore = max(1, int(round(len(dropped) * self.max_restore_fraction)))
        batch_size = max(1, int(round(len(dropped) * self.batch_fraction)))

        restored = 0
        pos = 0
        while (
            similarity < self.threshold
            and restored < max_restore
            and pos < len(dropped_by_score)
        ):
            batch = dropped_by_score[pos:pos + min(batch_size, max_restore - restored)]
            if not batch:
                break
            retained_set.update(batch)
            pos += len(batch)
            restored += len(batch)
            compressed_ids = [input_ids[i] for i in sorted(retained_set)]
            similarity = self.similarity_fn(input_ids, compressed_ids)

        info['final_similarity'] = similarity
        info['n_restored'] = restored
        info['retained_indices'] = sorted(retained_set)
        return compressed_ids, info


# ============================================================================
# Torch-dependent adapter: build a similarity_fn from a real model
# ============================================================================

def embed_ids_mean_pooled(model, input_ids: List[int], device=None):
    """
    Mean-pool a causal LM's last hidden state over sequence length as a
    cheap sentence embedding — avoids needing a separate embedding model
    (e.g. the tiny scorer model already loaded for perplexity is enough).
    """
    import torch

    if not input_ids:
        hidden_size = getattr(model.config, 'hidden_size', 1)
        return torch.zeros(hidden_size)

    dev = device or next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], device=dev)
    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0]  # [seq_len, hidden]
    return hidden.mean(dim=0).float().cpu()


def cosine_similarity(u, v) -> float:
    import torch

    u = u.flatten()
    v = v.flatten()
    denom = float(u.norm() * v.norm())
    if denom < 1e-8:
        return 1.0
    return float(torch.dot(u, v) / denom)


def make_similarity_fn_from_model(
    model,
    device: Optional[str] = None,
) -> Callable[[List[int], List[int]], float]:
    """
    Build a `(orig_ids, compressed_ids) -> cosine_similarity` callable
    backed by a real model's hidden states — the piece EnhancedCompressor /
    CombinedCompressor pass into SemanticQualityGate when a model is
    available (tiny scorer model or the main generation model both work).
    """

    def _similarity_fn(orig_ids: List[int], comp_ids: List[int]) -> float:
        emb_a = embed_ids_mean_pooled(model, orig_ids, device)
        emb_b = embed_ids_mean_pooled(model, comp_ids, device)
        return cosine_similarity(emb_a, emb_b)

    return _similarity_fn
