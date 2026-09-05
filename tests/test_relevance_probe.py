"""Tests for the wave-2 E4 query-relevance probe (vncompress/linguistics.py):
build_relevance_labels and RelevanceConsistencyLoss.

CPU-only, no model download: build_relevance_labels uses conftest's
MockTokenizer, and the probe is exercised on a random hidden-state tensor.
"""
import torch

from vncompress.linguistics import (
    PhonologicalConsistencyLoss,
    RelevanceConsistencyLoss,
    build_relevance_labels,
)


class TestBuildRelevanceLabels:
    def test_answer_tokens_positive_filler_negative(self, tokenizer):
        # Answer syllables should be labelled 1; unrelated filler 0.
        context = "công ty được thành lập tại Hà Nội vào năm 2010"
        answer = "Hà Nội"
        ids = tokenizer.encode(context)
        labels = build_relevance_labels(tokenizer, ids, answer)
        assert len(labels) == len(ids)
        words = context.split()
        pos = {i for i, w in enumerate(words) if w.lower() in ("hà", "nội")}
        for i, lab in enumerate(labels):
            if i in pos:
                assert lab == 1, f"expected token {words[i]!r} relevant"
        # A clearly-unrelated content word is negative.
        assert labels[words.index("thành")] == 0

    def test_short_tokens_are_ignored(self, tokenizer):
        # Sub-min_token_len pieces -> ignore_index (-100), neither train nor score.
        ids = tokenizer.encode("a bb ccc")
        labels = build_relevance_labels(tokenizer, ids, "ccc", min_token_len=2)
        assert labels[0] == -100          # "a" too short
        assert labels[2] == 1             # "ccc" overlaps answer

    def test_empty_answer_gives_no_positives(self, tokenizer):
        ids = tokenizer.encode("một hai ba bốn")
        labels = build_relevance_labels(tokenizer, ids, "")
        assert all(lab in (0, -100) for lab in labels)


class TestRelevanceConsistencyLoss:
    def test_forward_returns_scalar_loss(self):
        probe = RelevanceConsistencyLoss(hidden_dim=32)
        h = torch.randn(2, 5, 32)
        labels = torch.tensor([[1, 0, -100, 1, 0], [0, 1, 1, -100, 0]])
        mask = torch.ones(2, 5)
        loss = probe(h, labels, mask)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_score_importance_range_and_shape(self):
        probe = RelevanceConsistencyLoss(hidden_dim=32)
        h = torch.randn(2, 7, 32)
        scores = probe.score_importance(h)
        assert scores.shape == (2, 7)
        assert float(scores.min()) >= 0.5
        assert float(scores.max()) <= 3.0

    def test_length_mismatch_raises(self):
        probe = RelevanceConsistencyLoss(hidden_dim=16)
        h = torch.randn(1, 4, 16)
        labels = torch.zeros(1, 3, dtype=torch.long)
        try:
            probe(h, labels)
            assert False, "expected ValueError on S mismatch"
        except ValueError:
            pass


class TestInterfaceCompatibleWithToneProbe:
    """The relevance probe must be a drop-in for the tone probe so LACCScorer /
    models.load_scorer consume it through the identical code path."""

    def test_shares_tone_probe_surface(self):
        hidden_dim = 48
        probe = RelevanceConsistencyLoss(hidden_dim=hidden_dim)
        # Attributes LACCScorer / load_scorer read:
        assert hasattr(probe, "tone_classifier")
        assert hasattr(probe, "num_tones")
        assert hasattr(probe, "score_importance")
        assert probe.num_tones == 2
        # First classifier layer is Linear(hidden_dim, ...), as load_scorer's
        # state['tone_classifier.0.weight'].shape[1] dim-check expects.
        assert probe.tone_classifier[0].weight.shape[1] == hidden_dim

    def test_state_dict_keys_match_loader_expectation(self):
        probe = RelevanceConsistencyLoss(hidden_dim=24)
        state = probe.state_dict()
        assert "tone_classifier.0.weight" in state
        assert state["tone_classifier.0.weight"].shape[1] == 24

    def test_same_classifier_shape_as_tone_probe(self):
        # Both build the same Sequential shape (only the final class count differs),
        # so a checkpoint loads into whichever class matches its meta.
        rel = RelevanceConsistencyLoss(hidden_dim=40)
        tone = PhonologicalConsistencyLoss(hidden_dim=40)
        assert type(rel.tone_classifier[0]) is type(tone.tone_classifier[0])
        assert rel.tone_classifier[0].weight.shape == tone.tone_classifier[0].weight.shape
