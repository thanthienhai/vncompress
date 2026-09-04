"""CPU-only tests for the metrics added to match compression-paper reporting
conventions: needle-retrieval recall and the paired-bootstrap significance
used for the probe-vs-rule headline delta.
"""
import pytest

from vncompress.evaluation.metrics import compute_needle_recall
from vncompress.evaluation.significance import paired_bootstrap_delta


class TestNeedleRecall:
    def test_recovered_needle_scores_one(self):
        # The planted answer appears verbatim in a longer generated answer.
        assert compute_needle_recall(
            ["mật khẩu bí mật là VIETCOMPRESS2026"], ["VIETCOMPRESS2026"]
        ) == pytest.approx(1.0)

    def test_missing_needle_scores_zero(self):
        assert compute_needle_recall(["tôi không biết"], ["VIETCOMPRESS2026"]) == pytest.approx(0.0)

    def test_partial_needle_is_credited(self):
        # 1 of 2 reference syllables present -> recall 0.5.
        assert compute_needle_recall(["Hà xxxx"], ["Hà Nội"]) == pytest.approx(0.5)

    def test_recall_ignores_surrounding_filler(self):
        # Unlike F1, extra words in the answer are not penalized for retrieval.
        assert compute_needle_recall(
            ["câu trả lời đầy đủ là Hà Nội thủ đô"], ["Hà Nội"]
        ) == pytest.approx(1.0)

    def test_tone_marks_are_not_collapsed(self):
        assert compute_needle_recall(["ma"], ["mã"]) == pytest.approx(0.0)

    def test_empty_predictions_does_not_crash(self):
        assert compute_needle_recall([], []) == pytest.approx(0.0)


class TestPairedBootstrap:
    def test_clear_improvement_is_significant(self):
        # Probe uniformly beats rule -> delta positive, CI excludes 0.
        probe = [0.8, 0.9, 0.7, 0.85, 0.95, 0.6, 0.88, 0.72]
        rule = [0.5, 0.6, 0.4, 0.55, 0.65, 0.3, 0.58, 0.42]
        res = paired_bootstrap_delta(probe, rule, n_boot=2000, seed=0)
        assert res.mean_delta > 0
        assert res.ci_low > 0
        assert res.significant is True
        assert res.win_rate == pytest.approx(1.0)
        assert res.p_value < 0.05

    def test_no_difference_is_not_significant(self):
        xs = [0.5, 0.6, 0.4, 0.55, 0.65, 0.3]
        res = paired_bootstrap_delta(list(xs), list(xs), n_boot=2000, seed=0)
        assert res.mean_delta == pytest.approx(0.0)
        assert res.significant is False
        assert res.p_value == pytest.approx(1.0)

    def test_pairs_with_none_are_dropped(self):
        probe = [0.8, None, 0.7, 0.9]
        rule = [0.5, 0.6, None, 0.6]
        res = paired_bootstrap_delta(probe, rule, n_boot=1000, seed=0)
        assert res.n == 2  # only sample 0 and 3 survive pairing

    def test_too_few_pairs_returns_none(self):
        assert paired_bootstrap_delta([0.5], [0.4]) is None
        assert paired_bootstrap_delta([None], [0.4]) is None

    def test_is_paired_not_unpaired(self):
        # Same value multisets but paired so that probe==rule on every sample:
        # a paired test must report zero effect even though the arms' sorted
        # values look different in aggregate.
        probe = [0.9, 0.1, 0.9, 0.1]
        rule = [0.9, 0.1, 0.9, 0.1]
        res = paired_bootstrap_delta(probe, rule, n_boot=1000, seed=0)
        assert res.mean_delta == pytest.approx(0.0)
        assert res.significant is False
