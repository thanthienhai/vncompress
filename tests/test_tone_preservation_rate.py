"""Tests for the canonical Tone Preservation Rate (TPR) metric
(vncompress/tone_aware/vietnamese_tones.py), resolving P1 issue
"research: xac thuc va chuan hoa Tone Preservation Rate".

TPR = |tone-bearing tokens retained| / |tone-bearing tokens|, where a
token is "tone-bearing" iff it carries at least one non-'ngang' tone
mark (TokenToneInfo.tones_present is non-empty). See
docs/tone_preservation_rate.md for the full definition and edge cases.
"""
import pytest

from vncompress.tone_aware.vietnamese_tones import (
    VietnameseToneAnalyzer,
    compute_tone_preservation_rate,
    majority_tone_baseline_rate,
)


@pytest.fixture
def analyzer():
    return VietnameseToneAnalyzer()


class TestComputeToneRreservationRate:
    def test_all_tone_bearing_tokens_retained_gives_tpr_1(self, analyzer):
        # "má", "mà", "mã" each carry a tone mark.
        infos = analyzer.analyze_tokens(["má", "mà", "mã"])
        assert compute_tone_preservation_rate(infos, retained_indices=[0, 1, 2]) == 1.0

    def test_no_tone_bearing_tokens_retained_gives_tpr_0(self, analyzer):
        infos = analyzer.analyze_tokens(["má", "mà", "mã"])
        assert compute_tone_preservation_rate(infos, retained_indices=[]) == 0.0

    def test_partial_retention_is_exact_fraction(self, analyzer):
        infos = analyzer.analyze_tokens(["má", "mà", "mã", "mạ"])
        # Keep 1 of 4 tone-bearing tokens.
        assert compute_tone_preservation_rate(infos, retained_indices=[0]) == pytest.approx(0.25)

    def test_ngang_tokens_excluded_from_denominator(self, analyzer):
        # "ma" (ngang, no diacritic) is not tone-bearing; dropping it
        # must not affect TPR at all.
        infos = analyzer.analyze_tokens(["má", "ma", "ma"])
        # Keep only the tone-bearing token (index 0); the two ngang
        # tokens are dropped but don't count against TPR.
        assert compute_tone_preservation_rate(infos, retained_indices=[0]) == 1.0

    def test_punctuation_and_digits_are_never_tone_bearing(self, analyzer):
        infos = analyzer.analyze_tokens([".", "123", ","])
        assert compute_tone_preservation_rate(infos, retained_indices=[]) == 1.0

    def test_empty_input_returns_1_by_convention(self, analyzer):
        infos = analyzer.analyze_tokens([])
        assert compute_tone_preservation_rate(infos, retained_indices=[]) == 1.0

    def test_all_ngang_sequence_returns_1_regardless_of_retention(self, analyzer):
        infos = analyzer.analyze_tokens(["xin", "chao", "cac", "ban"])  # no diacritics
        assert compute_tone_preservation_rate(infos, retained_indices=[]) == 1.0
        assert compute_tone_preservation_rate(infos, retained_indices=[0, 1, 2, 3]) == 1.0

    def test_accepts_set_or_list_for_retained_indices(self, analyzer):
        infos = analyzer.analyze_tokens(["má", "mà"])
        assert compute_tone_preservation_rate(infos, retained_indices={0}) == \
            compute_tone_preservation_rate(infos, retained_indices=[0])

    def test_is_bounded_in_0_1(self, analyzer):
        infos = analyzer.analyze_tokens(["má", "mà", "mã", "mạ", "mả"])
        for retained in ([], [0], [0, 1], [0, 1, 2, 3, 4]):
            tpr = compute_tone_preservation_rate(infos, retained)
            assert 0.0 <= tpr <= 1.0


class TestMajorityToneBaselineRate:
    def test_all_tone_bearing_text_has_zero_baseline(self, analyzer):
        infos = analyzer.analyze_tokens(["má", "mà", "mã"])
        assert majority_tone_baseline_rate(infos) == 0.0

    def test_all_ngang_text_has_baseline_1(self, analyzer):
        infos = analyzer.analyze_tokens(["xin", "chao", "ban"])
        assert majority_tone_baseline_rate(infos) == 1.0

    def test_mixed_text_baseline_is_non_tone_bearing_fraction(self, analyzer):
        # 1 tone-bearing ("má") out of 4 tokens -> 3/4 non-tone-bearing.
        infos = analyzer.analyze_tokens(["má", "xin", "chao", "ban"])
        assert majority_tone_baseline_rate(infos) == pytest.approx(0.75)

    def test_empty_input_returns_1(self, analyzer):
        assert majority_tone_baseline_rate([]) == 1.0

    def test_no_compressor_always_beats_or_matches_baseline(self, analyzer, vi_text):
        # A NoCompressor (keep-everything) baseline retains every index,
        # so its TPR is always 1.0 -- structurally >= the majority-class
        # floor. This is the sanity check the metric doc calls for: a
        # method that reports a TPR *below* the no-compression ceiling
        # is measurably worse than doing nothing, which the raw TPR
        # number alone wouldn't tell you.
        tokens = vi_text.split()
        infos = analyzer.analyze_tokens(tokens)
        keep_everything_tpr = compute_tone_preservation_rate(infos, retained_indices=range(len(tokens)))
        assert keep_everything_tpr == 1.0
        assert keep_everything_tpr >= majority_tone_baseline_rate(infos)
