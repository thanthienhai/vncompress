"""CPU-only tests for the SLM-scorer compressor wiring and the token-F1 metric.

Deliberately does not load any model: these cover the parts that broke or
could silently break -- registry/taxonomy registration, the "no scorer
configured" error path, and F1 arithmetic on Vietnamese syllables.
"""
import json

import pytest

from vncompress.evaluation.metrics import compute_exact_match, compute_token_f1
from vncompress.evaluation.method_taxonomy import MethodCategory, categorize


class TestTokenF1:
    def test_identical_answers_score_one(self):
        assert compute_token_f1(["Phạm Văn Đồng"], ["Phạm Văn Đồng"]) == pytest.approx(1.0)

    def test_disjoint_answers_score_zero(self):
        assert compute_token_f1(["Hà Nội"], ["Sài Gòn"]) == pytest.approx(0.0)

    def test_partial_overlap_is_credited_where_exact_match_is_not(self):
        # 5 predicted syllables, 3 reference, 3 overlap -> P=3/5, R=1, F1=0.75.
        pred, ref = ["Thủ tướng Phạm Văn Đồng"], ["Phạm Văn Đồng"]
        assert compute_token_f1(pred, ref) == pytest.approx(0.75)
        # The whole point of adding F1: EM calls this a total failure.
        assert compute_exact_match(pred, ref) == pytest.approx(0.0)

    def test_case_and_punctuation_are_normalized_away(self):
        assert compute_token_f1(["hà nội."], ["Hà Nội"]) == pytest.approx(1.0)

    def test_both_empty_agree_but_one_empty_does_not(self):
        assert compute_token_f1([""], [""]) == pytest.approx(1.0)
        assert compute_token_f1([""], ["Hà Nội"]) == pytest.approx(0.0)

    def test_empty_input_does_not_crash(self):
        assert compute_token_f1([], []) == pytest.approx(0.0)

    def test_diacritics_are_not_stripped(self):
        # Tone marks distinguish words; normalization must not collapse them.
        assert compute_token_f1(["ma"], ["mã"]) == pytest.approx(0.0)


class TestSLMScorerRegistration:
    def test_both_slm_methods_are_registered(self):
        from vncompress.compressors import COMPRESSOR_REGISTRY

        assert "slm_scorer" in COMPRESSOR_REGISTRY
        assert "slm_scorer_base" in COMPRESSOR_REGISTRY

    def test_taxonomy_separates_the_proposed_method_from_its_ablation(self):
        assert categorize("slm_scorer") is MethodCategory.PROPOSED
        assert categorize("slm_scorer_base") is MethodCategory.ABLATION

    def test_missing_scorer_raises_an_actionable_error(self):
        from vncompress.compressors.slm_scorer import SLMScorerCompressor

        with pytest.raises(ValueError, match="scorer_adapter_dir"):
            SLMScorerCompressor(tokenizer=None, scorer=None, scorer_adapter_dir=None)

    def test_scorer_kwargs_are_not_leaked_to_other_compressors(self, tokenizer):
        # A single --scorer-adapter-dir must be passable for every method, so
        # non-SLM compressors have to ignore it rather than crash.
        from vncompress.compressors import create_compressor

        comp = create_compressor(
            "none", tokenizer, None, config=None, device="cpu",
            scorer_adapter_dir="trained_slm/final",
        )
        assert comp.get_name() == "NoCompression"


class TestVietnameseRougeTokenization:
    """Regression guard for the tone-blind ROUGE bug.

    rouge_score's default tokenizer strips every character outside [a-z0-9],
    which deletes Vietnamese tone marks and scored 'bạn' against 'bàn' as a
    perfect 1.0. See VietnameseRougeTokenizer.
    """

    def test_tone_marks_survive_tokenization(self):
        from vncompress.evaluation.metrics import VietnameseRougeTokenizer

        tok = VietnameseRougeTokenizer()
        assert tok.tokenize("bàn") == ["bàn"]
        assert tok.tokenize("bạn") == ["bạn"]
        assert tok.tokenize("Hà Nội là thủ đô") == ["hà", "nội", "là", "thủ", "đô"]

    def test_different_tones_are_not_scored_as_identical(self):
        from vncompress.evaluation.metrics import compute_rouge_l

        # The exact regression: these differ only by tone mark and must not
        # score 1.0. With the library default tokenizer this returned 1.0000.
        score = compute_rouge_l(["bàn của tôi"], ["bạn của tôi"])["rougeL_f1"]
        assert score < 1.0

    def test_identical_vietnamese_text_still_scores_one(self):
        from vncompress.evaluation.metrics import compute_rouge_l

        text = "Hà Nội là thủ đô của Việt Nam"
        assert compute_rouge_l([text], [text])["rougeL_f1"] == pytest.approx(1.0)

    def test_rouge_and_token_f1_count_the_same_units(self):
        from vncompress.evaluation.metrics import compute_rouge_l, compute_token_f1

        # Consistency matters: a reader comparing the two columns should not
        # be seeing two different tokenizations.
        pred, ref = ["Hà Nội là thủ đô"], ["Sài Gòn là thành phố"]
        assert compute_rouge_l(pred, ref)["rougeL_f1"] == pytest.approx(
            compute_token_f1(pred, ref)
        )


class TestRegressionGuards:
    """One test per bug found in the audit, so none of them come back."""

    def test_high_ratio_still_compresses(self, tokenizer, vi_ids):
        # Was: mid_budget==0 took an `else` branch returning every index, so
        # asking for MORE compression returned NONE (CR 1.0 at ratio 16+).
        from vncompress.compressors import create_compressor

        for ratio in (8.0, 16.0, 32.0):
            c = create_compressor("combined", tokenizer, None, config=None, device="cpu")
            c.config.target_ratio = ratio
            result = c.compress(list(vi_ids))
            assert result.compressed_length < len(vi_ids)
            assert result.compression_ratio > 1.0

    def test_output_is_a_subsequence_never_longer_than_input(self, tokenizer, vi_ids):
        # Was: unclamped `range(n-k, n)` produced negative indices that Python
        # wrapped to the end, duplicating and reordering tokens (n=3, k=5 on
        # "A B C" returned "B C A B C").
        from vncompress.compressors import create_compressor

        for n_small, k in ((1, 2), (2, 2), (3, 5)):
            c = create_compressor("random", tokenizer, None, config=None, device="cpu")
            c.config.target_ratio, c.config.keep_boundary_tokens = 4.0, k
            sub = list(vi_ids[:n_small])
            out = c.compress(sub).compressed_ids
            assert len(out) <= len(sub)
            indices = c.select_with_boundary([0.0] * n_small, n_small)
            assert indices == sorted(set(indices))
            assert all(0 <= i < n_small for i in indices)

    def test_random_compressor_is_reproducible(self, tokenizer, vi_ids):
        # Was: global `random`, so an unrelated random.seed() call elsewhere
        # changed the benchmark's headline baseline.
        from vncompress.compressors import create_compressor

        outs = []
        for _ in range(2):
            c = create_compressor("random", tokenizer, None, config=None, device="cpu")
            c.config.target_ratio = 4.0
            outs.append(c.compress(list(vi_ids)).compressed_ids)
        assert outs[0] == outs[1]

    def test_aggregate_never_emits_nan(self):
        # Was: np.mean([]) -> NaN, and json.dump writes a bare `NaN` token,
        # which is not valid JSON (RFC 8259).
        from vncompress.evaluation.metrics import CompressionMetrics, VCCBench

        failed = [CompressionMetrics(compression_ratio=2.0) for _ in range(3)]
        agg = VCCBench()._aggregate_metrics(failed)
        assert agg["mean_rouge_l_f1"] is None
        assert agg["mean_bleu"] is None
        json.dumps(agg, allow_nan=False)  # raises if any NaN slipped through

    def test_harmonized_score_survives_negative_efficiency(self):
        # Was: 2*q*e/(q+e) with e<0 returned -18,000,000, which sorted to the
        # top of a table labelled "higher is better".
        from vncompress.evaluation.metrics import VCCBench

        results = {"m": {"t": {"ratio_2.0": {
            "mean_quality_score": 0.3, "mean_efficiency_score": -0.3, "num_samples": 10,
        }}}}
        assert VCCBench()._compute_summary(results)["m"]["harmonized_score"] >= 0.0

    def test_summary_weights_tasks_by_sample_count(self):
        # Was: plain mean over cells, so an 8-sample task counted as much as a
        # 160-sample one in the ranking table.
        from vncompress.evaluation.metrics import VCCBench

        results = {"m": {
            "big": {"ratio_2.0": {"mean_quality_score": 0.9, "mean_efficiency_score": 0.5,
                                  "num_samples": 1000}},
            "small": {"ratio_2.0": {"mean_quality_score": 0.1, "mean_efficiency_score": 0.5,
                                    "num_samples": 2}},
        }}
        avg_q = VCCBench()._compute_summary(results)["m"]["avg_quality"]
        assert avg_q == pytest.approx((0.9 * 1000 + 0.1 * 2) / 1002)
        assert avg_q > 0.85  # not the unweighted 0.5


class TestVietnameseLinguisticsRegressions:
    def test_nfd_and_nfc_text_are_scored_identically(self):
        # Was: MANUAL_TONE_MAP holds composed characters only, so decomposed
        # text had zero tone-bearing tokens and TPR reported 1.0 (perfect
        # preservation) for text where every tone had in fact been dropped.
        import unicodedata

        from vncompress.tone_aware import (
            compute_tone_preservation_rate,
            get_tone_analyzer,
            is_vietnamese,
        )

        analyzer = get_tone_analyzer()
        words = ["Tiếng", "Việt", "rất", "đẹp"]
        out = []
        for form in ("NFC", "NFD"):
            toks = [unicodedata.normalize(form, w) for w in words]
            infos = analyzer.analyze_tokens(toks)
            out.append((
                sum(1 for i in infos if i.tones_present),
                compute_tone_preservation_rate(infos, []),
                is_vietnamese(unicodedata.normalize(form, "Tiếng Việt")),
            ))
        assert out[0] == out[1]
        assert out[0][1] == pytest.approx(0.0)  # dropped everything -> 0, not 1
        assert out[0][2] is True

    @pytest.mark.parametrize("a,b", [("ma", "mà"), ("bàn", "bán"), ("đá", "đã"), ("má", "mạ")])
    def test_tone_minimal_pairs_are_not_reduplication(self, a, b):
        # Was: _phonetic_similarity stripped tones before comparing, scoring
        # every minimal pair 1.0. The right-hand token then got a 0.1
        # multiplier -- deleting the tone contrast the project exists to keep.
        from vncompress.morphology.merge_policy import get_morphology_analyzer

        assert get_morphology_analyzer()._phonetic_similarity(a, b) < 0.6

    @pytest.mark.parametrize("a,b", [("nhè", "nhẹ"), ("đo", "đỏ"), ("trăng", "trắng")])
    def test_same_register_reduplication_is_still_detected(self, a, b):
        # The fix must not throw out real tone-alternating reduplication:
        # Vietnamese tone harmony keeps both tones in one register.
        from vncompress.morphology.merge_policy import get_morphology_analyzer

        assert get_morphology_analyzer()._phonetic_similarity(a, b) >= 0.6

    def test_unrelated_tokens_are_not_paired_as_reduplication(self):
        # Was: `right in self.redup_pairs` matched any token preceding a known
        # second syllable, pairing "tôi"/"vàng" and "mua"/"vàng" and dropping
        # the content word "vàng".
        from vncompress.morphology.merge_policy import get_morphology_analyzer

        assert get_morphology_analyzer().find_reduplicative_pairs(
            ["tôi", "mua", "vàng", "ở", "chợ"]
        ) == []

    @pytest.mark.parametrize("word", ["vàng", "tình", "năng", "thái", "linh"])
    def test_standalone_second_syllables_are_content_not_redup(self, word):
        # Was: classified REDUP and compressed at r_redup=0.5 instead of
        # r_content=0.85, purely for being some reduplication's 2nd syllable.
        from vncompress.morphology.merge_policy import WordClass, get_morphology_analyzer

        assert get_morphology_analyzer().classify_word(word) is WordClass.CONTENT

    @pytest.mark.parametrize("token", ["", "  ", ",", "123", "3.5%"])
    def test_punctuation_and_numbers_are_not_content_words(self, token):
        # Was: fell through to CONTENT and were protected like real vocabulary.
        from vncompress.morphology.merge_policy import WordClass, get_morphology_analyzer

        assert get_morphology_analyzer().classify_word(token) is WordClass.OTHER
