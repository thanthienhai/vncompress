"""Tests for vncompress/linguistics.py: tone analysis, Tone Preservation
Rate, and morphology classification. No torch/GPU needed for any of these
(PhonologicalConsistencyLoss etc. are exercised indirectly via
test_compression.py's fake-scorer tests)."""
import pytest

from vncompress.linguistics import (
    VietnameseToneAnalyzer,
    WordClass,
    MorphologyAnalyzer,
    VIETNAMESE_FUNCTION_WORDS,
    compute_tone_preservation_rate,
    extract_tone_marks,
    get_morphology_analyzer,
    get_tone_analyzer,
    is_vietnamese,
    majority_tone_baseline_rate,
    strip_tone,
)


# ============================================================================
# Tone detection / analysis
# ============================================================================


def test_is_vietnamese_true_for_diacritic_heavy_text():
    assert is_vietnamese("Xin chào các bạn, hôm nay trời đẹp quá") is True


def test_is_vietnamese_false_for_plain_english():
    assert is_vietnamese("Hello world, this is a plain English sentence") is False


def test_is_vietnamese_empty_string_is_false():
    assert is_vietnamese("") is False


def test_strip_tone_removes_diacritics():
    assert strip_tone("à á ả ã ạ") == "a a a a a"
    # strip_tone removes only the 5 tone marks, not vowel-quality modifiers
    # like the circumflex on ê -- so "ế" (e + circumflex + acute) becomes
    # "ê", not "e".
    assert strip_tone("Tiếng Việt") == "Tiêng Viêt"


def test_strip_tone_leaves_ascii_untouched():
    assert strip_tone("hello world 123") == "hello world 123"


def test_extract_tone_marks_matches_text_length():
    text = "má mà"
    marks = extract_tone_marks(text)
    assert len(marks) == len(text)
    assert marks[1] == "sắc"  # text[1] == 'á'
    assert "huyền" in marks


def test_get_tone_analyzer_returns_singleton_type():
    assert isinstance(get_tone_analyzer(), VietnameseToneAnalyzer)


def test_nfd_and_nfc_text_are_scored_identically():
    # MANUAL_TONE_MAP holds composed (NFC) characters only, so decomposed
    # text must be NFC-normalized internally (_nfc) or it silently reads as
    # zero tone-bearing tokens and TPR reports 1.0 for text that in fact had
    # every tone dropped.
    import unicodedata

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


class TestVietnameseToneAnalyzer:
    def test_analyze_tokens_returns_one_result_per_token(self):
        analyzer = VietnameseToneAnalyzer()
        tokens = ["xin", "chào", "các", "bạn"]
        infos = analyzer.analyze_tokens(tokens)
        assert len(infos) == len(tokens)
        for token, info in zip(tokens, infos):
            assert info.token == token

    def test_analyze_tokens_empty_list_does_not_crash(self):
        assert VietnameseToneAnalyzer().analyze_tokens([]) == []

    def test_tone_bearing_token_has_higher_preservation_weight_than_ngang(self):
        infos = VietnameseToneAnalyzer().analyze_tokens(["a", "á"])
        plain, toned = infos
        assert toned.preservation_weight >= plain.preservation_weight

    def test_tones_present_detected_for_marked_syllable(self):
        info = VietnameseToneAnalyzer().analyze_tokens(["mã"])[0]
        assert info.tones_present


# ============================================================================
# Tone Preservation Rate
# ============================================================================


@pytest.fixture
def analyzer():
    return VietnameseToneAnalyzer()


class TestComputeTonePreservationRate:
    def test_all_tone_bearing_tokens_retained_gives_tpr_1(self, analyzer):
        infos = analyzer.analyze_tokens(["má", "mà", "mã"])
        assert compute_tone_preservation_rate(infos, retained_indices=[0, 1, 2]) == 1.0

    def test_no_tone_bearing_tokens_retained_gives_tpr_0(self, analyzer):
        infos = analyzer.analyze_tokens(["má", "mà", "mã"])
        assert compute_tone_preservation_rate(infos, retained_indices=[]) == 0.0

    def test_partial_retention_is_exact_fraction(self, analyzer):
        infos = analyzer.analyze_tokens(["má", "mà", "mã", "mạ"])
        assert compute_tone_preservation_rate(infos, retained_indices=[0]) == pytest.approx(0.25)

    def test_ngang_tokens_excluded_from_denominator(self, analyzer):
        infos = analyzer.analyze_tokens(["má", "ma", "ma"])
        assert compute_tone_preservation_rate(infos, retained_indices=[0]) == 1.0

    def test_punctuation_and_digits_are_never_tone_bearing(self, analyzer):
        infos = analyzer.analyze_tokens([".", "123", ","])
        assert compute_tone_preservation_rate(infos, retained_indices=[]) == 1.0

    def test_empty_input_returns_1_by_convention(self, analyzer):
        assert compute_tone_preservation_rate(analyzer.analyze_tokens([]), retained_indices=[]) == 1.0

    def test_all_ngang_sequence_returns_1_regardless_of_retention(self, analyzer):
        infos = analyzer.analyze_tokens(["xin", "chao", "cac", "ban"])
        assert compute_tone_preservation_rate(infos, retained_indices=[]) == 1.0
        assert compute_tone_preservation_rate(infos, retained_indices=[0, 1, 2, 3]) == 1.0

    def test_accepts_set_or_list_for_retained_indices(self, analyzer):
        infos = analyzer.analyze_tokens(["má", "mà"])
        assert compute_tone_preservation_rate(infos, retained_indices={0}) == \
            compute_tone_preservation_rate(infos, retained_indices=[0])

    def test_is_bounded_in_0_1(self, analyzer):
        infos = analyzer.analyze_tokens(["má", "mà", "mã", "mạ", "mả"])
        for retained in ([], [0], [0, 1], [0, 1, 2, 3, 4]):
            assert 0.0 <= compute_tone_preservation_rate(infos, retained) <= 1.0


class TestMajorityToneBaselineRate:
    def test_all_tone_bearing_text_has_zero_baseline(self, analyzer):
        assert majority_tone_baseline_rate(analyzer.analyze_tokens(["má", "mà", "mã"])) == 0.0

    def test_all_ngang_text_has_baseline_1(self, analyzer):
        assert majority_tone_baseline_rate(analyzer.analyze_tokens(["xin", "chao", "ban"])) == 1.0

    def test_mixed_text_baseline_is_non_tone_bearing_fraction(self, analyzer):
        infos = analyzer.analyze_tokens(["má", "xin", "chao", "ban"])
        assert majority_tone_baseline_rate(infos) == pytest.approx(0.75)

    def test_empty_input_returns_1(self):
        assert majority_tone_baseline_rate([]) == 1.0

    def test_no_compressor_always_beats_or_matches_baseline(self, analyzer, vi_text):
        # A keep-everything baseline retains every index, so its TPR is
        # always 1.0 -- structurally >= the majority-class floor.
        tokens = vi_text.split()
        infos = analyzer.analyze_tokens(tokens)
        keep_everything_tpr = compute_tone_preservation_rate(infos, retained_indices=range(len(tokens)))
        assert keep_everything_tpr == 1.0
        assert keep_everything_tpr >= majority_tone_baseline_rate(infos)


# ============================================================================
# Morphology classification
# ============================================================================


def test_get_morphology_analyzer_returns_analyzer_instance():
    assert isinstance(get_morphology_analyzer(), MorphologyAnalyzer)


class TestClassifyWord:
    def test_known_function_word_classified_as_func(self):
        analyzer = MorphologyAnalyzer(use_word_segmentation=False, use_pos_tagger=False)
        sample = next(iter(VIETNAMESE_FUNCTION_WORDS))
        assert analyzer.classify_word(sample) == WordClass.FUNC

    def test_unknown_word_defaults_to_content(self):
        analyzer = MorphologyAnalyzer(use_word_segmentation=False, use_pos_tagger=False)
        assert analyzer.classify_word("blorptastic") == WordClass.CONTENT

    def test_underscore_joined_token_is_compound(self):
        analyzer = MorphologyAnalyzer(use_word_segmentation=False, use_pos_tagger=False)
        assert analyzer.classify_word("máy_tính") in (WordClass.COMPOUND, WordClass.SINO)

    def test_classification_is_case_insensitive(self):
        analyzer = MorphologyAnalyzer(use_word_segmentation=False, use_pos_tagger=False)
        sample = next(iter(VIETNAMESE_FUNCTION_WORDS))
        assert analyzer.classify_word(sample.upper()) == WordClass.FUNC

    @pytest.mark.parametrize("token", ["", "  ", ",", "123", "3.5%"])
    def test_punctuation_and_numbers_are_not_content_words(self, token):
        analyzer = MorphologyAnalyzer(use_word_segmentation=False, use_pos_tagger=False)
        assert analyzer.classify_word(token) is WordClass.OTHER

    @pytest.mark.parametrize("word", ["vàng", "tình", "năng", "thái", "linh"])
    def test_standalone_second_syllables_are_content_not_redup(self, word):
        # Was: classified REDUP purely for being some reduplication's 2nd
        # syllable, compressed at r_redup=0.5 instead of r_content=0.85.
        assert get_morphology_analyzer().classify_word(word) is WordClass.CONTENT


class TestClassifyBatch:
    def test_returns_one_result_per_token(self):
        analyzer = MorphologyAnalyzer(use_word_segmentation=False, use_pos_tagger=False)
        tokens = ["chúng", "tôi", "đã", "đi", "học"]
        infos = analyzer.classify_batch(tokens)
        assert len(infos) == len(tokens)
        for token, info in zip(tokens, infos):
            assert info.token == token

    def test_empty_batch_does_not_crash(self):
        assert MorphologyAnalyzer(use_word_segmentation=False, use_pos_tagger=False).classify_batch([]) == []

    def test_function_word_flags_are_consistent_with_class(self):
        analyzer = MorphologyAnalyzer(use_word_segmentation=False, use_pos_tagger=False)
        sample = next(iter(VIETNAMESE_FUNCTION_WORDS))
        info = analyzer.classify_batch([sample])[0]
        assert info.word_class == WordClass.FUNC
        assert info.is_function_word is True
        assert info.is_content_word is False


class TestReduplicationDetection:
    @pytest.mark.parametrize("a,b", [("ma", "mà"), ("bàn", "bán"), ("đá", "đã"), ("má", "mạ")])
    def test_tone_minimal_pairs_are_not_reduplication(self, a, b):
        # Was: _phonetic_similarity stripped tones before comparing, scoring
        # every minimal pair 1.0 -- deleting the tone contrast this project
        # exists to preserve.
        assert get_morphology_analyzer()._phonetic_similarity(a, b) < 0.6

    @pytest.mark.parametrize("a,b", [("nhè", "nhẹ"), ("đo", "đỏ"), ("trăng", "trắng")])
    def test_same_register_reduplication_is_still_detected(self, a, b):
        # Vietnamese tone harmony keeps both tones of a real reduplicated
        # pair in one register -- the fix must not throw this out too.
        assert get_morphology_analyzer()._phonetic_similarity(a, b) >= 0.6

    def test_unrelated_tokens_are_not_paired_as_reduplication(self):
        # Was: `right in self.redup_pairs` matched any token preceding a
        # known second syllable, pairing "tôi"/"vàng" and "mua"/"vàng".
        assert get_morphology_analyzer().find_reduplicative_pairs(
            ["tôi", "mua", "vàng", "ở", "chợ"]
        ) == []
