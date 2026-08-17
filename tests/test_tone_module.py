"""Tests for vncompress/tone_aware/vietnamese_tones.py (no torch needed)."""
from vncompress.tone_aware.vietnamese_tones import (
    VietnameseToneAnalyzer,
    extract_tone_marks,
    get_tone_analyzer,
    is_vietnamese,
    strip_tone,
)


def test_is_vietnamese_true_for_diacritic_heavy_text():
    assert is_vietnamese("Xin chào các bạn, hôm nay trời đẹp quá") is True


def test_is_vietnamese_false_for_plain_english():
    assert is_vietnamese("Hello world, this is a plain English sentence") is False


def test_is_vietnamese_empty_string_is_false():
    assert is_vietnamese("") is False


def test_strip_tone_removes_diacritics():
    assert strip_tone("à á ả ã ạ") == "a a a a a"
    # strip_tone removes only the 5 tone marks (grave/acute/hook/tilde/dot
    # below), not vowel-quality modifiers like the circumflex on ê -- so
    # "ế" (e + circumflex + acute tone) becomes "ê", not "e".
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
    analyzer = get_tone_analyzer()
    assert isinstance(analyzer, VietnameseToneAnalyzer)


class TestVietnameseToneAnalyzer:
    def test_analyze_tokens_returns_one_result_per_token(self):
        analyzer = VietnameseToneAnalyzer()
        tokens = ["xin", "chào", "các", "bạn"]
        infos = analyzer.analyze_tokens(tokens)
        assert len(infos) == len(tokens)
        for token, info in zip(tokens, infos):
            assert info.token == token

    def test_analyze_tokens_empty_list_does_not_crash(self):
        analyzer = VietnameseToneAnalyzer()
        assert analyzer.analyze_tokens([]) == []

    def test_tone_bearing_token_has_higher_preservation_weight_than_ngang(self):
        analyzer = VietnameseToneAnalyzer()
        infos = analyzer.analyze_tokens(["a", "á"])
        plain, toned = infos
        assert toned.preservation_weight >= plain.preservation_weight

    def test_tones_present_detected_for_marked_syllable(self):
        analyzer = VietnameseToneAnalyzer()
        info = analyzer.analyze_tokens(["mã"])[0]
        assert info.tones_present
