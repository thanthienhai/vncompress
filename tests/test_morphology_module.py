"""Tests for vncompress/morphology/merge_policy.py (no torch needed)."""
from vncompress.morphology.merge_policy import (
    VIETNAMESE_FUNCTION_WORDS,
    MorphologyAnalyzer,
    WordClass,
    get_morphology_analyzer,
)


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
        result = analyzer.classify_word("máy_tính")
        assert result in (WordClass.COMPOUND, WordClass.SINO)

    def test_classification_is_case_insensitive(self):
        analyzer = MorphologyAnalyzer(use_word_segmentation=False, use_pos_tagger=False)
        sample = next(iter(VIETNAMESE_FUNCTION_WORDS))
        assert analyzer.classify_word(sample.upper()) == WordClass.FUNC


class TestClassifyBatch:
    def test_returns_one_result_per_token(self):
        analyzer = MorphologyAnalyzer(use_word_segmentation=False, use_pos_tagger=False)
        tokens = ["chúng", "tôi", "đã", "đi", "học"]
        infos = analyzer.classify_batch(tokens)
        assert len(infos) == len(tokens)
        for token, info in zip(tokens, infos):
            assert info.token == token

    def test_empty_batch_does_not_crash(self):
        analyzer = MorphologyAnalyzer(use_word_segmentation=False, use_pos_tagger=False)
        assert analyzer.classify_batch([]) == []

    def test_function_word_flags_are_consistent_with_class(self):
        analyzer = MorphologyAnalyzer(use_word_segmentation=False, use_pos_tagger=False)
        sample = next(iter(VIETNAMESE_FUNCTION_WORDS))
        info = analyzer.classify_batch([sample])[0]
        assert info.word_class == WordClass.FUNC
        assert info.is_function_word is True
        assert info.is_content_word is False
