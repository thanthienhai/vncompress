"""Tests for vncompress/dataset_build.py -- the v2 rebuild transforms.

Each rule is tested on its own. A dataset builder exercised only end-to-end is
a builder whose individual rules are never checked, and a wrong rule here
silently ships in 140k rows.
"""
import pytest

from vncompress.dataset_build import (
    assign_split,
    budget_compliance,
    document_key,
    normalize_compression,
    count_tokens,
    detect_lang,
    extractive_ratio,
    is_long_document,
    map_domain,
    numbers_preserved,
    qa_task_name,
    realized_ratio,
    sanitize_identifiers,
    sentence_extractive_ratio,
    unsupported_numbers,
)
from vncompress.dataset_schema import CONTROLLED_DOMAINS


class TestMapDomain:
    @pytest.mark.parametrize("wikidata_type,expected", [
        ("Hiến pháp", "legal"),
        ("war crimes trial", "legal"),
        ("non-aggression pact", "legal"),
        ("bệnh truyền nhiễm", "medical"),
        ("triều đại phong kiến trong lịch sử Trung Quốc", "history"),
        ("Chiến dịch quân sự", "history"),
        ("naval battle", "history"),
        ("nguyên tố hóa học", "science"),
        ("đơn vị phân loại", "science"),
        ("khái niệm toán học", "science"),
        ("môn học", "education"),
        ("tỉnh của Việt Nam", "admin"),
        ("huyện của Việt Nam", "admin"),
        ("quốc gia có chủ quyền", "admin"),
    ])
    def test_maps_known_types(self, wikidata_type, expected):
        assert map_domain(wikidata_type) == expected

    @pytest.mark.parametrize("wikidata_type", [
        "người", "general", "loạt manga", "biến thể cờ vua", "câu lạc bộ bóng đá",
    ])
    def test_entity_types_without_a_task_domain_become_other(self, wikidata_type):
        # The honest answer for general encyclopedia content. Inventing a
        # domain here would recreate the inflated diversity P8 objects to.
        assert map_domain(wikidata_type) == "other"

    def test_missing_type_is_other_not_an_error(self):
        assert map_domain(None) == "other"
        assert map_domain("") == "other"

    def test_only_ever_returns_controlled_values(self):
        for value in ("Führerpartei", "viand", "aircraft family", "", None, "ngẫu nhiên xyz"):
            assert map_domain(value) in CONTROLLED_DOMAINS


class TestDetectLang:
    def test_vietnamese(self):
        assert detect_lang("Nền kinh tế Việt Nam tăng trưởng đều đặn trong năm nay.") == "vi"

    def test_english(self):
        # v1 stamped language='vi' on rows like this one.
        assert detect_lang("The economy is projected to grow steadily this year.") == "en"

    def test_empty_text_returns_none_rather_than_guessing(self):
        assert detect_lang("") is None
        assert detect_lang(None) is None
        assert detect_lang("   ") is None

    def test_a_number_has_no_language(self):
        # v2 labelled 144 bare-number answers `en`. "1827" is not English.
        assert detect_lang("1827") is None
        assert detect_lang("25.000", fallback="vi") is None

    def test_a_span_too_short_to_judge_inherits_the_context(self):
        # The real v2 rows: "gan" (cholesterol), "Amsterdam" (Den Haag). Both
        # sit inside Vietnamese contexts and both came back `en`.
        assert detect_lang("gan", fallback="vi") == "vi"
        assert detect_lang("Amsterdam", fallback="vi") == "vi"
        # With nothing to inherit, the honest answer is that we do not know.
        assert detect_lang("gan") is None

    def test_a_long_enough_foreign_span_still_reports_itself(self):
        # The floor must not turn every answer into the context's language:
        # genuinely non-Vietnamese answers exist in the same dataset.
        assert detect_lang("Sanctae Romanae Ecclesiae cardinalis", fallback="vi") == "en"
        assert detect_lang("Chinese Nationalist Party", fallback="vi") == "en"

    def test_a_long_enough_vietnamese_span_is_unaffected(self):
        assert detect_lang("rất căng thẳng và phức tạp", fallback="en") == "vi"


class TestSentenceExtractiveRatio:
    """SS3.1: is the gold quoted from the context, or rewritten from it?"""

    CONTEXT = ("Hà Nội là thủ đô của Việt Nam. Thành phố nằm bên bờ sông Hồng. "
               "Dân số khoảng tám triệu người.")

    def test_quoted_sentences_score_one(self):
        assert sentence_extractive_ratio(
            self.CONTEXT, "Hà Nội là thủ đô của Việt Nam.") == pytest.approx(1.0)

    def test_a_fluent_rewrite_scores_zero_where_token_overlap_does_not(self):
        # The v2 failure in miniature: every word is from the context, no
        # sentence is. Token overlap calls this extractive; this does not.
        rewrite = "Thủ đô của Việt Nam là Hà Nội, nằm bên bờ sông Hồng."
        assert extractive_ratio(self.CONTEXT, rewrite) > 0.7
        assert sentence_extractive_ratio(self.CONTEXT, rewrite) == 0.0

    def test_partially_quoted_scores_between(self):
        gold = "Hà Nội là thủ đô của Việt Nam. Dân số hiện nay rất đông đúc."
        assert sentence_extractive_ratio(self.CONTEXT, gold) == pytest.approx(0.5)

    def test_no_scoreable_sentence_returns_none(self):
        assert sentence_extractive_ratio(self.CONTEXT, "") is None
        assert sentence_extractive_ratio(self.CONTEXT, "Ừ.") is None


class TestUnsupportedNumbers:
    """A number the compression asserts and the context never states."""

    def test_clean_compression_reports_nothing(self):
        assert unsupported_numbers("Sinh năm 1906, mất năm 2000.",
                                   "Sinh năm 1906.") == []

    def test_a_fabricated_year_is_caught(self):
        # viquad:Phạm_Văn_Đồng: the compression dated events to 1925 and 1996
        # from a context containing neither.
        assert unsupported_numbers("Ông sinh năm 1906.",
                                   "Năm 1925 ông tham gia phong trào.") == ['1925']

    def test_punctuation_after_a_number_is_not_part_of_it(self):
        # The greedy `\d[\d.,]*` read "1925," and "1925" as different numbers,
        # under-reporting on 500 of the 973 shipped rows.
        assert unsupported_numbers("Năm 1925, ông đi Quảng Châu.",
                                   "Ông đi Quảng Châu năm 1925.") == []
        assert numbers_preserved("Năm 1925, ông đi.", "Năm 1925 ông đi.") == pytest.approx(1.0)

    def test_a_decimal_stays_one_number(self):
        assert unsupported_numbers("Tỷ lệ là 3.14 phần trăm.", "Tỷ lệ 3.14.") == []

    def test_no_numbers_to_check_returns_none(self):
        assert unsupported_numbers("Ngữ cảnh không có số.", "Bản nén không số.") is None


class TestBudgetCompliance:
    def test_two_sided_rule(self):
        assert budget_compliance(100, 100) is True
        assert budget_compliance(110, 100) is True     # +10%, inside tolerance
        assert budget_compliance(90, 100) is True      # -10%, inside tolerance
        assert budget_compliance(130, 100) is False    # too many tokens
        assert budget_compliance(50, 100) is False     # over-compressed

    def test_over_compression_is_not_compliant(self):
        # v1's stored flag was `realized <= target`, which called this True and
        # is why it reported 88.6% compliance where the spec gives 19.8%.
        assert budget_compliance(3, 61) is False

    def test_unmeasurable_returns_none_not_false(self):
        # None means "cannot be judged"; the caller must not fold that into a
        # compliance rate as though it had been measured and passed.
        assert budget_compliance(50, None) is None
        assert budget_compliance(None, 100) is None
        assert budget_compliance(50, 0) is None


class TestCompressionMetrics:
    def test_realized_ratio_is_measured_from_the_texts(self):
        assert realized_ratio("a b c d e f g h", "a b") == 4.0

    def test_realized_ratio_none_when_either_side_empty(self):
        assert realized_ratio("", "a b") is None
        assert realized_ratio("a b c", "") is None

    def test_numbers_preserved(self):
        assert numbers_preserved("giá 100 và 200 đồng", "giá 100 đồng") == 0.5
        assert numbers_preserved("giá 100 và 200", "100 200") == 1.0

    def test_numbers_preserved_none_when_context_has_no_numbers(self):
        assert numbers_preserved("không có số nào", "không có") is None

    def test_extractive_ratio(self):
        assert extractive_ratio("một hai ba bốn", "một hai") == 1.0
        assert extractive_ratio("một hai ba bốn", "một năm") == 0.5

    def test_count_tokens(self):
        assert count_tokens("một hai  ba") == 3
        assert count_tokens(None) == 0


class TestLongDocument:
    def test_short_span_extraction_is_not_a_long_document(self):
        # v1 called a ~480-character span task "long_document_qa".
        assert is_long_document("từ " * 150) is False
        assert qa_task_name("từ " * 150) == "short_context_extractive_qa"

    def test_genuinely_long_context_is_flagged(self):
        assert is_long_document("từ " * 4000) is True
        assert qa_task_name("từ " * 4000) == "long_document_qa"


class TestSanitizeIdentifiers:
    def test_rewrites_a_realistic_bank_account(self):
        # The exact needle docs/dataset_review.md flagged (P10).
        text, changed = sanitize_identifiers(
            "Số tài khoản là 1903666888666, Vietcombank chi nhánh Hà Nội")
        assert changed is True
        assert "1903666888666" not in text

    def test_rewrites_a_vietnamese_phone_number(self):
        text, changed = sanitize_identifiers("Liên hệ 0987654321 để biết thêm")
        assert changed is True
        assert "0987654321" not in text

    def test_leaves_ordinary_numbers_alone(self):
        # Years, prices and percentages are content, not identifiers; rewriting
        # them would corrupt the text the dataset is about.
        text, changed = sanitize_identifiers("Năm 2026 doanh thu đạt 5000 tỷ, tăng 12,5%")
        assert changed is False
        assert text == "Năm 2026 doanh thu đạt 5000 tỷ, tăng 12,5%"

    def test_empty_input(self):
        assert sanitize_identifiers(None) == (None, False)
        assert sanitize_identifiers("") == ("", False)


class TestDocumentKey:
    """E1: the same article must not get two splits because two corpora named it
    differently. This is the bug that put 168 of 1,000 test rows onto articles
    sitting in the training corpus, under a leak report reading CLEAN."""

    def test_the_same_article_from_two_sources_is_one_document(self):
        assert document_key('uvw:Đế_quốc_La_Mã') == document_key('viquad:Đế_quốc_La_Mã')

    def test_that_makes_the_split_agree_across_sources(self):
        # The actual guarantee. Without it these are two independent coin flips.
        for title in ('Đế_quốc_La_Mã', 'Họ_Đậu', 'Chiến_tranh_Vùng_Vịnh', 'Iran'):
            assert assign_split(document_key(f'uvw:{title}')) == \
                   assign_split(document_key(f'viquad:{title}'))

    def test_different_articles_stay_different(self):
        assert document_key('uvw:Hà_Nội') != document_key('uvw:Hải_Phòng')

    def test_spacing_and_case_do_not_create_a_second_document(self):
        assert document_key('viquad:Thành phố Hồ Chí Minh') == \
               document_key('viquad:thành_phố_hồ_chí_minh')

    def test_multi_segment_ids_keep_everything_after_the_source(self):
        # `legal:<law>:<chapter>` -- only the source prefix goes.
        assert document_key('legal:civil_2015:ch1') == 'civil_2015:ch1'
        assert document_key('legal:civil_2015:ch1') != document_key('legal:civil_2015:ch2')

    def test_missing_or_unprefixed_ids_do_not_raise(self):
        assert document_key(None) == ''
        assert document_key('') == ''
        assert document_key('noprefix') == 'noprefix'


class TestNormalizeCompression:
    """B2: duplicate detection must not be defeated by whitespace."""

    def test_whitespace_differences_collapse(self):
        assert normalize_compression("Hà  Nội\nlà thủ đô") == \
               normalize_compression(" Hà Nội là  thủ đô ")

    def test_different_texts_stay_different(self):
        assert normalize_compression("Hà Nội") != normalize_compression("Hải Phòng")

    def test_empty_input_is_an_empty_key(self):
        assert normalize_compression(None) == '' and normalize_compression('  ') == ''
