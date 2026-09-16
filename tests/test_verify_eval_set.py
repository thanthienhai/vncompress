"""Tests for scripts/verify_eval_set.py -- stage 4, eval verification.

The property under test throughout is that an LLM verdict can never be
mistaken for a human one. Everything else here (kappa, stratification, refusal
parsing) exists to make the human sample small enough to be affordable while
still saying something about the 850 rows nobody looked at.
"""
import importlib.util
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SPEC = importlib.util.spec_from_file_location(
    "verify_eval_set",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "verify_eval_set.py"),
)
verify = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify)

from vncompress.dataset_schema import (
    MIN_HUMAN_VERIFIED_SAMPLES,
    MIN_JUDGE_AGREEMENT_KAPPA,
    VERIFICATION_METHODS,
)


class TestRiskScore:
    def test_full_overlap_is_zero_risk(self):
        assert verify.risk_score("Thủ đô của Việt Nam là gì?",
                                 "Thủ đô của Việt Nam là Hà Nội.") == 0.0

    def test_the_grandfather_case_scores_high(self):
        # The real row from the eval set: the answer string is present and the
        # compression still cannot answer the question, because the link the
        # question asks about ("ông nội") never appears.
        risk = verify.risk_score("Ông nội của Thái Tông Trần Cảnh có tên là gì?",
                                 "Trần Lý sinh ra Trần Thừa.")
        assert risk > 0.5

    def test_empty_question_is_maximum_risk_not_a_crash(self):
        assert verify.risk_score("", "bất kỳ văn bản nào") == 1.0
        assert verify.risk_score(None, None) == 1.0

    def test_stopwords_do_not_earn_overlap_credit(self):
        # A compression made only of function words shares words with the
        # question and still contains nothing. If stopwords counted, this would
        # score as low risk.
        assert verify.risk_score("Ai là người của tổ chức này?", "là của này") == 1.0


class TestParseJudgeAnswer:
    def test_matching_answer_is_answerable(self):
        verdict, f1, _ = verify.parse_judge_answer("Hà Nội", "Hà Nội")
        assert verdict is True and f1 == pytest.approx(1.0)

    def test_explicit_refusal_is_unanswerable(self):
        for refusal in ("KHÔNG ĐỦ", "không đủ thông tin", "KHÔNG CÓ"):
            verdict, f1, _ = verify.parse_judge_answer(refusal, "Hà Nội")
            assert verdict is False, refusal

    def test_refusal_is_not_scored_as_a_wrong_answer(self):
        # Both come back False, but the refusal must not be handed an F1 that
        # makes it look like an attempted answer -- the two failures get
        # different follow-up, so they cannot be merged.
        refused, refused_f1, _ = verify.parse_judge_answer("KHÔNG ĐỦ", "Hà Nội")
        wrong, wrong_f1, _ = verify.parse_judge_answer("Sài Gòn", "Hà Nội")
        assert refused is wrong is False
        assert refused_f1 == 0.0 and wrong_f1 == 0.0

    def test_empty_response_is_unanswerable(self):
        assert verify.parse_judge_answer("", "Hà Nội")[0] is False
        assert verify.parse_judge_answer(None, "Hà Nội")[0] is False

    def test_partial_overlap_below_threshold_fails(self):
        verdict, _, _ = verify.parse_judge_answer(
            "một thành phố lớn ở miền bắc có tên nào đó", "Hà Nội")
        assert verdict is False


class TestCohenKappa:
    def test_perfect_agreement(self):
        kappa, agreement, _ = verify.cohen_kappa([(True, True), (False, False)] * 10)
        assert kappa == pytest.approx(1.0) and agreement == pytest.approx(1.0)

    def test_a_rater_that_always_says_yes_scores_zero_not_ninety_percent(self):
        # THE reason kappa is used instead of raw agreement. The judge is right
        # 90% of the time here purely because 90% of rows are answerable, and
        # raw agreement would certify it.
        pairs = [(True, True)] * 90 + [(True, False)] * 10
        kappa, agreement, _ = verify.cohen_kappa(pairs)
        assert agreement == pytest.approx(0.90)
        assert kappa is None or kappa < 0.1

    def test_realistic_disagreement_lands_between(self):
        pairs = [(True, True)] * 60 + [(False, False)] * 30 + \
                [(True, False)] * 5 + [(False, True)] * 5
        kappa, _, _ = verify.cohen_kappa(pairs)
        assert 0.6 < kappa < 0.9

    def test_no_pairs_is_undefined_not_zero(self):
        # Reporting 0.0 would read as "the judge disagrees", when the truth is
        # "nothing was measured".
        assert verify.cohen_kappa([]) == (None, None, {})

    def test_confusion_matrix_is_reported(self):
        _, _, confusion = verify.cohen_kappa([(True, False), (True, False), (False, False)])
        assert confusion["True|False"] == 2 and confusion["False|False"] == 1


class TestApplyVerdicts:
    def _llm(self, verdict=True):
        return {'verdict': verdict, 'model': 'JudgeModel', 'f1': 0.9}

    def _human(self, verdict=True):
        return {'verdict': verdict, 'annotator': 'reviewer_1'}

    def test_llm_only_is_never_labelled_human(self):
        row = verify.apply_verdicts({}, llm=self._llm())
        assert row['verification_method'] == 'llm'
        assert 'human' not in row['verified_by']

    def test_human_review_upgrades_the_method(self):
        row = verify.apply_verdicts({}, llm=self._llm())
        row = verify.apply_verdicts(row, human=self._human())
        assert row['verification_method'] == 'llm+human'
        assert 'llm:JudgeModel' in row['verified_by']
        assert 'human:reviewer_1' in row['verified_by']

    def test_the_human_verdict_overrides_the_judge(self):
        row = verify.apply_verdicts({}, llm=self._llm(verdict=True))
        assert row['answerable_from_compression'] is True
        row = verify.apply_verdicts(row, human=self._human(verdict=False))
        assert row['answerable_from_compression'] is False
        # ...and the judge's own verdict is still on the row, because the
        # disagreement is the measurement.
        assert row['verification']['llm']['verdict'] is True

    def test_method_is_always_a_declared_value(self):
        for llm, human in ((None, None), (self._llm(), None),
                           (None, self._human()), (self._llm(), self._human())):
            row = verify.apply_verdicts({}, llm=llm, human=human)
            assert row['verification_method'] in VERIFICATION_METHODS


class TestStratifiedSample:
    def _rows(self):
        rows = []
        for i in range(400):
            answerable = i % 4 != 0           # 75% pass, an imbalanced pool
            rows.append({
                'id': f'r{i}',
                # A prime modulus, so document does NOT correlate with verdict.
                # The correlated version of this fixture is worth keeping too --
                # see test_minority_survives_correlation_with_document.
                'doc_id': f'doc{i % 7}',
                'query': 'Thủ đô của Việt Nam là gì?',
                'gold_compression': ('Thủ đô của Việt Nam là Hà Nội.' if i % 3 == 0
                                     else 'Một câu hoàn toàn khác.'),
                'answer': 'Hà Nội',
                'verification': {'llm': {'verdict': answerable}},
            })
        return rows

    def test_returns_the_requested_size(self):
        assert len(verify.stratified_sample(self._rows(), 150)) == 150

    def test_both_verdicts_are_represented(self):
        picked = verify.stratified_sample(self._rows(), 150)
        verdicts = {r['verification']['llm']['verdict'] for r in picked}
        assert verdicts == {True, False}

    def test_the_minority_verdict_is_not_swamped(self):
        # A uniform sample of this pool would be ~75% pass. The point of
        # stratifying is that the cases the judge is most likely to have got
        # wrong get reviewed, so they must not appear in proportion.
        picked = verify.stratified_sample(self._rows(), 150)
        rejected = sum(1 for r in picked if not r['verification']['llm']['verdict'])
        assert rejected / len(picked) > 0.30

    def test_documents_are_spread(self):
        picked = verify.stratified_sample(self._rows(), 150)
        assert len({r['doc_id'] for r in picked}) == 7

    def test_minority_survives_correlation_with_document(self):
        # The case that broke the first implementation: every rejection lives
        # in 2 of 8 documents, so document strata outnumber verdict strata and
        # a flat stratification dilutes rejections straight back to their base
        # rate. Kappa is estimated from the off-diagonal, so this matters.
        rows = []
        for i in range(400):
            answerable = i % 4 != 0
            rows.append({
                'id': f'c{i}', 'doc_id': f'doc{i % 8}',
                'query': 'Thủ đô của Việt Nam là gì?',
                'gold_compression': ('Thủ đô của Việt Nam là Hà Nội.' if i % 3 == 0
                                     else 'Một câu hoàn toàn khác.'),
                'answer': 'Hà Nội',
                'verification': {'llm': {'verdict': answerable}},
            })
        picked = verify.stratified_sample(rows, 150)
        rejected = sum(1 for r in picked if not r['verification']['llm']['verdict'])
        assert rejected / len(picked) > 0.40

    def test_is_deterministic_for_a_seed(self):
        rows = self._rows()
        first = [r['id'] for r in verify.stratified_sample(rows, 50, seed=1)]
        second = [r['id'] for r in verify.stratified_sample(rows, 50, seed=1)]
        assert first == second

    def test_asking_for_more_than_exists_does_not_hang(self):
        picked = verify.stratified_sample(self._rows()[:10], 500)
        assert len(picked) == 10


class TestJudgeIsolation:
    def test_the_judge_never_receives_the_context(self):
        # If the judge could see the context it would answer from the context
        # and certify a compression that does not contain the answer at all --
        # which is the entire failure being screened for.
        prompt = verify.JUDGE_PROMPT.format(
            gold_compression="BẢN NÉN NGẮN", query="CÂU HỎI")
        assert 'BẢN NÉN NGẮN' in prompt and 'CÂU HỎI' in prompt
        assert '{context}' not in verify.JUDGE_PROMPT
        assert 'context' not in verify.JUDGE_PROMPT

    def test_default_judge_is_not_the_teacher(self):
        assert verify.DEFAULT_JUDGE_MODEL != verify.TEACHER_MODEL


class TestCalibrationGate:
    """The rule the whole design exists to enforce."""

    def _report_inputs(self, human_reviewed, kappa):
        return (human_reviewed >= MIN_HUMAN_VERIFIED_SAMPLES
                and kappa is not None and kappa >= MIN_JUDGE_AGREEMENT_KAPPA)

    def test_a_large_sample_with_poor_agreement_is_not_calibrated(self):
        assert self._report_inputs(1000, 0.40) is False

    def test_perfect_agreement_on_too_few_rows_is_not_calibrated(self):
        assert self._report_inputs(20, 0.99) is False

    def test_both_conditions_together_calibrate(self):
        assert self._report_inputs(MIN_HUMAN_VERIFIED_SAMPLES,
                                   MIN_JUDGE_AGREEMENT_KAPPA) is True

    def test_undefined_kappa_never_calibrates(self):
        assert self._report_inputs(1000, None) is False
