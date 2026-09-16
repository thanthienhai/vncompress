"""Tests for scripts/validate_dataset_v2.py -- the v2 acceptance checklist.

This script decides whether a dataset rebuild is accepted, so its failure
modes matter as much as its successes: a check that cannot fail certifies
nothing, and a check that fires on correct data (as the leak check once did on
v1's legitimately shared validation documents) is worse than no check at all.
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SPEC = importlib.util.spec_from_file_location(
    "validate_dataset_v2",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "validate_dataset_v2.py"),
)
validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validator)

PASS, FAIL, MANUAL = validator.PASS, validator.FAIL, validator.MANUAL


def _status(checklist, ref, name_fragment):
    for r_ref, name, status, detail in checklist.rows:
        if r_ref == ref and name_fragment in name:
            return status, detail
    raise AssertionError(f"no check {ref}/{name_fragment} in {[r[:2] for r in checklist.rows]}")


def _comp_row(i, realized=25, target=25, ratio=4.0, **extra):
    row = {
        'id': f'c{i}', 'doc_id': f'doc{i}', 'split': 'train', 'source': 'uvw-2026',
        'domain': 'news', 'context': 'x ' * 100, 'query': 'câu hỏi?',
        'gold_compression': f'bản nén số {i}', 'requested_ratio': ratio,
        'target_tokens': target, 'realized_tokens': realized,
        'answerable_from_compression': True,
    }
    row.update(extra)
    return row


class TestBudgetCompliance:
    def test_passes_when_output_matches_the_budget(self):
        cl = validator.Checklist()
        validator.check_realized_metrics([_comp_row(i) for i in range(20)], cl)
        assert _status(cl, 'P1', 'budget_compliance')[0] == PASS

    def test_fails_when_the_teacher_blows_the_budget(self):
        rows = [_comp_row(i, realized=3) for i in range(20)]  # asked 25, produced 3
        cl = validator.Checklist()
        validator.check_realized_metrics(rows, cl)
        assert _status(cl, 'P1', 'budget_compliance')[0] == FAIL

    def test_unmeasurable_rows_are_not_counted_as_compliant(self):
        # A row with no target cannot be shown to respect one. Counting it as
        # passing would let an unmeasured dataset clear the 95% gate.
        rows = [_comp_row(i, target=None) for i in range(20)]
        cl = validator.Checklist()
        validator.check_realized_metrics(rows, cl)
        status, detail = _status(cl, 'P1', 'budget_compliance')
        assert status == FAIL
        assert 'unmeasurable' in detail

    def test_ratio_is_reported_as_median_not_mean(self):
        # 19 rows on target, one compressing 1000x. The mean would read ~54x
        # and suggest the teacher ignores budgets; the median reads 4x and says
        # what is actually true -- the tail is broken, not the whole run.
        rows = [_comp_row(i, realized=25, target=25) for i in range(19)]
        rows.append(_comp_row(99, realized=1, target=25))
        cl = validator.Checklist()
        validator.check_realized_metrics(rows, cl)
        detail = _status(cl, 'P1', 'requested vs realized')[1]
        assert 'median 4.0x' in detail, detail
        assert 'max 100x' in detail, detail


class TestDuplicates:
    def test_flags_ratio_levels_that_produced_identical_output(self):
        rows = [_comp_row(1, ratio=r) for r in (2.0, 4.0, 8.0)]  # same doc, same text
        cl = validator.Checklist()
        validator.check_duplicates(rows, cl)
        status, detail = _status(cl, 'P2', 'duplicate')
        assert status == FAIL
        assert '2 duplicate' in detail

    def test_passes_when_every_level_differs(self):
        rows = [_comp_row(1, ratio=r, gold_compression=f'text {r}') for r in (2.0, 4.0, 8.0)]
        cl = validator.Checklist()
        validator.check_duplicates(rows, cl)
        assert _status(cl, 'P2', 'duplicate')[0] == PASS


class TestAnswerability:
    def test_fails_when_the_label_is_missing(self):
        rows = [_comp_row(i, answerable_from_compression=None) for i in range(5)]
        cl = validator.Checklist()
        validator.check_answerability(rows, cl)
        assert _status(cl, 'P4', 'answerable')[0] == FAIL

    def test_fails_when_unanswerable_rows_reach_train(self):
        rows = [_comp_row(i) for i in range(5)]
        rows[0]['answerable_from_compression'] = False
        cl = validator.Checklist()
        validator.check_answerability(rows, cl)
        status, detail = _status(cl, 'P4', 'answerable')
        assert status == FAIL
        assert 'still in train' in detail

    def test_passes_when_all_labelled_and_train_is_clean(self):
        cl = validator.Checklist()
        validator.check_answerability([_comp_row(i) for i in range(5)], cl)
        assert _status(cl, 'P4', 'answerable')[0] == PASS


class TestExtractiveFidelity:
    """The check the first v2 run had no equivalent of.

    It passed every SS7 box while shipping compressions that quote 30% of
    their sentences and assert numbers their context never states.
    """

    def test_flags_a_gold_that_is_rewritten_rather_than_quoted(self):
        rows = [_comp_row(i, sentence_extractive_ratio=0.30, unsupported_numbers=[])
                for i in range(5)]
        cl = validator.Checklist()
        validator.check_extractive_fidelity(rows, cl)
        assert _status(cl, 'B1', 'gold is extractive')[0] == FAIL

    def test_passes_a_quoted_gold(self):
        rows = [_comp_row(i, sentence_extractive_ratio=0.95, unsupported_numbers=[])
                for i in range(5)]
        cl = validator.Checklist()
        validator.check_extractive_fidelity(rows, cl)
        assert _status(cl, 'B1', 'gold is extractive')[0] == PASS

    def test_flags_numbers_the_context_never_states(self):
        rows = [_comp_row(i, sentence_extractive_ratio=0.95, unsupported_numbers=[])
                for i in range(5)]
        rows[0]['unsupported_numbers'] = ['1925', '1996']
        cl = validator.Checklist()
        validator.check_extractive_fidelity(rows, cl)
        status, detail = _status(cl, 'B1', 'unsupported numbers')
        assert status == FAIL
        assert '1925' in detail        # names the offender, not just a count

    def test_an_older_build_without_the_fields_is_manual_not_pass(self):
        # Silently passing a dataset that cannot be checked is how the first
        # run cleared the checklist.
        cl = validator.Checklist()
        validator.check_extractive_fidelity([_comp_row(i) for i in range(3)], cl)
        assert _status(cl, 'B1', 'gold is extractive')[0] == MANUAL
        assert _status(cl, 'B1', 'unsupported numbers')[0] == MANUAL


class TestEvalDocumentDiversity:
    """E1: 1,000 rows over 14 documents is not 1,000 independent samples."""

    def _eval_rows(self, per_doc):
        rows = []
        for doc, n in per_doc.items():
            rows += [{'doc_id': doc, 'id': f'{doc}-{i}'} for i in range(n)]
        return rows

    def test_flags_too_few_documents(self):
        rows = self._eval_rows({f'viquad:doc{i}': 70 for i in range(14)})
        cl = validator.Checklist()
        validator.check_eval_document_diversity(rows, cl)
        status, detail = _status(cl, 'E1', 'spans >=')
        assert status == FAIL
        assert '14 document' in detail

    def test_flags_one_document_dominating(self):
        # Enough documents, but one article carries a fifth of the rows.
        per_doc = {f'viquad:doc{i}': 5 for i in range(200)}
        per_doc['viquad:Michael_Jackson'] = 250
        cl = validator.Checklist()
        validator.check_eval_document_diversity(self._eval_rows(per_doc), cl)
        status, detail = _status(cl, 'E1', 'dominates')
        assert status == FAIL
        assert 'Michael_Jackson' in detail

    def test_passes_a_well_spread_test_set(self):
        rows = self._eval_rows({f'viquad:doc{i}': 6 for i in range(180)})
        cl = validator.Checklist()
        validator.check_eval_document_diversity(rows, cl)
        assert _status(cl, 'E1', 'spans >=')[0] == PASS
        assert _status(cl, 'E1', 'dominates')[0] == PASS


class TestLeakCheck:
    def test_does_not_fire_on_documents_shared_with_a_validation_split(self):
        # The v1 regression: 14 documents appear in both `vcc_bench/validation`
        # and `records/validation`. They are the same held-out documents seen
        # through two configs, not a leak.
        configs = {
            'eval': [{'doc_id': 'd1'}, {'doc_id': 'd2'}],
            'qa': [{'doc_id': 'd1', 'split': 'validation'},
                   {'doc_id': 'd9', 'split': 'train'}],
        }
        cl = validator.Checklist()
        validator.check_leakage(configs, cl)
        status, detail = _status(cl, 'E1', 'leak-check')
        assert status == PASS, detail
        assert 'CLEAN' in detail

    def test_fires_on_a_real_train_leak(self):
        configs = {
            'eval': [{'doc_id': 'd1'}],
            'qa': [{'doc_id': 'd1', 'split': 'train'}],
        }
        cl = validator.Checklist()
        validator.check_leakage(configs, cl)
        assert _status(cl, 'E1', 'leak-check')[0] == FAIL

    def test_is_manual_when_there_is_nothing_to_compare(self):
        cl = validator.Checklist()
        validator.check_leakage({'eval': [{'doc_id': 'd1'}], 'qa': []}, cl)
        assert _status(cl, 'E1', 'leak-check')[0] == MANUAL


class TestDomains:
    def test_rejects_wikidata_entity_types(self):
        rows = [{'domain': d} for d in ('người', 'quốc gia có chủ quyền', 'đơn vị phân loại')]
        cl = validator.Checklist()
        validator.check_domains({'qa': rows}, cl)
        assert _status(cl, 'P8', 'controlled vocabulary')[0] == FAIL

    def test_accepts_the_controlled_set(self):
        rows = [{'domain': d} for d in ('news', 'legal', 'medical', 'other')]
        cl = validator.Checklist()
        validator.check_domains({'qa': rows}, cl)
        assert _status(cl, 'P8', 'controlled vocabulary')[0] == PASS


class TestLanguageLabels:
    def test_flags_english_text_labelled_vietnamese(self):
        rows = [{'context': 'The economy is projected to grow steadily this year.',
                 'context_lang': 'vi'}]
        cl = validator.Checklist()
        validator.check_language_labels({'qa': rows}, cl)
        assert _status(cl, 'P9', 'language labels')[0] == FAIL

    def test_passes_when_labels_match(self):
        rows = [{'context': 'Nền kinh tế Việt Nam tăng trưởng đều đặn trong năm nay.',
                 'context_lang': 'vi'},
                {'context': 'The economy grows steadily.', 'context_lang': 'en'}]
        cl = validator.Checklist()
        validator.check_language_labels({'qa': rows}, cl)
        assert _status(cl, 'P9', 'language labels')[0] == PASS

    def test_a_span_too_short_to_judge_is_counted_not_scored(self):
        """The check must not re-run the predicate that produced the label.

        `answer_lang` comes from `is_vietnamese`; re-running it on a
        three-letter span reproduces the same call, so the validator agreed
        with itself on all 6,000 v2 rows while 983 answers were mislabelled.
        Short spans are reported as unjudgeable instead of silently confirmed.
        """
        rows = [{'context': 'Nền kinh tế Việt Nam tăng trưởng đều đặn trong năm nay.',
                 'context_lang': 'vi', 'answer': 'gan', 'answer_lang': 'vi'}]
        cl = validator.Checklist()
        validator.check_language_labels({'qa': rows}, cl)
        status, detail = _status(cl, 'P9', 'language labels')
        assert status == PASS
        assert 'too short to judge' in detail


class TestCorpusPurity:
    def test_flags_raw_corpus_rows_wearing_a_task_label(self):
        cl = validator.Checklist()
        validator.check_corpus_purity([{'task': 'context_compression'}], [_comp_row(1)], cl)
        assert _status(cl, 'P7', 'no task label')[0] == FAIL

    def test_flags_empty_gold_compression(self):
        cl = validator.Checklist()
        validator.check_corpus_purity([], [_comp_row(1, gold_compression='  ')], cl)
        assert _status(cl, 'B3', 'non-empty gold_compression')[0] == FAIL


class TestIndependentTest:
    def test_flags_teacher_output_and_echo_answers(self):
        rows = [{'doc_id': f'd{i}', 'context': 'ngữ cảnh ' * 20,
                 'reference_answer': 'ngữ cảnh ' * 20, 'teacher': 'GLM-5.2'} for i in range(5)]
        cl = validator.Checklist()
        validator.check_independent_test({'eval': rows}, None, cl)
        assert _status(cl, 'E3', 'no teacher output')[0] == FAIL
        assert _status(cl, 'E2', 'not context echoes')[0] == FAIL

    def test_size_gate(self):
        cl = validator.Checklist()
        validator.check_independent_test(
            {'eval': [{'doc_id': 'd'} for _ in range(299)]}, None, cl)
        assert _status(cl, 'E1', '>= 300')[0] == FAIL


class TestVerificationMethodIsNotConflated:
    """The check that exists because filling `verified_by` with an LLM string
    used to flip a line reading "human-verified" to PASS with no person
    involved -- the self-certification SS6's "người kiểm" is there to prevent."""

    def _rows(self, n, method, model='JudgeModel'):
        return [{'id': f'e{i}', 'doc_id': f'd{i}', 'verification_method': method,
                 'verified_by': f'llm:{model}',
                 'verification': {'llm': {'verdict': True, 'model': model}}}
                for i in range(n)]

    def test_llm_only_fails_the_human_check(self):
        cl = validator.Checklist()
        validator._check_verification(self._rows(1000, 'llm'), None, cl)
        assert _status(cl, 'E1', 'verified by a person')[0] == FAIL

    def test_llm_only_still_passes_the_has_a_verdict_check(self):
        # The two are different claims and are reported as two lines.
        cl = validator.Checklist()
        validator._check_verification(self._rows(1000, 'llm'), None, cl)
        assert _status(cl, 'E2', 'carries a verification verdict')[0] == PASS

    def test_unverified_rows_fail_the_verdict_check(self):
        cl = validator.Checklist()
        validator._check_verification(self._rows(10, 'none'), None, cl)
        assert _status(cl, 'E2', 'carries a verification verdict')[0] == FAIL

    def test_a_human_sample_without_calibration_still_fails(self, tmp_path):
        rows = self._rows(850, 'llm') + self._rows(150, 'llm+human')
        (tmp_path / 'provenance').mkdir()
        (tmp_path / 'provenance' / 'verification_report.json').write_text(
            json.dumps({'agreement': {'cohen_kappa': 0.40}}), encoding='utf-8')
        cl = validator.Checklist()
        validator._check_verification(rows, str(tmp_path), cl)
        assert _status(cl, 'E1', 'verified by a person')[0] == FAIL

    def test_enough_human_rows_plus_good_kappa_passes(self, tmp_path):
        rows = self._rows(850, 'llm') + self._rows(150, 'llm+human')
        (tmp_path / 'provenance').mkdir()
        (tmp_path / 'provenance' / 'verification_report.json').write_text(
            json.dumps({'agreement': {'cohen_kappa': 0.82}}), encoding='utf-8')
        cl = validator.Checklist()
        validator._check_verification(rows, str(tmp_path), cl)
        assert _status(cl, 'E1', 'verified by a person')[0] == PASS

    def test_a_judge_that_is_the_teacher_is_flagged(self, tmp_path):
        (tmp_path / 'provenance').mkdir()
        (tmp_path / 'provenance' / 'verification_report.json').write_text(
            json.dumps({'judge_model': 'GLM-5.2', 'teacher_model': 'GLM-5.2',
                        'agreement': {'cohen_kappa': 0.95}}), encoding='utf-8')
        cl = validator.Checklist()
        validator._check_verification(self._rows(300, 'llm+human'), str(tmp_path), cl)
        assert _status(cl, 'E3', 'differs from the teacher')[0] == FAIL

    def test_dry_run_verdicts_never_ship(self):
        rows = [{'id': 'e1', 'verification_method': 'llm',
                 'verification': {'llm': {'verdict': True, 'dry_run': True}}}]
        cl = validator.Checklist()
        validator._check_verification(rows, None, cl)
        assert _status(cl, 'E3', 'no dry-run verdicts')[0] == FAIL


class TestChecklistReport:
    def test_exit_code_is_nonzero_when_anything_failed(self, capsys):
        cl = validator.Checklist()
        cl.add('P1', 'a', PASS, '')
        cl.add('P2', 'b', FAIL, '')
        assert cl.report() == 1

    def test_manual_items_do_not_fail_the_run_but_are_not_counted_as_passing(self, capsys):
        cl = validator.Checklist()
        cl.add('P1', 'a', PASS, '')
        cl.add('E1', 'b', MANUAL, '')
        assert cl.report() == 0
        out = capsys.readouterr().out
        assert '1 pass' in out and '1 manual' in out


class TestCardSourceMatching:
    """P11 compares identifiers, not their typography."""

    CARD = ("# VNCompress-VI v2\n\n"
            "| Nguồn | Upstream |\n"
            "| UIT-ViQuAD 2.0 | taidng/UIT-ViQuAD2.0 |\n"
            "| UVW-2026 | undertheseanlp/UVW-2026 |\n"
            "Tập nào dùng để báo cáo, tập nào chỉ kiểm tra nhanh.\n")

    def _run(self, tmp_path, sources):
        (tmp_path / 'README.md').write_text(self.CARD, encoding='utf-8')
        configs = {'qa': [{'source': s} for s in sources]}
        cl = validator.Checklist()
        validator.check_card(str(tmp_path), configs, cl)
        return _status(cl, 'P11', 'card matches')

    def test_a_differently_typed_identifier_is_not_reported_missing(self, tmp_path):
        # The card says "UIT-ViQuAD 2.0"; the data says `uit-viquad-2.0`.
        status, detail = self._run(tmp_path, ['uit-viquad-2.0', 'uvw-2026'])
        assert status == PASS, detail

    def test_a_genuinely_undocumented_source_still_fails(self, tmp_path):
        # `legal` appears in the data and nowhere in the card, so someone
        # filtering on `source` cannot learn the value from the card.
        status, detail = self._run(tmp_path, ['uvw-2026', 'legal'])
        assert status == FAIL
        assert 'legal' in detail


class TestConfigFileRouting:
    """Each file belongs to exactly one config.

    `qa*.jsonl` used to swallow `qa_synthetic.jsonl`, merging stage 2b's
    teacher-written questions into the human `qa` config -- the exact mixing
    the separate file exists to prevent, happening inside the tool whose job
    is to catch it.
    """

    def _write(self, tmp_path, name, rows):
        with open(tmp_path / name, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_synthetic_qa_is_its_own_config(self, tmp_path):
        self._write(tmp_path, 'qa.jsonl', [{'id': 'h1', 'question_source': 'human'}])
        self._write(tmp_path, 'qa_synthetic.jsonl',
                    [{'id': 's1', 'question_source': 'teacher-generated'}])
        configs = validator.load_v2(str(tmp_path))
        assert [r['id'] for r in configs['qa']] == ['h1']
        assert [r['id'] for r in configs['qa_synthetic']] == ['s1']

    def test_a_sharded_config_still_reads_as_one(self, tmp_path):
        # The behaviour the prefix glob exists for must survive the fix.
        self._write(tmp_path, 'compression.jsonl', [{'id': 'c1'}])
        self._write(tmp_path, 'compression_batch2.jsonl', [{'id': 'c2'}])
        configs = validator.load_v2(str(tmp_path))
        assert {r['id'] for r in configs['compression']} == {'c1', 'c2'}
