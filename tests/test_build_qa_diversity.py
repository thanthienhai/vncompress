"""Tests for the test-set diversity controls in scripts/build_vncompress_vi_v2.py.

The first v2 build produced 1,000 test rows over 14 documents, four of which
supplied 534 of them. Every acceptance box passed, because every box counted
rows. These tests cover the two things that count documents instead: the
per-document cap, and the reporting that makes the concentration visible
whether or not the cap is set.
"""
import importlib.util
import json
import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SPEC = importlib.util.spec_from_file_location(
    "build_vncompress_vi_v2",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "build_vncompress_vi_v2.py"),
)
build = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build)

from vncompress.dataset_build import assign_split, document_key


def _viquad_rows(titles_and_counts, passage_chars=600):
    """Minimal ViQuAD-shaped rows: one article, many questions, real offsets."""
    rows = []
    for title, n in titles_and_counts.items():
        # One passage per article is enough; the answer must really be in it.
        answer = "Hà Nội"
        passage = ("Thành phố này là một đô thị lớn của Việt Nam. " * 6
                   + f"Thủ đô là {answer}. "
                   + "Dân cư đông đúc và kinh tế phát triển nhanh. " * 6)
        for i in range(n):
            rows.append({
                'title': title,
                'doc_id': f'viquad:{title}',
                'source': 'uit-viquad-2.0',
                'source_id': f'{title}_{i:04d}',
                'context': passage,
                'question': f'Thủ đô là thành phố nào? ({i})',
                'answer': answer,
                'answer_start': passage.index(answer),
                'is_impossible': False,
                'source_url': f'https://vi.wikipedia.org/wiki/{title}',
                'license': 'CC-BY-SA-4.0',
            })
    return rows


def _test_titles(candidates, seed=42):
    """The subset of `candidates` the deterministic split sends to test."""
    return [t for t in candidates
            if assign_split(document_key(f'viquad:{t}'), seed) == 'test']


def _titles_until(n_test_docs, seed=42, pool=400):
    """Enough article titles that at least `n_test_docs` of them land in test."""
    found, i = [], 0
    while len(found) < n_test_docs and i < pool:
        title = f'Article_{i}'
        if assign_split(document_key(f'viquad:{title}'), seed) == 'test':
            found.append(title)
        i += 1
    return found


class TestEvalPerDocumentCap:
    def test_unset_reproduces_the_concentrated_behaviour(self):
        """The default must not silently change an existing build."""
        titles = _titles_until(3)
        rows = _viquad_rows({t: 60 for t in titles})
        _, eval_rows, _ = build.build_qa(rows, {}, 42, 24000, 6000, 100)
        assert len(eval_rows) == 100
        # Round-robin spreads them, but nothing stops one document supplying
        # a third of a 3-document test set.
        assert max(Counter(r['doc_id'] for r in eval_rows).values()) > 10

    def test_the_cap_limits_questions_per_document(self):
        titles = _titles_until(3)
        rows = _viquad_rows({t: 60 for t in titles})
        _, eval_rows, skipped = build.build_qa(
            rows, {}, 42, 24000, 6000, 100, max_eval_per_doc=5)
        per_doc = Counter(r['doc_id'] for r in eval_rows)
        assert per_doc and max(per_doc.values()) <= 5
        assert skipped['eval per-document cap'] > 0

    def test_the_cap_trades_rows_for_independence_and_says_so(self):
        """Fewer rows is the POINT, and the skip counter records the price.

        The two runs stop for different reasons -- uncapped fills `max_eval`,
        capped runs out of per-document allowance -- so the price is read off
        the counter, not from the difference between them.
        """
        titles = _titles_until(3)
        available = 3 * 60
        rows = _viquad_rows({t: 60 for t in titles})
        _, uncapped, _ = build.build_qa(rows, {}, 42, 24000, 6000, 100)
        _, capped, skipped = build.build_qa(
            rows, {}, 42, 24000, 6000, 100, max_eval_per_doc=5)
        assert len(capped) < len(uncapped)
        assert len(capped) == 5 * len({r['doc_id'] for r in capped})
        # Everything the rule turned away is attributed to the rule, so the
        # build log says why the set is smaller rather than just that it is.
        assert skipped['eval per-document cap'] == available - len(capped)

    def test_the_cap_does_not_touch_the_train_split(self):
        titles = _titles_until(3)
        train_titles = [t for t in (f'Train_{i}' for i in range(40))
                        if assign_split(document_key(f'viquad:{t}'), 42) != 'test'][:3]
        rows = _viquad_rows({t: 20 for t in titles + train_titles})
        qa_rows, _, _ = build.build_qa(
            rows, {}, 42, 24000, 6000, 100, max_eval_per_doc=2)
        per_doc = Counter(r['doc_id'] for r in qa_rows)
        assert per_doc and max(per_doc.values()) == 20


class TestAnswerLanguageOnShortSpans:
    def test_a_short_vietnamese_answer_is_not_labelled_english(self):
        """983 of 6,000 v2 answers said `en` inside a Vietnamese context."""
        titles = _titles_until(1)
        rows = _viquad_rows({titles[0]: 3})
        _, eval_rows, _ = build.build_qa(rows, {}, 42, 24000, 6000, 100)
        assert eval_rows
        for row in eval_rows:
            assert row['context_lang'] == 'vi'
            # "Hà Nội" is six characters -- far too short for the diacritic
            # ratio to be evidence, so it inherits the context.
            assert row['answer_lang'] == 'vi'


class TestExternalBenchmarkHoldout:
    """A document an external benchmark evaluates on must not enter TRAIN.

    The measured case: `viquad:Hà_Nội` in this build's train split versus 32
    of 414 VCC-Bench v2 samples for the same article.
    """

    def _bench(self, tmp_path, titles):
        path = tmp_path / 'vcc_bench_v2.json'
        path.write_text(json.dumps(
            {'samples': [{'sample_id': f's{i}', 'title': t}
                         for i, t in enumerate(titles)]}), encoding='utf-8')
        return str(path)

    def test_reads_titles_past_the_source_prefix(self, tmp_path):
        # VCC-Bench says "Hà Nội"; the dataset says `viquad:Hà_Nội`.
        keys = build.external_eval_documents([self._bench(tmp_path, ['Hà Nội'])])
        assert document_key('viquad:Hà_Nội') in keys

    def test_holdout_keeps_the_document_out_of_train(self, tmp_path):
        train_titles = [t for t in (f'Doc_{i}' for i in range(60))
                        if assign_split(document_key(f'viquad:{t}'), 42) != 'test'][:3]
        rows = _viquad_rows({t: 4 for t in train_titles})
        bench = self._bench(tmp_path, [train_titles[0]])
        holdout = build.external_eval_documents([bench])

        qa_rows, _, skipped = build.build_qa(
            rows, {}, 42, 24000, 6000, 100, holdout_docs=holdout)
        kept = {r['doc_id'] for r in qa_rows}
        assert f'viquad:{train_titles[0]}' not in kept
        assert f'viquad:{train_titles[1]}' in kept
        assert skipped['external benchmark holdout'] == 4

    def test_without_the_flag_nothing_is_removed(self, tmp_path):
        """Reporting is default; removing rows stays an explicit choice."""
        train_titles = [t for t in (f'Doc_{i}' for i in range(60))
                        if assign_split(document_key(f'viquad:{t}'), 42) != 'test'][:2]
        rows = _viquad_rows({t: 4 for t in train_titles})
        qa_rows, _, skipped = build.build_qa(rows, {}, 42, 24000, 6000, 100)
        assert len(qa_rows) == 8
        assert skipped['external benchmark holdout'] == 0

    def test_a_missing_benchmark_file_is_an_error_not_a_silent_pass(self, tmp_path):
        # Silently holding out nothing because of a typo'd path is the failure
        # this whole mechanism exists to prevent.
        with pytest.raises(SystemExit):
            build.external_eval_documents([str(tmp_path / 'nope.json')])
