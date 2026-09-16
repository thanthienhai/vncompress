"""Tests for scripts/generate_qa_from_corpus.py -- stage 2b.

Two properties carry the whole script, and both are about what it REFUSES:

  * a generated answer that is not verbatim in the context never becomes a row,
    so `answer_span` is computed rather than trusted -- the same invariant the
    7,000 human rows hold;
  * a test document never gets a generated question, which is what keeps B5/E3
    ("tập test độc lập KHÔNG dùng output teacher") true by construction.

The rest is parsing whatever the model wrapped its JSON in.
"""
import importlib.util
import json
import os
import sys
import threading
from collections import Counter

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SPEC = importlib.util.spec_from_file_location(
    "generate_qa_from_corpus",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "generate_qa_from_corpus.py"),
)
qagen = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qagen)

from vncompress.dataset_build import assign_split, document_key

CONTEXT = ("Hà Nội là thủ đô của nước Cộng hòa Xã hội chủ nghĩa Việt Nam. "
           "Thành phố nằm bên bờ sông Hồng ở miền Bắc. "
           "Dân số Hà Nội vào khoảng tám triệu người.")


def _document(doc_id='uvw:Hà_Nội', split='train', context=CONTEXT):
    return {'doc_id': doc_id, 'split': split, 'context': context, 'num_passages': 3,
            'title': 'Hà Nội', 'source': 'uvw-2026',
            'source_url': 'https://vi.wikipedia.org/wiki/Hà_Nội',
            'license': 'CC-BY-SA-4.0', 'wikidata_type': 'thành phố thuộc tỉnh của Việt Nam'}


class _Teacher:
    model = 'GLM-5.2'
    base_url = 'http://x'
    temperature = 0.3
    dry_run = True

    def __init__(self, replies=('',)):
        self.replies = list(replies)
        self.prompts = []

    def chat(self, system, user, api_key=None, max_tokens=None, stop=None):
        self.prompts.append(user)
        return self.replies.pop(0) if self.replies else ''


class TestLocateAnswer:
    def test_finds_an_exact_span(self):
        assert qagen.locate_answer(CONTEXT, "sông Hồng") == (CONTEXT.index("sông Hồng"),
                                                             CONTEXT.index("sông Hồng") + 9)

    def test_tolerates_collapsed_whitespace(self):
        # The model copied the span but flattened a line break inside it. That
        # is still a copy; rejecting it would discard good rows over layout.
        context = "Thành phố nằm bên bờ\nsông Hồng ở miền Bắc."
        span = qagen.locate_answer(context, "bên bờ sông Hồng")
        assert span is not None
        assert context[span[0]:span[1]] == "bên bờ\nsông Hồng"

    def test_rejects_a_paraphrase(self):
        # The failure this check exists for: fluent, correct, and not a span.
        assert qagen.locate_answer(CONTEXT, "thủ đô Việt Nam là Hà Nội") is None

    def test_rejects_an_empty_answer(self):
        assert qagen.locate_answer(CONTEXT, "") is None


class TestParseQaJson:
    def test_plain_array(self):
        assert qagen.parse_qa_json('[{"question": "q?", "answer": "a"}]') == \
            [{'question': 'q?', 'answer': 'a'}]

    def test_markdown_fenced(self):
        raw = '```json\n[{"question": "q?", "answer": "a"}]\n```'
        assert qagen.parse_qa_json(raw) == [{'question': 'q?', 'answer': 'a'}]

    def test_prefixed_with_prose(self):
        raw = 'Đây là các câu hỏi:\n[{"question": "q?", "answer": "a"}]\nHết.'
        assert qagen.parse_qa_json(raw) == [{'question': 'q?', 'answer': 'a'}]

    def test_unparseable_returns_empty_rather_than_guessing(self):
        # Half-parsed output would put malformed rows in the dataset; dropping
        # the document is visible in the run stats instead.
        assert qagen.parse_qa_json('[{"question": "q?", "answer":') == []
        assert qagen.parse_qa_json('không có JSON ở đây') == []
        assert qagen.parse_qa_json('') == []

    def test_non_object_entries_are_discarded(self):
        assert qagen.parse_qa_json('["chuỗi", {"question": "q?", "answer": "a"}]') == \
            [{'question': 'q?', 'answer': 'a'}]


class TestBuildQaRows:
    def test_a_verbatim_answer_becomes_a_span_verified_row(self):
        rows, dropped = qagen.build_qa_rows(
            _document(), [{'question': 'Thủ đô ở đâu?', 'answer': 'sông Hồng'}], _Teacher())
        assert len(rows) == 1 and not dropped
        row = rows[0]
        start, end = row['answer_span']
        assert row['context'][start:end] == row['answer']
        assert row['question_source'] == 'teacher-generated'
        assert row['split'] == 'train'

    def test_a_paraphrased_answer_is_dropped_with_a_reason(self):
        rows, dropped = qagen.build_qa_rows(
            _document(), [{'question': 'Thủ đô?', 'answer': 'Hà Nội là thủ đô của Việt Nam'}],
            _Teacher())
        assert rows == []
        assert dropped['answer not verbatim in context'] == 1

    def test_an_answer_that_is_really_a_paragraph_is_dropped(self):
        long_answer = ' '.join(['từ'] * (qagen.MAX_ANSWER_WORDS + 1))
        rows, dropped = qagen.build_qa_rows(
            _document(context=long_answer), [{'question': 'q?', 'answer': long_answer}],
            _Teacher())
        assert rows == []
        assert dropped['answer too long to be a span'] == 1

    def test_duplicate_questions_and_spans_collapse(self):
        items = [{'question': 'Sông nào?', 'answer': 'sông Hồng'},
                 {'question': 'sông nào?', 'answer': 'sông Hồng'},      # same question
                 {'question': 'Khác hẳn?', 'answer': 'sông Hồng'}]      # same span
        rows, dropped = qagen.build_qa_rows(_document(), items, _Teacher())
        assert len(rows) == 1
        assert dropped['duplicate question'] == 1
        assert dropped['duplicate answer span'] == 1

    def test_the_answer_text_comes_from_the_context_not_the_model(self):
        # Whitespace-tolerant match: the stored answer must be what the context
        # actually says, so the span and the text can never disagree.
        context = "Thành phố nằm bên bờ\nsông Hồng ở miền Bắc."
        rows, _ = qagen.build_qa_rows(
            _document(context=context),
            [{'question': 'Ở đâu?', 'answer': 'bên bờ sông Hồng'}], _Teacher())
        start, end = rows[0]['answer_span']
        assert rows[0]['answer'] == context[start:end] == "bên bờ\nsông Hồng"


class TestTestSplitIsRefused:
    def test_the_worker_refuses_a_test_document(self):
        """B5/E3 enforced where it would be violated, not only where filtered."""
        worker = qagen.make_worker(_Teacher(), 4, Counter(), threading.Lock())
        with pytest.raises(RuntimeError, match='test document'):
            worker(_document(split='test'), 'key', None)

    def test_load_corpus_documents_skips_test_documents(self, tmp_path):
        # Pick two real doc ids that the deterministic split sends different ways.
        train_id = next(f'uvw:Doc_{i}' for i in range(500)
                        if assign_split(document_key(f'uvw:Doc_{i}'), 42) == 'train')
        test_id = next(f'uvw:Doc_{i}' for i in range(500)
                       if assign_split(document_key(f'uvw:Doc_{i}'), 42) == 'test')
        corpus = tmp_path / 'corpus.jsonl'
        with open(corpus, 'w', encoding='utf-8') as f:
            for doc_id in (train_id, test_id):
                for part in range(4):
                    f.write(json.dumps({
                        'doc_id': doc_id, 'text': f'Đoạn {part}. ' + 'nội dung dài. ' * 60,
                        'source': 'uvw-2026', 'split': 'x'}, ensure_ascii=False) + "\n")

        documents, skipped = qagen.load_corpus_documents(str(tmp_path), seed=42)
        ids = {d['doc_id'] for d in documents}
        assert train_id in ids
        assert test_id not in ids
        assert skipped['test document (human QA only)'] == 1

    def test_documents_too_short_to_question_are_skipped(self, tmp_path):
        train_id = next(f'uvw:Doc_{i}' for i in range(500)
                        if assign_split(document_key(f'uvw:Doc_{i}'), 42) == 'train')
        corpus = tmp_path / 'corpus.jsonl'
        corpus.write_text(json.dumps(
            {'doc_id': train_id, 'text': 'Quá ngắn.', 'source': 'uvw-2026',
             'split': 'train'}, ensure_ascii=False) + "\n", encoding='utf-8')
        documents, skipped = qagen.load_corpus_documents(str(tmp_path), seed=42)
        assert documents == []
        assert skipped['document too short'] == 1
