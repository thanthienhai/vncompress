"""Tests for vncompress/dataset.py -- the canonical schema, the §6.1
deterministic checks, and above all the §9 invariant: a document must never
straddle the train/eval boundary.

These run without torch/transformers on purpose: the split has to stay
verifiable in a bare checkout, and the no-leakage assertion is the one thing
in the data pipeline that silently invalidates every number downstream when it
breaks.
"""
import json
import os

import pytest

from vncompress.dataset import (
    KIND_BENCHMARK,
    KIND_CORPUS,
    Record,
    check_split_leakage,
    group_by_document,
    normalize_benchmark,
    normalize_corpus,
    normalize_file,
    read_jsonl,
    split_by_document,
    verify_records,
    write_jsonl,
    write_vcc_bench_json,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VCC_BENCH_V1 = os.path.join(REPO, 'data', 'benchmark', 'vcc_bench_v1.json')


def _para(topic_id, index):
    # Distinct text per paragraph: identical contexts are themselves a leak
    # signal (check_split_leakage looks at content, not just document ids),
    # so a fixture that repeats one string would test the wrong thing.
    text = f'Đoạn {index} của tài liệu {topic_id}. ' + ('Nội dung tiếng Việt. ' * 15)
    return {'source': 'uvw-2026', 'topic_id': topic_id, 'paragraph_index': index,
            'title': f'Bài {topic_id}', 'text': text}


def _corpus(n_docs, paras_per_doc):
    return [_para(f'doc{d}', i) for d in range(n_docs) for i in range(paras_per_doc)]


# ============================================================================
# Normalization (§5)
# ============================================================================


def test_corpus_paragraphs_of_one_article_share_a_document():
    records = normalize_corpus(_corpus(n_docs=3, paras_per_doc=5))
    assert len(records) == 15
    assert len({r.doc_key for r in records}) == 3
    assert all(r.kind == KIND_CORPUS for r in records)


def test_corpus_drops_paragraphs_at_or_below_the_min_length():
    records = normalize_corpus([{'topic_id': 'a', 'text': 'quá ngắn'}], min_chars=200)
    assert records == []


def test_corpus_falls_back_to_a_content_hash_when_no_document_id_exists():
    records = normalize_corpus([{'text': 'Xin chào. ' * 40}])
    assert len(records) == 1
    assert records[0].doc_id.startswith('sha:')


def test_benchmark_query_variants_of_one_paragraph_share_a_document():
    samples = [{'sample_id': f'doc_qa_0007_q{i}', 'task': 'long_document_qa',
                'context': 'ctx ' * 60, 'query': f'q{i}', 'reference_answer': 'a'} for i in range(3)]
    records = normalize_benchmark(samples)
    assert {r.doc_id for r in records} == {'doc_qa_0007'}


def test_benchmark_groups_law_chapters_and_wikipedia_paragraphs_by_source_document():
    legal = normalize_benchmark([
        {'sample_id': 'legal_0000', 'law_id': 'hien_phap_2013', 'task': 'long_document_qa',
         'context': 'x' * 300, 'query': 'q', 'reference_answer': 'a'},
        {'sample_id': 'legal_0001', 'law_id': 'hien_phap_2013', 'task': 'long_document_qa',
         'context': 'y' * 300, 'query': 'q', 'reference_answer': 'a'},
    ])
    assert {r.doc_id for r in legal} == {'law:hien_phap_2013'}

    wiki = normalize_benchmark([
        {'sample_id': 'doc_qa_0000_q0', 'source': 'wikipedia', 'domain': 'Ha_Noi',
         'task': 'long_document_qa', 'context': 'x' * 300, 'query': 'q', 'reference_answer': 'a'},
        {'sample_id': 'doc_qa_0031_q2', 'source': 'wikipedia', 'domain': 'Ha_Noi',
         'task': 'long_document_qa', 'context': 'y' * 300, 'query': 'q', 'reference_answer': 'a'},
    ])
    assert {r.doc_id for r in wiki} == {'wiki:Ha_Noi'}


def test_cross_lingual_direction_variants_share_a_document():
    samples = [{'sample_id': f'cross_0000_{d}', 'task': 'cross_lingual', 'context': 'c' * 300,
                'query': 'q', 'reference_answer': 'a'}
               for d in ('vi_to_vi', 'vi_to_en', 'en_to_en')]
    assert {r.doc_id for r in normalize_benchmark(samples)} == {'cross_0000'}


# ============================================================================
# The §9 invariant
# ============================================================================


def test_split_never_puts_a_document_on_both_sides():
    records = normalize_corpus(_corpus(n_docs=40, paras_per_doc=6))
    train, eval_, _ = split_by_document(records, eval_ratio=0.1, seed=42)
    assert check_split_leakage(train, eval_) == []
    assert {r.doc_key for r in train}.isdisjoint({r.doc_key for r in eval_})


def test_split_keeps_every_record_exactly_once():
    records = normalize_corpus(_corpus(n_docs=25, paras_per_doc=4))
    train, eval_, _ = split_by_document(records, eval_ratio=0.1)
    assert len(train) + len(eval_) == len(records)
    assert {r.id for r in train} | {r.id for r in eval_} == {r.id for r in records}


def test_split_honours_the_requested_ratio_within_one_document():
    records = normalize_corpus(_corpus(n_docs=100, paras_per_doc=5))
    train, eval_, manifest = split_by_document(records, eval_ratio=0.1)
    # 500 records, documents of 5 -> an exact 10% is reachable.
    assert len(eval_) == 50
    assert manifest['eval_ratio_realized'] == pytest.approx(0.1, abs=0.02)
    assert len(train) == 450


def test_split_is_deterministic_and_seed_dependent():
    records = normalize_corpus(_corpus(n_docs=30, paras_per_doc=3))
    a_train, a_eval, _ = split_by_document(records, seed=42)
    b_train, b_eval, _ = split_by_document(records, seed=42)
    assert [r.id for r in a_eval] == [r.id for r in b_eval]
    assert [r.id for r in a_train] == [r.id for r in b_train]
    _, c_eval, _ = split_by_document(records, seed=7)
    assert {r.id for r in c_eval} != {r.id for r in a_eval}


def test_split_stratifies_per_source():
    records = (normalize_corpus(_corpus(n_docs=50, paras_per_doc=4))
               + normalize_corpus([{'source': 'vietnamese-poetry', 'topic_id': f'p{i}',
                                    'text': 'Thơ ca. ' * 40} for i in range(50)]))
    _, eval_, manifest = split_by_document(records, eval_ratio=0.1)
    sources = {r.source for r in eval_}
    assert sources == {'uvw-2026', 'vietnamese-poetry'}, 'every source must reach the eval side'
    for stats in manifest['strata'].values():
        assert stats['eval_records'] > 0


def test_eval_only_source_sends_a_whole_source_to_eval():
    records = (normalize_corpus(_corpus(n_docs=20, paras_per_doc=4))
               + normalize_benchmark([{'sample_id': f'needle_{i:04d}', 'task': 'needle_in_haystack',
                                       'context': 'c' * 400, 'query': 'q', 'reference_answer': 'a'}
                                      for i in range(9)], default_source='vcc_bench'))
    train, eval_, manifest = split_by_document(records, eval_ratio=0.1, eval_only_sources=['vcc_bench'])
    assert not [r for r in train if r.source == 'vcc_bench']
    assert len([r for r in eval_ if r.source == 'vcc_bench']) == 9
    assert manifest['strata']['benchmark/vcc_bench']['mode'] == 'eval_only'


def test_split_never_hands_a_whole_stratum_to_eval():
    # Two documents, ratio 0.9: the naive greedy fill would take both.
    records = normalize_corpus(_corpus(n_docs=2, paras_per_doc=3))
    train, eval_, _ = split_by_document(records, eval_ratio=0.9)
    assert train and eval_


def test_split_holds_out_the_smallest_document_when_none_fits_the_target():
    # One 10-paragraph document and one 1-paragraph document; a 10% target of
    # 11 records rounds to 1, which only the small document fits.
    records = normalize_corpus([_para('big', i) for i in range(10)] + [_para('small', 0)])
    train, eval_, _ = split_by_document(records, eval_ratio=0.1)
    assert {r.doc_id for r in eval_} == {'small'}
    assert len(train) == 10


def test_rejects_an_out_of_range_eval_ratio():
    records = normalize_corpus(_corpus(n_docs=4, paras_per_doc=2))
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            split_by_document(records, eval_ratio=bad)


def test_leakage_checker_actually_detects_a_leak():
    records = normalize_corpus(_corpus(n_docs=10, paras_per_doc=4))
    # The pattern §9 forbids: same document, records scattered by position.
    train, eval_ = records[::2], records[1::2]
    issues = check_split_leakage(train, eval_)
    assert issues and 'document' in issues[0]


# ============================================================================
# Verification (§6.1)
# ============================================================================


def test_verify_flags_empty_context_and_duplicate_ids_as_errors():
    records = [
        Record(id='a', kind=KIND_CORPUS, source='s', source_id='1', doc_id='d1',
               task='context_compression', context=''),
        Record(id='b', kind=KIND_CORPUS, source='s', source_id='2', doc_id='d2',
               task='context_compression', context='x' * 300),
        Record(id='b', kind=KIND_CORPUS, source='s', source_id='3', doc_id='d3',
               task='context_compression', context='y' * 300),
    ]
    report = verify_records(records)
    assert not report.ok
    assert report.stats['counters']['empty_context'] == 1
    assert report.stats['counters']['duplicate_id'] == 1


def test_verify_flags_a_query_conditioned_sample_with_no_query():
    records = normalize_benchmark([{'sample_id': 's1', 'task': 'long_document_qa',
                                    'context': 'x' * 300, 'query': '', 'reference_answer': 'a'}])
    report = verify_records(records)
    assert not report.ok
    assert report.stats['counters']['missing_query'] == 1


def test_verify_flags_a_reference_answer_that_is_a_copy_of_the_context():
    context = 'Nội dung bài viết. ' * 30
    records = normalize_benchmark([{'sample_id': 's1', 'task': 'long_document_qa',
                                    'context': context, 'query': 'q', 'reference_answer': context}])
    report = verify_records(records)
    assert report.ok, 'a degenerate reference is a warning, not a blocking error'
    assert report.stats['counters']['degenerate_reference'] == 1


def test_verify_counts_duplicate_contexts():
    context = 'Trùng lặp. ' * 40
    records = normalize_benchmark([
        {'sample_id': 'a', 'task': 'needle_in_haystack', 'context': context,
         'query': 'q', 'reference_answer': 'Trùng'},
        {'sample_id': 'b', 'task': 'needle_in_haystack', 'context': context,
         'query': 'q', 'reference_answer': 'Trùng'},
    ])
    assert verify_records(records).stats['counters']['duplicate_context'] == 1


# ============================================================================
# IO round-trips (§12)
# ============================================================================


def test_jsonl_round_trip_preserves_the_split_unit(tmp_path):
    records = normalize_corpus(_corpus(n_docs=5, paras_per_doc=3))
    path = str(tmp_path / 'records.jsonl')
    assert write_jsonl(path, records) == 15
    back = list(read_jsonl(path))
    assert [r.doc_key for r in back] == [r.doc_key for r in records]
    assert [r.context for r in back] == [r.context for r in records]


def test_normalize_file_auto_detects_all_three_legacy_shapes(tmp_path):
    paragraphs = tmp_path / 'p.json'
    paragraphs.write_text(json.dumps({'paragraphs': [{'text': 'a' * 300, 'topic_id': 't1'}]}), encoding='utf-8')
    assert normalize_file(str(paragraphs))[0].kind == KIND_CORPUS

    samples = tmp_path / 's.json'
    samples.write_text(json.dumps({'samples': [{'sample_id': 'x', 'task': 'long_document_qa',
                                                'context': 'b' * 300, 'query': 'q',
                                                'reference_answer': 'r'}]}), encoding='utf-8')
    assert normalize_file(str(samples))[0].kind == KIND_BENCHMARK

    flat = tmp_path / 'l.json'
    flat.write_text(json.dumps(['c' * 300]), encoding='utf-8')
    assert len(normalize_file(str(flat))) == 1


def test_vcc_bench_json_output_is_readable_by_the_legacy_loader(tmp_path):
    records = normalize_benchmark([{'sample_id': 'needle_0000', 'task': 'needle_in_haystack',
                                    'context': 'c' * 400, 'query': 'q', 'reference_answer': 'a'}])
    path = str(tmp_path / 'vcc_bench_eval.json')
    write_vcc_bench_json(path, records, {'split': 'eval'})
    payload = json.loads(open(path, encoding='utf-8').read())
    assert payload['metadata']['split'] == 'eval'
    sample = payload['samples'][0]
    assert {'task', 'context', 'query', 'reference_answer', 'char_length'} <= set(sample)
    assert sample['doc_id'] == 'needle_0000'


# ============================================================================
# Against the real committed benchmark
# ============================================================================


@pytest.mark.skipif(not os.path.exists(VCC_BENCH_V1), reason='vcc_bench_v1.json not present')
def test_real_benchmark_splits_without_leaking_a_document():
    records = normalize_file(VCC_BENCH_V1, default_source='vcc_bench')
    train, eval_, manifest = split_by_document(records, eval_ratio=0.1, seed=42)
    assert check_split_leakage(train, eval_) == []
    assert manifest['eval_ratio_realized'] == pytest.approx(0.1, abs=0.03)

    # The concrete leak the old record-level split produced: the three query
    # variants of one Wikipedia paragraph, and the chapters of one law.
    docs = group_by_document(records)
    assert any(len(group) >= 3 for group in docs.values()), 'expected multi-sample documents'
    eval_docs = {r.doc_key for r in eval_}
    for doc_key, group in docs.items():
        assert (doc_key in eval_docs) == all(r in eval_ for r in group)
