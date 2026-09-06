"""Tests for the teacher-LLM distillation flow (docs/dataset_pipeline.md §4, §6, §14).

Every test here runs offline. That is the point of the design: the flow must be
developable and verifiable without an endpoint, because the alternative is
debugging prompt plumbing against a billed API.

No test reads the repository's real `.env` -- each one points `env_file` at a
temporary path and clears the variables first, so a developer's credentials
never leak into the suite's environment.
"""
import json
import os

import pytest

from vncompress.teacher import (
    CachedTeacherClient,
    DryRunTeacherClient,
    HTTPTeacherClient,
    TeacherConfig,
    TeacherConfigError,
    TeacherOutputError,
    extract_json,
    load_dotenv,
)
from vncompress.teacher_prompts import (
    PROMPT_VERSION,
    build_compression_messages,
    build_query_messages,
    count_words,
    dry_run_response,
    target_tokens,
)

ENV_VARS = ('VNCOMPRESS_TEACHER_BASE_URL', 'VNCOMPRESS_TEACHER_API_KEY', 'VNCOMPRESS_TEACHER_MODEL',
            'VNCOMPRESS_TEACHER_TEMPERATURE', 'VNCOMPRESS_TEACHER_MAX_TOKENS',
            'VNCOMPRESS_TEACHER_TIMEOUT', 'VNCOMPRESS_TEACHER_MAX_RETRIES',
            'VNCOMPRESS_TEACHER_CACHE_DIR', 'TEACHER_API_KEY', 'OPENAI_API_KEY',
            'apiKey', 'baseURL', 'model_name')


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ============================================================================
# .env loading and configuration (§14)
# ============================================================================


def test_dotenv_parses_comments_blank_lines_and_quotes(clean_env, tmp_path):
    path = tmp_path / '.env'
    path.write_text(
        '# a comment\n\n'
        'VNCOMPRESS_TEACHER_BASE_URL=https://example.test/v1\n'
        'VNCOMPRESS_TEACHER_API_KEY="sk-quoted"\n'
        "VNCOMPRESS_TEACHER_MODEL='quoted-model'\n"
        'MALFORMED_LINE\n', encoding='utf-8')
    loaded = load_dotenv(str(path))
    assert loaded['VNCOMPRESS_TEACHER_BASE_URL'] == 'https://example.test/v1'
    assert loaded['VNCOMPRESS_TEACHER_API_KEY'] == 'sk-quoted'
    assert loaded['VNCOMPRESS_TEACHER_MODEL'] == 'quoted-model'
    assert 'MALFORMED_LINE' not in loaded


def test_real_environment_wins_over_dotenv(clean_env, tmp_path):
    """CI and secret managers must override a developer's local file."""
    clean_env.setenv('VNCOMPRESS_TEACHER_MODEL', 'from-environment')
    path = tmp_path / '.env'
    path.write_text('VNCOMPRESS_TEACHER_MODEL=from-file\n', encoding='utf-8')
    load_dotenv(str(path))
    assert os.environ['VNCOMPRESS_TEACHER_MODEL'] == 'from-environment'


def test_config_accepts_provider_style_aliases(clean_env, tmp_path):
    """A provider hands you {"baseURL": ..., "apiKey": ..., "model_name": ...};
    that blob should paste into .env and work."""
    path = tmp_path / '.env'
    path.write_text('baseURL=https://example.test/v1\napiKey=sk-alias\nmodel_name=aliased\n',
                    encoding='utf-8')
    config = TeacherConfig.from_env(str(path))
    assert (config.base_url, config.model, config.api_key) == ('https://example.test/v1', 'aliased', 'sk-alias')


def test_config_error_names_the_missing_variables_and_the_fix(clean_env, tmp_path):
    with pytest.raises(TeacherConfigError) as excinfo:
        TeacherConfig.from_env(str(tmp_path / 'absent.env'))
    message = str(excinfo.value)
    assert 'VNCOMPRESS_TEACHER_BASE_URL' in message
    assert '.env.example' in message and '--dry-run' in message


def test_config_can_skip_the_key_for_dry_runs(clean_env, tmp_path):
    path = tmp_path / '.env'
    path.write_text('VNCOMPRESS_TEACHER_BASE_URL=https://example.test/v1\n'
                    'VNCOMPRESS_TEACHER_MODEL=m\n', encoding='utf-8')
    config = TeacherConfig.from_env(str(path), require_key=False)
    assert config.model == 'm'


def test_provenance_never_contains_the_api_key(clean_env, tmp_path):
    path = tmp_path / '.env'
    path.write_text('VNCOMPRESS_TEACHER_BASE_URL=https://example.test/v1\n'
                    'VNCOMPRESS_TEACHER_API_KEY=sk-secret-value\n'
                    'VNCOMPRESS_TEACHER_MODEL=m\n', encoding='utf-8')
    config = TeacherConfig.from_env(str(path))
    blob = json.dumps(config.provenance(PROMPT_VERSION)) + repr(config)
    assert 'sk-secret-value' not in blob


# ============================================================================
# Response parsing (§14: retry on invalid output)
# ============================================================================


@pytest.mark.parametrize('text', [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    '```\n{"a": 1}\n```',
    'Đây là kết quả:\n{"a": 1}',
    '{"a": 1}\nHy vọng giúp ích!',
])
def test_extract_json_survives_chatty_models(text):
    assert extract_json(text) == {'a': 1}


def test_extract_json_handles_arrays():
    assert extract_json('[1, 2]') == [1, 2]


@pytest.mark.parametrize('text', ['', '   ', 'không có JSON ở đây'])
def test_extract_json_raises_on_unparseable(text):
    with pytest.raises(TeacherOutputError):
        extract_json(text)


# ============================================================================
# HTTP client: retry policy
# ============================================================================


def _config(**kw):
    base = dict(base_url='https://example.test/v1', model='m', api_key='k',
                max_attempts=3, retry_delay=30.0)
    base.update(kw)
    return TeacherConfig(**base)


def test_http_client_waits_the_fixed_delay_before_resending():
    """A failed call waits retry_delay seconds and is sent again -- a fixed
    pause, not exponential backoff."""
    import urllib.error

    slept = []
    client = HTTPTeacherClient(_config(), sleep=slept.append)
    calls = {'n': 0}

    def fake_post(payload):
        calls['n'] += 1
        if calls['n'] == 1:
            raise urllib.error.HTTPError('u', 429, 'Too Many Requests', {}, None)
        return {'choices': [{'message': {'content': ' ok '}}]}

    client._post = fake_post
    assert client.complete([{'role': 'user', 'content': 'x'}]) == 'ok'
    assert calls['n'] == 2 and client.n_retries == 1
    assert slept == [30.0]


def test_http_client_does_not_retry_a_client_error():
    """Retrying a 400 burns quota and hides the real bug."""
    import io as _io
    import urllib.error

    from vncompress.teacher import TeacherCallError

    client = HTTPTeacherClient(_config(), sleep=lambda _s: None)
    calls = {'n': 0}

    def fake_post(payload):
        calls['n'] += 1
        raise urllib.error.HTTPError('u', 400, 'Bad Request', {}, _io.BytesIO(b'bad model'))

    client._post = fake_post
    with pytest.raises(TeacherCallError, match='HTTP 400') as excinfo:
        client.complete([{'role': 'user', 'content': 'x'}])
    assert calls['n'] == 1
    assert excinfo.value.status == 400 and excinfo.value.attempts == 1


def test_http_client_gives_up_after_three_attempts_and_says_so():
    """Three failures end the call. The error carries what the failure log needs."""
    import urllib.error

    from vncompress.teacher import TeacherCallError

    slept = []
    client = HTTPTeacherClient(_config(max_attempts=3), sleep=slept.append)
    client._post = lambda payload: (_ for _ in ()).throw(urllib.error.URLError('down'))
    with pytest.raises(TeacherCallError, match='after 3 attempt') as excinfo:
        client.complete([{'role': 'user', 'content': 'x'}])
    assert client.n_calls == 3
    assert slept == [30.0, 30.0], 'waits between attempts, not after the last one'
    assert excinfo.value.attempts == 3
    assert 'URLError' in excinfo.value.to_dict()['last_error']


# ============================================================================
# Cache (§14: cache teacher output)
# ============================================================================


def test_cache_returns_the_stored_response_without_calling_again(tmp_path):
    inner = DryRunTeacherClient(responder=lambda m: '{"n": 1}')
    messages = build_query_messages('Một đoạn văn tiếng Việt đủ dài để hỏi.')

    def cached(version='v1'):
        return CachedTeacherClient(inner, str(tmp_path / 'cache'), 'model-x', version, 512)

    first = cached()
    assert first.complete(messages) == '{"n": 1}'
    assert inner.n_calls == 1

    second = cached()
    assert second.complete(messages) == '{"n": 1}'
    assert inner.n_calls == 1, 'a cache hit must not reach the inner client'
    assert second.n_hits == 1


def test_bumping_the_prompt_version_invalidates_only_that_stage(tmp_path):
    inner = DryRunTeacherClient(responder=lambda m: '{"n": 1}')
    messages = build_query_messages('Một đoạn văn tiếng Việt đủ dài để hỏi.')
    CachedTeacherClient(inner, str(tmp_path / 'c'), 'model-x', 'v1', 512).complete(messages)
    CachedTeacherClient(inner, str(tmp_path / 'c'), 'model-x', 'v2', 512).complete(messages)
    assert inner.n_calls == 2


def test_cache_survives_a_corrupt_entry(tmp_path):
    inner = DryRunTeacherClient(responder=lambda m: '{"n": 1}')
    client = CachedTeacherClient(inner, str(tmp_path / 'c'), 'model-x', 'v1', 512)
    messages = [{'role': 'user', 'content': 'x'}]
    client.complete(messages)
    path = client._path(client._key(messages, 0.0))
    open(path, 'w', encoding='utf-8').write('not json')
    assert client.complete(messages) == '{"n": 1}'


# ============================================================================
# Prompts (§4.2, §4.3)
# ============================================================================


def test_compression_prompt_states_the_budget_and_the_extractive_constraint():
    context = 'Một câu. ' * 40
    messages = build_compression_messages(context, 'Câu hỏi?', 4)
    system, user = messages[0]['content'], messages[1]['content']
    assert 'KHÔNG được thêm thông tin' in system and 'TRÍCH XUẤT' in system
    assert str(target_tokens(context, 4)) in user
    assert 'Câu hỏi?' in user and context.strip()[:20] in user


def test_query_prompt_forbids_the_degenerate_question_shape():
    system = build_query_messages('x' * 300)[0]['content']
    assert 'nguyên văn' in system.lower()
    assert 'đoạn văn nói về gì' in system


def test_target_tokens_scales_with_the_ratio():
    context = 'từ ' * 100
    assert target_tokens(context, 2) == 50
    # 100/8 = 12.5; round() is banker's rounding, so 12. Immaterial against a
    # budget the filter allows 25% slack on.
    assert target_tokens(context, 8) == 12
    assert target_tokens('', 4) == 1, 'budget must never be zero'


def test_dry_run_produces_valid_output_for_both_stages():
    context = ('Hà Nội là thủ đô của Việt Nam. Dân số khoảng 8,5 triệu người vào năm 2024. '
               'Thành phố không giáp biển và có bốn mùa rõ rệt trong năm.')
    compression = extract_json(dry_run_response(build_compression_messages(context, 'Dân số?', 2)))
    assert compression['compressed_text']
    assert count_words(compression['compressed_text']) <= target_tokens(context, 2) * 1.5

    queries = extract_json(dry_run_response(build_query_messages(context)))
    assert queries['queries']
    for item in queries['queries']:
        assert item['answer_span'] in context, 'the stub must produce verbatim spans'


# ============================================================================
# Filtering (§6)
# ============================================================================


def _source_record(context, doc_id='doc1'):
    from vncompress.dataset import KIND_CORPUS, Record

    return Record(id='r1', kind=KIND_CORPUS, source='uvw-2026', source_id='1', doc_id=doc_id,
                  task='context_compression', context=context)


def _args(**kw):
    from types import SimpleNamespace

    base = dict(ratio_tolerance=0.25, min_words=4, budget_slack=8, min_extractive=0.95)
    base.update(kw)
    return SimpleNamespace(**base)


def test_extractive_ratio_detects_invented_text():
    from scripts.filter_dataset import extractive_ratio

    context = 'Hà Nội là thủ đô của Việt Nam.'
    assert extractive_ratio('Hà Nội là thủ đô', context) == 1.0
    assert extractive_ratio('Hà Nội là thành phố Paris', context) < 0.95


@pytest.mark.parametrize('compressed,reason', [
    ('', 'empty_output'),
    ('một hai ba bốn năm sáu bảy tám chín mười ' * 6, 'over_budget'),
])
def test_compression_filter_rejects_bad_output(compressed, reason):
    from collections import Counter

    from scripts.filter_dataset import filter_compression

    context = 'một hai ba bốn năm sáu bảy tám chín mười ' * 8
    counters, rejected = Counter(), []
    rows = [{'key': 'r1|abc|4', 'record_id': 'r1', 'target_tokens': 20, 'context_tokens': 80,
             'teacher_output': {'compressed_text': compressed}}]
    kept = filter_compression(rows, {'r1': _source_record(context)}, _args(), counters, rejected)
    assert kept == [] and counters[reason] == 1
    assert rejected[0]['reason'] == reason


def test_dropping_a_source_number_the_query_never_asked_about_is_accepted():
    """The teacher's `numbers` field lists numbers found in the SOURCE, not
    numbers that must survive. Requiring all of them rejected correct answers."""
    from collections import Counter

    from scripts.filter_dataset import filter_compression

    context = 'Doanh thu đạt 8500 tỷ đồng trong năm 2026. ' + ('thêm nội dung nền. ' * 30)
    counters, rejected = Counter(), []
    rows = [{'key': 'k', 'record_id': 'r1', 'query': 'Năm nào?', 'target_tokens': 20,
             'context_tokens': 80, 'teacher': {},
             'teacher_output': {'compressed_text': 'Doanh thu đạt trong năm 2026.',
                                'numbers': ['8500', '2026']}}]
    kept = filter_compression(rows, {'r1': _source_record(context)}, _args(), counters, rejected)
    assert len(kept) == 1, f'rejected as {[r["reason"] for r in rejected]}'
    # How many survived is recorded for analysis rather than enforced.
    assert kept[0].metadata['quality']['numbers_preserved'] == 0.5


def test_a_number_the_model_invented_is_rejected():
    """Fabricating a figure is the number failure that actually matters for a
    finance or legal dataset."""
    from collections import Counter

    from scripts.filter_dataset import filter_compression

    context = 'Doanh thu đạt 8500 tỷ đồng trong năm 2026. ' + ('thêm nội dung nền. ' * 30)
    counters, rejected = Counter(), []
    rows = [{'key': 'k', 'record_id': 'r1', 'target_tokens': 20, 'context_tokens': 80,
             'teacher_output': {'compressed_text': 'Doanh thu đạt 9900 tỷ đồng trong năm 2026.'}}]
    assert filter_compression(rows, {'r1': _source_record(context)}, _args(), counters, rejected) == []
    assert counters['number_altered'] == 1


def test_invented_numbers_ignores_trailing_punctuation():
    from scripts.filter_dataset import invented_numbers

    assert invented_numbers('năm 1948.', 'thành lập năm 1948 tại Paris') == []
    assert invented_numbers('năm 1949', 'thành lập năm 1948 tại Paris') == ['1949']


def test_compression_filter_accepts_and_populates_the_schema_five_fields():
    from collections import Counter

    from scripts.filter_dataset import filter_compression

    context = 'một hai ba bốn năm sáu bảy tám chín mười ' * 8
    counters, rejected = Counter(), []
    rows = [{'key': 'r1|abc|4', 'record_id': 'r1', 'query': 'Câu hỏi?', 'compression_ratio': 4.0,
             'target_tokens': 20, 'context_tokens': 80, 'token_unit': 'whitespace',
             'teacher': {'model': 'm', 'prompt_version': 'v1'},
             'teacher_output': {'compressed_text': 'một hai ba bốn năm sáu bảy tám chín mười ' * 2,
                                'important_spans': ['một hai'], 'numbers': [],
                                'compression_reason': 'giữ phần đầu'}}]
    kept = filter_compression(rows, {'r1': _source_record(context)}, _args(), counters, rejected)
    assert len(kept) == 1 and counters['accepted'] == 1
    meta = kept[0].metadata
    assert meta['compressed_text'] and meta['compression_ratio'] == 4.0
    assert meta['quality']['extractive_ratio'] == 1.0
    assert meta['teacher']['prompt_version'] == 'v1'


def test_query_filter_rejects_a_non_verbatim_span_and_a_whole_paragraph_answer():
    from collections import Counter

    from scripts.filter_dataset import filter_queries

    context = 'Hà Nội là thủ đô của Việt Nam và có lịch sử lâu đời hơn một nghìn năm.'
    counters, rejected = Counter(), []
    rows = [{'record_id': 'r1', 'queries': [
        {'query': 'Thủ đô?', 'answer': 'Paris', 'answer_span': 'Paris'},
        {'query': 'Nội dung?', 'answer': context, 'answer_span': context},
        {'query': 'Thủ đô của Việt Nam?', 'answer': 'Hà Nội', 'answer_span': 'Hà Nội'},
    ]}]
    kept = filter_queries(rows, {'r1': _source_record(context)}, _args(), counters, rejected)
    assert counters['answer_not_verbatim'] == 1
    assert counters['degenerate_answer'] == 1
    assert len(kept) == 1 and kept[0].reference_answer == 'Hà Nội'


def test_synthetic_questions_inherit_the_source_document_id():
    """Otherwise questions generated from one article could be split across
    train and eval, reintroducing exactly the §9 leak."""
    from collections import Counter

    from scripts.filter_dataset import filter_queries

    context = 'Hà Nội là thủ đô của Việt Nam và có lịch sử lâu đời.'
    counters, rejected = Counter(), []
    rows = [{'record_id': 'r1', 'queries': [
        {'query': 'Thủ đô?', 'answer': 'Hà Nội', 'answer_span': 'Hà Nội'},
        {'query': 'Lịch sử?', 'answer': 'lâu đời', 'answer_span': 'lâu đời'},
    ]}]
    kept = filter_queries(rows, {'r1': _source_record(context, doc_id='Ha_Noi')}, _args(),
                          counters, rejected)
    assert {r.doc_id for r in kept} == {'Ha_Noi'}
    assert {r.source for r in kept} == {'teacher-synth'}


# ============================================================================
# Defects the first real endpoint run exposed (dry-run could not surface them)
# ============================================================================


def test_query_stage_skips_records_that_already_carry_a_query(tmp_path):
    """The first live run spent every one of its calls on benchmark samples
    that already had a query -- this stage exists for corpus paragraphs."""
    from types import SimpleNamespace

    from scripts.generate_teacher_dataset import stage_queries
    from vncompress.dataset import KIND_BENCHMARK, KIND_CORPUS, Record

    records = [
        Record(id='has_q', kind=KIND_BENCHMARK, source='vcc_bench', source_id='1', doc_id='d1',
               task='long_document_qa', context='Ngữ cảnh đủ dài. ' * 20, query='Có sẵn?'),
        Record(id='no_q', kind=KIND_CORPUS, source='uvw-2026', source_id='2', doc_id='d2',
               task='context_compression', context='Ngữ cảnh khác đủ dài. ' * 20),
    ]
    from scripts.generate_teacher_dataset import ResultWriter

    client = DryRunTeacherClient()
    args = SimpleNamespace(n_queries=3, json_retries=1, json_retry_delay=0, provenance={}, include_answered=False,
                           workers=1)
    with open(tmp_path / 'out.jsonl', 'w', encoding='utf-8') as out:
        writer = ResultWriter(out, str(tmp_path / 'fail.jsonl'), 'queries')
        stage_queries(args, client, records, writer, set())

    assert writer.n_written == 1, 'only the record without a query should cost a call'
    assert client.n_calls == 1

    args.include_answered = True
    with open(tmp_path / 'out2.jsonl', 'w', encoding='utf-8') as out:
        writer2 = ResultWriter(out, str(tmp_path / 'fail2.jsonl'), 'queries')
        stage_queries(args, client, records, writer2, set())
    assert writer2.n_written == 2


def test_synthesized_questions_are_deduplicated_by_context_not_record_id():
    """`conv_0012_q0..q2` are three records sharing one context verbatim. Keyed
    on record id, the same question was emitted three times."""
    from collections import Counter

    from scripts.filter_dataset import filter_queries

    context = 'Hà Nội là thủ đô của Việt Nam và có lịch sử lâu đời hơn một nghìn năm.'
    queries = [{'query': 'Thủ đô của Việt Nam?', 'answer': 'Hà Nội', 'answer_span': 'Hà Nội'}]
    rows = [{'record_id': rid, 'queries': queries} for rid in ('c_q0', 'c_q1', 'c_q2')]
    sources = {rid: _source_record(context, doc_id='conv_0012') for rid in ('c_q0', 'c_q1', 'c_q2')}

    counters, rejected = Counter(), []
    kept = filter_queries(rows, sources, _args(), counters, rejected)
    assert len(kept) == 1, 'one context + one question = one record'
    assert counters['duplicate_query'] == 2


def test_dedup_is_insensitive_to_whitespace_and_case_only():
    from collections import Counter

    from scripts.filter_dataset import filter_queries

    context = 'Hà Nội là thủ đô của Việt Nam và có lịch sử lâu đời.'
    rows = [{'record_id': 'r1', 'queries': [
        {'query': 'Thủ  đô?', 'answer': 'Hà Nội', 'answer_span': 'Hà Nội'},
        {'query': 'THỦ ĐÔ?', 'answer': 'Hà Nội', 'answer_span': 'Hà Nội'},
        {'query': 'Lịch sử bao lâu?', 'answer': 'lâu đời', 'answer_span': 'lâu đời'},
    ]}]
    counters, rejected = Counter(), []
    kept = filter_queries(rows, {'r1': _source_record(context)}, _args(), counters, rejected)
    assert len(kept) == 2 and counters['duplicate_query'] == 1


def test_an_exhausted_call_is_written_to_the_failure_log_not_dropped(tmp_path):
    """A hole in a 20k-row dataset is invisible unless something writes down
    that it happened."""
    from types import SimpleNamespace

    from scripts.generate_teacher_dataset import ResultWriter, stage_queries
    from vncompress.dataset import KIND_CORPUS, Record
    from vncompress.teacher import TeacherCallError

    def always_fails(messages):
        raise TeacherCallError('endpoint down', attempts=3, last_error="URLError('down')", status=None)

    records = [Record(id=f'r{i}', kind=KIND_CORPUS, source='uvw-2026', source_id=str(i),
                      doc_id=f'd{i}', task='context_compression', context='Ngữ cảnh đủ dài. ' * 20)
               for i in range(3)]
    args = SimpleNamespace(n_queries=3, json_retries=1, json_retry_delay=0, provenance={}, include_answered=False,
                           workers=1)
    failures = tmp_path / 'failures.jsonl'
    with open(tmp_path / 'out.jsonl', 'w', encoding='utf-8') as out:
        writer = ResultWriter(out, str(failures), 'queries')
        stage_queries(args, DryRunTeacherClient(responder=always_fails), records, writer, set())

    assert writer.n_written == 0 and writer.n_failed == 3
    logged = [json.loads(line) for line in failures.read_text(encoding='utf-8').splitlines() if line]
    assert {r['record_id'] for r in logged} == {'r0', 'r1', 'r2'}
    assert all(r['attempts'] == 3 and r['stage'] == 'queries' for r in logged)
    assert all(r['failed_at'] and r['error_type'] == 'TeacherCallError' for r in logged)


def test_parallel_workers_write_every_row_exactly_once(tmp_path):
    """Workers share one output handle; a lost or interleaved row would be a
    silent data-loss bug over a multi-hour run."""
    from types import SimpleNamespace

    from scripts.generate_teacher_dataset import ResultWriter, stage_queries
    from vncompress.dataset import KIND_CORPUS, Record

    records = [Record(id=f'r{i}', kind=KIND_CORPUS, source='uvw-2026', source_id=str(i),
                      doc_id=f'd{i}', task='context_compression',
                      context=f'Ngữ cảnh số {i} đủ dài để hỏi. ' * 20)
               for i in range(60)]
    args = SimpleNamespace(n_queries=3, json_retries=1, json_retry_delay=0, provenance={}, include_answered=False,
                           workers=8)
    out_path = tmp_path / 'out.jsonl'
    with open(out_path, 'w', encoding='utf-8') as out:
        writer = ResultWriter(out, str(tmp_path / 'fail.jsonl'), 'queries')
        stage_queries(args, DryRunTeacherClient(), records, writer, set())

    lines = [line for line in out_path.read_text(encoding='utf-8').splitlines() if line]
    assert len(lines) == 60 and writer.n_written == 60
    rows = [json.loads(line) for line in lines]        # every line must be valid JSON
    assert {r['record_id'] for r in rows} == {f'r{i}' for i in range(60)}


def test_resume_skips_keys_already_present_in_the_output(tmp_path):
    from types import SimpleNamespace

    from scripts.generate_teacher_dataset import ResultWriter, stage_queries
    from vncompress.dataset import KIND_CORPUS, Record

    records = [Record(id=f'r{i}', kind=KIND_CORPUS, source='uvw-2026', source_id=str(i),
                      doc_id=f'd{i}', task='context_compression', context='Ngữ cảnh đủ dài. ' * 20)
               for i in range(5)]
    args = SimpleNamespace(n_queries=3, json_retries=1, json_retry_delay=0, provenance={}, include_answered=False,
                           workers=2)
    with open(tmp_path / 'out.jsonl', 'w', encoding='utf-8') as out:
        writer = ResultWriter(out, str(tmp_path / 'fail.jsonl'), 'queries')
        n_skipped = stage_queries(args, DryRunTeacherClient(), records, writer, {'r0', 'r1'})
    assert n_skipped == 2 and writer.n_written == 3


def test_compression_prefers_queries_that_passed_verification(tmp_path):
    """Compressing against a question the filter already rejected wastes calls
    and aims supervision at a target judged unusable."""
    import json as _json
    from types import SimpleNamespace

    from scripts.generate_teacher_dataset import _load_synthesized_queries

    teacher_dir = tmp_path / 'teacher'
    teacher_dir.mkdir()
    (teacher_dir / 'queries_raw.jsonl').write_text(_json.dumps({
        'record_id': 'r1', 'queries': [{'query': 'câu bị loại'}, {'query': 'câu hợp lệ'}],
    }, ensure_ascii=False) + '\n', encoding='utf-8')

    processed = tmp_path / 'processed'
    processed.mkdir()
    (processed / 'records_synthetic_qa.jsonl').write_text(_json.dumps({
        'id': 'synthqa_r1_1', 'kind': 'benchmark', 'source': 'teacher-synth', 'source_id': 'r1',
        'doc_id': 'd1', 'task': 'long_document_qa', 'context': 'ctx', 'query': 'câu hợp lệ',
        'reference_answer': 'a', 'metadata': {'origin_record': 'r1'},
    }, ensure_ascii=False) + '\n', encoding='utf-8')

    args = SimpleNamespace(teacher_dir=str(teacher_dir))
    monkey = os.environ.get('VNCOMPRESS_PROCESSED_DIR')
    os.environ['VNCOMPRESS_PROCESSED_DIR'] = str(processed)
    try:
        got = _load_synthesized_queries(args, [])
    finally:
        if monkey is None:
            del os.environ['VNCOMPRESS_PROCESSED_DIR']
        else:
            os.environ['VNCOMPRESS_PROCESSED_DIR'] = monkey

    assert got == {'r1': [{'query': 'câu hợp lệ'}]}, 'the rejected question must not be compressed'


def test_compression_falls_back_to_raw_queries_with_a_warning(tmp_path, capsys):
    from types import SimpleNamespace

    from scripts.generate_teacher_dataset import _load_synthesized_queries

    teacher_dir = tmp_path / 'teacher'
    teacher_dir.mkdir()
    (teacher_dir / 'queries_raw.jsonl').write_text(json.dumps({
        'record_id': 'r1', 'queries': [{'query': 'chưa lọc'}],
    }, ensure_ascii=False) + '\n', encoding='utf-8')

    empty = tmp_path / 'empty'
    empty.mkdir()
    os.environ['VNCOMPRESS_PROCESSED_DIR'] = str(empty)
    try:
        got = _load_synthesized_queries(SimpleNamespace(teacher_dir=str(teacher_dir)), [])
    finally:
        del os.environ['VNCOMPRESS_PROCESSED_DIR']

    assert got == {'r1': [{'query': 'chưa lọc'}]}
    assert 'RAW (unverified)' in capsys.readouterr().out


def test_unresolved_failures_counts_only_what_is_still_missing(tmp_path, monkeypatch):
    """The failure log is append-only history: a key in it may have succeeded on
    a later pass, so only the difference against the output is a real gap."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        'run_pipeline', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'scripts', 'run_pipeline.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    teacher = tmp_path / 'data' / 'teacher'
    teacher.mkdir(parents=True)
    (teacher / 'failures_queries.jsonl').write_text(
        '\n'.join(json.dumps({'key': k}) for k in ('a', 'b', 'c')) + '\n', encoding='utf-8')
    (teacher / 'queries_raw.jsonl').write_text(
        '\n'.join(json.dumps({'key': k}) for k in ('a', 'b')) + '\n', encoding='utf-8')

    monkeypatch.setattr(module, 'REPO', str(tmp_path))
    assert module._unresolved_failures('queries') == 1, 'only "c" is still missing'


def test_unresolved_failures_is_zero_without_a_failure_log(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        'run_pipeline', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'scripts', 'run_pipeline.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'REPO', str(tmp_path))
    assert module._unresolved_failures('compression') == 0


# ============================================================================
# Cache poisoning: the defect that made retries impossible
# ============================================================================


def test_an_empty_completion_is_never_cached(tmp_path):
    """The cache holds raw text and cannot tell a good response from a broken
    one. Caching an empty completion makes the failure permanent: every later
    retry, including the pipeline's retry passes, is served the same empty
    string from disk and can never recover."""
    responses = ['', '{"queries": []}']
    inner = DryRunTeacherClient(responder=lambda m: responses.pop(0))
    client = CachedTeacherClient(inner, str(tmp_path / 'c'), 'm', 'v1', 512)
    messages = [{'role': 'user', 'content': 'x'}]

    assert client.complete(messages) == ''
    assert client.complete(messages) == '{"queries": []}', 'the empty response must not be replayed'
    assert inner.n_calls == 2


def test_invalidate_removes_a_cached_response(tmp_path):
    responses = ['not json', '{"ok": 1}']
    inner = DryRunTeacherClient(responder=lambda m: responses.pop(0))
    client = CachedTeacherClient(inner, str(tmp_path / 'c'), 'm', 'v1', 512)
    messages = [{'role': 'user', 'content': 'x'}]

    assert client.complete(messages) == 'not json'
    assert client.complete(messages) == 'not json', 'still cached until invalidated'
    assert client.invalidate(messages) is True
    assert client.complete(messages) == '{"ok": 1}'


def test_invalidate_is_safe_when_nothing_is_cached(tmp_path):
    client = CachedTeacherClient(DryRunTeacherClient(), str(tmp_path / 'c'), 'm', 'v1', 512)
    assert client.invalidate([{'role': 'user', 'content': 'absent'}]) is False


def test_retry_drops_the_cached_bad_response_so_the_model_is_actually_re_asked(tmp_path):
    """The end-to-end symptom: six query records re-failed in 0.0 minutes across
    three retry passes because nothing was ever re-asked."""
    from scripts.generate_teacher_dataset import _call_with_retry

    responses = ['', '', '{"queries": [{"query": "q"}]}']
    inner = DryRunTeacherClient(responder=lambda m: responses.pop(0))
    client = CachedTeacherClient(inner, str(tmp_path / 'c'), 'm', 'v1', 512)

    payload, _raw = _call_with_retry(client, [{'role': 'user', 'content': 'x'}],
                                     max_attempts=3, retry_delay=0)
    assert payload == {'queries': [{'query': 'q'}]}
    assert inner.n_calls == 3, 'each attempt must reach the model, not the cache'


def test_an_empty_response_is_retried_with_the_original_prompt(tmp_path):
    """Echoing an empty assistant turn back and asking the model to fix the
    format gives it nothing to fix."""
    from scripts.generate_teacher_dataset import _call_with_retry

    seen = []

    def responder(messages):
        seen.append(list(messages))
        return '' if len(seen) == 1 else '{"ok": 1}'

    client = DryRunTeacherClient(responder=responder)
    _call_with_retry(client, [{'role': 'user', 'content': 'prompt gốc'}],
                     max_attempts=3, retry_delay=0)

    assert len(seen) == 2
    assert seen[1] == seen[0], 'an empty response must be retried with the original prompt'


def test_a_malformed_but_non_empty_response_is_echoed_back_for_correction(tmp_path):
    from scripts.generate_teacher_dataset import _call_with_retry

    seen = []

    def responder(messages):
        seen.append(list(messages))
        return 'đây không phải JSON' if len(seen) == 1 else '{"ok": 1}'

    _call_with_retry(DryRunTeacherClient(responder=responder),
                     [{'role': 'user', 'content': 'prompt gốc'}], max_attempts=3, retry_delay=0)

    assert len(seen[1]) == 3, 'original + assistant echo + correction'
    assert seen[1][1]['role'] == 'assistant'
    assert 'JSON hợp lệ' in seen[1][2]['content']


def test_aggressive_query_focused_compression_is_not_rejected_as_too_short():
    """§4.3 makes the budget an upper bound. A needle task legitimately reduces
    a huge haystack to a one-sentence needle; the old relative floor rejected
    exactly the best output in the set."""
    from collections import Counter

    from scripts.filter_dataset import filter_compression

    needle = 'Mật khẩu truy cập hệ thống là VIETCOMPRESS2026_SECURE.'
    context = ('nội dung nền không liên quan. ' * 2000) + needle
    counters, rejected = Counter(), []
    rows = [{'key': 'k', 'record_id': 'r1', 'query': 'Mật khẩu là gì?', 'compression_ratio': 2.0,
             'target_tokens': 3000, 'context_tokens': 6000, 'token_unit': 'whitespace',
             'teacher': {}, 'teacher_output': {'compressed_text': needle, 'numbers': []}}]
    kept = filter_compression(rows, {'r1': _source_record(context)}, _args(), counters, rejected)
    assert len(kept) == 1, f'rejected as {[r["reason"] for r in rejected]}'
    assert counters['accepted'] == 1


def test_truly_degenerate_output_is_still_rejected():
    from collections import Counter

    from scripts.filter_dataset import filter_compression

    counters, rejected = Counter(), []
    rows = [{'key': 'k', 'record_id': 'r1', 'target_tokens': 100, 'context_tokens': 400,
             'teacher_output': {'compressed_text': 'một hai'}}]
    assert filter_compression(rows, {'r1': _source_record('một hai ba ' * 100)},
                              _args(), counters, rejected) == []
    assert counters['too_short'] == 1


def test_tiny_budgets_get_absolute_slack_on_top_of_the_percentage():
    """At 8x the target can be ~11 words, where a 25% tolerance is under three
    words of room and rejects on rounding alone."""
    from collections import Counter

    from scripts.filter_dataset import filter_compression

    context = 'một hai ba bốn năm sáu bảy tám chín mười ' * 10
    counters, rejected = Counter(), []
    # target 11, realized 17: over 25% (13.75) but inside the absolute slack (19).
    rows = [{'key': 'k', 'record_id': 'r1', 'target_tokens': 11, 'context_tokens': 100,
             'teacher': {}, 'teacher_output': {
                 'compressed_text': 'một hai ba bốn năm sáu bảy tám chín mười một hai ba bốn năm sáu bảy',
                 'numbers': []}}]
    kept = filter_compression(rows, {'r1': _source_record(context)}, _args(), counters, rejected)
    assert len(kept) == 1, f'rejected as {[r["reason"] for r in rejected]}'
