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
    base = dict(base_url='https://example.test/v1', model='m', api_key='k', max_retries=2)
    base.update(kw)
    return TeacherConfig(**base)


def test_http_client_retries_rate_limits_then_succeeds():
    import urllib.error

    client = HTTPTeacherClient(_config(), sleep=lambda _s: None)
    calls = {'n': 0}

    def fake_post(payload):
        calls['n'] += 1
        if calls['n'] == 1:
            raise urllib.error.HTTPError('u', 429, 'Too Many Requests', {}, None)
        return {'choices': [{'message': {'content': ' ok '}}]}

    client._post = fake_post
    assert client.complete([{'role': 'user', 'content': 'x'}]) == 'ok'
    assert calls['n'] == 2 and client.n_retries == 1


def test_http_client_does_not_retry_a_client_error():
    """Retrying a 400 burns quota and hides the real bug."""
    import io as _io
    import urllib.error

    client = HTTPTeacherClient(_config(), sleep=lambda _s: None)
    calls = {'n': 0}

    def fake_post(payload):
        calls['n'] += 1
        raise urllib.error.HTTPError('u', 400, 'Bad Request', {}, _io.BytesIO(b'bad model'))

    client._post = fake_post
    with pytest.raises(RuntimeError, match='HTTP 400'):
        client.complete([{'role': 'user', 'content': 'x'}])
    assert calls['n'] == 1


def test_http_client_gives_up_after_max_retries():
    import urllib.error

    client = HTTPTeacherClient(_config(max_retries=2), sleep=lambda _s: None)
    client._post = lambda payload: (_ for _ in ()).throw(urllib.error.URLError('down'))
    with pytest.raises(RuntimeError, match='after 3 attempts'):
        client.complete([{'role': 'user', 'content': 'x'}])


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

    base = dict(ratio_tolerance=0.25, min_budget_fraction=0.2, min_extractive=0.95)
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


def test_compression_filter_rejects_a_number_the_teacher_itself_marked_important():
    from collections import Counter

    from scripts.filter_dataset import filter_compression

    context = 'Doanh thu đạt 8500 tỷ đồng trong năm 2026. ' + ('thêm nội dung nền. ' * 30)
    counters, rejected = Counter(), []
    rows = [{'key': 'k', 'record_id': 'r1', 'target_tokens': 20, 'context_tokens': 80,
             'teacher_output': {'compressed_text': 'Doanh thu đạt trong năm 2026.',
                                'numbers': ['8500']}}]
    assert filter_compression(rows, {'r1': _source_record(context)}, _args(), counters, rejected) == []
    assert counters['number_dropped'] == 1


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
