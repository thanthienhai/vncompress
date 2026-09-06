#!/usr/bin/env python3
"""generate_teacher_dataset.py -- teacher-LLM distillation (docs/dataset_pipeline.md §4, §15 Phase 1-2).

Turns canonical records into the supervision §5 describes, using a strong
instruction-following teacher over an OpenAI-compatible endpoint. Two stages:

    --stage queries       corpus paragraph -> query-conditioned instances
                          (§15 Phase 1: a paragraph carries no question, so
                          query-conditioned supervision must construct one)
    --stage compression   (context, query, ratio) -> compressed_text +
                          important/removed spans + entity/number/date/
                          condition/negation markup (§4.2)

Raw output goes to `data/teacher/*.jsonl` and is **kept**, per §12: teacher
calls are the expensive part, so re-filtering must never require re-generating.
Verification and the merge into canonical records happen in the next stage,
`scripts/filter_dataset.py` -- this script does not decide what is good enough.

**Input defaults to the TRAIN split.** §10 requires the benchmark to stay
independent of the teacher pipeline; pointing this at `eval.jsonl` would train
a student on supervision derived from its own test set, so that needs an
explicit override.

Every row carries §14 provenance: model, prompt version, temperature, stage,
timestamp, and the source record and document id.

Usage:
    # Exercise the whole flow with no endpoint and no spend:
    python scripts/generate_teacher_dataset.py --stage queries --dry-run --limit 20
    python scripts/generate_teacher_dataset.py --stage compression --dry-run --limit 20

    # Against the configured endpoint (.env), small trial first:
    python scripts/generate_teacher_dataset.py --stage queries --limit 50
    python scripts/generate_teacher_dataset.py --stage compression --ratios 2,4,8 --limit 50
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vncompress.dataset import KIND_BENCHMARK, KIND_CORPUS, normalize_file, processed_path  # noqa: E402
from vncompress.teacher import (  # noqa: E402
    TeacherCallError,
    TeacherConfig,
    TeacherConfigError,
    TeacherOutputError,
    build_client,
    extract_json,
)
from vncompress.teacher_prompts import (  # noqa: E402
    PROMPT_VERSION,
    STAGE_COMPRESSION,
    STAGE_QUERIES,
    build_compression_messages,
    build_query_messages,
    count_words,
    target_tokens,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEACHER_DIR = os.path.join(REPO, 'data', 'teacher')
OUTPUT_NAME = {STAGE_QUERIES: 'queries_raw.jsonl', STAGE_COMPRESSION: 'compression_raw.jsonl'}


def _done_keys(path):
    """Resume support: keys already generated, so a re-run continues instead of
    paying for the same rows twice."""
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line)['key'])
            except (ValueError, KeyError):
                continue
    return keys


def _query_key(record_id, query):
    return f'{record_id}|{hashlib.blake2b(query.encode("utf-8"), digest_size=6).hexdigest()}'


class ResultWriter:
    """Serializes writes from every worker to the output and failure logs.

    Rows are flushed as they complete rather than buffered to the end: a run
    over the full corpus takes hours, and a crash at hour six must not throw
    away everything -- the resume path reads back exactly what was flushed.
    """

    def __init__(self, out, failures_path, stage):
        self.out = out
        self.failures_path = failures_path
        self.stage = stage
        self.lock = threading.Lock()
        self.n_written = 0
        self.n_failed = 0

    def row(self, obj):
        line = json.dumps(obj, ensure_ascii=False) + '\n'
        with self.lock:
            self.out.write(line)
            self.out.flush()
            self.n_written += 1

    def failure(self, key, record_id, error, attempts=None, detail=None):
        """Record an exhausted call so it can be traced and replayed later.

        Never silently dropped: a hole in a 20k-row dataset is invisible unless
        something writes down that it happened.
        """
        entry = {
            'key': key, 'record_id': record_id, 'stage': self.stage,
            'error': str(error), 'error_type': type(error).__name__,
            'attempts': attempts, 'detail': detail,
            'failed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        line = json.dumps(entry, ensure_ascii=False) + '\n'
        with self.lock:
            with open(self.failures_path, 'a', encoding='utf-8') as f:
                f.write(line)
            self.n_failed += 1


def _failure_info(exc):
    if isinstance(exc, TeacherCallError):
        return exc.attempts, exc.to_dict()
    return None, None


def run_tasks(tasks, worker, writer, workers, label):
    """Map `worker` over `tasks`, in parallel when asked.

    Threads, not processes: every task is one blocking HTTPS request, so the
    GIL is released for essentially the whole task. Sequential execution over
    the full corpus is ~26h for the query stage alone, which is why this exists.
    """
    total = len(tasks)
    if not total:
        return
    started = time.time()
    done = 0

    def report():
        elapsed = time.time() - started
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        print(f'  {done}/{total} {label} | {rate:.2f}/s | ok={writer.n_written} '
              f'fail={writer.n_failed} | còn ~{remaining / 60:.0f} phút', flush=True)

    def guarded(task):
        try:
            worker(task)
        except (TeacherOutputError, TeacherCallError, RuntimeError) as exc:
            attempts, detail = _failure_info(exc)
            writer.failure(getattr(task, 'key', None) or str(task)[:80],
                           getattr(task, 'record_id', None), exc, attempts, detail)

    step = max(1, min(50, total // 20 or 1))
    if workers <= 1:
        for task in tasks:
            guarded(task)
            done += 1
            if done % step == 0 or done == total:
                report()
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(guarded, task) for task in tasks]
        for future in as_completed(futures):
            future.result()  # guarded() never raises; re-raise a real bug loudly
            done += 1
            if done % step == 0 or done == total:
                report()


def _call_with_retry(client, messages, max_attempts, temperature=None):
    """§14: retry when the output is invalid, rather than dropping the sample."""
    last = None
    for attempt in range(max_attempts):
        raw = client.complete(messages, temperature=temperature)
        try:
            return extract_json(raw), raw
        except TeacherOutputError as exc:
            last = exc
            if attempt + 1 < max_attempts:
                messages = messages + [
                    {'role': 'assistant', 'content': raw[:500]},
                    {'role': 'user', 'content': 'Sai định dạng. Chỉ trả về đúng một object JSON hợp lệ.'},
                ]
    raise TeacherOutputError(f'invalid JSON after {max_attempts} attempts: {last}')


class QueryTask:
    __slots__ = ('key', 'record_id', 'record')

    def __init__(self, record):
        self.key = self.record_id = record.id
        self.record = record


def stage_queries(args, client, records, writer, done):
    tasks, n_skipped, n_answered = [], 0, 0
    for record in records:
        if record.id in done:
            n_skipped += 1
            continue
        # A benchmark sample already carries a query; synthesizing another one
        # spends tokens re-deriving what is already there. This stage exists for
        # corpus paragraphs, which have none.
        if record.query and not args.include_answered:
            n_answered += 1
            continue
        tasks.append(QueryTask(record))

    def worker(task):
        messages = build_query_messages(task.record.context, args.n_queries)
        payload, _raw = _call_with_retry(client, messages, args.json_retries)
        queries = payload.get('queries') if isinstance(payload, dict) else None
        writer.row({
            'key': task.key,
            'stage': STAGE_QUERIES,
            'record_id': task.record.id,
            'doc_id': task.record.doc_id,
            'source': task.record.source,
            'queries': queries if isinstance(queries, list) else [],
            'teacher': {**args.provenance, 'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S')},
        })

    if n_answered:
        print(f'  {n_answered} record(s) already had a query and were skipped '
              f'(pass --include-answered to synthesize anyway)')
    run_tasks(tasks, worker, writer, args.workers, 'records')
    return n_skipped


def _queries_for(record, synthesized):
    """Where a record's query comes from: its own field for benchmark samples,
    the stage-1 output for corpus paragraphs."""
    if record.query:
        return [record.query]
    return [q.get('query', '') for q in synthesized.get(record.id, []) if q.get('query')]


class CompressionTask:
    __slots__ = ('key', 'record_id', 'record', 'query', 'ratio')

    def __init__(self, key, record, query, ratio):
        self.key = key
        self.record_id = record.id
        self.record = record
        self.query = query
        self.ratio = ratio


def stage_compression(args, client, records, writer, done):
    synthesized = {}
    queries_path = os.path.join(args.teacher_dir, OUTPUT_NAME[STAGE_QUERIES])
    if os.path.exists(queries_path):
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    synthesized[row['record_id']] = row.get('queries', [])
        print(f'  loaded synthesized queries for {len(synthesized)} records')
    elif any(not r.query for r in records):
        print(f'  [WARN] {queries_path} not found -- corpus paragraphs have no query and will be '
              f'skipped. Run --stage queries first.')

    tasks, n_skipped, n_noquery = [], 0, 0
    for record in records:
        queries = _queries_for(record, synthesized)
        if args.max_queries_per_record > 0:
            queries = queries[:args.max_queries_per_record]
        if not queries:
            n_noquery += 1
            continue
        for query in queries:
            for ratio in args.ratios:
                key = f'{_query_key(record.id, query)}|{ratio:g}'
                if key in done:
                    n_skipped += 1
                    continue
                tasks.append(CompressionTask(key, record, query, ratio))

    def worker(task):
        messages = build_compression_messages(task.record.context, task.query, task.ratio)
        payload, _raw = _call_with_retry(client, messages, args.json_retries)
        if not isinstance(payload, dict):
            raise TeacherOutputError('expected a JSON object')
        compressed = (payload.get('compressed_text') or '').strip()
        writer.row({
            'key': task.key,
            'stage': STAGE_COMPRESSION,
            'record_id': task.record.id,
            'doc_id': task.record.doc_id,
            'source': task.record.source,
            'task': task.record.task,
            'query': task.query,
            'compression_ratio': task.ratio,
            'target_tokens': target_tokens(task.record.context, task.ratio),
            'realized_tokens': count_words(compressed),
            'context_tokens': count_words(task.record.context),
            'token_unit': 'whitespace',
            'teacher_output': payload,
            'teacher': {**args.provenance, 'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S')},
        })

    if n_noquery:
        print(f'  {n_noquery} record(s) had no query available and were skipped')
    run_tasks(tasks, worker, writer, args.workers, 'instances')
    return n_skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=[STAGE_QUERIES, STAGE_COMPRESSION], required=True)
    ap.add_argument('--input', default=None,
                    help='Canonical .jsonl or raw dataset file. Default: data/processed/train.jsonl')
    ap.add_argument('--kind', choices=[KIND_CORPUS, KIND_BENCHMARK, 'all'], default='all',
                    help='Restrict to one record kind.')
    ap.add_argument('--teacher-dir', default=TEACHER_DIR, help='Default: data/teacher/')
    ap.add_argument('--ratios', default='2,4,8',
                    help='Compression ratios for --stage compression (§8). Default: 2,4,8')
    ap.add_argument('--n-queries', type=int, default=3, help='Queries to synthesize per paragraph.')
    ap.add_argument('--include-answered', action='store_true',
                    help='--stage queries: also synthesize for records that already carry a query. '
                         'Off by default -- it spends tokens re-deriving an existing field.')
    ap.add_argument('--limit', type=int, default=0, help='Cap input records (0 = all). Use for trials.')
    ap.add_argument('--workers', type=int, default=1,
                    help='Concurrent in-flight requests. Each task is one blocking HTTPS call, so '
                         'this is close to a linear speedup until the endpoint rate-limits. '
                         'Sequential (1) over the full corpus is ~26h for --stage queries alone.')
    ap.add_argument('--max-queries-per-record', type=int, default=0,
                    help='--stage compression: cap queries used per record (0 = all). The instance '
                         'count is records x queries x ratios, so this is the main cost dial.')
    ap.add_argument('--json-retries', type=int, default=2,
                    help='Attempts to get valid JSON out of one prompt (§14).')
    ap.add_argument('--temperature', type=float, default=None, help='Override the .env temperature.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Run the whole flow against a deterministic offline stub: no endpoint, no spend.')
    ap.add_argument('--no-cache', action='store_true', help='Bypass the on-disk teacher cache.')
    ap.add_argument('--allow-eval-input', action='store_true',
                    help='Permit an eval-split input. §10 says do not: it makes the benchmark '
                         'dependent on the teacher pipeline.')
    args = ap.parse_args()

    args.ratios = [float(r) for r in args.ratios.split(',') if r.strip()]

    path = args.input or processed_path('train.jsonl')
    if not os.path.exists(path):
        ap.error(f'{path} not found -- build the splits first:\n'
                 f'  python scripts/normalize_dataset.py && python scripts/split_dataset.py')
    if 'eval' in os.path.basename(path) and not args.allow_eval_input:
        ap.error(f'refusing to generate teacher supervision from {os.path.basename(path)} (§10: the '
                 f'benchmark must stay independent of the teacher pipeline). '
                 f'Pass --allow-eval-input only if you know why.')

    try:
        config = TeacherConfig.from_env(require_key=not args.dry_run)
    except TeacherConfigError as exc:
        if not args.dry_run:
            print(f'[FAIL] {exc}')
            return 1
        config = TeacherConfig(base_url='dry-run://local', model='dry-run', api_key='')

    records = normalize_file(path)
    if args.kind != 'all':
        records = [r for r in records if r.kind == args.kind]
    if args.limit:
        records = records[:args.limit]

    os.makedirs(args.teacher_dir, exist_ok=True)
    # Stub output must never share a file with real output: the rows look
    # structurally identical, and a --dry-run appended into the real dataset is
    # silent contamination that only shows up as inexplicably bad supervision.
    name = OUTPUT_NAME[args.stage]
    if args.dry_run:
        name = name.replace('.jsonl', '.dryrun.jsonl')
    out_path = os.path.join(args.teacher_dir, name)
    done = _done_keys(out_path)
    client = build_client(config, PROMPT_VERSION, dry_run=args.dry_run, use_cache=not args.no_cache)
    args.provenance = {**config.provenance(PROMPT_VERSION), 'stage': args.stage,
                       'client': client.name, 'dry_run': bool(args.dry_run)}

    print(f'Stage      : {args.stage} (prompt {PROMPT_VERSION})')
    print(f'Teacher    : {config.model} @ {config.base_url}'
          + ('  [DRY RUN -- no requests are sent]' if args.dry_run else ''))
    print(f'Input      : {os.path.relpath(path, REPO)} -> {len(records)} records')
    print(f'Concurrency: {args.workers} worker(s) | retry {config.max_attempts} lần, '
          f'cách nhau {config.retry_delay:g}s')
    print(f'Output     : {os.path.relpath(out_path, REPO)}'
          + (f' (resuming, {len(done)} already done)' if done else ''))
    if args.stage == STAGE_COMPRESSION:
        print(f'Ratios     : {", ".join(f"{r:g}x" for r in args.ratios)}')
    print()

    failures_path = os.path.join(
        args.teacher_dir, f'failures_{args.stage}{".dryrun" if args.dry_run else ""}.jsonl')
    started = time.time()
    with open(out_path, 'a', encoding='utf-8') as out:
        writer = ResultWriter(out, failures_path, args.stage)
        runner = stage_queries if args.stage == STAGE_QUERIES else stage_compression
        n_skipped = runner(args, client, records, writer, done)

    elapsed = time.time() - started
    print(f'\nWrote {writer.n_written} row(s) to {os.path.relpath(out_path, REPO)} '
          f'in {elapsed / 60:.1f} min')
    if n_skipped:
        print(f'  skipped (already generated): {n_skipped}')
    hits = getattr(client, 'n_hits', None)
    if hits is not None:
        print(f'  cache: {hits} hit(s), {client.n_misses} miss(es)')
    if writer.n_failed:
        print(f'  FAILED after {config.max_attempts} attempt(s): {writer.n_failed} '
              f'-> {os.path.relpath(failures_path, REPO)}')
        print('  Trace them: python scripts/inspect_failures.py --stage ' + args.stage)
        print('  Replay them: re-run this exact command -- failed rows were never written to the '
              'output, so the resume check picks them up again.')
    print('\nNext: python scripts/filter_dataset.py --stage ' + args.stage)
    return 0


if __name__ == '__main__':
    sys.exit(main())
