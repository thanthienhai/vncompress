#!/usr/bin/env python3
"""verify_eval_set.py -- stage 4: decide whether each test row's
`gold_compression` is actually ENOUGH to answer its question (SS6 E1/E2).

    python scripts/verify_eval_set.py --stage judge  --dry-run   # LLM over all rows
    python scripts/verify_eval_set.py --stage sample             # queue for a person
    python scripts/verify_eval_set.py --stage merge              # fold reviews back in

What is being checked is SUFFICIENCY, not containment. The answer string is
inside the compression by construction -- it is cut from the human-marked
answer span -- so a containment check passes 1000/1000 and tells you nothing:

    Q    : Ông nội của Thái Tông Trần Cảnh có tên là gì?
    GOLD : "Trần Lý sinh ra Trần Thừa."
    ANS  : Trần Lý                       <- present, and still unanswerable,
                                            because nothing here says Trần Thừa
                                            is Trần Cảnh's father.

A cheap lexical heuristic flags ~9% of rows as this shape, which is why the
check cannot be skipped and also why it cannot be left to the heuristic.

THE DIVISION OF LABOUR
  LLM   judges every row. It reads (query, gold_compression) and NOTHING else
        -- handing it the context would let it answer from the context and
        certify a compression that does not contain the answer at all.
  HUMAN reviews a stratified sample, which is what the agreement number is
        computed from. Without it the judge's error rate is unknown and
        "verified" is a word rather than a measurement.

Neither is allowed to impersonate the other: every row records
`verification_method` in {none, llm, human, llm+human}, and the validator
refuses to read 'llm' as human verification (SS7's "người kiểm" line).

The judge MUST NOT be the teacher model. A judge that shares the teacher's
blind spots certifies exactly the compressions the teacher botches, and the
failure is invisible because both agree. The script refuses that combination.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vncompress.dataset_schema import (
    ANSWERABLE_F1,
    MIN_HUMAN_VERIFIED_SAMPLES,
    MIN_JUDGE_AGREEMENT_KAPPA,
)
from vncompress.api_pool import (
    DEFAULT_ACCOUNT_RPM,
    DEFAULT_CCU_PER_KEY,
    DEFAULT_RPM,
    KeyPool,
    collect_api_keys,
    load_dotenv,
    run_pool,
)
from vncompress.evaluation import compute_token_f1

load_dotenv()

DEFAULT_BASE_URL = 'https://token-api.fpt.ai/v1'
# Deliberately not GLM-5.2 (scripts/generate_compression_pairs.py DEFAULT_MODEL),
# and deliberately a different FAMILY, not just a different size. Two checkpoints
# of one family share their training data and therefore their blind spots, so
# the judge would certify exactly the compressions the teacher botches.
# Confirmed present on token-api.fpt.ai alongside GLM-5.2, and confirmed NOT a
# reasoning model -- gpt-oss-120b is, and returned empty content at max_tokens=64
# because the whole budget went to its reasoning trace. A judge whose answer can
# vanish into a thinking budget is a judge that scores rows unanswerable for a
# reason that has nothing to do with the row.
DEFAULT_JUDGE_MODEL = 'Llama-3.3-70B-Instruct'
TEACHER_MODEL = 'GLM-5.2'

JUDGE_SYSTEM = (
    "Bạn là người chấm dữ liệu. Bạn chỉ được dùng đúng đoạn văn bản được đưa cho, "
    "không được dùng kiến thức bên ngoài."
)

JUDGE_PROMPT = """Dưới đây là một BẢN NÉN và một CÂU HỎI.

Chỉ dựa vào BẢN NÉN, hãy trả lời CÂU HỎI thật ngắn gọn.
Nếu bản nén KHÔNG chứa đủ thông tin để trả lời chắc chắn, hãy trả lời đúng hai chữ: KHÔNG ĐỦ.
Tuyệt đối không suy đoán và không dùng kiến thức bên ngoài.

BẢN NÉN:
{gold_compression}

CÂU HỎI:
{query}

ĐÁP ÁN:"""

INSUFFICIENT_MARKERS = ('không đủ', 'không có', 'không thể', 'khong du', 'khong co')

# Vietnamese function words carry no entity information, so their presence or
# absence in the compression says nothing about whether the question's subject
# survived. Stripping them is what makes the overlap score discriminative.
_STOPWORDS = set(
    "là gì nào ai đâu bao nhiêu của có được và với cho từ trong khi nếu thì mà này đó "
    "các những một hai người tên năm tại về theo như đã sẽ không phải hay hoặc ra vào "
    "lên xuống bởi do vì nên cái sự việc thế thứ mấy đầu tiên gọi nghĩa biết".split()
)


# ============================================================================
# Pure helpers (tested in tests/test_verify_eval_set.py)
# ============================================================================


def content_words(text):
    normalized = unicodedata.normalize('NFC', (text or '').lower())
    return [w for w in re.findall(r'\w+', normalized)
            if w not in _STOPWORDS and len(w) > 1]


def risk_score(query, gold_compression):
    """0.0 (every content word of the question is present) to 1.0 (none are).

    Not a verdict -- a sampling weight. Its only job is to make sure the human
    queue is not 150 easy rows: rows the question's own subject never reaches
    are where the judge is most likely to be wrong, so they must be represented
    even though they are a minority.
    """
    question = set(content_words(query))
    if not question:
        return 1.0
    return 1.0 - len(question & set(content_words(gold_compression))) / len(question)


def parse_judge_answer(raw, gold_answer, threshold=ANSWERABLE_F1):
    """(verdict, f1, cleaned). An explicit refusal is a `False`, not an F1 of 0.

    Kept apart because the two failures are different: "the judge said the
    compression is insufficient" is the signal being asked for, while a low F1
    on an attempted answer may just be a paraphrase. Collapsing them would hide
    how often the judge actually refuses.
    """
    cleaned = (raw or '').strip()
    lowered = unicodedata.normalize('NFC', cleaned.lower())
    if not cleaned:
        return False, 0.0, cleaned
    if any(lowered.startswith(m) or lowered == m for m in INSUFFICIENT_MARKERS):
        return False, 0.0, cleaned
    f1 = compute_token_f1([cleaned], [gold_answer or ''])
    return bool(f1 >= threshold), f1, cleaned


def cohen_kappa(pairs):
    """Agreement between two binary raters, corrected for chance.

    Raw agreement is unusable at this base rate: if 90% of rows are genuinely
    answerable, a judge that always says "answerable" agrees 90% of the time
    and has learned nothing. Kappa scores that judge at 0.

    Returns (kappa, raw_agreement, confusion). Kappa is None when one rater
    never varies -- undefined, and reporting 0.0 there would read as
    "disagrees" when the truth is "cannot be computed from this sample".
    """
    if not pairs:
        return None, None, {}
    n = len(pairs)
    confusion = Counter(pairs)
    observed = sum(v for (a, b), v in confusion.items() if a == b) / n
    a_true = sum(1 for a, _ in pairs if a) / n
    b_true = sum(1 for _, b in pairs if b) / n
    expected = a_true * b_true + (1 - a_true) * (1 - b_true)
    if expected >= 1.0:
        return None, observed, {f"{a}|{b}": v for (a, b), v in confusion.items()}
    kappa = (observed - expected) / (1 - expected)
    return kappa, observed, {f"{a}|{b}": v for (a, b), v in confusion.items()}


def stratified_sample(rows, size, seed=42):
    """Pick the human queue so it spans the axes the judge could fail along.

    The judge verdict is allocated FIRST, half the budget each way, and risk
    tier and document are spread within each half. Treating all three as one
    flat set of strata looked equivalent and is not: when the verdict happens
    to correlate with the document -- which it does, since a badly-cut article
    produces a run of bad compressions -- document strata outnumber verdict
    strata and the minority verdict gets diluted back to its base rate. A test
    pool where rejections lived in 2 of 8 documents came back 26.7% rejections
    against a 25% base rate, i.e. stratified in name only.

    Verdict balance is not one nicety among three: Cohen's kappa is estimated
    from the off-diagonal cells, and a sample with few rejections has almost no
    off-diagonal to estimate from.
    """
    rng = random.Random(seed)

    def spread(group, budget):
        """Round-robin over (risk tier, document) within one verdict group."""
        buckets = defaultdict(list)
        for row in group:
            risk = risk_score(row.get('query'), row.get('gold_compression'))
            tier = 0 if risk < 0.34 else (1 if risk < 0.67 else 2)
            buckets[(tier, row.get('doc_id'))].append(row)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        keys = sorted(buckets, key=lambda k: (k[0], str(k[1])))
        picked = []
        while len(picked) < budget and any(buckets[k] for k in keys):
            for key in keys:
                if buckets[key] and len(picked) < budget:
                    picked.append(buckets[key].pop())
        return picked

    by_verdict = defaultdict(list)
    for row in rows:
        by_verdict[bool((row.get('verification') or {}).get('llm', {}).get('verdict'))].append(row)

    half = size // 2
    rejected = spread(by_verdict[False], half)
    # Whatever the minority side could not fill goes to the majority rather
    # than shrinking the sample -- a short queue costs agreement precision.
    accepted = spread(by_verdict[True], size - len(rejected))
    if len(rejected) + len(accepted) < size:
        rejected += spread([r for r in by_verdict[False] if r not in rejected],
                           size - len(rejected) - len(accepted))
    return rejected + accepted


def apply_verdicts(row, llm=None, human=None):
    """Fold judgements into a row and derive the fields the validator reads.

    The human verdict wins where it exists; the LLM verdict stands everywhere
    else. Both are kept in full -- the point of the exercise is being able to
    say later which rows a person actually looked at.
    """
    verification = dict(row.get('verification') or {})
    if llm is not None:
        verification['llm'] = llm
    if human is not None:
        verification['human'] = human
    row['verification'] = verification

    has_llm, has_human = 'llm' in verification, 'human' in verification
    method = ('llm+human' if has_llm and has_human else
              'human' if has_human else 'llm' if has_llm else 'none')
    row['verification_method'] = method

    who = []
    if has_llm:
        who.append(f"llm:{verification['llm'].get('model')}")
    if has_human:
        who.append(f"human:{verification['human'].get('annotator')}")
    row['verified_by'] = '+'.join(who) if who else None

    if has_human:
        row['answerable_from_compression'] = bool(verification['human'].get('verdict'))
    elif has_llm:
        row['answerable_from_compression'] = bool(verification['llm'].get('verdict'))
    return row


# ============================================================================
# I/O
# ============================================================================


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class Judge:
    """One model, many keys. Retry and cool-down belong to the pool, not here.

    An in-call retry would hide a 429 from the pool, so the key that is being
    rate-limited never cools down and keeps being handed work -- which is how a
    four-key pool degrades to one slow key without anything looking wrong.
    """

    def __init__(self, model, base_url, temperature=0.0, dry_run=False, seed=42,
                 max_tokens=4096, request_timeout=300.0, max_connections=256):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.max_connections = max_connections
        self.dry_run = dry_run
        self.calls = Counter()
        self._rng = random.Random(seed)
        self._clients = {}
        self._lock = threading.Lock()

    def _client_for(self, api_key):
        with self._lock:
            client = self._clients.get(api_key.index)
            if client is None:
                import httpx
                from openai import OpenAI
                # max_retries=0 is load-bearing. The SDK retries twice by
                # default, inside the call, so a 429 was absorbed before the
                # pool ever saw it: the rate-limited key never cooled down and
                # kept being handed work, and the client-side limiter counted
                # one request where up to three were sent -- which is how a
                # pool configured for 200 req/min was actually exceeding 150.
                #
                # The connection pool must also fit the worker count. httpx
                # defaults to 100 max connections; at 4 keys x 30 concurrent
                # that is 120 workers fighting over 100 slots, which surfaced
                # as 52 APIConnectionError in a single batch.
                #
                # No connection REUSE either, for the same reason stage 3 gave
                # it up: a pooled connection this gateway has already closed
                # comes back as "Server disconnected without sending a
                # response" on its next use. One TLS handshake per call is the
                # cheaper side of that trade.
                client = OpenAI(
                    api_key=api_key.key, base_url=self.base_url,
                    max_retries=0, timeout=self.request_timeout,
                    http_client=httpx.Client(
                        limits=httpx.Limits(max_connections=self.max_connections,
                                            max_keepalive_connections=0,
                                            keepalive_expiry=5.0),
                        timeout=self.request_timeout),
                )
                self._clients[api_key.index] = client
            return client

    def ask(self, gold_compression, query, gold_answer, api_key=None):
        if self.dry_run:
            with self._lock:
                self.calls['dry_run'] += 1
            return self._fake(gold_compression, query, gold_answer)
        with self._lock:
            self.calls['requests'] += 1
        response = self._client_for(api_key).chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            messages=[{'role': 'system', 'content': JUDGE_SYSTEM},
                      {'role': 'user', 'content': JUDGE_PROMPT.format(
                          gold_compression=gold_compression, query=query)}])
        choice = response.choices[0]
        text = (choice.message.content or '').strip()
        if choice.finish_reason == 'length':
            # Must not be parsed. An answer cut off at the token limit scores a
            # low F1 and would be recorded as "the compression is insufficient"
            # -- a verdict about the judge's budget, filed against the data.
            with self._lock:
                self.calls['truncated'] += 1
            raise RuntimeError(f"judge response truncated at max_tokens={self.max_tokens}")
        return text

    def _fake(self, gold_compression, query, gold_answer):
        """Offline stand-in with the failure mode the real judge has.

        It answers only when the question's own content words are present in
        the compression, and refuses otherwise -- so --dry-run exercises both
        branches, the stratified sampler sees a real mix of verdicts, and the
        kappa path runs on something other than a constant.
        """
        if risk_score(query, gold_compression) > 0.5:
            return 'KHÔNG ĐỦ'
        return gold_answer or 'KHÔNG ĐỦ'


def build_judge_pool(args):
    """Judge keys, falling back to the teacher's -- same provider, same quota.

    Two prefixes rather than one because the judge may need to run on a
    different account: the judge must not be the teacher MODEL, and on some
    providers that also means different credentials.
    """
    keys = collect_api_keys('VNCOMPRESS_JUDGE_API_KEY') or \
        collect_api_keys('VNCOMPRESS_TEACHER_API_KEY')
    if args.api_key and args.api_key not in keys:
        keys.insert(0, args.api_key)
    if not keys:
        raise SystemExit(
            "No API keys. Set VNCOMPRESS_JUDGE_API_KEY_1..._4 in .env "
            "(see .env.example),\nor reuse VNCOMPRESS_TEACHER_API_KEY_*, "
            "or pass --api-key.\nTo exercise the flow without one: --dry-run")
    pool = KeyPool(keys, rpm=args.rpm, ccu_per_key=args.ccu_per_key,
                   account_rpm=args.account_rpm)
    print(f"Keys: {len(keys)} distinct · {args.ccu_per_key} concurrent each "
          f"· {args.rpm} req/min each -> ~{pool.theoretical_rpm()} req/min")
    if len(keys) == 1:
        print("  NOTE: one key only. Add _2.._4 to .env to run 4x wider.")
    return pool


# ============================================================================
# Stages
# ============================================================================


def stage_judge(args, eval_path, rows):
    if args.judge_model == TEACHER_MODEL and not args.allow_same_model:
        raise SystemExit(
            f"The judge model is the teacher model ({TEACHER_MODEL}).\n"
            "A judge that shares the teacher's blind spots certifies precisely the\n"
            "compressions the teacher gets wrong, and agrees with itself while doing it.\n"
            "Pass --judge-model <something else>, or --allow-same-model if you have a\n"
            "reason and intend to record it in the report.")

    todo = [r for r in rows
            if args.rejudge or 'llm' not in (r.get('verification') or {})]
    if not todo:
        print("Every row already has an LLM verdict. Pass --rejudge to redo them.")
        return 0

    pool = (KeyPool(['dry-run'], rpm=10 ** 6, ccu_per_key=args.ccu_per_key, account_rpm=0)
            if args.dry_run else build_judge_pool(args))
    judge = Judge(args.judge_model, args.base_url, dry_run=args.dry_run,
                  seed=args.seed, max_tokens=args.max_tokens,
                  max_connections=max(64, len(pool.keys) * args.ccu_per_key * 2))
    print(f"Judging {len(todo):,} of {len(rows):,} row(s) with {args.judge_model}"
          + (" [DRY RUN]" if args.dry_run else ""))
    print("The judge sees (query, gold_compression) only -- never the context.\n")

    stats, stats_lock = Counter(), threading.Lock()

    def work(row, api_key, stop):
        """One row = one call, so a requeue after a 429 costs exactly one call.

        Exceptions escape on purpose: that is what cools the key that raised
        them and hands this row to a different one.
        """
        raw = judge.ask(row['gold_compression'], row['query'], row['answer'], api_key)
        verdict, f1, cleaned = parse_judge_answer(raw, row['answer'])
        return row, {
            'verdict': verdict, 'f1': round(f1, 4), 'answer': cleaned[:500],
            'model': args.judge_model, 'threshold': ANSWERABLE_F1,
            'sees_context': False, 'judged_at': now(), 'dry_run': bool(args.dry_run),
        }

    def on_result(row, outcome):
        _, llm = outcome
        apply_verdicts(row, llm=llm)
        with stats_lock:
            stats['answerable' if llm['verdict'] else 'NOT answerable'] += 1

    def on_giveup(row, exc):
        # Left at verification_method='none'. The validator fails on that, which
        # is the correct outcome: a row nothing could judge is not a verified
        # row, and quietly defaulting it either way would be inventing a label.
        with stats_lock:
            stats['unjudged: all keys failed the row'] += 1


    error_log_path = os.path.join(args.input, 'provenance', 'api_errors.jsonl')
    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
    error_log = open(error_log_path, 'a', encoding='utf-8')
    error_lock = threading.Lock()

    def on_error(info, item):
        """Every failure goes to provenance/api_errors.jsonl for review.

        429 is expected under load and is handled by a fixed 30s cool-down on
        that key; it is still recorded, labelled, so that afterwards "the run
        was slow" and "the run was broken" can be told apart from the file
        rather than from whatever scrolled past.
        """
        with error_lock:
            error_log.write(json.dumps(
                dict(info, at=now(), item=item.get('id')), ensure_ascii=False) + "\n")
            error_log.flush()

    try:
        _, pool_stats = run_pool(
            todo, work, pool, on_result=on_result, on_giveup=on_giveup,
            on_error=on_error,
            max_item_retries=args.max_item_retries,
            progress_every=max(1, len(todo) // 20))
        stats.update(pool_stats)
    finally:
        error_log.close()
        # Written even on interrupt: 800 judged rows are worth keeping, and the
        # next run skips them because they already carry a verdict.
        write_jsonl(eval_path, rows)

    flagged = [r for r in rows if not (r.get('verification') or {}).get('llm', {}).get('verdict', True)]
    judged = [r for r in rows if (r.get('verification') or {}).get('llm')]
    print(f"\nLLM verdicts written to {eval_path}")
    print(f"  judged: {len(judged):,}/{len(rows):,}")
    print(f"  answerable: {len(judged) - len(flagged):,} · NOT answerable: {len(flagged):,} "
          f"({len(flagged) / max(len(judged), 1):.1%})")
    for key, n in stats.most_common():
        print(f"  {key}: {n:,}")
    if not args.dry_run:
        print(f"  keys: {pool.summary()}")
    print("\nEvery judged row is verification_method='llm'. That is NOT human verification.")
    print(f"Next: python scripts/verify_eval_set.py --input {args.input} --stage sample")
    return 0


def stage_sample(args, eval_path, rows):
    judged = [r for r in rows if 'llm' in (r.get('verification') or {})]
    if not judged:
        raise SystemExit("No LLM verdicts yet. Run --stage judge first.")

    picked = stratified_sample(judged, args.sample_size, args.seed)
    queue_path = os.path.join(args.input, 'provenance', 'human_review_queue.jsonl')
    write_jsonl(queue_path, [{
        'id': r['id'],
        'doc_id': r.get('doc_id'),
        'query': r['query'],
        'gold_compression': r['gold_compression'],
        'gold_answer': r['answer'],
        # The judge's verdict is deliberately NOT shown to the annotator. Seeing
        # it turns an independent judgement into a confirmation, and the
        # agreement number computed from confirmations is not an agreement
        # number. It is carried only so --stage merge can pair them up.
        '_llm_verdict': bool(r['verification']['llm']['verdict']),
        '_risk': round(risk_score(r['query'], r['gold_compression']), 3),
        'human_verdict': None,
        'annotator': None,
        'note': '',
    } for r in picked])

    # A spreadsheet-shaped copy, because the person doing this should not have
    # to edit JSONL by hand to do it.
    csv_path = os.path.join(args.input, 'provenance', 'human_review_queue.tsv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("id\tquery\tgold_compression\tgold_answer\thuman_verdict\tannotator\tnote\n")
        for r in picked:
            cell = lambda s: str(s or '').replace('\t', ' ').replace('\n', ' ')
            f.write(f"{r['id']}\t{cell(r['query'])}\t{cell(r['gold_compression'])}\t"
                    f"{cell(r['answer'])}\t\t\t\n")

    tiers = Counter('low' if risk_score(r['query'], r['gold_compression']) < 0.34
                    else 'mid' if risk_score(r['query'], r['gold_compression']) < 0.67
                    else 'high' for r in picked)
    verdicts = Counter(bool(r['verification']['llm']['verdict']) for r in picked)
    print(f"Human review queue: {len(picked):,} row(s) over "
          f"{len({r.get('doc_id') for r in picked})} document(s)")
    print(f"  by LLM verdict: {dict(verdicts)}")
    print(f"  by risk tier:   {dict(tiers)}")
    print(f"\n  {queue_path}")
    print(f"  {csv_path}   <- fill `human_verdict` with 1 (answerable) / 0 (not), and `annotator`")
    if len(picked) < MIN_HUMAN_VERIFIED_SAMPLES:
        print(f"\nWARNING: {len(picked)} < {MIN_HUMAN_VERIFIED_SAMPLES}, the floor for a usable "
              "agreement estimate. The validator will not accept a smaller sample.")
    print(f"\nThen: python scripts/verify_eval_set.py --input {args.input} --stage merge")
    return 0


def _load_reviews(args):
    """Accept the queue back as either the JSONL or the TSV a person edited."""
    jsonl_path = os.path.join(args.input, 'provenance', 'human_review_queue.jsonl')
    tsv_path = os.path.join(args.input, 'provenance', 'human_review_queue.tsv')
    reviews = {}
    for record in read_jsonl(args.reviews or jsonl_path):
        if record.get('human_verdict') is not None:
            reviews[record['id']] = (bool(int(record['human_verdict'])),
                                     record.get('annotator') or 'unknown',
                                     record.get('note') or '')
    if not reviews and os.path.exists(tsv_path):
        with open(tsv_path, encoding='utf-8') as f:
            header = f.readline().rstrip('\n').split('\t')
            for line in f:
                cells = line.rstrip('\n').split('\t')
                if len(cells) < len(header):
                    cells += [''] * (len(header) - len(cells))
                record = dict(zip(header, cells))
                if record.get('human_verdict', '').strip() not in ('', None):
                    reviews[record['id']] = (bool(int(record['human_verdict'])),
                                             record.get('annotator') or 'unknown',
                                             record.get('note') or '')
    return reviews


def stage_merge(args, eval_path, rows):
    reviews = _load_reviews(args)
    if not reviews:
        raise SystemExit(
            "No filled-in human verdicts found.\n"
            "Fill `human_verdict` (1/0) in provenance/human_review_queue.tsv "
            "(or .jsonl) and re-run.")

    by_id = {r['id']: r for r in rows}
    pairs, unknown = [], 0
    for row_id, (verdict, annotator, note) in reviews.items():
        row = by_id.get(row_id)
        if row is None:
            unknown += 1
            continue
        apply_verdicts(row, human={'verdict': verdict, 'annotator': annotator,
                                   'note': note, 'reviewed_at': now()})
        llm = (row.get('verification') or {}).get('llm')
        if llm is not None:
            pairs.append((bool(llm['verdict']), verdict))

    kappa, agreement, confusion = cohen_kappa(pairs)
    methods = Counter(r.get('verification_method', 'none') for r in rows)
    human_reviewed = methods['human'] + methods['llm+human']
    final_true = sum(1 for r in rows if r.get('answerable_from_compression'))

    calibrated = (human_reviewed >= MIN_HUMAN_VERIFIED_SAMPLES
                  and kappa is not None and kappa >= MIN_JUDGE_AGREEMENT_KAPPA)
    report = {
        'generated_at': now(),
        'eval_rows': len(rows),
        'verification_methods': dict(methods),
        'human_reviewed': human_reviewed,
        'human_sample_floor': MIN_HUMAN_VERIFIED_SAMPLES,
        'judge_model': next((r['verification']['llm']['model'] for r in rows
                             if (r.get('verification') or {}).get('llm')), None),
        'teacher_model': TEACHER_MODEL,
        'judge_is_teacher': False,
        'agreement': {
            'paired_rows': len(pairs),
            'cohen_kappa': None if kappa is None else round(kappa, 4),
            'raw_agreement': None if agreement is None else round(agreement, 4),
            'kappa_floor': MIN_JUDGE_AGREEMENT_KAPPA,
            'confusion_llm|human': confusion,
        },
        'calibrated': calibrated,
        'answerable_final': final_true,
        'unanswerable_final': len(rows) - final_true,
        'unknown_ids_in_review_file': unknown,
        # The sentence that has to survive into the paper. Written here, from
        # the numbers, so it cannot drift from them in the prose.
        'claim': (
            f"LLM-verified ({human_reviewed} of {len(rows)} rows independently reviewed by a "
            f"person; Cohen's kappa {kappa:.3f})" if calibrated and kappa is not None else
            f"LLM-verified, NOT human-calibrated "
            f"({human_reviewed}/{MIN_HUMAN_VERIFIED_SAMPLES} human sample"
            + (f", kappa {kappa:.3f} < {MIN_JUDGE_AGREEMENT_KAPPA}" if kappa is not None
               else ", kappa undefined")
            + ") -- must not be reported as human-verified"),
    }
    path = os.path.join(args.input, 'provenance', 'verification_report.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_jsonl(eval_path, rows)

    print(f"Merged {len(reviews):,} human verdict(s)"
          + (f" ({unknown} id(s) not in the eval set, ignored)" if unknown else ""))
    print(f"  verification_method: {dict(methods)}")
    print(f"  paired rows: {len(pairs)} · raw agreement: "
          f"{'n/a' if agreement is None else f'{agreement:.1%}'} · Cohen's kappa: "
          f"{'undefined' if kappa is None else f'{kappa:.3f}'} "
          f"(floor {MIN_JUDGE_AGREEMENT_KAPPA})")
    print(f"  confusion (llm|human): {confusion}")
    print(f"\n  {'CALIBRATED' if calibrated else 'NOT CALIBRATED'}: {report['claim']}")
    print(f"\nWrote {path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default='data/vncompress_vi_v2')
    ap.add_argument('--stage', choices=['judge', 'sample', 'merge'], required=True)
    ap.add_argument('--judge-model', default=os.environ.get('VNCOMPRESS_JUDGE_MODEL',
                                                            DEFAULT_JUDGE_MODEL))
    ap.add_argument('--base-url', default=os.environ.get('VNCOMPRESS_JUDGE_BASE_URL',
                                                         DEFAULT_BASE_URL))
    ap.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY'),
                    help='A single extra key, on top of whatever .env provides')
    ap.add_argument('--allow-same-model', action='store_true',
                    help='Permit judging with the teacher model (records it in the report)')
    ap.add_argument('--rejudge', action='store_true', help='Re-judge rows that already have a verdict')
    ap.add_argument('--sample-size', type=int, default=MIN_HUMAN_VERIFIED_SAMPLES)
    ap.add_argument('--reviews', default=None, help='Filled-in review file (defaults to the queue)')
    ap.add_argument('--rpm', type=int, default=int(os.environ.get('VNCOMPRESS_RPM', DEFAULT_RPM)),
                    help='Provider rate limit PER KEY, requests/minute')
    ap.add_argument('--account-rpm', type=int,
                    default=int(os.environ.get('VNCOMPRESS_ACCOUNT_RPM', DEFAULT_ACCOUNT_RPM)),
                    help='Ceiling shared by every key. 0 disables.')
    ap.add_argument('--ccu-per-key', type=int,
                    default=int(os.environ.get('VNCOMPRESS_CCU_PER_KEY', DEFAULT_CCU_PER_KEY)),
                    help='In-flight requests per key')
    ap.add_argument('--max-item-retries', type=int, default=4,
                    help='Times a row may be requeued onto another key before it is dropped')
    ap.add_argument('--max-tokens', type=int, default=4096,
                    help='Output cap per judge call')
    ap.add_argument('--dry-run', action='store_true', help='No API key, no network')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    eval_path = os.path.join(args.input, 'eval', 'test.jsonl')
    rows = read_jsonl(eval_path)
    if not rows:
        raise SystemExit(f"{eval_path} not found or empty. "
                         "Run scripts/build_vncompress_vi_v2.py first.")

    return {'judge': stage_judge, 'sample': stage_sample,
            'merge': stage_merge}[args.stage](args, eval_path, rows)


if __name__ == '__main__':
    sys.exit(main())
