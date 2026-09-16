#!/usr/bin/env python3
"""generate_compression_pairs.py -- stage 3: the teacher-generated `compression` config.

    # what a full run would cost, without calling anything
    python scripts/generate_compression_pairs.py --plan

    # exercise the whole flow with no API key and no network
    python scripts/generate_compression_pairs.py --limit 10 --dry-run

    # the real pilot
    export VNCOMPRESS_TEACHER_API_KEY=...
    python scripts/generate_compression_pairs.py --limit 10

Implements SS3 of docs/dataset_rebuild_spec.md:

  SS3.1  the token budget is enforced AT GENERATION -- the prompt carries a hard
         `target_tokens`, the output is measured, and a miss is re-prompted at
         most twice before the sample is DROPPED. Never labelled afterwards:
         v1 labelled afterwards and ended up with 19.8% of rows inside their
         budget while its own flag claimed 88.6% (docs/dataset_v1_baseline.md).
  SS3.2  2x/4x/8x are generated, and levels that return the same text collapse
         into one row recording `stable_across_ratios`. v1 kept them as three
         rows, which is one label counted three times.
  SS3.3  answerability: a judge answers using ONLY the compression and is scored
         against the human ViQuAD answer. Rows that fail go to extras/, not train.
  SS3.4  compression is conditioned on the query.

Rate limits are expected. Every call retries with exponential backoff plus
jitter on 429/5xx, and progress is checkpointed per row, so an interrupted run
resumes instead of re-spending the budget it already spent.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vncompress.dataset_schema import MIN_SENTENCE_EXTRACTIVE
from vncompress.dataset_build import (
    budget_compliance,
    count_tokens,
    detect_lang,
    extractive_ratio,
    normalize_compression,
    numbers_preserved,
    realized_ratio,
    sentences,
    sentence_extractive_ratio,
    unsupported_numbers,
)
from vncompress.api_pool import (
    DEFAULT_ACCOUNT_RPM,
    DEFAULT_CCU_PER_KEY,
    DEFAULT_RPM,
    KeyPool,
    classify,
    collect_api_keys,
    load_dotenv,
    run_pool,
)
from vncompress.evaluation import compute_token_f1

load_dotenv()

def now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


DEFAULT_BASE_URL = 'https://token-api.fpt.ai/v1'
DEFAULT_MODEL = 'GLM-5.2'
RATIOS = (2.0, 4.0, 8.0)
MAX_REPROMPTS = 2
PROMPT_VERSION = 'v3'

# SS3.3: the compression is judged sufficient when an answer produced from it
# alone recovers the human answer this well. Token-F1 rather than exact match:
# the question is whether the information survived, not whether the judge
# happened to phrase it identically.
ANSWERABLE_F1 = 0.6

# ...but token-F1 ALONE is not that question, and measuring the first v2 run
# shows how far off it lands. 295 of the 585 rows it sent to extras/ -- 50.4%
# -- still contain the human answer verbatim in the compression. The gate was
# scoring the judge's PHRASING against a short extractive span:
#
#   Q: Chế độ nhà nước của Ả Rập Xê Út là gì?   gold: "quân chủ chuyên chế"
#   judge: "Quân chủ chuyên chế, chế độ độc tài thế tập do hoàng tộc cai trị."
#   -> F1 0.42, discarded. The judge was right and more complete than the gold.
#
#   Q: Lòng chảo nội lục của châu Á đổ vào đâu nhiều nhất?
#   gold: "biển Caspian, biển Aral và nhiều hồ nhỏ hơn"   judge: "Biển Caspian"
#   -> F1 0.36, discarded. Right, and shorter than the gold.
#
# Both directions of that asymmetry are the same mistake: F1 punishes a length
# mismatch between two correct answers. So containment is accepted as a second
# route -- if the gold answer sits inside the judge's answer, or the judge's
# answer inside the gold, the information demonstrably survived compression.
#
# The filter is not merely wasteful, it is BIASED: what survives a phrasing
# gate is the question whose answer is a short literal string, which is
# precisely the question a query-agnostic compressor also gets right. Keeping
# it would shrink the very gap E1 exists to measure.
#
# The third route, `span`, is the strictest evidence available and needs no
# judge at all: the human answer span occurs verbatim in the compression.
ANSWERABLE_SPAN_MIN_CHARS = 4  # below this, containment is coincidence


def _normalize_answer(text):
    """Casefold and collapse whitespace -- 'Thủ tướng' vs 'thủ tướng.' """
    return ' '.join((text or '').lower().split()).strip(' .,;:!?"\'()')


def answerability_verdict(predicted, gold_answer, compression, f1):
    """Did the information survive? Returns (answerable, method).

    `method` is recorded on every row so the gate's own behaviour stays
    auditable: a later run can count how much each route contributed and
    tighten or drop one without re-judging anything.
    """
    if f1 >= ANSWERABLE_F1:
        return True, 'f1'
    gold = _normalize_answer(gold_answer)
    pred = _normalize_answer(predicted)
    if len(gold) >= ANSWERABLE_SPAN_MIN_CHARS and pred:
        if gold in pred or pred in gold:
            return True, 'containment'
    if len(gold) >= ANSWERABLE_SPAN_MIN_CHARS and gold in _normalize_answer(compression):
        return True, 'span'
    return False, 'none'


COMPRESS_SYSTEM = (
    "Bạn là công cụ trích xuất ngữ cảnh tiếng Việt. Bạn CHỌN các câu nguyên văn "
    "từ văn bản để trả lời được một câu hỏi cụ thể. Bạn không bao giờ viết lại "
    "hay bổ sung nội dung của riêng mình."
)

# v3: EXTRACTIVE. v2 asked the model to "nén" (compress) and to "giữ nguyên văn
# mọi câu chữ cần thiết", and got fluent rewrites -- 973 shipped rows with a
# median `sentence_extractive_ratio` of 0.038 and 21% asserting numbers their
# context never states. "Keep the necessary wording verbatim" left the model
# free to decide that rewriting the rest was compression. It is not, for this
# project: every wave-2 arm SELECTS from the context (perplexity over tokens,
# sentences, class budgets, encoder keep/drop labels), so a gold containing
# sentences the context does not is a target none of them can ever reach.
#
# So the instruction is no longer "compress" but "choose": copy whole sentences,
# change nothing. The gate behind it (see `extractive_problem`) verifies that
# rather than trusting it.
COMPRESS_PROMPT = """Chọn các câu từ ĐOẠN VĂN để trả lời được CÂU HỎI.

RÀNG BUỘC BẮT BUỘC:
- CHỈ được chép lại NGUYÊN VĂN các câu có sẵn trong ĐOẠN VĂN.
- KHÔNG viết lại, KHÔNG tóm tắt, KHÔNG diễn giải, KHÔNG rút gọn câu.
- KHÔNG thêm bất kỳ thông tin nào không có trong ĐOẠN VĂN — kể cả khi bạn biết nó đúng.
- Nối các câu đã chọn theo đúng thứ tự xuất hiện, cách nhau một dấu cách.
- Tổng độ dài phải **khoảng {target_tokens} từ** (đếm theo khoảng trắng). Đây là ngân sách cứng.
- Chọn đúng những câu cần để trả lời câu hỏi; bỏ phần còn lại.
- Chỉ xuất các câu đã chọn. Không giải thích, không mở đầu, không đánh dấu.

CÂU HỎI:
{query}

ĐOẠN VĂN:
{context}

CÁC CÂU ĐÃ CHỌN (khoảng {target_tokens} từ, nguyên văn):"""

REPROMPT = """Bản nén vừa rồi dài {actual} từ, nhưng ngân sách là {target_tokens} từ.

Chọn lại cho đúng khoảng {target_tokens} từ. {direction}
Vẫn phải trả lời được câu hỏi, và vẫn CHỈ được chép nguyên văn các câu có trong ĐOẠN VĂN.
Chỉ xuất các câu đã chọn.

BẢN NÉN TRƯỚC:
{previous}"""

# A targeted re-prompt, because "you rewrote it" and "you invented a number"
# need different corrections and a generic "try again" earns a generic retry.
# The offending text is quoted back: naming the sentence the model invented is
# what makes the second attempt different from the first.
EXTRACTIVE_REPROMPT = """Bản nén vừa rồi KHÔNG hợp lệ: {problem}

Quy tắc: chỉ được chép NGUYÊN VĂN các câu có sẵn trong ĐOẠN VĂN, không sửa một chữ nào.
Chọn lại các câu nguyên văn từ ĐOẠN VĂN, khoảng {target_tokens} từ, vẫn trả lời được câu hỏi.
Chỉ xuất các câu đã chọn.

BẢN NÉN TRƯỚC (bản không hợp lệ):
{previous}"""

ANSWER_SYSTEM = "Bạn trả lời câu hỏi chỉ dựa trên văn bản được cung cấp."

ANSWER_PROMPT = """Chỉ dùng VĂN BẢN dưới đây, trả lời CÂU HỎI thật ngắn gọn.
Nếu văn bản không chứa đáp án, trả lời đúng hai chữ: KHÔNG CÓ.

VĂN BẢN:
{compression}

CÂU HỎI:
{query}

ĐÁP ÁN:"""


class TeacherError(RuntimeError):
    pass


class TruncatedResponse(TeacherError):
    """The model hit `max_tokens` before finishing.

    Its own failure class because the alternative is catastrophic and silent:
    a truncated compression is short, so it reads as a BUDGET MISS, gets
    re-prompted twice, misses again, and the sample is dropped. A whole run
    would report "the teacher cannot hit its budgets" when what actually
    happened is that the answer was cut off mid-sentence. Measured on GLM-5.2:
    max_tokens=16 returned finish_reason='length' with empty content and 42
    tokens of usage, all of it reasoning.
    """


# GLM-5.2 is a REASONING model: `max_tokens` covers reasoning tokens plus the
# answer, and the reasoning runs first. A one-word reply cost 499 tokens with
# thinking on and 24 with it off -- and at max_tokens=1024 a 2,000-word
# compression would be truncated before it started.
#
# Compression here is extractive rewriting against a hard budget. There is no
# reasoning for the chain of thought to do, so it buys nothing and costs a 20x
# token multiple plus the truncation risk above. Off by default, --thinking to
# restore it.
THINKING_OFF = {'thinking': {'type': 'disabled'}}

# A dropped connection is repaired where it happened. See Teacher._send for why
# this does not violate the no-in-call-retry rule that 429 lives under.
TRANSPORT_RETRIES = 3
TRANSPORT_RETRY_WAIT = 1.0

# 32k, flat. An earlier version sized this from `target_tokens` on the theory
# that a 100-word budget needs less room than a 2,000-word one. Measured on
# GLM-5.2 over this endpoint, that produced finish_reason='length' on 4 of 9
# real pilot calls: the model does not treat the requested word count as a
# ceiling, and a compression cut off mid-sentence comes back SHORT, so it reads
# as a budget miss, gets re-prompted twice, and the sample is discarded.
#
# max_tokens is a cap, not a reservation -- billing is on tokens actually
# generated -- so a high cap costs nothing when unused. Truncation costs the
# whole sample. Measured ceiling for this model is ~8.6k output tokens, so 32k
# is headroom the model will never reach, which is the point.
MAX_TOKENS_DEFAULT = 32768

# The answerability judge writes one short answer, but a model that starts
# rambling must still be allowed to finish -- a truncated answer scores a low
# F1 and gets filed as "the compression is insufficient", a verdict about the
# token budget masquerading as a verdict about the data.
ANSWER_MAX_TOKENS = 1024


class Teacher:
    """One model, many keys. Holds a client per key and nothing else stateful.

    Retry and backoff deliberately do NOT live here any more. They used to, and
    two layers of backoff -- one inside the call, one in the pool -- compound
    into waits nobody chose, and worse, an in-call retry hides a 429 from the
    pool so the key never cools and keeps being handed work. The pool owns the
    decision: cool THIS key, requeue the item, let another key take it.
    """

    def __init__(self, model, base_url, temperature=0.1, max_tokens=MAX_TOKENS_DEFAULT,
                 dry_run=False, seed=42, thinking=False, request_timeout=600.0,
                 max_connections=256, keepalive_expiry=5.0,
                 keepalive_connections=0, account_limiter=None):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.max_reprompts = MAX_REPROMPTS
        # Generous: a 2x compression of a 5,000-word context legitimately runs
        # minutes. Too short a timeout turns slow-but-working calls into 499s.
        self.request_timeout = request_timeout
        self.max_connections = max_connections
        self.keepalive_expiry = keepalive_expiry
        self.keepalive_connections = keepalive_connections
        # Held here, not read from the pool, because chat() is what has to
        # respect it and chat() is not given the pool.
        self.account_limiter = account_limiter
        self.dry_run = dry_run
        self.calls = Counter()
        self._rng = random.Random(seed)
        self._clients = {}
        self._lock = threading.Lock()

    def _client_for(self, api_key):
        """Cached per key. The OpenAI client is thread-safe; building a fresh
        one per request would open a new connection pool per call."""
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
                client = OpenAI(
                    api_key=api_key.key, base_url=self.base_url,
                    max_retries=0, timeout=self.request_timeout,
                    http_client=httpx.Client(
                        limits=httpx.Limits(
                            max_connections=self.max_connections,
                            # NO connection reuse. A pooled connection the
                            # gateway has already reaped comes back as "Server
                            # disconnected without sending a response" on its
                            # next use -- 131 lost calls in one 1,284-item
                            # batch, and the race is unwinnable from this side
                            # because the close and the reuse are concurrent.
                            #
                            # A fresh connection costs one TLS handshake, call
                            # it 100ms, against calls that measured 12-40s. Two
                            # tenths of a percent to delete an entire class of
                            # failure. The retry in api_pool.classify() stays
                            # as the backstop for the ones that still slip
                            # through (a mid-response death owes nothing to
                            # pooling).
                            max_keepalive_connections=self.keepalive_connections,
                            keepalive_expiry=self.keepalive_expiry),
                        timeout=self.request_timeout),
                )
                self._clients[api_key.index] = client
            return client

    def chat(self, system, user, api_key=None, max_tokens=None, stop=None):
        """One request, rate-limited HERE because this is where requests happen.

        The pool cannot do the counting for us: it hands out one permit per
        work item, and one item is up to four calls. A limiter set to the
        provider's 150/min was therefore admitting up to 600 -- invisible while
        a separate account-wide ceiling happened to throttle by a similar
        factor, and immediately visible as 429s once that ceiling was removed.
        """
        if self.dry_run:
            with self._lock:
                self.calls['dry_run'] += 1
            return self._fake(user)
        extra = {} if self.thinking else {'extra_body': dict(THINKING_OFF)}
        response = self._send(system, user, api_key, max_tokens, extra, stop)
        choice = response.choices[0]
        text = (choice.message.content or '').strip()
        if choice.finish_reason == 'length':
            with self._lock:
                self.calls['truncated'] += 1
            raise TruncatedResponse(
                f"finish_reason=length at max_tokens="
                f"{max_tokens or self.max_tokens} (got {len(text.split())} words)")
        return text

    def _send(self, system, user, api_key, max_tokens, extra, stop=None):
        """Send, retrying a dropped connection IN PLACE.

        The no-in-call-retry rule exists so a 429 reaches the pool: the pool
        owns cooling that key, and a retry that hides the 429 leaves a
        rate-limited key in rotation. A dropped socket carries no such state --
        there is nothing about it for the pool to learn -- and the item-level
        retry is the expensive way to handle it: one item is four calls, so
        redoing the item to repair one bad call throws away three good ones.
        Measured at a ~16% per-call drop rate that compounded to 124 items
        (8% of a batch) given up after four full re-runs each.

        429 still escapes untouched.
        """
        last = None
        for attempt in range(TRANSPORT_RETRIES + 1):
            # Inside the loop, not before it. A retry is another real request;
            # taking one permit and then sending up to four is the same
            # under-count this method exists to stop, only smaller.
            if api_key is not None:
                api_key.limiter.acquire(stop)
                if self.account_limiter is not None:
                    self.account_limiter.acquire(stop)
            with self._lock:
                self.calls['requests'] += 1
            try:
                return self._client_for(api_key).chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    messages=[{'role': 'system', 'content': system},
                              {'role': 'user', 'content': user}],
                    **extra,
                )
            except Exception as exc:                      # noqa: BLE001
                if classify(exc)[1] != 'timeout' or attempt == TRANSPORT_RETRIES:
                    raise
                last = exc
                with self._lock:
                    self.calls['transport_retry'] += 1
                time.sleep(TRANSPORT_RETRY_WAIT * (attempt + 1))
        raise last

    def _fake(self, user):
        """Offline stand-in: a real extractive compression of the right length.

        Not a fixed string -- it selects WHOLE sentences from the passage up to
        the requested budget, so the budget, dedup, answerability AND extractive
        paths all run on plausible input and a bug in them shows up in
        --dry-run rather than only once the API is being paid for.

        Whole sentences specifically: a token-count prefix ends mid-sentence, so
        its last fragment is not verbatim in the context and the v3 extractive
        gate rejects it. A dry run where every sample is dropped tests the drop
        path and nothing else.

        The passage is read between the two markers rather than up to a fixed
        trailing string -- the v2 version split on '\\n\\nBẢN NÉN', which the v3
        prompt no longer contains, and so silently fed the prompt's own trailing
        instructions back as if they were the passage.
        """
        if user.rstrip().endswith('JSON:'):
            # Stage 2b asks for question/answer pairs. The fake quotes real
            # sentences out of the passage as answers, so the span-verification
            # path in generate_qa_from_corpus.py runs for real in --dry-run
            # instead of being skipped by output it always rejects.
            passage = user.split('ĐOẠN VĂN:\n')[1].rsplit('\n\nJSON:', 1)[0]
            picked = [s.strip() for s in re.split(r'(?<=[.!?])\s+', passage.strip())
                      if 4 <= len(s.split()) <= 25][:int(user.split('đặt ')[1].split(' câu')[0])]
            return json.dumps([{'question': f"Câu hỏi {i + 1} về đoạn văn là gì?",
                                'answer': sentence}
                               for i, sentence in enumerate(picked)], ensure_ascii=False)
        if 'ĐOẠN VĂN:' in user:
            target = int(user.split('khoảng ')[1].split(' từ')[0])
            passage = user.split('ĐOẠN VĂN:\n')[1]
            # Everything after the passage is the closing instruction line,
            # which always follows a blank line and ends in a colon.
            passage = re.split(r'\n\n[^\n]*:$', passage, flags=re.MULTILINE)[0]
            picked, used = [], 0
            for sentence in re.split(r'(?<=[.!?])\s+', passage.strip()):
                length = len(sentence.split())
                if not length:
                    continue
                if used + length > target:
                    break
                picked.append(sentence)
                used += length
            if picked:
                return ' '.join(picked)
            # No whole sentence fits -- a real teacher hits this at 8x on a
            # passage of long sentences. Fall back to a token prefix, which is
            # still a verbatim substring of the passage (so the extractive gate
            # is satisfied) and still respects the budget (so the happy path,
            # not the drop path, is what --dry-run exercises).
            return ' '.join(passage.split()[:target])
        if 'VĂN BẢN:' in user:
            return user.split('VĂN BẢN:\n')[1].split('\n\nCÂU HỎI')[0][:200]
        return ''


def extractive_problem(context, text):
    """Why this output is not an extraction of `context`, or None if it is.

    Two independent ways to fail, reported as prose because the string is fed
    straight back to the model:

      rewriting   -- fewer than MIN_SENTENCE_EXTRACTIVE of its sentences occur
                     verbatim in the context;
      fabrication -- it states a number the context never states.

    Checked separately from the budget because they are separately fixable,
    and because a run that drops samples needs to say which of the two it is
    dropping them for. v2 measured neither, which is how a median
    sentence-extractive ratio of 0.038 shipped as "compression".
    """
    if not (text or '').strip():
        return None
    ratio = sentence_extractive_ratio(context, text)
    if ratio is not None and ratio < MIN_SENTENCE_EXTRACTIVE:
        offenders = [s for s in sentences(text) if s not in (context or '')]
        example = offenders[0][:120] if offenders else ''
        return (f"có câu không xuất hiện nguyên văn trong ĐOẠN VĂN "
                f"(chỉ {ratio:.0%} số câu là nguyên văn). Ví dụ câu bị viết lại: \"{example}\"")
    invented = unsupported_numbers(context, text)
    if invented:
        return (f"có số liệu không hề xuất hiện trong ĐOẠN VĂN: "
                f"{', '.join(invented[:5])}")
    return None


def compress_to_budget(teacher, context, query, target_tokens, api_key=None,
                       max_reprompts=MAX_REPROMPTS, stop=None,
                       require_extractive=True):
    """SS3.1: generate, measure, re-prompt at most twice, else give up.

    Returns (text, attempts, achieved, problem); `text` is None when the sample
    was dropped and `problem` says it was dropped for fabrication rather than
    for the budget.
    Giving up is the point: a sample kept after three misses is a sample whose
    label says one thing and whose text says another.

    Records nothing itself. It used to take a `stats` counter and increment it,
    and so did the caller -- so every drop was counted twice (a 9-item pilot
    reported 12 drops), and the increments happened off the results lock while
    40 threads shared the counter. Outcomes are derived from the return value
    by the one caller that holds the lock.
    """
    prompt = COMPRESS_PROMPT.format(target_tokens=target_tokens, query=query, context=context)
    text = teacher.chat(COMPRESS_SYSTEM, prompt, api_key, stop=stop)
    actual = 0
    for attempt in range(max_reprompts + 1):
        actual = count_tokens(text)
        budget_ok = budget_compliance(actual, target_tokens)
        problem = extractive_problem(context, text) if require_extractive else None
        if budget_ok and not problem:
            return text, attempt + 1, actual / max(target_tokens, 1), None
        if attempt == max_reprompts:
            break
        # Budget first: a wildly over-length answer is usually also a rewrite,
        # and correcting one thing per turn is what keeps the re-prompt
        # specific enough to act on.
        if not budget_ok:
            direction = ("Bản nén đang quá dài, hãy rút ngắn."
                         if actual > target_tokens else "Bản nén đang quá ngắn, hãy bổ sung chi tiết liên quan.")
            text = teacher.chat(COMPRESS_SYSTEM, REPROMPT.format(
                actual=actual, target_tokens=target_tokens, direction=direction,
                previous=text), api_key, stop=stop)
        else:
            text = teacher.chat(COMPRESS_SYSTEM, EXTRACTIVE_REPROMPT.format(
                problem=problem, target_tokens=target_tokens, previous=text),
                api_key, stop=stop)
    # The miss magnitude, not just the fact of it. "Off by 18%" and "off by
    # 300%" call for different fixes -- one more re-prompt versus a different
    # tolerance -- and without the number the choice between them is a guess.
    # `problem` distinguishes the two ways to fail, so the run reports how much
    # of the drop rate is budget and how much is fabrication.
    final_problem = extractive_problem(context, text) if require_extractive else None
    if budget_compliance(actual, target_tokens) and final_problem:
        return None, max_reprompts + 1, actual / max(target_tokens, 1), final_problem
    return None, max_reprompts + 1, actual / max(target_tokens, 1), None


def judge_answerability(teacher, compression, query, gold_answer, api_key=None, stop=None):
    """SS3.3: can the question still be answered from the compression alone?

    Returns (answerable, f1, predicted, method). `f1` stays the raw score --
    it is the diagnostic, not the verdict -- while `method` says which of the
    three routes in `answerability_verdict` accepted the row.
    """
    predicted = teacher.chat(ANSWER_SYSTEM, ANSWER_PROMPT.format(
        compression=compression, query=query), api_key,
        max_tokens=ANSWER_MAX_TOKENS, stop=stop)
    f1 = compute_token_f1([predicted], [gold_answer])
    answerable, method = answerability_verdict(predicted, gold_answer, compression, f1)
    return bool(answerable), f1, predicted, method


def build_row(source_row, ratio, gold, target_tokens, attempts, answerable, f1,
              predicted, teacher, dry_run=False, method='f1'):
    """One (source row, ratio) pair -> one compression row.

    `stable_across_ratios` starts as a single-element list and GROWS in
    `deduplicate`, which is where levels that produced identical text get
    merged. Collapsing them here instead -- the previous design -- could only
    ever see one source row's three levels, and so missed the more common
    duplicate: two different questions about the same document converging on
    the same compression.
    """
    ratio_levels = [ratio]
    context = source_row['context']
    realized = count_tokens(gold)
    return {
        'id': f"comp_{source_row['id']}_{int(min(ratio_levels))}x",
        'doc_id': source_row['doc_id'],
        'source': source_row.get('source'),
        'domain': source_row.get('domain', 'other'),
        'split': source_row['split'],
        'context': context,
        'context_lang': detect_lang(context),
        'query': source_row.get('query', ''),
        'query_lang': detect_lang(source_row.get('query', '')),
        'gold_compression': gold,
        'requested_ratio': min(ratio_levels),
        'stable_across_ratios': sorted(ratio_levels),
        'target_tokens': target_tokens,
        'realized_tokens': realized,
        'realized_ratio': realized_ratio(context, gold),
        'budget_compliance': budget_compliance(realized, target_tokens),
        'extractive_ratio': extractive_ratio(context, gold),
        # Token overlap says the words are familiar; these two say whether the
        # text was QUOTED and whether it stayed inside the evidence. Shipped v2
        # scored 0.933 on the first and 0.300 on the second, and asserted
        # unsupported numbers on 21% of rows -- a gap only visible once all
        # three are on the row. See vncompress/dataset_build.py.
        'sentence_extractive_ratio': sentence_extractive_ratio(context, gold),
        'unsupported_numbers': unsupported_numbers(context, gold),
        'numbers_preserved': numbers_preserved(context, gold),
        'answerable_from_compression': answerable,
        'answerability_f1': round(f1, 4),
        'answerability_method': method,
        'answerability_prediction': predicted,
        # Stage 4 has not run on this row. The field exists here, and not only
        # in eval/test.jsonl, because the dataset card claims a
        # `verification_method` for every row -- and a field the card
        # describes but the data omits is a claim nobody can check.
        'verification_method': 'none',
        'teacher': teacher.model,
        'gen_config': {
            'base_url': teacher.base_url, 'temperature': teacher.temperature,
            'max_tokens': teacher.max_tokens, 'prompt_version': PROMPT_VERSION,
            'max_reprompts': teacher.max_reprompts, 'attempts': attempts,
            'dry_run': bool(dry_run),
            # WHICH model judged answerability. Shipped v2 recorded the teacher
            # and the endpoint but never the judge, so a reader cannot tell
            # whether the teacher graded its own work -- here, it does, and
            # saying so plainly is the point. scripts/verify_eval_set.py is
            # where an independent judge belongs, and it refuses teacher==judge.
            'judge_model': teacher.model,
            'judge_is_teacher': True,
            'answerable_f1_threshold': ANSWERABLE_F1,
        },
    }


def load_sources(input_dir, splits, include_synthetic=False):
    """Source QA rows for compression.

    `qa_synthetic.jsonl` (stage 2b) is opt-in rather than picked up
    automatically: those questions are teacher-written, and a stage that
    silently absorbs them makes "which rows came from a human" a question
    nobody can answer afterwards. Requesting them is one flag; discovering
    later that a run used them is a re-run.

    Synthetic rows are refused on the test split even when asked for -- the
    file should never contain any (stage 2b will not emit one), and the two
    places that guarantee it are cheaper than the one report that would have
    to be withdrawn.
    """
    path = os.path.join(input_dir, 'qa.jsonl')
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run scripts/build_vncompress_vi_v2.py first.")
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if include_synthetic:
        synthetic_path = os.path.join(input_dir, 'qa_synthetic.jsonl')
        if not os.path.exists(synthetic_path):
            raise SystemExit(
                f"--include-synthetic-qa was passed but {synthetic_path} does not exist. "
                "Run scripts/generate_qa_from_corpus.py first.")
        with open(synthetic_path, encoding='utf-8') as f:
            synthetic = [json.loads(line) for line in f if line.strip()]
        leaked = [r for r in synthetic if r.get('split') == 'test']
        if leaked:
            raise SystemExit(
                f"{synthetic_path} contains {len(leaked)} row(s) on the TEST split. "
                "Teacher-written questions must never reach the independent test set "
                "(B5/E3); refusing to continue.")
        print(f"Including {len(synthetic):,} teacher-written QA row(s) from qa_synthetic.jsonl")
        rows += synthetic

    return [r for r in rows if r.get('split') in splits and r.get('query') and r.get('answer')]


def checkpoint_path_for(input_dir, dry_run):
    """Dry-run and real runs must never share a checkpoint.

    They did. A `--dry-run` wrote its fabricated rows into the shared
    checkpoint, and the next real run opened with "Resuming: 3 source row(s)
    already processed" and shipped the fabrications -- a dry run silently
    consuming the budget of the real one and poisoning its output. Separate
    files, and `mode` recorded in every record as a second line of defence
    against a stale file from before this fix.
    """
    name = 'compression_checkpoint.dryrun.jsonl' if dry_run else 'compression_checkpoint.jsonl'
    return os.path.join(input_dir, 'provenance', name)


def load_checkpoint(path, dry_run):
    done, rows, rejected = set(), [], 0
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if bool(record.get('dry_run')) != bool(dry_run):
                    rejected += 1
                    continue
                # Keyed per (source row, ratio): the unit of work is one ratio,
                # so resuming must not re-spend the two levels that already
                # succeeded because the third failed.
                done.add(record.get('work_id') or record['source_id'])
                rows.append(record)
    if rejected:
        print(f"Ignored {rejected:,} checkpoint record(s) from the other run mode")
    return done, rows


def deduplicate(rows, stats):
    """B2/P2: one (doc_id, gold_compression) may appear once, run-wide.

    Collapsing only within a source row -- which is all the per-row `by_text`
    map can do -- leaves the far more likely duplicate untouched: two DIFFERENT
    questions about the same document whose answers live in the same passage
    converge on the same compression. Measured on the v2 eval set, 271 of 1,000
    rows share a (doc_id, gold_compression) with another row, so this is the
    common case, not the corner case. Ratio levels merge into the survivor so
    no information about which budgets produced the text is lost.
    """
    kept, index = [], {}
    for row in rows:
        key = (row.get('doc_id'), normalize_compression(row.get('gold_compression')))
        if key in index:
            survivor = index[key]
            merged = set(survivor.get('stable_across_ratios') or []) | \
                set(row.get('stable_across_ratios') or [])
            survivor['stable_across_ratios'] = sorted(merged)
            survivor['requested_ratio'] = min(merged) if merged else survivor.get('requested_ratio')
            stats['rows merged (duplicate across source rows)'] += 1
            continue
        index[key] = row
        kept.append(row)
    return kept


def shard_dir(input_dir, dry_run=False):
    """Per-batch results live under provenance/, NOT under `compression/`.

    The validator globs `<input>/compression*.jsonl` and `<input>/compression/*.jsonl`,
    so a directory named `compression/` would be read as the config itself --
    and shards still contain the cross-batch duplicates that `deduplicate` has
    not seen yet. Scoring a dataset on its own un-merged intermediates is
    exactly the kind of thing the validator exists to catch, so the
    intermediates are kept somewhere it does not look.

    Dry-run shards go somewhere else again. The checkpoint already learned this
    lesson the hard way; adding shards re-opened the same hole, because a
    rehearsal that writes fabricated batches into the real provenance directory
    gets them merged into the real config by the next genuine run.
    """
    name = 'shards_dryrun' if dry_run else 'shards'
    return os.path.join(input_dir, 'provenance', name)


def manifest_path(input_dir, dry_run=False):
    name = 'batch_manifest.dryrun.json' if dry_run else 'batch_manifest.json'
    return os.path.join(input_dir, 'provenance', name)


def load_manifest(input_dir, dry_run=False):
    path = manifest_path(input_dir, dry_run)
    if not os.path.exists(path):
        return {'batches': {}}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_manifest(input_dir, manifest, dry_run=False):
    path = manifest_path(input_dir, dry_run)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    # Atomic: a manifest half-written by a kill is a manifest that lies about
    # which batches are done, and the next run would skip work it never did.
    os.replace(tmp, path)


def write_shard(input_dir, index, rows, dry_run=False):
    os.makedirs(shard_dir(input_dir, dry_run), exist_ok=True)
    path = os.path.join(shard_dir(input_dir, dry_run), f'batch-{index:04d}.jsonl')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def read_shards(input_dir, dry_run=False):
    """Every shard written so far, in batch order."""
    rows = []
    for path in sorted(glob.glob(os.path.join(shard_dir(input_dir, dry_run), 'batch-*.jsonl'))):
        with open(path, encoding='utf-8') as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def merge_shards(input_dir, output_dir, stats=None, dry_run=False):
    """Rebuild the merged configs from every shard on disk.

    Run after EVERY batch, not once at the end. A 15-hour job that merges only
    on completion has no usable output at hour 14 -- and the reason to shard at
    all is that hour 14 is exactly when something goes wrong.

    Dedup happens HERE, across all shards together. Doing it per batch would
    miss the duplicate this is most likely to produce: two questions about the
    same document, landing in different batches, compressed to the same text.
    """
    stats = stats if stats is not None else Counter()
    rows = sorted(read_shards(input_dir, dry_run),
                  key=lambda r: (str(r.get('id')), r.get('requested_ratio') or 0))
    rows = deduplicate(rows, stats)
    keep = [r for r in rows if r['answerable_from_compression']]
    reject = [r for r in rows if not r['answerable_from_compression']]

    os.makedirs(output_dir, exist_ok=True)
    compression_path = os.path.join(output_dir, 'compression.jsonl')
    tmp = compression_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for row in keep:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, compression_path)

    reject_path = os.path.join(output_dir, 'extras', 'compression_unanswerable.jsonl')
    os.makedirs(os.path.dirname(reject_path), exist_ok=True)
    tmp = reject_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for row in reject:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, reject_path)
    update_checksums(input_dir, [compression_path, reject_path])
    return keep, reject


def update_checksums(input_dir, paths):
    """Fold this stage's outputs into provenance/CHECKSUMS.json.

    Stage 2 writes the manifest for the files IT produces -- corpus, qa,
    eval/test -- and then stage 3 writes two more files beside them and says
    nothing. The shipped v2 manifest therefore covered three files and omitted
    both that the teacher had just generated: the only two whose contents
    depend on a paid, non-deterministic run, and so the only two where a silent
    change is plausible. SS8 asks for CHECKSUMS.json over the hand-off, not
    over part of it.

    Merged rather than rewritten, because stage 2's entries must survive, and
    re-hashing its multi-hundred-megabyte corpus on every batch merge would
    make the per-batch merge the slowest thing in the run.
    """
    manifest_path = os.path.join(input_dir, 'provenance', 'CHECKSUMS.json')
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding='utf-8') as f:
                manifest = json.load(f)
        except (OSError, ValueError):
            # A corrupt or half-written manifest must not take the run down --
            # the rows are the deliverable. It is rebuilt from what we know.
            manifest = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            data = f.read().replace(b'\r\n', b'\n')
        # Forward slashes always. The manifest is compared across machines, and
        # a key written `eval\test.jsonl` on Windows never matches the same file
        # recorded as `eval/test.jsonl` on Linux -- the path equivalent of the
        # CRLF mismatch `_normalized_bytes` exists to prevent.
        key = os.path.relpath(path).replace(os.sep, '/')
        manifest[key] = {
            'sha256': hashlib.sha256(data).hexdigest(),
            'size_bytes': len(data),
        }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    tmp = manifest_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)


def make_worker(teacher, stats, stats_lock, max_reprompts=MAX_REPROMPTS,
                require_extractive=True):
    """The unit of work: one (source row, ratio) -> one row, or None if dropped.

    Deliberately ONE ratio, not one source row. A source row is ~12 calls; when
    a key 429s on the last of them the pool requeues the item and all twelve are
    paid for again. A ratio is ~4, so a requeue costs a third as much -- and it
    triples the number of independent items, which is what keeps 40 workers fed.

    Exceptions are allowed to escape: that is the signal the pool cools this key
    on. Swallowing them here would leave a sick key in rotation.
    """
    def work(item, api_key, stop):
        source, ratio = item
        original = count_tokens(source['context'])
        target = max(1, int(original / ratio))
        try:
            gold, attempts, achieved, problem = compress_to_budget(
                teacher, source['context'], source['query'], target, api_key,
                max_reprompts=max_reprompts, stop=stop,
                require_extractive=require_extractive)
        except TruncatedResponse as exc:
            # Not requeued: another key would truncate identically. Dropped with
            # its own reason so the run report distinguishes "the teacher missed
            # the budget" from "we never let it finish".
            with stats_lock:
                stats['dropped: response truncated at max_tokens'] += 1
            return None
        if gold is None:
            with stats_lock:
                if problem:
                    # Budget was fine; the model would not stop rewriting. A
                    # separate line because the fix is a prompt, not a wider
                    # tolerance -- and because this rate is the honest cost of
                    # demanding extraction, which the run should report.
                    stats['dropped: not extractive after re-prompts'] += 1
                    stats['  invented a number' if 'số liệu' in problem
                          else '  rewrote the sentences'] += 1
                else:
                    stats['dropped: budget never met'] += 1
                    # Bucketed, because the fix depends on the size of the miss.
                    stats['  miss within 1.5x of target' if 0.66 <= achieved <= 1.5
                          else '  miss within 3x of target' if 0.33 <= achieved <= 3.0
                          else '  miss beyond 3x of target'] += 1
            return None
        with stats_lock:
            stats[f'budget met on attempt {attempts}'] += 1
        answerable, f1, predicted, method = judge_answerability(
            teacher, gold, source['query'], source['answer'], api_key, stop=stop)
        with stats_lock:
            stats['answerable' if answerable else 'not answerable'] += 1
            # Per-route counts, so the run reports how much each acceptance
            # rule contributed rather than leaving the gate a black box.
            stats[f'  answerable via {method}' if answerable
                  else '  rejected by every route'] += 1
        return build_row(source, ratio, gold, target, attempts, answerable, f1,
                         predicted, teacher, teacher.dry_run, method=method)
    return work


def build_pool(args, prefix='VNCOMPRESS_TEACHER_API_KEY'):
    """Assemble the key pool, and say plainly what it found.

    Silence here is expensive: a run that quietly fell back to one key takes
    four times as long, and the only symptom is that it is slow.
    """
    keys = collect_api_keys(prefix)
    if args.api_key and args.api_key not in keys:
        keys.insert(0, args.api_key)
    if not keys:
        raise SystemExit(
            f"No API keys. Set {prefix}_1..{prefix}_4 in .env (see .env.example),\n"
            f"or {prefix} for a single key, or pass --api-key.\n"
            "To exercise the flow without one: --dry-run")
    pool = KeyPool(keys, rpm=args.rpm, ccu_per_key=args.ccu_per_key,
                   account_rpm=args.account_rpm)
    print(f"Keys: {len(keys)} distinct · {args.ccu_per_key} concurrent each "
          f"· {args.rpm} req/min each -> ~{pool.theoretical_rpm()} req/min"
          + (f" (account ceiling {args.account_rpm})" if args.account_rpm else ""))
    if len(keys) == 1:
        print("  NOTE: one key only. Add _2.._4 to .env to run 4x wider.")
    return pool


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default='data/vncompress_vi_v2')
    ap.add_argument('--output', default=None,
                    help='Where the compression config is written. Defaults to --input, '
                         'or <input>/_dryrun with --dry-run so a rehearsal cannot overwrite '
                         'the real dataset.')
    ap.add_argument('--limit', type=int, default=None, help='Number of source rows to process')
    ap.add_argument('--batch-size', type=int, default=500,
                    help='Source rows per batch. Each batch is written to its own shard '
                         'and the merged configs are rebuilt, so a failure costs one '
                         'batch rather than the whole run.')
    ap.add_argument('--merge-only', action='store_true',
                    help='Rebuild compression.jsonl from the shards on disk. No API calls.')
    ap.add_argument('--splits', default='train,validation')
    ap.add_argument('--model', default=os.environ.get('VNCOMPRESS_TEACHER_MODEL', DEFAULT_MODEL))
    ap.add_argument('--base-url', default=os.environ.get('VNCOMPRESS_TEACHER_BASE_URL', DEFAULT_BASE_URL))
    ap.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY'),
                    help='A single extra key, on top of whatever .env provides')
    ap.add_argument('--rpm', type=int, default=int(os.environ.get('VNCOMPRESS_RPM', DEFAULT_RPM)),
                    help='Provider rate limit PER KEY, requests/minute')
    ap.add_argument('--account-rpm', type=int,
                    default=int(os.environ.get('VNCOMPRESS_ACCOUNT_RPM', DEFAULT_ACCOUNT_RPM)),
                    help='Ceiling shared by every key. 0 disables. Correct whether the '
                         "provider's 150 RPM is per key or per account.")
    ap.add_argument('--request-timeout', type=float, default=600.0,
                    help='Per-call timeout in seconds. Too short turns slow-but-working '
                         'compressions into 499s.')
    ap.add_argument('--ccu-per-key', type=int,
                    default=int(os.environ.get('VNCOMPRESS_CCU_PER_KEY', DEFAULT_CCU_PER_KEY)),
                    help='In-flight requests per key')
    ap.add_argument('--max-item-retries', type=int, default=4,
                    help='Times an item may be requeued onto another key before it is dropped')
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--max-reprompts', type=int, default=MAX_REPROMPTS,
                    help='Times the teacher is told it missed the budget before the '
                         'sample is dropped (SS3.1)')
    ap.add_argument('--include-synthetic-qa', action='store_true',
                    help='Also read qa_synthetic.jsonl (stage 2b, teacher-written '
                         'questions over corpus documents). Off by default so a run '
                         'never mixes human and generated questions unasked.')
    ap.add_argument('--allow-abstractive', action='store_true',
                    help='Drop the extractive gate: accept a gold that rewrites the '
                         'context instead of quoting it. This is what v2 did, and it '
                         'produces data no wave-2 arm can be supervised on -- use it '
                         'only to rebuild the abstractive config on purpose.')
    ap.add_argument('--max-tokens', type=int, default=MAX_TOKENS_DEFAULT,
                    help='Output cap per call. A cap, not a reservation: billing is on '
                         'tokens generated, so high costs nothing and truncation costs '
                         'the whole sample.')
    ap.add_argument('--thinking', action='store_true',
                    help='Leave the reasoning trace ON. GLM-5.2 spends ~20x the tokens '
                         'with it enabled and compression has nothing to reason about.')
    ap.add_argument('--dry-run', action='store_true', help='No API key, no network')
    ap.add_argument('--plan', action='store_true', help='Print the cost of a full run and exit')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    sources = load_sources(args.input, set(args.splits.split(',')),
                           args.include_synthetic_qa)
    calls_per_row = len(RATIOS) * (1 + MAX_REPROMPTS) + len(RATIOS)  # worst case: compress + judge

    if args.plan:
        keys = collect_api_keys('VNCOMPRESS_TEACHER_API_KEY') or ['(none)']
        rpm = max(1, len([k for k in keys if k != '(none)'])) * args.rpm
        print(f"Source rows available ({args.splits}): {len(sources):,}")
        print(f"Ratios per row: {list(RATIOS)} · re-prompts allowed: {MAX_REPROMPTS}")
        print(f"Work items (source x ratio): {len(sources) * len(RATIOS):,}")
        print(f"API calls per row: {len(RATIOS) * 2} best case, {calls_per_row} worst case")
        print(f"Full run: {len(sources) * len(RATIOS) * 2:,} - {len(sources) * calls_per_row:,} calls")
        print(f"\nKeys found in .env: {len([k for k in keys if k != '(none)'])} "
              f"· ceiling ~{rpm} req/min")
        for n in (10, 100, 1000, 5000):
            if n <= len(sources):
                low, high = n * len(RATIOS) * 2, n * calls_per_row
                print(f"  --limit {n:>5}: {low:>8,} - {high:>8,} calls "
                      f"= {low / rpm:>6.0f} - {high / rpm:>6.0f} min")
        return 0

    if args.limit:
        sources = sources[:args.limit]
    output_dir = args.output or (os.path.join(args.input, '_dryrun') if args.dry_run else args.input)
    if args.dry_run and output_dir == args.input:
        raise SystemExit(
            "--dry-run must not write into the real dataset directory.\n"
            "Drop --output, or point it somewhere else.")

    if args.merge_only:
        keep, reject = merge_shards(args.input, output_dir, dry_run=args.dry_run)
        print(f"Merged {len(read_shards(args.input, args.dry_run)):,} shard row(s) -> "
              f"{len(keep):,} compression + {len(reject):,} unanswerable")
        return 0

    checkpoint_path = checkpoint_path_for(args.input, args.dry_run)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    done, cached = load_checkpoint(checkpoint_path, args.dry_run)
    by_work_id = {record['work_id']: record['row'] for record in cached
                  if record.get('work_id') and record.get('row')}

    pool = (KeyPool(['dry-run'], rpm=10 ** 6, ccu_per_key=args.ccu_per_key, account_rpm=0)
            if args.dry_run else build_pool(args))
    teacher = Teacher(args.model, args.base_url, args.temperature, args.max_tokens,
                      dry_run=args.dry_run, seed=args.seed, thinking=args.thinking,
                      request_timeout=args.request_timeout,
                      max_connections=max(64, len(pool.keys) * args.ccu_per_key * 2),
                      account_limiter=pool.account_limiter)
    teacher.max_reprompts = args.max_reprompts

    manifest = load_manifest(args.input, args.dry_run)
    batches = [sources[i:i + args.batch_size]
               for i in range(0, len(sources), args.batch_size)]
    print(f"Teacher: {args.model} @ {args.base_url}" + (" [DRY RUN]" if args.dry_run else ""))
    print(f"{len(sources):,} source row(s) in {len(batches)} batch(es) of {args.batch_size}")
    print(f"Shards: {shard_dir(args.input, args.dry_run)}  ·  merged after every batch\n")

    stats, stats_lock = Counter(), threading.Lock()
    checkpoint = open(checkpoint_path, 'a', encoding='utf-8')
    write_lock = threading.Lock()
    error_log_path = os.path.join(
        args.input, 'provenance',
        'api_errors.dryrun.jsonl' if args.dry_run else 'api_errors.jsonl')
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
                dict(info, at=now_iso(), item=f"{item[0]['id']}:{item[1]:g}"),
                ensure_ascii=False) + "\n")
            error_log.flush()

    try:
        for index, batch in enumerate(batches):
            key = str(index)
            if manifest['batches'].get(key, {}).get('status') == 'done':
                print(f"[batch {index + 1}/{len(batches)}] already done, skipping")
                continue

            items = [(source, ratio) for source in batch for ratio in RATIOS
                     if f"{source['id']}:{ratio:g}" not in done]
            batch_rows = [by_work_id[f"{source['id']}:{ratio:g}"]
                          for source in batch for ratio in RATIOS
                          if f"{source['id']}:{ratio:g}" in by_work_id]
            started = time.monotonic()
            print(f"[batch {index + 1}/{len(batches)}] {len(items):,} item(s) to run"
                  + (f" ({len(batch_rows):,} already checkpointed)" if batch_rows else ""))

            def on_result(item, row, _rows=batch_rows):
                if row is None:
                    return
                source, ratio = item
                work_id = f"{source['id']}:{ratio:g}"
                with write_lock:
                    checkpoint.write(json.dumps(
                        {'source_id': source['id'], 'work_id': work_id,
                         'dry_run': bool(args.dry_run), 'row': row},
                        ensure_ascii=False) + "\n")
                    checkpoint.flush()
                _rows.append(row)

            def on_giveup(item, exc):
                with stats_lock:
                    stats['dropped: all keys failed the item'] += 1

            interrupted = False
            if items:
                try:
                    _, pool_stats = run_pool(
                        items, make_worker(teacher, stats, stats_lock, args.max_reprompts,
                                           not args.allow_abstractive),
                        pool, on_result=on_result, on_giveup=on_giveup, on_error=on_error,
                        max_item_retries=args.max_item_retries,
                        progress_every=max(1, len(items) // 10),
                        # One item is up to 4 calls; Teacher.chat() holds the
                        # limiter so the count matches what the provider counts.
                        rate_limit_per_item=False)
                    stats.update(pool_stats)
                except KeyboardInterrupt:
                    interrupted = True

            # Shard first, manifest second. If the process dies between them the
            # batch simply reruns -- the checkpoint makes that cheap. The reverse
            # order would mark a batch done whose rows were never written.
            write_shard(args.input, index, batch_rows, args.dry_run)
            elapsed = time.monotonic() - started
            # 'partial' is the whole point of recording the status. An
            # interrupted batch used to be written as 'done', so the rows that
            # never ran were skipped by every later resume and vanished without
            # appearing anywhere as missing. The shard is still written -- those
            # rows were paid for -- but the batch stays claimable.
            manifest['batches'][key] = {
                'status': 'partial' if interrupted else 'done',
                'source_rows': len(batch), 'rows': len(batch_rows),
                'items_run': len(items), 'seconds': round(elapsed, 1),
                'finished_at': now_iso(),
            }
            save_manifest(args.input, manifest, args.dry_run)
            if interrupted:
                raise KeyboardInterrupt

            keep, reject = merge_shards(args.input, output_dir, dry_run=args.dry_run)
            remaining = len(batches) - index - 1
            eta = f", ETA {remaining * elapsed / 60:.0f} min" if remaining and elapsed else ""
            print(f"[batch {index + 1}/{len(batches)}] {len(batch_rows):,} row(s) in "
                  f"{elapsed / 60:.1f} min -> merged total: {len(keep):,} compression, "
                  f"{len(reject):,} unanswerable{eta}")
            print(f"    keys: {pool.summary()}\n")
    except KeyboardInterrupt:
        print("\nInterrupted. Shards and manifest kept -- re-run to resume.")
    finally:
        checkpoint.close()
        error_log.close()

    keep, reject = merge_shards(args.input, output_dir, stats, args.dry_run)
    print("\n" + "=" * 60)
    print(f"Output: {output_dir}" + ("  [DRY RUN -- not the real dataset]" if args.dry_run else ""))
    print(f"compression.jsonl: {len(keep):,} rows")
    print(f"extras/compression_unanswerable.jsonl: {len(reject):,} rows (kept for analysis, not train)")
    print(f"shards: {len(manifest['batches'])} batch file(s) in "
          f"{shard_dir(args.input, args.dry_run)}")
    for key, n in stats.most_common():
        print(f"  {key}: {n:,}")
    # Reported, not just counted. Dropped connections are now repaired inside
    # the call, which means they stop appearing in the pool's error log -- and
    # a failure that is absorbed into eventual success and never printed is a
    # failure nobody can see getting worse. This line is how "the endpoint is
    # fine" stays distinguishable from "the endpoint is dropping a third of our
    # requests and we are paying to send them twice".
    sent = teacher.calls.get('requests', 0)
    retried = teacher.calls.get('transport_retry', 0)
    if sent:
        print(f"  api calls sent: {sent:,}  "
              f"(of which re-sent after a dropped connection: {retried:,} = "
              f"{retried / sent:.1%})")
    if teacher.calls.get('truncated'):
        print(f"  truncated at max_tokens: {teacher.calls['truncated']:,}")
    if not args.dry_run:
        print(f"  keys: {pool.summary()}")
    if keep:
        compliant = sum(1 for r in keep if r['budget_compliance'])
        print(f"  budget_compliance: {compliant}/{len(keep)} = {compliant / len(keep):.1%}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
