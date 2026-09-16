"""Multi-key API fan-out: a pool of keys, each with its own rate budget.

One key on this provider allows ~50 requests/minute. A full compression run is
tens of thousands of calls, so a single key turns a two-hour job into a
two-day one. Four keys used properly do not just multiply throughput -- they
also decouple failures: a key that starts returning 429 stops being asked
while the other three keep working, instead of stalling the whole run.

WHAT IS PER-KEY AND WHY

  Rate limit   Per key. The provider counts against the key, so one shared
               limiter would either throttle four keys to one key's budget or
               (worse) let four keys burst into one key's quota.
  Concurrency  Per key. 10 in flight each; the cap is what keeps a burst from
               becoming the 429 the limiter is there to avoid.
  Cool-down    Per key, EXPONENTIAL. A key that 429s is not a run that failed;
               it is one key that needs a minute. Cooling that key alone is the
               entire reason to hold four.
  Retry        Per ITEM, not per key. Work that a cooling key could not finish
               goes back on the queue and is picked up by whichever key is
               healthy next, so a cool-down costs latency and never coverage.

THE FAILURE THIS IS BUILT AROUND

A dead key (revoked, out of quota, typo'd) returns 401 on every call, instantly.
Treated as retryable, it drains the whole queue at full speed and fails every
item -- fastest possible way to lose a run. Auth errors therefore disable the
key permanently and immediately; only transient statuses cool down.
"""
from __future__ import annotations

import os
import queue
import random
import threading
import time
from collections import Counter, deque
from typing import Callable, Iterable, List, Optional

# ONLY these two are retried. Everything else is logged and left for analysis.
#
# The narrow list is deliberate. A broad retry policy hides the shape of the
# failures behind eventual success, and the first real run produced 52
# APIConnectionError in one batch -- a number worth SEEING rather than
# absorbing. Nothing is lost by not retrying either: a failed item is never
# written to the checkpoint, so the next run picks it up again. Not retrying
# costs a resume; retrying costs the evidence.
RETRYABLE_STATUS = (429,)

# 499 is nginx's "client closed request" -- the connection cut before the server
# answered. Compression calls run ~40s with a 32k output cap, squarely in the
# range where proxies give up, so this is a routine event rather than an error
# in any useful sense.
#
# Separate from 429 because the BLAME differs. A 429 is the key's fault, so
# parking that key is right. A 499 is the request's or the network's, and
# parking a healthy key for it removes capacity for nothing.
TIMEOUT_STATUS = (499,)

AUTH_STATUS = (401, 403)

# A 499 that never got far enough to be given a number.
#
# When the gateway hangs up, `openai` raises APIConnectionError -- which carries
# NO status, because no HTTP response ever arrived to put one in. Underneath,
# httpx says what actually happened: "Server disconnected without sending a
# response." That is the same event as a 499 (connection cut before an answer),
# so it is classified the same way; refusing to retry it because the wire gave
# us no integer would be reading the policy off the transport rather than off
# the failure.
#
# It is also the one failure that is unconditionally safe to repeat: "without
# sending a response" means the request was never answered, and a compression
# call has no side effect to duplicate.
#
# Measured, not guessed: a 1,284-item batch lost 131 calls (~25%) to exactly
# this, all of them given up after one attempt, while the same endpoint served
# 80 concurrent requests with zero errors in a short burst. The difference is
# time -- pooled connections go stale between calls, and a long run is made of
# gaps.
TRANSPORT_DISCONNECT = (
    'RemoteProtocolError',      # server closed a pooled connection mid-request
    'ConnectError',             # never got a connection
    'ConnectTimeout',
    'ReadError',                # connection died mid-response
    'WriteError',
    'PoolTimeout',
    'APITimeoutError',
)


def _transport_kind(exc, limit: int = 5) -> Optional[str]:
    """Walk the cause chain for a transport failure httpx named."""
    seen, cur = set(), exc
    for _ in range(limit):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if type(cur).__name__ in TRANSPORT_DISCONNECT:
            return type(cur).__name__
        cur = cur.__cause__ or cur.__context__
    return None

DEFAULT_RPM = 150
DEFAULT_CCU_PER_KEY = 10

# MEASURED, not assumed: the 150 is PER KEY.
#
# The provider's 429 body says only "Your maximum thresholds are 150 RPM",
# which does not say whose. A shared ceiling was held at first because it is
# safe under both readings -- but safe in the wrong direction: it throttled
# four keys to one key's budget. A run at 4x20 concurrent with the ceiling
# removed produced 241 errors and ZERO of them 429, which settles it.
#
# 0 disables the shared limiter and leaves the per-key ones, which are the
# real constraint. Set --account-rpm 150 to put the ceiling back if a future
# key set turns out to share a quota.
DEFAULT_ACCOUNT_RPM = 0

# A 429 says one thing only: this key is over its per-minute budget. The budget
# refills on a fixed clock, so the useful wait is one window -- not a number
# that doubles. Exponential backoff on a rate limit is backoff against a clock
# that already told you the answer: after four 429s it would be parking a
# healthy key for eight minutes to wait out a sixty-second window.
RATE_LIMIT_COOLDOWN = 60.0

# A timeout says nothing bad about the key. Just enough pause to let a network
# blip pass, then the key is back in rotation.
TIMEOUT_COOLDOWN = 3.0

# A stale pooled connection is not the key's fault AT ALL, so the key is not
# parked for it -- the item is simply requeued.
#
# This is the same blame rule that separates 429 from 499, applied one step
# further, and getting it wrong was expensive: cooling a key parks every one of
# its 20 workers, so ~40 dropped sockets a minute across 4 keys held each key
# idle ~35 seconds out of 60. Throughput fell from 64 items/min to 10 -- the
# retry that stopped the data loss cost six times the speed until the blame was
# put back where it belonged.
DISCONNECT_COOLDOWN = 0.0

# Everything else transient (5xx, connection resets) has no such clock. There
# the doubling is right, because the cause and its duration are both unknown.
TRANSIENT_BASE = 5.0
TRANSIENT_CAP = 300.0


def load_dotenv(path: str = '.env', override: bool = False) -> dict:
    """Minimal `.env` reader -- no dependency, because one file needs no library.

    Understands `KEY=value`, `export KEY=value`, `#` comments, and quoted
    values. Does NOT expand variables or run anything: a .env holding API keys
    is not a file that should be able to execute.
    """
    loaded = {}
    if not os.path.exists(path):
        return loaded
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            if line.startswith('export '):
                line = line[len('export '):].lstrip()
            name, _, value = line.partition('=')
            name, value = name.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if not name:
                continue
            loaded[name] = value
            if override or name not in os.environ:
                os.environ[name] = value
    return loaded


def collect_api_keys(prefix: str, env: Optional[dict] = None) -> List[str]:
    """Gather `<PREFIX>`, `<PREFIX>_1` ... `<PREFIX>_N` into an ordered list.

    Both spellings are read so a single-key setup keeps working untouched.
    Duplicates are dropped: the same key pasted into two slots is one rate
    budget wearing two names, and counting it twice is how a pool that looks
    4x provisioned runs straight into 429s.
    """
    env = env if env is not None else os.environ
    keys, seen = [], set()
    for name in [prefix] + [f'{prefix}_{i}' for i in range(1, 33)]:
        value = (env.get(name) or '').strip()
        if value and not value.startswith('<') and value not in seen:
            seen.add(value)
            keys.append(value)
    return keys


class RateLimiter:
    """Sliding-window limiter: at most `limit` acquisitions per `period`.

    A sliding window rather than a token bucket because the provider's own
    limit is phrased that way. A bucket refilled at 50/60s would allow a burst
    of 50 in one second after any idle stretch, which is exactly the shape that
    trips the limit this exists to respect.
    """

    def __init__(self, limit: int = DEFAULT_RPM, period: float = 60.0):
        self.limit = max(1, int(limit))
        self.period = float(period)
        self._events = deque()
        self._lock = threading.Lock()

    def acquire(self, stop: Optional[threading.Event] = None) -> bool:
        """Block until a slot frees. False if `stop` was set while waiting."""
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0] >= self.period:
                    self._events.popleft()
                if len(self._events) < self.limit:
                    self._events.append(now)
                    return True
                wait = self.period - (now - self._events[0])
            if stop is not None and stop.wait(min(wait, 0.5)):
                return False
            if stop is None:
                time.sleep(min(wait, 0.5))


class ApiKey:
    """One key: its rate budget, its in-flight cap, and its health."""

    def __init__(self, key: str, index: int, rpm: int = DEFAULT_RPM,
                 ccu: int = DEFAULT_CCU_PER_KEY,
                 rate_limit_wait: float = RATE_LIMIT_COOLDOWN,
                 timeout_wait: float = TIMEOUT_COOLDOWN,
                 disconnect_wait: float = DISCONNECT_COOLDOWN):
        self.key = key
        self.index = index
        self.label = f"key{index + 1}"
        # Held on the key rather than read from the module constants at the
        # call site, because `run_pool` cools keys itself and had no way to be
        # told otherwise: a test that provokes a 429 through the pool parked a
        # thread for the production 60 seconds, so the suite took minutes and
        # timed out instead of failing honestly.
        self.rate_limit_wait = rate_limit_wait
        self.timeout_wait = timeout_wait
        self.disconnect_wait = disconnect_wait
        self.limiter = RateLimiter(rpm)
        self.semaphore = threading.BoundedSemaphore(max(1, ccu))
        self.stats = Counter()
        self._lock = threading.Lock()
        self._cooldown_until = 0.0
        self._consecutive_failures = 0
        self.disabled_reason = None

    # -- health -------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self.disabled_reason is None and time.monotonic() >= self._cooldown_until

    @property
    def cooldown_remaining(self) -> float:
        return max(0.0, self._cooldown_until - time.monotonic())

    def note_success(self):
        with self._lock:
            self._consecutive_failures = 0
            self.stats['ok'] += 1

    def cool_down(self, status=None, kind: Optional[str] = None,
                  base: float = TRANSIENT_BASE,
                  cap: float = TRANSIENT_CAP, rate_limit_wait: Optional[float] = None):
        """Park this key. Fixed wait for 429, exponential for everything else.

        Per-key, never global: three keys behaving and one rate-limited is the
        normal state of a four-key pool, and a shared backoff would punish the
        three that are fine -- which is the whole reason to hold four.

        429 gets a flat `rate_limit_wait` because the provider's window is
        flat. Consecutive-failure doubling here would park a perfectly healthy
        key for minutes to wait out a sixty-second budget.
        """
        with self._lock:
            self._consecutive_failures += 1
            if status == 429:
                delay = self.rate_limit_wait if rate_limit_wait is None else rate_limit_wait
                self.stats['cooldown_429'] += 1
            elif kind == 'timeout' and status is None:
                # Transport-level: the socket died, the key is fine. Requeue
                # the item and keep all of this key's workers serving.
                delay = self.disconnect_wait
                self.stats['disconnect'] += 1
            elif kind == 'timeout' or status in TIMEOUT_STATUS:
                # Keyed on the KIND, not on the presence of a status integer.
                # A dropped connection carries no number, and falling through
                # to the exponential branch here parked a healthy key for
                # minutes over a stale socket.
                delay = self.timeout_wait
                self.stats[f'timeout_{status or "conn"}'] += 1
            else:
                delay = min(cap, base * (2 ** (self._consecutive_failures - 1)))
                self.stats[f'cooldown_{status or "error"}'] += 1
            # Jitter regardless: four keys that 429 in the same second must not
            # come back in the same second and do it again.
            delay += random.uniform(0, delay * 0.25)
            self._cooldown_until = time.monotonic() + delay
        return delay

    def disable(self, reason: str):
        with self._lock:
            self.disabled_reason = reason
            self.stats['disabled'] += 1

    def __repr__(self):
        state = (f"DISABLED ({self.disabled_reason})" if self.disabled_reason
                 else f"cooling {self.cooldown_remaining:.0f}s" if not self.available
                 else "ok")
        return f"<{self.label} {state} {dict(self.stats)}>"


class KeyPool:
    """The set of keys, plus the accounting a run needs to be explainable."""

    def __init__(self, keys: Iterable[str], rpm: int = DEFAULT_RPM,
                 ccu_per_key: int = DEFAULT_CCU_PER_KEY,
                 account_rpm: int = DEFAULT_ACCOUNT_RPM,
                 rate_limit_wait: float = RATE_LIMIT_COOLDOWN,
                 timeout_wait: float = TIMEOUT_COOLDOWN,
                 disconnect_wait: float = DISCONNECT_COOLDOWN):
        self.keys = [ApiKey(k, i, rpm, ccu_per_key, rate_limit_wait, timeout_wait,
                            disconnect_wait)
                     for i, k in enumerate(keys)]
        if not self.keys:
            raise ValueError("KeyPool needs at least one key")
        self.rpm = rpm
        self.ccu_per_key = ccu_per_key
        # Shared across every key. Held IN ADDITION to the per-key limiter, not
        # instead of it: if the quota really is per key the per-key limiter is
        # the right one, and if it is per account this is. Whichever binds
        # first is the true constraint, and neither reading can overshoot.
        self.account_limiter = RateLimiter(account_rpm) if account_rpm else None

    @property
    def workers(self) -> int:
        return len(self.keys) * self.ccu_per_key

    @property
    def live_keys(self) -> List[ApiKey]:
        return [k for k in self.keys if k.disabled_reason is None]

    def theoretical_rpm(self) -> int:
        per_key_total = len(self.live_keys) * self.rpm
        if self.account_limiter is None:
            return per_key_total
        return min(per_key_total, self.account_limiter.limit)

    def summary(self) -> str:
        return " · ".join(repr(k) for k in self.keys)


def describe_exception(exc, limit: int = 4) -> str:
    """Render an exception WITH its cause chain.

    `openai.APIConnectionError` stringifies to the four useless words
    "Connection error." -- the same text whether the socket was refused, the
    gateway hung up mid-response, DNS failed, or TLS did. A run that logs 131
    of those has logged that something went wrong 131 times and nothing else.
    The cause chain is where httpx says which one it was, so the chain is what
    gets recorded.
    """
    parts, seen, cur = [], set(), exc
    while cur is not None and len(parts) < limit:
        if id(cur) in seen:
            break
        seen.add(id(cur))
        text = str(cur).strip()
        parts.append(f"{type(cur).__name__}: {text}" if text else type(cur).__name__)
        cur = cur.__cause__ or cur.__context__
    return ' <- '.join(parts)


def classify(exc) -> tuple:
    """(status, kind) where kind is 'retry' | 'timeout' | 'auth' | 'fatal'.

    Only 429 and 499 come back retryable. A transport failure with no status
    (APIConnectionError, APITimeoutError, a bare reset) is `fatal` here, which
    reads harsher than it is: fatal means "logged, not retried in THIS run",
    and since a failed item is never checkpointed, the next run retries it
    anyway. The trade is deliberate -- absorbing those failures into eventual
    success is precisely what hides how often they happen.
    """
    status = getattr(exc, 'status_code', None) or getattr(
        getattr(exc, 'response', None), 'status_code', None)
    if status in AUTH_STATUS:
        return status, 'auth'
    if status in TIMEOUT_STATUS:
        return status, 'timeout'
    if status in RETRYABLE_STATUS:
        return status, 'retry'
    if status is None and _transport_kind(exc):
        return None, 'timeout'
    return status, 'fatal'


class PoolExhausted(RuntimeError):
    """Every key is disabled. Continuing would fail every remaining item."""


def run_pool(items, work, pool: KeyPool, on_result=None, on_giveup=None, on_error=None,
             max_item_retries: int = 4, progress_every: int = 0,
             rate_limit_per_item: bool = True,
             log: Callable[[str], None] = print):
    """Run `work(item, api_key, stop)` across the pool, one thread per key slot.

    Each thread is bound to one key for its lifetime, which is what makes the
    per-key concurrency cap real: `ccu_per_key` threads on a key means at most
    that many of its requests in flight, with no coordination needed. Threads
    on a cooling key simply stop pulling work, and the queue drains through the
    healthy ones.

    `work` raises to signal failure. Retryable failures cool the key and put the
    item BACK ON THE QUEUE -- the item is not tied to the key that failed it,
    so a cool-down costs time and never coverage. `on_result` is called under a
    lock, so it is the safe place to write a checkpoint.

    Returns (results, stats).
    """
    work_queue = queue.Queue()
    for item in items:
        work_queue.put((item, 0))
    total = work_queue.qsize()
    if not total:
        return [], Counter()

    results, stats = [], Counter()
    results_lock = threading.Lock()
    stop = threading.Event()
    done = [0]

    def worker(api_key: ApiKey):
        while not stop.is_set():
            try:
                item, attempts = work_queue.get(timeout=0.5)
            except queue.Empty:
                # Empty is not finished: another worker may still requeue an
                # item it could not complete. Only the counter is authoritative.
                with results_lock:
                    if done[0] >= total:
                        return
                continue
            try:
                if api_key.disabled_reason is not None:
                    work_queue.put((item, attempts))
                    return
                while not api_key.available and not stop.is_set():
                    time.sleep(min(1.0, max(0.05, api_key.cooldown_remaining)))
                if stop.is_set():
                    work_queue.put((item, attempts))
                    return
                # Rate limiting belongs wherever the REQUEST is made, and for
                # a work function that makes several it is not here.
                #
                # Acquiring once per item counts items, and the provider counts
                # calls: one compression item is up to 4 calls (3 attempts plus
                # the answerability judge), so a limiter set to 150/min was
                # letting through as many as 600. A separate account-wide
                # ceiling had been masking that by roughly the same factor;
                # removing the ceiling did not cause the 429s, it revealed
                # them. Work functions that call more than once pass
                # rate_limit_per_item=False and acquire per call themselves.
                if rate_limit_per_item:
                    if not api_key.limiter.acquire(stop):
                        work_queue.put((item, attempts))
                        return
                    if (pool.account_limiter is not None
                            and not pool.account_limiter.acquire(stop)):
                        work_queue.put((item, attempts))
                        return

                with api_key.semaphore:
                    outcome = work(item, api_key, stop)

                api_key.note_success()
                with results_lock:
                    results.append(outcome)
                    done[0] += 1
                    stats['completed'] += 1
                    if on_result is not None:
                        on_result(item, outcome)
                    if progress_every and done[0] % progress_every == 0:
                        log(f"  [{done[0]}/{total}] " + pool.summary())

            except BaseException as exc:                  # noqa: BLE001
                if not isinstance(exc, Exception):
                    # KeyboardInterrupt, MemoryError, SystemExit -- not ours to
                    # classify, but the item is ours to not lose. A worker that
                    # died holding one left `done` permanently short of `total`
                    # while its siblings sat on an empty queue, so the wait loop
                    # never finished: the run hung rather than failed.
                    work_queue.put((item, attempts))
                    raise
                status, kind = classify(exc)
                if on_error is not None:
                    # Every failure is recorded, 429s included but labelled, so
                    # "the run was slow" and "the run was broken" can be told
                    # apart afterwards from the file rather than from memory.
                    with results_lock:
                        on_error({'key': api_key.label, 'status': status, 'kind': kind,
                                  'error': describe_exception(exc)[:400],
                                  'attempt': attempts + 1}, item)
                if kind == 'auth':
                    api_key.disable(f"HTTP {status}")
                    log(f"  {api_key.label} DISABLED (HTTP {status}) -- "
                        f"{len(pool.live_keys)} key(s) left")
                    work_queue.put((item, attempts))
                    if not pool.live_keys:
                        stop.set()
                    return
                if kind in ('retry', 'timeout') and attempts + 1 < max_item_retries:
                    delay = api_key.cool_down(status, kind=kind)
                    stats[f'retry_{status or "conn"}'] += 1
                    if delay > 0:
                        log(f"  {api_key.label} cooling {delay:.0f}s "
                            f"({f'HTTP {status}' if status else _transport_kind(exc) or 'transport'}); "
                            f"item requeued (attempt {attempts + 1}/{max_item_retries})")
                    work_queue.put((item, attempts + 1))
                else:
                    with results_lock:
                        done[0] += 1
                        stats['gave_up'] += 1
                        if on_giveup is not None:
                            on_giveup(item, exc)
                    log(f"  GAVE UP after {attempts + 1} attempt(s) on {item!r}: "
                        f"{describe_exception(exc)[:220]}")
            finally:
                work_queue.task_done()

    threads = []
    for api_key in pool.keys:
        for slot in range(pool.ccu_per_key):
            thread = threading.Thread(target=worker, args=(api_key,),
                                      name=f"{api_key.label}-{slot}", daemon=True)
            thread.start()
            threads.append(thread)

    log(f"Pool: {len(pool.keys)} key(s) x {pool.ccu_per_key} concurrent "
        f"= {pool.workers} worker(s), ~{pool.theoretical_rpm()} req/min ceiling")
    interrupted = False
    try:
        while True:
            with results_lock:
                finished = done[0] >= total
            if finished or not any(t.is_alive() for t in threads):
                break
            if not pool.live_keys:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        interrupted = True
        log("\nInterrupted -- finishing in-flight requests, checkpoint kept.")
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5.0)

    # Checked AFTER the threads are joined, not inside the wait loop. Every
    # worker on a dying pool disables its key and returns, so the loop's
    # "are any threads alive" condition goes false first and the exhaustion
    # branch was never reached -- a run where all four keys were revoked
    # returned an empty result set and an exit code saying it had worked.
    if interrupted:
        # Re-raised, not swallowed. Catching it here and returning normally
        # told the caller nothing, so a batch loop marked the half-finished
        # batch 'done' in the manifest and moved to the next one -- and the
        # skipped work was then invisible to every resume. An interrupt has to
        # reach the code that decides what counts as finished.
        raise KeyboardInterrupt(f"interrupted after {done[0]}/{total} item(s)")

    if not pool.live_keys and done[0] < total:
        raise PoolExhausted(
            "Every API key is disabled (auth/quota). "
            f"{done[0]}/{total} item(s) completed. "
            "Check the keys in .env -- a revoked or out-of-quota key fails "
            "instantly on every call.")

    for api_key in pool.keys:
        for name, count in api_key.stats.items():
            stats[f'{api_key.label}_{name}'] += count
    return results, stats
