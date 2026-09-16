"""Tests for vncompress/api_pool.py -- multi-key fan-out.

The paths that matter are the ones a healthy run never touches: a key that
starts refusing, a key that is dead, a run where every key dies. Each is driven
by a fake worker whose failures the test chooses, because the real versions
only show up hours into a paid run.
"""
import os
import signal
import threading
import time
from collections import Counter

import httpx
import pytest

from vncompress.api_pool import (
    describe_exception,
    AUTH_STATUS,
    RATE_LIMIT_COOLDOWN,
    RETRYABLE_STATUS,
    TIMEOUT_COOLDOWN,
    TIMEOUT_STATUS,
    ApiKey,
    KeyPool,
    PoolExhausted,
    RateLimiter,
    classify,
    collect_api_keys,
    load_dotenv,
    run_pool,
)


class FakeHttpError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


def quiet(_message):
    """run_pool logs progress; tests do not need it on stdout."""


class TestCollectApiKeys:
    def test_reads_numbered_slots_in_order(self):
        env = {'K_1': 'a', 'K_2': 'b', 'K_3': 'c', 'K_4': 'd'}
        assert collect_api_keys('K', env) == ['a', 'b', 'c', 'd']

    def test_the_bare_name_still_works_for_a_single_key_setup(self):
        assert collect_api_keys('K', {'K': 'solo'}) == ['solo']

    def test_bare_and_numbered_combine(self):
        assert collect_api_keys('K', {'K': 'a', 'K_1': 'b'}) == ['a', 'b']

    def test_the_same_key_in_two_slots_is_counted_once(self):
        # It is ONE rate budget. Counting it twice makes the pool look 4x
        # provisioned and walks it straight into the limit it is avoiding.
        assert collect_api_keys('K', {'K_1': 'same', 'K_2': 'same', 'K_3': 'other'}) \
            == ['same', 'other']

    def test_unfilled_template_placeholders_are_not_keys(self):
        assert collect_api_keys('K', {'K_1': '<key-1>', 'K_2': 'real'}) == ['real']

    def test_blank_and_whitespace_slots_are_skipped(self):
        assert collect_api_keys('K', {'K_1': '', 'K_2': '   ', 'K_3': 'real'}) == ['real']

    def test_gaps_in_the_numbering_do_not_stop_the_scan(self):
        assert collect_api_keys('K', {'K_1': 'a', 'K_4': 'd'}) == ['a', 'd']


class TestLoadDotenv:
    def test_parses_comments_quotes_and_export(self, tmp_path):
        path = tmp_path / '.env'
        path.write_text(
            "# a comment\n"
            "PLAIN=value\n"
            "export EXPORTED=value2\n"
            'QUOTED="value 3"\n'
            "SINGLE='value 4'\n"
            "\n"
            "EMPTY=\n", encoding='utf-8')
        loaded = load_dotenv(str(path))
        assert loaded['PLAIN'] == 'value'
        assert loaded['EXPORTED'] == 'value2'
        assert loaded['QUOTED'] == 'value 3'
        assert loaded['SINGLE'] == 'value 4'
        assert loaded['EMPTY'] == ''

    def test_equals_inside_the_value_survives(self, tmp_path):
        # Base64- and JWT-shaped keys routinely contain or end in '='. Splitting
        # on every '=' instead of the first would truncate them silently, and a
        # truncated key fails as 401 -- read as "revoked", not "mis-parsed".
        path = tmp_path / '.env'
        path.write_text("KEY=abc==def\n", encoding='utf-8')
        assert load_dotenv(str(path))['KEY'] == 'abc==def'

    def test_does_not_clobber_the_real_environment_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv('ALREADY', 'from-shell')
        path = tmp_path / '.env'
        path.write_text("ALREADY=from-file\n", encoding='utf-8')
        load_dotenv(str(path))
        assert os.environ['ALREADY'] == 'from-shell'

    def test_override_is_opt_in(self, tmp_path, monkeypatch):
        monkeypatch.setenv('ALREADY', 'from-shell')
        path = tmp_path / '.env'
        path.write_text("ALREADY=from-file\n", encoding='utf-8')
        load_dotenv(str(path), override=True)
        assert os.environ['ALREADY'] == 'from-file'

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert load_dotenv(str(tmp_path / 'nope.env')) == {}


class TestRateLimiter:
    def test_allows_up_to_the_limit_without_waiting(self):
        limiter = RateLimiter(limit=5, period=60.0)
        start = time.monotonic()
        for _ in range(5):
            limiter.acquire()
        assert time.monotonic() - start < 0.1

    def test_blocks_once_the_window_is_full(self):
        limiter = RateLimiter(limit=2, period=0.4)
        limiter.acquire()
        limiter.acquire()
        start = time.monotonic()
        limiter.acquire()
        assert time.monotonic() - start >= 0.25

    def test_the_window_slides_rather_than_resetting(self):
        # A bucket refilled all at once would let 2 more through immediately
        # after the period; a sliding window releases them one at a time as the
        # old events age out. The provider counts the sliding way.
        limiter = RateLimiter(limit=2, period=0.3)
        limiter.acquire()
        time.sleep(0.16)
        limiter.acquire()
        start = time.monotonic()
        limiter.acquire()          # waits for the FIRST event to age out only
        waited = time.monotonic() - start
        assert 0.05 < waited < 0.25

    def test_stop_event_breaks_the_wait(self):
        limiter = RateLimiter(limit=1, period=30.0)
        limiter.acquire()
        stop = threading.Event()
        threading.Timer(0.1, stop.set).start()
        assert limiter.acquire(stop) is False

    def test_is_safe_under_concurrency(self):
        limiter = RateLimiter(limit=10, period=60.0)
        granted = []
        lock = threading.Lock()

        def take():
            if limiter.acquire(threading.Event() if False else None):
                with lock:
                    granted.append(1)

        threads = [threading.Thread(target=take) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(granted) == 10


def _api_connection_error(inner):
    """An openai APIConnectionError raised from an httpx failure, as the SDK does."""
    from openai import APIConnectionError
    try:
        try:
            raise inner
        except Exception as cause:
            raise APIConnectionError(request=httpx.Request('POST', 'https://x/v1/chat')) from cause
    except APIConnectionError as exc:
        return exc


class TestClassify:
    """Only 429 and 499 are retried. Everything else is evidence, not noise."""

    def test_only_429_is_retried(self):
        assert classify(FakeHttpError(429))[1] == 'retry'
        assert set(RETRYABLE_STATUS) == {429}

    def test_499_is_a_timeout_and_is_retried(self):
        # nginx "client closed request". It used to classify as `fatal`, so
        # every proxy timeout silently discarded a sample -- and at ~40s per
        # compression call this is a routine event, not a rare one.
        assert classify(FakeHttpError(499))[1] == 'timeout'
        assert set(TIMEOUT_STATUS) == {499}

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 408, 522, 524, 400, 404, 422])
    def test_everything_else_is_logged_not_retried(self, status):
        # "fatal" is narrower than it sounds: logged and not retried in THIS
        # run. A failed item is never checkpointed, so a later run picks it up.
        assert classify(FakeHttpError(status))[1] == 'fatal'

    @pytest.mark.parametrize("status", AUTH_STATUS)
    def test_auth_failures_disable_the_key(self, status):
        # A revoked key returns 401 instantly on every call; retried, it drains
        # the whole queue at full speed and fails every item.
        assert classify(FakeHttpError(status))[1] == 'auth'

    def test_a_bare_error_with_no_transport_cause_is_still_fatal(self):
        # The default stays narrow. Only a failure httpx actually NAMED is
        # retried; an exception that says nothing about the transport is
        # evidence, and absorbing it into eventual success is what hides how
        # often it happens.
        assert classify(ConnectionResetError("reset"))[1] == 'fatal'
        assert classify(ValueError("nonsense"))[1] == 'fatal'

    def test_a_dropped_connection_is_the_same_event_as_a_499(self):
        # MEASURED: a 1,284-item batch lost 131 calls (~25%) to this, every one
        # given up after a single attempt, while the same endpoint served 80
        # concurrent requests with zero errors in a short burst.
        #
        # openai raises APIConnectionError with NO status -- no HTTP response
        # ever arrived to carry one -- and it stringifies to the four words
        # "Connection error." Underneath, httpx says what happened. Classifying
        # on the presence of an integer rather than on the failure meant a
        # stale pooled connection discarded a paid-for sample.
        exc = _api_connection_error(
            httpx.RemoteProtocolError("Server disconnected without sending a response."))
        assert classify(exc) == (None, 'timeout')

    @pytest.mark.parametrize("inner", [
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("timed out"),
        httpx.ReadError("died mid-response"),
        httpx.PoolTimeout("no free connection"),
    ])
    def test_every_transport_failure_is_retried_not_dropped(self, inner):
        # Safe to repeat without exception: none of these produced an answer,
        # and a compression call has no side effect to duplicate.
        assert classify(_api_connection_error(inner))[1] == 'timeout'

    def test_the_cause_chain_is_what_gets_logged(self):
        # "APIConnectionError: Connection error." is the same string whether the
        # socket was refused, the gateway hung up, DNS failed or TLS did. A log
        # full of it records that something went wrong and nothing else.
        text = describe_exception(_api_connection_error(
            httpx.RemoteProtocolError("Server disconnected without sending a response.")))
        assert 'RemoteProtocolError' in text
        assert 'Server disconnected' in text

    def test_a_dropped_connection_does_not_park_the_key_at_all(self):
        # The blame rule, one step past 429-vs-499: a stale socket is not the
        # key's fault, so the item is requeued and the key keeps serving.
        #
        # MEASURED: cooling a key parks all 20 of its workers. At ~40 dropped
        # sockets a minute across 4 keys that held each key idle ~35s out of
        # 60, and throughput fell from 64 items/min to 10 -- the retry that
        # stopped the data loss cost 6x the speed until the blame moved back.
        key = ApiKey('k', 0)
        assert all(key.cool_down(None, kind='timeout') == 0.0 for _ in range(5))
        assert key.stats['disconnect'] == 5
        assert key.available

    def test_a_499_still_parks_the_key_briefly(self):
        # A real 499 DOES carry a status, and keeps the short pause: unlike a
        # stale socket it says the request path itself is struggling.
        delay = ApiKey('k', 0).cool_down(499, kind='timeout')
        assert 0 < delay <= TIMEOUT_COOLDOWN * 1.25

    def test_a_timeout_barely_touches_the_key(self):
        # Different blame: 429 is the key's fault, 499 is the request's. Parking
        # a healthy key for someone else's timeout removes capacity for nothing.
        timeout_delay = ApiKey('k', 0).cool_down(499)
        rate_limit_delay = ApiKey('k', 1).cool_down(429)
        assert timeout_delay <= TIMEOUT_COOLDOWN * 1.25
        assert timeout_delay < rate_limit_delay

    def test_a_timeout_does_not_escalate_across_failures(self):
        key = ApiKey('k', 0)
        assert all(d <= TIMEOUT_COOLDOWN * 1.25 for d in (key.cool_down(499) for _ in range(5)))

    def test_a_429_waits_the_full_window_the_provider_named(self):
        # The provider's own 429 body says "Please try again in 60s."
        delays = [ApiKey('k', 0).cool_down(429) for _ in range(3)]
        assert all(RATE_LIMIT_COOLDOWN <= d <= RATE_LIMIT_COOLDOWN * 1.25 for d in delays)

    def test_a_timed_out_item_is_requeued_and_completes(self):
        pool = KeyPool(['a', 'b'], rpm=10 ** 6, ccu_per_key=2, account_rpm=0)
        state, lock = {'n': 0}, threading.Lock()

        def work(item, api_key, stop):
            with lock:
                state['n'] += 1
                if state['n'] <= 3:
                    raise FakeHttpError(499)
            return item

        results, stats = run_pool(range(15), work, pool, log=quiet)
        assert len(results) == 15 and stats['gave_up'] == 0

    def test_a_500_is_given_up_on_immediately(self):
        pool = KeyPool(['a'], rpm=10 ** 6, ccu_per_key=2, account_rpm=0)
        attempts, lock = Counter(), threading.Lock()

        def work(item, api_key, stop):
            with lock:
                attempts[item] += 1
            raise FakeHttpError(500)

        _, stats = run_pool(range(3), work, pool, max_item_retries=5, log=quiet)
        assert stats['gave_up'] == 3
        assert all(count == 1 for count in attempts.values())


class TestApiKeyHealth:
    def test_a_fresh_key_is_available(self):
        assert ApiKey('k', 0).available is True

    def test_cooldown_makes_it_unavailable_then_available_again(self):
        key = ApiKey('k', 0)
        key.cool_down(429, rate_limit_wait=0.15)
        assert key.available is False
        time.sleep(0.45)
        assert key.available is True

    def test_a_success_resets_the_backoff(self):
        key = ApiKey('k', 0)
        key.cool_down(503, base=1.0, cap=1000)
        key.cool_down(503, base=1.0, cap=1000)
        key.note_success()
        assert key.cool_down(503, base=1.0, cap=1000) < 2.0

    def test_backoff_is_capped(self):
        key = ApiKey('k', 0)
        for _ in range(20):
            delay = key.cool_down(503, base=1.0, cap=10.0)
        assert delay <= 12.5   # cap plus the <=25% jitter

    def test_a_disabled_key_never_becomes_available(self):
        key = ApiKey('k', 0)
        key.disable('HTTP 401')
        assert key.available is False
        time.sleep(0.05)
        assert key.available is False


class TestKeyPoolShape:
    def test_workers_is_keys_times_concurrency(self):
        assert KeyPool(['a', 'b', 'c', 'd'], ccu_per_key=10).workers == 40

    def test_theoretical_rpm_scales_with_live_keys(self):
        pool = KeyPool(['a', 'b', 'c', 'd'], rpm=50, account_rpm=0)
        assert pool.theoretical_rpm() == 200
        pool.keys[0].disable('HTTP 401')
        assert pool.theoretical_rpm() == 150

    def test_the_account_ceiling_caps_the_per_key_total(self):
        # Still available, no longer the default: a live run at 4x20 concurrent
        # with the shared ceiling removed produced 241 errors and ZERO of them
        # 429, so this provider's 150 RPM is per key. Holding a shared ceiling
        # "to be safe" was throttling four keys to one key's budget.
        pool = KeyPool(['a', 'b', 'c', 'd'], rpm=150, account_rpm=150)
        assert pool.theoretical_rpm() == 150

    def test_the_default_leaves_the_per_key_budgets_alone(self):
        pool = KeyPool(['a', 'b', 'c', 'd'], rpm=150)
        assert pool.account_limiter is None
        assert pool.theoretical_rpm() == 600

    def test_the_account_ceiling_actually_throttles(self):
        pool = KeyPool(['a', 'b', 'c', 'd'], rpm=10 ** 6, ccu_per_key=4, account_rpm=10)
        pool.account_limiter = RateLimiter(limit=10, period=0.5)
        start = time.monotonic()
        results, _ = run_pool(range(25), lambda i, k, s: i, pool, log=quiet)
        assert len(results) == 25
        assert time.monotonic() - start >= 0.4

    def test_the_ceiling_can_be_switched_off(self):
        assert KeyPool(['a'], rpm=150, account_rpm=0).account_limiter is None

    def test_an_empty_pool_is_rejected_at_construction(self):
        with pytest.raises(ValueError):
            KeyPool([])


class TestRunPool:
    def test_every_item_is_processed_exactly_once(self):
        pool = KeyPool(['a', 'b', 'c', 'd'], rpm=10 ** 6, ccu_per_key=4, account_rpm=0)
        seen, lock = Counter(), threading.Lock()

        def work(item, api_key, stop):
            with lock:
                seen[item] += 1
            return item * 2

        results, _ = run_pool(range(200), work, pool, log=quiet)
        assert sorted(results) == [i * 2 for i in range(200)]
        assert all(count == 1 for count in seen.values())
        assert len(seen) == 200

    def test_work_is_spread_across_all_keys(self):
        pool = KeyPool(['a', 'b', 'c', 'd'], rpm=10 ** 6, ccu_per_key=4, account_rpm=0)
        used, lock = Counter(), threading.Lock()

        def work(item, api_key, stop):
            time.sleep(0.002)
            with lock:
                used[api_key.label] += 1
            return item

        run_pool(range(400), work, pool, log=quiet)
        assert len(used) == 4
        assert min(used.values()) > 0

    def test_concurrency_never_exceeds_the_per_key_cap(self):
        # The cap is what keeps a burst from becoming the 429 the limiter exists
        # to avoid, so it has to hold under load, not just on paper.
        pool = KeyPool(['a', 'b'], rpm=10 ** 6, ccu_per_key=3, account_rpm=0)
        in_flight, peak, lock = Counter(), {}, threading.Lock()

        def work(item, api_key, stop):
            with lock:
                in_flight[api_key.label] += 1
                peak[api_key.label] = max(peak.get(api_key.label, 0), in_flight[api_key.label])
            time.sleep(0.01)
            with lock:
                in_flight[api_key.label] -= 1
            return item

        run_pool(range(120), work, pool, log=quiet)
        assert max(peak.values()) <= 3

    def test_a_rate_limited_key_cools_down_and_the_item_moves_on(self):
        # Short cool-down on purpose. The production wait is the 60s the
        # provider's own 429 body names; asserting the REQUEUE with that number
        # parks a thread for a minute and tests the clock, not the behaviour.
        pool = KeyPool(['a', 'b'], rpm=10 ** 6, ccu_per_key=2, rate_limit_wait=0.05, account_rpm=0)
        failed_once = {'done': False}
        lock = threading.Lock()

        def work(item, api_key, stop):
            with lock:
                if not failed_once['done']:
                    failed_once['done'] = True
                    raise FakeHttpError(429)
            return item

        results, stats = run_pool(range(20), work, pool, log=quiet)
        # The item that hit the 429 was requeued, not lost.
        assert len(results) == 20
        assert stats['retry_429'] == 1
        assert stats['gave_up'] == 0

    def test_a_dead_key_is_disabled_and_the_run_finishes_on_the_others(self):
        pool = KeyPool(['dead', 'b', 'c', 'd'], rpm=10 ** 6, ccu_per_key=2, account_rpm=0)

        def work(item, api_key, stop):
            if api_key.label == 'key1':
                raise FakeHttpError(401)
            return item

        results, _ = run_pool(range(60), work, pool, log=quiet)
        assert pool.keys[0].disabled_reason == 'HTTP 401'
        assert len(pool.live_keys) == 3
        assert sorted(results) == list(range(60))

    def test_when_every_key_dies_the_run_raises_rather_than_reporting_success(self):
        # Silently returning partial results here would look like a completed
        # run that produced very little data.
        pool = KeyPool(['a', 'b'], rpm=10 ** 6, ccu_per_key=2, account_rpm=0)

        def work(item, api_key, stop):
            raise FakeHttpError(403)

        with pytest.raises(PoolExhausted):
            run_pool(range(50), work, pool, log=quiet)

    def test_an_item_that_keeps_failing_is_given_up_not_looped_forever(self):
        pool = KeyPool(['a', 'b'], rpm=10 ** 6, ccu_per_key=2, rate_limit_wait=0.05, account_rpm=0)
        gave_up = []

        def work(item, api_key, stop):
            if item == 7:
                raise FakeHttpError(429)
            return item

        results, stats = run_pool(
            range(20), work, pool, on_giveup=lambda i, e: gave_up.append(i),
            max_item_retries=2, log=quiet)
        assert gave_up == [7]
        assert stats['gave_up'] == 1
        assert len(results) == 19

    def test_fatal_errors_are_not_retried(self):
        pool = KeyPool(['a'], rpm=10 ** 6, ccu_per_key=2, account_rpm=0)
        attempts, lock = Counter(), threading.Lock()

        def work(item, api_key, stop):
            with lock:
                attempts[item] += 1
            raise FakeHttpError(400)

        _, stats = run_pool(range(3), work, pool, max_item_retries=5, log=quiet)
        assert stats['gave_up'] == 3
        assert all(count == 1 for count in attempts.values())

    def test_on_result_is_serialised_so_it_can_write_a_checkpoint(self):
        pool = KeyPool(['a', 'b', 'c', 'd'], rpm=10 ** 6, ccu_per_key=5, account_rpm=0)
        written, overlaps = [], []
        active = {'n': 0}

        def on_result(item, outcome):
            active['n'] += 1
            if active['n'] > 1:
                overlaps.append(True)
            written.append(outcome)
            active['n'] -= 1

        run_pool(range(300), lambda i, k, s: i, pool, on_result=on_result, log=quiet)
        assert len(written) == 300
        assert not overlaps

    def test_every_failure_is_reported_for_review(self):
        # 429s are logged too, but labelled, so "slow run" and "broken run" can
        # be told apart from the file afterwards instead of from memory.
        pool = KeyPool(['a', 'b'], rpm=10 ** 6, ccu_per_key=2, account_rpm=0,
                       rate_limit_wait=0.05)
        errors = []
        state = {'n': 0}
        lock = threading.Lock()

        def work(item, api_key, stop):
            with lock:
                state['n'] += 1
                if state['n'] == 1:
                    raise FakeHttpError(429)
                if state['n'] == 2:
                    raise FakeHttpError(500)
            return item

        run_pool(range(10), work, pool,
                 on_error=lambda info, item: errors.append(info), log=quiet)
        assert {e['status'] for e in errors} == {429, 500}
        # BOTH are logged; they are not treated the same. The 429 is retried,
        # the 500 is kept as evidence -- which is the distinction the log has
        # to preserve for "slow run" and "broken run" to be separable later.
        by_status = {e['status']: e['kind'] for e in errors}
        assert by_status == {429: 'retry', 500: 'fatal'}
        assert all(e['key'].startswith('key') for e in errors)

    def test_an_interrupt_is_raised_not_swallowed(self):
        # Returning normally after a Ctrl-C told the caller nothing, so the
        # batch loop wrote the half-finished batch to the manifest as 'done'
        # and moved on -- and every later resume then SKIPPED the work that
        # never ran. Silent partial data is worse than a loud stop.
        #
        # Delivered as a real SIGINT, because that is how it arrives: to the
        # MAIN thread, while the workers are mid-call. Raising it inside a
        # worker instead tests a path no Ctrl-C ever takes.
        pool = KeyPool(['a', 'b'], rpm=10 ** 6, ccu_per_key=2, account_rpm=0)
        started = threading.Event()

        def work(item, api_key, stop):
            started.set()
            time.sleep(0.05)
            return item

        def interrupt_once():
            started.wait(5)
            time.sleep(0.05)
            signal.raise_signal(signal.SIGINT)

        threading.Thread(target=interrupt_once, daemon=True).start()
        with pytest.raises(KeyboardInterrupt):
            run_pool(range(400), work, pool, log=quiet)

    def test_a_worker_that_dies_outright_does_not_hang_the_run(self):
        # Its item goes back on the queue. Without that the counter stays one
        # short of the total for ever and the wait loop spins on an empty
        # queue -- a hang, which reads exactly like a slow API.
        class Boom(BaseException):
            """Outside the Exception hierarchy, so classify() never sees it."""

        pool = KeyPool(['a', 'b'], rpm=10 ** 6, ccu_per_key=2, account_rpm=0)
        state = {'n': 0}
        lock = threading.Lock()

        def work(item, api_key, stop):
            with lock:
                state['n'] += 1
                if state['n'] == 2:
                    raise Boom("worker died")
            return item

        results, _ = run_pool(range(30), work, pool, log=quiet)
        assert sorted(results) == list(range(30))

    def test_the_limiter_can_be_left_to_a_work_function_that_calls_many_times(self):
        # One permit per ITEM counts items; the provider counts CALLS. A work
        # function making 4 calls per item under a 150/min limiter was sending
        # up to 600 -- masked for a while by a separate account-wide ceiling
        # throttling by a similar factor, so removing that ceiling did not
        # cause the 429s, it revealed them.
        pool = KeyPool(['a'], rpm=10 ** 6, ccu_per_key=1, account_rpm=0)
        acquired = []
        real = pool.keys[0].limiter.acquire
        pool.keys[0].limiter.acquire = lambda stop=None: (acquired.append(1), real(stop))[1]

        run_pool(range(5), lambda i, k, s: i, pool, rate_limit_per_item=False, log=quiet)
        assert acquired == []

        run_pool(range(5), lambda i, k, s: i, pool, log=quiet)
        assert len(acquired) == 5

    def test_a_dead_key_is_reported_to_the_error_log(self):
        pool = KeyPool(['dead', 'b'], rpm=10 ** 6, ccu_per_key=2, account_rpm=0)
        errors = []

        def work(item, api_key, stop):
            if api_key.label == 'key1':
                raise FakeHttpError(401)
            return item

        run_pool(range(20), work, pool,
                 on_error=lambda info, item: errors.append(info), log=quiet)
        assert any(e['kind'] == 'auth' and e['status'] == 401 for e in errors)

    def test_no_items_is_not_an_error(self):
        pool = KeyPool(['a'], rpm=10 ** 6, account_rpm=0)
        assert run_pool([], lambda i, k, s: i, pool, log=quiet) == ([], Counter())

    def test_the_rate_limit_is_actually_enforced(self):
        # 4 keys x 5/sec should take ~1s for 40 items, not ~0s.
        pool = KeyPool(['a', 'b', 'c', 'd'], ccu_per_key=4, account_rpm=0)
        for key in pool.keys:
            key.limiter = RateLimiter(limit=5, period=0.5)
        start = time.monotonic()
        results, _ = run_pool(range(40), lambda i, k, s: i, pool, log=quiet)
        elapsed = time.monotonic() - start
        assert len(results) == 40
        assert elapsed >= 0.4
