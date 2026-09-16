#!/usr/bin/env python3
"""probe_api_pool.py -- fire N real requests through the key pool and report.

    python scripts/probe_api_pool.py --requests 20
    python scripts/probe_api_pool.py --requests 200 --rpm 50   # provoke a 429

Run this before any long job. It costs 20 calls and answers the four questions
that otherwise surface three hours into a paid run:

  1. Does every key in .env actually work? A revoked or mistyped key returns
     401 on every call and is indistinguishable from a working one until used.
  2. Is the work spread across all four keys, or did it quietly collapse onto
     one? A pool that fell back to a single key is not slow-looking, it is just
     slow.
  3. Does the client-side rate limit hold? 429s here mean the real per-key
     budget is below what .env claims.
  4. What comes back on failure? Providers differ in how they shape errors, and
     `classify()` has to read the status correctly or the whole retry policy
     misfires.

Nothing is written to the dataset. Errors go to the probe's own log.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vncompress.api_pool import (
    DEFAULT_CCU_PER_KEY,
    DEFAULT_RPM,
    KeyPool,
    PoolExhausted,
    collect_api_keys,
    load_dotenv,
    run_pool,
)

load_dotenv()

PROMPT = ("Trả lời bằng đúng một từ tiếng Việt: thủ đô của Việt Nam là gì? "
          "Chỉ trả lời tên thành phố.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--requests', type=int, default=20)
    ap.add_argument('--model', default=os.environ.get('VNCOMPRESS_TEACHER_MODEL', 'GLM-5.2'))
    ap.add_argument('--base-url', default=os.environ.get('VNCOMPRESS_TEACHER_BASE_URL'))
    ap.add_argument('--key-prefix', default='VNCOMPRESS_TEACHER_API_KEY')
    ap.add_argument('--rpm', type=int, default=int(os.environ.get('VNCOMPRESS_RPM', DEFAULT_RPM)))
    ap.add_argument('--ccu-per-key', type=int,
                    default=int(os.environ.get('VNCOMPRESS_CCU_PER_KEY', DEFAULT_CCU_PER_KEY)))
    ap.add_argument('--max-item-retries', type=int, default=3)
    ap.add_argument('--max-tokens', type=int, default=64)
    ap.add_argument('--thinking', action='store_true',
                    help='Leave the reasoning trace on (the pipeline turns it off)')
    ap.add_argument('--error-log', default='data/vncompress_vi_v2/provenance/probe_errors.jsonl')
    args = ap.parse_args()

    keys = collect_api_keys(args.key_prefix)
    if not keys:
        raise SystemExit(f"No keys under {args.key_prefix}* in .env (see .env.example)")
    pool = KeyPool(keys, rpm=args.rpm, ccu_per_key=args.ccu_per_key)

    print(f"Endpoint : {args.base_url}")
    print(f"Model    : {args.model}")
    print(f"Keys     : {len(keys)} distinct -> " +
          ", ".join(f"{k[:11]}...{k[-5:]}" for k in keys))
    print(f"Pool     : {args.ccu_per_key} concurrent/key, {args.rpm} req/min/key "
          f"-> ~{pool.theoretical_rpm()} req/min ceiling")
    print(f"Sending  : {args.requests} request(s)\n")

    from openai import OpenAI
    clients, clients_lock = {}, threading.Lock()

    def client_for(api_key):
        with clients_lock:
            if api_key.index not in clients:
                clients[api_key.index] = OpenAI(api_key=api_key.key, base_url=args.base_url)
            return clients[api_key.index]

    latencies, by_key, lock = [], Counter(), threading.Lock()
    errors = []

    def work(index, api_key, stop):
        started = time.monotonic()
        # Sent exactly as the pipeline sends it. A probe that leaves reasoning
        # on is not a probe of the pipeline: GLM-5.2 spends its whole token
        # budget thinking and returns empty content, which looks like a broken
        # endpoint rather than a probe configured differently from the job.
        extra = {} if args.thinking else {'extra_body': {'thinking': {'type': 'disabled'}}}
        response = client_for(api_key).chat.completions.create(
            model=args.model, temperature=0.0, max_tokens=args.max_tokens,
            messages=[{'role': 'user', 'content': PROMPT}], **extra)
        elapsed = time.monotonic() - started
        choice = response.choices[0]
        text = (choice.message.content or '').strip()
        if choice.finish_reason == 'length':
            raise RuntimeError(
                f"truncated at max_tokens={args.max_tokens} -- content empty. "
                "A reasoning model spends the budget before it answers.")
        usage = getattr(response, 'usage', None)
        with lock:
            latencies.append(elapsed)
            by_key[api_key.label] += 1
        return {'i': index, 'key': api_key.label, 'seconds': round(elapsed, 2),
                'reply': text[:80],
                'tokens': getattr(usage, 'total_tokens', None) if usage else None}

    def on_error(info, item):
        errors.append(dict(info, item=item,
                           at=datetime.now(timezone.utc).isoformat(timespec='seconds')))

    started = time.monotonic()
    try:
        results, stats = run_pool(range(args.requests), work, pool,
                                  on_error=on_error,
                                  max_item_retries=args.max_item_retries,
                                  progress_every=max(1, args.requests // 4))
    except PoolExhausted as exc:
        print(f"\nFAILED: {exc}")
        results, stats = [], Counter()
    wall = time.monotonic() - started

    print("\n" + "=" * 66)
    print(f"Completed        : {len(results)}/{args.requests}")
    print(f"Wall clock       : {wall:.1f}s  ->  {len(results) / max(wall, 1e-9) * 60:.0f} req/min actual")
    if latencies:
        latencies.sort()
        print(f"Latency          : median {latencies[len(latencies) // 2]:.2f}s · "
              f"min {latencies[0]:.2f}s · max {latencies[-1]:.2f}s")
    print(f"Per key          : {dict(by_key) or 'none'}")
    print(f"Key health       : {pool.summary()}")
    if stats:
        print(f"Pool stats       : {dict(stats)}")

    rate_limited = [e for e in errors if e['status'] == 429]
    auth_failed = [e for e in errors if e['kind'] == 'auth']
    other = [e for e in errors if e['status'] != 429 and e['kind'] != 'auth']
    print(f"429 (rate limit) : {len(rate_limited)}  [30s cool-down per key, item requeued]")
    print(f"Auth failures    : {len(auth_failed)}" +
          (f"  <- {sorted({e['key'] for e in auth_failed})} DEAD, fix .env" if auth_failed else ""))
    print(f"Other errors     : {len(other)}" +
          (f"  <- logged for review" if other else ""))
    for entry in other[:5]:
        print(f"    {entry['key']} HTTP {entry['status']}: {entry['error'][:120]}")

    if errors:
        os.makedirs(os.path.dirname(args.error_log) or '.', exist_ok=True)
        with open(args.error_log, 'a', encoding='utf-8') as f:
            for entry in errors:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nError log        : {args.error_log} ({len(errors)} entries)")

    if results:
        print("\nSample replies:")
        for row in sorted(results, key=lambda r: r['i'])[:5]:
            print(f"  [{row['i']:>3}] {row['key']} {row['seconds']:>5.2f}s "
                  f"{row['tokens'] or '?':>4} tok  {row['reply']!r}")

    unused = [k.label for k in pool.keys if not by_key.get(k.label)]
    if unused:
        print(f"\nNOTE: {unused} never served a request. With {args.requests} request(s) "
              f"and {pool.workers} workers that may just be too little work to reach them -- "
              f"re-probe with --requests {pool.workers * 3} to be sure.")
    return 0 if len(results) == args.requests else 1


if __name__ == '__main__':
    sys.exit(main())
