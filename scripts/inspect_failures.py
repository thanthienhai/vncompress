#!/usr/bin/env python3
"""inspect_failures.py -- read the teacher failure log (docs/dataset_pipeline.md §14).

`generate_teacher_dataset.py` writes every call that exhausted its attempts to
`data/teacher/failures_<stage>.jsonl` instead of dropping it. Over a run of tens
of thousands of calls that file is the only record that a hole exists, so this
summarizes it: what failed, how it failed, and which records to look at.

Failures are **not** removed from the log when a later run succeeds -- the log
is an append-only history of what went wrong, and a row appearing here does not
mean it is still missing. Cross-check against the output file (`--check-output`)
to see which failures remain unresolved.

Replaying is just re-running the generate command: a failed call never reached
the output file, so the resume check schedules it again.

Usage:
    python scripts/inspect_failures.py --stage queries
    python scripts/inspect_failures.py --stage compression --check-output --limit 20
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEACHER_DIR = os.path.join(REPO, 'data', 'teacher')
OUTPUT_NAME = {'queries': 'queries_raw.jsonl', 'compression': 'compression_raw.jsonl'}


def _read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['queries', 'compression'], required=True)
    ap.add_argument('--teacher-dir', default=TEACHER_DIR)
    ap.add_argument('--limit', type=int, default=10, help='Example rows to print.')
    ap.add_argument('--check-output', action='store_true',
                    help='Cross-check against the output file to see which failures are still missing.')
    args = ap.parse_args()

    path = os.path.join(args.teacher_dir, f'failures_{args.stage}.jsonl')
    rows = _read_jsonl(path)
    if not rows:
        print(f'No failures recorded for stage={args.stage} '
              f'({os.path.relpath(path, REPO)} is absent or empty).')
        return 0

    print(f'{len(rows)} failure(s) logged in {os.path.relpath(path, REPO)}')
    print('\nBy error type:')
    for kind, n in Counter(r.get('error_type', '?') for r in rows).most_common():
        print(f'  {n:6d}  {kind}')

    statuses = Counter(str((r.get('detail') or {}).get('status')) for r in rows if r.get('detail'))
    if statuses:
        print('\nBy HTTP status:')
        for status, n in statuses.most_common():
            print(f'  {n:6d}  {status}')

    print('\nBy attempts made:')
    for attempts, n in Counter(str(r.get('attempts')) for r in rows).most_common():
        print(f'  {n:6d}  {attempts}')

    affected = {r.get('record_id') for r in rows if r.get('record_id')}
    print(f'\nDistinct records affected: {len(affected)}')

    if args.check_output:
        done = {r.get('key') for r in _read_jsonl(os.path.join(args.teacher_dir, OUTPUT_NAME[args.stage]))}
        unresolved = [r for r in rows if r.get('key') not in done]
        print(f'Still missing from the output: {len(unresolved)} of {len(rows)} '
              f'({len(rows) - len(unresolved)} later succeeded)')
        rows = unresolved or rows

    print(f'\nExamples (up to {args.limit}):')
    for row in rows[:args.limit]:
        print(f"  {row.get('key')}  [{row.get('error_type')}] {str(row.get('error'))[:110]}")

    print('\nReplay: re-run the same generate command -- these keys are not in the output file, '
          'so the resume check schedules them again.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
