#!/usr/bin/env python3
"""run_pipeline.py -- run the whole data pipeline end to end (docs/dataset_pipeline.md).

One command for the full chain, so the final dataset is reproducible from the
raw sources rather than from a shell history:

    raw sources
      -> normalize_dataset.py        canonical records + doc_id
      -> verify_dataset.py           §6.1 deterministic checks
      -> split_dataset.py            90/10 by document
      -> generate_teacher_dataset.py --stage queries        (teacher, optional)
      -> filter_dataset.py           --stage queries        §6 verify + merge
      -> generate_teacher_dataset.py --stage compression    (teacher, optional)
      -> filter_dataset.py           --stage compression
      -> merge + re-split            final train/eval over every record
      -> verify + checksum           final report

The teacher stages are skipped unless --with-teacher is passed, because they
cost money and hours; without it this is the cheap deterministic pipeline.

Re-running is safe and cheap: every stage resumes, and teacher calls that
already completed are served from the on-disk cache.

Usage:
    python scripts/run_pipeline.py                          # deterministic stages only
    python scripts/run_pipeline.py --with-teacher --workers 64 --dry-run
    python scripts/run_pipeline.py --with-teacher --workers 64 --max-queries-per-record 1
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(REPO, 'data', 'processed')
PY = sys.executable


def run(step, argv, allow_fail=False):
    print(f'\n{"=" * 78}\n== {step}\n{"=" * 78}', flush=True)
    started = time.time()
    result = subprocess.run([PY] + argv, cwd=REPO)
    elapsed = time.time() - started
    if result.returncode != 0 and not allow_fail:
        print(f'\n[FAIL] {step} exited {result.returncode} after {elapsed / 60:.1f} min')
        sys.exit(result.returncode)
    print(f'-- {step}: {elapsed / 60:.1f} min', flush=True)
    return result.returncode


def concat(sources, target):
    """Merge record streams into one file for the final split.

    Kept as a plain concatenation rather than a merge step with its own logic:
    every source is already canonical, and split_by_document groups by doc_key,
    so records that share a document reunite there no matter which file they
    came from.
    """
    present = [p for p in sources if os.path.exists(p)]
    with open(target, 'w', encoding='utf-8') as out:
        n = 0
        for path in present:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        n += 1
            print(f'   + {os.path.relpath(path, REPO)}')
    print(f'   = {n} records -> {os.path.relpath(target, REPO)}')
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--with-teacher', action='store_true',
                    help='Include the teacher stages (costs money and hours).')
    ap.add_argument('--dry-run', action='store_true', help='Teacher stages use the offline stub.')
    ap.add_argument('--workers', type=int, default=1, help='Concurrent teacher requests.')
    ap.add_argument('--limit', type=int, default=0, help='Cap records for the teacher stages.')
    ap.add_argument('--ratios', default='2,4,8')
    ap.add_argument('--max-queries-per-record', type=int, default=1,
                    help='Queries per record for the compression stage. 1 matches the §11 target '
                         'of 20k contexts x 3 ratios ~ 60k instances; 3 triples the cost.')
    ap.add_argument('--eval-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    started = time.time()

    run('1/7 normalize', ['scripts/normalize_dataset.py'])
    run('2/7 verify (source data)', ['scripts/verify_dataset.py', '--fail-on-error'])
    run('3/7 split 90/10 by document',
        ['scripts/split_dataset.py', '--eval-ratio', str(args.eval_ratio), '--seed', str(args.seed)])

    teacher_outputs = []
    if args.with_teacher:
        common = ['--workers', str(args.workers)]
        if args.dry_run:
            common.append('--dry-run')
        if args.limit:
            common += ['--limit', str(args.limit)]

        run('4/7 teacher: synthesize queries',
            ['scripts/generate_teacher_dataset.py', '--stage', 'queries'] + common)
        run('4/7 filter queries', ['scripts/filter_dataset.py', '--stage', 'queries'])
        teacher_outputs.append(os.path.join(PROCESSED, 'records_synthetic_qa.jsonl'))

        run('5/7 teacher: compress',
            ['scripts/generate_teacher_dataset.py', '--stage', 'compression',
             '--ratios', args.ratios,
             '--max-queries-per-record', str(args.max_queries_per_record)] + common)
        run('5/7 filter compression', ['scripts/filter_dataset.py', '--stage', 'compression'])
        teacher_outputs.append(os.path.join(PROCESSED, 'records_teacher.jsonl'))

        for stage in ('queries', 'compression'):
            run(f'   failures ({stage})',
                ['scripts/inspect_failures.py', '--stage', stage, '--check-output'], allow_fail=True)
    else:
        print('\n(teacher stages skipped -- pass --with-teacher to include them)')

    print(f'\n{"=" * 78}\n== 6/7 merge every record stream\n{"=" * 78}')
    merged = os.path.join(PROCESSED, 'records_all.jsonl')
    concat([os.path.join(PROCESSED, 'records.jsonl')] + teacher_outputs, merged)

    run('6/7 verify (final)', ['scripts/verify_dataset.py', '--input', merged])
    run('6/7 final split 90/10 by document',
        ['scripts/split_dataset.py', '--input', merged,
         '--eval-ratio', str(args.eval_ratio), '--seed', str(args.seed)])
    run('7/7 checksums', ['scripts/checksum_datasets.py'], allow_fail=True)

    print(f'\n{"=" * 78}')
    print(f'PIPELINE COMPLETE in {(time.time() - started) / 60:.1f} min')
    print(f'{"=" * 78}')
    print('Final artifacts:')
    for name in ('train.jsonl', 'eval.jsonl', 'vcc_bench_train.json', 'vcc_bench_eval.json',
                 'split_manifest.json', 'verification_report.json'):
        path = os.path.join(PROCESSED, name)
        if os.path.exists(path):
            print(f'  {os.path.relpath(path, REPO):46s} {os.path.getsize(path) / 1024:10.1f} KB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
