#!/usr/bin/env python3
"""verify_dataset.py -- stage 2 of the data pipeline (docs/dataset_pipeline.md §6.1).

Runs the deterministic checks §6 requires *before* anything is allowed into
training: schema validity, non-empty context, a query on every
query-conditioned sample, duplicate ids/contexts, and length bounds.

Two checks are specific to this dataset and worth calling out because they are
silent failures rather than crashes:

  - `degenerate_reference` -- `reference_answer` is a verbatim copy of
    `context`. 166 of 243 samples in `vcc_bench_v1.json` are built this way
    (`build_vcc_bench.py` sets `reference_answer = text` for long_document_qa
    and joins all turns for multi_turn), so ROUGE-L/BERTScore measure "how much
    text survived compression", which penalizes every ratio by construction.
  - `answer_not_in_context` -- a span-answer task whose answer is not a
    verbatim span, which weakens E4's span-overlap supervision.

Neither blocks the pipeline (they describe the dataset we have), but they must
be visible in the report rather than discovered in a results table.

Usage:
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --input data/processed/train.jsonl --fail-on-error
    python scripts/verify_dataset.py --input data/benchmark/vcc_bench_v1.json   # raw file too
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vncompress.dataset import normalize_file, processed_path, verify_records  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default=None,
                    help='Canonical .jsonl or any raw dataset file. Default: data/processed/records.jsonl')
    ap.add_argument('--report', default=None, help='Default: <input dir>/verification_report.json')
    ap.add_argument('--max-issues', type=int, default=50, help='Cap listed examples per bucket.')
    ap.add_argument('--fail-on-error', action='store_true',
                    help='Exit 1 when a blocking check fails (use in CI).')
    args = ap.parse_args()

    path = args.input or processed_path('records.jsonl')
    if not os.path.exists(path):
        ap.error(f'{path} not found -- run: python scripts/normalize_dataset.py')

    records = normalize_file(path)
    report = verify_records(records, max_issues=args.max_issues)

    print(f'Verified {report.n_records} records from {os.path.relpath(path, REPO)}')
    print(f"  documents: {report.stats['n_documents']} | unique contexts: {report.stats['n_unique_contexts']}")
    for label in ('by_kind', 'by_source', 'by_task'):
        print(f'  {label}: ' + ', '.join(f'{k}={v}' for k, v in report.stats[label].items()))

    print('\nChecks (§6.1):')
    for name, count in report.stats['counters'].items():
        flag = 'OK  ' if count == 0 else 'HIT '
        print(f'  [{flag}] {name}: {count}')

    if report.errors:
        print(f'\nERRORS ({len(report.errors)} shown):')
        for issue in report.errors:
            print(f'  - {issue}')
    if report.warnings:
        print(f'\nWARNINGS ({len(report.warnings)} shown):')
        for issue in report.warnings[:20]:
            print(f'  - {issue}')
        if len(report.warnings) > 20:
            print(f'  ... {len(report.warnings) - 20} more (see the JSON report)')

    report_path = args.report or os.path.join(os.path.dirname(os.path.abspath(path)), 'verification_report.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({'input': os.path.relpath(path, REPO), **report.to_dict()}, f, ensure_ascii=False, indent=2)
    print(f'\nReport: {os.path.relpath(report_path, REPO)}')

    print('\nVerdict:', 'PASS' if report.ok else 'FAIL (blocking errors above)')
    if args.fail_on_error and not report.ok:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
