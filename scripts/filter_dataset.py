#!/usr/bin/env python3
"""filter_dataset.py -- verify, filter and merge teacher output (docs/dataset_pipeline.md §6, §5).

`generate_teacher_dataset.py` keeps every raw teacher response, good or bad.
This is the stage that decides what is allowed into training, per §6: "Không
đưa toàn bộ teacher output vào training. Mỗi sample phải qua verification."

Two stages, mirroring the generator:

    --stage queries       validate synthesized questions and emit them as
                          canonical **benchmark** records, so E4 finally has
                          query-conditioned training data that is not the
                          benchmark it is scored on. The corpus paragraph's
                          `doc_id` is inherited, so the §9 document-level split
                          keeps every question about one article on one side.

    --stage compression   validate compressed instances and merge the §5
                          teacher fields onto the source record.

Rejected rows are not deleted -- they go to `data/teacher/quarantine.jsonl`
with the reason, which is what §6.3 asks for. A row rejected here can be
re-examined without re-running the teacher.

Checks applied (the §6.1 items that only make sense once teacher output exists):

  - `empty_output`      nothing usable came back
  - `not_extractive`    compressed text contains words absent from the source,
                        i.e. the teacher rewrote or invented rather than
                        extracted (the one failure mode that silently poisons a
                        compression dataset)
  - `over_budget`       realized length exceeds the target beyond tolerance.
                        Tolerance is a percentage OR a small absolute slack,
                        whichever is larger: at 8x the target can be ~11 words,
                        where 25% is under three words of room.
  - `too_short`         output is degenerate in absolute terms. Deliberately
                        NOT "far below the target": §4.3 makes the budget an
                        upper bound, and for a query-conditioned compressor the
                        right length is set by the question. A needle task
                        legitimately reduces a 7,000-word haystack to the
                        26-word needle -- rejecting that as under-budget threw
                        away the single best output in the set.
  - `degenerate`        "compressed" text is essentially the whole context
  - `number_dropped`    a number the teacher itself marked important is missing
                        from its own compressed text
  - `answer_not_verbatim` (queries) `answer_span` is not a literal span of the
                        paragraph, so it cannot anchor span-overlap supervision
  - `degenerate_answer` (queries) the answer is the whole paragraph -- the
                        exact defect that makes 166/243 VCC-Bench samples
                        unusable for measuring compression

Usage:
    python scripts/filter_dataset.py --stage queries
    python scripts/filter_dataset.py --stage compression --ratio-tolerance 0.25
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vncompress.dataset import (  # noqa: E402
    KIND_BENCHMARK,
    Record,
    normalize_file,
    processed_path,
    write_jsonl,
)
from vncompress.teacher_prompts import STAGE_COMPRESSION, STAGE_QUERIES, count_words  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEACHER_DIR = os.path.join(REPO, 'data', 'teacher')

_WORD = re.compile(r'\w+', re.UNICODE)


def _words(text):
    return _WORD.findall((text or '').lower())


def extractive_ratio(compressed, context):
    """Share of the compressed text's words that the source actually contains,
    counting multiplicity. 1.0 means purely extractive."""
    comp = Counter(_words(compressed))
    if not comp:
        return 0.0
    src = Counter(_words(context))
    covered = sum(min(n, src.get(word, 0)) for word, n in comp.items())
    return covered / sum(comp.values())


def _quarantine(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def filter_queries(raw_rows, records_by_id, args, counters, rejected):
    """Validated synthetic QA -> canonical benchmark records."""
    out = []
    seen = set()
    for row in raw_rows:
        source = records_by_id.get(row.get('record_id'))
        if source is None:
            counters['unknown_record'] += 1
            rejected.append({'reason': 'unknown_record', 'row': row})
            continue
        context = source.context
        for i, item in enumerate(row.get('queries') or []):
            query = (item.get('query') or '').strip()
            answer = (item.get('answer') or '').strip()
            span = (item.get('answer_span') or '').strip()
            reason = None
            if not query or not answer:
                reason = 'empty_output'
            elif span and span not in context:
                reason = 'answer_not_verbatim'
            elif not span and answer not in context:
                reason = 'answer_not_verbatim'
            elif len(answer) >= 0.6 * len(context):
                reason = 'degenerate_answer'
            if reason:
                counters[reason] += 1
                rejected.append({'reason': reason, 'record_id': row['record_id'], 'item': item})
                continue

            # Deduplicate on CONTENT, not on record id. VCC-Bench fans one
            # paragraph out into three query variants (`conv_0012_q0..q2`) that
            # share a context verbatim, so an id-keyed check let the same
            # synthesized question through three times.
            key = (hashlib.blake2b(context.encode('utf-8'), digest_size=8).hexdigest(),
                   ' '.join(query.lower().split()))
            if key in seen:
                counters['duplicate_query'] += 1
                continue
            seen.add(key)

            out.append(Record(
                id=f"synthqa_{row['record_id']}_{i}",
                kind=KIND_BENCHMARK,
                # A distinct source so split_dataset.py stratifies it separately
                # and split_manifest.json shows exactly how much of the training
                # signal is teacher-generated.
                source='teacher-synth',
                source_id=row['record_id'],
                # Inherited on purpose: questions about one article must not be
                # split across train and eval (§9).
                doc_id=source.doc_id,
                # And the split unit is the ORIGIN's, not this record's own: the
                # question carries the paragraph verbatim, so a different
                # doc_key would let the paragraph go to train while a question
                # about it goes to eval.
                split_group=source.doc_key,
                task='long_document_qa',
                context=context,
                query=query,
                reference_answer=span or answer,
                domain=source.domain,
                metadata={'answer': answer, 'answer_span': span,
                          'question_type': item.get('type', ''),
                          'origin_record': row['record_id'],
                          'teacher': row.get('teacher', {})},
            ))
            counters['accepted'] += 1
    return out


def filter_compression(raw_rows, records_by_id, args, counters, rejected):
    """Validated compression instances -> source record + §5 teacher fields."""
    out = []
    for row in raw_rows:
        source = records_by_id.get(row.get('record_id'))
        if source is None:
            counters['unknown_record'] += 1
            rejected.append({'reason': 'unknown_record', 'key': row.get('key')})
            continue
        payload = row.get('teacher_output') or {}
        compressed = (payload.get('compressed_text') or '').strip()
        realized = count_words(compressed)
        target = row.get('target_tokens') or 1

        # The budget is an upper bound (§4.3). A percentage tolerance alone is
        # unfair at aggressive ratios, where the target can be ~11 words, so a
        # small absolute slack applies too.
        ceiling = max(target * (1 + args.ratio_tolerance), target + args.budget_slack)
        reason = None
        if not compressed:
            reason = 'empty_output'
        elif realized > ceiling:
            reason = 'over_budget'
        elif realized < args.min_words:
            reason = 'too_short'
        elif len(compressed) >= 0.9 * len(source.context):
            reason = 'degenerate'
        else:
            score = extractive_ratio(compressed, source.context)
            if score < args.min_extractive:
                reason = 'not_extractive'
            else:
                missing = [n for n in (payload.get('numbers') or [])
                           if str(n).strip() and str(n) not in compressed]
                if missing:
                    reason = 'number_dropped'
        if reason:
            counters[reason] += 1
            rejected.append({'reason': reason, 'key': row.get('key'),
                             'realized_tokens': realized, 'target_tokens': target})
            continue

        record = Record(
            id=f"comp_{row['key'].replace('|', '_')}",
            kind=source.kind,
            source=source.source,
            source_id=source.id,
            doc_id=source.doc_id,
            split_group=source.doc_key,
            task='context_compression',
            context=source.context,
            query=row.get('query', ''),
            reference_answer=source.reference_answer,
            domain=source.domain,
            metadata={
                # §5 teacher fields, now actually populated.
                'compression_ratio': row.get('compression_ratio'),
                'target_tokens': target,
                'realized_tokens': realized,
                'token_unit': row.get('token_unit', 'whitespace'),
                'compressed_text': compressed,
                'important_spans': payload.get('important_spans') or [],
                'removed_spans': payload.get('removed_spans') or [],
                'entities': payload.get('entities') or [],
                'numbers': payload.get('numbers') or [],
                'dates': payload.get('dates') or [],
                'conditions': payload.get('conditions') or [],
                'negations': payload.get('negations') or [],
                'compression_reason': payload.get('compression_reason', ''),
                'quality': {
                    'extractive_ratio': round(extractive_ratio(compressed, source.context), 4),
                    'budget_compliance': realized <= target,
                    'realized_ratio': round(row.get('context_tokens', realized) / max(realized, 1), 3),
                },
                'teacher': row.get('teacher', {}),
            },
        )
        out.append(record)
        counters['accepted'] += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=[STAGE_QUERIES, STAGE_COMPRESSION], required=True)
    ap.add_argument('--raw', default=None, help='Default: data/teacher/<stage>_raw.jsonl')
    ap.add_argument('--records', default=None,
                    help='Canonical records the raw rows refer to. Default: data/processed/train.jsonl')
    ap.add_argument('--output', default=None,
                    help='Default: data/processed/records_synthetic_qa.jsonl (queries) '
                         'or records_teacher.jsonl (compression)')
    ap.add_argument('--ratio-tolerance', type=float, default=0.25,
                    help='Allowed overshoot of the token budget (§8). 0.25 = up to 25%% over.')
    ap.add_argument('--min-words', type=int, default=4,
                    help='Absolute floor on compressed length. Catches truncated or degenerate '
                         'output without punishing legitimately aggressive, query-focused '
                         'compression (a needle task may reduce thousands of words to a dozen).')
    ap.add_argument('--budget-slack', type=int, default=8,
                    help='Absolute words of headroom added to the percentage tolerance, so tiny '
                         'budgets at high ratios are not rejected for rounding.')
    ap.add_argument('--min-extractive', type=float, default=0.95,
                    help='Minimum share of compressed words present in the source (§6.1: no '
                         'out-of-source text for extractive compression).')
    args = ap.parse_args()

    raw_path = args.raw or os.path.join(
        TEACHER_DIR, 'queries_raw.jsonl' if args.stage == STAGE_QUERIES else 'compression_raw.jsonl')
    if not os.path.exists(raw_path):
        ap.error(f'{raw_path} not found -- run:\n'
                 f'  python scripts/generate_teacher_dataset.py --stage {args.stage}')
    records_path = args.records or processed_path('train.jsonl')
    if not os.path.exists(records_path):
        ap.error(f'{records_path} not found -- run scripts/split_dataset.py first')

    raw_rows = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_rows.append(json.loads(line))
    records_by_id = {r.id: r for r in normalize_file(records_path)}

    counters = Counter()
    rejected = []
    runner = filter_queries if args.stage == STAGE_QUERIES else filter_compression
    kept = runner(raw_rows, records_by_id, args, counters, rejected)

    default_out = ('records_synthetic_qa.jsonl' if args.stage == STAGE_QUERIES else 'records_teacher.jsonl')
    out_path = args.output or processed_path(default_out)
    write_jsonl(out_path, kept)
    quarantine_path = os.path.join(TEACHER_DIR, f'quarantine_{args.stage}.jsonl')
    _quarantine(rejected, quarantine_path)

    total = counters['accepted'] + sum(v for k, v in counters.items() if k != 'accepted')
    print(f'Stage      : {args.stage}')
    print(f'Raw rows   : {len(raw_rows)} from {os.path.relpath(raw_path, REPO)}')
    print(f'Accepted   : {counters["accepted"]} / {total} '
          f'({counters["accepted"] / max(total, 1):.1%})')
    if len(counters) > 1:
        print('Rejected (§6):')
        for reason, n in sorted(counters.items()):
            if reason != 'accepted':
                print(f'  {reason}: {n}')
    print(f'\nWrote {len(kept)} record(s) -> {os.path.relpath(out_path, REPO)}')
    if rejected:
        print(f'Quarantined {len(rejected)} row(s) -> {os.path.relpath(quarantine_path, REPO)}')

    report_path = os.path.join(TEACHER_DIR, f'filter_report_{args.stage}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({'stage': args.stage, 'raw_rows': len(raw_rows), 'accepted': counters['accepted'],
                   'counters': dict(counters), 'thresholds': {
                       'ratio_tolerance': args.ratio_tolerance,
                       'budget_slack': args.budget_slack,
                       'min_words': args.min_words,
                       'min_extractive': args.min_extractive}},
                  f, ensure_ascii=False, indent=2)
    print(f'Report: {os.path.relpath(report_path, REPO)}')

    print('\nNext: fold it into the split (documents stay intact across both files):')
    print(f'  cat data/processed/records.jsonl {os.path.relpath(out_path, REPO)} '
          f'> data/processed/records_all.jsonl')
    print('  python scripts/split_dataset.py --input data/processed/records_all.jsonl')
    return 0


if __name__ == '__main__':
    sys.exit(main())
