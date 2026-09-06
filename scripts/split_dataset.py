#!/usr/bin/env python3
"""split_dataset.py -- stage 3 of the data pipeline (docs/dataset_pipeline.md §9, §10, §12).

Splits the canonical record stream 90/10 into train/eval **by document**, then
asserts the invariant §9 exists to protect: no document, record id, or context
appears on both sides.

Why this replaces the old behaviour: `run_slm_training()` used
`random_split(..., seed 42)` over *paragraphs*. `build_training_corpus.py`
segments one UVW-2026 article into up to 162 paragraphs sharing a `topic_id`,
and `build_vcc_bench.py` emits 3 query variants per paragraph and up to 5
chapter samples per law -- so a record-level split put ~65% of UVW articles on
both sides at once. That is the "Không được làm" diagram in §9, drawn exactly.

Outputs (§12 `data/processed/`):
    train.jsonl / eval.jsonl        canonical records, all kinds
    vcc_bench_train.json            benchmark-kind train split, legacy shape
                                    (E4 relevance-probe training input)
    vcc_bench_eval.json             benchmark-kind eval split, INDEPENDENT
                                    sources only (benchmark.py --data-path input)
    vcc_bench_eval_synthetic.json   benchmark-kind eval split, teacher-generated
                                    sources only -- scored separately, never pooled
    split_manifest.json             policy, seed, realized ratios, per-stratum
                                    counts, eval document keys, checksums

The legacy-shape files exist so `VCCBench.load_from_json()` and
`load_relevance_samples()` consume a split with no parser change.

**The eval side is split by provenance on purpose.** Once teacher-generated
questions are in the dataset they dominate by volume -- in the first full run,
482 of 506 eval samples. Pooling them with the original benchmark produces one
average in which a model trained on teacher output is largely being scored
against more teacher output from the same model, and the 24 independent samples
disappear into the mean. Emitting two files makes the pooled number something
you have to build deliberately rather than something you get by default.

Neither file is a substitute for the other: the synthetic set measures
generalization to unseen *documents*, the independent set measures behaviour on
data the teacher never touched. Report them separately.

Note on §10: by default the benchmark is split 90/10 like everything else, so
E4 has query-conditioned training data today. That is a compromise -- §10 would
rather the benchmark were eval-only. Pass `--eval-only-source vcc_bench
--eval-only-source wikipedia` once a separate query-conditioned training source
exists (UIT-ViQuAD's train split is the intended one, §3) to enforce the strict
policy.

Usage:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --eval-ratio 0.1 --seed 42
    python scripts/split_dataset.py --eval-only-source vcc_bench --eval-only-source wikipedia
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vncompress.dataset import (  # noqa: E402
    KIND_BENCHMARK,
    check_split_leakage,
    normalize_file,
    processed_path,
    sha256_file,
    split_by_document,
    verify_records,
    write_jsonl,
    write_vcc_bench_json,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default=None, help='Default: data/processed/records.jsonl')
    ap.add_argument('--output-dir', default=None, help='Default: data/processed/')
    ap.add_argument('--eval-ratio', type=float, default=0.1,
                    help='Record-level share of the eval side (default 0.1 -> 90/10).')
    ap.add_argument('--seed', type=int, default=42, help='Salts the document hash ordering.')
    ap.add_argument('--eval-only-source', action='append', default=[],
                    help='Send a whole source to eval (§10). Repeatable.')
    ap.add_argument('--teacher-source', action='append', default=['teacher-synth'],
                    help='Sources treated as teacher-generated, kept out of the independent eval '
                         'file. Repeatable.')
    ap.add_argument('--allow-leakage', action='store_true',
                    help='Write the split even if the no-leakage assertion fails (debugging only).')
    args = ap.parse_args()

    path = args.input or processed_path('records.jsonl')
    if not os.path.exists(path):
        ap.error(f'{path} not found -- run: python scripts/normalize_dataset.py')
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(path))

    records = normalize_file(path)
    print(f'Loaded {len(records)} canonical records from {os.path.relpath(path, REPO)}')

    train, eval_, manifest = split_by_document(
        records, eval_ratio=args.eval_ratio, seed=args.seed,
        eval_only_sources=args.eval_only_source,
    )

    print(f"\nSplit policy: {manifest['split_policy']}")
    print(f"  requested eval ratio: {args.eval_ratio}  |  realized: {manifest['eval_ratio_realized']}")
    print(f"  train: {len(train):6d} records / {manifest['train']['n_documents']:5d} documents")
    print(f"  eval : {len(eval_):6d} records / {manifest['eval']['n_documents']:5d} documents")
    print('\nPer stratum (kind/source):')
    for stratum, stats in manifest['strata'].items():
        print(f"  {stratum:34s} docs={stats['n_documents']:5d} "
              f"train={stats['train_records']:6d} eval={stats['eval_records']:5d} "
              f"({stats['eval_ratio_realized']:.1%}, {stats['mode']})")

    issues = check_split_leakage(train, eval_)
    print('\nLeakage check (§9):', 'CLEAN' if not issues else 'VIOLATIONS')
    for issue in issues:
        print(f'  - {issue}')
    if issues and not args.allow_leakage:
        print('\nRefusing to write a leaking split. Pass --allow-leakage to override.')
        return 1

    train_path = os.path.join(out_dir, 'train.jsonl')
    eval_path = os.path.join(out_dir, 'eval.jsonl')
    write_jsonl(train_path, train)
    write_jsonl(eval_path, eval_)

    teacher_sources = set(args.teacher_source)
    bench_train = [r for r in train if r.kind == KIND_BENCHMARK]
    bench_eval = [r for r in eval_ if r.kind == KIND_BENCHMARK]
    bench_eval_independent = [r for r in bench_eval if r.source not in teacher_sources]
    bench_eval_synthetic = [r for r in bench_eval if r.source in teacher_sources]

    bench_train_path = os.path.join(out_dir, 'vcc_bench_train.json')
    bench_eval_path = os.path.join(out_dir, 'vcc_bench_eval.json')
    bench_eval_synth_path = os.path.join(out_dir, 'vcc_bench_eval_synthetic.json')
    write_vcc_bench_json(bench_train_path, bench_train,
                         {'name': 'vcc-bench-train', 'split': 'train', 'seed': args.seed})
    write_vcc_bench_json(bench_eval_path, bench_eval_independent,
                         {'name': 'vcc-bench-eval', 'split': 'eval', 'seed': args.seed,
                          'provenance': 'independent (not teacher-generated)',
                          'excluded_sources': sorted(teacher_sources)})
    write_vcc_bench_json(bench_eval_synth_path, bench_eval_synthetic,
                         {'name': 'vcc-bench-eval-synthetic', 'split': 'eval', 'seed': args.seed,
                          'provenance': 'teacher-generated',
                          'sources': sorted(teacher_sources),
                          'warning': 'Do not pool with vcc_bench_eval.json: a model trained on '
                                     'teacher output scored against teacher output measures '
                                     'generalization across documents, not independent quality.'})

    manifest.update({
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input': os.path.relpath(path, REPO),
        'input_sha256': sha256_file(path),
        'leakage_check': {'clean': not issues, 'issues': issues},
        'verification': {
            'train': verify_records(train).to_dict()['stats']['counters'],
            'eval': verify_records(eval_).to_dict()['stats']['counters'],
        },
        'outputs': {
            os.path.basename(p): {'sha256': sha256_file(p), 'bytes': os.path.getsize(p)}
            for p in (train_path, eval_path, bench_train_path, bench_eval_path, bench_eval_synth_path)
        },
    })
    manifest_path = os.path.join(out_dir, 'split_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    rel = lambda p: os.path.relpath(p, REPO)  # noqa: E731
    print(f'\nWrote:\n  {rel(train_path)}\n  {rel(eval_path)}'
          f'\n  {rel(bench_train_path)} ({len(bench_train)} samples)'
          f'\n  {rel(bench_eval_path)} ({len(bench_eval_independent)} samples, INDEPENDENT)'
          f'\n  {rel(bench_eval_synth_path)} ({len(bench_eval_synthetic)} samples, teacher-generated)'
          f'\n  {rel(manifest_path)}')
    if bench_eval_synthetic:
        print(f'\n  Note: the eval side is {len(bench_eval_synthetic)} teacher-generated + '
              f'{len(bench_eval_independent)} independent samples, written to separate files. '
              f'Score them separately -- pooling hides the independent set in the mean.')
    print('\nConsumers now read the split automatically:')
    print('  train.py --mode slm                     (corpus train/eval, document-level)')
    print('  scripts/train_encoder_compressor.py     (corpus train split)')
    print(f'  scripts/train_relevance_probe.py --data-path {rel(bench_train_path)}')
    print(f'  benchmark.py --data-path {rel(bench_eval_path)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
