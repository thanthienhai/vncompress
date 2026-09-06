#!/usr/bin/env python3
"""normalize_dataset.py -- stage 1 of the data pipeline (docs/dataset_pipeline.md §5, §12).

Reads every raw source in its own ad-hoc shape and writes ONE canonical
stream, `data/processed/records.jsonl`, where each line is a
`vncompress.dataset.Record`: a stable `id`, the **`doc_id` the §9 split needs**,
`kind` (corpus vs benchmark), source/domain/task, and the text fields.

Why a separate stage: `doc_id` is the whole point. `training_corpus_v1.json`
carries the source article in `topic_id` but `load_training_texts()` threw it
away (it returned bare strings), so nothing downstream *could* split by
document. Normalization is where that provenance is preserved instead of lost.

Usage:
    python scripts/normalize_dataset.py
    python scripts/normalize_dataset.py --corpus data/benchmark/training_corpus_v1.json \
        --benchmark data/benchmark/vcc_bench_v1.json --output data/processed/records.jsonl
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vncompress.dataset import (  # noqa: E402
    KIND_BENCHMARK,
    KIND_CORPUS,
    MIN_CORPUS_CHARS,
    normalize_file,
    processed_path,
    sha256_file,
    write_jsonl,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data', 'benchmark')

DEFAULT_CORPUS = [os.path.join(DATA, 'training_corpus_v1.json')]
DEFAULT_CORPUS_FALLBACK = [os.path.join(DATA, 'wikipedia_vi_raw.json')]
DEFAULT_BENCHMARK = [os.path.join(DATA, 'vcc_bench_v1.json')]
# Built on demand by scripts/build_viquad_eval.py; gitignored, so optional.
OPTIONAL_BENCHMARK = [os.path.join(DATA, 'vcc_bench_uit_viquad_qa.json')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', action='append', default=None,
                    help='Raw corpus file (repeatable). Default: training_corpus_v1.json, '
                         'falling back to wikipedia_vi_raw.json.')
    ap.add_argument('--benchmark', action='append', default=None,
                    help='Raw VCC-Bench-shaped file (repeatable). Default: vcc_bench_v1.json '
                         '(+ vcc_bench_uit_viquad_qa.json when present).')
    ap.add_argument('--benchmark-source', default='vcc_bench',
                    help='Source tag for benchmark samples that carry none of their own.')
    ap.add_argument('--min-corpus-chars', type=int, default=MIN_CORPUS_CHARS,
                    help='Drop corpus paragraphs at or below this length (matches the filter '
                         'load_training_texts has always applied, so records.jsonl == what training sees).')
    ap.add_argument('--output', default=None, help='Default: data/processed/records.jsonl')
    args = ap.parse_args()

    corpus_paths = args.corpus
    if corpus_paths is None:
        corpus_paths = [p for p in DEFAULT_CORPUS if os.path.exists(p)]
        if not corpus_paths:
            corpus_paths = [p for p in DEFAULT_CORPUS_FALLBACK if os.path.exists(p)]
    benchmark_paths = args.benchmark
    if benchmark_paths is None:
        benchmark_paths = [p for p in DEFAULT_BENCHMARK + OPTIONAL_BENCHMARK if os.path.exists(p)]

    missing = [p for p in corpus_paths + benchmark_paths if not os.path.exists(p)]
    if missing:
        ap.error('input file(s) not found: ' + ', '.join(missing))
    if not corpus_paths and not benchmark_paths:
        ap.error('no input files found -- pass --corpus/--benchmark explicitly')

    records = []
    inputs = []
    for path in corpus_paths:
        got = normalize_file(path, min_chars=args.min_corpus_chars)
        for record in got:
            record.kind = KIND_CORPUS
        records.extend(got)
        inputs.append({'path': os.path.relpath(path, REPO), 'kind': KIND_CORPUS,
                       'n_records': len(got), 'sha256': sha256_file(path)})
        print(f'  corpus    {os.path.relpath(path, REPO)}: {len(got)} records')

    for path in benchmark_paths:
        got = normalize_file(path, default_source=args.benchmark_source)
        for record in got:
            record.kind = KIND_BENCHMARK
        records.extend(got)
        inputs.append({'path': os.path.relpath(path, REPO), 'kind': KIND_BENCHMARK,
                       'n_records': len(got), 'sha256': sha256_file(path)})
        print(f'  benchmark {os.path.relpath(path, REPO)}: {len(got)} records')

    output = args.output or processed_path('records.jsonl')
    n = write_jsonl(output, records)

    documents = {r.doc_key for r in records}
    strata = sorted({r.stratum for r in records})
    meta = {
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'min_corpus_chars': args.min_corpus_chars,
        'n_records': n,
        'n_documents': len(documents),
        'strata': strata,
        'inputs': inputs,
        'output': os.path.relpath(output, REPO),
    }
    meta_path = os.path.join(os.path.dirname(output), 'records_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'\nWrote {n} canonical records ({len(documents)} documents, {len(strata)} strata) -> '
          f'{os.path.relpath(output, REPO)}')
    for stratum in strata:
        rs = [r for r in records if r.stratum == stratum]
        print(f'  {stratum}: {len(rs)} records / {len({r.doc_key for r in rs})} documents')
    print(f'Provenance: {os.path.relpath(meta_path, REPO)}')
    print('\nNext: python scripts/verify_dataset.py && python scripts/split_dataset.py')


if __name__ == '__main__':
    main()
