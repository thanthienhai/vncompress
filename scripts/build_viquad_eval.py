#!/usr/bin/env python3
"""Build a VCC-Bench-compatible QA evaluation task from UIT-ViQuAD2.0.

Purpose: measure how much a compression method degrades *downstream QA
performance* (via existing EM/ROUGE-L metrics in vncompress.evaluation)
on real Vietnamese Wikipedia reading-comprehension questions, as opposed
to the fully synthetic `long_document_qa` task in vcc_bench_v1.json.

Only the `test` split is used, and it is used for evaluation only --
never for training a compressor or scorer (this repo's compressor is
training-free; see docs/benchmark.md). Written to its own file
(`vcc_bench_uit_viquad_qa.json`), separate from vcc_bench_v1.json, so it
does not change the checksums/composition of the existing official
benchmark -- pass it explicitly via `--data-path` to opt in.

Samples are tagged task="long_document_qa" -- the same name the synthetic
task uses in vcc_bench_v1.json -- because benchmark.py's VCCBenchConfig
has a fixed `tasks` allow-list (no --tasks CLI flag to extend it); a
different task name would be silently dropped with zero samples evaluated.
This is safe because exactly one dataset file is loaded per run
(--data-path), so the synthetic and real ViQuAD samples are never mixed
in the same run -- only reused across runs on purpose.

Output schema matches the `samples` list read by
vncompress.evaluation.VCCBench.load_from_json():
    {"task", "context", "query", "reference_answer", "char_length", ...metadata}

Usage:
    python scripts/build_viquad_eval.py
    python scripts/build_viquad_eval.py --max-samples 500   # quick iteration subset
    python benchmark.py --data-path data/benchmark/vcc_bench_uit_viquad_qa.json --config configs/benchmark.json
"""
import argparse
import json
import os
import random
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'benchmark')
os.makedirs(DATA_DIR, exist_ok=True)

TASK_NAME = "long_document_qa"  # must match VCCBenchConfig's fixed task allow-list; see module docstring


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dataset", default="taidng/UIT-ViQuAD2.0",
                    help="HF dataset id (nhphuc210/UIT-ViQuAD2.0 is an alternate mirror)")
    ap.add_argument("--split", default="test", help="Only 'test' should be used for reported eval numbers")
    ap.add_argument("--include-unanswerable", action="store_true",
                    help="Also emit is_impossible=True samples with empty reference_answer "
                         "(off by default: current EM/ROUGE-L metrics assume a real answer string)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Optional cap for quick iteration (default: use the full split)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=os.path.join(DATA_DIR, "vcc_bench_uit_viquad_qa.json"))
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.hf_dataset} split={args.split} ...")
    ds = load_dataset(args.hf_dataset, split=args.split)
    print(f"Loaded {len(ds)} raw rows.")

    samples = []
    n_skipped_unanswerable = 0
    for row in ds:
        is_impossible = bool(row.get("is_impossible", False))
        if is_impossible and not args.include_unanswerable:
            n_skipped_unanswerable += 1
            continue

        context = (row.get("context") or "").strip()
        question = (row.get("question") or "").strip()
        answers = row.get("answers") or {}
        answer_texts = answers.get("text") or []
        reference_answer = answer_texts[0] if answer_texts else ""

        if not context or not question:
            continue
        if not is_impossible and not reference_answer:
            continue  # answerable but no answer text -- malformed row, skip

        samples.append({
            "task": TASK_NAME,
            "context": context,
            "query": question,
            "reference_answer": reference_answer,
            "char_length": len(context),
            "source": "UIT-ViQuAD2.0",
            "uit_id": row.get("uit_id", row.get("id", "")),
            "title": row.get("title", ""),
            "is_impossible": is_impossible,
        })

    n_unanswerable_kept = sum(1 for s in samples if s["is_impossible"])
    print(f"Kept {len(samples)} samples "
          f"({len(samples) - n_unanswerable_kept} answerable, {n_unanswerable_kept} unanswerable kept).")
    if n_skipped_unanswerable:
        print(f"Skipped {n_skipped_unanswerable} unanswerable samples "
              f"(pass --include-unanswerable to keep them).")

    if args.max_samples is not None and len(samples) > args.max_samples:
        random.Random(args.seed).shuffle(samples)
        samples = samples[:args.max_samples]
        print(f"Subsampled to --max-samples={args.max_samples}.")

    output = {
        "metadata": {
            "name": "vcc-bench-uit-viquad-qa",
            "version": "1.0.0",
            "date": time.strftime("%Y-%m-%d"),
            "source_dataset": args.hf_dataset,
            "source_split": args.split,
            "license": "Not stated on the Hugging Face dataset page as of build time -- "
                       "verify against the original UIT-ViQuAD paper/repository before any "
                       "redistribution; used here for local evaluation only.",
            "tasks": [TASK_NAME],
            "total_samples": len(samples),
            "includes_unanswerable": args.include_unanswerable,
        },
        "samples": samples,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.1f} KB")
    print("This file is gitignored (UIT-ViQuAD2.0's license isn't stated on Hugging Face) -- "
          "it stays local unless you've separately confirmed you can redistribute it.")
    print("\nNext: python benchmark.py --data-path "
          f"{os.path.relpath(args.output)} --config configs/benchmark.json")


if __name__ == "__main__":
    main()
