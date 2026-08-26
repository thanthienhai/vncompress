#!/usr/bin/env python3
"""Build a larger Vietnamese training corpus for the SLM external-scorer +
tone probe (run_train_slm.py), mixing two Hugging Face sources:

  - undertheseanlp/UVW-2026 (Vietnamese Wikipedia, cleaned, CC BY-SA 4.0)
    as the main source, filtered by `quality_score`.
  - bigscience-data/roots_vi_vietnamese_poetry (Fsoft AI Lab poetry, MIT)
    as a small augmentation slice for tone/diacritic density.

The Poetry dataset is *gated*: before running this script, accept the
BigScience Ethical Charter on its Hugging Face page and log in locally
with `huggingface-cli login` (or set HF_TOKEN), otherwise the poetry
step will fail with a 401/403 -- use --skip-poetry to build
Wikipedia-only in that case.

Output schema matches vcc_bench_data/wikipedia_vi_raw.json's `paragraphs`
list, so run_training.load_training_texts() (used by run_train_slm.py)
reads it with no code changes:

    {"metadata": {...}, "paragraphs": [{"text": ..., "char_length": ..., ...}, ...]}

Usage:
    python scripts/build_training_corpus.py
    python scripts/build_training_corpus.py --uvw-n 20000 --poetry-ratio 0.1
    python scripts/build_training_corpus.py --skip-poetry
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fetch_vietnamese_data import segment_paragraphs

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vcc_bench_data')
os.makedirs(DATA_DIR, exist_ok=True)

MIN_CHARS = 200  # must match run_training.load_training_texts()'s >200-char filter


def fetch_uvw_paragraphs(target_n: int, min_quality: int, seed: int, shuffle_buffer: int) -> list:
    """Stream undertheseanlp/UVW-2026, filter by quality_score, segment
    articles into paragraphs, stop once target_n paragraphs are collected."""
    from datasets import load_dataset

    print(f"Streaming undertheseanlp/UVW-2026 (train split), target={target_n} paragraphs, "
          f"min_quality_score={min_quality} ...")
    ds = load_dataset("undertheseanlp/UVW-2026", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    paragraphs = []
    articles_seen = articles_used = 0
    for row in ds:
        articles_seen += 1
        if row.get("quality_score", 0) < min_quality:
            continue
        articles_used += 1
        segs = segment_paragraphs(row.get("content", ""), min_chars=MIN_CHARS)
        for pi, para in enumerate(segs):
            paragraphs.append({
                "source": "uvw-2026",
                "topic_id": row.get("id", ""),
                "title": row.get("title", ""),
                "paragraph_index": pi,
                "text": para,
                "char_length": len(para),
                "quality_score": row.get("quality_score"),
                "main_category": row.get("main_category"),
                "wikidata_id": row.get("wikidata_id"),
            })
            if len(paragraphs) >= target_n:
                break
        if len(paragraphs) >= target_n:
            break
        if articles_seen % 2000 == 0:
            print(f"  ... scanned {articles_seen} articles, kept {articles_used}, "
                  f"{len(paragraphs)}/{target_n} paragraphs so far")

    print(f"UVW-2026: scanned {articles_seen} articles, {articles_used} passed quality filter, "
          f"collected {len(paragraphs)} paragraphs.")
    return paragraphs


def fetch_poetry_paragraphs(target_n: int, seed: int) -> list:
    """Load bigscience-data/roots_vi_vietnamese_poetry and concatenate
    consecutive poems (shuffled) into >=MIN_CHARS chunks, since most
    individual poems are shorter than the 200-char training filter."""
    from datasets import load_dataset

    print("Loading bigscience-data/roots_vi_vietnamese_poetry "
          "(requires prior HF login + Ethical Charter acceptance) ...")
    ds = load_dataset("bigscience-data/roots_vi_vietnamese_poetry", split="train")

    text_field = next((f for f in ("text", "content", "poem") if f in ds.column_names), None)
    if text_field is None:
        raise RuntimeError(
            f"Could not find a text column in roots_vi_vietnamese_poetry; "
            f"available columns: {ds.column_names}. Update fetch_poetry_paragraphs() "
            f"to point at the right field."
        )

    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)

    paragraphs = []
    buf, buf_ids = [], []
    for pi, idx in enumerate(indices):
        poem = (ds[idx][text_field] or "").strip()
        if not poem:
            continue
        buf.append(poem)
        buf_ids.append(idx)
        joined = "\n\n".join(buf)
        if len(joined) >= MIN_CHARS:
            paragraphs.append({
                "source": "vietnamese-poetry",
                "topic_id": "+".join(str(i) for i in buf_ids),
                "title": "",
                "paragraph_index": pi,
                "text": joined,
                "char_length": len(joined),
            })
            buf, buf_ids = [], []
        if len(paragraphs) >= target_n:
            break

    print(f"Poetry: collected {len(paragraphs)} paragraph chunks "
          f"(from a pool of {len(ds)} poems).")
    return paragraphs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uvw-n", type=int, default=5000, help="Target number of UVW-2026 paragraphs")
    ap.add_argument("--poetry-ratio", type=float, default=0.10,
                    help="Fraction of the final corpus drawn from Poetry (0 disables it)")
    ap.add_argument("--min-quality-score", type=int, default=7, help="Minimum UVW-2026 quality_score (1-10)")
    ap.add_argument("--shuffle-buffer", type=int, default=10000, help="Streaming shuffle buffer size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-poetry", action="store_true", help="Wikipedia-only corpus (no gated dataset needed)")
    ap.add_argument("--output", default=os.path.join(DATA_DIR, "training_corpus_v1.json"))
    args = ap.parse_args()
    if not 0 <= args.poetry_ratio < 1:
        ap.error("--poetry-ratio must be in [0, 1)")

    random.seed(args.seed)

    uvw_paragraphs = fetch_uvw_paragraphs(args.uvw_n, args.min_quality_score, args.seed, args.shuffle_buffer)

    poetry_paragraphs = []
    use_poetry = not args.skip_poetry and args.poetry_ratio > 0
    if use_poetry:
        poetry_n = max(1, round(len(uvw_paragraphs) * args.poetry_ratio / (1 - args.poetry_ratio)))
        poetry_paragraphs = fetch_poetry_paragraphs(poetry_n, args.seed)

    all_paragraphs = uvw_paragraphs + poetry_paragraphs
    random.Random(args.seed).shuffle(all_paragraphs)

    total = len(all_paragraphs)
    n_poetry = len(poetry_paragraphs)
    print(f"\nFinal corpus: {total} paragraphs "
          f"({total - n_poetry} UVW-2026 [{(total - n_poetry) / max(total, 1):.1%}], "
          f"{n_poetry} Poetry [{n_poetry / max(total, 1):.1%}])")

    output = {
        "metadata": {
            "version": "1.0.0",
            "date": time.strftime("%Y-%m-%d"),
            "sources": {
                "uvw-2026": {
                    "hf_dataset": "undertheseanlp/UVW-2026",
                    "license": "CC-BY-SA 4.0",
                    "num_paragraphs": len(uvw_paragraphs),
                    "min_quality_score": args.min_quality_score,
                },
                "vietnamese-poetry": {
                    "hf_dataset": "bigscience-data/roots_vi_vietnamese_poetry",
                    "license": "MIT",
                    "num_paragraphs": n_poetry,
                    "note": "Gated dataset; access requires accepting the BigScience Ethical Charter.",
                } if use_poetry else None,
            },
            "num_paragraphs": total,
            "seed": args.seed,
        },
        "paragraphs": all_paragraphs,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.1f} KB")
    print(f"\nNext: python run_train_slm.py --train-data-path {os.path.relpath(args.output)}")
    print("If you plan to `git add` this file (its sources are CC-BY-SA 4.0 + MIT, so it's "
          "safe to commit), also run `python scripts/checksum_datasets.py --write` and commit "
          "the updated CHECKSUMS.json in the SAME commit -- CI verifies checksums and will fail "
          "if CHECKSUMS.json references a file that isn't actually tracked.")


if __name__ == "__main__":
    main()
