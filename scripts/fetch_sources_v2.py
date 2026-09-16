#!/usr/bin/env python3
"""fetch_sources_v2.py -- stage 1 of the VNCompress-VI v2 build: fetch raw sources.

    python scripts/fetch_sources_v2.py --uvw-articles 4000
    python scripts/fetch_sources_v2.py --only viquad          # one source at a time

Writes one JSONL per source under `data/raw_v2/`, with provenance (source id,
URL, license, fetch date) attached to every record. Nothing here is derived
from the v1 dataset: v2 is built from the primary sources so that no v1
labelling decision -- notably its ratio labels and its one-sided budget flag --
can leak into it (docs/dataset_v1_baseline.md).

Sources:
  uvw      undertheseanlp/UVW-2026        Vietnamese Wikipedia, CC-BY-SA-4.0
  viquad   taidng/UIT-ViQuAD2.0           human-written QA over Wikipedia, CC-BY-SA-4.0
  legal    embedded in build_vcc_bench.py Vietnamese state legal texts, public domain
  poetry   bigscience-data/roots_vi_...   gated; skipped with a message if unavailable
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw_v2')

UVW_DATASET = 'undertheseanlp/UVW-2026'
VIQUAD_DATASET = 'taidng/UIT-ViQuAD2.0'
POETRY_DATASET = 'bigscience-data/roots_vi_vietnamese_poetry'

MIN_PARAGRAPH_CHARS = 200

# UVW-2026 ships Wikipedia's own maintenance pages ("Trang Chính", quality 3,
# content that is mostly wiki markup). They are not Vietnamese prose and must
# not become corpus rows.
UVW_MIN_QUALITY = 7
WIKI_INTERNAL_MARKERS = ('mục nội bộ Wikimedia', 'trang định hướng', 'Wikimedia')


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):,} rows -> {path}")
    return path


def segment_paragraphs(text: str, min_chars: int = MIN_PARAGRAPH_CHARS):
    """Split an article into paragraphs long enough to be worth compressing."""
    if not text:
        return []
    out = []
    for block in re.split(r'\n\s*\n', text):
        block = re.sub(r'\s+', ' ', block).strip()
        if len(block) >= min_chars:
            out.append(block)
    return out


def looks_like_wiki_markup(text: str) -> bool:
    """Reject articles that are templates/markup rather than prose."""
    if not text:
        return True
    markup = text.count('|') + text.count('{{') + text.count('__')
    return markup > max(5, len(text) / 200)


def fetch_uvw(target_articles: int, seed: int, shuffle_buffer: int):
    from datasets import load_dataset

    print(f"UVW-2026: streaming (target {target_articles:,} articles, quality >= {UVW_MIN_QUALITY})")
    ds = load_dataset(UVW_DATASET, split='train', streaming=True).shuffle(
        seed=seed, buffer_size=shuffle_buffer)
    rows, seen, kept = [], 0, 0
    for article in ds:
        seen += 1
        if article.get('quality_score', 0) < UVW_MIN_QUALITY:
            continue
        category = article.get('main_category') or ''
        if any(m in category for m in WIKI_INTERNAL_MARKERS):
            continue
        content = article.get('content') or ''
        if looks_like_wiki_markup(content):
            continue
        paragraphs = segment_paragraphs(content)
        if not paragraphs:
            continue
        kept += 1
        title = article.get('title') or article.get('id') or f'uvw_{kept}'
        for index, paragraph in enumerate(paragraphs):
            rows.append({
                'source': 'uvw-2026',
                'source_id': article.get('id'),
                'doc_id': f"uvw:{title.replace(' ', '_')}",
                'title': title,
                'paragraph_index': index,
                'text': paragraph,
                'main_category': article.get('main_category'),
                'wikidata_id': article.get('wikidata_id'),
                'quality_score': article.get('quality_score'),
                'source_url': f"https://vi.wikipedia.org/wiki/{title.replace(' ', '_')}",
                'license': 'CC-BY-SA-4.0',
                'fetched': str(date.today()),
            })
        if kept >= target_articles:
            break
        if seen % 5000 == 0:
            print(f"  scanned {seen:,} articles, kept {kept:,}, {len(rows):,} paragraphs")
    print(f"  scanned {seen:,}, kept {kept:,} articles -> {len(rows):,} paragraphs")
    return rows


def fetch_viquad():
    """Human-written questions with human-marked answer spans.

    This is what makes an honest independent test set possible: the questions
    and answers come from annotators, not from a teacher model, so scoring a
    compressor on them is not scoring it on its own teacher's output (E3).
    """
    from datasets import load_dataset

    print(f"ViQuAD: loading {VIQUAD_DATASET}")
    ds = load_dataset(VIQUAD_DATASET)
    rows = []
    for split, data in ds.items():
        for record in data:
            answers = record.get('answers') or {}
            texts = answers.get('text') or []
            starts = answers.get('answer_start') or []
            rows.append({
                'source': 'uit-viquad-2.0',
                'source_id': record.get('uit_id') or record.get('id'),
                'doc_id': f"viquad:{(record.get('title') or '').replace(' ', '_')}",
                'title': record.get('title'),
                'context': record.get('context'),
                'question': record.get('question'),
                'answer': texts[0] if texts else '',
                'answer_start': starts[0] if starts else None,
                'is_impossible': bool(record.get('is_impossible')),
                'viquad_split': split,
                'source_url': f"https://vi.wikipedia.org/wiki/{(record.get('title') or '').replace(' ', '_')}",
                'license': 'CC-BY-SA-4.0',
                'fetched': str(date.today()),
            })
    print(f"  {len(rows):,} QA rows across {sorted(ds)} splits")
    return rows


def fetch_legal():
    """Vietnamese state legal documents, embedded in the repo.

    S3: the v1 card claimed a legal source the data did not really contain.
    Emitting these as their own rows with source='legal' makes the claim true
    or visibly false -- there is no third option.
    """
    from build_vcc_bench import VIETNAMESE_LEGAL_TEXTS  # noqa: E402

    rows = []
    for law_id, law in VIETNAMESE_LEGAL_TEXTS.items():
        for chapter, articles in law.get('chapters', {}).items():
            for index, article in enumerate(articles):
                rows.append({
                    'source': 'legal',
                    'source_id': f"{law_id}:{chapter}:{index}",
                    'doc_id': f"legal:{law_id}:{chapter}",
                    'title': law.get('title'),
                    'chapter': chapter,
                    'law_id': law_id,
                    'text': article,
                    'main_category': 'legal',
                    'source_url': None,
                    'license': 'Public Domain (Vietnamese state legal documents)',
                    'fetched': str(date.today()),
                })
    print(f"Legal: {len(rows):,} articles from {len(VIETNAMESE_LEGAL_TEXTS)} law(s)")
    return rows


def fetch_poetry(target_chunks: int, seed: int):
    """Gated on HuggingFace. Returns [] with an explanation rather than failing.

    T3 keeps poetry out of the compression and QA objectives entirely, so its
    absence costs those configs nothing -- it only means no `poetry` config.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(POETRY_DATASET, split='train', streaming=True).shuffle(
            seed=seed, buffer_size=2000)
    except Exception as exc:
        print(f"Poetry: SKIPPED -- {type(exc).__name__}: {str(exc)[:160]}")
        print(f"  {POETRY_DATASET} requires accepting the BigScience Ethical Charter.")
        print("  Accept it on HuggingFace and `huggingface-cli login`, then re-run with --only poetry.")
        return []

    rows, buffer = [], ''
    try:
        for record in ds:
            buffer = (buffer + "\n" + (record.get('text') or '')).strip()
            if len(buffer) >= MIN_PARAGRAPH_CHARS:
                rows.append({
                    'source': 'vietnamese-poetry',
                    'source_id': f'poetry_{len(rows)}',
                    'doc_id': f'poetry:{len(rows) // 50}',
                    'text': buffer, 'main_category': 'poetry',
                    'source_url': None,
                    'license': 'MIT content; source dataset gated (BigScience Ethical Charter)',
                    'fetched': str(date.today()),
                })
                buffer = ''
            if len(rows) >= target_chunks:
                break
    except Exception as exc:
        print(f"Poetry: stopped early -- {type(exc).__name__}: {str(exc)[:160]}")
    print(f"Poetry: {len(rows):,} chunks")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir', default=RAW_DIR)
    ap.add_argument('--only', choices=['uvw', 'viquad', 'legal', 'poetry'], default=None)
    ap.add_argument('--uvw-articles', type=int, default=4000)
    ap.add_argument('--poetry-chunks', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--shuffle-buffer', type=int, default=10000)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for build_vcc_bench import
    written = {}
    want = (lambda name: args.only in (None, name))

    if want('legal'):
        written['legal'] = write_jsonl(os.path.join(args.output_dir, 'legal.jsonl'), fetch_legal())
    if want('viquad'):
        written['viquad'] = write_jsonl(os.path.join(args.output_dir, 'viquad.jsonl'), fetch_viquad())
    if want('uvw'):
        written['uvw'] = write_jsonl(os.path.join(args.output_dir, 'uvw.jsonl'),
                                     fetch_uvw(args.uvw_articles, args.seed, args.shuffle_buffer))
    if want('poetry'):
        rows = fetch_poetry(args.poetry_chunks, args.seed)
        if rows:
            written['poetry'] = write_jsonl(os.path.join(args.output_dir, 'poetry.jsonl'), rows)

    print(f"\nRaw sources in {args.output_dir}: {', '.join(sorted(written)) or 'none'}")
    print("Next: python scripts/build_vncompress_vi_v2.py")
    return 0


if __name__ == '__main__':
    sys.exit(main())
