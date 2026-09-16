#!/usr/bin/env python3
"""build_vncompress_vi_v2.py -- stage 2: raw sources -> the v2 configs.

    python scripts/fetch_sources_v2.py            # stage 1
    python scripts/build_vncompress_vi_v2.py      # this
    python scripts/generate_compression_pairs.py  # stage 3 (teacher)
    python scripts/validate_dataset_v2.py --input data/vncompress_vi_v2

Builds `corpus`, `qa` and the independent test set from the primary sources in
`data/raw_v2/`. Nothing is carried over from the v1 dataset: v1's ratio labels
and its one-sided budget flag were wrong in ways that are invisible once mixed
into a new build (docs/dataset_v1_baseline.md), so v2 shares no rows with it.

The `compression` config is NOT written here -- it requires the teacher, and
stage 3 writes it. A build with no compression config is an unfinished build,
and the validator says so.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vncompress.dataset_build import (
    assign_split,
    build_long_context,
    count_tokens,
    detect_lang,
    document_key,
    is_long_document,
    map_domain,
    qa_task_name,
    sanitize_identifiers,
    sentence_containing,
)

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw_v2')

# S1: UVW-2026's `main_category` IS the Wikidata instance-of label -- measured
# on the 4,000 fetched articles, its most common values are "người" (685),
# "đơn vị phân loại" (234) and "quốc gia có chủ quyền" (40). That is exactly
# what `map_domain` is written against, so the mapping lives there (one rule
# set, tested once) and the raw label is kept in `wikidata_type` per S1.
#
# An earlier version of this script mapped a hand-written topic table over the
# ViQuAD ARTICLE TITLE instead, which is not a category at all: 5,775 of 6,000
# qa rows fell to `other` and the 225 that did not were titles that happened to
# contain a keyword ("Kinh tế Hàn Quốc" -> news). That is noise wearing a
# controlled vocabulary's clothes.


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _corpus_row(record, seed):
    """One raw-text row: provenance, no task label whatsoever (B3/P7)."""
    text, _ = sanitize_identifiers(record.get('text'))
    if not text:
        return None
    doc_id = record['doc_id']
    return {
        'id': f"corpus_{hashlib.sha1((doc_id + str(record.get('source_id')) + str(record.get('paragraph_index'))).encode()).hexdigest()[:16]}",
        'doc_id': doc_id,
        'source': record['source'],
        'source_url': record.get('source_url'),
        'license': record.get('license'),
        'text': text,
        'char_length': len(text),
        'token_count': count_tokens(text),
        'text_lang': detect_lang(text),
        # S1: the Wikidata instance-of label is kept, unmapped, beside the row
        # it describes. `domain` is deliberately absent -- SS2.1 gives corpus no
        # domain, and a raw-text row has no task to have a domain of.
        'wikidata_type': record.get('main_category') or None,
        'split': assign_split(document_key(doc_id), seed),
    }


def build_corpus(raw_dir, seed):
    """Prose corpus. Poetry is NOT part of it.

    T3 keeps poetry out of the compression and QA objectives; folding it into
    `corpus` would leave it one join away from both. It gets its own config so
    that using it for tone/morphology signal is a deliberate act.
    """
    rows, poetry = [], []
    for name in ('uvw', 'legal'):
        for record in read_jsonl(os.path.join(raw_dir, f'{name}.jsonl')):
            row = _corpus_row(record, seed)
            if row:
                rows.append(row)
    for record in read_jsonl(os.path.join(raw_dir, 'poetry.jsonl')):
        row = _corpus_row(record, seed)
        if row:
            poetry.append(row)
    return rows, poetry


def uvw_categories(raw_dir):
    """Wikipedia article title -> its Wikidata instance-of label.

    ViQuAD records carry no category of their own, so the only honest domain
    signal for a ViQuAD row is the one attached to the SAME article in the UVW
    dump. Coverage is partial (29 of 184 ViQuAD titles), and the rows it does
    not cover get `domain='other'` with `domain_source='none'` rather than a
    guess -- which is why `domain_source` exists: it separates "this really is
    general content" from "we had nothing to go on".
    """
    index = {}
    for record in read_jsonl(os.path.join(raw_dir, 'uvw.jsonl')):
        title = record.get('title')
        if title and title not in index:
            index[title] = record.get('main_category') or None
    return index


def group_viquad(viquad_rows):
    """Article title -> its passages in a stable order, and each question."""
    passages = defaultdict(list)
    seen = defaultdict(set)
    for record in viquad_rows:
        title = record['title']
        context = record['context']
        if context not in seen[title]:
            seen[title].add(context)
            passages[title].append(context)
    return passages


def external_eval_documents(paths):
    """Document keys already used by an EXTERNAL benchmark, so training on them
    contaminates that benchmark.

    Measured case: `viquad:Hà_Nội` sits in this build's qa TRAIN split, and 32
    of the 414 samples in data/benchmark/vcc_bench_v2.json (7.7%) are the same
    Wikipedia article -- 54 sentences verbatim on both sides. WAVE2_HANDOFF.md
    already warns against training E4 on vcc_bench; the leak arrives from the
    other direction, and this build is where it can be seen.

    Reads VCC-Bench-shaped JSON (`samples` with `title`) and the v2 JSONL
    shape, so the same flag covers a benchmark file or another dataset's
    eval split. Titles are reduced with `document_key`, which is what makes
    "Hà Nội" and `viquad:Hà_Nội` the same document.
    """
    keys = set()
    for path in paths or ():
        if not os.path.exists(path):
            raise SystemExit(f"--holdout-docs-from: {path} not found")
        with open(path, encoding='utf-8') as f:
            if path.endswith('.jsonl'):
                records = [json.loads(line) for line in f if line.strip()]
            else:
                payload = json.load(f)
                records = payload.get('samples') if isinstance(payload, dict) else payload
        for record in records or ():
            if not isinstance(record, dict):
                continue
            for field in ('doc_id', 'title'):
                value = record.get(field)
                if value:
                    keys.add(document_key(str(value).replace(' ', '_')))
                    break
    return keys


# Benchmarks in this repo that a v2 TRAIN split must not overlap. Checked on
# every build and reported whether or not --holdout-docs-from is passed, so a
# contaminated build is loud rather than silent; removing the rows stays an
# explicit choice, because it shrinks the training set.
KNOWN_EVAL_BENCHMARKS = (
    os.path.join('data', 'benchmark', 'vcc_bench_v2.json'),
)


def report_external_overlap(qa_rows, already_held_out):
    """Say plainly whether TRAIN shares documents with a benchmark in this repo."""
    train_keys = {document_key(r['doc_id']) for r in qa_rows if r.get('split') != 'test'}
    for path in KNOWN_EVAL_BENCHMARKS:
        if not os.path.exists(path):
            continue
        overlap = sorted(train_keys & external_eval_documents([path]))
        if not overlap:
            print(f"External benchmark {path}: no shared training document")
            continue
        print(f"WARNING: {len(overlap)} training document(s) also appear in {path}: "
              + ', '.join(overlap[:5]) + ('...' if len(overlap) > 5 else ''))
        if not already_held_out:
            print(f"         Training on these contaminates that benchmark. "
                  f"Re-run with --holdout-docs-from {path} to drop them from train.")


def build_qa(viquad_rows, categories, seed, long_context_chars, max_train, max_eval,
             max_eval_per_doc=None, holdout_docs=None):
    """Human-written QA, with a real document around each answer.

    Questions and answers come from UIT-ViQuAD 2.0 annotators, not a teacher,
    which is what lets the test split satisfy E3. Answers are question-specific
    by construction -- the P3 failure (every turn of a conversation sharing one
    answer) cannot occur here.
    """
    passages = group_viquad(viquad_rows)
    index_of = {title: {c: i for i, c in enumerate(ctxs)} for title, ctxs in passages.items()}

    # Round-robin over documents rather than taking the head of the file.
    # ViQuAD groups every question of an article together, so a straight cap
    # drew 1,000 test questions from 7 documents -- 1,000 rows, but nothing
    # close to 1,000 independent samples, and a per-document quirk would look
    # like a general result. Interleaving spreads the cap over all 18 test
    # documents (and all 139 train/validation ones).
    answerable = [r for r in viquad_rows if not r.get('is_impossible') and r.get('answer')]
    skipped = Counter()
    skipped['unanswerable'] = len(viquad_rows) - len(answerable)

    by_document = defaultdict(list)
    for record in answerable:
        by_document[record['doc_id']].append(record)
    ordered = []
    documents = sorted(by_document)
    for position in range(max(len(v) for v in by_document.values()) if by_document else 0):
        for doc_id in documents:
            if position < len(by_document[doc_id]):
                ordered.append(by_document[doc_id][position])

    qa_rows, eval_rows = [], []
    eval_per_doc = Counter()
    for record in ordered:
        title = record['title']
        doc_id = record['doc_id']
        # E1: split on the SOURCE-INDEPENDENT document key. `uvw:Iran` and
        # `viquad:Iran` are the same Wikipedia article; splitting on the
        # prefixed id gave them independent draws and put 3 of 18 test articles
        # into the training corpus while the leak-check still read CLEAN.
        split = assign_split(document_key(doc_id), seed)
        # A document an external benchmark already evaluates on must not enter
        # TRAIN. It stays allowed in this build's own test split: two eval sets
        # sharing a document is a reporting question, training on your own test
        # document is a broken result.
        if holdout_docs and split != 'test' and document_key(doc_id) in holdout_docs:
            skipped['external benchmark holdout'] += 1
            continue
        bucket = eval_rows if split == 'test' else qa_rows
        if split == 'test' and len(eval_rows) >= max_eval:
            skipped['eval cap'] += 1
            continue
        # Round-robin spreads the cap over the documents that EXIST; it cannot
        # conjure more. ViQuAD yields 14 test articles, so 1,000 rows meant 134
        # questions about Michael Jackson and 133 about Texas -- four articles
        # supplying 534 of them. Rows are not independent samples when they
        # share a document: a per-document quirk moves the headline number, and
        # a bootstrap CI over 1,000 such rows is narrow about nothing.
        #
        # The cap makes the trade explicit. Left unset the behaviour is
        # unchanged, so this never silently shrinks an existing build; set, it
        # buys independence at the cost of rows, and the honest report is
        # "N rows over D documents" rather than N alone.
        if split == 'test' and max_eval_per_doc and eval_per_doc[doc_id] >= max_eval_per_doc:
            skipped['eval per-document cap'] += 1
            continue
        if split != 'test' and len(qa_rows) >= max_train:
            skipped['train cap'] += 1
            continue

        context, answer_start, used = build_long_context(
            passages[title], index_of[title][record['context']],
            record.get('answer_start'), long_context_chars,
        )
        answer = record['answer']
        if answer_start is None or context[answer_start:answer_start + len(answer)] != answer:
            # The recomputed offset must land on the answer. If it does not,
            # the row is dropped rather than shipped with a span that points at
            # the wrong text -- a silently wrong answer_span is worse than none.
            skipped['answer offset mismatch'] += 1
            continue

        context, changed = sanitize_identifiers(context)
        row = {
            'id': f"qa_{record['source_id']}",
            'doc_id': doc_id,
            'source': record['source'],
            'domain': map_domain(categories.get(title)),
            # `wikidata_type` is deliberately NOT carried onto qa rows. Only 29
            # of 184 ViQuAD titles appear in the UVW dump, so the column would
            # be 86% empty -- a B4 violation, and a column that empty invites
            # being read as "this article has no Wikidata type" rather than
            # "we did not fetch it". `domain_source` says which, in one word,
            # on every row.
            'domain_source': 'uvw-category' if categories.get(title) else 'none',
            'split': split,
            'context': context,
            'context_lang': detect_lang(context),
            'query': record['question'],
            'question_type': 'extractive',
            'answer': answer,
            'answer_span': [answer_start, answer_start + len(answer)],
            # An answer span is a few words cut out of `context`, and the
            # diacritic heuristic cannot read a few words: it labelled "gan",
            # "Amsterdam" and every bare year `en`, 983 rows in all. Too short
            # to judge means inherit the language of the text it came from;
            # only a span long enough to carry the signal gets its own verdict,
            # which still lets genuinely foreign answers -- "Sanctae Romanae
            # Ecclesiae cardinalis" -- come back `en` on their own merits.
            'answer_lang': detect_lang(answer, fallback=detect_lang(context)),
            'is_long_document': is_long_document(context),
            'task': qa_task_name(context),
            'num_passages': used,
            'title': title,
            'source_url': record.get('source_url'),
            'license': record.get('license'),
            # Stated on the row, not just implied by which file it came from.
            # stage 2b writes `teacher-generated` into qa_synthetic.jsonl, and
            # the two get merged for training -- at which point the only thing
            # separating a human question from a generated one is this field.
            'question_source': 'human',
        }
        if changed:
            row['synthetic_pii'] = True
        if split == 'test':
            # E2: a gold compression that provably contains the answer. Derived
            # from the human-marked span, so it is verifiable rather than
            # trusted; `verified_by` stays null until something checks that it
            # is SUFFICIENT to answer, which stage 3 does.
            row['gold_compression'] = sentence_containing(
                context, answer_start, answer_start + len(answer))
            row['gold_compression_source'] = 'answer-sentence (human span)'
            # E2 is about SUFFICIENCY, not containment. The answer string is in
            # the sentence by construction, but that is not the same as being
            # able to answer from it: "Trần Lý sinh ra Trần Thừa." contains the
            # answer to "who was Trần Cảnh's grandfather?" and still cannot
            # answer it. Stage 4 (verify_eval_set.py) decides that and fills
            # `verified_by`. Until then the row says so rather than carrying an
            # always-null column -- an empty `verified_by` on every row is a
            # B4 violation that also reads as "checked, found nothing".
            row['verification_method'] = 'none'
        bucket.append(row)
        if split == 'test':
            eval_per_doc[doc_id] += 1
    return qa_rows, eval_rows, skipped


def leak_check(corpus, qa_rows, eval_rows):
    """E1: document-level separation, compared on the SOURCE-INDEPENDENT key.

    Comparing `doc_id` verbatim compares `uvw:Đế_quốc_La_Mã` against
    `viquad:Đế_quốc_La_Mã` and finds no overlap -- which is how the previous
    build shipped 168 of 1,000 test rows whose article was sitting in the
    training corpus, under a CLEAN leak report. The prefix is the source, not
    the document; the check has to see past it.
    """
    eval_docs = {document_key(r['doc_id']) for r in eval_rows}
    train_docs = {document_key(r['doc_id']) for r in corpus + qa_rows if r.get('split') != 'test'}
    overlap = sorted(eval_docs & train_docs)
    return {'method': 'document_key (source prefix stripped, NFC, case-folded)',
            'eval_documents': len(eval_docs), 'train_documents': len(train_docs),
            'overlap': len(overlap), 'overlapping_doc_ids': overlap[:50],
            'status': 'CLEAN' if not overlap else 'LEAK'}


def checksums(paths):
    out = {}
    for path in paths:
        with open(path, 'rb') as f:
            data = f.read().replace(b'\r\n', b'\n')
        # Forward slashes always -- see update_checksums in
        # scripts/generate_compression_pairs.py, which merges into this file.
        out[os.path.relpath(path).replace(os.sep, '/')] = {
            'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw-dir', default=RAW_DIR)
    ap.add_argument('--output', default='data/vncompress_vi_v2')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--long-context-chars', type=int, default=24000,
                    help='Budget for the document assembled around each answer')
    ap.add_argument('--max-qa', type=int, default=6000)
    ap.add_argument('--max-eval', type=int, default=1000,
                    help='Independent test rows (spec floor is 300)')
    ap.add_argument('--holdout-docs-from', action='append', default=None, metavar='PATH',
                    help='Keep documents used by an external benchmark out of TRAIN. '
                         'Repeatable. Accepts VCC-Bench JSON or v2 JSONL. Without it '
                         'the overlap is only reported, never removed.')
    ap.add_argument('--max-eval-per-doc', type=int, default=None,
                    help='Cap questions per test document. Unset keeps current '
                         'behaviour; the shipped v2 set put 534 of 1000 rows on '
                         'four articles, so rows and independent samples are not '
                         'the same number. 10 is a reasonable starting point.')
    args = ap.parse_args()

    raw_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.raw_dir, '*.jsonl')))
    if not raw_files:
        raise SystemExit(f"No raw sources in {args.raw_dir}. Run scripts/fetch_sources_v2.py first.")
    print(f"Raw sources: {', '.join(raw_files)}")

    corpus, poetry = build_corpus(args.raw_dir, args.seed)
    print(f"corpus: {len(corpus):,} rows · poetry (separate config, T3): {len(poetry):,} rows")

    categories = uvw_categories(args.raw_dir)
    viquad = read_jsonl(os.path.join(args.raw_dir, 'viquad.jsonl'))
    holdout = external_eval_documents(args.holdout_docs_from)
    qa_rows, eval_rows, skipped = build_qa(
        viquad, categories, args.seed, args.long_context_chars, args.max_qa, args.max_eval,
        args.max_eval_per_doc, holdout)
    report_external_overlap(qa_rows, holdout)
    print(f"qa: {len(qa_rows):,} rows over {len({r['doc_id'] for r in qa_rows})} documents "
          f"· independent test: {len(eval_rows):,} rows over "
          f"{len({r['doc_id'] for r in eval_rows})} documents")
    for reason, n in skipped.most_common():
        print(f"  skipped {n:,}: {reason}")

    covered = sum(1 for r in qa_rows + eval_rows if r.get('domain_source') == 'uvw-category')
    print(f"domain: {covered:,}/{len(qa_rows) + len(eval_rows):,} rows have a real category signal "
          f"(rest are 'other'/domain_source=none)")
    print("  " + ", ".join(f"{k}={v}" for k, v in
                           Counter(r['domain'] for r in qa_rows + eval_rows).most_common()))

    written = [
        write_jsonl(os.path.join(args.output, 'corpus.jsonl'), corpus),
        write_jsonl(os.path.join(args.output, 'qa.jsonl'), qa_rows),
        write_jsonl(os.path.join(args.output, 'eval', 'test.jsonl'), eval_rows),
    ]
    if poetry:
        written.append(write_jsonl(os.path.join(args.output, 'poetry.jsonl'), poetry))
    leak = leak_check(corpus, qa_rows, eval_rows)

    prov = os.path.join(args.output, 'provenance')
    os.makedirs(prov, exist_ok=True)
    with open(os.path.join(prov, 'leak_check.json'), 'w', encoding='utf-8') as f:
        json.dump(leak, f, indent=2, ensure_ascii=False)
    with open(os.path.join(prov, 'split_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({'seed': args.seed,
                   'method': 'sha256(seed:document_key(doc_id)), 80/10/10',
                   'document_key': 'source prefix stripped, NFC, case-folded -- '
                                   'one Wikipedia article gets one split across all sources',
                   'long_context_chars': args.long_context_chars,
                   'split_counts': {
                       'corpus': dict(Counter(r['split'] for r in corpus)),
                       'poetry': dict(Counter(r['split'] for r in poetry)),
                       'qa': dict(Counter(r['split'] for r in qa_rows)),
                       'eval': dict(Counter(r['split'] for r in eval_rows))},
                   'raw_sources': raw_files}, f, indent=2)
    with open(os.path.join(prov, 'CHECKSUMS.json'), 'w', encoding='utf-8') as f:
        json.dump(checksums(written), f, indent=2)

    print(f"\nLeak-check: {leak['status']} "
          f"({leak['eval_documents']} eval docs vs {leak['train_documents']} train docs)")
    print(f"Wrote {args.output}/  (corpus, qa, eval/test"
          + (", poetry)" if poetry else ")"))
    print("\nNOT built yet:")
    print("  - the `compression` config -- it needs the teacher (stage 3)")
    print("  - eval verification -- every test row is verification_method='none' until stage 4")
    print(f"Next: python scripts/generate_compression_pairs.py --input {args.output} --limit 10")
    print(f"      python scripts/verify_eval_set.py --input {args.output} --stage judge")
    return 0


if __name__ == '__main__':
    sys.exit(main())
