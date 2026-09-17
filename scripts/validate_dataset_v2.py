#!/usr/bin/env python3
"""validate_dataset_v2.py -- score a dataset against the VNCompress-VI v2
acceptance checklist (docs/dataset_rebuild_spec.md SS7).

    # baseline: what the v1 HuggingFace dataset scores today
    python scripts/validate_dataset_v2.py --legacy-v1

    # a v2 build
    python scripts/validate_dataset_v2.py --input data/vncompress_vi_v2

Written BEFORE the builder on purpose: the checklist is the specification, and
a builder scored by a checker written afterwards tends to be scored by a
checker that agrees with it. Running it on v1 first turns "v1 has problems"
into a number per problem, which is what the rebuild is measured against.

Exit code 0 only if every automatable check passes. Checks that cannot be
automated (human verification, dataset card prose) are reported as MANUAL and
do not fail the run -- but they are never reported as passing either.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same reasoning as build_vncompress_vi_v2.py: report lines quote Vietnamese
# document titles/text, which a Windows console's cp1252 default cannot encode.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from vncompress.dataset_schema import (
    CONFIG_FIELDS,
    CONTROLLED_DOMAINS,
    LONG_DOCUMENT_MIN_TOKENS,
    MAX_EMPTY_COLUMN_RATE,
    MIN_BUDGET_COMPLIANCE_RATE,
    MAX_EVAL_DOC_SHARE,
    MIN_EVAL_DOCUMENTS,
    MIN_HUMAN_VERIFIED_SAMPLES,
    MIN_INDEPENDENT_TEST_SAMPLES,
    MIN_JUDGE_AGREEMENT_KAPPA,
    MIN_SENTENCE_EXTRACTIVE,
    budget_compliant,
)
from vncompress.dataset_build import LANG_MIN_ALPHA_CHARS
from vncompress.linguistics import is_vietnamese

HF_CACHE_V1 = os.path.expanduser(
    '~/.cache/huggingface/hub/datasets--anhalu--vncompress-vi-dataset/snapshots'
)

PASS, FAIL, MANUAL = 'PASS', 'FAIL', 'MANUAL'


class Checklist:
    """Collects one result per SS7 item, then prints them as one table."""

    def __init__(self):
        self.rows = []

    def add(self, ref: str, name: str, status: str, detail: str):
        self.rows.append((ref, name, status, detail))

    def report(self) -> int:
        width = max(len(r[1]) for r in self.rows)
        print("\n" + "=" * (width + 46))
        print(f"{'':6} {'check':<{width}} {'status':<7} detail")
        print("-" * (width + 46))
        for ref, name, status, detail in self.rows:
            print(f"{ref:<6} {name:<{width}} {status:<7} {detail}")
        print("=" * (width + 46))
        failed = [r for r in self.rows if r[2] == FAIL]
        manual = [r for r in self.rows if r[2] == MANUAL]
        print(f"{len(self.rows) - len(failed) - len(manual)} pass · {len(failed)} FAIL · "
              f"{len(manual)} manual")
        if manual:
            print("\nManual items still have to be verified by a person before hand-off; "
                  "they are not counted as passing.")
        return 1 if failed else 0


# ============================================================================
# Loading
# ============================================================================


# Config files are globbed by prefix so a sharded build reads as one config --
# `compression_batch2.jsonl` joins `compression`. But the prefix glob is greedy,
# and `qa*.jsonl` swallows `qa_synthetic.jsonl`, which
# would merge stage 2b's teacher-written questions into the human `qa` config
# and report on the union. That is precisely the mixing stage 2b keeps in a
# separate file to prevent, so each file is assigned to the LONGEST config name
# that matches it and to that config only.
V2_CONFIGS = ('corpus', 'compression', 'qa', 'qa_synthetic', 'poetry')


def _config_for(basename: str) -> str:
    """The one config a file belongs to: longest matching name wins."""
    candidates = [name for name in V2_CONFIGS if basename.startswith(name)]
    return max(candidates, key=len) if candidates else ''


def load_v2(input_dir: str) -> dict:
    """A v2 build: one JSONL per config under `input_dir`."""
    configs = {}
    for name in V2_CONFIGS:
        matches = [p for p in glob.glob(os.path.join(input_dir, f'{name}*.jsonl'))
                   if _config_for(os.path.basename(p)) == name] + \
                  glob.glob(os.path.join(input_dir, name, '*.jsonl'))
        rows = []
        for path in sorted(matches):
            with open(path, encoding='utf-8') as f:
                rows.extend(json.loads(line) for line in f if line.strip())
        if rows:
            configs[name] = rows
    eval_rows = []
    for path in sorted(glob.glob(os.path.join(input_dir, 'eval', '*.json*'))):
        with open(path, encoding='utf-8') as f:
            content = f.read().strip()
        if path.endswith('.jsonl'):
            eval_rows.extend(json.loads(line) for line in content.splitlines() if line.strip())
        else:
            data = json.loads(content)
            eval_rows.extend(data.get('samples', data if isinstance(data, list) else []))
    if eval_rows:
        configs['eval'] = eval_rows
    return configs


def _json_meta(raw):
    """v1 keeps the interesting fields as a JSON string in `metadata`."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def load_legacy_v1(snapshot_dir: str = None) -> dict:
    """The v1 HuggingFace dataset, mapped onto v2 field names.

    The mapping is not cosmetic. v1's compression pairs are NOT the 496 rows
    with a top-level `compressed_text` in the `vcc_bench` config -- they are
    the ~52k rows whose `records.metadata` JSON carries one, and that nested
    blob also already holds `realized_ratio`, `budget_compliance`,
    `extractive_ratio` and `numbers_preserved`. Reading only the flat columns
    scores v1 on a hundredth of its compression data and reports fields as
    missing that are present, which would turn the rebuild's to-do list into
    fiction. Everything is mapped here so the checks stay written against the
    v2 contract.
    """
    import pyarrow.parquet as pq

    snapshot_dir = snapshot_dir or HF_CACHE_V1
    files = glob.glob(os.path.join(snapshot_dir, '*', '*', '*.parquet'))
    if not files:
        raise SystemExit(
            f"No v1 parquet found under {snapshot_dir}.\n"
            "Fetch it first:  python -c \"from datasets import load_dataset; "
            "load_dataset('anhalu/vncompress-vi-dataset')\""
        )

    compression, qa, corpus, independent = [], [], [], []
    for path in sorted(files):
        config = os.path.basename(os.path.dirname(path))
        split = os.path.basename(path).split('-')[0]
        for r in pq.read_table(path).to_pylist():
            meta = _json_meta(r.get('metadata'))
            quality = meta.get('quality') or {}
            teacher = meta.get('teacher') or {}
            gold = meta.get('compressed_text') or r.get('compressed_text')

            if config == 'vcc_bench' and split == 'validation':
                independent.append(dict(r, split=split))
                continue

            if gold:
                compression.append({
                    'id': r.get('id') or r.get('sample_id'), 'doc_id': r.get('doc_id'),
                    'source': r.get('source'), 'domain': r.get('domain'), 'split': split,
                    'context': r.get('context'), 'query': r.get('query') or '',
                    'gold_compression': gold,
                    'requested_ratio': meta.get('compression_ratio', r.get('compression_ratio')),
                    'target_tokens': meta.get('target_tokens', r.get('target_tokens')),
                    'realized_tokens': meta.get('realized_tokens', r.get('realized_tokens')),
                    'realized_ratio': quality.get('realized_ratio'),
                    'budget_compliance': quality.get('budget_compliance'),
                    'extractive_ratio': quality.get('extractive_ratio'),
                    'numbers_preserved': quality.get('numbers_preserved'),
                    'answerable_from_compression': meta.get('answerable_from_compression'),
                    'teacher': teacher.get('model') if isinstance(teacher, dict) else teacher,
                    'gen_config': teacher if isinstance(teacher, dict) else None,
                    'language': r.get('language'),
                })
            elif config == 'records' and r.get('kind') == 'corpus':
                corpus.append({
                    'id': r.get('id'), 'doc_id': r.get('doc_id'), 'source': r.get('source'),
                    'split': split, 'kind': r.get('kind'), 'task': r.get('task'),
                    'text': r.get('context'), 'char_length': r.get('char_length'),
                    'language': r.get('language'), 'title': meta.get('title'),
                })
            else:
                qa.append({
                    'id': r.get('id') or r.get('sample_id'), 'doc_id': r.get('doc_id'),
                    'source': r.get('source'), 'domain': r.get('domain'), 'split': split,
                    'task': r.get('task'), 'context': r.get('context'),
                    'query': r.get('query') or '',
                    'answer': meta.get('answer') or r.get('answer') or r.get('reference_answer') or '',
                    'answer_span': meta.get('answer_span') or r.get('answer_span'),
                    'question_type': meta.get('question_type') or r.get('question_type'),
                    'language': r.get('language'),
                    'needle': r.get('needle'), 'insert_position': r.get('insert_position'),
                    'num_turns': r.get('num_turns'),
                })

    out = {'compression': compression, 'qa': qa, 'corpus': corpus, 'eval': independent}
    return {k: v for k, v in out.items() if v}


# ============================================================================
# Checks (one per SS7 bullet)
# ============================================================================


def _rate(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)" if denominator else "0/0"


def check_realized_metrics(rows, cl: Checklist):
    """P1: every compression row measured from its own output, >=95% on budget."""
    if not rows:
        cl.add('P1', 'compression: realized_tokens/ratio present', FAIL, 'no compression rows')
        return
    measured = [r for r in rows if r.get('realized_tokens') is not None]
    status = PASS if len(measured) == len(rows) else FAIL
    cl.add('P1', 'compression: realized_tokens present', status, _rate(len(measured), len(rows)))

    compliant = [budget_compliant(r.get('realized_tokens'), r.get('target_tokens')) for r in rows]
    ok = sum(1 for c in compliant if c is True)
    unmeasurable = sum(1 for c in compliant if c is None)
    rate = ok / len(rows)
    detail = f"{_rate(ok, len(rows))}, need >={MIN_BUDGET_COMPLIANCE_RATE:.0%}"
    if unmeasurable:
        detail += f"; {unmeasurable} rows unmeasurable (missing target/realized)"
    cl.add('P1', 'compression: budget_compliance', PASS if rate >= MIN_BUDGET_COMPLIANCE_RATE else FAIL, detail)

    # Requested vs realized, in the SAME units on both sides: the original
    # length is target_tokens * requested_ratio, which is how the target was
    # derived in the first place. Comparing whitespace words against teacher
    # tokens instead produces nonsense (measured: a spurious "31.8x").
    #
    # Reported as MEDIAN plus the tail, never the mean. The mean is 36-47x on
    # v1 purely because a handful of rows compress 1000x+; the typical row
    # tracks its target closely. "The teacher ignores budgets" and "the teacher
    # usually hits the budget but blows out on a minority" call for different
    # fixes, and the mean cannot tell them apart.
    by_req = defaultdict(list)
    for r in rows:
        req, target, realized = r.get('requested_ratio'), r.get('target_tokens'), r.get('realized_tokens')
        if req and target and realized:
            by_req[float(req)].append((target * float(req)) / max(realized, 1))
    if by_req:
        parts = []
        for req, vals in sorted(by_req.items()):
            vals.sort()
            median = vals[len(vals) // 2]
            p90 = vals[min(len(vals) - 1, int(0.9 * len(vals)))]
            parts.append(f"asked {req:g}x -> median {median:.1f}x, p90 {p90:.1f}x, max {vals[-1]:.0f}x")
        cl.add('P1', 'compression: requested vs realized', MANUAL, "; ".join(parts))


def check_duplicates(rows, cl: Checklist):
    """P2: the three ratio levels must not be three copies of one output."""
    if not rows:
        cl.add('P2', 'compression: no duplicate (doc_id, gold_compression)', FAIL, 'no rows')
        return
    seen = Counter((r.get('doc_id'), r.get('gold_compression')) for r in rows)
    dupes = sum(c - 1 for c in seen.values() if c > 1)
    distinct_texts = len({r.get('gold_compression') for r in rows})
    cl.add('P2', 'compression: no duplicate (doc_id, gold_compression)',
           PASS if dupes == 0 else FAIL,
           f"{dupes} duplicate row(s); {distinct_texts} distinct texts over {len(rows)} rows")


def check_answerability(rows, cl: Checklist):
    """P4: the only quality signal that asks whether the compression still works."""
    if not rows:
        cl.add('P4', 'compression: answerable_from_compression', FAIL, 'no rows')
        return
    with_query = [r for r in rows if (r.get('query') or '').strip()]
    labelled = [r for r in with_query if r.get('answerable_from_compression') is not None]
    status = PASS if with_query and len(labelled) == len(with_query) else FAIL
    detail = _rate(len(labelled), len(with_query)) + " of query rows labelled"
    train_bad = sum(1 for r in labelled
                    if r.get('split') == 'train' and r.get('answerable_from_compression') is False)
    if train_bad:
        status = FAIL
        detail += f"; {train_bad} unanswerable row(s) still in train"
    cl.add('P4', 'compression: answerable_from_compression', status, detail)


def check_extractive_fidelity(rows, cl: Checklist):
    """SS3.1/B1: is the gold a COMPRESSION of the context, or a rewrite of it?

    Nothing in the SS7 checklist asks this, which is how the first v2 run
    passed every box while shipping compressions that quote 30% of their
    sentences and assert numbers their context never states. Budget compliance
    and answerability both measure the OUTPUT; neither measures whether the
    output came from the input.

    Two independent failures, so two lines:

      quotability -- `sentence_extractive_ratio`. Every wave-2 arm selects
        from the context; a gold nobody can select cannot supervise them.
      support     -- `unsupported_numbers`. A number in the compression and
        not the context is fabrication, however true it happens to be.

    Both are reported over the rows that CARRY the fields, so an older build
    without them says so instead of silently scoring zero.
    """
    if not rows:
        cl.add('B1', 'compression: gold is extractive', FAIL, 'no rows')
        return

    scored = [r for r in rows if r.get('sentence_extractive_ratio') is not None]
    if not scored:
        cl.add('B1', 'compression: gold is extractive', MANUAL,
               'no sentence_extractive_ratio field -- rebuild with the current pipeline')
    else:
        mean = sum(r['sentence_extractive_ratio'] for r in scored) / len(scored)
        low = [r for r in scored if r['sentence_extractive_ratio'] < MIN_SENTENCE_EXTRACTIVE]
        cl.add('B1', 'compression: gold is extractive',
               PASS if not low else FAIL,
               f"mean {mean:.3f}; {len(low)} row(s) below {MIN_SENTENCE_EXTRACTIVE} "
               f"of {len(scored)} scored")

    checked = [r for r in rows if r.get('unsupported_numbers') is not None]
    if not checked:
        cl.add('B1', 'compression: no unsupported numbers', MANUAL,
               'no unsupported_numbers field -- rebuild with the current pipeline')
        return
    offenders = [r for r in checked if r['unsupported_numbers']]
    example = ''
    if offenders:
        worst = max(offenders, key=lambda r: len(r['unsupported_numbers']))
        example = (f" (worst: {worst.get('doc_id')} asserts "
                   f"{worst['unsupported_numbers'][:4]})")
    cl.add('B1', 'compression: no unsupported numbers',
           PASS if not offenders else FAIL,
           _rate(len(checked) - len(offenders), len(checked)) + " clean" + example)


def check_eval_document_diversity(rows, cl: Checklist):
    """E1: 1,000 rows over 14 documents is not 1,000 independent samples.

    SS6 E1 reads "at least 300 samples, separated BY DOCUMENT from train". The
    row count alone satisfied it while four articles supplied 534 of the 1,000
    test rows -- Michael Jackson 134, Hà Nam 134, Nhà Hán 133, Texas 133. Rows
    drawn from one article share its vocabulary, its subject and its quirks, so
    a confidence interval computed over them is narrow about the article rather
    than about the method.

    Two things are reported because they fail differently: too few documents
    (nothing to generalise over) and too many rows on one document (an
    average that one article can move on its own).
    """
    if not rows:
        return
    per_doc = Counter(r.get('doc_id') for r in rows)
    documents = len(per_doc)
    cl.add('E1', f'independent test spans >= {MIN_EVAL_DOCUMENTS} documents',
           PASS if documents >= MIN_EVAL_DOCUMENTS else FAIL,
           f"{len(rows)} row(s) over {documents} document(s)")

    top_doc, top_n = per_doc.most_common(1)[0]
    share = top_n / len(rows)
    cl.add('E1', 'no single test document dominates',
           PASS if share <= MAX_EVAL_DOC_SHARE else FAIL,
           f"largest is {top_doc} with {top_n} row(s) = {share:.1%} "
           f"(cap {MAX_EVAL_DOC_SHARE:.0%})")


def check_corpus_purity(corpus_rows, compression_rows, cl: Checklist):
    """B3/P7: raw corpus must not wear a task label; no empty gold compression."""
    labelled_corpus = [r for r in corpus_rows if r.get('task')]
    cl.add('P7', 'corpus rows carry no task label',
           PASS if not labelled_corpus else FAIL,
           f"{len(labelled_corpus)} of {len(corpus_rows)} corpus rows have a task")

    empty = [r for r in compression_rows if not (r.get('gold_compression') or '').strip()]
    cl.add('B3', 'compression rows have a non-empty gold_compression',
           PASS if not empty else FAIL, f"{len(empty)} empty of {len(compression_rows)}")


def check_empty_columns(configs, cl: Checklist):
    """P6/B4: a 43-column schema that is 99% empty describes ambition, not data."""
    offenders = []
    for name, rows in configs.items():
        if not rows:
            continue
        keys = {k for r in rows for k in r}
        for key in sorted(keys):
            filled = sum(1 for r in rows if r.get(key) not in (None, '', [], {}))
            empty_rate = 1 - filled / len(rows)
            if empty_rate > MAX_EMPTY_COLUMN_RATE:
                offenders.append(f"{name}.{key} {empty_rate:.0%}")
    cl.add('P6', f'no column more than {MAX_EMPTY_COLUMN_RATE:.0%} empty',
           PASS if not offenders else FAIL,
           f"{len(offenders)} offending column(s)" + (f": {', '.join(offenders[:6])}"
                                                      + (" ..." if len(offenders) > 6 else "")
                                                      if offenders else ""))


def check_schema_fields(configs, cl: Checklist):
    """SS2: each config carries the fields the spec defines for it."""
    for name, required in CONFIG_FIELDS.items():
        rows = configs.get(name)
        if not rows:
            cl.add('SS2', f'{name}: config present', FAIL, 'missing')
            continue
        present = {k for r in rows[:2000] for k in r}
        missing = [f for f in required if f not in present]
        cl.add('SS2', f'{name}: required fields', PASS if not missing else FAIL,
               f"{len(required) - len(missing)}/{len(required)} present"
               + (f"; missing {', '.join(missing)}" if missing else ""))


def check_long_document_flag(qa_rows, cl: Checklist):
    """P5: `is_long_document` must mean what it says."""
    if not qa_rows:
        cl.add('P5', 'qa: is_long_document is correct', FAIL, 'no qa rows')
        return
    flagged = [r for r in qa_rows if r.get('is_long_document') is not None]
    if not flagged:
        lengths = sorted(len((r.get('context') or '')) for r in qa_rows)
        median = lengths[len(lengths) // 2] if lengths else 0
        cl.add('P5', 'qa: is_long_document is correct', FAIL,
               f"field absent; median context {median} chars "
               f"({'not long-document' if median < 2000 else 'plausible'})")
        return
    wrong = [r for r in flagged
             if bool(r['is_long_document']) != (len((r.get('context') or '').split()) >= LONG_DOCUMENT_MIN_TOKENS)]
    cl.add('P5', 'qa: is_long_document is correct', PASS if not wrong else FAIL,
           f"{len(wrong)} mislabelled of {len(flagged)}")


def check_multi_turn_answers(qa_rows, cl: Checklist):
    """P3/T2: each turn needs its own answer, not a copy of the transcript."""
    turns = [r for r in qa_rows if r.get('task') == 'multi_turn_conversation'
             or str(r.get('id', '')).startswith('conv_')]
    if not turns:
        cl.add('P3', 'multi-turn answers depend on the question', MANUAL, 'no multi-turn rows found')
        return
    by_doc = defaultdict(set)
    for r in turns:
        by_doc[r.get('doc_id') or r.get('context', '')[:80]].add((r.get('answer') or r.get('reference_answer') or '')[:200])
    shared = [k for k, v in by_doc.items() if len(v) == 1]
    multi = [k for k, v in by_doc.items() if len(by_doc[k]) >= 1]
    cl.add('P3', 'multi-turn answers depend on the question',
           PASS if not shared else FAIL,
           f"{len(shared)} of {len(multi)} conversation(s) reuse one answer for every question")


def check_domains(configs, cl: Checklist):
    """P8: a controlled task vocabulary, not 429 Wikidata entity types."""
    values = Counter()
    for name in ('compression', 'qa'):
        for r in configs.get(name, []):
            if r.get('domain'):
                values[r['domain']] += 1
    if not values:
        cl.add('P8', 'domain uses the controlled vocabulary', FAIL, 'no domain values')
        return
    outside = {k: v for k, v in values.items() if k not in CONTROLLED_DOMAINS}
    cl.add('P8', 'domain uses the controlled vocabulary',
           PASS if not outside else FAIL,
           f"{len(values)} distinct value(s), {len(outside)} outside the vocabulary"
           + (f" e.g. {', '.join(list(outside)[:4])}" if outside else ""))


def check_language_labels(configs, cl: Checklist, sample_size: int = 200):
    """P9/S2: language is a property of each text, not of the document.

    Only texts LONG ENOUGH to judge are scored. Re-running `is_vietnamese` over
    a three-word answer span reproduces exactly the call that produced the
    label, so the check agreed with itself on 6,000 rows while 983 of them said
    a Vietnamese answer was English -- a validator confirming a bug because it
    shares the bug's predicate. Short spans are counted and reported instead:
    they are the builder's business (`detect_lang` inherits the context's
    language there), and re-deriving them here could only ever agree.
    """
    checked = mismatched = unjudgeable = 0
    examples = []
    for name in ('compression', 'qa'):
        for r in configs.get(name, [])[:sample_size]:
            for text_field, lang_field in (('context', 'context_lang'), ('query', 'query_lang'),
                                           ('answer', 'answer_lang')):
                text, lang = r.get(text_field), r.get(lang_field) or r.get('language')
                if not text or not lang:
                    continue
                if sum(1 for c in text[:1500] if c.isalpha()) < LANG_MIN_ALPHA_CHARS:
                    unjudgeable += 1
                    continue
                checked += 1
                looks_vi = is_vietnamese(text[:1500])
                if (lang == 'vi') != looks_vi:
                    mismatched += 1
                    if len(examples) < 3:
                        examples.append(f"{name}.{text_field} labelled {lang}")
    if not checked:
        cl.add('P9', 'language labels match content', FAIL, 'no *_lang fields to check')
        return
    cl.add('P9', 'language labels match content',
           PASS if mismatched == 0 else FAIL,
           f"{mismatched} mismatch(es) in {checked} checked"
           + (f", {unjudgeable} span(s) too short to judge" if unjudgeable else "")
           + (f" ({'; '.join(examples)})" if examples else ""))


def check_synthetic_pii(configs, cl: Checklist):
    """P10/T4: identifiers that look real must be flagged as fake."""
    rows = [r for rows in configs.values() for r in rows if r.get('needle') or r.get('needle_payload')]
    if not rows:
        cl.add('P10', 'needles marked synthetic_pii', MANUAL, 'no needle rows in this dataset')
        return
    unflagged = [r for r in rows if not r.get('synthetic_pii')]
    cl.add('P10', 'needles marked synthetic_pii',
           PASS if not unflagged else FAIL, f"{len(unflagged)} unflagged of {len(rows)}")


def _check_verification(rows, input_dir, cl: Checklist):
    """E1/E2: who actually verified the test set -- and refuse to conflate them.

    An earlier version passed this check whenever `verified_by` was non-empty.
    Writing "llm:GLM-5.2" into that field would have flipped a line reading
    "independent test is human-verified" to PASS with no person involved, which
    is the self-certification the spec's "người kiểm" exists to prevent. The
    distinction is now structural: 'llm' can never satisfy the human line, and
    the human line additionally requires the judge to have been CALIBRATED
    against those human labels -- a large human sample that agrees with the
    judge by chance is not evidence about the other 850 rows.
    """
    methods = Counter(r.get('verification_method') or 'none' for r in rows)
    unverified = methods['none']
    cl.add('E2', 'every test row carries a verification verdict',
           PASS if not unverified else FAIL,
           f"{_rate(len(rows) - unverified, len(rows))} verified; "
           + ", ".join(f"{k}={v}" for k, v in sorted(methods.items())))

    human = methods['human'] + methods['llm+human']
    report_path = os.path.join(input_dir or '', 'provenance', 'verification_report.json')
    report = {}
    if os.path.exists(report_path):
        with open(report_path, encoding='utf-8') as f:
            report = json.load(f)
    kappa = (report.get('agreement') or {}).get('cohen_kappa')

    detail = (f"{human}/{len(rows)} rows human-reviewed "
              f"(floor {MIN_HUMAN_VERIFIED_SAMPLES}); kappa "
              f"{'n/a' if kappa is None else f'{kappa:.3f}'} "
              f"(floor {MIN_JUDGE_AGREEMENT_KAPPA})")
    if human == 0:
        status = FAIL
        detail += "; NO human verification -- LLM verdicts alone do not satisfy E1"
    elif human >= MIN_HUMAN_VERIFIED_SAMPLES and kappa is not None \
            and kappa >= MIN_JUDGE_AGREEMENT_KAPPA:
        status = PASS
        detail += "; judge calibrated, LLM verdicts admissible for the remainder"
    else:
        status = FAIL
        detail += "; judge NOT calibrated -- report as LLM-verified, not human-verified"
    cl.add('E1', 'independent test verified by a person (E1)', status, detail)

    # The judge must not be the teacher. Two models with one blind spot agree
    # with each other exactly where both are wrong.
    if report.get('judge_model') and report.get('teacher_model'):
        same = report['judge_model'] == report['teacher_model']
        cl.add('E3', 'judge model differs from the teacher', FAIL if same else PASS,
               f"judge={report['judge_model']}, teacher={report['teacher_model']}")

    simulated = [r for r in rows
                 if ((r.get('verification') or {}).get('llm') or {}).get('dry_run')]
    if simulated:
        cl.add('E3', 'no dry-run verdicts in the shipped test set', FAIL,
               f"{len(simulated)} row(s) carry a --dry-run verdict")


def check_independent_test(configs, input_dir, cl: Checklist):
    """E1-E3: the set results are reported on."""
    rows = configs.get('eval') or []
    cl.add('E1', f'independent test >= {MIN_INDEPENDENT_TEST_SAMPLES} samples',
           PASS if len(rows) >= MIN_INDEPENDENT_TEST_SAMPLES else FAIL, f"{len(rows)} sample(s)")
    if not rows:
        return

    teacher_tainted = [r for r in rows if r.get('teacher') or r.get('compressed_text')]
    cl.add('E3', 'independent test contains no teacher output',
           PASS if not teacher_tainted else FAIL,
           f"{len(teacher_tainted)} row(s) carry teacher output")

    # E2: an answer that merely echoes the context measures text overlap, not
    # answer quality -- the failure that made v1's 132-row validation unusable.
    echo = [r for r in rows
            if (r.get('reference_answer') or r.get('answer'))
            and (r.get('reference_answer') or r.get('answer')) in (r.get('context') or '')
            and len(r.get('reference_answer') or r.get('answer')) > 0.5 * len(r.get('context') or ' ')]
    cl.add('E2', 'test answers are not context echoes',
           PASS if not echo else FAIL, f"{len(echo)} echo-like answer(s) of {len(rows)}")

    _check_verification(rows, input_dir, cl)

    gold = [r for r in rows if (r.get('gold_compression') or '').strip()]
    cl.add('E2', 'test rows carry a verified gold_compression',
           PASS if len(gold) == len(rows) else FAIL, _rate(len(gold), len(rows)))


def check_leakage(configs, cl: Checklist):
    """E1: document-level separation between the test set and TRAIN.

    Against train specifically. An earlier version compared the test set
    against everything not named 'test', which flagged the 14 documents that
    legitimately appear in both `vcc_bench/validation` and `records/validation`
    -- the same held-out documents seen through two configs -- and reported a
    leak where there is none. A false leak report is worse than no check.
    """
    eval_docs = {r.get('doc_id') for r in configs.get('eval', []) if r.get('doc_id')}
    if not eval_docs:
        cl.add('E1', 'test/train leak-check', MANUAL, 'eval rows have no doc_id to check')
        return
    train_docs = {r.get('doc_id') for name in ('compression', 'qa', 'corpus')
                  for r in configs.get(name, []) if r.get('split') == 'train' and r.get('doc_id')}
    if not train_docs:
        cl.add('E1', 'test/train leak-check', MANUAL, "no rows with split == 'train' to compare against")
        return
    overlap = eval_docs & train_docs
    cl.add('E1', 'test/train leak-check', PASS if not overlap else FAIL,
           f"CLEAN ({len(eval_docs)} eval docs vs {len(train_docs)} train docs)"
           if not overlap else f"{len(overlap)} doc_id(s) shared with train")


def _squash(text: str) -> str:
    """Lowercase, keep only letters and digits -- 'UIT-ViQuAD 2.0' -> 'uitviquad20'."""
    return re.sub(r'[^0-9a-z]', '', (text or '').lower())


def check_card(input_dir, configs, cl: Checklist):
    """P11/S3/E4: the card has to describe the data that is actually here."""
    # Only look inside the dataset directory. Falling back to the working
    # directory reads the project README and scores the wrong document --
    # which reported "card does not mention uvw-2026" about a card that does.
    card = None
    if input_dir:
        for name in ('README.md', 'dataset_card.md', 'CARD.md'):
            path = os.path.join(input_dir, name)
            if os.path.exists(path):
                card = open(path, encoding='utf-8').read()
                break
    sources = Counter(r.get('source') for rows in configs.values() for r in rows if r.get('source'))
    top = ', '.join(f"{k}={v}" for k, v in sources.most_common(6))
    if card is None:
        cl.add('P11', 'card matches the real source distribution', MANUAL,
               f"no card found; actual sources: {top}")
        return
    # Compare identifiers, not their typography. A card writes "UIT-ViQuAD 2.0"
    # and "UVW-2026" where the data carries `uit-viquad-2.0` and `uvw-2026`;
    # a raw substring test called both absent from a card that describes them
    # in its first table. Stripping case and punctuation matches the identifier
    # a reader would recognise while still catching one that is genuinely
    # undocumented -- `legal` really is missing from the v2 card, which
    # describes the source in prose ("Văn bản pháp luật VN") and never gives
    # the value someone filtering on `source` has to type.
    normalized_card = _squash(card)
    missing = [s for s in sources if s and _squash(s) not in normalized_card]
    cl.add('P11', 'card matches the real source distribution',
           PASS if not missing else FAIL,
           f"sources absent from card: {', '.join(missing) or 'none'}")
    # The card is written for the team that uses it, so it may be in Vietnamese
    # or English. Matching only English keywords reported a card that states
    # this plainly ("tập nào dùng để báo cáo ... KHÔNG báo cáo") as not stating it.
    lower = card.lower()
    says_report = any(k in lower for k in ('report', 'báo cáo'))
    says_sanity = any(k in lower for k in ('sanity', 'kiểm tra nhanh'))
    says = says_report and says_sanity
    cl.add('E4', 'card names the reporting set vs sanity-check set',
           PASS if says else MANUAL, 'stated' if says else 'not stated explicitly')


def _legacy_card_dir(snapshot_dir: str = None) -> str:
    """The v1 dataset card lives beside the parquet files in the HF snapshot."""
    matches = glob.glob(os.path.join(snapshot_dir or HF_CACHE_V1, '*', 'README.md'))
    return os.path.dirname(matches[0]) if matches else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default=None, help='A v2 build directory (configs as JSONL)')
    ap.add_argument('--legacy-v1', action='store_true',
                    help='Score the v1 HuggingFace dataset from the local HF cache instead')
    ap.add_argument('--snapshot-dir', default=None, help='Override the v1 HF snapshot location')
    args = ap.parse_args()

    if args.legacy_v1:
        print("Scoring the LEGACY v1 dataset against the v2 checklist.")
        print("Failures here are the rebuild's to-do list, not regressions.\n")
        configs = load_legacy_v1(args.snapshot_dir)
        args.input = _legacy_card_dir(args.snapshot_dir)
    elif args.input:
        configs = load_v2(args.input)
    else:
        ap.error("Pass --input <v2 build dir> or --legacy-v1")

    if not configs:
        ap.error("No rows loaded.")
    print("Loaded: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(configs.items())))

    cl = Checklist()
    compression = configs.get('compression', [])
    check_schema_fields(configs, cl)
    check_realized_metrics(compression, cl)
    check_duplicates(compression, cl)
    check_answerability(compression, cl)
    check_extractive_fidelity(compression, cl)
    check_corpus_purity(configs.get('corpus', []), compression, cl)
    check_empty_columns(configs, cl)
    check_long_document_flag(configs.get('qa', []), cl)
    check_multi_turn_answers(configs.get('qa', []), cl)
    check_domains(configs, cl)
    check_language_labels(configs, cl)
    check_synthetic_pii(configs, cl)
    check_independent_test(configs, args.input, cl)
    check_eval_document_diversity(configs.get('eval') or [], cl)
    check_leakage(configs, cl)
    check_card(args.input, configs, cl)
    return cl.report()


if __name__ == '__main__':
    sys.exit(main())
