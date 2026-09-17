#!/usr/bin/env python3
"""generate_qa_from_corpus.py -- stage 2b: teacher-written QA over the corpus.

    # what a full run would cost, without calling anything
    python scripts/generate_qa_from_corpus.py --plan

    # exercise the whole flow with no API key and no network
    python scripts/generate_qa_from_corpus.py --limit 5 --dry-run

    # the real run
    export VNCOMPRESS_TEACHER_API_KEY=...
    python scripts/generate_qa_from_corpus.py --limit 50

WHY THIS EXISTS
UIT-ViQuAD 2.0 contains 184 Wikipedia articles, and after the document-level
split that is 143 for training. Every supervised row in v2 -- 6,000 QA pairs
and every compression built from them -- therefore stands on 143 documents,
about 126 compression pairs per document once stage 3 finishes. A relevance
probe (E4) or an encoder classifier (E6) trained on that can learn "which part
of the Nhà Minh article tends to hold answers" instead of learning the relation
between a query and a token, and no amount of additional PAIRS fixes it,
because the pairs are not the scarce thing. Documents are.

`corpus.jsonl` already holds 4,010 documents that nothing supervised uses. What
it lacks is questions, so this asks the teacher for them.

WHAT THIS IS ALLOWED TO TOUCH
TRAIN and VALIDATION only. The independent test set stays human-written
ViQuAD, which is what lets it satisfy B5/E3 ("tập test độc lập KHÔNG dùng
output teacher"). The split is re-derived here and a test document is refused
rather than filtered, so the invariant cannot quietly lapse if the corpus is
rebuilt with different splits.

Output goes to `qa_synthetic.jsonl`, NOT into `qa.jsonl`. One file, one
provenance: a reader who wants only human questions opens the human file, and
no future stage can pick up synthetic questions by accident. Stage 3 reads it
only when asked (`--include-synthetic-qa`).

EVERY ROW IS SPAN-VERIFIED
A generated answer is kept only when it occurs verbatim in the context, so
`answer_span` is computed from the text rather than trusted from the model. A
question whose answer the model paraphrased is dropped: a QA row whose span
does not land on its answer is worse than no row, and v2's human rows hold
that invariant on 7,000 of 7,000 rows. Synthetic rows meet the same bar.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same reasoning as generate_compression_pairs.py: cp1252 (Windows console
# default) cannot encode Vietnamese text, so make stdout UTF-8 defensively.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from vncompress.dataset_build import (
    assign_split,
    count_tokens,
    detect_lang,
    document_key,
    is_long_document,
    map_domain,
    qa_task_name,
    sanitize_identifiers,
)
from vncompress.api_pool import (
    DEFAULT_CCU_PER_KEY,
    DEFAULT_RPM,
    KeyPool,
    collect_api_keys,
    load_dotenv,
    run_pool,
)

load_dotenv()

# The teacher client, its retry/transport handling and its offline faker all
# live in stage 3. Loaded by path rather than duplicated: two copies of an HTTP
# client with two sets of timeout constants is how one of them silently stops
# matching the provider. Same mechanism the test-suite uses for these scripts.
_STAGE3 = importlib.util.spec_from_file_location(
    "generate_compression_pairs",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_compression_pairs.py"))
_stage3 = importlib.util.module_from_spec(_STAGE3)
_STAGE3.loader.exec_module(_stage3)
Teacher = _stage3.Teacher
TruncatedResponse = _stage3.TruncatedResponse
DEFAULT_BASE_URL = _stage3.DEFAULT_BASE_URL
DEFAULT_MODEL = _stage3.DEFAULT_MODEL

PROMPT_VERSION = 'qa-v1'

# Long enough that a question has something to be about; short enough that the
# whole document fits one prompt comfortably. Matches the context budget the
# human rows are built to, so synthetic and human rows are the same shape.
DEFAULT_CONTEXT_CHARS = 24000
MIN_CONTEXT_CHARS = 1200

# Over this many words an "extractive answer" has stopped being a span and
# become a paragraph, which makes the QA trivially answerable by copying.
MAX_ANSWER_WORDS = 25

QA_SYSTEM = (
    "Bạn là người soạn đề đọc hiểu tiếng Việt. Bạn đặt câu hỏi mà đáp án là "
    "một đoạn trích nguyên văn từ văn bản được cung cấp."
)

QA_PROMPT = """Đọc ĐOẠN VĂN và đặt {n} câu hỏi đọc hiểu.

RÀNG BUỘC BẮT BUỘC:
- Đáp án của mỗi câu hỏi phải là một đoạn NGUYÊN VĂN cắt ra từ ĐOẠN VĂN, chép chính xác từng ký tự.
- Đáp án phải ngắn (dưới {max_answer_words} từ) và cụ thể: tên, số liệu, ngày tháng, mệnh đề ngắn.
- Câu hỏi phải trả lời được CHỈ bằng ĐOẠN VĂN, không cần kiến thức bên ngoài.
- Mỗi câu hỏi hỏi về một thông tin KHÁC NHAU, trải đều trên cả đoạn văn.
- Không hỏi những câu mà đáp án là cả một đoạn dài.

Xuất đúng một mảng JSON, không kèm giải thích, không kèm markdown:
[{{"question": "...", "answer": "..."}}]

ĐOẠN VĂN:
{context}

JSON:"""


def load_corpus_documents(input_dir, max_chars=DEFAULT_CONTEXT_CHARS, seed=42,
                          holdout_docs=None):
    """Corpus paragraphs -> one training document per doc_id.

    Paragraphs are joined in file order up to `max_chars`. A document whose
    split is `test` is skipped here AND refused later; see the module docstring
    for why that invariant is enforced twice.
    """
    path = os.path.join(input_dir, 'corpus.jsonl')
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run scripts/build_vncompress_vi_v2.py first.")

    paragraphs, meta = defaultdict(list), {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get('doc_id')
            text = (row.get('text') or '').strip()
            if not doc_id or not text:
                continue
            paragraphs[doc_id].append(text)
            meta.setdefault(doc_id, row)

    documents, skipped = [], Counter()
    for doc_id, parts in paragraphs.items():
        key = document_key(doc_id)
        if assign_split(key, seed) == 'test':
            skipped['test document (human QA only)'] += 1
            continue
        if holdout_docs and key in holdout_docs:
            skipped['external benchmark holdout'] += 1
            continue
        context, used = '', 0
        for part in parts:
            if context and len(context) + len(part) + 2 > max_chars:
                break
            context = f"{context}\n\n{part}" if context else part
            used += 1
        if len(context) < MIN_CONTEXT_CHARS:
            skipped['document too short'] += 1
            continue
        source_row = meta[doc_id]
        documents.append({
            'doc_id': doc_id,
            'split': assign_split(key, seed),
            'context': context,
            'num_passages': used,
            'title': doc_id.partition(':')[2].replace('_', ' ') or doc_id,
            'source': source_row.get('source'),
            'source_url': source_row.get('source_url'),
            'license': source_row.get('license'),
            'wikidata_type': source_row.get('wikidata_type'),
        })
    documents.sort(key=lambda d: d['doc_id'])
    return documents, skipped


def parse_qa_json(raw):
    """The JSON array out of whatever the model wrapped it in.

    Models fence JSON in markdown, prefix it with a sentence, or emit it with
    trailing prose. Returning [] on anything unparseable is deliberate: the
    document is then simply dropped and counted, which is visible, whereas
    guessing at half-parsed output puts malformed rows in the dataset.
    """
    if not raw:
        return []
    text = raw.strip()
    fenced = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find('['), text.rfind(']')
    if start == -1 or end == -1 or end < start:
        return []
    try:
        parsed = json.loads(text[start:end + 1])
    except ValueError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def locate_answer(context, answer):
    """Char span of `answer` in `context`, or None if it is not verbatim.

    Exact match first, then a whitespace-tolerant retry -- a model that copies
    a span correctly but collapses a line break inside it has still copied the
    span, and rejecting that would discard good rows for a formatting
    difference. Anything looser would be paraphrase, which is what this check
    exists to reject.
    """
    if not answer:
        return None
    index = context.find(answer)
    if index != -1:
        return index, index + len(answer)
    pattern = r'\s+'.join(re.escape(part) for part in answer.split())
    match = re.search(pattern, context)
    return (match.start(), match.end()) if match else None


def build_qa_rows(document, items, teacher, dry_run=False):
    """Validated QA rows for one document, plus the reasons others were dropped."""
    rows, dropped = [], Counter()
    context = document['context']
    seen_questions, seen_spans = set(), set()

    for order, item in enumerate(items):
        question = (item.get('question') or '').strip()
        answer = (item.get('answer') or '').strip()
        if not question or not answer:
            dropped['empty question or answer'] += 1
            continue
        if len(answer.split()) > MAX_ANSWER_WORDS:
            dropped['answer too long to be a span'] += 1
            continue
        span = locate_answer(context, answer)
        if span is None:
            # The model paraphrased instead of quoting. This is the drop that
            # matters: it is exactly the failure that produced v2's rewritten
            # compressions, caught here before it becomes a label.
            dropped['answer not verbatim in context'] += 1
            continue
        normalized = ' '.join(question.lower().split())
        if normalized in seen_questions:
            dropped['duplicate question'] += 1
            continue
        if span in seen_spans:
            dropped['duplicate answer span'] += 1
            continue
        seen_questions.add(normalized)
        seen_spans.add(span)

        answer_text = context[span[0]:span[1]]
        rows.append({
            'id': f"qa_syn_{document_key(document['doc_id'])}_{order:03d}",
            'doc_id': document['doc_id'],
            'source': document['source'],
            'domain': map_domain(document.get('wikidata_type')),
            'domain_source': 'wikidata' if document.get('wikidata_type') else 'none',
            'split': document['split'],
            'context': context,
            'context_lang': detect_lang(context),
            'query': question,
            'question_type': 'extractive',
            'answer': answer_text,
            'answer_span': [span[0], span[1]],
            'answer_lang': detect_lang(answer_text, fallback=detect_lang(context)),
            'is_long_document': is_long_document(context),
            'task': qa_task_name(context),
            'num_passages': document['num_passages'],
            'title': document['title'],
            'source_url': document.get('source_url'),
            'license': document.get('license'),
            # The whole point of a separate file, repeated on every row so a
            # merged copy is still self-describing.
            'question_source': 'teacher-generated',
            'teacher': teacher.model,
            'gen_config': {
                'base_url': teacher.base_url, 'temperature': teacher.temperature,
                'prompt_version': PROMPT_VERSION, 'dry_run': bool(dry_run),
            },
        })
    return rows, dropped


def make_worker(teacher, questions_per_doc, stats, stats_lock):
    def work(document, api_key, stop):
        if document['split'] == 'test':
            # Refused, not filtered. Reaching here means the split changed
            # under us, and a synthetic question on a test document silently
            # breaks B5 for every result reported on that set afterwards.
            raise RuntimeError(f"refusing to generate QA for a test document: {document['doc_id']}")
        try:
            raw = teacher.chat(QA_SYSTEM, QA_PROMPT.format(
                n=questions_per_doc, max_answer_words=MAX_ANSWER_WORDS,
                context=document['context']), api_key, stop=stop)
        except TruncatedResponse:
            with stats_lock:
                stats['dropped: response truncated at max_tokens'] += 1
            return []
        items = parse_qa_json(raw)
        if not items:
            with stats_lock:
                stats['dropped: unparseable JSON'] += 1
            return []
        rows, dropped = build_qa_rows(document, items, teacher, teacher.dry_run)
        with stats_lock:
            stats['questions requested'] += questions_per_doc
            stats['questions returned'] += len(items)
            stats['questions kept'] += len(rows)
            for reason, n in dropped.items():
                stats[f'  dropped: {reason}'] += n
            if not rows:
                stats['documents yielding nothing'] += 1
        return rows
    return work


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default='data/vncompress_vi_v2')
    ap.add_argument('--output', default=None,
                    help='Defaults to <input>/qa_synthetic.jsonl')
    ap.add_argument('--limit', type=int, default=None, help='Documents to process')
    ap.add_argument('--questions-per-doc', type=int, default=8)
    ap.add_argument('--context-chars', type=int, default=DEFAULT_CONTEXT_CHARS)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--holdout-docs-from', action='append', default=None, metavar='PATH',
                    help='Skip documents used by an external benchmark. Repeatable.')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--base-url', default=DEFAULT_BASE_URL)
    ap.add_argument('--temperature', type=float, default=0.3,
                    help='Higher than stage 3: question variety is the point here, '
                         'whereas a compression wants the same answer every time.')
    ap.add_argument('--rpm', type=int, default=DEFAULT_RPM)
    ap.add_argument('--ccu-per-key', type=int, default=DEFAULT_CCU_PER_KEY)
    ap.add_argument('--max-item-retries', type=int, default=4)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--plan', action='store_true',
                    help='Print what a full run would cover, call nothing')
    args = ap.parse_args()

    holdout = set()
    if args.holdout_docs_from:
        build_spec = importlib.util.spec_from_file_location(
            "build_vncompress_vi_v2",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "build_vncompress_vi_v2.py"))
        build = importlib.util.module_from_spec(build_spec)
        build_spec.loader.exec_module(build)
        holdout = build.external_eval_documents(args.holdout_docs_from)

    documents, skipped = load_corpus_documents(
        args.input, args.context_chars, args.seed, holdout)
    print(f"Corpus documents eligible for synthetic QA: {len(documents):,}")
    for reason, n in skipped.most_common():
        print(f"  skipped: {reason}: {n:,}")

    if args.limit:
        documents = documents[:args.limit]
    if args.plan:
        print(f"\nWould send {len(documents):,} request(s), asking for "
              f"{args.questions_per_doc} question(s) each = "
              f"{len(documents) * args.questions_per_doc:,} candidate rows.")
        return 0
    if not documents:
        print("Nothing to do.")
        return 0

    api_keys = ['dry-run'] if args.dry_run else collect_api_keys('VNCOMPRESS_TEACHER_API_KEY')
    if not api_keys:
        raise SystemExit("No API key. Set VNCOMPRESS_TEACHER_API_KEY or use --dry-run.")
    pool = KeyPool(api_keys, rpm=args.rpm, ccu_per_key=args.ccu_per_key)
    teacher = Teacher(args.model, args.base_url, temperature=args.temperature,
                      dry_run=args.dry_run, seed=args.seed,
                      account_limiter=getattr(pool, 'account_limiter', None))
    print(f"Teacher: {args.model} @ {args.base_url}" + (" [DRY RUN]" if args.dry_run else ""))

    stats, stats_lock = Counter(), threading.Lock()
    rows = []
    started = time.monotonic()
    results, pool_stats = run_pool(
        documents, make_worker(teacher, args.questions_per_doc, stats, stats_lock),
        pool, max_item_retries=args.max_item_retries,
        progress_every=max(1, len(documents) // 10))
    stats.update(pool_stats)
    for produced in results:
        rows.extend(produced or [])

    output = args.output or os.path.join(args.input, 'qa_synthetic.jsonl')
    if args.dry_run:
        output = os.path.join(args.input, '_dryrun', 'qa_synthetic.jsonl')
    write_jsonl(output, rows)

    print("\n" + "=" * 60)
    print(f"Output: {output}" + ("  [DRY RUN -- not the real dataset]" if args.dry_run else ""))
    print(f"qa_synthetic.jsonl: {len(rows):,} rows over "
          f"{len({r['doc_id'] for r in rows})} document(s) in "
          f"{(time.monotonic() - started) / 60:.1f} min")
    for label, value in stats.most_common():
        print(f"  {label}: {value:,}")
    if rows:
        kept = stats.get('questions kept', 0) or len(rows)
        returned = stats.get('questions returned', 0)
        if returned:
            print(f"  span-verified yield: {kept}/{returned} = {100 * kept / returned:.1f}%")
        print("\nNext: python scripts/generate_compression_pairs.py "
              f"--input {args.input} --include-synthetic-qa")
    return 0


if __name__ == '__main__':
    sys.exit(main())
