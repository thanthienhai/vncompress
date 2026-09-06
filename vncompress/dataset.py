"""Canonical dataset layer -- docs/dataset_pipeline.md §5/§6/§9/§12, made executable.

Before this module, wave-2 read three ad-hoc JSON shapes directly and split
them with a *record-level* `random_split`. That is exactly the pattern §9
marks "Không được làm": `build_training_corpus.py` segments one UVW-2026
article into up to 162 paragraphs that all carry the same `topic_id`, and
`build_vcc_bench.py` emits three query variants per paragraph (`doc_qa_0007_q0..q2`)
plus up to 5 chapter samples per law -- so a record-level split scatters a
single document across train and eval.

This module is the one place that:

  1. **normalizes** every raw source into ONE canonical record (§5 core
     fields; the teacher-distillation fields are *reserved* by name but not
     materialized, since nothing produces them yet -- see RESERVED_FIELDS);
  2. runs the deterministic **§6.1 checks** (`verify_records`);
  3. **splits by document**, stratified per (kind, source), at a configurable
     eval ratio -- default 0.1, i.e. 90/10 (`split_by_document`);
  4. reads/writes the **§12 `data/processed/` layout**.

Deliberately stdlib-only: the split must be reproducible from a checkout with
no torch/transformers installed, and the no-leakage invariant must be testable
in milliseconds.

The split is *hash-ordered*, not RNG-shuffled: document assignment is a pure
function of (doc_key, seed), so it is reproducible across machines and Python
versions, and re-running after adding new documents leaves existing
assignments largely intact instead of reshuffling the whole corpus.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SCHEMA_VERSION = '1.0.0'

#: Canonical record fields that are always written (§5 core).
CORE_FIELDS = (
    'id', 'schema_version', 'kind', 'source', 'source_id', 'doc_id',
    'language', 'domain', 'task', 'context', 'query', 'reference_answer',
    'char_length',
)

#: §5 teacher-distillation fields. Reserved by name so a future
#: `generate_compression_dataset.py` has a fixed target, and so readers can
#: tell "not produced yet" from "misspelled". Nothing writes them today.
RESERVED_FIELDS = (
    'compression_ratio', 'target_tokens', 'compressed_text', 'important_spans',
    'token_labels', 'tone_sensitive_spans', 'entities', 'numbers', 'dates',
    'conditions', 'negations', 'relations', 'removed_spans', 'hard_negative',
    'preference', 'compression_reason', 'teacher', 'quality',
)

KIND_CORPUS = 'corpus'
KIND_BENCHMARK = 'benchmark'

#: §2.5-A: `load_training_texts` has always kept only paragraphs > 200 chars.
#: Normalization applies the same floor so records.jsonl == what training sees.
MIN_CORPUS_CHARS = 200

#: §6.1 sanity bounds, matching build_vcc_bench.validate_dataset().
SHORT_CONTEXT_CHARS = 200
LONG_CONTEXT_CHARS = 50_000

DEFAULT_PROCESSED_DIR = os.path.join('data', 'processed')

#: Tasks whose reference_answer is a span/needle inside the context, so
#: span-overlap supervision (E4) is meaningful. Mirrors
#: training.load_relevance_samples' allow-list.
SPAN_ANSWER_TASKS = ('long_document_qa', 'needle_in_haystack')

# Query-variant suffixes build_vcc_bench.py appends to a document's sample_id:
# `doc_qa_0007_q2`, `conv_0003_q1`, `cross_0000_vi_to_en`.
_VARIANT_SUFFIX = re.compile(r'_(?:q\d+|(?:vi|en)_to_(?:vi|en))$')


# ============================================================================
# Canonical record (§5)
# ============================================================================


@dataclass
class Record:
    """One canonical sample. `kind` separates the two consumer contracts:

    - ``corpus``    -- raw text only (E6 encoder distillation, SLM/tone).
    - ``benchmark`` -- (context, query, reference_answer) (E4 probe, VCC-Bench).
    """

    id: str
    kind: str
    source: str
    source_id: str
    doc_id: str
    task: str
    context: str
    query: str = ''
    reference_answer: str = ''
    language: str = 'vi'
    domain: str = 'general'
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: Explicit split unit for a DERIVED record, set to the doc_key of the
    #: record it was derived from. A teacher-generated question carries its
    #: source paragraph's text verbatim but has its own kind/source, so without
    #: this it would get a different doc_key and could land on the opposite side
    #: of the split from the paragraph it came from -- training on the text and
    #: evaluating on a question about that same text. See
    #: filter_dataset.filter_queries / filter_compression.
    split_group: Optional[str] = None

    @property
    def char_length(self) -> int:
        return len(self.context)

    @property
    def doc_key(self) -> str:
        """The unit §9 forbids splitting. Namespaced by kind+source so two
        sources can reuse an id without being merged into one document, unless
        the record explicitly declares the group it was derived from."""
        return self.split_group or f'{self.kind}/{self.source}/{self.doc_id}'

    @property
    def stratum(self) -> str:
        """Derived from doc_key, not from this record's own kind/source: a
        split group must never span two strata, or stratified splitting could
        not keep it together."""
        return '/'.join(self.doc_key.split('/')[:2])

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'id': self.id,
            'schema_version': SCHEMA_VERSION,
            'kind': self.kind,
            'source': self.source,
            'source_id': self.source_id,
            'doc_id': self.doc_id,
            'language': self.language,
            'domain': self.domain,
            'task': self.task,
            'context': self.context,
            'query': self.query,
            'reference_answer': self.reference_answer,
            'char_length': self.char_length,
        }
        if self.split_group:
            d['split_group'] = self.split_group
        if self.metadata:
            d['metadata'] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Record':
        return cls(
            id=d['id'], kind=d['kind'], source=d.get('source', 'unknown'),
            source_id=d.get('source_id', ''), doc_id=d.get('doc_id', ''),
            task=d.get('task', ''), context=d.get('context', ''),
            query=d.get('query', ''), reference_answer=d.get('reference_answer', ''),
            language=d.get('language', 'vi'), domain=d.get('domain', 'general'),
            metadata=d.get('metadata', {}), split_group=d.get('split_group'),
        )

    def to_vcc_bench_sample(self) -> Dict[str, Any]:
        """Legacy `{"samples": [...]}` shape read by
        `evaluation.VCCBench.load_from_json` and `training.load_relevance_samples`,
        so the split is a drop-in for both without touching their parsers."""
        out = dict(self.metadata)
        out.update({
            'sample_id': self.id, 'task': self.task, 'context': self.context,
            'query': self.query, 'reference_answer': self.reference_answer,
            'char_length': self.char_length, 'doc_id': self.doc_id,
            'source': self.source, 'domain': self.domain,
        })
        return out


def _blake(text: str, size: int = 8) -> str:
    return hashlib.blake2b(text.encode('utf-8'), digest_size=size).hexdigest()


# ============================================================================
# Normalization: raw shapes -> canonical records
# ============================================================================


def _corpus_doc_id(para: Dict[str, Any]) -> str:
    """UVW-2026 keeps the source article id in `topic_id`; poetry chunks get a
    composite id per chunk. Fall back to title, then to a content hash."""
    for key in ('topic_id', 'wikidata_id'):
        value = str(para.get(key) or '').strip()
        if value:
            return value
    title = str(para.get('title') or '').strip()
    if title:
        return f'title:{title}'
    return f'sha:{_blake(para.get("text", ""))}'


def _benchmark_doc_id(sample: Dict[str, Any]) -> str:
    """Group VCC-Bench samples back into the documents they were built from.

    A law is one document (its chapters are 5 separate samples); a Wikipedia
    *article* is one document (its paragraphs are separate samples, and each
    paragraph is further fanned out into 3 query variants).
    """
    law_id = str(sample.get('law_id') or '').strip()
    if law_id:
        return f'law:{law_id}'
    # `domain` carries the Wikipedia topic id for wiki-derived samples.
    if str(sample.get('source') or '') == 'wikipedia':
        topic = str(sample.get('domain') or '').strip()
        if topic:
            return f'wiki:{topic}'
    title = str(sample.get('title') or '').strip()
    sample_id = str(sample.get('sample_id') or '').strip()
    if sample_id:
        return _VARIANT_SUFFIX.sub('', sample_id)
    if title:
        return f'title:{title}'
    return f'sha:{_blake(sample.get("context", ""))}'


def normalize_corpus(
    paragraphs: Iterable[Dict[str, Any]],
    default_source: str = 'unknown',
    min_chars: int = MIN_CORPUS_CHARS,
) -> List[Record]:
    """`{"paragraphs": [{"text", "topic_id", ...}]}` -> corpus records."""
    out: List[Record] = []
    for i, para in enumerate(paragraphs):
        text = (para.get('text') or para.get('context') or '').strip()
        if len(text) <= min_chars:
            continue
        source = str(para.get('source') or default_source)
        source_id = str(para.get('topic_id') or para.get('id') or i)
        idx = para.get('paragraph_index', i)
        out.append(Record(
            id=f'corpus_{source}_{source_id}_{idx}',
            kind=KIND_CORPUS,
            source=source,
            source_id=source_id,
            doc_id=_corpus_doc_id(para),
            task='context_compression',
            context=text,
            domain=str(para.get('main_category') or 'general'),
            metadata={k: v for k, v in para.items()
                      if k not in ('text', 'context', 'char_length') and v is not None},
        ))
    return out


def normalize_benchmark(samples: Iterable[Dict[str, Any]], default_source: str = 'vcc_bench') -> List[Record]:
    """`{"samples": [{"task", "context", "query", "reference_answer", ...}]}`
    -> benchmark records."""
    out: List[Record] = []
    for i, sample in enumerate(samples):
        context = (sample.get('context') or '').strip()
        sample_id = str(sample.get('sample_id') or sample.get('uit_id') or f'{default_source}_{i:05d}')
        out.append(Record(
            id=sample_id,
            kind=KIND_BENCHMARK,
            source=str(sample.get('source') or default_source),
            source_id=sample_id,
            doc_id=_benchmark_doc_id(sample),
            task=str(sample.get('task') or 'unknown'),
            context=context,
            query=(sample.get('query') or '').strip(),
            reference_answer=(sample.get('reference_answer') or '').strip(),
            domain=str(sample.get('domain') or 'general'),
            metadata={k: v for k, v in sample.items()
                      if k not in ('task', 'context', 'query', 'reference_answer',
                                   'char_length', 'sample_id', 'source', 'domain')
                      and v is not None},
        ))
    return out


def normalize_file(path: str, default_source: Optional[str] = None, min_chars: int = MIN_CORPUS_CHARS) -> List[Record]:
    """Auto-detect a raw file's shape and normalize it.

    Handles the three legacy shapes `load_training_texts` accepts plus the
    VCC-Bench `{"samples": [...]}` shape, and canonical `.jsonl`.
    """
    if path.endswith('.jsonl'):
        return list(read_jsonl(path))
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    stem = default_source or os.path.splitext(os.path.basename(path))[0]
    if isinstance(data, dict) and 'paragraphs' in data:
        return normalize_corpus(data['paragraphs'], default_source=stem, min_chars=min_chars)
    if isinstance(data, dict) and 'samples' in data:
        return normalize_benchmark(data['samples'], default_source=stem)
    if isinstance(data, list):
        paras = [{'text': item} if isinstance(item, str) else item for item in data]
        return normalize_corpus(paras, default_source=stem, min_chars=min_chars)
    raise ValueError(f'Unrecognized dataset shape in {path}: top-level keys '
                     f'{sorted(data)[:8] if isinstance(data, dict) else type(data).__name__}')


# ============================================================================
# JSONL / legacy IO (§12)
# ============================================================================


def write_jsonl(path: str, records: Iterable[Record]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[Record]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield Record.from_dict(json.loads(line))


def write_vcc_bench_json(path: str, records: Sequence[Record], metadata: Optional[Dict[str, Any]] = None) -> int:
    """Emit benchmark records in the legacy `{"metadata", "samples"}` shape so
    `VCCBench.load_from_json` and `load_relevance_samples` read a split
    without any parser change."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    samples = [r.to_vcc_bench_sample() for r in records]
    tasks = sorted({r.task for r in records})
    meta = {'name': 'vcc-bench-split', 'version': SCHEMA_VERSION,
            'tasks': tasks, 'total_samples': len(samples)}
    meta.update(metadata or {})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'metadata': meta, 'samples': samples}, f, ensure_ascii=False, indent=2)
    return len(samples)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# Verification (§6.1 -- the deterministic checks)
# ============================================================================


@dataclass
class VerificationReport:
    n_records: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {'n_records': self.n_records, 'ok': self.ok, 'stats': self.stats,
                'errors': self.errors, 'warnings': self.warnings}


def verify_records(records: Sequence[Record], max_issues: int = 50) -> VerificationReport:
    """§6.1 deterministic checks that apply to a pre-teacher dataset.

    Errors block the pipeline (empty context, duplicate ids, a
    query-conditioned sample with no query). Warnings are reported but do not
    block -- notably `degenerate_reference`, where `reference_answer` is a
    verbatim copy of `context`: 166/243 samples of `vcc_bench_v1.json` are
    built that way, which turns "answer preservation" into "how much text
    survived" and penalizes every compression ratio by construction.
    """
    report = VerificationReport(n_records=len(records))
    seen_ids: Dict[str, int] = {}
    by_context: Dict[str, List[str]] = {}
    counters = {
        'empty_context': 0, 'duplicate_id': 0, 'missing_query': 0,
        'missing_reference': 0, 'short_context': 0, 'long_context': 0,
        'degenerate_reference': 0, 'duplicate_context': 0,
        'answer_not_in_context': 0,
    }

    def note(bucket: List[str], message: str) -> None:
        if len(bucket) < max_issues:
            bucket.append(message)

    for record in records:
        if record.id in seen_ids:
            counters['duplicate_id'] += 1
            note(report.errors, f'duplicate id: {record.id}')
        seen_ids[record.id] = seen_ids.get(record.id, 0) + 1

        if not record.context.strip():
            counters['empty_context'] += 1
            note(report.errors, f'{record.id}: empty context')
            continue

        by_context.setdefault(_blake(record.context, 16), []).append(record.id)

        if record.char_length < SHORT_CONTEXT_CHARS:
            counters['short_context'] += 1
            note(report.warnings, f'{record.id}: short context ({record.char_length} chars)')
        if record.char_length > LONG_CONTEXT_CHARS:
            counters['long_context'] += 1
            note(report.warnings, f'{record.id}: very long context ({record.char_length} chars)')

        if record.kind == KIND_BENCHMARK:
            if not record.query.strip():
                counters['missing_query'] += 1
                note(report.errors, f'{record.id}: query-conditioned sample with empty query')
            if not record.reference_answer.strip():
                counters['missing_reference'] += 1
                note(report.warnings, f'{record.id}: empty reference_answer')
            elif record.reference_answer.strip() == record.context.strip():
                counters['degenerate_reference'] += 1
                note(report.warnings,
                     f'{record.id}: reference_answer is a verbatim copy of context '
                     f'(compression metrics degrade to text-overlap)')
            elif (record.task in SPAN_ANSWER_TASKS
                  and record.reference_answer.strip() not in record.context):
                counters['answer_not_in_context'] += 1
                note(report.warnings,
                     f'{record.id}: task={record.task} but reference_answer is not a '
                     f'verbatim span of context (span-overlap supervision is weaker here)')

    for digest, ids in by_context.items():
        if len(ids) > 1:
            counters['duplicate_context'] += len(ids) - 1
            note(report.warnings, f'duplicate context shared by {len(ids)} records: {ids[:4]}')

    report.stats = {
        'counters': counters,
        'n_documents': len({r.doc_key for r in records}),
        'n_unique_contexts': len(by_context),
        'by_kind': _count_by(records, lambda r: r.kind),
        'by_source': _count_by(records, lambda r: r.source),
        'by_task': _count_by(records, lambda r: r.task),
    }
    return report


def _count_by(records: Sequence[Record], key) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        counts[key(record)] = counts.get(key(record), 0) + 1
    return dict(sorted(counts.items()))


# ============================================================================
# Document-level split (§9)
# ============================================================================


def _doc_order_key(doc_key: str, seed: int) -> str:
    """Deterministic, seed-salted ordering of documents. Pure hash, no RNG:
    reproducible across machines/Python versions, and adding new documents
    does not reshuffle the ones already assigned."""
    return _blake(f'{seed}:{doc_key}', 16)


def group_by_document(records: Iterable[Record]) -> Dict[str, List[Record]]:
    groups: Dict[str, List[Record]] = {}
    for record in records:
        groups.setdefault(record.doc_key, []).append(record)
    return groups


def split_by_document(
    records: Sequence[Record],
    eval_ratio: float = 0.1,
    seed: int = 42,
    eval_only_sources: Sequence[str] = (),
) -> Tuple[List[Record], List[Record], Dict[str, Any]]:
    """Split records into (train, eval) so that **every document lands wholly
    on one side** (§9), stratified per (kind, source).

    `eval_ratio` is honoured at the *record* level -- documents are taken in
    hash order until the stratum's eval bucket is full -- while documents stay
    intact. Because documents have wildly different sizes (a UVW article
    contributes 1-162 paragraphs), the realized record ratio is close to but
    not exactly `eval_ratio`; the manifest reports what was actually realized.

    `eval_only_sources` sends a whole source to eval (§10: a benchmark built
    to measure downstream behaviour should never also be training data).
    """
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(f'eval_ratio must be in (0, 1), got {eval_ratio}')

    eval_only = set(eval_only_sources)
    strata: Dict[str, List[Record]] = {}
    for record in records:
        strata.setdefault(record.stratum, []).append(record)

    train: List[Record] = []
    eval_: List[Record] = []
    stratum_report: Dict[str, Any] = {}

    for stratum in sorted(strata):
        stratum_records = strata[stratum]
        docs = group_by_document(stratum_records)
        source = stratum.split('/', 1)[1] if '/' in stratum else stratum

        if source in eval_only:
            eval_.extend(stratum_records)
            stratum_report[stratum] = _stratum_stats(docs, set(docs), len(stratum_records), 'eval_only')
            continue

        ordered = sorted(docs, key=lambda k: _doc_order_key(k, seed))
        target = round(eval_ratio * len(stratum_records))
        chosen: set = set()
        taken = 0
        for doc_key in ordered:
            size = len(docs[doc_key])
            if taken + size <= target:
                chosen.add(doc_key)
                taken += size
        # A stratum whose documents are all bigger than the target would get an
        # empty eval side; take its smallest document instead, so every source
        # is represented on both sides whenever it has >= 2 documents.
        if not chosen and len(docs) >= 2:
            smallest = min(ordered, key=lambda k: (len(docs[k]), _doc_order_key(k, seed)))
            chosen.add(smallest)
            taken = len(docs[smallest])
        # Never let eval swallow a stratum outright.
        if chosen and len(chosen) == len(docs):
            largest = max(ordered, key=lambda k: (len(docs[k]), _doc_order_key(k, seed)))
            chosen.discard(largest)

        for doc_key in ordered:
            (eval_ if doc_key in chosen else train).extend(docs[doc_key])
        stratum_report[stratum] = _stratum_stats(docs, chosen, len(stratum_records), 'split')

    manifest = {
        'schema_version': SCHEMA_VERSION,
        'split_policy': 'document-level, hash-ordered, stratified by (kind, source)',
        'eval_ratio_requested': eval_ratio,
        'seed': seed,
        'eval_only_sources': sorted(eval_only),
        'n_records': len(records),
        'n_documents': len(group_by_document(records)),
        'train': {'n_records': len(train), 'n_documents': len(group_by_document(train))},
        'eval': {'n_records': len(eval_), 'n_documents': len(group_by_document(eval_))},
        'eval_ratio_realized': round(len(eval_) / max(len(records), 1), 4),
        'strata': stratum_report,
        # Derived records (teacher-generated questions, compressed instances)
        # share their origin's stratum so they cannot be split away from it, so
        # the stratum table alone no longer shows how much of the data is
        # teacher-generated. This does.
        'by_source': {
            'train': _count_by(train, lambda r: r.source),
            'eval': _count_by(eval_, lambda r: r.source),
        },
        'eval_doc_keys': sorted({r.doc_key for r in eval_}),
        'train_doc_keys_sha256': hashlib.sha256(
            '\n'.join(sorted({r.doc_key for r in train})).encode('utf-8')).hexdigest(),
    }
    return train, eval_, manifest


def _stratum_stats(docs: Dict[str, List[Record]], chosen: set, n_records: int, mode: str) -> Dict[str, Any]:
    eval_n = sum(len(docs[k]) for k in chosen)
    return {
        'mode': mode,
        'n_documents': len(docs),
        'n_records': n_records,
        'eval_documents': len(chosen),
        'eval_records': eval_n,
        'train_records': n_records - eval_n,
        'eval_ratio_realized': round(eval_n / max(n_records, 1), 4),
    }


def check_split_leakage(train: Sequence[Record], eval_: Sequence[Record]) -> List[str]:
    """The invariant §9 exists to protect. Returns a list of violations
    (empty == clean), so callers can assert on it."""
    issues: List[str] = []
    train_docs = {r.doc_key for r in train}
    eval_docs = {r.doc_key for r in eval_}
    shared_docs = train_docs & eval_docs
    if shared_docs:
        issues.append(f'{len(shared_docs)} document(s) appear in BOTH splits: {sorted(shared_docs)[:5]}')

    train_ids = {r.id for r in train}
    shared_ids = train_ids & {r.id for r in eval_}
    if shared_ids:
        issues.append(f'{len(shared_ids)} record id(s) in both splits: {sorted(shared_ids)[:5]}')

    train_ctx = {_blake(r.context, 16) for r in train}
    shared_ctx = train_ctx & {_blake(r.context, 16) for r in eval_}
    if shared_ctx:
        issues.append(f'{len(shared_ctx)} identical context(s) appear in both splits')
    return issues


# ============================================================================
# Consumer-facing loaders (§2.5 data contract, now split-aware)
# ============================================================================


def processed_path(name: str, processed_dir: Optional[str] = None) -> str:
    root = processed_dir or os.environ.get('VNCOMPRESS_PROCESSED_DIR') or _repo_path(DEFAULT_PROCESSED_DIR)
    return os.path.join(root, name)


def _repo_path(*parts: str) -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, *parts)


def has_processed_split(processed_dir: Optional[str] = None) -> bool:
    return all(os.path.exists(processed_path(f'{s}.jsonl', processed_dir)) for s in ('train', 'eval'))


def load_split(
    split: str = 'train',
    kind: Optional[str] = None,
    processed_dir: Optional[str] = None,
) -> List[Record]:
    """Read `data/processed/<split>.jsonl`, optionally filtered to one kind."""
    if split not in ('train', 'eval'):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    path = processed_path(f'{split}.jsonl', processed_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found -- build the splits first:\n'
            f'  python scripts/split_dataset.py')
    records = list(read_jsonl(path))
    return [r for r in records if kind is None or r.kind == kind]


def load_manifest(processed_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = processed_path('split_manifest.json', processed_dir)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
