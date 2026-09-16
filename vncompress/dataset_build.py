"""Pure transforms for building VNCompress-VI v2 from the v1 dataset.

Kept out of the build script so each rule can be tested on its own: a dataset
builder that is only exercised end-to-end is a builder whose individual rules
are never checked.

See docs/dataset_rebuild_spec.md and docs/dataset_v1_baseline.md.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Optional

from .dataset_schema import BUDGET_TOLERANCE, LONG_DOCUMENT_MIN_TOKENS
from .linguistics import is_vietnamese

# ============================================================================
# Domain (P8 / S1)
# ============================================================================
#
# v1's `domain` is the Wikidata instance-of type of the article's SUBJECT --
# 490 values including "người" (human) and "đơn vị phân loại" (taxon). That
# describes the entity, not the task domain, which is why v1 appeared to span
# hundreds of domains while actually being general-purpose Wikipedia text.
#
# These rules match on substrings of the lowercased Wikidata label, most
# specific first. A large share landing in `other` is the CORRECT outcome, not
# a gap in the rules: general encyclopedia content has no task domain, and
# inventing one would reproduce exactly the inflated diversity P8 objects to.
# The original label is always preserved in `wikidata_type`.

_DOMAIN_RULES = (
    ('legal', (
        'luật', 'hiến pháp', 'nghị định', 'thông tư', 'bộ luật', 'hiệp ước', 'hiệp định',
        'điều ước', 'tòa án', 'toà án', 'treaty', 'pact', 'trial', 'court', 'legal',
        'law', 'tội phạm', 'war crimes',
    )),
    ('medical', (
        'bệnh', 'y học', 'y tế', 'thuốc', 'vắc', 'virus', 'vi rút', 'dịch bệnh', 'giải phẫu',
        'disease', 'medical', 'medicine', 'health', 'syndrome', 'pandemic', 'protein',
    )),
    ('history', (
        'lịch sử', 'history', 'historical', 'triều đại', 'vương triều', 'chiến tranh', 'chiến dịch',
        'trận đánh', 'battle', 'war', 'nổi loạn', 'khởi nghĩa', 'cách mạng', 'revolution',
        'cựu quốc gia', 'thời pháp thuộc', 'đế quốc', 'empire', 'dynasty', 'thời kỳ',
    )),
    ('science', (
        'nguyên tố', 'hóa học', 'hoá học', 'toán học', 'vật lý', 'sinh học', 'thiên văn',
        'đơn vị phân loại', 'taxon', 'khoa học', 'science', 'scientific', 'chemical',
        'mathemat', 'physics', 'biolog', 'astronom', 'ngôn ngữ lập trình', 'ngôn ngữ hướng đối tượng',
        'phần mềm', 'thuật toán', 'algorithm', 'software', 'công nghệ', 'technolog',
    )),
    ('education', (
        'môn học', 'trường', 'đại học', 'trung học', 'tiểu học', 'giáo dục', 'học viện',
        'school', 'universit', 'educat', 'academic', 'nhánh khoa học', 'academic discipline',
    )),
    ('admin', (
        'tỉnh', 'huyện', 'quận', 'thị xã', 'thành phố', 'phường', 'xã của', 'đơn vị hành chính',
        'quốc gia', 'thủ đô', 'vùng', 'lãnh thổ', 'chính phủ', 'bộ trưởng', 'công chức',
        'tổ chức liên chính phủ', 'municipal', 'province', 'district', 'city', 'country',
        'state', 'government', 'administrative',
    )),
    ('news', (
        'sự kiện', 'thời sự', 'báo chí', 'news', 'journalis', 'current event',
    )),
)


def map_domain(wikidata_type: Optional[str]) -> str:
    """A v1 Wikidata entity type -> one of the controlled task domains."""
    if not wikidata_type:
        return 'other'
    label = unicodedata.normalize('NFC', str(wikidata_type)).lower()
    for domain, needles in _DOMAIN_RULES:
        if any(n in label for n in needles):
            return domain
    return 'other'


# ============================================================================
# Language (P9 / S2)
# ============================================================================


# Below this many alphabetic characters the diacritic-ratio heuristic cannot
# be trusted; see detect_lang for the measurement that picked the number.
LANG_MIN_ALPHA_CHARS = 20


def detect_lang(text: Optional[str], fallback: Optional[str] = None) -> Optional[str]:
    """Language of THIS text, not of the document it came from.

    v1 stamped `language='vi'` on every row, including the English side of
    cross-lingual pairs. Returns None for empty text rather than guessing, so a
    missing label stays visibly missing.

    THREE TIERS, because one rule cannot cover a 4,000-word context and a
    three-letter answer span. `is_vietnamese` fires on the RATIO of
    Vietnamese-specific characters, so it needs enough characters to see one:

      no letters at all -> None. "371" and "1827" are not English; they are not
        any language. v2 shipped 144 such answers labelled `en`.
      under LANG_MIN_ALPHA_CHARS -> `fallback`, normally the language of the
        text the span was cut from. Measured on 1,002 answers that are
        unambiguously Vietnamese prose, truncating them to a 3-character
        budget leaves the detector recognising only 77.8%; at 20 characters it
        is 99.4%. Below that floor the detector is guessing, and a guess
        inherited from the enclosing context is right far more often.
      otherwise -> the heuristic, which is reliable at this length.

    Without the floor, 983 of 6,000 v2 answers (17%) were labelled `en` while
    sitting inside a Vietnamese context -- "gan", "Amsterdam", "Louis XIV" --
    which is P9/S2 read backwards: the label described the absence of diacritics
    rather than the language of the text.
    """
    if not text or not text.strip():
        return None
    alpha_chars = sum(1 for c in text[:2000] if c.isalpha())
    if alpha_chars == 0:
        return None
    if alpha_chars < LANG_MIN_ALPHA_CHARS:
        return fallback
    return 'vi' if is_vietnamese(text[:2000]) else 'en'


# ============================================================================
# Compression metrics (P1 / B1)
# ============================================================================

# `\d[\d.,]*` would swallow the punctuation that follows a number, so "1925,"
# in the context never matches "1925" in the compression and the number reads
# as lost. Measured on the 973 shipped v2 rows, the greedy form disagrees with
# this one on 500 of them -- always by UNDER-reporting -- and drags median
# `numbers_preserved` from 0.228 down to 0.191. A separator only counts when
# digits continue after it, which is what makes "3.14" one number and "1925,"
# one number plus a comma.
_NUMBER_RE = re.compile(r'\d+(?:[.,]\d+)*')

# Sentence boundaries, shared by the extractive checks below. Vietnamese uses
# the same terminal punctuation as English, and a blank line ends a sentence
# whether or not it is punctuated.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')

# Fragments shorter than this are not evidence either way: "Năm 1945." appears
# verbatim in almost any Vietnamese history article, so counting it as
# extracted would inflate the score with coincidences.
_MIN_SENTENCE_CHARS = 15


def count_tokens(text: Optional[str]) -> int:
    """Whitespace tokens -- the unit v1's `token_unit` field declares."""
    return len(text.split()) if text else 0


def realized_ratio(context: Optional[str], gold_compression: Optional[str]) -> Optional[float]:
    """Compression actually achieved, measured from the two texts themselves.

    B1: never a stored or requested number. v1's own `realized_ratio` was
    derived from a teacher-reported token count; recomputing from the text is
    the only version that cannot drift from the data it describes.
    """
    original, compressed = count_tokens(context), count_tokens(gold_compression)
    return original / compressed if original and compressed else None


def numbers_preserved(context: Optional[str], gold_compression: Optional[str]) -> Optional[float]:
    """Fraction of the numbers in the context that survive compression."""
    source = _NUMBER_RE.findall(context or '')
    if not source:
        return None
    kept = set(_NUMBER_RE.findall(gold_compression or ''))
    return sum(1 for n in source if n in kept) / len(source)


def extractive_ratio(context: Optional[str], gold_compression: Optional[str]) -> Optional[float]:
    """Fraction of compressed tokens that appear verbatim in the context."""
    compressed = (gold_compression or '').split()
    if not compressed:
        return None
    source = set((context or '').split())
    return sum(1 for t in compressed if t in source) / len(compressed)


def sentences(text: Optional[str], min_chars: int = _MIN_SENTENCE_CHARS) -> list:
    """Sentences worth scoring -- see `_MIN_SENTENCE_CHARS` for the floor."""
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text)
            if len(s.strip()) >= min_chars]


def sentence_extractive_ratio(context: Optional[str],
                              gold_compression: Optional[str]) -> Optional[float]:
    """Fraction of the compression's SENTENCES that appear verbatim in the context.

    `extractive_ratio` counts tokens against a bag of context words, so a
    fluent rewrite that reuses the vocabulary scores near 1.0 while sharing no
    sentence with the source. On the 973 shipped v2 rows it reads a reassuring
    median 0.933 -- and only 30.0% of the compressions' sentences are actually
    present in the context, with 61 of 973 compressions being a substring of
    their context.

    That gap is the difference between a dataset that can supervise an
    EXTRACTIVE compressor and one that cannot. Every wave-2 arm selects from
    the context -- token perplexity (E1), sentences (E5), class budgets (E7),
    encoder keep/drop labels (E6) -- and none of them can emit a sentence the
    context does not contain. A gold that is 93% familiar words but 30%
    quotable is a summarisation target wearing a compression label.
    """
    gold_sentences = sentences(gold_compression)
    if not gold_sentences:
        return None
    haystack = context or ''
    return sum(1 for s in gold_sentences if s in haystack) / len(gold_sentences)


def unsupported_numbers(context: Optional[str],
                        gold_compression: Optional[str]) -> Optional[list]:
    """Numbers asserted by the compression that its context never states.

    The inverse of `numbers_preserved`, and the one that catches fabrication
    rather than loss. A compression can only ever LOSE information; a number
    that appears in the output and not the input came from the teacher's own
    parametric knowledge, not from the text it was asked to compress.

    Real and frequent: 206 of the 973 shipped v2 rows (21.2%) assert at least
    one such number. In `viquad:Pham_Van_Dong` at 4x the compression dates the
    Phan Chau Trinh mourning movement to 1925 and the 8th Party Congress to
    1996 from a 4,712-character context that contains neither -- true facts,
    invented supervision. Truncation is not the explanation: only 16 of those
    206 rows sit at the 24,000-character context cap.

    Returns the sorted offenders so a reviewer can judge them, not just a
    count; an empty list means clean, None means the compression has no
    numbers to check.
    """
    if not gold_compression:
        return None
    produced = set(_NUMBER_RE.findall(gold_compression))
    if not produced:
        return None
    supported = set(_NUMBER_RE.findall(context or ''))
    return sorted(produced - supported)


def budget_compliance(realized_tokens: Optional[int], target_tokens: Optional[int],
                      tolerance: float = BUDGET_TOLERANCE) -> Optional[bool]:
    """Two-sided budget check, per docs/dataset_rebuild_spec.md SS2.2.

    Deliberately NOT v1's rule. v1 stored `realized <= target`, which only
    penalises producing too MANY tokens -- verified to agree with its stored
    flag on 100.0% of 57,529 rows. Because the teacher over-compresses
    systematically, that one-sided rule reported 88.6% compliance where the
    two-sided rule gives 19.8%. Inheriting the flag would inherit a 4.5x
    overstatement (see docs/dataset_v1_baseline.md).
    """
    if not target_tokens or realized_tokens is None:
        return None
    return abs(realized_tokens - target_tokens) / target_tokens <= tolerance


# ============================================================================
# QA (P5)
# ============================================================================


def is_long_document(context: Optional[str]) -> bool:
    """True only above the spec's threshold.

    v1 called a 480-character span-extraction task "long_document_qa". The flag
    has to be a measurement, not a name.
    """
    return count_tokens(context) >= LONG_DOCUMENT_MIN_TOKENS


def qa_task_name(context: Optional[str]) -> str:
    return 'long_document_qa' if is_long_document(context) else 'short_context_extractive_qa'


# ============================================================================
# Synthetic identifiers (P10 / T4)
# ============================================================================
#
# A needle that reads like a real bank account is a hazard whether or not it
# was generated. These rewrite such payloads into forms that cannot be valid.

_VN_PHONE_RE = re.compile(r'\b0\d{9,10}\b')
_LONG_DIGITS_RE = re.compile(r'\b\d{9,}\b')


def sanitize_identifiers(text: Optional[str]) -> tuple:
    """Rewrite real-looking identifiers into obviously invalid ones.

    Returns (text, changed). Phone numbers become the reserved 0000-prefixed
    form and long digit runs (account numbers) are prefixed with a marker, so
    a reader can tell at a glance that nothing here is a real identifier.
    """
    if not text:
        return text, False
    original = text
    text = _VN_PHONE_RE.sub(lambda m: '0000' + '0' * (len(m.group()) - 4), text)
    text = _LONG_DIGITS_RE.sub(lambda m: '0000000000' + m.group()[10:], text)
    return text, text != original


# ============================================================================
# Deterministic document-level split
# ============================================================================


def assign_split(doc_key: str, seed: int = 42,
                 ratios: tuple = (0.80, 0.10, 0.10)) -> str:
    """train / validation / test, as a pure function of the document key.

    Re-derived, never inherited -- including from the source corpora. UIT-ViQuAD
    2.0's own splits share 17 article titles and 449 contexts between train and
    test, so adopting them would put the same documents on both sides of the
    evaluation before a single row was written. Hashing the document key makes
    document-level separation true by construction and reproducible from the
    seed alone.
    """
    digest = hashlib.sha256(f"{seed}:{doc_key}".encode('utf-8')).digest()
    position = int.from_bytes(digest[:8], 'big') / 2 ** 64
    train, validation, _ = ratios
    if position < train:
        return 'train'
    if position < train + validation:
        return 'validation'
    return 'test'


def build_long_context(contexts, focus_index: int, answer_start: Optional[int],
                       max_chars: int = 24000):
    """Grow a real long document around the passage holding the answer.

    P5: v1 called a 480-character span-extraction task "long_document_qa".
    Real long-document QA needs a real long document, so neighbouring passages
    of the same article are joined around the answer-bearing one until the
    budget is full. The answer offset is recomputed for the joined text --
    carrying the original offset over would point into the wrong passage.

    Returns (text, answer_start_in_text, passages_used).
    """
    if not contexts:
        return '', None, 0
    focus_index = max(0, min(focus_index, len(contexts) - 1))
    chosen = [focus_index]
    total = len(contexts[focus_index])
    low, high = focus_index - 1, focus_index + 1
    while total < max_chars and (low >= 0 or high < len(contexts)):
        # Alternate outwards so the answer does not always land at one edge;
        # position within the document is itself a variable worth varying.
        if high < len(contexts) and (low < 0 or (high - focus_index) <= (focus_index - low)):
            nxt, high = high, high + 1
        else:
            nxt, low = low, low - 1
        if total + len(contexts[nxt]) + 2 > max_chars:
            break
        chosen.append(nxt)
        total += len(contexts[nxt]) + 2
    chosen.sort()
    parts = [contexts[i] for i in chosen]
    text = "\n\n".join(parts)
    new_start = None
    if answer_start is not None:
        offset = sum(len(contexts[i]) + 2 for i in chosen if i < focus_index)
        new_start = offset + answer_start
    return text, new_start, len(chosen)


def sentence_containing(text: str, char_start: int, char_end: int) -> str:
    """The sentence(s) spanning [char_start, char_end) -- an extractive
    gold-compression candidate that provably contains the answer."""
    if not text or char_start is None:
        return ''
    boundaries = [0] + [m.end() for m in re.finditer(r'[.!?]\s+|\n+', text)] + [len(text)]
    start = max((b for b in boundaries if b <= char_start), default=0)
    end = min((b for b in boundaries if b >= char_end), default=len(text))
    return text[start:end].strip()


# ============================================================================
# Cross-source document identity (E1)
# ============================================================================


def document_key(doc_id: Optional[str]) -> str:
    """The identity of a DOCUMENT, independent of which source delivered it.

    Both source corpora are Vietnamese Wikipedia, so the same article arrives
    twice under two ids -- `uvw:Đế_quốc_La_Mã` and `viquad:Đế_quốc_La_Mã`.
    Splitting on the prefixed id gives the two copies independent coin flips,
    which put 3 of 18 test articles (168 of 1000 test rows) into the training
    corpus while the leak-check compared prefixed ids and reported CLEAN.

    Stripping the prefix makes that impossible by construction rather than
    detectable afterwards: one article, one key, one split.
    """
    if not doc_id:
        return ''
    _, separator, rest = str(doc_id).partition(':')
    body = rest if separator else str(doc_id)
    return unicodedata.normalize('NFC', body).strip().lower().replace(' ', '_')


def normalize_compression(text: Optional[str]) -> str:
    """Whitespace-insensitive key for B2 duplicate detection."""
    return ' '.join((text or '').split())
