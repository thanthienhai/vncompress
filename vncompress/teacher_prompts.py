"""Versioned teacher prompts (docs/dataset_pipeline.md §4.2, §4.3, §14).

§14 requires prompts to be versioned and stored with every generated row, so
`PROMPT_VERSION` is bumped whenever a template below changes in a way that
would alter output. It is part of the cache key, so a bump invalidates exactly
the affected stage and nothing else.

Two stages, matching §15's phases:

- ``queries``     (Phase 1) -- a corpus paragraph carries no question, so
  query-conditioned supervision has to start by constructing one. This is the
  "query-conditioned sample generator" §15 Phase 1 asks for.
- ``compression`` (Phase 2) -- the §4.2 task list: decide what the query needs,
  compress to a budget, and mark the entity/number/date/negation/condition
  spans that must survive.

The instructions are written in Vietnamese on purpose: the material is
Vietnamese, the §4.3 constraints are specified in Vietnamese in the doc, and
keeping prompt and spec in one language makes drift between them visible.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

PROMPT_VERSION = 'v1'

STAGE_QUERIES = 'queries'
STAGE_COMPRESSION = 'compression'

# Marker the dry-run client uses to tell the stages apart. Also makes the stage
# explicit to the model, which measurably helps small instruction-tuned models.
_STAGE_TAG = '[[vncompress-stage:{stage}|{version}]]'

_JSON_ONLY = (
    'Chỉ trả về MỘT object JSON hợp lệ, không kèm giải thích, không bọc trong khối mã.'
)

QUERY_SYSTEM = f"""Bạn là chuyên gia xây dựng dữ liệu huấn luyện cho bài toán nén ngữ cảnh tiếng Việt.

Nhiệm vụ: đọc một đoạn văn tiếng Việt và sinh ra các câu hỏi mà đoạn văn đó TRẢ LỜI ĐƯỢC.

Ràng buộc bắt buộc:
- Câu hỏi phải trả lời được HOÀN TOÀN bằng thông tin có trong đoạn văn. Không suy diễn.
- `answer_span` phải là một đoạn trích NGUYÊN VĂN, sao chép chính xác từ đoạn văn.
- `answer` là câu trả lời ngắn gọn, tự nhiên (có thể khác `answer_span` về cách diễn đạt).
- Ưu tiên câu hỏi mà đáp án là thực thể, con số, ngày tháng, điều kiện hoặc phủ định —
  đây là những thông tin dễ mất nhất khi nén.
- Tránh câu hỏi chung chung kiểu "đoạn văn nói về gì" — loại này khiến đáp án trùng
  với cả đoạn văn và làm hỏng phép đo.
- Nếu đoạn văn quá nghèo thông tin để đặt câu hỏi có đáp án rõ ràng, trả về danh sách rỗng.

{_JSON_ONLY}

Định dạng:
{{"queries": [{{"query": "...", "answer": "...", "answer_span": "...", "type": "factual|numeric|temporal|conditional|negation|multi_hop"}}]}}"""

COMPRESSION_SYSTEM = f"""Bạn là bộ nén ngữ cảnh tiếng Việt có điều kiện theo câu hỏi (query-conditioned compressor).

Nhiệm vụ: rút gọn NGỮ CẢNH sao cho một mô hình ngôn ngữ khác vẫn trả lời đúng CÂU HỎI.

Ràng buộc bắt buộc (§4.3):
- Nén phải phục vụ CÂU HỎI, không phải tóm tắt chung chung.
- KHÔNG được thay đổi sự thật. KHÔNG được thêm thông tin không có trong ngữ cảnh.
- Bắt buộc giữ nguyên văn: tên riêng, con số, đơn vị, tỷ lệ phần trăm, ngày tháng,
  từ phủ định ("không", "chưa", "không được") và từ điều kiện ("nếu", "trừ khi",
  "trong trường hợp") nào có ảnh hưởng tới câu trả lời.
- `compressed_text` phải là văn bản TRÍCH XUẤT: chỉ gồm các câu/cụm lấy từ ngữ cảnh gốc,
  giữ nguyên thứ tự xuất hiện. Không diễn đạt lại.
- Bám sát ngân sách độ dài được giao. Vượt quá là lỗi.
- Nếu ngữ cảnh không đủ để trả lời, hãy giữ lại phần bằng chứng liên quan nhất
  thay vì cố bịa ra câu trả lời.

{_JSON_ONLY}

Định dạng:
{{"compressed_text": "...",
  "important_spans": ["trích nguyên văn ..."],
  "removed_spans": ["trích nguyên văn ..."],
  "entities": ["..."], "numbers": ["..."], "dates": ["..."],
  "conditions": ["..."], "negations": ["..."],
  "compression_reason": "một câu giải thích vì sao giữ những phần này"}}"""


def count_words(text: str) -> int:
    """Whitespace-unit length.

    The budget is expressed in whitespace units, not model tokens: the teacher
    has no access to the student's tokenizer, and a Vietnamese syllable count is
    stable across the PhoBERT / Qwen / GPT-2 tokenizers this repo mixes. The
    mapping to real model tokens is measured separately by
    scripts/measure_token_inflation.py.
    """
    return len(re.findall(r'\S+', text or ''))


def build_query_messages(context: str, n_queries: int = 3) -> List[Dict[str, str]]:
    tag = _STAGE_TAG.format(stage=STAGE_QUERIES, version=PROMPT_VERSION)
    user = (f'{tag}\n'
            f'Sinh tối đa {n_queries} câu hỏi cho đoạn văn dưới đây.\n\n'
            f'=== ĐOẠN VĂN ===\n{context}\n=== HẾT ===')
    return [{'role': 'system', 'content': QUERY_SYSTEM}, {'role': 'user', 'content': user}]


def build_compression_messages(context: str, query: str, ratio: float) -> List[Dict[str, str]]:
    budget = max(1, round(count_words(context) / ratio))
    tag = _STAGE_TAG.format(stage=STAGE_COMPRESSION, version=PROMPT_VERSION)
    user = (f'{tag}\n'
            f'Ngân sách: tối đa {budget} âm tiết/từ (ngữ cảnh gốc có {count_words(context)}), '
            f'tức tỷ lệ nén khoảng {ratio:g}x.\n\n'
            f'=== CÂU HỎI ===\n{query}\n\n'
            f'=== NGỮ CẢNH ===\n{context}\n=== HẾT ===')
    return [{'role': 'system', 'content': COMPRESSION_SYSTEM}, {'role': 'user', 'content': user}]


def target_tokens(context: str, ratio: float) -> int:
    return max(1, round(count_words(context) / ratio))


# ============================================================================
# Offline stand-in
# ============================================================================


def _stage_of(messages: List[Dict[str, str]]) -> str:
    joined = '\n'.join(m.get('content', '') for m in messages)
    match = re.search(r'\[\[vncompress-stage:([a-z_]+)\|', joined)
    return match.group(1) if match else STAGE_COMPRESSION


def _context_of(messages: List[Dict[str, str]]) -> str:
    joined = '\n'.join(m.get('content', '') for m in messages if m.get('role') == 'user')
    match = re.search(r'=== (?:ĐOẠN VĂN|NGỮ CẢNH) ===\n(.*?)\n=== HẾT ===', joined, re.S)
    return match.group(1) if match else ''


def _budget_of(messages: List[Dict[str, str]]) -> int:
    joined = '\n'.join(m.get('content', '') for m in messages if m.get('role') == 'user')
    match = re.search(r'tối đa (\d+) âm tiết', joined)
    return int(match.group(1)) if match else 64


def dry_run_response(messages: List[Dict[str, str]]) -> str:
    """A structurally valid response with no model behind it.

    Enough to prove the plumbing works end to end -- parsing, verification,
    merge, split -- and nothing more. The "compression" is a lead-sentence cut,
    which is a deliberately weak baseline, not a teacher.
    """
    context = _context_of(messages)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if s.strip()]

    if _stage_of(messages) == STAGE_QUERIES:
        payload: Dict[str, Any] = {'queries': []}
        for sentence in sentences[:2]:
            words = re.findall(r'\S+', sentence)
            if len(words) < 6:
                continue
            span = ' '.join(words[:5])
            payload['queries'].append({
                'query': f'Theo đoạn văn, "{span}" được nói tới trong ngữ cảnh nào?',
                'answer': span,
                'answer_span': span,
                'type': 'factual',
            })
        return json.dumps(payload, ensure_ascii=False)

    budget = _budget_of(messages)
    kept: List[str] = []
    used = 0
    for sentence in sentences:
        n = count_words(sentence)
        if used + n > budget and kept:
            break
        kept.append(sentence)
        used += n
    if not kept and sentences:
        kept = [' '.join(re.findall(r'\S+', sentences[0])[:budget])]

    compressed = ' '.join(kept)
    return json.dumps({
        'compressed_text': compressed,
        'important_spans': kept[:3],
        'removed_spans': sentences[len(kept):][:3],
        'entities': [], 'numbers': re.findall(r'\d[\d.,]*', compressed)[:5],
        'dates': [], 'conditions': [], 'negations': [
            w for w in ('không', 'chưa', 'không được') if w in compressed.lower()],
        'compression_reason': 'dry-run: giữ các câu đầu cho tới khi hết ngân sách',
    }, ensure_ascii=False)
