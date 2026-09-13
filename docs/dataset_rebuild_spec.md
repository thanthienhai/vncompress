# Yêu cầu xây dựng lại dataset VNCompress-VI (v2)

**Ngày:** 2026-09-10 · **Cơ sở:** `docs/dataset_review.md` (P1–P12) · **Đối tượng:** đội xây dựng dữ liệu
**Nguyên tắc chung:** giữ nguyên phần hạ tầng đang tốt (split tất định theo document, leak-check, provenance/checksums). Chỉ sửa nội dung & nhãn theo yêu cầu dưới đây. Mỗi yêu cầu ghi rõ vấn đề nó giải quyết.

---

## 1. Nguyên tắc bắt buộc (không được vi phạm)

- **B1.** Không dùng ratio *yêu cầu* làm nhãn. Mọi target/metric nén phải tính từ output thật. *(P1)*
- **B2.** Không có hàng trùng `compressed_text` trong cùng `doc_id`. *(P2)*
- **B3.** Không gắn nhãn task cho hàng corpus thô; `kind=corpus` và hàng giám sát phải tách bạch. *(P7)*
- **B4.** Mọi trường trong schema phải được điền hoặc bị loại; cấm giữ cột luôn rỗng. *(P6)*
- **B5.** Tập test độc lập **không** dùng output teacher; phải có người kiểm. *(P3, P4)*

---

## 2. Schema chuẩn v2

Tách làm **3 file/config riêng biệt**, không nhồi vào một schema rộng:

### 2.1 `corpus` — văn bản thô (không nhãn)
`id, doc_id, source, source_url, license, text, char_length, token_count, text_lang, split`

### 2.2 `compression` — cặp nén giám sát *(sửa P1, P2, P12)*
| Trường | Ý nghĩa |
|---|---|
| `id, doc_id, source, domain, split` | định danh & phân loại |
| `context, context_lang` | ngữ cảnh gốc + ngôn ngữ nội dung *(P9)* |
| `query, query_lang` | câu hỏi/chỉ dẫn (rỗng nếu nén query-agnostic) |
| `gold_compression` | bản nén (đây là nhãn nén, KHÔNG dùng `reference_answer`) *(P12)* |
| `requested_ratio` | ratio **yêu cầu** — chỉ để tra cứu, KHÔNG phải target *(P1)* |
| `target_tokens, realized_tokens, realized_ratio` | ngân sách & kết quả thật *(P1)* |
| `budget_compliance` | `abs(realized-target)/target ≤ 0.15` |
| `extractive_ratio, numbers_preserved` | metric nội tại |
| `answerable_from_compression` | **bắt buộc** với mẫu có query: bản nén còn trả lời đúng không *(P4)* |
| `teacher, gen_config` | model & tham số sinh |

### 2.3 `qa` — hỏi đáp *(sửa P5)*
`id, doc_id, source, domain, split, context, context_lang, query, question_type, answer, answer_span, answer_lang, is_long_document`

> Đổi tên task cũ `long_document_qa`. Đặt `is_long_document=true` chỉ khi `context ≥ 4000` token; còn lại là `short_context_extractive_qa`. *(P5)*

---

## 3. Quy trình sinh dữ liệu nén *(P1, P2, P4)*

1. **Ép ngân sách khi sinh:** prompt teacher kèm `target_tokens` cứng. Sau khi sinh, đo `realized_tokens`; nếu `budget_compliance=false` → **re-prompt tối đa 2 lần**, vẫn trượt thì **loại mẫu** (không giữ để gắn nhãn sau).
2. **Đa mức chỉ khi khác nhau:** sinh các mức {2×, 4×, 8×}; nếu 2 mức cho `gold_compression` trùng nhau → **gộp còn 1 hàng**, ghi `stable_across_ratios=[...]`. *(P2)*
3. **Answerability:** với mẫu có `query`, dùng một model trả lời chỉ từ `gold_compression`, so với `answer` gốc → `answerable_from_compression ∈ {true,false}`. **Chỉ giữ mẫu `true`** vào tập train nén (mẫu `false` để riêng phục vụ phân tích). *(P4)*
4. **Ưu tiên query-conditioned:** mẫu có query phải nén theo query, không nén chung chung. *(P4)*

---

## 4. Task hiếm & nội dung đặc thù

- **T1.** Các task `agent / needle / multi_turn / cross_lingual`: hoặc nâng lên **≥ 300 mẫu/task** (đủ để eval), hoặc **tách ra `extras/` và ghi rõ "minh hoạ, chưa đủ đánh giá"**. Không trộn vào bộ chính. *(P6)*
- **T2.** Multi-turn: mỗi câu hỏi phải có **đáp án riêng phụ thuộc câu hỏi**; cấm dùng transcript làm đáp án chung. *(P3)*
- **T3.** Thơ: **loại khỏi** `compression`/`qa`. Nếu cần cho tín hiệu thanh điệu/hình thái → để ở config `poetry` riêng, không nhãn nén/QA. *(P7)*
- **T4.** Needle & mọi PII: dùng định danh **giả rõ ràng, không hợp lệ**; gắn cờ `synthetic_pii=true`. *(P10)*

---

## 5. Nguồn, domain, ngôn ngữ *(P8, P9, P11)*

- **S1.** `domain` dùng **bộ có kiểm soát** (news, legal, medical, education, science, history, admin, other). Nhãn Wikidata (nếu giữ) để ở `wikidata_type`. *(P8)*
- **S2.** `*_lang` gán theo **nội dung từng text**, không gán ở mức document. *(P9)*
- **S3.** Đồng bộ dataset card với phân bố `source` thật; nếu có legal thì phải xuất hiện trong `source` với khối lượng khai báo đúng. *(P11)*

---

## 6. Tập test độc lập *(P3, P4)*

- **E1.** ≥ 300 mẫu, người kiểm, tách riêng theo document khỏi train (leak-check CLEAN).
- **E2.** Mỗi mẫu: `context, query, gold_answer` (đáp án phụ thuộc câu hỏi) + `gold_compression` do người xác nhận là đủ để trả lời.
- **E3.** Không chứa output teacher. `validation_synthetic` (nếu giữ) phải để **file riêng**, có nhãn cảnh báo không được gộp.
- **E4.** Card ghi rõ: tập nào dùng để **báo cáo kết quả**, tập nào chỉ sanity-check.

---

## 7. Tiêu chí nghiệm thu (checklist đo được)

Bản v2 chỉ được chấp nhận khi **tất cả** đúng:

- [ ] `compression`: 100% hàng có `realized_tokens` & `realized_ratio`; **≥ 95%** `budget_compliance=true`. *(P1)*
- [ ] `compression`: 0 hàng trùng `(doc_id, gold_compression)`. *(P2)*
- [ ] `compression`: 100% mẫu có query đã tính `answerable_from_compression`; tập train nén chỉ gồm `true`. *(P4)*
- [ ] 0 hàng corpus thô mang nhãn task; 0 hàng nén có `gold_compression` rỗng. *(P7, B3)*
- [ ] Schema không còn cột rỗng > 50%. *(P6, B4)*
- [ ] `qa`: gán đúng `is_long_document`; multi-turn có đáp án riêng theo câu hỏi. *(P5, P3)*
- [ ] `domain` chỉ nhận giá trị trong bộ kiểm soát; `*_lang` khớp nội dung (kiểm mẫu ≥ 100). *(P8, P9)*
- [ ] Tập test độc lập ≥ 300 mẫu, người kiểm, leak-check CLEAN, không có teacher output. *(E1–E3)*
- [ ] Mọi PII là giả + `synthetic_pii=true`. *(P10)*
- [ ] Dataset card khớp phân bố `source` thật & nêu rõ tập báo cáo vs sanity-check. *(P11, E4)*

---

## 8. Bàn giao

1. Dataset v2 (3 config: `corpus`, `compression`, `qa` + `extras/` nếu có).
2. `provenance/`: `split_manifest.json`, `verification_report.json`, `CHECKSUMS.json`, leak-check.
3. `report_v2.md`: phân bố task/domain/source/lang, tỷ lệ `budget_compliance`, tỷ lệ `answerable_from_compression`, số hàng bị loại theo từng bước lọc.
4. Dataset card cập nhật.
