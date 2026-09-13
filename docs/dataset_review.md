# Báo cáo rà soát dataset `anhalu/vncompress-vi-dataset`

**Ngày:** 2026-09-09
**Người rà soát:** thien.than (hỗ trợ bởi Claude Code)
**Đối tượng:** đội xây dựng dữ liệu VNCompress-VI
**Mục tiêu:** chỉ ra các vấn đề dữ liệu chưa ổn và đề xuất cải thiện, phục vụ vòng lặp làm dữ liệu tiếp theo.

---

## 1. Phạm vi & phương pháp

- Nguồn: HuggingFace `anhalu/vncompress-vi-dataset` (đọc qua datasets-server: `/info`, `/rows`, `/statistics`).
- Đã xem: metadata 2 config, ~60 bản ghi lấy rải ở nhiều offset của `records/train` (offset 0, 35k, 65k, 95k, 122k), `records/validation`, `vcc_bench/train` (offset 25k), `vcc_bench/validation` (132 mẫu độc lập).
- Trọng tâm: chất lượng nhãn nén (`compressed_text` + metadata), chất lượng QA, phân bố task/domain, và tính dùng được của tập validation.

**Cấu trúc dataset:**

| Config | Splits | Số lượng | Số cột |
|---|---|---|---|
| `records` | train / validation | 127,189 / 14,105 | 15 |
| `vcc_bench` | train / validation / validation_synthetic | 56,759 / **132** / 5,192 | 43 |

Điểm hạ tầng làm **tốt** (cần giữ): split tất định theo document (hash), leak-check **CLEAN**, provenance/checksums đầy đủ, metadata nén giàu trường (`realized_tokens`, `extractive_ratio`, `numbers_preserved`, `budget_compliance`).

---

## 2. Tổng hợp vấn đề theo mức độ

| # | Vấn đề | Mức độ | Ảnh hưởng |
|---|---|---|---|
| P1 | Nhãn `compression_ratio` là ratio **yêu cầu**, không phải đạt được; lệch rất lớn | 🔴 Nghiêm trọng | Sai target khi train student |
| P2 | Teacher bỏ qua ngân sách token → 3 mức nén (2×/4×/8×) thường **cho ra output y hệt nhau** | 🔴 Nghiêm trọng | Phình dữ liệu bằng bản trùng; nhãn nhiễu |
| P3 | Tập validation độc lập gần như không dùng được (~6/132); multi-turn có `reference_answer` **không phụ thuộc câu hỏi** | 🔴 Nghiêm trọng | Không đo được chất lượng thật |
| P4 | Không có kiểm chứng "nén giữ được khả năng trả lời" (task-utility) | 🔴 Nghiêm trọng | Chất lượng nén chỉ đo nội tại, có thể vô nghĩa |
| P5 | "long_document_qa" thực chất là span-extraction ngữ cảnh ngắn (median ~480 ký tự) | 🟠 Cao | Sai kỳ vọng năng lực; benchmark nông |
| P6 | Mất cân bằng task cực đoan; schema 43 cột ~99% rỗng | 🟠 Cao | Không train/eval được agent/needle/multi-turn/cross-lingual |
| P7 | Hàng `context_compression` rỗng/thoái hoá (query & answer trống); thơ bị nén thành câu ngẫu nhiên | 🟠 Cao | Nhiễu nhãn |
| P8 | `domain` là loại thực thể Wikidata, không phải domain tác vụ (429 "domain" ảo) | 🟡 Trung bình | Thổi phồng độ đa dạng |
| P9 | `language` gán 100% `vi` nhưng có nội dung tiếng Anh/chèn mã | 🟡 Trung bình | Nhãn ngôn ngữ sai ở mức nội dung |
| P10 | Nội dung dạng PII (số tài khoản ngân hàng) làm "needle" | 🟡 Trung bình | Rủi ro nhầm dữ liệu thật; cần đánh dấu synthetic |
| P11 | Card ghi nguồn "legal" nhưng phân bố `source` không có; chỉ 8 chương legal | 🟡 Trung bình | Tài liệu không khớp dữ liệu |
| P12 | `reference_answer` mang ý nghĩa khác nhau theo task, ở hàng nén chỉ lặp lại context | 🟡 Trung bình | Dễ dùng sai khi chấm điểm |

---

## 3. Chi tiết & bằng chứng

### P1 — `compression_ratio` là ratio *yêu cầu*, lệch rất lớn so với thực tế 🔴

Metadata ghi ratio được **yêu cầu**, không phải đạt được. Ví dụ:

- `comp_corpus_uvw-2026_Armenia_65...` : yêu cầu 2.0 (target 133 token) → thực chỉ 15 token ⇒ **realized_ratio ≈ 17.7×**.
- `comp_cross_0003_vi_to_vi..._2` : yêu cầu 2.0 (target 61) → realized 33 ⇒ ~3.7×.
- Card tự ghi nhận: yêu cầu 2× nhưng trung bình đạt **~7.3×**.

Teacher (GLM-5.2) có xu hướng **trích một câu nổi bật nhất** bất kể ngân sách, nên `compression_ratio` gần như tách rời output.

**Khuyến nghị:**
- Dùng `realized_tokens` / `realized_ratio` làm target khi train; **không** dùng `compression_ratio` làm nhãn giám sát.
- Đổi tên trường thành `requested_compression_ratio` để tránh hiểu nhầm.
- Nếu muốn dữ liệu điều khiển được theo mức nén, phải **ép ngân sách ở khâu sinh** (re-prompt/hậu kiểm & loại mẫu không đạt) chứ không gắn nhãn sau.

### P2 — Ba mức nén thường trùng output ⇒ phình dữ liệu bằng bản lặp 🔴

Với `vietnamese-poetry_119358_807`, cả 3 target 32/16/8 token cho ra **cùng một chuỗi** `"giấc chiêm bao bảo vệ những phiêu bồng"` (8 token). Ở `vcc_bench/train` có 421 hàng nén nhưng chỉ **390 `compressed_text` duy nhất** — trùng lặp diện rộng.

**Khuyến nghị:**
- Khử trùng theo `(doc_id, compressed_text)`; nếu nhiều mức trùng nhau, giữ **một** hàng và ghi rõ đây là output ổn định ở mọi mức.
- Chỉ sinh nhiều mức khi output thực sự khác nhau; nếu không, giảm số biến thể.

### P3 — Tập validation độc lập gần như không dùng được 🔴

`vcc_bench/validation` = 132 mẫu, nhưng chỉ **~6** là QA độc lập thật (card tự nêu). Mẫu multi-turn (vd `conv_0013_q0/q1/q2`, `conv_0004_*`) có **3 câu hỏi khác nhau nhưng `reference_answer` y hệt nhau** (lặp lại toàn bộ hội thoại) ⇒ đáp án **không phụ thuộc câu hỏi**, chỉ đo việc chép lại transcript, không đo chất lượng trả lời. `validation_synthetic` (5,192) **không được gộp** với validation vì dính bias teacher.

**Khuyến nghị:**
- Xây tập test **độc lập, người gán nhãn kiểm** (≥ vài trăm mẫu) với đáp án phụ thuộc câu hỏi.
- Sửa mẫu multi-turn: mỗi câu hỏi phải có đáp án riêng, chấm bằng metric có ý nghĩa (không phải overlap transcript).
- Ghi rõ trong card: validation hiện tại chỉ dùng để sanity-check, **không** dùng để báo cáo kết quả.

### P4 — Thiếu kiểm chứng "nén giữ được khả năng trả lời" 🔴

Chất lượng nén hiện chỉ đo **nội tại** (`extractive_ratio`, `numbers_preserved`). Chưa có kiểm tra **downstream**: cho model trả lời QA **chỉ từ `compressed_text`** và so với trả lời từ context gốc. Card xác nhận chưa train student nào trên `compressed_text`.

**Khuyến nghị:**
- Thêm nhãn/metric "answerability": với mỗi (context, query), kiểm tra câu trả lời từ `compressed_text` có còn đúng `reference_answer` không. Đây là tín hiệu chất lượng nén quan trọng nhất và hiện đang thiếu.
- Ưu tiên nén **query-conditioned** cho các mẫu có query, thay vì nén query-agnostic.

### P5 — "long_document_qa" thực chất là span-extraction ngữ cảnh ngắn 🟠

Median context ~480–483 ký tự (~150 token); median độ dài đáp án 14 ký tự; đáp án trùng khớp verbatim trong context; 3 câu hỏi/đoạn, mẫu hoá. Đây là QA span kiểu SQuAD trên ngữ cảnh ngắn, **không phải** long-document.

**Khuyến nghị:**
- Đổi tên task thành `short_context_extractive_qa` (hoặc gộp context nhiều đoạn để thành long-document thật).
- Đa dạng loại câu hỏi (suy luận, tổng hợp nhiều câu, phủ định) thay vì chỉ factual/temporal verbatim.

### P6 — Mất cân bằng task & schema rỗng 🟠

- `records/train` task: `context_compression` 70,851 · `long_document_qa` 56,268 · **`multi_turn` 39 · `cross_lingual` 18 · `needle` 8 · `agent` 5**.
- `vcc_bench/train` (56,759): agent 5, legal-chapter 8, `compressed_text` chỉ 421/56,759. Các trường nâng cao (`needle`, `insert_position`, `num_turns`, `expected_tools`, `negations`, `conditions`, `entities`, `dates`, `removed_spans`, `important_spans`) **rỗng gần như hoàn toàn**.

Schema 43 cột hiện là **tham vọng**, không phản ánh dữ liệu thật.

**Khuyến nghị:**
- Hoặc bổ sung đủ mẫu cho các task hiếm (tối thiểu hàng trăm/nghìn để train hoặc vài trăm để eval), hoặc **tách chúng ra khỏi bộ chính** và ghi rõ là "mẫu minh hoạ, chưa đủ để đánh giá".
- Cắt bớt các cột luôn rỗng để schema phản ánh đúng nội dung.

### P7 — Hàng nén rỗng/thoái hoá; nén thơ vô nghĩa 🟠

- `corpus_vietnamese-poetry_20534_2078`: task = `context_compression` nhưng **query rỗng, reference_answer rỗng, không có metadata nén** — thực chất là đoạn corpus thô bị gắn nhãn task.
- Nén thơ = trích một dòng ngẫu nhiên; QA trên thơ = chép một dòng ⇒ ý nghĩa thấp. Poetry chiếm 7,188 bản ghi.

**Khuyến nghị:**
- Không gắn nhãn task cho hàng corpus thô; tách `kind=corpus` khỏi hàng có nhãn giám sát.
- Cân nhắc **loại thơ khỏi mục tiêu nén/QA** (hoặc gắn cờ riêng), giữ lại chỉ nếu dùng cho tín hiệu thanh điệu/hình thái với pipeline riêng.

### P8 — `domain` là loại thực thể Wikidata, không phải domain tác vụ 🟡

429 "domain" gồm `người`, `general`, `quốc gia có chủ quyền`, `đơn vị phân loại`, `tỉnh của Việt Nam`... — đây là `instance-of` của Wikidata, không phải lĩnh vực tác vụ (legal/medical/news/...). Độ đa dạng domain bị **thổi phồng**; thực tế legal chỉ 8 chương.

**Khuyến nghị:** ánh xạ về một bộ domain tác vụ nhỏ, có kiểm soát; giữ nhãn Wikidata ở trường phụ nếu cần.

### P9 — `language` không chính xác ở mức nội dung 🟡

Gán 100% `vi` nhưng có nội dung tiếng Anh (mẫu `en_to_en`, `reference_answer` tiếng Anh, chèn mã như "four-terminal sensing").

**Khuyến nghị:** gán `language` theo nội dung từng text (hoặc thêm `context_lang`/`answer_lang`), đặc biệt cho nhánh cross-lingual.

### P10 — Nội dung dạng PII làm "needle" 🟡

`needle_0001` (context 63,517 ký tự — cũng là doc dài nhất) có needle là **số tài khoản ngân hàng** `"1903666888666, Vietcombank chi nhánh Hà Nội"`. Dù là synthetic, nó **trông như PII thật**.

**Khuyến nghị:** đảm bảo mọi PII trong needle là giả rõ ràng (dùng số/định danh không hợp lệ), và ghi cờ `synthetic_pii=true`.

### P11 — Card không khớp phân bố nguồn 🟡

Card liệt kê "Vietnamese legal documents (Public Domain)" là nguồn, nhưng phân bố `source` chỉ có `uvw-2026`, `teacher-synth`, `vietnamese-poetry`, `wikipedia`, `vcc_bench`; legal chỉ xuất hiện dưới dạng 8 chương trong `vcc_bench`.

**Khuyến nghị:** đồng bộ card với dữ liệu thật; nếu legal quan trọng, bổ sung nguồn và gắn `source=legal` rõ ràng.

### P12 — Ngữ nghĩa `reference_answer` không nhất quán 🟡

`reference_answer` khi ở QA là đáp án; khi ở hàng nén lại **lặp lại/echo** nội dung gốc chứ không phải "gold compression". Dễ dùng sai khi chấm.

**Khuyến nghị:** tách trường theo task (`qa_answer` vs `gold_compression`), hoặc tài liệu hoá rõ ý nghĩa theo `task`.

---

## 4. Việc nên làm ngay (quick wins)

1. Đổi tên `compression_ratio` → `requested_compression_ratio`; thêm cột `realized_ratio` hiển thị sẵn (P1).
2. Khử trùng `compressed_text` theo `(doc_id, text)`, gộp các mức trùng (P2).
3. Loại/gắn cờ các hàng `context_compression` rỗng metadata (P7).
4. Sửa hoặc loại các mẫu multi-turn có đáp án không phụ thuộc câu hỏi (P3).
5. Đồng bộ dataset card với phân bố nguồn thật (P11).
6. Cắt các cột luôn rỗng khỏi schema chính (P6).

## 5. Việc mang tính cấu trúc (cần vòng làm dữ liệu mới)

1. **Xây tập test độc lập, người kiểm**, đáp án phụ thuộc câu hỏi, đủ lớn để báo cáo kết quả (P3, P4).
2. **Thêm nhãn answerability** đo chất lượng nén qua khả năng trả lời từ bản nén (P4).
3. **Ép ngân sách token ở khâu sinh** để 3 mức nén thực sự khác nhau và đạt target (P1, P2).
4. **Long-document thật** bằng cách gộp nhiều đoạn; đa dạng loại câu hỏi (P5).
5. **Bổ sung hoặc tách** các task hiếm (agent/needle/multi-turn/cross-lingual) (P6).

---

## 6. Kết luận

Hạ tầng dữ liệu (split, provenance, metadata) làm bài bản và đáng tin. Tuy nhiên, **nhánh dùng để train nén** có nhãn ratio không đáng tin và nhiều bản trùng, còn **nhánh dùng để đánh giá** gần như chưa dùng được (tập độc lập quá nhỏ + đáp án không phụ thuộc câu hỏi + thiếu kiểm chứng answerability). Ưu tiên cao nhất trước khi train/paper: **(a) sửa nhãn/khử trùng dữ liệu nén, (b) dựng tập test độc lập có kiểm, (c) thêm metric answerability**.
