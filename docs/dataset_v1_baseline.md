# Baseline đo được của dataset v1 theo checklist v2

**Ngày đo:** 2026-09-14 · **Công cụ:** `python scripts/validate_dataset_v2.py --legacy-v1`
**Đối tượng:** `anhalu/vncompress-vi-dataset` (snapshot `ff8ff9dd`, đọc từ HF cache local)
**Mục đích:** biến "v1 có vấn đề" (`docs/dataset_review.md`) thành **một con số cho mỗi vấn đề**, làm mốc nghiệm thu cho bản rebuild.

Chạy lại lệnh trên bất cứ lúc nào để so tiến độ. Mỗi dòng FAIL là một việc cụ thể trong `docs/dataset_rebuild_spec.md`.

## Kết quả: 4 pass · 17 FAIL · 3 manual

Dữ liệu nạp được: `compression=57.950` · `qa=123.670` · `corpus=22.178` · `eval=132`.

> **Đọc trước:** cặp nén của v1 **không** nằm ở 496 hàng có cột `compressed_text` trong config `vcc_bench`. Chúng nằm trong chuỗi JSON `records.metadata` — **57.950 hàng**, và blob đó cũng đã chứa sẵn `realized_ratio`, `budget_compliance`, `extractive_ratio`, `numbers_preserved`. Chỉ đọc cột phẳng thì chấm v1 trên 1% dữ liệu nén của nó và báo thiếu những trường vốn có.

| Ref | Hạng mục | Kết quả |
|---|---|---|
| §2 | `corpus` đủ trường | FAIL — thiếu `source_url`, `license`, `token_count`, `text_lang` |
| §2 | `compression` đủ trường | FAIL — 18/20; thiếu `context_lang`, `query_lang` |
| §2 | `qa` đủ trường | FAIL — 10/13; thiếu `context_lang`, `answer_lang`, `is_long_document` |
| P1 | có `realized_tokens` | **PASS** — 57.950/57.950 |
| P1 | `budget_compliance` ≥ 95% | FAIL — **20,1%** |
| P2 | 0 trùng `(doc_id, gold_compression)` | FAIL — **19.367** hàng trùng; 38.562 text duy nhất / 57.950 hàng |
| P4 | `answerable_from_compression` | FAIL — **0/57.950** |
| P7 | corpus không mang nhãn task | FAIL — **22.178/22.178** |
| B3 | `gold_compression` không rỗng | **PASS** — 0 rỗng |
| P6 | không cột nào rỗng > 50% | FAIL — trên schema gốc `vcc_bench`: **29/43 cột**, 12 cột rỗng **100%** |
| P5 | `is_long_document` đúng | FAIL — trường không tồn tại; median context **482 ký tự** ⇒ không phải long-document |
| P3 | multi-turn có đáp án riêng | FAIL — **16/16** hội thoại dùng chung một đáp án |
| P8 | `domain` trong bộ kiểm soát | FAIL — **465** giá trị, 463 ngoài bộ (`general`, `người`, `đơn vị phân loại`…) |
| P9 | `*_lang` khớp nội dung | FAIL — **117/1.000** mẫu kiểm lệch nhãn |
| P10 | needle gắn `synthetic_pii` | FAIL — 9/9 chưa gắn cờ |
| E1 | test độc lập ≥ 300 | FAIL — **132** mẫu |
| E3 | test không có output teacher | FAIL — **75/132** |
| E2 | đáp án test không phải echo context | FAIL — **108/132** là echo |
| E2 | test có `gold_compression` xác nhận | FAIL — 0/132 |
| E1 | test có người kiểm | manual — 0/132 có `verified_by` |
| E1 | leak-check test/train | **PASS** — CLEAN (14 eval docs vs 2.796 train docs) |
| P11 | card khớp phân bố `source` | **PASS** |
| E4 | card nêu rõ tập báo cáo vs sanity-check | manual — chưa nêu rõ |

---

## Phát hiện chính: cờ `budget_compliance` của v1 đo sai định nghĩa

v1 tự ghi `budget_compliance = True` cho **50.960/57.529 hàng (88,6%)**. Tính lại từ chính `target_tokens` và `realized_tokens` của cùng hàng đó theo công thức §2.2 — `abs(realized − target)/target ≤ 0,15` — chỉ còn **19,8%**. **44.515 hàng bất đồng.**

Nguyên nhân xác định được, khớp **100,0%**:

| Quy tắc | Mức khớp với cờ v1 |
|---|---|
| `realized ≤ target` | **100,0%** |
| `realized ≤ target × 1,15` | 95,7% |
| `abs(realized − target)/target ≤ 0,15` (§2.2) | 22,6% |

Cờ của v1 là kiểm tra **một phía**: chỉ phạt khi sinh ra **quá nhiều** token, không bao giờ phạt nén **quá tay**. Vì teacher có xu hướng nén quá tay một cách hệ thống, cờ này gần như luôn báo đạt.

**Hệ quả:** bất kỳ thống kê nào dựa trên cờ `budget_compliance` của v1 đều cao hơn thực tế khoảng 4,5 lần. Builder v2 phải **tính lại** cờ này, không được kế thừa (dùng `vncompress.dataset_schema.budget_compliant`, có test ở `tests/test_validate_dataset_v2.py`).

## Teacher gần như không phản ứng với mức nén yêu cầu

Đọc thẳng `realized_ratio` mà chính v1 đã ghi:

| Yêu cầu | Median thực (v1 tự ghi) | n |
|---|---|---|
| 2× | **5,3×** | 19.782 |
| 4× | **6,7×** | 19.452 |
| 8× | **9,3×** | 18.295 |

Yêu cầu tăng gấp 4 lần (2× → 8×) chỉ làm kết quả tăng 1,75 lần. Teacher nén về một độ dài gần như cố định bất kể ngân sách — và đó chính là cơ chế sinh ra **19.367 hàng trùng** ở P2: ba mức nén cho ra cùng một output.

**Đính chính:** một bản đo trước đó của tài liệu này báo "median 2,1× ở mức 2×, teacher bám target rất sát". Con số đó lấy từ 496 hàng trong config `vcc_bench` — một mẫu không đại diện cho 57.950 hàng thật. Kết luận đúng là kết luận của `docs/dataset_review.md`: nhãn ratio của v1 không dùng được, và dữ liệu nén phải **sinh lại** chứ không chỉ vá phần đuôi.

## Đính chính thứ hai: không có leak

Một bản đo trước báo "14 doc_id rò rỉ". Sai: 14 doc_id đó nằm giữa `vcc_bench/validation` và `records/validation` — cùng phía held-out nhìn qua hai config. Trùng với `records/train` = **0**. Card ghi CLEAN là đúng.

Validator đã sửa để chỉ so với `split == 'train'`, có test chặn đúng tình huống này (`tests/test_validate_dataset_v2.py::TestLeakCheck`).

---

## Phân chia công việc theo chi phí

**Không cần teacher** — chỉ biến đổi dữ liệu, xử lý phần lớn số FAIL:

1. Tách `kind=corpus` khỏi nhãn task (P7 — 22k hàng).
2. Map 465 `domain` Wikidata → 8 giá trị kiểm soát, giữ nhãn gốc ở `wikidata_type` (P8).
3. Gán `*_lang` theo từng text (P9).
4. Khử trùng `(doc_id, gold_compression)` → loại ~19,4k hàng trùng (P2).
5. **Tính lại `budget_compliance`** theo §2.2 và loại/đánh dấu hàng không đạt (P1).
6. Drop cột rỗng > 50%, gán `is_long_document`, tách thơ sang config riêng (P6, P5, T3).

**Cần teacher GLM-5.2:**

7. Sinh lại cặp nén có **ép ngân sách cứng** + re-prompt ≤ 2 lần → loại (P1, P2). Quy mô thật: ~80% số hàng không đạt ngưỡng §2.2, nên đây là sinh lại chứ không phải vá.
8. Gán `answerable_from_compression` (P4 — hiện 0/57.950).

**Cần người:**

9. Tập test độc lập ≥ 300 mẫu có `gold_compression` được xác nhận (E1–E3).
