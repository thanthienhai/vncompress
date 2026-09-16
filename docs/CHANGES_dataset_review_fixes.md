# Sửa pipeline theo kết quả rà soát dataset v2

**Ngày:** 2026-09-16 · **Căn cứ:** [`dataset_v2_review.md`](dataset_v2_review.md)
**Trạng thái:** đã land vào project. `python -m pytest -q` → **570 passed**.

Bốn patch `0001`–`0004` trong `vncompress-code-changes/` đã được apply, rồi chồng
các sửa đổi dưới đây lên. Thư mục đó nay thừa và đã xoá; cây project là bản đúng
duy nhất. **Chưa commit** — cần review rồi commit.

---

## Tóm tắt

| # | Vấn đề (mục trong bản rà soát) | Hiệu quả đo được |
|---|---|---|
| 1 | Cổng answerability chấm token-F1 → loại nhầm 50,4% (§3.2) | **lấy lại 310/585 hàng**; 973 → **1.283** (+32%), 0 hàng đang giữ bị loại ngược |
| 2 | Gold là văn viết lại, không trích xuất (§3.1) | prompt **v3 trích xuất** + gate cứng + 2 metric + 2 check validator |
| 3 | Test 1.000 dòng / 14 tài liệu (§3.3) | `--max-eval-per-doc`; validator **FAIL** đúng dataset đã ship |
| 4 | `answer_lang` sai ~17% (§4.4) | ngưỡng 20 ký tự chọn từ đo đạc; số thuần hết bị gán `en` |
| 5 | `numbers_preserved` báo thiếu | regex nuốt dấu câu → sai **500/973 hàng**; median 0,191 → 0,228 |
| 6 | Thiếu `verification_method`, tên judge (§5.2) | 3 trường mới trong `gen_config` |
| 7 | CHECKSUMS bỏ sót `compression.jsonl` + `extras/` (§6) | stage 3 tự merge vào manifest |
| 8 | Rò "Hà Nội" sang VCC-Bench v2 (§5.3) | `--holdout-docs-from`; phát hiện đúng `hà_nội`/46 hàng trên dữ liệu thật |
| 9 | Toàn bộ train trên 143 tài liệu (§4.1) | **stage 2b mới**: 3.420 tài liệu đủ điều kiện (**gấp 24 lần**) |
| 10 | `check_card` báo FAIL giả (§5 cũ) | so khớp định danh thay vì kiểu chữ |

---

## 1. Gold phải là trích xuất — prompt v3 + gate cứng

`PROMPT_VERSION` `v2` → `v3`. Prompt cũ nói "nén" và "giữ nguyên văn mọi câu chữ
cần thiết", để model tự quyết rằng viết lại phần còn lại cũng là nén. Prompt mới
không bảo *nén* nữa mà bảo *chọn*:

```
- CHỈ được chép lại NGUYÊN VĂN các câu có sẵn trong ĐOẠN VĂN.
- KHÔNG viết lại, KHÔNG tóm tắt, KHÔNG diễn giải, KHÔNG rút gọn câu.
- KHÔNG thêm bất kỳ thông tin nào không có trong ĐOẠN VĂN — kể cả khi bạn biết nó đúng.
```

Và `extractive_problem()` **kiểm chứng** thay vì tin: trượt
`sentence_extractive_ratio ≥ 0.80` hoặc có `unsupported_numbers` → re-prompt
**có chỉ đích danh câu vi phạm** (trích lại đúng câu model bịa), tối đa 2 lần,
vẫn trượt thì **loại mẫu**. Run report tách riêng hai lý do loại (`rewrote the
sentences` / `invented a number`) vì cách sửa khác nhau.

`--allow-abstractive` để dựng lại nhánh abstractive một cách có chủ ý.

**Cần làm ngoài code:** 1.283 hàng hiện có là sản phẩm của prompt v2 → chuyển
sang config `compression_abstractive` riêng, không trộn vào bản sinh lại.

## 2. Cổng answerability — `answerability_verdict()`

Thêm hai lối chấp nhận ngoài F1: `containment` (đáp án gốc nằm trong câu trả lời
của judge hoặc ngược lại) và `span` (đáp án gốc còn nguyên văn trong bản nén —
bằng chứng chắc nhất, không cần judge). Mỗi hàng ghi `answerability_method`.

Chấm lại 585 hàng `extras/` sẵn có: **`span` 188 · `containment` 122 · vẫn loại
275** → lấy lại **310 hàng (53,0%)**. Cổng chỉ nới không siết: 973 hàng đang giữ
không hàng nào bị đảo.

**Cần làm ngoài code:** chạy lại 585 hàng đó. Không tốn API — mọi đầu vào
(`answerability_prediction`, `answerability_f1`, `gold_compression`) đã có sẵn.

## 3. Stage 2b mới — `scripts/generate_qa_from_corpus.py`

Mở rộng **số tài liệu**, thứ thực sự khan hiếm. ViQuAD 2.0 có 184 bài, còn 143
sau khi chia split; `corpus.jsonl` có 4.010 tài liệu mà chưa gì dùng tới, chỉ
thiếu câu hỏi. Script này hỏi teacher.

```bash
python scripts/generate_qa_from_corpus.py --plan          # 3.420 tài liệu đủ điều kiện
python scripts/generate_qa_from_corpus.py --limit 5 --dry-run
python scripts/generate_qa_from_corpus.py --questions-per-doc 8
python scripts/generate_compression_pairs.py --include-synthetic-qa
```

Ba ràng buộc được cài cứng:

- **Chỉ train/validation.** Tài liệu split `test` bị loại ở `load_corpus_documents`
  *và* bị `raise` ở worker — B5/E3 ("tập test độc lập KHÔNG dùng output teacher")
  đúng theo cấu trúc, không phải theo bộ lọc.
- **Mọi đáp án đều span-verified.** Giữ hàng chỉ khi đáp án xuất hiện nguyên văn
  trong context (có nới cho khác biệt khoảng trắng); `answer_span` **tính từ văn
  bản**, không tin model. Đáp án diễn giải → loại. Chính là lỗi đã tạo ra bản nén
  viết lại của v2, chặn trước khi nó thành nhãn.
- **Ghi ra file riêng** `qa_synthetic.jsonl`, và stage 3 chỉ đọc khi có cờ
  `--include-synthetic-qa`. Nếu file đó chứa hàng split `test`, stage 3 **dừng
  hẳn** thay vì chạy tiếp.

Mỗi hàng mang `question_source: teacher-generated`; hàng người viết trong
`qa.jsonl` nay mang `question_source: human`, nên bản trộn vẫn tự mô tả được.

> Một bug do chính file mới này lộ ra và đã sửa: validator glob `qa*.jsonl` nên
> **nuốt luôn `qa_synthetic.jsonl` vào config `qa`** — đúng cái trộn ngầm mà việc
> tách file sinh ra để tránh, xảy ra ngay bên trong công cụ có nhiệm vụ bắt nó.
> Giờ mỗi file thuộc về đúng một config (tên khớp dài nhất thắng), và bản build
> chia shard vẫn đọc gộp như cũ.

## 4. Rò tài liệu sang benchmark ngoài

`--holdout-docs-from PATH` (lặp được, đọc VCC-Bench JSON hoặc v2 JSONL) loại tài
liệu mà một benchmark ngoài đang dùng ra khỏi **train**; vẫn cho phép ở test của
chính bản build này, vì hai tập eval trùng tài liệu là chuyện báo cáo, còn train
trên tài liệu test của mình là kết quả hỏng.

Không có cờ thì **vẫn phát hiện và cảnh báo**, chỉ là không xoá hàng — nhìn thấy
là mặc định, đổi dữ liệu là lựa chọn. Chạy trên dữ liệu thật: phát hiện đúng
`hà_nội`, 46 hàng train.

## 5. Đo độ trích xuất và ngôn ngữ

- `sentence_extractive_ratio`, `unsupported_numbers` (trả danh sách số bịa, không
  chỉ đếm) vào `COMPRESSION_FIELDS` + 2 check validator. Dataset cũ thiếu trường
  → validator báo **MANUAL**, không PASS.
- `detect_lang(text, fallback)` ba tầng, ngưỡng `LANG_MIN_ALPHA_CHARS = 20` chọn
  từ đo đạc (1.002 mẫu tiếng Việt chắc chắn: 77,8% nhận đúng ở 3 ký tự → 99,4% ở
  20). Validator cũng thôi chạy lại đúng hàm đã sinh ra nhãn — span quá ngắn nay
  báo "too short to judge" thay vì tự xác nhận.

## 6. Kiểm chứng trên dữ liệu thật

Validator đã sửa, chạy trên chính bản v2 đã ship:

```
B1  compression: gold is extractive      FAIL  mean 0.363; 678/973 dưới 0.80
B1  compression: no unsupported numbers  FAIL  633/839 (75.4%) sạch
E1  independent test spans >= 150 docs   FAIL  1000 row(s) over 14 document(s)
E1  no single test document dominates    FAIL  Hà_Nam_(Trung_Quốc) 134 = 13.4%
P9  language labels match content        PASS  0 sai/665, 135 span quá ngắn để xét
```

Dry-run toàn chuỗi 2b → 3 (không gọi API): span-verified yield 100%,
`sentence_extractive_ratio` median 1.0, `budget_compliance` 100%.

---

## 7. Còn lại — việc vận hành, không phải code

- **Sinh lại `compression` bằng prompt v3.** Bản 1.283 hàng hiện có thuộc v2.
- **Chạy lại 585 hàng `extras/`** qua cổng mới (miễn phí).
- **Chạy stage 2b** rồi stage 3 với `--include-synthetic-qa`.
- **Stage 4 (`verify_eval_set.py`) chưa chạy** — code sẵn sàng và đã từ chối
  judge trùng teacher; cần API key + người review.
- **`MIN_EVAL_DOCUMENTS = 150` vẫn chưa đạt được** bằng ViQuAD (14 bài test).
  Stage 2b không giải quyết được chỗ này vì test phải là câu hỏi người viết —
  cần thêm nguồn QA người viết nếu muốn tập test đa dạng tài liệu.
- **Cập nhật dataset card**: thêm định danh `legal` (validator báo đúng, card
  chỉ mô tả bằng văn xuôi "Văn bản pháp luật VN"), nêu rõ tập báo cáo vs
  sanity-check, sửa "72.270 tài liệu" → đoạn văn.
