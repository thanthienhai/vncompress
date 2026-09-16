# VNCompress-VI v2 — tổng quan quy mô & nguồn dữ liệu

Cập nhật: 2026-09-16 — stage 3 tạm dừng để sửa pipeline theo
[`dataset_v2_review.md`](dataset_v2_review.md); xem
[`CHANGES_dataset_review_fixes.md`](CHANGES_dataset_review_fixes.md).

Pipeline hiện có 5 stage:

| Stage | Script | Ra cái gì |
|---|---|---|
| 1 | `fetch_sources_v2.py` | `data/raw_v2/` |
| 2 | `build_vncompress_vi_v2.py` | `corpus`, `qa`, `eval/test` |
| **2b** | **`generate_qa_from_corpus.py`** | **`qa_synthetic.jsonl` — câu hỏi teacher trên 3.420 tài liệu corpus, chỉ train/validation** |
| 3 | `generate_compression_pairs.py` | `compression`, `extras/` |
| 4 | `verify_eval_set.py` | `verification_method` cho tập test (**chưa chạy**) |

## 1. Nguồn gốc dữ liệu

Toàn bộ dataset dựng từ **3 nguồn upstream**, tất cả đều public:

| Nguồn | HF dataset | Dùng cho | License |
|---|---|---|---|
| UVW-2026 | `undertheseanlp/UVW-2026` | `corpus.jsonl` — kho tài liệu retrieval | CC-BY-SA-4.0 |
| UIT-ViQuAD 2.0 | `taidng/UIT-ViQuAD2.0` | `qa.jsonl` + `eval/test.jsonl` — câu hỏi/đáp án | CC-BY-SA-4.0 |
| Văn bản pháp luật VN | (thu thập rời) | 31 doc trong corpus | Public Domain |

Poetry (`bigscience-data/roots_vi_vietnamese_poetry`) có trong script fetch nhưng **không có mặt** trong v2 hiện tại.

## 2. Quy mô từng file

| File | Rows | Nội dung |
|---|---|---|
| `corpus.jsonl` | 72.301 | kho tài liệu (uvw-2026: 72.270 · legal: 31) |
| `qa.jsonl` | 6.000 | câu hỏi **người viết**, nguồn đầu vào của stage 3 (train 5.482 · validation 518) |
| `qa_synthetic.jsonl` | chưa sinh | câu hỏi **teacher sinh** trên tài liệu corpus (stage 2b); stage 3 chỉ đọc khi có `--include-synthetic-qa` |
| `eval/test.jsonl` | 1.000 | test set độc lập, split `test`, không trùng document với train |
| `compression.jsonl` | đang sinh | cặp (context → compressed) do teacher GLM-5.2 tạo |
| `extras/compression_unanswerable.jsonl` | đang sinh | các cặp mà judge kết luận không trả lời được |

## 3. Stage 3 sinh ra bao nhiêu

```
6.000 qa rows  ×  3 tỷ lệ nén (2.0, 4.0, 8.0)  =  18.000 work item
yield đo được: 47%            →  ~8.460 compression rows cuối cùng
```

Phần chênh 53% là các item bị quality gate loại (sai budget compliance ±15%,
judge kết luận unanswerable, teacher bỏ cuộc).

> **Đính chính (2026-09-16).** Câu "phần chênh không phải lỗi" ở bản trước là
> sai với nhánh answerability. Rà soát 585 hàng `extras/` cho thấy **295 hàng
> (50,4%) vẫn chứa nguyên văn đáp án người gán** — cổng cũ chấm token-F1 của
> *cách diễn đạt* judge so với một span ngắn, nên loại nhầm cả câu trả lời
> đúng-nhưng-dài hơn lẫn đúng-nhưng-ngắn hơn. Cổng đã được sửa (thêm hai lối
> chấp nhận `containment` và `span`, xem `answerability_verdict`): chấm lại
> đúng 585 hàng đó **lấy lại 310 hàng (53,0%)**, đưa bộ nén 973 → 1.283
> (+32%) mà không gọi thêm API, và không hàng nào đang giữ bị loại ngược.
> Yield 47% ở trên vì thế là **ước lượng thấp**; cần đo lại sau lần chạy tới.
> Chi tiết: `docs/dataset_v2_review.md` §3.2.

**Tiến độ (tạm dừng 2026-09-15 để review):**

| | |
|---|---|
| work item đã chạy qua teacher | ~3.052 / 18.000 = **17%** |
| row thu được | 1.558 (compression 973 + extras 585) |
| batch 1 | 1.500 item → 677 row · `done` |
| batch 2 | 1.500 item → 903 row · `done` |
| batch 3 | 52 item → 45 row · `partial`, dừng tại đây |

Lưu ý khi đọc số: *work item* (= 1 câu hỏi × 1 tỷ lệ nén) và *row* là hai đại
lượng khác nhau — chênh lệch là phần bị quality gate loại.

Bản chụp tại thời điểm này đã đẩy lên
<https://huggingface.co/datasets/anhalu/vncompress-vi-v2> (public, kèm dataset card).
Tốc độ 20 item/phút → **~12,6 giờ** nữa.

## 4. Đặc điểm cần lưu ý

- **Chỉ 1 nguồn QA.** Cả 6.000 qa rows lẫn 1.000 test rows đều là
  uit-viquad-2.0, `question_type` 100% là `extractive`. Volume lớn nhưng
  đa dạng nguồn = 1. Với một benchmark, đây là điểm reviewer sẽ hỏi.
- **Domain gần như không phân loại:** `other` 5.331/6.000 (89%).
  Chỉ 833 row có domain thật từ uvw-category (admin/history/science).
- **Context dài:** median 19.453 ký tự, p90 23.799, trần cứng 24.000.
- **Bias theo độ dài (chưa xử lý):** tỷ lệ sống sót giảm theo độ dài context —
  <5k: 96% · 5–10k: 91% · 10–20k: 76% · >20k: 62%. Dataset cuối sẽ lệch về
  tài liệu ngắn, ngược hướng mong muốn của một benchmark long-context.
