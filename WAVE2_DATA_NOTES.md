# Dữ liệu v2 — những gì đã đổi và đội training cần làm gì

**Ngày:** 2026-09-16 · Đọc cùng [`WAVE2_HANDOFF.md`](WAVE2_HANDOFF.md)
Chi tiết kỹ thuật: [`docs/dataset_v2_review.md`](docs/dataset_v2_review.md) (rà soát) và
[`docs/CHANGES_dataset_review_fixes.md`](docs/CHANGES_dataset_review_fixes.md) (sửa gì).

---

## 1. Ba điều cần biết trước khi chạy bất cứ thứ gì

**1. Bản `anhalu/vncompress-vi-v2` trên HuggingFace hiện tại KHÔNG dùng để train
E5/E6 được.** Không phải vì ít dữ liệu, mà vì sai loại: `gold_compression` là
văn bản **model viết lại**, không phải trích xuất. Đo trên 973 hàng đã ship:

| | |
|---|---|
| câu trong gold xuất hiện nguyên văn trong context | **30,0%** (median mỗi hàng: 0,038) |
| hàng khẳng định con số **không có** trong context | **206/973 = 21,2%** |

Mọi arm wave 2 đều là *chọn lọc* — E1 chấm token, E5 chọn câu, E7 phân bổ theo
lớp, E6 học nhãn giữ/bỏ mức token. Không arm nào sinh được một câu mà context
không chứa, nên gold hiện tại là đích không với tới. Pipeline đã sửa (prompt v3
trích xuất + gate cứng) nhưng **dữ liệu chưa sinh lại**.

**2. Có một đường rò train → VCC-Bench v2 phải chặn trước khi chạy E4/E6.**
Bài **Hà Nội** nằm ở split `train` của `qa.jsonl` (46 hàng) **và** chiếm 32/414
mẫu (7,7%) của `data/benchmark/vcc_bench_v2.json` — 54 câu trùng nguyên văn.
`WAVE2_HANDOFF.md` §4 đã dặn đừng train E4 trên vcc_bench; rò đến từ chiều ngược lại.

**3. Tập test 1.000 dòng chỉ trải trên 14 tài liệu**, 4 bài chiếm 534 dòng.
n hiệu dụng là 14, không phải 1000 — **đừng báo cáo khoảng tin cậy tính trên
1.000 dòng này**, nó hẹp giả tạo. Dùng `eval/test.jsonl` để sanity-check, còn
kết quả chính vẫn báo trên VCC-Bench v2.

---

## 2. Dùng được gì, chưa dùng được gì

| Arm | Dữ liệu | Trạng thái |
|---|---|---|
| **E4** relevance probe | `qa.jsonl` (6.000 hàng, `answer_span` đúng 100%) | ✅ **dùng được ngay** — tốt hơn `reference_answer` của VCC-Bench. Phải loại `viquad:Hà_Nội` |
| **E6** encoder classifier | `compression.jsonl` | ❌ chờ sinh lại bằng prompt v3 |
| **E5** chọn mức câu | `compression.jsonl` | ❌ chờ sinh lại (gold phải là tập con câu của context) |
| **E1 / E7** | không cần gold | ✅ chạy được, eval trên VCC-Bench v2 |
| **E5/E8/E9** phần eval | cần needle / cross_lingual | ❌ v2 **không có** 4 task này → vẫn dùng VCC-Bench v2 |

Nói gọn: **v2 là nguồn *train* bổ sung, không thay thế VCC-Bench cho eval.**

---

## 3. Việc cụ thể

### E4 — probe liên-quan (chạy được ngay)

`qa.jsonl` cần convert sang shape VCC-Bench `{context, query, reference_answer, task}`,
và **loại tài liệu Hà Nội**:

```python
from vncompress.dataset_build import document_key
BLOCKED = {'hà_nội'}          # trùng với vcc_bench_v2.json
rows = [r for r in qa_rows
        if r['split'] == 'train' and document_key(r['doc_id']) not in BLOCKED]
```

Cảnh báo về quy mô: toàn bộ `qa.jsonl` chỉ trải trên **143 tài liệu**
(~42 câu hỏi/bài). Probe hoàn toàn có thể học "đoạn nào của bài *Nhà Minh* hay
chứa đáp án" thay vì học quan hệ query↔token. Khi A/B, **tách theo document**
chứ đừng tách theo hàng, nếu không kết quả sẽ đẹp một cách giả tạo.

Muốn nhiều tài liệu hơn thì chạy stage 2b (mới): sinh câu hỏi teacher trên
**3.420 tài liệu** của `corpus.jsonl`, chỉ train/validation, mọi đáp án đều
span-verified:

```bash
python scripts/generate_qa_from_corpus.py --plan                 # xem quy mô
python scripts/generate_qa_from_corpus.py --questions-per-doc 8  # cần API key
```

### E6 / E5 — chờ sinh lại `compression`

```bash
python scripts/generate_compression_pairs.py --input data/vncompress_vi_v2
# prompt v3 (trích xuất) + gate cứng là mặc định
# --include-synthetic-qa  : dùng thêm câu hỏi từ stage 2b
# --allow-abstractive     : tắt gate, dựng lại bản cũ (đừng dùng để train E5/E6)
```

Đợi xong rồi hãy train. Bản 1.283 hàng hiện có thuộc prompt v2 — nếu vẫn muốn
giữ, để ở config riêng, **đừng trộn**.

---

## 4. Trường mới trên mỗi hàng `compression` (dùng để lọc)

```python
row['sentence_extractive_ratio']  # ≥0.80 mới là trích xuất thật
row['unsupported_numbers']        # [] là sạch; list số model bịa ra
row['answerability_method']       # 'f1' | 'containment' | 'span'
row['verification_method']        # 'none' = stage 4 chưa chạy
row['gen_config']['judge_model']  # + judge_is_teacher: teacher tự chấm bài mình
```

Lọc an toàn nhất cho E6:

```python
usable = [r for r in rows
          if r['sentence_extractive_ratio'] and r['sentence_extractive_ratio'] >= 0.8
          and not r['unsupported_numbers']]
```

Hàng của bản đã ship **không có** các trường này — thiếu trường nghĩa là chưa
kiểm được, không phải là đạt.

## 5. Vì sao cổng answerability nới ra (ảnh hưởng số lượng mẫu)

Cổng cũ chấm token-F1 *cách diễn đạt* của judge so với một span ngắn, nên loại
nhầm cả câu trả lời đúng-nhưng-dài hơn lẫn đúng-nhưng-ngắn hơn:

> `Chế độ nhà nước của Ả Rập Xê Út là gì?` — đáp án `quân chủ chuyên chế`,
> judge trả `Quân chủ chuyên chế, chế độ độc tài thế tập do hoàng tộc cai trị.`
> → F1 0,42 → **loại**.

295/585 hàng trong `extras/` bị loại oan kiểu đó. Đã thêm hai lối chấp nhận
(`containment`, `span`): chấm lại lấy về **310 hàng (53%)**, và không hàng nào
đang giữ bị đảo ngược. Nghĩa là bộ train nén sắp tới sẽ **lớn hơn ~32%** so với
con số các bạn thấy trên HF, và bớt lệch về câu hỏi có đáp án khớp mặt chữ —
vốn là loại câu mà compressor query-agnostic cũng làm được, tức là làm *nhỏ đi*
chính khoảng cách E1 sinh ra để đo.

---

## 6. Checklist trước khi báo cáo số

- [ ] Đã loại `viquad:Hà_Nội` khỏi train (hoặc loại 32 mẫu đó khỏi VCC-Bench)
- [ ] `compression` đã sinh bằng prompt v3 (`gen_config.prompt_version == 'v3'`)
- [ ] A/B của E4 tách theo **document**, không phải theo hàng
- [ ] Kết quả chính báo trên VCC-Bench v2; `eval/test.jsonl` chỉ sanity-check
- [ ] Không trích dẫn CI tính trên 1.000 dòng test (14 tài liệu)
