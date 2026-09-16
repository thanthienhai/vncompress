# Kiểm định tập test độc lập (stage 4)

**Script:** `scripts/verify_eval_set.py` · **Spec:** `docs/dataset_rebuild_spec.md` §6 (E1–E3)

---

## 1. Đang kiểm cái gì

Không phải "bản nén có chứa đáp án không". Cái đó **đúng 1000/1000 theo cách dựng** — `gold_compression` được cắt từ chính câu chứa span người gán trong UIT-ViQuAD. Kiểm chứa đáp án sẽ pass hết và không nói lên điều gì.

Cái cần kiểm là **đọc bản nén đó có suy ra được đáp án không**:

```
[qa_uit_012357]
Q    : Ông nội của Thái Tông Trần Cảnh có tên là gì?
GOLD : "Trần Lý sinh ra Trần Thừa."
ANS  : Trần Lý
```

Chuỗi "Trần Lý" có mặt. Nhưng đọc đúng câu đó không thể biết Trần Lý là ông nội Trần Cảnh — thiếu mắt xích "Trần Thừa là cha Trần Cảnh". Bản nén **chứa đáp án mà không đủ để trả lời**.

Heuristic từ vựng (`risk_score`) gắn cờ ~9% số hàng có hình dạng này. Đó là lý do không được bỏ qua bước kiểm, và cũng là lý do không được giao hết cho heuristic.

---

## 2. Phân công LLM ↔ người

| | Làm gì | Khối lượng |
|---|---|---|
| **LLM** | chấm **toàn bộ** 1000 hàng | tự động |
| **Người** | review **mẫu phân tầng** | 150 hàng |

LLM chỉ nhìn `(query, gold_compression)` — **không bao giờ thấy `context`**. Nếu thấy context nó sẽ trả lời từ context và chứng nhận cả những bản nén không hề chứa đáp án, tức đúng lỗi đang cần sàng.

Mẫu 150 hàng của người **không phải để kiểm thêm 150 hàng**. Nó để đo LLM sai bao nhiêu. Không có nó thì tỷ lệ lỗi của judge là ẩn số và chữ "đã kiểm" chỉ là một từ, không phải một phép đo.

### Vì sao dùng LLM ở đây là hợp lệ

`gold_compression` của tập eval **không do model sinh** — nó cắt từ span người gán. Judge đang sàng nhãn của người khác, không chấm bài của chính nó. Đó là khác biệt then chốt với v1, nơi tập test **chính là** output của teacher (P3/P4).

### Hai ràng buộc không được phá

1. **Judge ≠ teacher.** Teacher là `GLM-5.2`, judge mặc định là `Qwen3-235B-A22B`. Hai model cùng điểm mù sẽ đồng thuận với nhau đúng ở chỗ cả hai cùng sai. Script **từ chối chạy** nếu trùng (trừ khi có `--allow-same-model`, và lựa chọn đó bị ghi vào report).
2. **`llm` không bao giờ được đọc thành `human`.** Mỗi hàng mang `verification_method ∈ {none, llm, human, llm+human}`, và validator có dòng riêng cho từng loại.

---

## 3. Quy trình

```bash
# 4a. LLM chấm toàn bộ  (thêm --dry-run để chạy thử không cần API key)
export VNCOMPRESS_JUDGE_API_KEY=...
python scripts/verify_eval_set.py --input data/vncompress_vi_v2 --stage judge

# 4b. Xuất hàng đợi cho người
python scripts/verify_eval_set.py --input data/vncompress_vi_v2 --stage sample

#     -> provenance/human_review_queue.tsv   (mở bằng Excel/Sheets)
#        điền cột `human_verdict`: 1 = trả lời được, 0 = không
#        điền cột `annotator`: tên/mã người kiểm

# 4c. Gộp lại, tính độ đồng thuận
python scripts/verify_eval_set.py --input data/vncompress_vi_v2 --stage merge
#     -> provenance/verification_report.json
```

**Người kiểm không được nhìn verdict của LLM.** File TSV cố tình không có cột đó. Thấy nó thì việc chấm độc lập biến thành việc xác nhận, và con số đồng thuận tính từ các xác nhận không còn là con số đồng thuận.

### Mẫu được chọn thế nào

Ngân sách chia **đôi theo verdict của LLM trước** (75 pass / 75 fail), rồi trong mỗi nửa mới trải đều theo risk tier và document.

Thứ tự này quan trọng. Coi cả ba trục ngang nhau trông có vẻ tương đương nhưng không phải: khi verdict tương quan với document — điều thực sự xảy ra, vì một bài bị cắt tệ sẽ sinh một loạt bản nén tệ — số tầng document áp đảo số tầng verdict và verdict thiểu số bị pha loãng về đúng tỷ lệ nền. Một pool thử nghiệm có 25% bản nén bị loại trả về 26,7%, tức là "phân tầng" chỉ trên danh nghĩa. Cohen's κ ước lượng từ các ô ngoài đường chéo, nên mẫu ít hàng bị loại gần như không có gì để ước lượng.

---

## 4. Cổng nghiệm thu

`verification_report.json` chỉ ghi `calibrated: true` khi **cả hai** điều kiện đúng:

| Điều kiện | Ngưỡng | Vì sao |
|---|---|---|
| Số hàng người kiểm | ≥ 150 | ở mức đồng thuận ~90%, khoảng tin cậy 95% là ±5 điểm |
| Cohen's κ | ≥ 0.70 | κ chứ không phải % đồng thuận thô |

**Vì sao là κ:** nếu 90% số hàng thực sự trả lời được, một judge luôn nói "được" đạt 90% đồng thuận thô mà không học được gì. κ chấm judge đó **bằng 0**. Chỉ κ phân biệt được hai trường hợp này.

κ = `None` (không xác định, khi một bên không bao giờ đổi verdict) **không bao giờ** vượt cổng — báo 0.0 ở đó sẽ bị đọc thành "judge bất đồng" trong khi sự thật là "không đo được từ mẫu này".

### Câu được phép viết vào bài báo

`verification_report.json` tự sinh ra câu đó từ chính các con số, để prose không trôi khỏi dữ liệu:

- **Đạt cổng:** `"LLM-verified (150 of 1000 rows independently reviewed by a person; Cohen's kappa 0.907)"`
- **Không đạt:** `"LLM-verified, NOT human-calibrated (... kappa 0.373 < 0.7) -- must not be reported as human-verified"`

Không đạt cổng thì **không được viết "human-verified"**. Cách xử lý: tăng mẫu người kiểm, sửa prompt judge, đổi model judge, hoặc thu nhỏ tập test xuống đúng phần đã có người kiểm.

---

## 5. Validator kiểm lại những gì

`scripts/validate_dataset_v2.py --input data/vncompress_vi_v2` có 4 dòng cho stage này:

| Ref | Dòng | PASS khi |
|---|---|---|
| E2 | every test row carries a verification verdict | không hàng nào còn `method='none'` |
| E1 | independent test verified by a person | ≥150 hàng người kiểm **và** κ ≥ 0.70 |
| E3 | judge model differs from the teacher | judge ≠ teacher trong report |
| E3 | no dry-run verdicts in the shipped test set | không hàng nào mang verdict `--dry-run` |

Dòng E1 **không thể** pass bằng verdict LLM. Bản trước của check này chỉ xét `verified_by` có rỗng hay không — ghi `"llm:GLM-5.2"` vào đó sẽ lật một dòng ghi "human-verified" thành PASS mà không có người nào tham gia. Đó đúng là kiểu tự chứng nhận mà chữ "người kiểm" trong spec sinh ra để chặn.
