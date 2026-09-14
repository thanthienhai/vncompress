# Chạy toàn bộ arm đã có, chưa train gì · Cổng 1

**Sản phẩm bàn giao:** bảng kết quả đầy đủ + hình C1 (đối chứng dấu thanh)
**Cần GPU:** Có, nhưng chỉ để chạy suy luận (inference), không train
**Nguyên tắc của của phần này:** không viết model mới, không train. Chỉ lấy số.

---

## 1. Tại sao chạy trước khi làm gì thêm

`WAVE2_HANDOFF.md` ghi rõ: *"code đã implement + test (CPU), chưa chạy GPU"*.
Bạn đang ngồi trên **năm arm đã code xong mà chưa có một con số nào**:

| Arm | Đề xuất gốc | Nội dung | Cần train? |
|---|---|---|---|
| `llmlingua_contrastive` | E1 | Perplexity có điều kiện câu hỏi (LongLLMLingua) | Không |
| `lacc_ppl_contrastive` | E1 | LACC chỉ dùng PPL query-conditioned | Không |
| `lacc_sentence` | E5 | Chọn theo **cả câu** thay vì token lẻ | Không |
| `lacc_ppl_morph` | E2 | PPL × morphology (nhân, không cộng) | Không |
| `lacc_classprop` | E7 | Phân bổ budget theo hạn ngạch lớp từ | Không |

Cửa sổ PPL cũng đã sửa từ 512/256 sang 2048/256 (E3), giảm khoảng 6–7× số lần
forward. Nghĩa là bảng chi phí ở §8.7 báo cáo Wave 1 **đã lỗi thời** và phải đo
lại trước khi trích dẫn.

`lacc_sentence` đáng chú ý nhất. Nó chọn theo cả câu, tức là né được lỗi "giữ
đáp án nhưng xoá mất tiền tố" một cách tự nhiên. Nhiều khả năng đây là arm mạnh
nhất của bạn, và nó hoàn toàn miễn phí.

---

## 2. Việc 1 — Ma trận arm đầy đủ

Chạy đủ 11 arm dưới đây. Mười arm đầu đã có sẵn hoặc rất rẻ; hai arm cuối phải
viết thêm (xem §3).

```
A0  none                      không nén (trần trên)
A1  random                    xoá ngẫu nhiên (sàn dưới)
A2  llmlingua                 PPL, query-agnostic
A3  llmlingua_contrastive     PPL có điều kiện câu hỏi        ← baseline query-aware
A4  lacc_ppl_contrastive      LACC + PPL query-conditioned
A5  lacc_ppl_morph            PPL × morphology
A6  lacc_cx_morph             PPL query-conditioned × morphology
A7  lacc_sentence             chọn theo câu                    ← ứng viên sáng giá
A8  lacc_classprop            hạn ngạch theo lớp từ
A9  lacc_tone                 arm Wave 1 (để tái hiện C1)
B1  sentence_retrieval        chọn câu bằng BM25 / embedding   ← PHẢI VIẾT THÊM
B2  translate_then_compress   dịch sang Anh → LLMLingua-2      ← PHẢI VIẾT THÊM
```

**A9 bắt buộc phải có.** Không có nó thì bạn không tái hiện được kết quả Wave 1
trên benchmark mới, và toàn bộ đóng góp C1 mất chỗ dựa.

---

## 3. Việc 2 — Hai baseline còn thiếu

Đây là hai baseline nguy hiểm nhất. Nếu chúng thắng, cả hướng nghiên cứu cần xem
lại — nên phải biết sớm, không phải ở tuần 10.

### B1 — Chọn câu bằng retrieval

Cắt context thành câu, chấm điểm mỗi câu bằng BM25 hoặc cosine similarity với
embedding câu hỏi, giữ các câu điểm cao nhất cho tới khi đầy budget.

Khoảng 60 dòng code, không cần GPU cho BM25. Đây là thứ mà một kỹ sư sẽ làm đầu
tiên trong thực tế, và nó thường rất mạnh trên QA. **Nếu VQLC không thắng được
nó ở cùng budget thì paper không có lý do tồn tại.** Biết điều này ở tuần 4 rẻ
hơn nhiều so với biết ở tuần 11.

### B2 — Dịch rồi nén

Dịch context sang tiếng Anh, nén bằng LLMLingua-2 (mô hình có sẵn, đã train),
rồi đưa cho model sinh.

Bài *Lost in Compression* (arXiv 2608.26175) cho thấy pipeline này ngang hoặc
thắng nén bản địa ở khoảng **một nửa chi phí token** trong 3/5 ngôn ngữ được
thử. Tiếng Việt chưa được thử. Nếu nó thắng trên tiếng Việt, đó **vẫn là một
kết quả đăng được** — chỉ là paper đổi thông điệp. Nếu nó thua, bạn có một đoạn
phản biện mạnh mà không paper nào khác có.

---

## 4. Việc 3 — Cấu hình chạy

```bash
python benchmark.py \
  --model <generator đã chốt> \
  --data-path data/benchmark/vcc_bench_v2.json \
  --ratios 2,4,8 \
  --methods none,random,llmlingua,llmlingua_contrastive,lacc_ppl_contrastive,\
lacc_ppl_morph,lacc_cx_morph,lacc_sentence,lacc_classprop,lacc_tone
```

**Bốn quy tắc không được vi phạm:**

1. **Cùng budget thực tế, không phải cùng tỉ lệ cấu hình.** Báo cáo Wave 1 cho
   thấy LLMLingua đạt realized CR 4.70× khi đặt 4×. So sánh ở tỉ lệ cấu hình là
   so sánh sai điểm vận hành. Ghi lại `realized_compression_ratio` từng mẫu và
   khi tổng hợp thì so ở CR thực.
2. **Hai generator model** để kiểm tra tính bền. Nếu kết luận đảo chiều giữa hai
   generator thì đó là kết luận về generator, không phải về compressor.
3. **Ba seed** cho arm `random`, lấy trung bình. Một seed của random không phải
   sàn dưới, nó là một mẫu ngẫu nhiên.
4. **Đo lại latency và VRAM** sau khi đã áp cửa sổ 2048/256. Bảng chi phí Wave 1
   không dùng lại được.

---

## 5. Việc 4 — Phân tích bắt buộc: đây chính là đóng góp C1

Đây là phần quan trọng nhất của hai tuần, quan trọng hơn cả bảng tổng hợp.

**Tách kết quả needle theo `needle_group`:**

| Arm | Nhóm A (không dấu) | Nhóm B (có dấu) | Nhóm C (hỗn hợp) |
|---|---|---|---|
| `lacc_tone` | ? | ? | ? |
| `llmlingua` | ? | ? | ? |
| `random` | ? | ? | ? |

Giả thuyết cần kiểm chứng: `lacc_tone` sụp trên nhóm A nhưng **bình thường hoặc
tốt** trên nhóm B, trong khi `llmlingua` và `random` không có khác biệt đáng kể
giữa hai nhóm.

Nếu đúng, bạn đã chứng minh con số 2.3% là **artifact của thiết kế benchmark**,
không phải bài học phương pháp luận. Đó là một hình vẽ duy nhất, rất sạch, và là
thứ reviewer sẽ nhớ lâu nhất trong paper.

Nếu sai — tức `lacc_tone` sụp trên cả hai nhóm — thì kết luận gốc của Wave 1 lại
đúng, và bạn có quyền giữ nguyên Contribution 1. Cả hai kết cục đều đăng được.
Điều duy nhất không chấp nhận được là không đo.

**Phân tích phụ, làm nếu còn thời gian:**
- Kết quả theo `insert_position` → hiệu ứng lost-in-the-middle trên tiếng Việt.
- Tương quan giữa TPR (tone preservation rate) và token-F1. Wave 1 báo cáo nó
  **nghịch biến**. Xác nhận lại trên v2 — con số này sẽ dùng lại ở tuần 5.

---

## 6. Metric phải ghi cho mỗi lần chạy

| Nhóm | Metric |
|---|---|
| Chất lượng | Exact Match, token-F1, needle recall, ROUGE-L |
| Nén | tỉ lệ nén thực tế, số token giữ lại |
| Chi phí | latency đầu-cuối, VRAM đỉnh |
| Chẩn đoán | TPR, và (từ tuần 5) Word Integrity Rate |

Lưu kết quả **ở cấp từng mẫu**, không chỉ trung bình. Tuần 10 bạn sẽ cần chạy
paired bootstrap trên dữ liệu mẫu-theo-mẫu, và nếu chỉ lưu trung bình thì phải
chạy lại toàn bộ.

---

## 7. 🚦 Cổng 1 — Quyết định cuối tuần 4

**Câu hỏi:** E1 (contrastive PPL) hoặc E5 (chọn theo câu) có lấy lại phần lớn
khoảng cách so với `llmlingua` không, ở cùng CR thực?

| Kịch bản | Dấu hiệu | Hành động |
|---|---|---|
| **GO** | `lacc_sentence` hoặc `lacc_ppl_contrastive` thắng `llmlingua` rõ rệt trên QA/needle | Đi tiếp tuần 5–7 theo kế hoạch |
| **PIVOT** | B1 (retrieval chọn câu) ngang hoặc thắng mọi arm LACC | Đổi thông điệp paper: *"baseline đơn giản là đủ mạnh cho tiếng Việt, và đây là lý do"*. Vẫn đủ cho ICISN, và trung thực hơn |
| **PIVOT** | B2 (dịch rồi nén) thắng ở chi phí thấp hơn | Paper thành *"đo transfer gap của nén prompt trên tiếng Việt"*, nối thẳng vào Lost in Compression. Đây là kết quả cộng đồng đang cần |
| **STOP** | Mọi arm đều không hơn `random` một cách có ý nghĩa | Dừng phần phương pháp. Viết paper thuần benchmark + chẩn đoán (C1 + C3). Vẫn đủ 10 trang |

**Ghi lại quyết định bằng văn bản ngay cuối tuần 4**, kèm bảng số làm căn cứ.
Đây là pre-registration nội bộ — nó bảo vệ bạn khỏi việc hợp lý hoá ngược ở
tuần 11 khi đã mệt và sắp hết hạn.