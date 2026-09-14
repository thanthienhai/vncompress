# Xây dựng VCC-Bench v2

**Sản phẩm bàn giao:** `data/benchmark/vcc_bench_v2.json` đã freeze + checksum
**Cần GPU:** Không
**Đây là việc quyết định toàn bộ paper.** Mọi con số ở 10 tuần sau đều đọc từ file này.

---

## 1. Tại sao phải làm lại benchmark trước

VCC-Bench v1 có một lỗi thiết kế làm hỏng kết luận trung tâm của Wave 1.

Công thức tone trong `vncompress/linguistics.py` (dòng 216):

```
w_tone(t) = 1 + α·ρ(t)·(1 + β·ν(t)/6)      ρ = mật độ ký tự mang dấu thanh
```

Token không có dấu thanh → ρ = 0 → `w_tone = 1.0`, đúng bằng sàn tuyệt đối.

Kiểm chứng trên chính `vcc_bench_needle_in_haystack.json`:

| Chỉ số đo được | Giá trị |
|---|---|
| Token payload của needle nằm ở sàn `w_tone = 1.000` | **16/16** |
| Percentile trung bình của token-needle trong context | **0.0%** |
| Tỉ lệ token trong context **có** dấu thanh | **70.8%** |

Toàn bộ 9 needle của v1 là chuỗi alphanumeric: `VIETCOMPRESS2026_SECURE`,
`192.168.55.107`, `35,000,000`, `SUMMER2026`, `500mg`, `09:00`.

**Cơ chế thất bại:** nén 2× giữ 50% token, nhưng 70.8% token có dấu được xếp
hạng cao hơn. Budget cạn trước khi chạm nhóm không dấu — nơi chứa toàn bộ
needle. Needle bị xoá **tất định**, ở mọi mẫu, không phải ngẫu nhiên.

Vì vậy con số 2.3% **không đo** điều mà báo cáo Wave 1 nói nó đo. Nó chỉ chứng
minh rằng heuristic tone là một bộ phát hiện chính tả tiếng Việt, và needle theo
thiết kế không có chính tả tiếng Việt.

Không có nhóm đối chứng needle **có dấu thanh**, bạn không thể tách "tone là tín
hiệu xấu" khỏi "needle không dấu". Đó là lỗ hổng mà reviewer sẽ tìm ra, và cũng
là cơ hội: sửa nó thì bạn có đóng góp C1 gần như miễn phí.

Vấn đề phụ nhưng nghiêm trọng: v1 chỉ có **9 mẫu** needle. Không đủ để phát biểu
bất cứ điều gì có ý nghĩa thống kê.

---

## 2. Việc 1 — Thiết kế lại tập needle (ưu tiên cao nhất)

Từ 9 mẫu lên **120 mẫu**, chia ba nhóm đối chứng 40 mẫu mỗi nhóm.

| Nhóm | Nội dung needle | Ví dụ | Mục đích |
|---|---|---|---|
| **A — không dấu** | mã, số, IP, ngày giờ | `Mã kích hoạt là VNC-2026-X7K.` | Tái hiện điều kiện Wave 1 |
| **B — có dấu** | tên người, địa danh, cụm từ thuần Việt | `Người phụ trách dự án là bà Trần Thị Mỹ Linh.` | **Nhóm đối chứng còn thiếu** |
| **C — hỗn hợp** | một câu chứa cả hai | `Hợp đồng HĐ-2026-451 do ông Nguyễn Quốc Thắng ký.` | Đo tương tác giữa hai loại |

**Yêu cầu thiết kế nhóm B.** Payload phải là cụm từ tiếng Việt **có dấu thanh
thật**, không phải từ tiếng Việt không dấu ngẫu nhiên. Câu hỏi và
`reference_answer` phải đòi đúng cụm đó, không đòi thông tin quanh nó. Nếu
`reference_answer` của nhóm B lẫn nhiều từ đệm không dấu thì bạn lại làm loãng
đúng thứ mình muốn đo.

**Ba biến phải kiểm soát chặt:**

1. **Vị trí chèn** rải đều đầu / giữa / cuối, tỉ lệ bằng nhau trong cả ba nhóm.
   Ghi lại để phân tích hiệu ứng lost-in-the-middle về sau.
2. **Độ dài context đồng nhất** giữa ba nhóm (v1 đang dao động 61k–67k ký tự).
   Nếu lệch, khác biệt giữa A và B có thể do độ dài chứ không do dấu thanh — và
   toàn bộ C1 sụp.
3. **Cùng một tập văn bản nền (haystack)** cho cả ba nhóm. Chỉ đổi needle, giữ
   nguyên mọi thứ khác. Đây là điều kiện để gọi nó là "controlled study".

---

## 3. Việc 2 — Bổ sung subset cho long_document_qa

Giữ nguyên 160 mẫu hiện có, thêm hai subset nhỏ để chuẩn bị cho phần thảo luận
về structural completeness:

- **Multi-hop** (~30 mẫu): đáp án đòi nối 2–3 câu ở các đoạn khác nhau.
- **Referential** (~30 mẫu): đáp án phụ thuộc vào tiền tố — đại từ, tên viết
  tắt, hoặc thực thể đã giới thiệu ở đoạn trước.

Hai subset này không phải để claim đóng góp, mà để bạn có chỗ quan sát xem nén
theo câu (`lacc_sentence`) có thật sự bảo toàn chuỗi bằng chứng tốt hơn nén theo
token không. Nếu không kịp, ưu tiên **multi-hop** và bỏ referential — multi-hop
dễ sinh tự động hơn.

---

## 4. Việc 3 — Schema trường mới

Mỗi mẫu needle phải mang thêm các trường sau để phân tích theo nhóm ở tuần 3–4:

```json
{
  "sample_id": "needle_v2_0007",
  "task": "needle_in_haystack",
  "needle_group": "B",
  "needle_has_diacritic": true,
  "needle_payload": "Trần Thị Mỹ Linh",
  "needle_char_len": 16,
  "insert_position": "middle",
  "insert_char_offset": 31204,
  "haystack_id": "hay_0012",
  "char_length": 62000
}
```

`haystack_id` là trường quan trọng nhất trong nhóm này — nó cho phép bạn ghép
cặp A/B/C trên cùng văn bản nền và chạy paired test ở tuần 10.

---

## 5. Việc 4 — Sửa ba lỗi chặn trong repo

Ba lỗi này sẽ làm hỏng mọi lần chạy ở tuần 3 nếu không sửa ngay bây giờ.

| Lỗi | Vị trí | Cách sửa |
|---|---|---|
| `vcc_bench_v2.json` được nhiều script trỏ tới nhưng **không tồn tại** | `scripts/train_relevance_probe.py`, `WAVE2_HANDOFF.md` | Tạo file thật ở tuần này |
| `max_encoder_len = 512` nhưng PhoBERT chỉ chịu 256, script train lại đặt `--max-length 256` | `vncompress/encoder_compression.py` vs `scripts/train_encoder_compressor.py` | Thống nhất **một** hằng số, khai báo ở `config.py` |
| Default `--data-path` của relevance probe trỏ thẳng vào **benchmark đánh giá** | `scripts/train_relevance_probe.py` | Đổi default sang `training_corpus_v1.json`. Train không bao giờ được đọc benchmark |

Lỗi thứ ba là nhiễm chéo train/test. Nếu để nguyên và chạy ở tuần 8, mọi con số
của arm học đều vô giá trị và không cứu được.

---

## 6. Việc 5 — Freeze và ghi nguồn gốc

Cập nhật `data/benchmark/CHECKSUMS.json` và `PROVENANCE.md` với:

- checksum của `vcc_bench_v2.json`
- nguồn văn bản nền, ngày crawl, license
- model và prompt version dùng để sinh câu hỏi (nếu có dùng LLM)
- seed ngẫu nhiên cho vị trí chèn needle
- danh sách `haystack_id` đã dùng, để tuần 8 loại trừ khỏi tập train

Sau khi freeze, **không sửa benchmark nữa**. Nếu phát hiện lỗi ở tuần 6 thì ghi
chú là v2.1 và chạy lại toàn bộ, đừng sửa lặng lẽ.

---

## 7. Tiêu chí hoàn thành

- [ ] `vcc_bench_v2.json` có ≥120 mẫu needle chia đều ba nhóm A/B/C
- [ ] Ba nhóm dùng chung tập haystack, độ dài context lệch nhau <5%
- [ ] Mỗi mẫu có đủ trường `needle_group`, `haystack_id`, `insert_position`
- [ ] long_document_qa có thêm subset multi-hop
- [ ] Ba lỗi chặn ở §5 đã sửa, `pytest -q` xanh
- [ ] CHECKSUMS + PROVENANCE cập nhật, benchmark đã freeze

---

## 8. Bẫy cần tránh

- **Đừng sinh needle nhóm B bằng cách bỏ dấu ngược lại từ nhóm A.** Nó tạo ra
  chuỗi vô nghĩa và model sinh sẽ xử lý khác hẳn.
- **Đừng để câu hỏi nhóm B chứa sẵn đáp án dạng lặp từ.** Nếu câu hỏi là "bà
  Trần Thị Mỹ Linh phụ trách gì?" thì bạn đã lộ payload vào query, và mọi
  compressor query-aware sẽ thắng một cách giả tạo.
- **Đừng tăng số mẫu bằng cách nhân bản haystack.** 120 mẫu trên 10 văn bản nền
  không phải 120 quan sát độc lập.