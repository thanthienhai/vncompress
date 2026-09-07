# Wave 2 — Data pipeline: căn chỉnh theo spec và chạy teacher distillation

**Ngày:** 2026-09-06, cập nhật 2026-09-07 · **Nhánh:** `refactor/wave2-data-pipeline` (22 commit từ `wave2-implementation`)
**Trạng thái:** **XONG** — chạy full end-to-end, split cuối sạch rò rỉ, 348 test xanh

Tài liệu này ghi lại ba thứ: **đã làm gì**, **đo được gì**, và **còn lại gì**.
Spec gốc: [`docs/dataset_pipeline.md`](../../../docs/dataset_pipeline.md).
Cách chạy: [`WAVE2_HANDOFF.md`](../../../WAVE2_HANDOFF.md).

---

## 1. Xuất phát điểm

`docs/dataset_pipeline.md` mô tả pipeline `normalize → verify → split → teacher → filter`.
Code thì đọc thẳng ba shape JSON và split ở **mức record**. Doc tự khai báo khoảng cách đó
(§2.5 và các nhãn `ROADMAP`), nhưng có bốn chỗ lệch thật, không được nhãn nào che:

| Chỗ | Doc yêu cầu | Code thực tế |
|---|---|---|
| §9 split | theo source/document | `random_split` mức paragraph → **~737/1136 bài UVW bị chia đôi** giữa train/eval |
| §9 (E4/E6) | có held-out | không có split nào; E4 train trên **toàn bộ** sample mà `benchmark.py` chấm |
| §14 provenance | model name phải nằm trong metadata | checkpoint E6 không ghi teacher/ratio/seed |
| §4 teacher | pipeline distillation | chưa tồn tại |

Ngoài ra `CHECKSUMS.json` đã commit lệch với **cả 8** file dữ liệu → CI đỏ sẵn.

---

## 2. Đã làm

### 2.1. Tầng dataset chuẩn hoá (`vncompress/dataset.py`, thuần stdlib)

- Canonical `Record` (§5 core; các trường teacher reserve theo tên trong `RESERVED_FIELDS`).
- `verify_records` — deterministic check §6.1.
- `split_by_document` — **90/10 theo document**, stratify theo `(kind, source)`, xếp thứ tự bằng
  `blake2b(seed:doc_key)` chứ không RNG: tái lập được across máy, thêm document mới không xáo lại
  phần đã gán.
- `check_split_leakage` — trùng document / record id / nguyên văn context. `split_dataset.py`
  **từ chối ghi** nếu bẩn.

Thuần stdlib có chủ ý: bất biến no-leakage phải test được trong checkout trống, không cần torch.

### 2.2. Ba script pipeline (§13)

`normalize_dataset.py` → `verify_dataset.py` → `split_dataset.py`, ghi ra layout §12 `data/processed/`.
Thêm `run_pipeline.py` chạy cả chuỗi bằng một lệnh.

### 2.3. Tầng teacher distillation (§4)

- `vncompress/teacher.py` — config từ `.env`, client OpenAI-compatible qua stdlib `urllib`
  (không thêm dependency), cache trên đĩa, retry, stub dry-run.
- `vncompress/teacher_prompts.py` — prompt có version, §4.2/§4.3.
- `generate_teacher_dataset.py` — `--stage queries|compression`.
- `filter_dataset.py` — verify §6 + quarantine + merge trường §5 vào record.
- `inspect_failures.py` — trace lỗi.

**Retry theo đúng yêu cầu:** lỗi → chờ **30s** → gửi lại, tối đa **3 lần**; hết lượt thì ghi vào
`data/teacher/failures_<stage>.jsonl` (key, record id, stage, loại lỗi, HTTP status, số lần thử,
timestamp). Ngoại lệ có chủ ý: **4xx không phải rate limit thì fail ngay** — gửi lại một request
sai chỉ đốt quota và che lỗi thật.

### 2.4. Nối dây consumer

| Consumer | Trước | Sau |
|---|---|---|
| SLM/tone, LACC | `random_split` / `Subset(range(0.9n))` | `load_train_eval_texts()` theo document |
| E4 relevance probe | không split, không đo | train split + **accuracy/P-R-F1 held-out** kèm baseline all-negative |
| E6 encoder | không split, không metadata | train split + metric held-out + `distillation_meta.json` (§14) |
| `benchmark.py` | file 243 sample | eval split, cảnh báo nếu bị ép chấm full |

---

## 3. Đo được gì

### 3.1. Split (dữ liệu nguồn)

```
                         documents   train    eval   realized
benchmark/vcc_bench             42      84       9      9.7%
benchmark/wikipedia             12     135      15     10.0%
corpus/uvw-2026               1136   17960    1996     10.0%
corpus/vietnamese-poetry      2222    2000     222     10.0%
-------------------------------------------------------------
tổng                          3412   20179    2242     10.0%     Leakage: CLEAN
```

150 sample wiki của VCC-Bench gộp về **12 bài viết**; 19.956 paragraph UVW gộp về **1.136 bài**.
Chính hai tỷ lệ đó là lý do split mức record rò rỉ.

### 3.2. Verify dữ liệu nguồn (22.421 record)

```
[OK ] empty_context 0      [HIT] long_context 9
[OK ] duplicate_id 0       [HIT] degenerate_reference 166
[OK ] missing_query 0      [HIT] duplicate_context 159
[OK ] missing_reference 0  [HIT] answer_not_in_context 5
```

**`degenerate_reference: 166`** không phải bug pipeline mà là tính chất của `vcc_bench_v1.json`:
`build_vcc_bench.py` đặt `reference_answer = text`. 166/243 sample (68%) có đáp án là bản sao
nguyên văn context → ROUGE-L/BERTScore ở đó đo "còn giữ bao nhiêu chữ", tức **phạt mọi mức nén
theo định nghĩa**.

### 3.3. Teacher — stage 1 (sinh câu hỏi): **XONG**

| | |
|---|---:|
| đoạn corpus đã xử lý | 19.960 (100%) |
| record QA sau lọc | **61.344 / 61.795 (99,3%)** |
| loại: `answer_not_verbatim` / `degenerate_answer` / `duplicate_query` | 400 / 14 / 51 |
| lỗi (toàn HTTP 429) | 1.115 → **phục hồi 1.115/1.115** |
| thời gian | 124,7 phút @ 2,63 req/s (64 worker) |

Phân bố loại câu hỏi cho thấy teacher tự nhắm đúng hard case §7:
`factual 4458 · numeric 1012 · temporal 954 · negation 280 · conditional 164` (mẫu 2.235 đoạn).

**Span nguyên văn 100%. Degenerate 0.**

### 3.4. Teacher — stage 2 (nén): **XONG**

| | |
|---|---:|
| instance sinh được | **60.171 / 60.237 (99,89%)** |
| chấp nhận sau lọc | **57.529 / 60.171 (95,6%)** |
| bỏ hẳn sau 3 pass | 66 (0,11%) — nằm trong `failures_compression.jsonl` để trace |
| thời gian | ~7,5 giờ ở 2,2 req/s, 32 worker |

Loại bởi filter (§6): `over_budget` 1.776 · `too_short` 783 · `not_extractive` 81 ·
`number_altered` **2**.

Chất lượng nén trên **toàn bộ 57.529 bản** (không phải mẫu):

| ratio yêu cầu | n | target | realized | **nén thực tế** | extractive | số giữ được |
|---:|---:|---:|---:|---:|---:|---:|
| 2x | 19.782 | 65,8 | 22,8 | **7,31x** | 1,000 | 0,939 |
| 4x | 19.452 | 32,8 | 17,1 | **8,83x** | 1,000 | 0,933 |
| 8x | 18.295 | 16,9 | 12,3 | **11,55x** | 1,000 | 0,910 |

Hai điều phải đọc kỹ ở bảng này:

- **`extractive_ratio = 1,000` trên cả 57.529 bản.** Teacher trích xuất thuần, không diễn đạt
  lại. Đây là thứ làm dữ liệu này dùng được để dạy token-selection.
- **Nhãn ratio không phải mức nén thật.** Yêu cầu 2x thì thực tế nhận 7,31x. Con số mẫu
  52 instance trước đó (3,76x) quá lạc quan. Ai dùng `compression_ratio` như biến độc lập
  sẽ đo sai; phải đọc `target_tokens` / `realized_tokens`.
- `number_altered` **2 lần trên 57.529** — teacher gần như không bịa số. Nhưng
  `numbers_preserved` chỉ 0,91–0,94: nó *bỏ* số thường xuyên (đúng, khi câu hỏi không hỏi tới),
  chỉ là không *sửa* số.

### 3.5. Split cuối (141.294 record)

| | |
|---|---:|
| record | 127.189 train / 14.105 eval |
| tài liệu | 2.796 / 616 |
| đơn vị chia | 3.402 nhóm (từ 3.412 tài liệu — 10 bị gộp do trùng nội dung) |
| tỷ lệ eval thật | **0,0998** |
| leakage §9 | **CLEAN** |

Theo nguồn:

| nguồn | train | eval |
|---|---:|---:|
| uvw-2026 | 63.242 | 7.889 |
| teacher-synth | 56.152 | 5.192 |
| vietnamese-poetry | 7.188 | 892 |
| wikipedia | 321 | 101 |
| vcc_bench | 286 | 31 |

### 3.6. Tác động lên E4

| | trước | sau |
|---|---:|---:|
| cặp (context, answer) để train | **153** | **61.344** |
| eval độc lập, tổng | 24 | 132 |
| eval độc lập, **thực sự dùng được** | **3** | **6** |

Dòng cuối là con số quan trọng nhất trong cả báo cáo này, xem §5.

## 4. Lỗi đã tìm và sửa

### 4.1. Lỗi trong dữ liệu / thiết kế cũ

1. **Rò rỉ document-level** ở `run_slm_training` / `run_lacc_training` (§9).
2. **E4 train trên chính bộ nó được chấm** — 169 sample `vcc_bench_v1.json`.
3. **`CHECKSUMS.json` lệch cả 8 file** — CI đỏ sẵn trên `wave2-implementation`.
4. **E6 checkpoint không ghi provenance** (§14).

### 4.2. Lỗi tôi tạo ra và sửa trong lúc chạy

| Lỗi | Triệu chứng | Cách sửa |
|---|---|---|
| **Cache lưu response hỏng** | retry đọc lại bản rỗng từ đĩa → 6 record fail 3 pass trong **0,0 phút** mà không hỏi model lần nào. Throughput 0,57 req/s, ETA **29 giờ** | không cache response rỗng; xoá entry trước khi retry; response rỗng thì hỏi lại prompt gốc → **2,26 req/s, ETA 4 giờ**, phục hồi 1.115/1.115 |
| **Ngân sách nén coi là sàn** | loại đúng output tốt nhất: needle 6.973 → 26 từ | §4.3 định nghĩa là **trần** → sàn tuyệt đối `--min-words` |
| **Tolerance chỉ theo %** | ở 8x target ~11 từ, 25% chưa tới 3 từ → loại do làm tròn | thêm `--budget-slack` tuyệt đối |
| **`number_dropped`** | trường `numbers` liệt kê số trong **nguồn**, không phải số phải giữ. `"Nam Phi biểu quyết trắng để bảo vệ chế độ apartheid"` bị loại vì rơi số `13` mà câu hỏi không hỏi | đổi thành `number_altered`; tỷ lệ giữ ghi vào `quality.numbers_preserved`, không dùng để loại |
| **Rò rỉ ở bước merge** | câu hỏi teacher mang nguyên văn đoạn gốc nhưng `doc_key` khác → split có thể tách đôi | thêm `Record.split_group` |
| **Eval bị teacher chi phối** | 482 teacher vs 24 độc lập, gộp chung thì số độc lập tan biến | tách hai file, gộp phải cố ý làm |
| **`pgrep -f` / `pkill -f` tự khớp** | 3 lần trong phiên; một lần làm pipeline **đứng yên 1,5 giờ** | bỏ hẳn pattern matching, chỉ dùng PID |
| **Tuple exception thiếu lỗi transport** | `RemoteDisconnected` là `ConnectionResetError`, **không phải** `URLError`, nên lọt khỏi vòng retry 30s×3 và **giết run ở phút 446** | nới thành `OSError` + `http.client.HTTPException`; test parametrize 7 kiểu lỗi thật, 5/7 fail với bản cũ |
| **Một task hỏng giết cả run** | `guarded()` cũng bắt tuple hẹp → exception thoát ra `as_completed` và abort 44.900 instance đã xong | bắt `BaseException`, ghi failure log, chỉ re-raise `KeyboardInterrupt`/`SystemExit` |
| **Đơn vị chia là doc id, không phải nội dung** | VCC-Bench `long_document_qa` **chính là** bài UVW nhưng doc_key khác → train đoạn văn rồi eval câu hỏi trên đúng đoạn đó; thêm 1 đoạn thư mục dùng chung 4 bài | `link_documents_by_content` union-find gộp doc_key trùng nội dung; manifest ghi `n_split_groups` để việc gộp không âm thầm |

Tỷ lệ chấp nhận bản nén qua ba lần sửa filter: **80,6% → 89,9% → 95,5%** (cuối cùng **95,6%** trên toàn bộ 60.171 row).
Cả ba đều là giả định của tôi về output của teacher mà **chưa từng đối chiếu với output thật**.

---

## 5. Còn lại

### 5.1. Chặn đường: bộ eval độc lập chỉ có **6 sample dùng được**

Đây là vấn đề nghiêm trọng nhất và nó **không** được lần chạy này giải quyết.

`vcc_bench_eval.json` có 132 sample, nhưng 126 trong số đó có `reference_answer` là
bản sao gần nguyên văn của `context`. Chấm điểm trên chúng thì mọi metric suy biến
thành đo độ trùng văn bản — bất kỳ phương pháp nào giữ nhiều token cũng "thắng".

| | tổng | dùng được | theo task |
|---|---:|---:|---|
| trước lần chạy này | 24 | 3 | agent 2, needle 1 |
| **sau** | 132 | **6** | agent 3, compression 2, needle 1 |

Số dùng được tăng gấp đôi, nhưng **6 mẫu không phải là một benchmark**. Không con số
nào báo cáo trên bộ này có ý nghĩa thống kê.

Lối thoát (không có đường tắt): nạp **UIT-ViQuAD** (§3) làm nguồn QA độc lập có câu
trả lời ngắn thật, rồi trả VCC-Bench về eval-only:

```bash
python scripts/split_dataset.py --input data/processed/records_all.jsonl \
    --eval-only-source vcc_bench --eval-only-source wikipedia
```

### 5.2. Việc còn phải làm

| Ưu tiên | Việc | Vì sao |
|---|---|---|
| **P0** | **Nạp UIT-ViQuAD, trả VCC-Bench về eval-only** | Xem §5.1. Chặn mọi kết quả báo cáo được |
| **P0** | **Chưa có student nào học từ `compressed_text`** | 57.529 bản nén chất lượng cao đang nằm không. E4 vẫn dùng span-overlap, E6 vẫn dùng perplexity — cả hai đều yếu hơn supervision vừa sinh ra |
| P1 | Train lại E4/E6 trên dữ liệu mới | E4 đi từ 153 → 61.344 cặp; số cũ không còn để so sánh |
| P1 | Trace 66 instance nén hỏng + 26 `TeacherOutputError` ở stage 1 | `inspect_failures.py --stage compression`. Phần lớn là "empty response" lặp lại — nghi context quá dài hoặc nội dung bị model từ chối |
| P1 | `generate_importance_dataset.py`, `generate_preference_dataset.py` | `token_labels`, `hard_negative`, `preference` vẫn chưa sinh |
| P1 | §6.2 verifier LLM, §6.3 teacher agreement | Chưa có |
| P2 | Làm rõ trường `numbers` trong prompt | Hiện mơ hồ giữa "số trong nguồn" và "số phải giữ"; sửa prompt sẽ vô hiệu hoá toàn bộ cache |
| P2 | Nguồn P0 còn thiếu ở §3 | ViLegalText, VNFinsQA, ViSecQA chưa có script |

### 5.3. Cảnh báo khi đọc kết quả

- **Chỉ 6/132 sample trong `vcc_bench_eval.json` dùng được.** Xem §5.1 trước khi báo cáo bất kỳ số nào.
- **Không gộp** `vcc_bench_eval.json` với `vcc_bench_eval_synthetic.json` (5.192 sample teacher sinh).
- `compression_ratio: 2` **không** phản ánh mức nén thật (thực tế 7,31x); đọc
  `target_tokens` / `realized_tokens`.
- `vcc_bench_train.json` (99MB) và `vcc_bench_eval_synthetic.json` (9MB) **không nằm trong git** —
  dựng lại bằng `scripts/run_pipeline.py`, đối chiếu sha256 trong `split_manifest.json`.
- Adapter train **trước** khi đổi sang split theo document có `val_split.json` không tái lập được;
  `train_probe_control.py` sẽ từ chối chạy và báo lý do.

---

## 6. Kiểm chứng

- **348 test** pass (thêm ~99 test mới), CI xanh cả 6 bước.
- CI dựng lại pipeline vào thư mục tạm và **fail nếu split rò rỉ**.
- Split là hàm thuần của `(nội dung, seed)` — chạy lại cho kết quả byte-identical.
- `split_manifest.json` ghi policy, seed, tỷ lệ thực tế theo stratum, **toàn bộ eval document key**,
  sha256 từng output, và kết quả leakage check.
