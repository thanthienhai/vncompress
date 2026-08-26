# Hướng dẫn: xây dataset, huấn luyện, đánh giá SLM (external scorer + tone probe)

Tài liệu này gộp lại toàn bộ quy trình cho SLM tiếng Việt nhỏ dùng làm
**external perplexity scorer** ở mode `lightweight` của LACC (xem
[README.md](../README.md#kiến-trúc-phần-cứng-3-mức)) — mở rộng dataset
training, huấn luyện LoRA + tone probe, đánh giá, và cách đọc kết quả eval
cho đúng. Đây **không phải** protocol của VCC-Bench (bộ đánh giá thuật
toán nén LACC) — xem [`docs/benchmark.md`](benchmark.md) cho việc đó.
Ghi kết quả từng lần chạy vào
[`docs/training_eval_report_template.md`](training_eval_report_template.md).

## Pipeline tổng quan

```
scripts/build_training_corpus.py   →  vcc_bench_data/training_corpus_v1.json
                                              │
                                              ▼
run_train_slm.py --train-data-path ...  →  trained_slm/final/  (LoRA adapter + tokenizer + val_split.json)
                                         →  trained_slm/tone_probe.pt
                                              │
                                              ▼
evaluate_slm.py --adapter-dir ...       →  NLL / Perplexity / Tone accuracy
evaluate_slm.py --no-adapter            →  baseline (base model, không LoRA) để so sánh
```

`scripts/build_viquad_eval.py` là một nhánh riêng, không nằm trong pipeline
training — xem mục 3.

---

## 1. Xây / mở rộng dataset training

`run_train_slm.py` cần văn bản tiếng Việt thô (không cần nhãn) để huấn
luyện LoRA (mục tiêu causal LM) + tone probe (mục tiêu phân loại thanh
điệu theo từng token). Nguồn mặc định cũ (`vcc_bench_data/wikipedia_vi_raw.json`,
393 đoạn) **quá nhỏ** — tone probe học gần như random (~19-20% accuracy
trên các thanh có dấu, xem mục 5). `scripts/build_training_corpus.py`
thay thế bằng corpus lớn hơn, trộn 2 nguồn Hugging Face:

| Nguồn | Vai trò | License | Ghi chú truy cập |
|---|---|---|---|
| `undertheseanlp/UVW-2026` | Chính (bulk, ~90%) | CC BY-SA 4.0 | Public, không cần xin quyền |
| `bigscience-data/roots_vi_vietnamese_poetry` | Augmentation (~10%), tăng mật độ/đa dạng thanh điệu | MIT | **Gated** — phải bấm "Agree and access repository" trên trang dataset (đăng nhập đúng tài khoản dùng cho `huggingface-cli login`), có thể mất vài phút để quyền lan truyền |

Chi tiết provenance đầy đủ (cách build, preprocessing, license) ở
[`vcc_bench_data/PROVENANCE.md`](../vcc_bench_data/PROVENANCE.md).

### Lệnh cơ bản

```bash
python scripts/build_training_corpus.py
```

Mặc định: 5,000 đoạn từ UVW-2026 (lọc `quality_score >= 7`) + ~10% Poetry
(ghép nhiều bài thơ ngắn thành đoạn ≥200 ký tự để không bị lọc bởi ngưỡng
`>200 ký tự` mà `run_training.load_training_texts()` áp dụng) → ghi ra
`vcc_bench_data/training_corpus_v1.json`.

### Các cờ quan trọng (`--help` để xem đầy đủ)

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--uvw-n` | 5000 | Số đoạn lấy từ UVW-2026 — tăng để có corpus lớn hơn |
| `--poetry-ratio` | 0.10 | Tỉ lệ Poetry trong corpus cuối (0 = tắt) |
| `--min-quality-score` | 7 | Ngưỡng `quality_score` (1-10) của UVW-2026 |
| `--skip-poetry` | off | Bỏ qua Poetry hoàn toàn (dùng khi chưa có gated access) |
| `--seed` | 42 | Seed cho shuffle/sample — cùng seed + cùng tham số = cùng corpus |
| `--output` | `vcc_bench_data/training_corpus_v1.json` | Đường dẫn ghi ra |

### Tăng quy mô corpus

`undertheseanlp/UVW-2026` có 894,579 bài trong train split — dư sức tăng
`--uvw-n` xa hơn mức mặc định. Corpus càng lớn thì **số optimizer step
mỗi epoch càng nhiều** → thời gian train tăng gần như tuyến tính theo số
đoạn. Trên GPU yếu (vd. GTX 1060 6GB), cân nhắc giảm `--epochs` khi tăng
`--uvw-n` để giữ thời gian train hợp lý:

```bash
# ~4x corpus hiện tại, giảm epochs để bù thời gian train
python scripts/build_training_corpus.py --uvw-n 20000
python run_train_slm.py --train-data-path vcc_bench_data/training_corpus_v1.json \
  --batch-size 1 --max-length 128 --grad-accum 8 --epochs 2
```

Công thức ước lượng số step (xem mục 2 để hiểu `updates_per_epoch`):
tăng `--uvw-n` N lần → số step mỗi epoch tăng ~N lần (nếu `--epochs`
không đổi, thời gian train cũng tăng ~N lần).

### Chạy lại `build_training_corpus.py` sẽ ghi đè

Mỗi lần chạy **ghi đè** `training_corpus_v1.json` cũ (không merge). Muốn
giữ lại corpus cũ để so sánh, đổi tên file cũ hoặc dùng `--output` khác
trước khi chạy lại.

### Commit corpus vào git? (tuỳ chọn)

`training_corpus_v1.json` **không tự động bị commit** (không có trong
`.gitignore`, nhưng cũng không được add tự động). License các nguồn rõ
ràng (CC-BY-SA 4.0 + MIT) nên an toàn để commit nếu muốn, nhưng phải:

```bash
git add vcc_bench_data/training_corpus_v1.json
python scripts/checksum_datasets.py --write   # cùng lúc, KHÔNG tách riêng
git add vcc_bench_data/CHECKSUMS.json
```

CI chạy `scripts/checksum_datasets.py` ở chế độ verify — nếu
`CHECKSUMS.json` ghi checksum cho một file không thực sự có trong git,
CI sẽ fail. Nếu chỉ generate để chạy local (không commit), test
`test_checksums_manifest_matches_files_on_disk` sẽ báo lỗi cục bộ (vô
hại, chỉ là cảnh báo checksum chưa đăng ký) — bỏ qua nếu không định
commit.

---

## 2. Huấn luyện (`run_train_slm.py`)

```bash
python run_train_slm.py --train-data-path vcc_bench_data/training_corpus_v1.json
```

Yêu cầu **GPU NVIDIA (CUDA) bắt buộc** — không hỗ trợ CPU. Model mặc định
`chronopt-research/vietnamese-gpt2-base` (137M).

### Cấu hình theo phần cứng

| GPU | Cờ khuyến nghị |
|---|---|
| T4 16GB (mặc định script đã tune sẵn) | không cần thêm cờ |
| ~6GB (vd. GTX 1060) | `--batch-size 1 --max-length 128 --grad-accum 8` |

Nếu thấy lỗi `CUDA out of memory`, hạ tiếp `--batch-size`/`--max-length`
hoặc tăng `--grad-accum`.

### Ước lượng số step / thời gian còn lại

Script không tự in tổng số step dự kiến. Công thức (theo đúng logic
trong `run_train_slm.py`):

```
train_n            = round(len(dataset) * 0.9)          # 90% train, 10% held-out
updates_per_epoch  = ceil(ceil(train_n / batch_size) / grad_accum)
planned_steps      = updates_per_epoch * epochs          # (trừ khi có --max-steps > 0)
```

Ví dụ thực tế đã chạy: corpus 5,556 đoạn (`--batch-size 1 --max-length 128
--grad-accum 8`, `--epochs 3` mặc định) → `train_n≈5000` →
`updates_per_epoch = ceil(5000/8) = 625` → `planned_steps = 1875`. Log in
mỗi 10 step (`step=... lm=... tone=... vram=...`) — so current step với
`planned_steps` tính được ở trên để biết còn bao lâu.

### Output

- `trained_slm/final/` — LoRA adapter + tokenizer + **`val_split.json`**
  (tập held-out 10% đã dùng lúc train, lưu lại để `evaluate_slm.py` dùng
  đúng lại — quan trọng, xem mục 4).
- `trained_slm/tone_probe.pt` — trọng số tone probe (`PhonologicalConsistencyLoss`).

### Bug đã gặp và đã sửa: `CUDA error: srcIndex < srcSelectDimSize`

Một số checkpoint community (vd. `chronopt-research/vietnamese-gpt2-base`)
có tokenizer khai báo nhiều hơn 1 token so với số hàng của embedding
matrix trong checkpoint (`<|endoftext|>` ở id 50257, embedding matrix chỉ
có 50257 hàng — id hợp lệ 0..50256). Vì token này là EOS/BOS/pad, nó xuất
hiện trong **hầu hết mọi batch** → tra bảng embedding vượt giới hạn →
crash ngay từ step đầu tiên trên GPU (không phải lỗi thiếu VRAM, dù triệu
chứng dễ nhầm với OOM). Đã vá bằng `model.resize_token_embeddings(len(tokenizer))`
trong cả `run_train_slm.py` và `evaluate_slm.py` (no-op nếu kích thước đã
khớp) — nếu đổi sang base model khác và gặp lại lỗi
`srcIndex < srcSelectDimSize`, đây chính là nguyên nhân cần kiểm tra
trước tiên.

---

## 3. Xây dataset eval thật từ UIT-ViQuAD2.0 (tuỳ chọn)

Bổ sung một task QA **thật** (không synthetic) vào VCC-Bench, để đo ảnh
hưởng của nén lên độ chính xác QA downstream — khác mục đích với dataset
training ở mục 1 (dataset này **chỉ dùng để eval thuật toán nén**, không
liên quan `run_train_slm.py`).

```bash
python scripts/build_viquad_eval.py
python run_benchmark.py --data-path vcc_bench_data/vcc_bench_uit_viquad_qa.json --config configs/example_experiment.json
```

- Chỉ dùng split `test` (7,301 mẫu) của `taidng/UIT-ViQuAD2.0`, không
  bao giờ dùng để train (nguyên tắc cứng, xem `PROVENANCE.md`).
- Samples gắn `task="long_document_qa"` — **trùng tên** với task synthetic
  trong `vcc_bench_v1.json` một cách cố ý, vì `VCCBenchConfig.tasks` là
  danh sách cố định không có cờ `--tasks` để mở rộng; tên khác sẽ bị lọc
  bỏ âm thầm, 0 mẫu được đánh giá. Không bị lẫn dữ liệu vì mỗi lần chạy
  chỉ load một file qua `--data-path`.
- File output **gitignored**, không commit — license của UIT-ViQuAD2.0
  chưa được xác nhận rõ trên trang Hugging Face.

---

## 4. Đánh giá (`evaluate_slm.py`)

### Đánh giá adapter đã train

```bash
python evaluate_slm.py --adapter-dir trained_slm/final --tone-probe trained_slm/tone_probe.pt
```

Nếu `trained_slm/final/val_split.json` tồn tại (các lần train gần đây có
lưu), script tự dùng đúng tập held-out đó — **không cần** truyền
`--train-data-path`/`--max-length`. Nếu thiếu file này (checkpoint cũ),
script phải tự dựng lại split từ `--train-data-path` + `--max-length`,
và **bắt buộc truyền đúng giá trị đã dùng lúc train** — sai sẽ cho ra
tập validation không khớp lúc train (có thể lẫn cả dữ liệu train), số
liệu không đáng tin.

### So sánh với baseline (base model, không LoRA)

```bash
python evaluate_slm.py --adapter-dir trained_slm/final --no-adapter
```

Dùng **đúng** `val_split.json` của adapter đó nhưng bỏ qua LoRA — cho
perplexity của base model gốc trên cùng tập held-out, để biết fine-tuning
có thực sự cải thiện hay không. Tone accuracy bị bỏ qua ở chế độ này (tone
probe được train gắn với hidden states đã qua LoRA, áp lên base model
thuần không có ý nghĩa).

### Output

```
Validation texts: <N>
LM validation loss (NLL): <...>
Perplexity: <...>
Tone accuracy (all tokens): <...>
Tone accuracy (marked tones only): <...>  (<M> tokens)
Majority-class baseline (always predict ngang): <...>
```

---

## 5. Cách đọc kết quả eval cho đúng

### ⚠️ Trần của tone probe là 100%, và nó miễn phí

Nhãn thanh điệu được `VietnameseToneDataset` sinh ra bằng
`analyzer.get_dominant_tone(tokenizer.decode([token_id]))` — tức là **hàm
xác định của riêng token id**, không phụ thuộc ngữ cảnh. Đã kiểm chứng
trên split 2,218 văn bản: 13,727 token id duy nhất, **0 id nào** ứng với
nhiều hơn một nhãn thanh.

Hệ quả: một **bảng tra cứu token_id → thanh**, dựng từ tokenizer, **không
cần huấn luyện, không cần dữ liệu**, đạt **100.00%** trên cả "all tokens"
lẫn "marked tones only". Probe đã train đạt 92.94% — tức là **thấp hơn**
trần miễn phí này ~7 điểm.

Vì vậy con số tone accuracy **không** chứng minh "model học được thanh
điệu tiếng Việt". Nó đo một thứ khác (vẫn hợp lệ, nhưng phải phát biểu
đúng): **bao nhiêu thông tin thanh điệu còn sống sót và đọc được tuyến
tính từ hidden states** — đây là bài toán *probing* (Hewitt & Liang 2019),
không phải bài toán dự đoán thanh điệu. Probe hiện tại là MLP 2 lớp
(768→192→7, ~150K tham số), đủ sức chứa để nhớ khá nhiều, nên càng cần
control.

**Bắt buộc báo cáo kèm 2 mốc**, nếu không con số 92.94% không diễn giải
được:

1. **Trần lookup = 100%** — `evaluate_slm.py` đã tự in ra.
2. **Probe trên base model đóng băng** (chưa có sẵn — xem mục 7): train
   đúng kiến trúc probe đó trên hidden states của base model *chưa* LoRA.
   Nếu base + probe cũng đạt ~92% thì `--lambda-tone` **không đóng góp
   gì**, và luận điểm "tone-aware training" sụp đổ.

### Tone accuracy: luôn nhìn "marked tones only", không nhìn "all tokens"

Tiếng Việt có 6 thanh; thanh "ngang" (không dấu) luôn là lớp đa số áp
đảo. Vì vậy:

- **"all tokens"** bị thổi phồng bởi việc đoán đúng các token ngang
  (không cần học gì cũng đạt 40-90% tuỳ tỉ lệ ngang trong tập).
- **"marked tones only"** (5 thanh có dấu) mới phản ánh model có học
  được tín hiệu thanh điệu thật hay không. Random guessing giữa 5 thanh
  ≈ **20%**.
- **"Majority-class baseline"** in ra là baseline "luôn đoán ngang" tính
  trên toàn bộ token (all tokens) — dùng để thấy con số "all tokens" bị
  thổi phồng bao nhiêu, không phải để so trực tiếp với "marked tones
  only" (baseline "luôn đoán ngang" trên riêng tập marked luôn là 0%,
  vì theo định nghĩa các token đó không phải ngang).

Phân bố lớp trên split 2,218 văn bản (rất mất cân bằng):

| thanh | tỉ lệ | | thanh | tỉ lệ |
|---|---|---|---|---|
| ngang | 46.15% | | nặng | 12.78% |
| sắc | 16.73% | | hỏi | 7.38% |
| huyền | 13.83% | | **ngã** | **3.13%** |

`ngã` chỉ chiếm 3.13% — model có thể **bỏ qua hoàn toàn** lớp này mà chỉ
mất ~3% accuracy, trong khi F1 của `ngã` tụt về ~0. Và `hỏi`↔`ngã` chính
là cặp dễ lẫn kinh điển trong tiếng Việt. Vì vậy `evaluate_slm.py` giờ in
thêm **macro-F1** (trung bình không trọng số qua các lớp) và **confusion
matrix 6×6** — đây mới là con số nên đưa vào báo cáo, không phải accuracy.

**Quy tắc đọc**: "marked tones only" càng cao hơn mức random ~20% thì
model càng học được tín hiệu thanh điệu thật. Ví dụ thực tế trong dự án:

| Corpus training | Marked tones accuracy | Đánh giá |
|---|---|---|
| 393 đoạn (Wikipedia, split lệch — không đại diện) | 19.64% | ≈ random, gần như không học được gì |
| 5,556 đoạn (UVW-2026 + Poetry, split đúng) | 82.24% | Học được tín hiệu thanh điệu rõ ràng |

### Perplexity: chỉ so sánh khi cùng validation split, và cần baseline

Hai cái bẫy dễ gặp:

1. **So sánh perplexity giữa các lần train trên corpus khác nhau là vô
   nghĩa** — corpus đa dạng/khó hơn (nhiều chủ đề, có thơ ca) tự nhiên
   cho perplexity cao hơn, không đồng nghĩa model tệ đi. Ví dụ thực tế:
   perplexity tăng từ 51.57 (corpus nhỏ, split lệch — số liệu không đáng
   tin) lên 124.26 (corpus lớn, split đúng) — nhìn thoáng qua tưởng tệ
   đi, nhưng thực chất số cũ không đáng so sánh (split sai) và corpus mới
   khó hơn hẳn.
2. **Không có baseline thì không biết fine-tuning có giúp ích không.**
   Luôn chạy thêm `--no-adapter` trên **cùng** `val_split.json` để so
   sánh công bằng. Ví dụ thực tế: base model perplexity 159.81 vs LoRA
   fine-tuned 124.26 trên cùng 555 văn bản held-out → LoRA giúp giảm
   perplexity 22.2% — xác nhận fine-tuning có tác dụng thật, không chỉ
   tone probe học tốt mà LM objective cũng cải thiện.

**Quy tắc đọc**: chỉ so sánh Perplexity giữa 2 lần chạy `evaluate_slm.py`
khi (a) cùng `val_split.json` (cùng adapter-dir, hoặc chạy `--no-adapter`
trên chính adapter-dir đó để lấy baseline khớp split), và (b) đã kiểm tra
`Validation texts: <N>` khớp nhau giữa 2 lần chạy.

### Chênh lệch perplexity có ý nghĩa thống kê không?

Một chênh lệch trần trụi ("110.76 vs 148.10") chưa phải một khẳng định
cho đến khi biết nó có sống sót qua resampling hay không. Vì hai lần chạy
chấm **đúng cùng một tập văn bản**, phép so sánh đúng là **theo cặp**:

```bash
python evaluate_slm.py --adapter-dir trained_slm/final --no-adapter \
  --dump-per-sample results_slm/base.json
python evaluate_slm.py --adapter-dir trained_slm/final \
  --tone-probe trained_slm/tone_probe.pt --dump-per-sample results_slm/lora.json
python scripts/compare_slm_runs.py results_slm/base.json results_slm/lora.json
```

Kết quả gồm: khoảng tin cậy 95% (bootstrap theo cặp, resample **văn bản**
chứ không phải token) cho ΔNLL và cho tỉ số perplexity, kiểm định Wilcoxon
signed-rank, và tỉ lệ văn bản mà model tốt hơn. Script **từ chối so sánh**
nếu hai dump có `split_fingerprint` khác nhau — bảo vệ đúng cái bẫy
`val_split.json` bị ghi đè mô tả ở trên.

### Một artifact cần biết: ghép thơ ngắn có thể thổi phồng perplexity

`build_training_corpus.py` ghép nhiều bài thơ ngắn không liên quan lại
với nhau (nối bằng `\n\n`) để đạt ngưỡng >200 ký tự. Việc này tạo ra các
điểm "đứt gãy ngữ nghĩa" giữa các bài thơ ghép trong cùng một đoạn huấn
luyện — dự đoán câu đầu bài thơ tiếp theo từ câu cuối bài thơ trước gần
như không thể, làm tăng perplexity tại các điểm nối một cách nhân tạo
(không phản ánh model kém). Nếu muốn tách biệt ảnh hưởng này, có thể sửa
`evaluate_slm.py`/script riêng để tính NLL theo từng `source` (`uvw-2026`
vs `vietnamese-poetry`) thay vì gộp chung — hiện chưa có sẵn, cần code
thêm nếu cần phân tích sâu hơn.

---

## 6. Sự cố thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `DatasetNotFoundError: ... gated dataset ... ask for access` (khi build corpus) | Chưa được duyệt truy cập `roots_vi_vietnamese_poetry` | Vào trang dataset trên HF, bấm "Agree and access repository", đợi vài phút rồi thử lại; hoặc dùng `--skip-poetry` để không bị chặn |
| `CUDA error: srcIndex < srcSelectDimSize` | Tokenizer/embedding mismatch của base model checkpoint (xem mục 2) | Đã vá trong `run_train_slm.py`/`evaluate_slm.py`; nếu đổi base model khác vẫn gặp, kiểm tra `len(tokenizer)` vs `model.get_input_embeddings().weight.shape[0]` |
| `RuntimeError: CUDA out of memory` | GPU không đủ VRAM cho `--batch-size`/`--max-length` hiện tại | Dùng cấu hình 6GB: `--batch-size 1 --max-length 128 --grad-accum 8`, hoặc giảm thêm |
| `[WARN] val_split.json not found; rebuilding the split...` (khi eval) | Adapter train bằng version script cũ chưa lưu `val_split.json` | Bắt buộc truyền đúng `--train-data-path`/`--max-length` đã dùng lúc train; số liệu vẫn kém tin cậy hơn trường hợp có sẵn `val_split.json` |
| `test_checksums_manifest_matches_files_on_disk` fail khi chạy `pytest` cục bộ | File dataset mới generate (`training_corpus_v1.json`, `vcc_bench_uit_viquad_qa.json`) chưa có trong `CHECKSUMS.json` | Vô hại nếu không định commit; nếu commit, chạy `scripts/checksum_datasets.py --write` cùng lúc (xem mục 1) |
| `RuntimeError: CUDA GPU is required.` | Chạy `run_train_slm.py`/`evaluate_slm.py` trên máy không có GPU CUDA | Không hỗ trợ CPU cho 2 script này (khác với `run_benchmark.py`/`run_ablation.py`, có `--device cpu`) |

---

---

## 7. Control cho tone probe (`scripts/train_probe_control.py`)

Trả lời câu hỏi mà con số 92.94% một mình không trả lời được: **LoRA và
`--lambda-tone` có thực sự đóng góp gì vào biểu diễn không?**

```bash
# 4 lần chạy, ghi kết quả dồn vào một file JSONL
python scripts/train_probe_control.py --mode frozen_base --out results_slm/probe.jsonl
python scripts/train_probe_control.py --mode lora        --out results_slm/probe.jsonl
python scripts/train_probe_control.py --mode frozen_base --control-task --out results_slm/probe.jsonl
python scripts/train_probe_control.py --mode lora        --control-task --out results_slm/probe.jsonl
```

Cách đọc:

| Hiệu số | Ý nghĩa |
|---|---|
| `lora` − `frozen_base` | LoRA/`--lambda-tone` **thực sự** thêm được bao nhiêu vào biểu diễn. Nếu ≈ 0 → claim "tone-aware training" không đứng vững |
| `real` − `control_task` | **Probe selectivity** (Hewitt & Liang 2019). Gần 0 → probe chỉ đang ghi nhớ token id, không đọc cấu trúc từ biểu diễn |

Script train probe với model **đóng băng** (chỉ probe có gradient, model
forward dưới `no_grad`) nên nhanh hơn fine-tune nhiều và chạy được trên
GPU nhỏ. Nó **rebuild lại train/val split** từ corpus rồi **đối chiếu**
nửa val với `val_split.json` — nếu corpus đã bị ghi đè kể từ lúc train
adapter, script báo lỗi thay vì âm thầm cho probe học nhầm dữ liệu.

Lưu ý về `--control-task`: vì nhãn thanh là hàm xác định của token id
(mục 5), control task **cũng** dễ học đúng như task thật, nên selectivity
gần 0 là kết quả **được dự đoán trước** — và đó chính là bằng chứng định
lượng cho việc đây là bài toán ghi nhớ token identity.

---

## 8. Đo tác động thật lên pipeline nén (quan trọng nhất cho paper)

Trước đây SLM được train/eval **hoàn toàn cô lập** — `TinyModelScorer` /
`EnhancedCompressor` không nằm trong `COMPRESSOR_REGISTRY` và không
implement interface `BaseCompressor`, nên `run_benchmark.py` **không có
đường nào** dùng tới SLM đã train. Giờ đã có 2 method mới:

| Method | Là gì | Phân loại |
|---|---|---|
| `slm_scorer` | LACC lightweight với SLM đã fine-tune làm scorer perplexity | `proposed` |
| `slm_scorer_base` | Đúng compressor đó nhưng **tắt LoRA adapter** | `ablation` |

```bash
python run_benchmark.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --methods none,random,combined,slm_scorer_base,slm_scorer \
  --scorer-adapter-dir trained_slm/final \
  --data-path vcc_bench_data/vcc_bench_uit_viquad_qa.json \
  --output-dir results/slm-impact-v1
python scripts/summarize_results.py results/slm-impact-v1
```

Bộ 3 phép so sánh cần đọc:

- `slm_scorer` vs `combined` → SLM có hơn heuristic thuần (0 VRAM) không?
- `slm_scorer` vs `slm_scorer_base` → **fine-tuning** có đóng góp gì không?
- mọi method vs `none` → mất bao nhiêu chất lượng để đổi lấy mức nén đó

**VRAM**: chọn `--model` vừa với card. Qwen2.5-7B **không** chạy được
trên 6GB; `Qwen/Qwen2.5-1.5B-Instruct` (fp16 ~3.1GB) + scorer (~0.3GB)
thì vừa. Kết quả với generation model nhỏ không so sánh được với kết quả
dùng 7B — ghi rõ model nào trong báo cáo (`config.json` tự lưu lại).

### Bất khớp tokenizer đã được xử lý

Benchmark đưa vào `compress()` token id của tokenizer **generation model**
(Qwen, ~152k từ vựng), trong khi scorer là model khác với tokenizer riêng
(GPT-2, ~50k). Đưa thẳng id của model này vào model kia là **vô nghĩa và
có thể lỗi index out-of-range** — `SLMScorerCompressor` vì vậy ánh xạ điểm
số qua **offset ký tự** trên văn bản đã decode. Đã kiểm chứng: nén đúng
tỉ lệ 2x/4x/8x với cặp tokenizer lệch nhau hoàn toàn.

### Cảnh báo: TPR thoái hoá khi tokenizer không hợp với tiếng Việt

Đo thực tế trên cùng một câu tiếng Việt:

| Tokenizer | Số token | Token mang thanh |
|---|---|---|
| `gpt2` (tiếng Anh) | 160 | 4 (**2.5%**) |
| `vietnamese-gpt2-base` | 43 | 26 (**60.5%**) |

Tokenizer tiếng Anh vừa **thổi phồng token 3.7×** (đúng vấn đề "token
inflation" README nêu, thậm chí tệ hơn mức 1.5–2.0× được ghi), vừa làm
TPR **luôn bằng 1.000** ở mọi ratio — vì gần như không token nào được
nhận là mang thanh, rơi vào edge case "không có token mang thanh → trả
1.0". Với tokenizer tiếng Việt, TPR giảm hợp lý: 0.577 (2x) → 0.308 (4x)
→ 0.115 (8x). **Đừng báo cáo TPR mà không kèm tỉ lệ token mang thanh của
tokenizer đang dùng.**

---

## 9. Độ đo còn thiếu

- **Chi phí thực đo của mode `lightweight`** — latency (ms/cửa sổ 512
  token) và VRAM thật của scorer. README đang ghi "~0.3GB VRAM ở INT4"
  cho SmolLM2-135M, nhưng scorer thực tế là `vietnamese-gpt2-base`; con
  số này cần **đo**, không phải giả định — nếu không, nhãn "lightweight"
  không có gì chống lưng.
- **Tách NLL theo `source`** (`uvw-2026` vs `vietnamese-poetry`) để cô
  lập artifact ghép thơ — cần sửa `run_train_slm.py` lưu thêm `source`
  vào `val_split.json`, chỉ có tác dụng cho các lần train **sau**, không
  hồi tố được checkpoint hiện tại.
- **Khoảng tin cậy cho các số VCC-Bench** — `compare_slm_runs.py` mới chỉ
  làm cho perplexity; các chỉ số downstream (F1/EM/ROUGE-L) hiện vẫn báo
  cáo dưới dạng trung bình trần trụi, chưa có CI.

---

## Tệp liên quan

- [`scripts/build_training_corpus.py`](../scripts/build_training_corpus.py) — build dataset training
- [`scripts/compare_slm_runs.py`](../scripts/compare_slm_runs.py) — kiểm định thống kê theo cặp giữa 2 lần eval
- [`scripts/train_probe_control.py`](../scripts/train_probe_control.py) — control frozen-base + probe selectivity
- [`vncompress/compressors/slm_scorer.py`](../vncompress/compressors/slm_scorer.py) — `slm_scorer` / `slm_scorer_base`
- [`scripts/build_viquad_eval.py`](../scripts/build_viquad_eval.py) — build dataset eval QA thật
- [`run_train_slm.py`](../run_train_slm.py) — huấn luyện LoRA + tone probe
- [`evaluate_slm.py`](../evaluate_slm.py) — đánh giá (+ `--no-adapter` baseline)
- [`vcc_bench_data/PROVENANCE.md`](../vcc_bench_data/PROVENANCE.md) — nguồn/license/gated-access chi tiết
- [`docs/training_eval_report_template.md`](training_eval_report_template.md) — mẫu ghi kết quả từng lần chạy
- [`docs/benchmark.md`](benchmark.md) — protocol đánh giá VCC-Bench (thuật toán nén, khác SLM)
