# Báo cáo huấn luyện & đánh giá SLM (Vietnamese Tone-Aware LoRA)

> Ghi lại theo mẫu [`docs/training_eval_report_template.md`](../training_eval_report_template.md).
> Báo cáo này gộp 2 lần train hợp lệ (Run 2, Run 3) trên cùng chuỗi mở
> rộng dataset, để so sánh trực tiếp. Run 1 (dataset gốc 393 đoạn, split
> lệch lúc eval) không đủ tin cậy để đưa vào bảng số liệu — xem mục 4.

## 0. Thông tin chung

| Trường | Giá trị |
|---|---|
| Người thực hiện | tranthuykieu1103@gmail.com (qua phiên làm việc với Claude Code) |
| Ngày chạy | 2026-08-19 → 2026-08-20 |
| GPU | NVIDIA GeForce GTX 1060 6GB |
| Model base | `chronopt-research/vietnamese-gpt2-base` (137M) |
| Mục đích | Khắc phục tone probe học ~random (Run 1) bằng cách mở rộng dataset training (`scripts/build_training_corpus.py`, nguồn UVW-2026 + Vietnamese Poetry); sau đó tiếp tục scale-up để đo xu hướng cải thiện |

---

## 1. Huấn luyện (`run_train_slm.py`)

### 1.1 Lệnh đã chạy

**Run 2:**
```bash
python scripts/build_training_corpus.py
python run_train_slm.py \
  --train-data-path vcc_bench_data/training_corpus_v1.json \
  --batch-size 1 --max-length 128 --grad-accum 8
```

**Run 3:**
```bash
python scripts/build_training_corpus.py --uvw-n 20000
python run_train_slm.py \
  --train-data-path vcc_bench_data/training_corpus_v1.json \
  --batch-size 1 --max-length 128 --grad-accum 8 --epochs 2
```

> `--uvw-n 20000` cho Run 3 là suy ra từ `Validation texts: 2218` (≈10%
> của tổng corpus ~22,180-22,222) khớp với khuyến nghị đã đưa ra trong
> phiên làm việc, không phải trích trực tiếp từ log lệnh `build_training_corpus.py`
> của người dùng — xác nhận lại nếu cần độ chính xác tuyệt đối.

### 1.2 Cấu hình

| Tham số | Run 2 | Run 3 |
|---|---|---|
| `--model` | `chronopt-research/vietnamese-gpt2-base` (mặc định) | như Run 2 |
| `--uvw-n` (build corpus) | 5,000 (mặc định) | 20,000 |
| `--poetry-ratio` (build corpus) | 0.10 (mặc định) | 0.10 (mặc định) |
| `--batch-size` | 1 | 1 |
| `--max-length` | 128 | 128 |
| `--grad-accum` | 8 | 8 |
| `--epochs` | 3 (mặc định) | 2 (giảm để bù thời gian train do corpus lớn hơn) |
| `--lora-r` / `--lambda-tone` | 8 / 0.1 (mặc định) | 8 / 0.1 (mặc định) |

### 1.3 Dataset huấn luyện

| Trường | Run 2 | Run 3 |
|---|---|---|
| Nguồn | UVW-2026 (CC-BY-SA 4.0) + Vietnamese Poetry (MIT) | như Run 2, tỉ lệ UVW-2026 lớn hơn |
| Tổng số đoạn | 5,556 (5,000 UVW-2026 + 556 Poetry) | ~22,180-22,222 (~20,000 UVW-2026 + ~2,220 Poetry) |
| Train / Val (90/10, seed=42) | ~5,000 / 555 | ~19,960-20,000 / 2,218 |

### 1.4 Kết quả log huấn luyện

| Trường | Run 2 | Run 3 |
|---|---|---|
| Optimizer steps dự kiến | 1,875 (`updates_per_epoch=625 × 3 epochs`) | không tính lại (corpus/epochs khác Run 2) |
| Optimizer steps thực tế | 1,872 | không ghi lại trong phiên làm việc |
| VRAM tối đa quan sát | 0.66 GB | không ghi lại |
| `val_split.json` được lưu | Có (555 texts) | Có (2,218 texts) |

---

## 2. Đánh giá (`evaluate_slm.py`)

### 2.1 Lệnh đã chạy

```bash
# Adapter đã train
python evaluate_slm.py --adapter-dir trained_slm/final --tone-probe trained_slm/tone_probe.pt

# Baseline (base model, không LoRA) trên ĐÚNG val_split.json của adapter đó
python evaluate_slm.py --adapter-dir trained_slm/final --no-adapter
```

Cả hai lệnh tự dùng `trained_slm/final/val_split.json` đã lưu lúc train
— không cần `--train-data-path`/`--max-length`.

> Lưu ý quan trọng: `trained_slm/final` bị **ghi đè** giữa Run 2 và Run 3
> (cùng `--output-dir` mặc định `./trained_slm`), nên baseline của Run 2
> (555 texts) và baseline của Run 3 (2,218 texts) được đo trên **hai
> `val_split.json` khác nhau** — không so sánh chéo base-Run2 với
> LoRA-Run3 hay ngược lại, chỉ so sánh trong cùng cột.

### 2.2 Kết quả

| Metric | Run 2 — Base | Run 2 — LoRA | Run 3 — Base | Run 3 — LoRA |
|---|---|---|---|---|
| Validation texts | 555 | 555 | 2,218 | 2,218 |
| LM validation loss (NLL) | 5.0740 | 4.8223 | 4.9979 | 4.7073 |
| Perplexity | 159.81 | 124.26 | 148.10 | 110.76 |
| Cải thiện Perplexity (LoRA vs Base, cùng split) | **−22.2%** | | **−25.2%** | |
| Tone accuracy (all tokens) | — (bỏ qua ở `--no-adapter`) | 88.51% | — | 95.45% |
| Tone accuracy (marked tones only) | — | **82.24%** (32,865 tokens) | — | **92.94%** (127,691 tokens) |
| Majority-class baseline (all tokens, luôn đoán ngang) | — | 44.93% | — | 46.15% |

### 2.3 Nhận xét

- **Tone learning cải thiện nhất quán khi tăng quy mô corpus**: 82.24%
  (Run 2, 5,556 đoạn) → 92.94% (Run 3, ~22,200 đoạn), cách xa mức random
  ~20% (5 thanh không-ngang) và majority baseline — model học được tín
  hiệu thanh điệu thật, không phải nhiễu.
- **LoRA fine-tuning cải thiện LM perplexity thật, có baseline xác
  nhận độc lập ở cả hai lần**: −22.2% (Run 2) và −25.2% (Run 3) so với
  base model trên cùng validation split. Tỉ lệ cải thiện còn tốt hơn khi
  corpus lớn hơn.
- Base perplexity tự nó khác nhau giữa Run 2 (159.81) và Run 3 (148.10)
  — **không phải base model thay đổi** (base cố định, không train), chỉ
  vì mỗi run có nội dung `val_split.json` khác nhau. Không suy luận gì
  từ chênh lệch này.
- **Artifact cần lưu ý khi đọc perplexity tuyệt đối**: `build_training_corpus.py`
  ghép nhiều bài thơ ngắn không liên quan để đạt ngưỡng lọc >200 ký tự —
  tạo ra các điểm "đứt gãy ngữ nghĩa" giữa các bài thơ ghép, có thể thổi
  phồng perplexity một phần không phản ánh chất lượng model (xem
  [`docs/slm_training_guide.md`](../slm_training_guide.md#5-cách-đọc-kết-quả-eval-cho-đúng)).

---

## 3. So sánh với các lần chạy trước

| Lần chạy | Ngày | Corpus (đoạn) | `--epochs` | Tone acc (marked) | Perplexity (LoRA) | Cải thiện vs base | Ghi chú |
|---|---|---|---|---|---|---|---|
| Run 1 (baseline, không đưa vào báo cáo chính thức) | 2026-08-19 | 393 | 3 (mặc định) | 19.64% | 51.57 | không đo | Split lúc eval bị lệch (thiếu `val_split.json`, tự dựng lại với `--max-length` có thể sai) — số liệu không đáng tin, chỉ dùng để phát hiện vấn đề dataset quá nhỏ |
| Run 2 | 2026-08-19/20 | 5,556 | 3 | 82.24% | 124.26 | −22.2% | `val_split.json` đúng, baseline đo cùng split |
| Run 3 | 2026-08-20 | ~22,200 | 2 | **92.94%** | **110.76** | **−25.2%** | Tốt nhất tính đến nay; `trained_slm/final` hiện tại là checkpoint này |

---

## 4. Việc tiếp theo

- [ ] Xác nhận lại chính xác `--uvw-n` đã dùng cho Run 3 (báo cáo đang suy
      ra từ `Validation texts: 2218`, xem ghi chú mục 1.1).
- [ ] Cân nhắc tách riêng NLL/perplexity theo `source` (`uvw-2026` vs
      `vietnamese-poetry`) để đo đúng mức độ artifact do ghép thơ ngắn
      gây ra (hiện chưa có script sẵn, cần code thêm).
- [ ] Nếu tiếp tục scale corpus (>20,000 đoạn UVW-2026 hoặc thêm nguồn
      khác như tin tức), lặp lại đúng quy trình: build → train → eval
      adapter → eval `--no-adapter` cùng split → ghi báo cáo mới, không
      ghi đè báo cáo này.
- [ ] Cân nhắc dùng checkpoint Run 3 (`trained_slm/final` hiện tại) làm
      external perplexity scorer thật trong pipeline LACC `lightweight`
      (hiện tại SLM mới chỉ được train/eval độc lập, chưa được đo tác
      động khi cắm vào `run_benchmark.py`).
