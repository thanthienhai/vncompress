# Báo cáo huấn luyện & đánh giá SLM (Vietnamese Tone-Aware LoRA)

> Copy file này thành `docs/reports/YYYY-MM-DD_<mô-tả-ngắn>.md` cho mỗi lần
> train/eval, điền vào các mục `TODO`. Dùng để so sánh các lần chạy với nhau.

## 0. Thông tin chung

| Trường | Giá trị |
|---|---|
| Người thực hiện | TODO |
| Ngày chạy | TODO (YYYY-MM-DD) |
| Git commit | TODO (`git rev-parse HEAD`) |
| Nhánh | TODO |
| GPU | TODO (`torch.cuda.get_device_name(0)`, VRAM) |
| Mục đích lần chạy | TODO (vd: thử `lambda_tone` cao hơn, đổi base model, ...) |

---

## 1. Huấn luyện (`run_train_slm.py`)

### 1.1 Lệnh đã chạy

```bash
python run_train_slm.py \
  --model TODO \
  --output-dir TODO \
  --train-data-path TODO \
  --epochs TODO \
  --batch-size TODO \
  --max-length TODO \
  --lr TODO \
  --lora-r TODO \
  --lambda-tone TODO \
  --grad-accum TODO \
  --max-steps TODO
```

### 1.2 Cấu hình (điền lại từ lệnh trên, để dễ so sánh giữa các lần chạy)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `--model` (base model) | TODO | |
| `--train-data-path` | TODO | Nếu để trống → fallback `vcc_bench_data/wikipedia_vi_raw.json` (xem `run_training.load_training_texts`) |
| `--epochs` | TODO | |
| `--batch-size` | TODO | |
| `--max-length` | TODO | Phải khớp khi eval nếu không có `val_split.json` |
| `--lr` | TODO | |
| `--lora-r` | TODO (`lora_alpha` = 2× giá trị này) | |
| `--lambda-tone` | TODO | Trọng số `PhonologicalConsistencyLoss` |
| `--grad-accum` | TODO | |
| `--max-steps` | TODO (`-1` = chạy hết epoch) | |
| `--quick` | TODO (có/không) | Nếu có: ép `epochs=1, max_steps=30, max_length=min(128, ...)` |
| Gradient checkpointing | TODO (bật mặc định, tắt nếu `--no-gradient-checkpointing`) | |

### 1.3 Dataset huấn luyện

| Trường | Giá trị |
|---|---|
| Nguồn dữ liệu | TODO (đường dẫn file / dataset) |
| Số văn bản hợp lệ (`len(dataset)`) | TODO |
| Số mẫu train / val (90/10 split, seed=42) | TODO / TODO |
| Checksum / phiên bản dataset (nếu là VCC-Bench) | TODO |

### 1.4 Kết quả log huấn luyện

| Trường | Giá trị |
|---|---|
| Tổng số optimizer steps | TODO |
| LM loss bước cuối | TODO |
| Tone loss bước cuối | TODO |
| VRAM tối đa (`max_memory_allocated`) | TODO GB |
| Thời gian chạy | TODO |
| Đường dẫn adapter đã lưu | TODO (`<output-dir>/final`) |
| Đường dẫn tone probe đã lưu | TODO (`<output-dir>/tone_probe.pt`) |
| Có lưu `val_split.json` không | TODO (có/không) |

**Log các bước tiêu biểu** (copy vài dòng in ra từ script, vd step 1/10/20/cuối):

```
TODO: dán log dạng
step=1 lm=... tone=... vram=... GB
step=10 lm=... tone=... vram=... GB
...
```

---

## 2. Đánh giá (`evaluate_slm.py`)

### 2.1 Lệnh đã chạy

```bash
python evaluate_slm.py \
  --adapter-dir TODO \
  --tone-probe TODO \
  --train-data-path TODO \
  --max-length TODO \
  --batch-size TODO
```

> Nếu adapter có sẵn `val_split.json`, `--train-data-path`/`--max-length` bị
> bỏ qua (script dùng đúng tập held-out đã lưu lúc train). Nếu không có
> file này, **bắt buộc** truyền đúng giá trị đã dùng lúc train ở mục 1.2,
> nếu không tập validation sẽ không khớp với lúc train.

### 2.2 Kết quả

| Metric | Giá trị |
|---|---|
| Số văn bản validation | TODO |
| LM validation loss (NLL) | TODO |
| Perplexity | TODO |
| Tone accuracy (all tokens) | TODO % |
| Tone accuracy (marked tones only) | TODO % (trên TODO tokens) |
| Baseline luôn đoán "ngang" (majority-class) | TODO % |
| Tone accuracy (marked) − baseline | TODO (điểm chênh lệch — thước đo thật của việc học thanh điệu) |

### 2.3 Nhận xét

- TODO: model có học được tín hiệu thanh điệu tốt hơn baseline "luôn đoán ngang" không?
- TODO: perplexity có hợp lý so với base model gốc (chưa fine-tune) không?
- TODO: vấn đề/bất thường quan sát được (overfit, loss không giảm, OOM, ...)

---

## 3. So sánh với các lần chạy trước (tuỳ chọn)

| Lần chạy | Ngày | `lambda_tone` | Epochs | Tone acc (marked) | Perplexity | Ghi chú |
|---|---|---|---|---|---|---|
| TODO (link tới report này) | TODO | TODO | TODO | TODO | TODO | TODO |
| ... | | | | | | |

---

## 4. Việc tiếp theo (tuỳ chọn)

- [ ] TODO
