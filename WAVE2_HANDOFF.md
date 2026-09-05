# Wave 2 — Handoff for the training team

**Ngày:** 2026-09-05 · **Nguồn đề xuất:** [`research/wave2_proposals.md`](research/wave2_proposals.md) · **Trạng thái:** code đã implement + test (CPU), **chưa chạy GPU**

Tất cả thay đổi ở đây là **code không cần GPU để viết/test**; việc còn lại là các
bạn **chạy lại benchmark/train** trên cluster. `python -m pytest -q` phải xanh
trước khi chạy (CPU-only, ~20s).

---

## 1. Chạy gì trước (đường đi khuyến nghị)

Không GPU-train — chỉ eval (dùng generator + scorer sẵn có):

```bash
# Core wave-2 arms trên VCC-Bench v2, 3 tỉ lệ, 2 generator để robustness check
python benchmark.py --model Qwen/Qwen2.5-7B-Instruct --ratios 2,4,8 \
  --data-path data/benchmark/vcc_bench_v2.json \
  --scorer-adapter-dir models/qwen3/final --tone-probe-path models/qwen3/tone_probe.pt \
  --methods none,random,llmlingua,llmlingua_contrastive,lacc_ppl_contrastive,lacc_ppl_morph,lacc_cx_morph,lacc_sentence,lacc_classprop

python benchmark.py --list-methods   # xem toàn bộ arm mới
```

Kỳ vọng theo đề xuất: `llmlingua_contrastive` và `lacc_ppl_contrastive` (E1) lấy
lại phần lớn khoảng cách trên **needle/QA**; `lacc_sentence` (E5) mạnh nhất ở
needle; so `lacc_ppl_morph` (E2) với `llmlingua` xem morphology có cộng thêm khi
đã có perplexity.

Đo P3 (không cần generator):

```bash
python scripts/measure_token_inflation.py --tokenizer Qwen/Qwen2.5-7B-Instruct
```

---

## 2. Các arm mới (chọn bằng `--methods`)

| Arm | Đề xuất | Là gì |
|---|---|---|
| `llmlingua_contrastive` | E1/E11 | LongLLMLingua: perplexity có điều kiện câu hỏi (contrastive). Baseline query-aware. |
| `lacc_ppl_contrastive` | E1 | LACC chỉ dùng perplexity query-conditioned (arm giá trị cao nhất). |
| `lacc_ppl_morph` | E2 | perplexity × morphology (nhân, không cộng). Tone tắt. |
| `lacc_cx_morph` | E1+E2 | perplexity query-conditioned × morphology. |
| `lacc_sentence` | E5 | chọn theo **câu** (extractive) thay vì token lẻ. |
| `lacc_classprop` | E7 | phân bổ budget theo **hạn ngạch lớp từ** (điều paper hứa). |
| `lacc_tone_gated` | E8 | tone chỉ bật cho task bề mặt (cross-lingual…), tắt cho QA/needle. |
| `encoder` | E6/E11 | compressor encoder token-classification (cần checkpoint, xem §4). |

Mọi arm nhận `query` và `task` từ harness (`evaluation.py`), nên E1/E8 tự hoạt động.

## 3. Thay đổi ảnh hưởng *mọi* run

- **E3 — cửa sổ perplexity 2048 / overlap 256** (`compression.DEFAULT_PPL_WINDOW`,
  `_default_stride`). Giảm ~6–7× số forward pass so với 512/stride-256 cũ. Đây là
  lý do các arm dùng SLM giờ rẻ hơn nhiều; nên **đo lại latency** để cập nhật bảng
  chi phí (§8.7 báo cáo wave 1). Không đổi thuật toán, chỉ cấu hình.

## 4. Hai pipeline cần GPU-train trước khi eval

### E4 — Relevance probe (điểm đột phá, cứu C2)

Đổi mục tiêu probe từ "thanh điệu" → "liên-quan-câu-hỏi", **tái dùng hạ tầng train
wave 1**. Probe mới `linguistics.RelevanceConsistencyLoss` cùng interface với tone
probe nên cắm thẳng vào `LACCCompressor(tone_source='model')`.

```bash
python scripts/train_relevance_probe.py \
  --adapter-dir models/qwen3/final --data-path data/benchmark/vcc_bench_v2.json \
  --output-dir models/qwen3 --load-4bit        # freeze base, chỉ train probe (rẻ)
```

Rồi A/B đúng khuôn wave 1 (probe-relevance vs probe-tone vs rule) bằng
`scripts/verify_tone_probe_e2e.py`. Kỳ vọng **đảo dấu** kết quả A/B của wave 1.

### E6 — Encoder classifier (LLMLingua-2 / PhoBERT)

Bỏ scorer generative 4B, dùng encoder phân loại token "giữ/bỏ" (rẻ hơn, 1 forward
pass song song).

```bash
python scripts/train_encoder_compressor.py \
  --encoder-id vinai/phobert-base --train-data-path data/benchmark/training_corpus_v1.json \
  --teacher-model Qwen/Qwen2.5-0.5B-Instruct --ratio 4 --output-dir models/encoder_cls

python benchmark.py --methods none,llmlingua,encoder --ratios 2,4,8 \
  --model Qwen/Qwen2.5-7B-Instruct    # arm 'encoder' đọc checkpoint từ --output-dir
```

## 5. Đề xuất chưa cần code

- **E10** (mask token cho probe task) — không cần nữa nếu theo E4; nhãn liên-quan
  không phải hàm tất định của token id.
- Các mục §12/§13 của báo cáo wave 1 (commit, sửa abstract, bootstrap CI, run
  λ_tone=0 control, hoàn tất nhánh `slm_tone_probe`) vẫn cần làm — không thuộc phạm
  vi code wave 2.

## 6. Kiểm thử

```bash
python -m pytest -q          # toàn bộ, CPU-only, ~20s — phải xanh trước khi chạy cluster
```
Test mới: `tests/test_compression.py` (E1/E2/E3/E5/E7/E8), `tests/test_relevance_probe.py`
(E4), `tests/test_encoder_compression.py` (E6).
