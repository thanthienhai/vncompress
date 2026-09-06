# VNCompress — LACC: Language-Aware Context Compression cho Tiếng Việt

Research codebase chứng minh **LACC (Language-Aware Context Compression)**: nén ngữ cảnh có ý thức về thanh điệu và hình thái học cho LLM trên tiếng Việt.

## 1. Tổng quan

Các phương pháp nén ngữ cảnh hiện tại (LLMLingua, SnapKV) **bất chấp ngôn ngữ**, gây ba vấn đề với tiếng Việt: mất thông tin thanh điệu (6 thanh, xóa dấu đổi nghĩa hoàn toàn), lãng phí ngân sách nén (30–40% token là hư từ), và token inflation (1.5–2.0× so với tiếng Anh).

## 2. Vấn đề nghiên cứu

Có thể kết hợp tri thức ngôn ngữ học tiếng Việt (thanh điệu, hình thái từ) với tín hiệu học máy nhẹ (perplexity, trained tone probe) trong **một** thuật toán nén duy nhất, đạt compression ratio cao mà vẫn bảo toàn thông tin quan trọng — mà không cần huấn luyện lại toàn bộ mô hình sinh?

## 3. Phương pháp đề xuất

`LACCCompressor` (xem [`vncompress/compression.py`](vncompress/compression.py)) kết hợp ba tín hiệu:

```
S(t) = w_ppl · S_ppl(t) + w_tone · S_tone(t) + w_morph · S_morph(t)
```

- **S_ppl**: perplexity từ mô hình sinh, một SLM tiếng Việt nhỏ đã fine-tune, hoặc trung lập nếu không có model (tier 0 VRAM).
- **S_tone**: heuristic thanh điệu (mật độ + đa dạng + tương phản với lân cận) hoặc trained tone probe (`tone_source='model'`) — cầu nối training→inference của paper.
- **S_morph**: hệ số bảo toàn theo lớp từ (hư từ/thực từ/từ láy/từ ghép/Sino-Việt).

Ablation chỉ là **config**, không phải class hay script riêng: `LACCCompressor(use_perplexity=False, use_tone=True, use_morphology=False)`.

## 4. Kiến trúc

```
train.py --mode lacc|slm      benchmark.py [--ablation]      evaluate.py --input results/...
        │                              │                              │
        ▼                              ▼                              ▼
vncompress/training.py     vncompress/compression.py         vncompress/evaluation.py
        │                    (LACCCompressor + baselines)              │
        ▼                              │                               │
vncompress/models.py ◄─────────────────┴──── vncompress/linguistics.py ┘
(model/scorer loading)                      (tone + morphology + tone probe)
```

Mỗi module có một trách nhiệm rõ ràng — chi tiết ở [Cấu trúc thư mục](#6-cấu-trúc-thư-mục).

## 5. Cài đặt

```bash
git clone https://github.com/thanthienhai/vncompress.git
cd vncompress
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CPU-only (test suite, lint, smoke test, không cần GPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 6. Data pipeline

Mọi thứ huấn luyện/đánh giá đọc **split 90/10 theo document**. Dựng một lần:

```bash
python scripts/normalize_dataset.py   # raw -> data/processed/records.jsonl (canonical, có doc_id)
python scripts/verify_dataset.py      # deterministic checks (§6.1), ghi verification_report.json
python scripts/split_dataset.py       # 90/10 theo document; từ chối ghi nếu phát hiện rò rỉ
```

Vì sao theo document chứ không theo record: một bài UVW-2026 bị cắt thành tối đa 162 paragraph cùng
`topic_id`, và VCC-Bench sinh 3 biến thể câu hỏi cho mỗi paragraph — split mức record đặt cùng một
tài liệu vào cả train lẫn eval. Chi tiết ở [`docs/dataset_pipeline.md`](docs/dataset_pipeline.md) §9.

Bỏ qua bước này vẫn chạy được (split được suy ra trong bộ nhớ từ file raw), chỉ khác là không có
manifest để audit.

### Tầng teacher distillation (tuỳ chọn)

Sinh supervision bằng một teacher LLM mạnh (§4): câu hỏi cho đoạn văn chưa có câu hỏi, rồi bản nén
kèm markup entity/số/ngày/phủ định/điều kiện. Cấu hình endpoint trong `.env` (`cp .env.example .env`;
`.env` đã gitignore), nhận mọi endpoint OpenAI-compatible.

```bash
python scripts/generate_teacher_dataset.py --stage queries --dry-run --limit 20
python scripts/filter_dataset.py --stage queries            # verify §6 + merge, loại vào quarantine
python scripts/generate_teacher_dataset.py --stage compression --dry-run --limit 20 --ratios 2,4,8
python scripts/filter_dataset.py --stage compression
```

`--dry-run` chạy hết đường ống bằng stub offline, không tốn token — **luôn chạy nó trước** khi trỏ
vào endpoint tính tiền. Bỏ `--dry-run` để gọi thật. Generator mặc định chỉ đọc split `train` và
**từ chối** `eval.jsonl` (§10). Kết quả thô được giữ ở `data/teacher/` để lọc lại mà không phải gọi
teacher lần nữa.

## 7. Huấn luyện

```bash
python train.py --mode lacc --config configs/training.json          # fine-tune model sinh + tone probe
python train.py --mode slm                                          # SLM scorer nhỏ (cần GPU), đọc split đã dựng
python train.py --mode slm --validate --adapter-dir models/slm/final --tone-probe models/slm/tone_probe.pt
```

Chi tiết đầy đủ (xây dataset, tuning theo GPU, cách đọc kết quả validate, đo tác động thật lên pipeline nén) ở [`docs/training.md`](docs/training.md).

### Wave 2 — hai pipeline huấn luyện mới

Sau khi wave 1 bác bỏ giả thuyết tone-aware, wave 2 thêm hai hướng huấn luyện (đề xuất & trạng thái ở [`research/wave2_proposals.md`](research/wave2_proposals.md), hướng dẫn chạy ở [`WAVE2_HANDOFF.md`](WAVE2_HANDOFF.md)):

```bash
# E4 — probe dự đoán liên-quan-câu-hỏi (thay cho probe thanh điệu), freeze base, chỉ train probe
python scripts/train_relevance_probe.py --adapter-dir models/qwen3/final \
    --data-path data/processed/vcc_bench_train.json --output-dir models/qwen3 --load-4bit
#   -> models/qwen3/relevance_probe.pt (+ relevance_probe_meta.json, kèm metric held-out)

# E6 — encoder token-classification compressor (LLMLingua-2 / PhoBERT), distill nhãn keep/drop từ teacher
python scripts/train_encoder_compressor.py --encoder-id vinai/phobert-base \
    --teacher-model Qwen/Qwen2.5-0.5B-Instruct --ratio 4 --output-dir models/encoder_cls
#   -> checkpoint + distillation_meta.json (teacher, ratio, seed, split, metric held-out)
```

Cả hai đọc phía **train** của split và báo cáo số liệu trên phía **eval** chưa từng nhìn thấy.

`load_scorer(..., probe_kind='relevance')` nạp relevance probe; A/B nó với probe thanh điệu bằng `scripts/verify_tone_probe_e2e.py`. Đo token inflation (P3): `python scripts/measure_token_inflation.py`.

## 8. Benchmark

```bash
python benchmark.py --list-methods                                   # gồm cả các arm wave-2
python benchmark.py --config configs/benchmark.json                  # VCC-Bench đầy đủ
python benchmark.py --ablation --model Qwen/Qwen2.5-7B-Instruct --ratios 2,4,8   # tách từng tín hiệu
python evaluate.py --input results/qwen2.5-7b-vcc-bench-v1            # bảng baseline/proposed/ablation

# Wave-2 arms (query-conditioned perplexity, ppl×morph, sentence-level, class-proportional budget):
python benchmark.py --model Qwen/Qwen2.5-7B-Instruct --ratios 2,4,8 \
    --methods none,llmlingua,llmlingua_contrastive,lacc_ppl_contrastive,lacc_ppl_morph,lacc_sentence,lacc_classprop
```

VCC-Bench: 5 tác vụ (Long-Document QA, Multi-turn Conversation, Needle-in-Haystack, Agent Tool-Calling, Cross-lingual). Metrics: ROUGE-L, BLEU, BERTScore, Exact Match, Token-F1, **Tone Preservation Rate**, Harmonized Score. Giao thức đầy đủ ở [`docs/benchmark.md`](docs/benchmark.md).

```bash
pytest tests/ -v                        # test suite (CPU-only, ~20s)
ruff check vncompress tests scripts *.py
python scripts/smoke_test.py            # nén end-to-end với tokenizer thật, không cần model lớn
python scripts/checksum_datasets.py     # xác thực checksum dataset
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) chạy toàn bộ trên mỗi push/PR vào `main`, dùng torch CPU-only.

## 9. Kết quả

Mỗi experiment là một thư mục tự mô tả dưới `results/` (`config.json` + `environment.json` + `metrics.json`/`vcc_bench_results.json` + `predictions.json`), truy nguyên được qua git commit + checksum dataset. Xem `research/experiments.md` cho danh sách các lần chạy còn cần thực hiện và kết quả sơ bộ đã có (`results/training/reports/`).

### Quy tắc đặt tên kết quả training

Mỗi lần chạy `train.py` tạo một thư mục riêng, **không ghi đè** lần chạy trước, theo mẫu:

```
results/training/<YYYY-MM-DD>_<mode>[-<mô-tả-ngắn>]/
├── config.json         # ExperimentConfig đã resolve (model, seed, ...)
├── environment.json    # git commit, package versions, timestamp
├── metrics.json         # NLL/perplexity, tone accuracy, macro-F1, ...
└── README.md            # báo cáo ngắn: lệnh đã chạy, nhận xét, so sánh lần trước
```

- `<YYYY-MM-DD>`: ngày **chạy** training (không phải ngày viết báo cáo), lấy theo `environment.json.timestamp_utc`.
- `<mode>`: khớp đúng giá trị `--mode` đã dùng — `lacc` hoặc `slm`.
- `-<mô-tả-ngắn>` (tuỳ chọn): phân biệt nhiều lần chạy cùng ngày/cùng mode, vd. `-scaleup`, `-qwen3-4b`, `-quick`. Không dùng số thứ tự chung chung (`-run2`) — mô tả phải nói lên *cái gì thay đổi* so với lần trước.

Ví dụ: `results/training/2026-09-05_slm-scaleup/`, `results/training/2026-09-12_lacc-qwen3-4b/`.

**Phân biệt với `models/`**: `results/training/...` chỉ chứa metadata + báo cáo (nhẹ, luôn commit được); checkpoint có thể tái sử dụng (LoRA adapter, tokenizer, `tone_probe.pt`) nằm ở `models/<mode>/` hoặc `models/<tên-tuỳ-chỉnh>/` (đặt qua `--output-dir`), bị ghi đè mỗi lần train lại và phần lớn bị gitignore (xem [`docs/training.md`](docs/training.md#5-filing-a-trainingeval-report)). Một báo cáo trong `results/training/` nên ghi rõ nó ứng với checkpoint nào trong `models/` tại thời điểm chạy.

## 10. Cấu trúc thư mục

```
vncompress/
├── train.py / benchmark.py / evaluate.py   # 3 entrypoint duy nhất
├── configs/{training,benchmark}.json
├── vncompress/
│   ├── config.py         # ExperimentConfig, seed, environment snapshot
│   ├── compression.py    # LACCCompressor + baselines (core algorithm)
│   ├── linguistics.py    # tone + morphology + tone probe (Vietnamese-specific)
│   ├── models.py         # model/scorer loading, device utilities
│   ├── training.py       # 2 pipeline huấn luyện + validate_slm
│   ├── evaluation.py     # VCC-Bench, metrics, significance, taxonomy
│   └── utils.py          # JSON I/O nhỏ, dùng chung
├── tests/                # 4 file test, CPU-only, MockTokenizer
│   ├── dataset.py        # canonical schema, verify §6.1, split theo document §9
├── tests/                # test CPU-only, MockTokenizer
├── scripts/               # data pipeline (normalize/verify/split), build dataset, checksum, phân tích
├── data/benchmark/        # nguồn raw + VCC-Bench + PROVENANCE.md + CHECKSUMS.json
├── data/processed/        # canonical records + split 90/10 + split_manifest.json
├── models/                # checkpoint đã train (adapter lớn bị gitignore)
├── results/               # kết quả benchmark/training, mỗi run tự mô tả
├── docs/{training,benchmark}.md
├── research/              # ghi chú nghiên cứu, ý tưởng, research gaps
└── paper/                 # LaTeX paper
```

## 11. Trích dẫn

```bibtex
@misc{thanthien2026lacc,
  title={LACC: Language-Aware Context Compression for Vietnamese},
  author={Thanthien},
  year={2026},
  note={Tone-aware and morphology-aware context compression}
}
```

## Tham khảo chính

- LLMLingua (EMNLP 2023): Jiang et al., *Compressing Prompts for Accelerated Inference*
- SnapKV (2024): Li et al., *LLM Knows What You are Looking for*
- StreamingLLM (ICLR 2024): Xiao et al., *Efficient Streaming Language Models with Attention Sinks*
- H2O (NeurIPS 2023): Zhang et al., *Heavy-Hitter Oracle for Efficient Generative Inference*

Danh sách đầy đủ và ghi chú nghiên cứu mở rộng ở [`research/references.md`](research/references.md).
