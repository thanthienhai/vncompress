# VNCompress — Language-Aware Context Compression cho Tiếng Việt

> **LACC (Language-Aware Context Compression)**: Nén ngữ cảnh có ý thức về thanh điệu và hình thái học cho LLM trên tiếng Việt.

## Tổng quan

Các phương pháp nén ngữ cảnh hiện tại (LLMLingua, SnapKV, H2O, StreamingLLM) **bất chấp ngôn ngữ** — xử lý mọi token đồng nhất, gây ra ba vấn đề nghiêm trọng với tiếng Việt:

1. **Mất thông tin thanh điệu**: Tiếng Việt có 6 thanh, xóa dấu làm thay đổi nghĩa hoàn toàn.
2. **Lãng phí ngân sách nén**: 30–40% token là hư từ (function words) mang ít thông tin ngữ nghĩa.
3. **Token inflation**: Tiếng Việt tốn 1.5–2.0× token so với tiếng Anh do tokenizer phụ thuộc khoảng trắng.

**VNCompress** giải quyết bằng ba tín hiệu điểm số mới, kết hợp tri thức ngôn ngữ học với học máy nhẹ.

## Phương pháp đề xuất

Điểm số nén tổng hợp cho token `t`:

```
S(t) = w_ppl · S_ppl(t) + w_tone · S_tone(t) + w_morph · S_morph(t)
```

với `w_ppl + w_tone + w_morph = 1.0`, mỗi thành phần được chuẩn hóa về [0, 1].

### 1. Tone-Aware Scoring

Lượng hóa mức độ quan trọng của token dựa trên đặc điểm thanh điệu:

```
S_tone(t) = w_tone(t) · f_contrast(t)
```

- **Mật độ thanh**: `ρ(t) = count_non_ngang(t) / len(t)`
- **Đa dạng thanh**: `ν(t)` — số thanh không-ngang phân biệt trong token
- **Trọng số bảo toàn**: `w_tone(t) = 1.0 + α · ρ(t) · (1 + β · ν(t)/6)`
- **Hệ số tương phản**: `f_contrast(t) = 1 + γ · mean(D_tone(tone(t), tone(n)))` với `n` là token lân cận

Ma trận tương phản `D_tone` (6×6) đo khoảng cách ngữ âm giữa các thanh (0.0–0.9). Mặc định: α=0.5, β=0.3, γ=0.4.

### 2. Morphology-Aware Scoring

Phân loại token thành 5 lớp từ, mỗi lớp có ngân sách nén riêng:

| Word Class | Hệ số giữ | Giải thích |
|-----------|-----------|------------|
| FUNC | 0.30 | Hư từ (và, thì, của, ...) — ưu tiên nén |
| CONTENT | 0.85 | Thực từ — giữ phần lớn |
| REDUP | 0.50 | Từ láy — nén vì dư thừa ngữ nghĩa |
| COMPOUND | 0.95 | Từ ghép — gần như giữ nguyên |
| OTHER | 0.50 | Còn lại |

**Cơ chế đặc biệt**:
- **Reduplicative Pair Merging**: Token ghép láy được gộp, token phụ đặt điểm 0.1.
- **Class-Aware Budget Allocation**: Ngân sách phân bổ theo tỷ lệ `|class| × keep_ratio`.

### 3. External Perplexity Scoring

Dùng mô hình nhỏ (SmolLM2-135M, ~0.3GB VRAM ở INT4) để tính độ quan trọng:

```
importance(tᵢ) = −log P(tᵢ | context)
```

Token có perplexity cao = bất ngờ = quan trọng. Xử lý theo sliding window (512 token).

### Thuật toán chọn token

1. Luôn giữ `k` token biên (đầu/cuối, mặc định 2).
2. Tính điểm `S(t)` cho mọi token ở giữa.
3. Chọn Top-K với `K = max(n/R − 2k, 0)`.
4. Tái tạo chuỗi theo thứ tự gốc.

## Kiến trúc phần cứng (3 mức)

| Mode | VRAM | Thành phần | Mô tả |
|------|------|-----------|-------|
| `no_model` | 0 GB | Tone + Morphology | CPU, heuristic ngôn ngữ thuần túy |
| `lightweight` | ~0.3 GB | + Tiny model scorer | Thêm perplexity từ SmolLM2-135M |
| `full` | ~7.8 GB | + INT4 7B generation | Pipeline đầy đủ với Qwen2.5-7B |

## VCC-Bench

Bộ đánh giá nén ngữ cảnh đầu tiên cho tiếng Việt với 5 tác vụ:

- **Long-Document QA**: Trả lời câu hỏi trên tài liệu dài
- **Multi-turn Conversation**: Hội thoại nhiều lượt
- **Needle-in-Haystack**: Truy xuất thông tin trong ngữ cảnh lớn
- **Agent Tool-Calling**: Gọi công cụ qua ngữ cảnh nén
- **Cross-lingual Compression**: Nén đa ngữ (Việt-Anh)

**Metrics**: ROUGE-L, BLEU, BERTScore, Exact Match, **Tone Preservation Rate**, Harmonized Score.

## Yêu cầu hệ thống

- **Python**: 3.10+ (đã kiểm thử trên 3.11 và 3.13).
- **CPU-only**: đủ để chạy toàn bộ test suite, lint, smoke test, và các compressor `no_model`/`none`/`random` (không cần GPU, không cần model lớn).
- **GPU (tuỳ chọn)**: cần cho benchmark/training với model thật (Qwen2.5-7B, v.v). Xem bảng VRAM ở mục [Kiến trúc phần cứng](#kiến-trúc-phần-cứng-3-mức) bên dưới — tối thiểu một GPU 16GB (T4/P100) cho pipeline `full`.

## Cài đặt

```bash
git clone https://github.com/thanthienhai/vncompress.git
cd vncompress
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r vncompress/requirements.txt
```

Để chỉ chạy trên CPU (test suite, smoke test, các compressor không cần model), có thể cài bản CPU-only của torch để giảm dung lượng tải:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r vncompress/requirements.txt
```

## Sử dụng

```bash
# Liệt kê phương pháp nén
python run_benchmark.py --list-methods

# Demo nhanh (một vài ví dụ, in kết quả chi tiết theo từng phương pháp)
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --demo

# Benchmark đầy đủ trên VCC-Bench
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --device cuda

# Ablation study (tách riêng từng tín hiệu: perplexity, tone, morphology)
python run_ablation.py --model Qwen/Qwen2.5-7B-Instruct --device cuda

# Huấn luyện tone-aware (model Qwen-family, LoRA)
python run_training.py --model Qwen/Qwen2.5-7B-Instruct --device cuda

# Huấn luyện LoRA cho SLM tiếng Việt nhỏ (tối ưu cho T4 16GB / GPU nhỏ hơn)
python run_train_slm.py --quick
python run_train_slm.py --batch-size 1 --max-length 128 --grad-accum 8  # GPU 6GB

# Đánh giá SLM đã huấn luyện (perplexity + tone-probe accuracy)
python evaluate_slm.py --adapter-dir trained_slm/final --tone-probe trained_slm/tone_probe.pt
```

`run_benchmark.py`, `run_ablation.py` và `run_training.py` hỗ trợ `--device cpu` (chậm hơn nhiều nhưng chạy được để thử nghiệm nhanh). `run_train_slm.py` và `evaluate_slm.py` yêu cầu GPU NVIDIA (CUDA) bắt buộc — không hỗ trợ CPU.

## Kiểm thử & CI

```bash
# Cài dev dependencies
pip install pytest ruff

# Chạy test suite (CPU-only, không cần GPU/model lớn, ~15s)
pytest tests/ -v

# Lint
ruff check vncompress tests

# CPU smoke test: import package + chạy nén end-to-end với tokenizer thật
python scripts/smoke_test.py
```

GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) chạy đúng ba bước trên (lint → test → smoke test) tự động trên mỗi push và pull request vào `main`, dùng torch CPU-only — không yêu cầu GPU hay model lớn.

## Cấu trúc thư mục

```
vncompress/
├── .github/workflows/ci.yml  # CI: lint, test suite, CPU smoke test
├── run_benchmark.py          # VCC-Bench evaluation
├── run_ablation.py           # Ablation study (tách từng tín hiệu LACC)
├── run_training.py           # Training pipeline (LoRA, tone-aware, model Qwen-family)
├── run_train_slm.py          # Training pipeline cho SLM tiếng Việt nhỏ
├── evaluate_slm.py           # Đánh giá SLM: perplexity + tone-probe accuracy
├── tests/                    # pytest test suite (CPU-only, MockTokenizer)
├── scripts/                  # Tiện ích: build VCC-Bench, fetch data, eval, smoke test
├── paper/
│   └── lacc_paper.tex        # Full LaTeX paper
├── vncompress/
│   ├── compressors/          # Base, tone_aware, llmlingua, snapkv, external_scorer, no_model
│   ├── tone_aware/           # Tones, scoring, linguistics
│   ├── morphology/           # merge_policy, word classes
│   ├── evaluation/           # VCC-Bench metrics
│   ├── calibration/          # Weight/parameter search cho scoring blend
│   └── docs/                 # Tài liệu toán học (math_framework.md)
├── pyproject.toml            # Cấu hình ruff (lint)
└── pytest.ini                # Cấu hình pytest
```

## Trích dẫn

```bibtex
@misc{thanthien2026lacc,
  title={LACC: Language-Aware Context Compression for Vietnamese},
  author={Thanthien},
  year={2026},
  note={Proposed method with tone-aware and morphology-aware scoring}
}
```

## Tham khảo chính

- LLMLingua (EMNLP 2023): Jiang et al., *LLMLingua: Compressing Prompts for Accelerated Inference*
- SnapKV (2024): Li et al., *SnapKV: LLM Knows What You are Looking for*
- StreamingLLM (2023): Xiao et al., *Efficient Streaming Language Models with Attention Sinks*
- H2O (2023): Zhang et al., *Heavy-Hitter Oracle for Efficient Generative Inference*
- SeCo (2026): Chen et al., *Semantic Compression with Large Language Models*
