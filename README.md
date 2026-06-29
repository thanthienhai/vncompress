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

## Cài đặt

```bash
git clone https://github.com/thanthien/vncompress.git
cd vncompress
pip install -r vncompress/requirements.txt
```

## Sử dụng

```bash
# Benchmark đầy đủ
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --device cuda

# Demo nhanh
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --demo

# Chạy trên Colab/Kaggle (T4 16GB)
python run_colab.py --auto

# Huấn luyện tone-aware
python run_training.py --model Qwen/Qwen2.5-7B-Instruct --mode tone_aware

# Liệt kê phương pháp nén
python run_benchmark.py --list-methods
```

## Cấu trúc thư mục

```
vncompress/
├── run_benchmark.py          # VCC-Bench evaluation
├── run_training.py           # Training pipeline (LoRA, tone-aware)
├── run_colab.py              # Colab/Kaggle T4-optimized
├── paper/
│   └── lacc_paper.tex        # Full LaTeX paper
├── vncompress/
│   ├── compressors/          # Base, tone_aware, llmlingua, snapkv, external_scorer
│   ├── tone_aware/           # Tones, scoring, linguistics
│   ├── morphology/           # merge_policy, word classes
│   ├── evaluation/           # VCC-Bench metrics
│   └── docs/                 # Mathematical documentation
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
