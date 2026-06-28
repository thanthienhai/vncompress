# Latent-Space Compression trong LLM — Báo cáo chuyên sâu

**Ngày:** 28/06/2026 | **Người tổng hợp:** Thanthien

> 📂 **Thư mục:** Các báo cáo liên quan:
> - [README.md](README.md) — Báo cáo tổng quan Latent-Space Compression (file hiện tại)
> - [research_gaps.md](research_gaps.md) — Phân tích 12 Research Gaps & đề xuất hướng nghiên cứu
> - [low_resource_language.md](low_resource_language.md) — Chuyên sâu: Low-Resource Language Context Compression (Tiếng Việt)

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Phân loại các hướng tiếp cận](#2-phân-loại-các-hướng-tiếp-cận)
3. [Các paper nền tảng](#3-các-paper-nền-tảng)
4. [Các paper mở rộng và cải tiến gần đây](#4-các-paper-mở-rộng-và-cải-tiến-gần-đây)
5. [So sánh các phương pháp](#5-so-sánh-các-phương-pháp)
6. [Open-source repositories](#6-open-source-repositories)
7. [Thách thức và hướng phát triển](#7-thách-thức-và-hướng-phát-triển)
8. [Tài liệu tham khảo](#8-tài-liệu-tham-khảo)

---

## 1. Tổng quan

### 1.1. Vấn đề

Trong các mô hình ngôn ngữ lớn (LLM), context window có giới hạn cố định (thường 4K–128K token với hầu hết model, và lên đến 1M+ token với một số model mới). Khi xử lý các tác vụ yêu cầu lượng lớn context như:

- Phân tích tài liệu dài (sách, báo cáo kỹ thuật, codebase)
- Hội thoại nhiều lượt (multi-turn conversations)
- Agent thực thi nhiều bước (multi-step agent tasks)
- Retrieval-Augmented Generation (RAG) với nhiều document chunks

...lượng token đầu vào vượt quá khả năng xử lý của model hoặc gây ra chi phí tính toán quá lớn (self-attention có độ phức tạp O(n²)).

### 1.2. Tại sao Latent-Space Compression?

Các phương pháp nén truyền thống hoạt động trong **token space** (xóa token, chọn subset token) có hạn chế cố hữu:

| Hạn chế | Mô tả |
|---------|-------|
| **Mất thông tin vĩnh viễn** | Token bị xóa là mất hoàn toàn, không thể phục hồi |
| **Phá vỡ coherence** | Xóa token riêng lẻ làm đứt gãy cấu trúc ngữ nghĩa |
| **Không tận dụng được redundancy** | Redundancy trong embedding space không được khai thác triệt để |
| **Discrete decision** | Quyết định xóa/giữ là rời rạc (binary), không có trung gian |

**Latent-space compression** giải quyết các hạn chế này bằng cách nén thông tin vào **continuous embedding space**, nơi:

- Thông tin từ nhiều token được "trộn" (merge) thành các vector embedding đại diện
- Compression ratio có thể đạt rất cao (10x–100x) nhờ khả năng nén liên tục
- Quá trình nén có thể học được (learnable) hoặc dựa trên cấu trúc ngữ nghĩa
- Thông tin được bảo toàn tốt hơn nhờ tính chất continuous

### 1.3. Sơ đồ phân loại

```
Latent-Space Compression
│
├── 1. Memory/Special Tokens (học token đặc biệt để chứa thông tin nén)
│   ├── Gist Tokens (NeurIPS 2023)
│   ├── ICAE (ICLR 2024)
│   ├── AutoCompressor (EMNLP 2023)
│   └── MMCompressor (2025)
│
├── 2. Embedding Merging (gộp embedding trong không gian liên tục)
│   ├── Token Merging (ToMe) — gốc từ Vision
│   ├── K-Token Merging (2026)
│   ├── SeCo — Semantic Consistency Weighted Merging (2026)
│   └── AVOC — Retrieval-Inspired Multi-modal Merging (2026)
│
├── 3. Low-Rank Projection (chiếu xuống không gian thấp chiều)
│   ├── STAR-KV (2026)
│   ├── LoRA-based compression
│   └── SVD-based KV cache compression
│
├── 4. Latent Communication (giao tiếp liên tục giữa các agent/model)
│   ├── Hidden state sharing
│   ├── KV-cache sharing
│   └── Embedding-based protocol
│
└── 5. Semantic Hierarchy Compression (nén dựa trên cấu trúc ngữ nghĩa phân cấp)
    ├── H2MT (2026)
    └── Hierarchical document compression
```

---

## 2. Phân loại các hướng tiếp cận

### 2.1. Memory/Special Tokens

**Ý tưởng cốt lõi:** Huấn luyện (hoặc thêm vào) model một tập các "memory tokens" hoặc "gist tokens" đặc biệt. Khi đưa context dài vào, model học cách nén thông tin từ context vào các token này. Sau đó, thay vì giữ toàn bộ context, chỉ cần giữ các memory tokens này cho các bước xử lý tiếp theo.

**Cơ chế:**
1. Thêm `M` token đặc biệt vào vocabulary
2. Huấn luyện model tái tạo thông tin từ các token này (autoencoding objective)
3. Khi inference: context dài → nén vào M memory tokens → chỉ dùng M tokens cho downstream tasks

**Đặc điểm:**
- ✅ Compression ratio rất cao (số memory tokens cố định, bất kể context dài bao nhiêu)
- ✅ Có thể cache và tái sử dụng
- ✅ Linh hoạt: có thể áp dụng cho nhiều task khác nhau
- ❌ Cần huấn luyện thêm (fine-tuning)
- ❌ Khó interpret: không biết memory tokens chứa thông tin gì
- ❌ Có thể mất thông tin chi tiết (lossy compression)

### 2.2. Embedding Merging

**Ý tưởng cốt lõi:** Thay vì thêm token đặc biệt, nhóm các token có đặc điểm tương đồng (về mặt ngữ nghĩa) lại và gộp embedding của chúng thành một embedding duy nhất. Việc gộp có thể là averaging, weighted averaging (dựa trên attention), hoặc học một phép biến đổi.

**Cơ chế:**
1. Xác định các nhóm token cần gộp (dựa trên position, similarity, hoặc query relevance)
2. Áp dụng phép gộp (merge) trong embedding space
3. Chuỗi embedding đã được rút gọn đi qua các layer tiếp theo của transformer

**Đặc điểm:**
- ✅ Có thể training-free (không cần fine-tune)
- ✅ Giữ được cấu trúc ngữ nghĩa (nếu merge đúng cách)
- ✅ Linh hoạt về compression ratio
- ❌ Merge quá nhiều token có thể gây "over-smoothing"
- ❌ Khó xác định chính xác token nào nên merge với nhau

### 2.3. Low-Rank Projection

**Ý tưởng cốt lõi:** Khai thác tính chất low-rank của ma trận Key-Value trong attention để nén dọc theo hidden dimension (thay vì nén theo sequence length như các phương pháp khác).

**Cơ chế:**
1. Phân tích ma trận K, V thành dạng low-rank (SVD, adaptive rank selection)
2. Lưu trữ dạng nén thay vì dạng đầy đủ
3. Khi cần attention: giải nén hoặc tính toán trực tiếp trên dạng nén

**Đặc điểm:**
- ✅ Tận dụng được redundancy trong hidden dimension
- ✅ Có thể kết hợp với quantization để đạt compression ratio rất cao
- ❌ Cần xác định rank phù hợp cho từng head/layer
- ❌ Overhead giải nén trong quá trình decode

### 2.4. Latent Communication

**Ý tưởng cốt lõi:** Trong multi-agent systems, thay vì giao tiếp bằng natural language (tốn token), các agent trao đổi trực tiếp continuous representations (embeddings, hidden states, hoặc KV-caches).

**Cơ chế:**
1. Agent A xử lý context → sinh ra continuous representation
2. Truyền representation này cho Agent B (thay vì text)
3. Agent B sử dụng representation này như một phần của input

**Đặc điểm:**
- ✅ Loại bỏ bottleneck của text generation
- ✅ Giảm chi phí inference đáng kể
- ❌ Khó interpret và debug
- ❌ Cần alignment giữa các model khác nhau

---

## 3. Các paper nền tảng

### 3.1. Gist Tokens — "Learning to Compress Prompts with Gist Tokens"

| Thuộc tính | Chi tiết |
|------------|----------|
| **arxiv** | 2304.08467 |
| **Venue** | **NeurIPS 2023** |
| **Authors** | Jesse Mu, Xiang Lisa Li, Noah Goodman (Stanford) |
| **Code** | Tích hợp trong LLaMA fine-tuning framework |

**Tóm tắt kỹ thuật:**

Gist tokens là một trong những paper đầu tiên và có ảnh hưởng nhất trong hướng latent-space compression. Ý tưởng chính:

1. **Thêm K "gist tokens"** vào trước prompt. Các token này được huấn luyện để "hấp thụ" thông tin từ prompt.
2. **Modified attention mask**: trong quá trình huấn luyện, các gist tokens được phép attend đến toàn bộ prompt, nhưng prompt tokens KHÔNG được attend đến các token sau prompt (chỉ được attend đến gist tokens). Điều này buộc thông tin phải "chảy" qua gist tokens.
3. **Khi inference**: chỉ cần forward-pass prompt một lần để compute KV của gist tokens, sau đó cache KV này và tái sử dụng cho tất cả các request tiếp theo với cùng prompt.

```
Standard Prompting:
  [Prompt tokens...] [Instruction] → Model → Output
  Mỗi lần gọi đều phải encode lại toàn bộ prompt.

Gisting:
  [GIST][GIST]...[GIST] [Prompt...] → Model → KV cache của GIST
  [GIST (cached)] [Instruction] → Model → Output
  Chỉ encode prompt MỘT LẦN.
```

**Kết quả chính:**
- Compression lên đến **26x** trên LLaMA-7B
- Giảm **40% FLOPs**, tăng **4.2% wall-time speedup**
- Chất lượng đầu ra gần như không đổi
- Áp dụng cho cả decoder-only (LLaMA) và encoder-decoder (FLAN-T5)

**Điểm hạn chế:**
- Cần huấn luyện (fine-tuning) model
- Số lượng gist tokens (K) là hyperparameter cần tune
- Hiệu quả phụ thuộc vào độ dài prompt: prompt càng dài, lợi ích càng lớn

---

### 3.2. ICAE — "In-context Autoencoder for Context Compression"

| Thuộc tính | Chi tiết |
|------------|----------|
| **arxiv** | 2307.06945 |
| **Venue** | **ICLR 2024** (Spotlight) |
| **Authors** | Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, Furu Wei (Microsoft) |
| **Code** | Có public |

**Tóm tắt kỹ thuật:**

ICAE (In-context Autoencoder) mở rộng ý tưởng của Gist Tokens lên một tầm cao mới, sử dụng kiến trúc autoencoder:

1. **Encoder**: Một LLM nhỏ (pretrained) được fine-tune để encode context dài thành một tập các "memory slots" (compact embeddings).
2. **Decoder**: LLM chính (lớn hơn, frozen hoặc fine-tune nhẹ) nhận memory slots này làm điều kiện (conditioning) để sinh output.
3. **Huấn luyện 2 giai đoạn**:
   - **Pretraining**: autoencoding objective (tái tạo context từ memory slots) + language modeling objective
   - **Fine-tuning**: trên instruction data cho các task cụ thể

```
Kiến trúc ICAE:

  Context dài (N tokens)
       │
       ▼
  ┌─────────────────┐
  │  ICAE Encoder    │  ← LLM nhỏ (~1% params của decoder)
  │  (Pretrained LM) │
  └────────┬────────┘
           │
           ▼
  Memory Slots (M << N tokens)
       │
       ▼
  ┌─────────────────┐
  │  LLM Decoder     │  ← LLM chính (frozen)
  │  + Memory Slots  │
  └────────┬────────┘
           │
           ▼
       Output
```

**Kết quả chính:**
- Compression **4x** với Llama (chỉ thêm ~1% parameters)
- Giảm latency và GPU memory trong inference
- Cho thấy mối liên hệ thú vị giữa "working memory" trong cognitive science và representation learning trong LLM
- Memory slots hoạt động như "bộ nhớ làm việc" (working memory) của model

**Điểm đặc biệt:**
- Là paper đầu tiên đặt nền móng cho việc hiểu context compression qua lăng kính cognitive science
- Pretraining trên massive text data → memory slots có khả năng tổng quát hóa tốt
- Thiết kế module hóa: encoder và decoder tách biệt, có thể thay thế độc lập

---

### 3.3. AutoCompressor — "Adapting Language Models to Compress Contexts"

| Thuộc tính | Chi tiết |
|------------|----------|
| **Venue** | **EMNLP 2023** |
| **Authors** | Alexis Chevalier, Alexander Wettig, Anirudh Ajith, Danqi Chen (Princeton) |

**Tóm tắt kỹ thuật:**

AutoCompressor đề xuất một phương pháp **recursive compression** — nén context thành summary tokens, sau đó dùng chính summary tokens này để tiếp tục nén context mới:

1. **Summary tokens**: Thêm M "summary tokens" vào cuối mỗi segment của document.
2. **Segment-by-segment processing**: Document được chia thành các segment. Mỗi segment được xử lý tuần tự, summary tokens từ segment trước được đưa vào làm prefix cho segment sau.
3. **Recursive compression**: Summary tokens liên tục tích lũy và cập nhật thông tin từ các segment mới.
4. **Huấn luyện**: LLM được fine-tune với objective: dự đoán token tiếp theo trong segment, đồng thời summary tokens phải chứa đủ thông tin để hỗ trợ dự đoán.

```
AutoCompressor:

  Seg 1 → [Summary₁] → cache
  Seg 2 + [Summary₁] → [Summary₂] → cache
  Seg 3 + [Summary₂] → [Summary₃] → cache
  ...
  Seg N + [Summary_{N-1}] → [Summary_N]

  Khi query: [Summary_N] + [Query] → LLM → Output
```

**Kết quả chính:**
- Nén document dài thành chỉ vài summary tokens
- Có thể xử lý document dài vô hạn (về mặt lý thuyết) qua recursive compression
- Hiệu quả tốt trên các tác vụ QA và summarization

**Điểm hạn chế:**
- Xử lý tuần tự (không parallel được)
- Tích lũy lỗi qua các bước compression
- Khó áp dụng cho các tác vụ cần cross-segment reasoning

---

### 3.4. LLoCO — "Learning Long Contexts Offline"

| Thuộc tính | Chi tiết |
|------------|----------|
| **Venue** | **2024** |
| **Authors** | Sijun Tan, Xiuyu Li, Shishir Patil, et al. (UC Berkeley) |

**Tóm tắt kỹ thuật:**

LLoCO kết hợp ý tưởng từ AutoCompressor và context compression để xử lý long documents:

1. **Offline compression**: Document dài được xử lý offline qua AutoCompressor, tạo ra các summary tokens.
2. **Online inference**: Khi có query, model chỉ cần xử lý summary tokens (đã được cache) + query tokens.
3. **Fine-tuning**: Model được fine-tune để sử dụng summary tokens hiệu quả.

**Kết quả chính:**
- Giảm đáng kể chi phí inference cho long-document tasks
- Giữ được chất lượng tương đương với xử lý full context
- Phù hợp cho các ứng dụng cần xử lý nhiều documents (RAG pipeline)

---

## 4. Các paper mở rộng và cải tiến gần đây (2025–2026)

### 4.1. SeCo — "Semantic Consistency Context Compression"

| Thuộc tính | Chi tiết |
|------------|----------|
| **arxiv** | 2605.09463 |
| **Năm** | 2026 |
| **Code** | Có public |

**Đóng góp chính:**

SeCo chuyển từ **position-driven** (gộp token theo vị trí) sang **semantic-driven** compression (gộp token theo ngữ nghĩa):

1. **Query-relevant semantic centers**: Chọn các token có độ liên quan cao với query làm "trung tâm ngữ nghĩa"
2. **Consistency-weighted merging**: Các token xung quanh được gộp vào semantic center với trọng số dựa trên độ nhất quán ngữ nghĩa
3. **Dynamic anchoring**: Vị trí gộp không cố định mà thay đổi theo query và context

```
SeCo:

  Position-driven (cũ):       Semantic-driven (SeCo):
  [T₁ T₂ T₃] [T₄ T₅ T₆]      [T₁] [T₂ T₃ T₅] [T₄ T₆]
  → gộp theo block cố định    → gộp theo semantic similarity

  Kết quả:                    Kết quả:
  ✗ Semantic fragmentation    ✓ Semantic coherence preserved
  ✗ Position bias             ✓ Position-invariant
```

**Kết quả:**
- Vượt trội trên 14 benchmarks, 2 backbone models
- Cải thiện cả downstream tasks, inference latency, và out-of-domain robustness

---

### 4.2. K-Token Merging — "Compressing Sequences in the Latent Embedding Space"

| Thuộc tính | Chi tiết |
|------------|----------|
| **arxiv** | 2604.15153 |
| **Năm** | 2026 |
| **Code** | [github.com/shsjxzh/K-Token-Merging](https://github.com/shsjxzh/K-Token-Merging) |

**Đóng góp chính:**

1. **Lightweight encoder**: Mỗi block K token liên tiếp được gộp thành 1 embedding qua một lightweight encoder (MLP hoặc tiny transformer)
2. **LoRA-adapted LLM**: LLM được fine-tune nhẹ với LoRA để thích nghi với compressed sequence
3. **Original vocabulary generation**: Đầu ra vẫn là text bình thường (không cần decode từ latent)

**Kết quả:**
- Giảm **75% input length** với performance degradation tối thiểu
- Nằm trên **Pareto frontier** của performance vs compression
- Áp dụng cho nhiều loại tác vụ: structural reasoning, sentiment classification, code editing

---

### 4.3. STAR-KV — "Low-Rank KV Cache Compression via Soft Thresholding"

| Thuộc tính | Chi tiết |
|------------|----------|
| **arxiv** | 2606.08382 |
| **Năm** | 2026 |
| **Code** | [github.com/PriyanshBhatnagar/STAR-KV](https://github.com/PriyanshBhatnagar/STAR-KV) |

**Đóng góp chính:**

1. **Adaptive low-rank compression**: Thay vì chọn rank cố định, dùng differentiable soft thresholding để chọn rank tối ưu cho từng head và block
2. **Hybrid decomposition**: Key và Value được decompose với các chiến lược khác nhau (do chúng có sensitivity khác nhau)
3. **Mixed-precision quantization**: Kết hợp low-rank với quantization cho compression ratio tối đa

**Kết quả:**
- **75% KV cache compression** (chỉ với low-rank)
- **20x overall** khi kết hợp quantization
- **6.9x speedup** attention module, **3.1x** end-to-end throughput

---

### 4.4. MMCompressor — Multi-modal Compression into KV Cache

| Thuộc tính | Chi tiết |
|------------|----------|
| **Năm** | 2025 |
| **Code** | [github.com/asvilesov/MMCompressor](https://github.com/asvilesov/MMCompressor) |

**Đóng góp chính:**

MMCompressor mở rộng ý tưởng memory tokens từ text sang multi-modal:

1. **Cross-modal compression**: Text, image, và video được nén vào cùng một không gian memory tokens trong KV cache
2. **Chỉ cần 1 token**: Một ảnh hoặc đoạn video có thể được nén thành chỉ 1 memory token
3. **Editable memory**: Memory tokens có thể được truy vấn, suy luận, và chỉnh sửa

```
MMCompressor:

  Text:   "The cat sat on..."  ──┐
  Image:  [🐱 on mat]          ──┼──→ [Memory Token] → LLM
  Video:  [▶️ 30s clip]        ──┘

  Compression ratio: Image = hàng trăm tokens → 1 token
```

---

### 4.5. AVOC — "Retrieval-Inspired Audio-Video Token Compression"

| Thuộc tính | Chi tiết |
|------------|----------|
| **arxiv** | 2606.24286 |
| **Năm** | 2026 |

**Đóng góp chính:**

1. **Top-K retrieval formulation**: Coi multi-modal token compression như bài toán retrieval — chọn K token "quan trọng nhất" từ tập lớn
2. **3 tiêu chí từ Information Retrieval**:
   - **Relevance**: token có liên quan đến query không?
   - **Importance**: token quan trọng với context không?
   - **Diversity**: token có đa dạng, không trùng lặp không?
3. **Ứng dụng**: Audio-video dài đến 1 giờ

**Kết quả:**
- **SOTA** trên OmniVideoBench (+4.9 điểm) và LVOmniBench (+5.5 điểm)
- Robust trên Needle-in-a-Haystack cho audio-video duration lên đến 1 giờ

---

### 4.6. H2MT — "Semantic Hierarchy-Aware Hierarchical Memory Transformer"

| Thuộc tính | Chi tiết |
|------------|----------|
| **arxiv** | 2605.24930 |
| **Năm** | 2026 |

**Đóng góp chính:**

1. **Semantic hierarchy tree**: Xây dựng cây phân cấp ngữ nghĩa offline từ document (ví dụ: chapter → section → paragraph)
2. **Bottom-up memory embedding**: Mỗi node trong cây có một memory embedding được tính từ dưới lên (từ leaf đến root)
3. **Coarse-to-fine pruning**: Khi query, duyệt cây từ gốc xuống, prune những nhánh không liên quan

**Kết quả:**
- Competitive quality-efficiency trade-off
- Giảm peak GPU memory và TTFT (Time-To-First-Token)

---

### 4.7. Beyond tokens — "A Unified Framework for Latent Communication in LLM Multi-Agent"

| Thuộc tính | Chi tiết |
|------------|----------|
| **arxiv** | 2606.05711 |
| **Năm** | 2026 |

**Đóng góp chính:**

Paper survey này đề xuất một framework 3 trục để phân loại các phương pháp latent communication:

1. **WHAT**: Thông tin gì được truyền? (Embeddings, Hidden States, KV-Caches)
2. **WHICH**: Alignment giữa sender và receiver như thế nào? (Latent-space alignment, Layer alignment)
3. **HOW**: Làm sao để fuse thông tin vào receiver? (Concatenation, Prepending, Cross-attention, Cache restoration)

Khảo sát **18 methods** từ 2024–2026, xác định 5 design patterns chính.

---

### 4.8. Token Merging (ToMe) — Gốc từ Vision, ứng dụng cho LLM

| Thuộc tính | Chi tiết |
|------------|----------|
| **arxiv** | 2210.09461 (Vision) → mở rộng cho LLM 2024 |
| **Venue** | ICLR 2023 |

**Đóng góp chính:**

Mặc dù xuất phát từ Vision Transformers, ToMe đã được mở rộng cho LLM:

1. **Bipartite matching**: Chia token thành 2 tập, match các cặp token tương đồng
2. **Merge bằng averaging**: Embedding của 2 token được gộp thành 1
3. **Training-free**: Không cần huấn luyện, áp dụng trực tiếp trong inference

**Kết quả (cho Vision):**
- Giảm **2x FLOPs** với <0.5% accuracy drop
- Có thể tích hợp vào bất kỳ ViT architecture nào

**Mở rộng cho LLM:**
- Áp dụng tương tự cho self-attention trong transformer decoder
- Token merging theo từng layer hoặc theo block

---

## 5. So sánh các phương pháp

| Phương pháp | Năm | Venue | Training? | Compression Ratio | Ưu điểm chính | Nhược điểm chính |
|-------------|-----|-------|-----------|-------------------|---------------|------------------|
| **Gist Tokens** | 2023 | NeurIPS | Có (FT) | Lên đến 26x | Đơn giản, hiệu quả cao | Cần fine-tune, cố định K |
| **ICAE** | 2024 | ICLR | Có (2-stage) | 4x | Module hóa, nền tảng lý thuyết vững | Thêm 1% params, phức tạp |
| **AutoCompressor** | 2023 | EMNLP | Có (FT) | Tùy ý | Recursive, xử lý vô hạn | Tuần tự, tích lũy lỗi |
| **LLoCO** | 2024 | — | Có (FT) | Tùy ý | Offline + Online hiệu quả | Cần preprocess |—| **2024** |
| **Authors** | Sijun Tan, Xiuyu Li, Shishir Patil, et al. (UC Berkeley) |
| **arXiv** | [2404.07979](https://arxiv.org/abs/2404.07979) |
| **SeCo** | 2026 | — | Không* | Linh hoạt | Semantic-driven, position-invariant | Phụ thuộc query |
| **K-Token Merging** | 2026 | — | Có (LoRA) | 75% length ↓ | Đơn giản, Pareto optimal | Cần LoRA fine-tune |
| **STAR-KV** | 2026 | — | Không | 20x (với quant) | Adaptive rank, tốc độ cao | Overhead giải nén |
| **MMCompressor** | 2025 | — | Có | 100x+ (hình ảnh) | Multi-modal, 1 token/ảnh | Phức tạp, cần train |
| **AVOC** | 2026 | — | Có | Tùy budget | IR-inspired, SOTA video | Phức tạp 3 tiêu chí |
| **H2MT** | 2026 | — | Không* | Hierarchical | Cấu trúc, coarse-to-fine | Cần index offline |
| **ToMe (LLM)** | 2023→ | ICLR | Không | 2x FLOPs ↓ | Training-free, đơn giản | Merge thô, lossy |

*Có thể training-free hoặc cần fine-tune tùy phiên bản

---

## 6. Open-Source Repositories

| Repository | Mô tả | Link |
|------------|-------|------|
| **ICAE** (Microsoft) | In-context Autoencoder cho LLaMA | Code public |
| **Gist Tokens** | Implementation cho LLaMA fine-tuning | Tích hợp trong HF transformers |
| **AutoCompressor** | Recursive compression cho LLM | Code public |
| **K-Token Merging** | Gộp K token trong embedding space | [github.com/shsjxzh/K-Token-Merging](https://github.com/shsjxzh/K-Token-Merging) |
| **STAR-KV** | Low-rank + quantization KV compression | [github.com/PriyanshBhatnagar/STAR-KV](https://github.com/PriyanshBhatnagar/STAR-KV) |
| **MMCompressor** | Multi-modal compression vào KV cache | [github.com/asvilesov/MMCompressor](https://github.com/asvilesov/MMCompressor) |
| **SeCo** | Semantic consistency compression | [anonymous.4open.science/r/seco-EE5E](https://anonymous.4open.science/r/seco-EE5E) |
| **ToMe (SD)** | Token Merging cho Stable Diffusion và ViT | Tích hợp trong diffusers |
| **LLMLingua** | Prompt compression (có liên quan) | [github.com/microsoft/LLMLingua](https://github.com/microsoft/LLMLingua) |

---

## 7. Thách thức và hướng phát triển

### 7.1. Thách thức hiện tại

| Thách thức | Mô tả |
|------------|-------|
| **Alignment gap** | Compressed representation và không gian embedding của frozen LLM không khớp hoàn toàn |
| **Interpretability** | Không thể biết memory tokens/summary tokens chứa thông tin gì → khó debug |
| **Task-specificity** | Compression tốt cho task này có thể không tốt cho task khác |
| **Cross-architecture** | Latent representation từ model A không dùng được cho model B |
| **Streaming** | Hầu hết phương pháp yêu cầu thấy toàn bộ context trước khi compress |
| **Training cost** | Các phương pháp cần training đòi hỏi compute đáng kể |
| **Metric** | Thiếu metric chuẩn để đánh giá chất lượng compression trong latent space |
| **Security** | Latent communication mở ra attack surface mới: adversarial latent injection |

### 7.2. Hướng phát triển tiềm năng

1. **Universal latent representation**: Một không gian embedding chung cho phép compress một lần, dùng cho nhiều model
2. **Adaptive compression rate**: Tự động điều chỉnh compression ratio dựa trên:
   - Độ phức tạp của context
   - Độ khó của task
   - Budget tính toán/memory hiện tại
3. **Hierarchical compression**: Kết hợp nhiều mức compression (token → sentence → paragraph → document) trong cùng một framework
4. **Multi-modal universal memory**: Một không gian memory tokens duy nhất cho text, image, audio, video
5. **Training-free semantic compression**: Nén dựa trên semantic similarity mà không cần fine-tune (hướng SeCo đang làm)
6. **Compression-aware training**: Huấn luyện model từ đầu với awareness về compression, không cần fine-tune sau
7. **Lossless latent compression**: Đảm bảo không mất thông tin trong quá trình nén (có thể phục hồi hoàn toàn)
8. **Integration với MCP (Model Context Protocol)**: Dùng latent compression như một transport layer trong MCP

### 7.3. Các câu hỏi nghiên cứu mở

- Làm thế nào để đo lường "information content" của một compressed embedding?
- Liệu có tồn tại một "critical compression ratio" mà dưới đó performance sụp đổ không?
- Mối quan hệ giữa compression trong latent space và "working memory" trong cognitive science?
- Làm sao để đảm bảo fairness và safety khi dùng latent compression?
- Liệu có thể học một "universal compressor" cho mọi LLM?

---

## 8. Tài liệu tham khảo

### Papers chính

1. **Gist Tokens**: Mu, J., Li, X.L., Goodman, N. "Learning to Compress Prompts with Gist Tokens." NeurIPS 2023. arxiv:2304.08467
2. **ICAE**: Ge, T., Hu, J., Wang, L., Wang, X., Chen, S.Q., Wei, F. "In-context Autoencoder for Context Compression in a Large Language Model." ICLR 2024. arxiv:2307.06945
3. **AutoCompressor**: Chevalier, A., Wettig, A., Ajith, A., Chen, D. "Adapting Language Models to Compress Contexts." EMNLP 2023.
4. **LLoCO**: Tan, S., Li, X., Patil, S., et al. "LLoCO: Learning Long Contexts Offline." 2024. [arxiv:2404.07979](https://arxiv.org/abs/2404.07979)
5. **SeCo**: Tang, J., Huang, Z., Zhang, X., et al. "Beyond Position Bias: Shifting Context Compression from Position-Driven to Semantic-Driven." 2026. arxiv:2605.09463
6. **K-Token Merging**: Xu, Z., Harvill, J., Fan, Z., et al. "Compressing Sequences in the Latent Embedding Space: K-Token Merging for LLMs." 2026. arxiv:2604.15153
7. **STAR-KV**: Bhatnagar, P., Moradifirouzabadi, A., et al. "STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding." 2026. arxiv:2606.08382
8. **AVOC**: Chen, Y., Tan, W., Yu, X., et al. "AVOC: Enhancing Hour-Level Audio-Video Understanding in Omni-Modal LLMs via Retrieval-Inspired Token Compression." 2026. arxiv:2606.24286
9. **H2MT**: Haghifam, M., He, Z., Cong, J., Sun, Y. "Semantic Hierarchy-Aware Hierarchical Memory Transformer." 2026. arxiv:2605.24930
10. **Latent Communication Survey**: Liu, Y. "Beyond tokens: A Unified Framework for Latent Communication in LLM-based Multi-agent Systems." 2026. arxiv:2606.05711
11. **ToMe**: Bolya, D., Fu, C.Y., Dai, X., et al. "Token Merging: Your ViT But Faster." ICLR 2023. arxiv:2210.09461
12. **MMCompressor**: Vilesov, A. "MMcompress: Multi-modal Compression into KV Cache." 2025. github.com/asvilesov/MMCompressor

### Papers liên quan (KV cache compression)

13. **CompressKV**: Lin, X., Wang, J., et al. "CompressKV: Semantic-Retrieval-Guided KV-Cache Compression." 2026. arxiv:2606.24467
14. **HyperQuant**: Domb, Y., et al. "HyperQuant: A Rate-Distortion-Optimal Quantization Pipeline." 2026. arxiv:2606.23406
15. **Beyond Uniform Tokens**: Gan, J., et al. "Adaptive Compression for Time Series Language Models." 2026. arxiv:2606.13624

---

*Báo cáo được tổng hợp từ các nguồn: arXiv, GitHub, conference proceedings (NeurIPS, ICLR, EMNLP, ACL, ICML).*

*Cập nhật lần cuối: 28/06/2026*
