# Research Gaps & Đề xuất hướng nghiên cứu — Latent-Space Compression cho LLM

**Ngày:** 28/06/2026 | **Người phân tích:** Thanthien

---

## 1. Tổng quan bối cảnh

Khoảng **2.5 năm** kể từ Gist Tokens (NeurIPS 2023) — paper khởi đầu cho latent-space compression trong LLM — lĩnh vực này đã phát triển nhanh chóng. Tuy nhiên, vẫn còn nhiều **khoảng trống nghiên cứu** (research gaps) chưa được giải quyết, đặc biệt là trong bối cảnh:

- Bùng nổ **LLM agents** với context management phức tạp
- Xu hướng **multi-modal** (text + image + audio + video)
- Nhu cầu **on-device/edge deployment** với tài nguyên hạn chế
- Yêu cầu về **safety & interpretability** ngày càng cao
- **Tiếng Việt và các ngôn ngữ low-resource** gần như chưa được nghiên cứu

---

## 2. Phân tích 12 Research Gaps

### Đánh giá theo thang điểm 1-5 cho từng tiêu chí:
- **Novelty (N):** Mức độ mới, chưa được khám phá
- **Impact (I):** Tầm ảnh hưởng nếu giải quyết được
- **Feasibility (F):** Tính khả thi với nguồn lực hạn chế
- **Fit (Fi):** Mức độ phù hợp với dự án `vncompress`

### Gap 1: Universal Cross-Architecture Latent Compression ⭐⭐⭐

| N | I | F | Fi |
|---|---|---|---|
| 4 | 4 | 2 | 2 |

**Mô tả:**
Hiện tại, mọi phương pháp latent compression (Gist Tokens, ICAE, AutoCompressor, SeCo...) đều **gắn chặt với một model cụ thể**. Không có cách nào để:
- Nén context bằng model A, rồi dùng compressed representation cho model B
- Chia sẻ memory tokens giữa các model khác nhau (vd: từ Llama sang Qwen)
- Xây dựng một "universal compressor" hoạt động độc lập với LLM backend

**Vấn đề cụ thể:**
- Embedding dimension của mỗi model khác nhau (4096 vs 8192 vs ...)
- Tokenizer và vocabulary khác nhau
- Kiến trúc attention khác nhau (MHA, GQA, MLA...)

**Hướng tiếp cận đề xuất:**
- Học một **projection layer** (adapter) giữa latent spaces của các model khác nhau
- Dùng **contrastive learning** để align compressed representations
- Hoặc: thiết kế **model-agnostic compression tokens** nằm ở input level, không phụ thuộc vào hidden dim

---

### Gap 2: Interpretable/Auditable Compression 🔥🔥🔥

| N | I | F | Fi |
|---|---|---|---|
| 5 | 5 | 4 | 4 |

**Mô tả:**
**KHÔNG AI BIẾT** memory tokens/gist tokens thực sự chứa thông tin gì. Khi compression fail:
- Là do mất thông tin gì?
- Token nào bị nén sai?
- Làm sao để sửa mà không cần train lại?

Đây là gap **lớn nhất** và **quan trọng nhất** trong toàn bộ lĩnh vực.

**Vấn đề cụ thể:**
- Memory tokens là "black box" — không thể decode ngược lại thành text
- Không có metric nào đo "information content" của một compressed embedding
- Khi model trả lời sai, không biết là do compression mất thông tin hay do model yếu

**Hướng tiếp cận đề xuất:**

**2a. Trainable Decoder cho Memory Tokens:**
- Huấn luyện một lightweight decoder có thể "giải nén" memory tokens ngược lại thành text
- Cho phép kiểm tra: "sau khi nén, model còn biết gì về context gốc?"
- Similar to probing classifiers nhưng cho compressed representations

**2b. Information-Theoretic Metrics:**
- Phát triển metric đo lượng thông tin được bảo toàn sau compression
- Dùng mutual information I(X; Z) giữa context gốc X và compressed representation Z
- Thiết lập "information budget" — biết chính xác bao nhiêu bits thông tin còn lại

**2c. Token Attribution in Latent Space:**
- Khi model dùng memory token để trả lời, token gốc nào đóng góp nhiều nhất?
- Similar to attention attribution nhưng cho compressed setting
- Cho phép "trace back" từ output về source context

**Tính khả thi cao** vì có thể thực hiện trên model đã có sẵn, không cần train LLM từ đầu.

---

### Gap 3: Adaptive Online Compression Rate 🔥🔥

| N | I | F | Fi |
|---|---|---|---|
| 4 | 4 | 3 | 3 |

**Mô tả:**
Mọi phương pháp hiện tại dùng **compression ratio cố định**. Nhưng:
- Một câu hỏi đơn giản không cần nén nhiều như một câu hỏi phức tạp
- Context đơn giản (tin tức) dễ nén hơn context phức tạp (code, toán)
- Budget (VRAM, latency) thay đổi theo thời gian thực

**Vấn đề cụ thể:**
- Làm sao model biết được "độ khó" của context trước khi compress?
- Làm sao để quyết định compression rate tối ưu mà không cần chạy thử nhiều lần?
- Cần cơ chế "dự đoán" information loss trước khi thực sự compress

**Hướng tiếp cận đề xuất:**
- **Complexity estimator**: Một lightweight classifier dự đoán "độ khó" của context → quyết định compression rate
- **Multi-pass progressive compression**: Nén thô trước, nếu model bắt đầu "bối rối" (high entropy output) thì giải nén thêm
- **Reinforcement learning**: Học policy chọn compression rate tối ưu qua reward = (quality - latency_cost)

---

### Gap 4: Streaming/Incremental Latent Compression 🔥

| N | I | F | Fi |
|---|---|---|---|
| 4 | 4 | 2 | 2 |

**Mô tả:**
Hầu hết phương pháp cần thấy **toàn bộ context** trước khi nén. Trong thực tế:
- Video streaming: frame đến liên tục
- Live conversation: tin nhắn đến từng turn
- Sensor data: readings liên tục

**Vấn đề cụ thể:**
- Làm sao cập nhật compressed representation khi có token mới mà không cần encode lại từ đầu?
- Làm sao quyết định khi nào cần "quên" thông tin cũ?
- Cân bằng giữa "nhớ" thông tin cũ và "cập nhật" thông tin mới

**Hướng tiếp cận đề xuất:**
- **Recurrent memory update**: Memory tokens được cập nhật incremental qua recurrent mechanism
- **Memory decay + consolidation**: Cơ chế giống human memory — thông tin quan trọng được củng cố, thông tin cũ phân rã
- **Chunked compression với overlap**: Nén từng chunk, dùng overlapping tokens để duy trì continuity

---

### Gap 5: Agent-Aware Latent Compression 🔥🔥🔥

| N | I | F | Fi |
|---|---|---|---|
| 5 | 5 | 4 | 5 |

**Mô tả:**
AGORA (2026)~\cite{zhang2026agora} đã chứng minh: token-level compression **thất bại hoàn toàn** trên agent tasks vì:
- Action tokens (identifiers, brackets, action verbs) bị xóa
- Plan được viết sớm, dùng cho nhiều bước, nhưng dễ bị evict
- Cấu trúc format (JSON, function call syntax) bị phá vỡ

Nhưng **CHƯA CÓ** phương pháp latent compression nào được thiết kế riêng cho agents.

**Vấn đề cụ thể:**
- Làm sao để compressed representation **bảo toàn action grammar**?
- Làm sao để memory tokens **duy trì plan information** qua nhiều bước?
- Làm sao để phân biệt "structural tokens" (cần giữ format) vs "content tokens" (có thể nén)?

**Hướng tiếp cận đề xuất:**

**5a. Structure-Aware Compression:**
- Tách biệt compression cho "structural stream" (format, actions) và "semantic stream" (nội dung)
- Structural tokens: nén nhẹ hoặc không nén
- Semantic tokens: nén mạnh trong latent space

**5b. Plan-Preserving Memory Tokens:**
- Dành riêng một tập memory tokens cho "plan information"
- Các token này được bảo vệ (không bị ghi đè) trong suốt quá trình agent thực thi
- Dùng probe để phát hiện khi plan information bắt đầu suy giảm

**5c. Action-Verified Compression:**
- Trước khi thực sự dùng compressed representation, kiểm tra xem model có thể tạo đúng action không
- Nếu fail → giải nén thêm
- Nếu pass → dùng compressed

**Lý do đây là gap quan trọng nhất cho `vncompress`:**
- Agent là xu hướng chính của LLM applications hiện nay
- Chưa có paper nào về latent compression cho agents (!!)
- Có thể kết hợp với Context Codec (commitment-level framework)
- Tính khả thi cao: có thể xây dựng benchmark đơn giản (tool-calling tasks)

---

### Gap 6: Formal Guarantees for Latent Compression

| N | I | F | Fi |
|---|---|---|---|
| 5 | 3 | 1 | 2 |

**Mô tả:**
Không có **formal guarantee** nào về những gì được bảo toàn sau compression:
- "Bao nhiêu % thông tin bị mất?"
- "Loại thông tin nào dễ bị mất nhất?"
- "Có đảm bảo gì về worst-case performance không?"

Context Codec (2026)~\cite{trukhina2026context} bắt đầu hướng này nhưng ở token level, chưa vào latent space.

**Hướng tiếp cận đề xuất:**
- Áp dụng **information bottleneck theory** cho latent compression
- Xây dựng **certified bounds** cho information loss dựa trên mutual information
- Phát triển **verifiable compression** — có thể chứng minh một số thuộc tính được bảo toàn

**Tính khả thi thấp** vì cần nền tảng toán học sâu, không phù hợp cho dự án nhỏ.

---

### Gap 7: Cross-modal Unified Latent Compression

| N | I | F | Fi |
|---|---|---|---|
| 4 | 5 | 2 | 3 |

**Mô tả:**
Mỗi modality có cách compression riêng:
- Text: memory tokens / gisting
- Image: visual token pruning / Q-Former
- Audio: spectral compression
- Video: temporal + spatial compression

**Không có** một không gian latent chung cho tất cả modalities.

**Hướng tiếp cận đề xuất:**
- **Universal modality token**: Một loại token duy nhất có thể chứa thông tin từ bất kỳ modality nào
- **Cross-modal contrastive compression**: Dùng contrastive learning để align compressed representations từ các modality khác nhau

**Tính khả thi thấp** vì cần dataset multi-modal lớn và compute cao.

---

### Gap 8: Compression-Retrieval Hybrid 🔥

| N | I | F | Fi |
|---|---|---|---|
| 3 | 4 | 4 | 4 |

**Mô tả:**
- **Compression thuần túy**: nén mọi thứ, mất chi tiết
- **Retrieval thuần túy**: cần index lớn, tốn storage

Chưa có phương pháp kết hợp thông minh cả hai:
- Nén để có "bức tranh tổng quan" (global context)
- Truy xuất để có "chi tiết cụ thể" (local details)
- Chọn linh hoạt giữa compression và retrieval tùy query

**Hướng tiếp cận đề xuất:**
- **Two-tier memory**: Compressed global memory + Retrievable detailed memory
- Model tự quyết định: "với query này, compressed representation đã đủ chưa? Nếu chưa, cần retrieve thêm gì?"

---

### Gap 9: Low-Resource Language Context Compression 🔥🔥🔥🔥

| N | I | F | Fi |
|---|---|---|---|
| 5 | 5 | 5 | 5 |

**Mô tả:**
**100% các paper** về context compression (cả token-level lẫn latent-space) đều đánh giá trên:
- Tiếng Anh (LongBench, RULER, Needle-in-Haystack với English data)
- Tiếng Trung (một số paper từ Trung Quốc)

**KHÔNG CÓ** nghiên cứu nào về context compression cho:
- Tiếng Việt
- Các ngôn ngữ Đông Nam Á
- Ngôn ngữ low-resource nói chung
- Ngôn ngữ có hình thái học phức tạp (morphologically rich)

**Tại sao đây là gap quan trọng:**

1. **Tokenization khác biệt**: Tiếng Việt dùng nhiều token hơn tiếng Anh cho cùng một nội dung (do dấu thanh, từ ghép). Một câu 100 ký tự tiếng Việt có thể thành 60-80 tokens, trong khi tiếng Anh chỉ 30-40 tokens → context window bị "phình to" nhanh hơn.

2. **Cấu trúc ngôn ngữ khác biệt**:
   - Tiếng Việt là ngôn ngữ đơn lập + thanh điệu (tonal, isolating)
   - Dấu thanh thay đổi nghĩa hoàn toàn (ma, má, mà, mả, mã, mạ)
   - Compression methods hiện tại không được thiết kế để bảo toàn tonal information

3. **Thiếu benchmark**:
   - Không có LongBench tiếng Việt
   - Không có Needle-in-Haystack tiếng Việt
   - Không có dataset đánh giá compression quality cho tiếng Việt

4. **Cơ hội đóng góp cao**:
   - Paper đầu tiên về context compression cho tiếng Việt có thể được accept tại các venue tốt
   - Có thể xây dựng benchmark và dataset → trở thành reference cho cả cộng đồng
   - Ứng dụng thực tế rất lớn (chatbot tiếng Việt, trợ lý ảo, RAG cho văn bản pháp luật, y tế...)

**Hướng tiếp cận đề xuất:**

**9a. Vietnamese Context Compression Benchmark (VCC-Bench):**
- Xây dựng dataset tiếng Việt cho các task: QA, summarization, multi-turn conversation, agent tasks
- Đánh giá các phương pháp hiện có (LLMLingua, SnapKV, ICAE...) trên tiếng Việt
- Phân tích: phương pháp nào fail, tại sao, đặc điểm gì của tiếng Việt gây khó?

**9b. Tone-Aware Compression:**
- Thiết kế compression method bảo toàn thông tin thanh điệu
- Dùng phonological features (6 thanh: ngang, huyền, sắc, hỏi, ngã, nặng) làm tín hiệu bổ sung
- So sánh với methods không tone-aware

**9c. Vietnamese Tokenization Impact Study:**
- Phân tích ảnh hưởng của tokenizer đến compression quality
- So sánh: subword tokenizer (BPE, SentencePiece) vs word-level tokenizer vs syllable-level
- Đề xuất tokenization strategy tối ưu cho compression

**9d. Morphology-Aware Compression:**
- Tận dụng đặc điểm hình thái tiếng Việt (từ láy, từ ghép, hư từ, thực từ)
- Nén mạnh hư từ (function words: đã, sẽ, đang, của, những...)
- Giữ nguyên thực từ (content words mang nghĩa chính)

**9e. Cross-lingual Transfer for Compression:**
- Nghiên cứu: compression model huấn luyện trên tiếng Anh có transfer được sang tiếng Việt không?
- Nếu không: fine-tune bao nhiêu data là đủ?
- Nếu có: điều kiện gì để transfer tốt?

**Lộ trình nghiên cứu đề xuất (6-12 tháng):**

```
Tháng 1-2:  Xây dựng VCC-Bench (dataset + evaluation framework)
Tháng 3-4:  Benchmark các phương pháp hiện có trên tiếng Việt
Tháng 5-6:  Phân tích failure modes, viết paper phân tích (survey/analysis paper)
Tháng 7-9:  Đề xuất phương pháp cải tiến (tone-aware / morphology-aware)
Tháng 10-12: Thực nghiệm, so sánh, viết paper phương pháp mới
```

---

### Gap 10: Energy-Aware Compression

| N | I | F | Fi |
|---|---|---|---|
| 3 | 3 | 2 | 2 |

**Mô tả:**
"The Compression Paradox" (2026)~\cite{johnson2026compression1} chỉ ra: input-token reduction không luôn đồng nghĩa với energy saving. Với DeepSeek, compression gây output expansion (+2140% energy). Chưa có work nào về energy-optimal compression decisions.

---

### Gap 11: Safety-Preserving Latent Compression

| N | I | F | Fi |
|---|---|---|---|
| 4 | 4 | 3 | 3 |

**Mô tả:**
AnchorKV~\cite{ni2026anchorkv} đã làm safety-aware KV compression ở token level. Nhưng chưa có work nào về:
- Liệu compressed latent representations có **khuếch đại bias** không?
- Có thể **nhúng malicious payload** vào memory tokens không?
- Làm sao để **audit safety** của compressed representations?

---

### Gap 12: Theoretical Understanding of Latent Compression

| N | I | F | Fi |
|---|---|---|---|
| 5 | 3 | 1 | 1 |

**Mô tả:**
Rất ít lý thuyết về **tại sao latent compression hoạt động**:
- Information Bottleneck theory áp dụng như thế nào?
- Tại sao chỉ vài memory tokens có thể chứa thông tin của hàng nghìn token?
- Có giới hạn lý thuyết nào cho compression ratio không?

---

## 3. Bảng tổng hợp và xếp hạng

| # | Gap | N | I | F | Fi | **Tổng** | Đề xuất |
|---|-----|---|---|---|---|----------|---------|
| 1 | Universal Cross-Architecture | 4 | 4 | 2 | 2 | **12** | Dài hạn |
| 2 | Interpretable/Auditable | 5 | 5 | 4 | 4 | **18** | ⭐ Rất nên làm |
| 3 | Adaptive Online Rate | 4 | 4 | 3 | 3 | **14** | Nên cân nhắc |
| 4 | Streaming/Incremental | 4 | 4 | 2 | 2 | **12** | Dài hạn |
| 5 | Agent-Aware Compression | 5 | 5 | 4 | 5 | **19** | ⭐⭐ Ưu tiên cao |
| 6 | Formal Guarantees | 5 | 3 | 1 | 2 | **11** | Học thuật sâu |
| 7 | Cross-modal Unified | 4 | 5 | 2 | 3 | **14** | Cần nhiều resource |
| 8 | Compression-Retrieval Hybrid | 3 | 4 | 4 | 4 | **15** | ⭐ Nên làm |
| 9 | **Low-Resource Language** | **5** | **5** | **5** | **5** | **20** | ⭐⭐⭐ ƯU TIÊN CAO NHẤT |
| 10 | Energy-Aware | 3 | 3 | 2 | 2 | **10** | Phụ |
| 11 | Safety-Preserving | 4 | 4 | 3 | 3 | **14** | Có thể kết hợp |
| 12 | Theoretical Understanding | 5 | 3 | 1 | 1 | **10** | Lý thuyết |

---

## 4. Đề xuất chiến lược nghiên cứu cho `vncompress`

### Chiến lược chính: Gap 9 (Low-Resource Language) + Gap 5 (Agent-Aware)

Đây là hai gap có tổng điểm cao nhất và bổ trợ cho nhau:

```
Gap 9 (Vietnamese) + Gap 5 (Agent) = Vietnamese Agent Context Compression
```

**Lý do:**
1. **Chưa ai làm**: context compression cho tiếng Việt + agent chưa có paper nào
2. **Tận dụng thế mạnh**: là người Việt, hiểu tiếng Việt → lợi thế cạnh tranh
3. **Tính thực tiễn cao**: ứng dụng ngay cho chatbot, trợ lý ảo tiếng Việt
4. **Có thể publish**: novelty đủ để nộp ACL, EMNLP, NAACL
5. **Khả thi**: không cần compute khổng lồ, có thể dùng model 7B-13B

### Kế hoạch triển khai (3 giai đoạn):

#### Giai đoạn 1: Benchmark & Analysis (3 tháng)

```
Mục tiêu: Xây dựng nền tảng đánh giá + hiểu rõ bài toán

Công việc:
□ Thu thập/Xây dựng Vietnamese long-context dataset
  - Văn bản pháp luật Việt Nam (Luật, Nghị định, Thông tư)
  - Văn bản báo chí (VnExpress, Tuổi Trẻ, Thanh Niên)
  - Hội thoại multi-turn (từ dữ liệu chatbot)
  - Tool-calling / agent traces (tổng hợp từ các agent framework)

□ Xây dựng VCC-Bench (Vietnamese Context Compression Benchmark)
  - Task: QA, Summarization, Multi-turn conversation, Agent tasks
  - Metrics: ROUGE-L, BLEU, BERTScore, task-specific accuracy
  - Context lengths: 4K, 8K, 16K, 32K tokens

□ Benchmark các phương pháp hiện có trên tiếng Việt:
  - Token-level: LLMLingua~\cite{jiang2023llmlingua}, Selective Context~\cite{li2023selective}
  - KV eviction: SnapKV~\cite{li2024snapkv}, H2O~\cite{zhang2023h2o}, StreamingLLM~\cite{xiao2024streamingllm}
  - Latent: Gist Tokens (fine-tune cho LLaMA/Vi), ICAE (nếu có resource)

□ Phân tích failure modes:
  - Phương pháp nào hoạt động tốt/tệ với tiếng Việt?
  - Đặc điểm nào của tiếng Việt gây khó? (thanh điệu, từ ghép, hư từ...)
  - Tokenization ảnh hưởng như thế nào?

Output: 1 paper phân tích (survey/benchmark paper)
```

#### Giai đoạn 2: Phương pháp cải tiến (3-4 tháng)

```
Mục tiêu: Đề xuất phương pháp compression cải tiến cho tiếng Việt

Hướng A — Tone-Aware Compression:
□ Thiết kế compression method bảo toàn tonal information
□ Thêm "tone embedding" bổ sung vào token representation
□ So sánh với baseline không tone-aware

Hướng B — Morphology-Aware Compression:
□ Phân loại từ: hư từ (function words) vs thực từ (content words)
□ Compression policy khác nhau cho từng loại
□ Đánh giá trên VCC-Bench

Hướng C — Agent-Aware Compression cho tiếng Việt:
□ Xây dựng Vietnamese Agent benchmark
□ Thiết kế compression bảo toàn action tokens + plan information
□ Kết hợp với Gap 5 insights

Output: 1-2 papers phương pháp mới
```

#### Giai đoạn 3: Hệ thống & Ứng dụng (3-4 tháng)

```
Mục tiêu: Đóng gói thành công cụ/thư viện, ứng dụng thực tế

□ Phát triển thư viện vncompress:
  - API đơn giản: compress(text, method, ratio) → compressed
  - Hỗ trợ nhiều phương pháp
  - Tích hợp với các LLM framework (LangChain, vLLM...)

□ Xây dựng demo ứng dụng:
  - Chatbot tiếng Việt long-context
  - RAG system cho văn bản pháp luật Việt Nam
  - Agent trợ lý ảo tiếng Việt

□ Open-source release + paper systems track
```

---

## 5. Các hướng kết hợp tiềm năng khác

### 5.1. Gap 2 (Interpretable) + Gap 9 (Vietnamese)

**Ý tưởng:** Khi compression fail trên tiếng Việt, cần biết tại sao → phát triển interpretability tools cho compression. Đây là cách tự nhiên để kết hợp hai gap.

### 5.2. Gap 8 (Hybrid) + Gap 9 (Vietnamese)

**Ý tưởng:** Vietnamese RAG systems hiện tại gặp vấn đề: context quá dài (văn bản pháp luật), retrieval không chính xác. Kết hợp compression (nén tổng quan) + retrieval (chi tiết khi cần).

### 5.3. Gap 2 (Interpretable) + Gap 5 (Agent)

**Ý tưởng:** Agent fail vì compression → cần interpretability để hiểu tại sao fail. Xây dựng "debug tool" cho compressed agents.

---

## 6. Tài nguyên cần thiết

### Compute
- **Tối thiểu:** 1× A100 80GB hoặc 2× RTX 4090 24GB
- **Khuyến nghị:** 2× A100 hoặc 4× RTX 4090
- **Model:** LLaMA-3.1-8B, Qwen-2.5-7B, Gemma-2-9B (có hỗ trợ tiếng Việt)

### Data
- Văn bản tiếng Việt (có sẵn từ báo chí, wikipedia, luật...)
- Cần xây dựng thêm: multi-turn conversations, agent traces

### Thời gian
- **6-12 tháng** cho một chu kỳ nghiên cứu đầy đủ (benchmark → method → paper)
- **3-4 tháng** nếu chỉ tập trung benchmark + analysis

---

## 7. Kết luận

**Gap quan trọng nhất nên theo đuổi: Gap 9 (Low-Resource Language Context Compression)**

Lý do:
1. **Chưa có ai làm** → novelty cao nhất
2. **Phù hợp nhất** với bối cảnh và thế mạnh của người Việt
3. **Khả thi nhất** — không cần compute khổng lồ, có thể bắt đầu ngay
4. **Impact lớn** — mở ra hướng nghiên cứu mới cho cả cộng đồng low-resource languages
5. **Có thể publish** tại các venue tốt (ACL, EMNLP)

**Gap bổ trợ nên kết hợp: Gap 5 (Agent-Aware) và Gap 2 (Interpretable)**

Kết hợp 3 gap này tạo thành một hướng nghiên cứu độc đáo:
**"Interpretable Agent Context Compression for Low-Resource Languages"**

---

*Phân tích bởi: Thanthien | 28/06/2026*

## Tài liệu tham khảo

- \cite{jiang2023llmlingua} Jiang et al. "LLMLingua: Compressing Prompts for Accelerated Inference of LLMs." EMNLP 2023. [arxiv:2310.05736](https://arxiv.org/abs/2310.05736)
- \cite{li2024snapkv} Li et al. "SnapKV: LLM Knows What You are Looking for Before Generation." 2024. [arxiv:2404.14469](https://arxiv.org/abs/2404.14469)
- \cite{zhang2023h2o} Zhang et al. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference." NeurIPS 2023. [arxiv:2306.14048](https://arxiv.org/abs/2306.14048)
- \cite{xiao2024streamingllm} Xiao et al. "Efficient Streaming Language Models with Attention Sinks." ICLR 2024. [arxiv:2309.17453](https://arxiv.org/abs/2309.17453)
- \cite{li2023selective} Li et al. "Selective Context: Compressing Prompts for Efficient In-Context Learning." 2023. [arxiv:2310.06201](https://arxiv.org/abs/2310.06201)
- \cite{zhang2026agora} Zhang & Sun. "AGORA: Adapter-Grounded Observation-Action Retention for LLM Agents." 2026. [arxiv:2605.26596](https://arxiv.org/abs/2605.26596)
- \cite{trukhina2026context} Trukhina & Vashkelis. "Context Codec: Compress the Context, Keep the Commitments." 2026. [arxiv:2605.17304](https://arxiv.org/abs/2605.17304)
- \cite{johnson2026compression1} Johnson. "The Compression Paradox: Provider-Dependent Energy Effects of Prompt Compression." 2026. [arxiv:2603.23528](https://arxiv.org/abs/2603.23528)
- \cite{ni2026anchorkv} Ni & Lao. "AnchorKV: Safety-Aware KV Cache Compression via Soft Penalty." 2026. [arxiv:2606.17872](https://arxiv.org/abs/2606.17872)
- \cite{mu2023gist} Mu et al. "Learning to Compress Prompts with Gist Tokens." NeurIPS 2023. [arxiv:2304.08467](https://arxiv.org/abs/2304.08467)
- \cite{ge2024icae} Ge et al. "In-context Autoencoder for Context Compression." ICLR 2024. [arxiv:2307.06945](https://arxiv.org/abs/2307.06945)
- \cite{lee2026equity} Lee et al. "Equity with Efficiency: Tokenizers for Multilingual LLMs." 2026. [arxiv:2606.15044](https://arxiv.org/abs/2606.15044)
- \cite{deng2026language} Deng et al. "The Language-Energy Divide." 2026. [arxiv:2606.21869](https://arxiv.org/abs/2606.21869)
- \cite{colak2026cross} Colak. "Cross-Lingual Token Arbitrage." 2026. [arxiv:2606.03618](https://arxiv.org/abs/2606.03618)
- \cite{guo2026brain} Guo et al. "Brain-LLM Alignment Tracks Training Data, Not Typology." CoNLL 2026. [arxiv:2605.23032](https://arxiv.org/abs/2605.23032)

Xem thêm danh sách đầy đủ tại: [references.md](references.md)
