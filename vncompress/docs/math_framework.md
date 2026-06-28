# Mathematical Framework — Context Compression cho Tiếng Việt

**Ngày:** 28/06/2026 | **Người tổng hợp:** Thanthien

---

## 1. Self-Attention & KV Cache

### 1.1. Standard Self-Attention

Với input sequence $X \in \mathbb{R}^{n \times d}$, self-attention được tính:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Trong đó:
- $Q = XW_Q$, $K = XW_K$, $V = XW_V$
- $W_Q, W_K \in \mathbb{R}^{d \times d_k}$, $W_V \in \mathbb{R}^{d \times d_v}$

**Độ phức tạp:** $O(n^2 d)$ — tăng bình phương theo độ dài sequence.

### 1.2. KV Cache Memory

Bộ nhớ cần cho KV cache:

$$M_{KV} = 2 \times L \times H \times d_{head} \times n \times \text{bytes\_per\_elem}$$

Ví dụ Llama-3.1-8B với $n=128K$:
- $L=32$, $H=32$, $d_{head}=128$
- FP16 (2 bytes): $M_{KV} = 2 \times 32 \times 32 \times 128 \times 131072 \times 2 = 68.7 \text{ GB}$

### 1.3. Grouped Query Attention (GQA)

Với GQA, số KV heads ($H_{kv}$) ít hơn query heads ($H_q$):

$$H_{kv} = H_q / g \quad \text{với } g \text{ là group size}$$

Ví dụ Llama-3.1-8B: $H_q=32$, $H_{kv}=8$ → $g=4$

---

## 2. Compression Ratio & Efficiency

### 2.1. Compression Ratio

$$CR = \frac{n_{\text{original}}}{n_{\text{compressed}}}$$

### 2.2. Token Savings

$$TS = \frac{n_{\text{original}} - n_{\text{compressed}}}{n_{\text{original}}} \times 100\%$$

### 2.3. Effective Compression Ratio (cho tiếng Việt)

Do token inflation, cần hiệu chỉnh:

$$CR_{\text{eff}} = \frac{CR}{\text{TIR}}$$

với $\text{TIR} = \frac{\text{tokens}_{vi}}{\text{tokens}_{en}}$ (thường 1.5-2.0)

### 2.4. Memory Saved

$$\Delta M = M_{KV}^{\text{original}} - M_{KV}^{\text{compressed}} = 2 L H d_{head} (n_o - n_c) \times \text{bytes}$$

---

## 3. Token-Level Compression (LLMLingua-style)

### 3.1. Perplexity-based Importance

$$\text{importance}(t_i) = \left|\log P(t_i | x_{<i}) - \log P(t_i | x_{<i}, x_{>i})\right|$$

Đo lường "token $t_i$ khó dự đoán đến mức nào nếu chỉ biết context trái".
Token có importance thấp → redundant → có thể xóa.

### 3.2. Budget Controller

Với target ratio $R$:

$$n_{\text{target}} = \max\left(\frac{n}{R}, n_{\min}\right)$$

Chọn $\lfloor n_{\text{target}} \rfloor$ token có importance cao nhất.

### 3.3. Coarse-to-Fine Pipeline

1. **Sentence-level:** Tính $\text{sent\_imp}(S) = \frac{1}{|S|}\sum_{t \in S} \text{importance}(t)$
   → Giữ $K$ sentences có score cao nhất.

2. **Token-level:** Trong các sentence được giữ, chọn token theo importance.

---

## 4. Attention-Based KV Eviction (SnapKV-style)

### 4.1. Observation Window

Dùng $W$ token cuối làm "observation window":

$$A_{obs}[h, i, j] = \text{attention từ token } i \in [n-W, n) \text{ đến token } j$$

### 4.2. Aggregated Importance

Cho mỗi head $h$:

$$\text{imp}_h(j) = \sum_{i=n-W}^{n-1} A_{obs}[h, i, j]$$

### 4.3. Per-Head Selection

Với budget $B_h$ cho head $h$:
$$\text{keep}_h = \text{TopK}(\text{imp}_h, B_h)$$

### 4.4. Memory Reduction

$$\Delta M \propto n - |\bigcup_h \text{keep}_h|$$

### 4.5. H2O Variant (Heavy Hitter Oracle)

$$\text{imp}(j) = \sum_{t=1}^{n} A[t, j] \quad \text{(cumulative attention)}$$

Giữ top heavy hitters + local window.

### 4.6. StreamingLLM Variant

Phát hiện "attention sinks" — 4 token đầu luôn nhận attention cao:

$$\text{keep} = \{\text{tokens } 0..3\} \cup \{\text{tokens } n-W..n-1\}$$

---

## 5. Tone-Aware Compression (Đóng góp chính #1)

### 5.1. Tone Density

$$\rho(t) = \frac{|\{c \in t : \text{tone}(c) \neq \text{ngang}\}|}{|t|}$$

với $t$ là token string, $c$ là ký tự trong token.

### 5.2. Tone Variety

$$\nu(t) = \left|\{\text{tone}(c) : c \in t \land \text{tone}(c) \neq \text{ngang}\}\right|$$

### 5.3. Tone Preservation Weight

$$w_{\text{tone}}(t) = 1.0 + \alpha \cdot \rho(t) \cdot \left(1 + \beta \cdot \frac{\nu(t)}{6}\right)$$

Với:
- $\alpha = 0.5$ (base importance)
- $\beta = 0.3$ (variety bonus)
- $6$ = số lượng thanh điệu tối đa

**Range:** $w_{\text{tone}} \in [1.0, 1.0 + \alpha \cdot (1 + \beta)] \approx [1.0, 1.65]$

### 5.4. Contrast Factor

$$f_{\text{contrast}}(t) = 1 + \gamma \cdot \frac{1}{|N|} \sum_{n \in N} D_{\text{tone}}(\text{tone}(t), \text{tone}(n))$$

Với:
- $N$ = neighbor tokens (window $\pm 2$)
- $D_{\text{tone}}$ = ma trận tương phản thanh điệu

Ma trận $D_{\text{tone}}$ (normalized):

| | Ngang | Huyền | Sắc | Hỏi | Ngã | Nặng |
|---|-------|-------|-----|-----|-----|------|
| **Ngang** | 0.0 | 0.5 | 0.7 | 0.8 | 0.9 | 0.6 |
| **Huyền** | 0.5 | 0.0 | 0.9 | 0.6 | 0.8 | 0.4 |
| **Sắc** | 0.7 | 0.9 | 0.0 | 0.7 | 0.4 | 0.8 |
| **Hỏi** | 0.8 | 0.6 | 0.7 | 0.0 | 0.7 | 0.8 |
| **Ngã** | 0.9 | 0.8 | 0.4 | 0.7 | 0.0 | 0.9 |
| **Nặng** | 0.6 | 0.4 | 0.8 | 0.8 | 0.9 | 0.0 |

(Based on phonetic features: register ± breathy/creaky, contour ± rising/falling)

### 5.5. Tone-Aware Score

$$S_{\text{tone}}(t) = S_{\text{base}}(t) \times w_{\text{tone}}(t) \times f_{\text{contrast}}(t)$$

### 5.6. Tone Preservation Rate (metric)

$$\text{TPR} = \frac{|\{t \in \text{compressed} : \text{tone preserved}\}|}{|\{t \in \text{original} : \text{tone}(t) \neq \text{ngang}\}|}$$

### 5.7. Phonological Consistency Loss

$$\mathcal{L}_{\text{tone}} = \frac{1}{N} \sum_{i=1}^{N} \text{CE}(\hat{y}_i, y_i)$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \lambda \cdot \mathcal{L}_{\text{tone}}$$

---

## 6. Morphology-Aware Compression (Đóng góp chính #2)

### 6.1. Word Classification

Mỗi token $t$ được gán class $c(t) \in \{\text{FUNC}, \text{CONTENT}, \text{REDUP}, \text{COMPOUND}, \text{OTHER}\}$

### 6.2. Class-Aware Preservation Factor

$$f_{\text{class}}(t) = \begin{cases}
0.3-0.5 & \text{nếu } c(t) = \text{FUNC} \quad \text{(hư từ → nén mạnh)} \\
1.0-1.5 & \text{nếu } c(t) = \text{CONTENT} \quad \text{(thực từ → bảo vệ)} \\
0.4-0.6 & \text{nếu } c(t) = \text{REDUP} \quad \text{(từ láy → merge)} \\
1.2-2.0 & \text{nếu } c(t) = \text{COMPOUND} \quad \text{(từ ghép → giữ)} \\
1.0 & \text{nếu } c(t) = \text{OTHER}
\end{cases}$$

### 6.3. Morphology-Aware Score

$$S_{\text{morph}}(t) = S_{\text{base}}(t) \times f_{\text{class}}(c(t))$$

### 6.4. Class-Aware Budget Allocation

Budget cho mỗi class được phân bổ theo:

$$B_c = B_{\text{total}} \times \frac{|T_c| \times r_c}{\sum_{c'} |T_{c'}| \times r_{c'}}$$

Với:
- $|T_c|$ = số token thuộc class $c$
- $r_c$ = keep ratio cho class $c$

### 6.5. Reduplicative Merge

Với cặp từ láy $(t_L, t_R)$:
- Giữ $t_L$ với score gấp 1.5 lần
- Giảm score $t_R$ về 0.1 → bị loại bỏ

---

## 7. Combined Compression (Đóng góp chính #3)

### 7.1. Combined Score

$$S_{\text{combined}}(t) = S_{\text{base}}(t) \times W_{\text{combined}}(t)$$

Với:

$$W_{\text{combined}}(t) = w_t \times [w_{\text{tone}}(t) \times f_{\text{contrast}}(t)] + (1 - w_t) \times f_{\text{class}}(c(t))$$

$w_t \in [0, 1]$ là trọng số cân bằng giữa tone và morphology.

### 7.2. Optimization

Có thể tối ưu $w_t$, $\alpha$, $\beta$, $\gamma$, $f_{\text{class}}$ qua grid search hoặc Bayesian optimization trên validation set.

---

## 8. Evaluation Metrics

### 8.1. Quality Metrics

**ROUGE-L F1:**
$$\text{F1} = 2 \times \frac{P \times R}{P + R}$$

với $P$ = LCS precision, $R$ = LCS recall.

**BLEU:**
$$\text{BLEU} = \text{BP} \times \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)$$

**BERTScore:**
$$\text{BERTScore} = \frac{1}{|x|} \sum_{x_i \in x} \max_{y_j \in y} \cos(\text{emb}(x_i), \text{emb}(y_j))$$

### 8.2. Efficiency Metrics

**Latency:**
$$T_{\text{total}} = T_{\text{compress}} + T_{\text{prefill}} + T_{\text{decode}}$$

**Speedup:**
$$\text{Speedup} = \frac{T_{\text{original}}}{T_{\text{compressed}}}$$

### 8.3. Combined Score

$$\text{Harmonized} = \frac{2 \times Q \times E}{Q + E}$$

với:
- $Q$ = quality score (weighted combination of ROUGE-L, BLEU, EM, BERTScore)
- $E$ = efficiency score (token savings %)

---

## 9. Training Objectives

### 9.1. LLM Loss (Standard)

$$\mathcal{L}_{\text{LM}} = -\frac{1}{N} \sum_{i=1}^{N} \log P(t_i | t_{<i})$$

### 9.2. Phonological Consistency Loss (Our Addition)

$$\mathcal{L}_{\text{tone}} = \frac{1}{N} \sum_{i=1}^{N} \sum_{c=0}^{6} -y_{i,c} \log(\hat{p}_{i,c})$$

với:
- $y_{i,c} = 1$ nếu token $i$ có tone class $c$
- $\hat{p}_{i,c}$ = predicted probability từ hidden state

### 9.3. Morphology Preservation Loss (Our Addition)

$$\mathcal{L}_{\text{morph}} = \frac{1}{N} \sum_{i=1}^{N} w_{c(t_i)} \cdot \|\text{emb}_{\text{orig}}(t_i) - \text{emb}_{\text{comp}}(t_i)\|_2^2$$

với $w_{c(t_i)}$ là class weight (cao hơn cho content words).

### 9.4. Total Training Loss

$$\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda_t \mathcal{L}_{\text{tone}} + \lambda_m \mathcal{L}_{\text{morph}}$$

---

## 10. Token Inflation Analysis

### 10.1. Token Inflation Ratio

$$\text{TIR}_{L_1, L_2} = \frac{\text{tokens}(T_{L_1})}{\text{tokens}(T_{L_2})}$$

với $T_{L}$ là cùng nội dung ở ngôn ngữ $L$.

### 10.2. Expected Values

Từ thực nghiệm:
- $\mathbb{E}[\text{TIR}_{\text{vi, en}}] \approx 1.65 \pm 0.2$
- $\mathbb{E}[\text{TIR}_{\text{zh, en}}] \approx 0.85 \pm 0.15$ (Chinese ít token hơn English do ký tự)
- $\mathbb{E}[\text{TIR}_{\text{th, en}}] \approx 2.1 \pm 0.3$ (Thai token inflation cao nhất)

### 10.3. Impact on Effective Context Capacity

$$\text{EffectiveWords} = \frac{\text{context\_window}}{\text{tokens\_per\_word}}$$

Tiếng Anh: $\frac{128K}{1.3} \approx 98K$ từ
Tiếng Việt: $\frac{128K}{2.1} \approx 61K$ từ

→ **Tiếng Việt chỉ có ~62% effective context capacity so với English.**

---

## 11. Tone Embedding Augmentation

### 11.1. Embedding Concatenation

$$\mathbf{e}'_t = [\mathbf{e}_t \;\|\; \mathbf{W}_{\text{tone}}[\text{tone\_id}(t)]]$$

với:
- $\mathbf{e}_t \in \mathbb{R}^{d}$ = token embedding gốc
- $\mathbf{W}_{\text{tone}} \in \mathbb{R}^{7 \times d_{\text{tone}}}$ = learnable tone embedding matrix
- $\mathbf{e}'_t \in \mathbb{R}^{d + d_{\text{tone}}}$ = augmented embedding

### 11.2. Optional Projection

$$\mathbf{e}''_t = \mathbf{P} \cdot \mathbf{e}'_t \in \mathbb{R}^{d}$$

với $\mathbf{P} \in \mathbb{R}^{d \times (d + d_{\text{tone}})}$

---

## Tài liệu tham khảo

1. Jiang et al. "LLMLingua: Compressing Prompts for Accelerated Inference of LLMs." EMNLP 2023.
2. Li et al. "SnapKV: LLM Knows What You are Looking for Before Generation." 2024.
3. Zhang et al. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference." NeurIPS 2023.
4. Xiao et al. "Efficient Streaming Language Models with Attention Sinks." ICLR 2024.
5. Mu et al. "Learning to Compress Prompts with Gist Tokens." NeurIPS 2023.
6. Ge et al. "In-context Autoencoder for Context Compression." ICLR 2024.
7. Lee et al. "Equity with Efficiency: Tokenizers for Multilingual LLMs." 2026.
8. Deng et al. "The Language-Energy Divide." 2026.
9. Colak. "Cross-Lingual Token Arbitrage." 2026.
