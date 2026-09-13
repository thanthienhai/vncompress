# LACC — Báo cáo kỹ thuật sau wave 1

**Ngày:** 2026-09-05 · **Phạm vi:** toàn bộ wave 1 (6×H100, 2026-09-04) · **Trạng thái:** kết quả đã đo xong, paper chưa đồng bộ

Tài liệu này gộp ba thứ trước đây nằm rời: mô tả bài toán (cho người mới đọc),
kết quả đo được (từ `results_cluster/`), và bản đối chiếu giữa những gì paper
tuyên bố lúc đầu với những gì thực sự đo được.

Báo cáo tường thuật kèm bảng của riêng wave 1:
[`2026-09-04_lacc-wave1-results.html`](2026-09-04_lacc-wave1-results.html).
Bản giao việc ngắn: [`../../HANDOFF.md`](../../HANDOFF.md).

---

## 1. Kết luận trong một đoạn

Giả thuyết trung tâm của paper không đứng được. Nén tone-aware giữ được 93.2%
token có dấu thanh nhưng chỉ lấy lại 2.3% needle ở 2×, trong khi một scorer
perplexity hoàn toàn không biết gì về thanh điệu lấy lại 74.2%. Trên QA trích
xuất, tone-aware còn **thấp hơn cả random dropout**. Ba đường độc lập cùng chỉ
một hướng, và sắc nhất là A/B có kiểm soát: giữ nguyên SLM, chỉ đổi nguồn tín
hiệu thanh điệu, probe học được giữ tone **kém hơn** (ΔTPR −0.246, CI
[−0.251, −0.241]) và trả lời **tốt hơn** (Δtoken-F1 +0.086, CI [+0.028, +0.146],
win-rate 76%). Phần đứng được là nửa hình thái học: phân bổ budget theo tỉ lệ
lớp từ thắng random 65–182% trên QA tiếng Việt ở 0 GB VRAM — nhưng LLMLingua và
SnapKV, cả hai dùng một scoring model nhỏ, thắng mọi biến thể LACC với khoảng
cách lớn.

Riêng biệt và không mâu thuẫn: **phương pháp huấn luyện thì hoạt động.** Probe
control 2×2 cho thấy LoRA + phonological consistency loss thêm +0.0625 macro-F1
trên task thanh điệu thật và −0.0147 trên control task token-identity, tức gain
là đặc hiệu cho tone. Cấu trúc thanh điệu là thật và học được; nó chỉ không phải
thứ làm cho việc nén tốt lên.

| Tuyên bố | Trạng thái |
|---|---|
| P1 — mất thông tin thanh điệu là vấn đề của nén | ❌ Bác bỏ |
| P2 — lãng phí budget cho hư từ là vấn đề | ✅ Xác nhận |
| P3 — token inflation | ⬜ Chưa đo |
| C1 — framework LACC blend 3 tín hiệu | ⚠️ Blend thua từng tín hiệu riêng |
| C2 — Phonological Consistency Loss | ⚠️ Học thật, nhưng thiếu control λ=0 |
| C3 — VCC-Bench | ✅ Đóng góp mạnh nhất còn lại |
| C4 — báo cáo train/eval | ✅ Hoàn thành, có CI |

---

## 2. Bài toán

Nén **context**, không nén model.

Có một tài liệu dài và một câu hỏi. Đưa cả tài liệu vào LLM thì đắt — chi phí
self-attention là O(n²). Ý tưởng: bỏ bớt token trong tài liệu **trước khi** đưa
vào LLM, sao cho LLM vẫn trả lời đúng.

Ví dụ thật, mẫu `viquad_qa_0002`:

```
Context : 28,854 ký tự về lịch sử bang Texas
Câu hỏi : "Quốc gia đầu tiên của châu Âu đã tuyên bố chủ quyền đối với Texas là gì?"
Đáp án  : "Tây Ban Nha"
```

Nén 2× = bỏ một nửa số token. Bài toán là **bỏ token nào**. Bỏ đúng câu chứa
"Tây Ban Nha" là mất đường trả lời.

Giả thuyết riêng của paper: với tiếng Việt, nên chọn token dựa vào đặc trưng
ngôn ngữ tiếng Việt — dấu thanh và loại từ — thay vì dùng phương pháp chung cho
mọi ngôn ngữ. Cơ sở: `bàn / bán / bạn / bản` là bốn từ khác nghĩa chỉ nhờ dấu.

---

## 3. Dataset — có hai dataset khác nhau

Đây là chỗ dễ lẫn nhất trong repo.

### 3.1. Corpus để TRAIN

`vcc_bench_data/training_corpus_v1.json` — 149,693 đoạn văn Wikipedia tiếng Việt
(UVW-2026, CC BY-SA 4.0). **Văn bản thô, không nhãn, không câu hỏi.** Split
90/10 deterministic (seed 42) → 134,723 train / **14,970 val**. Trung bình ~181
token/đoạn, tổng ~27M token.

Corpus thơ tiếng Việt (MIT, gated) không lấy được vì thiếu `HF_TOKEN`, nên
corpus là Wikipedia-only.

### 3.2. Benchmark để ĐO

`vcc_bench_data/vcc_bench_v2.json` — 614 mẫu, mỗi mẫu là
`(context, query, reference_answer)`. **Không dùng để train.**

| Task | n | Context (median) | Reference (median) | Nguồn reference |
|---|---|---|---|---|
| `long_document_qa` | 300 | 21,660 ký tự | 28 ký tự | người viết (UIT-ViQuAD2.0) |
| `needle_in_haystack` | 120 | 45,696 ký tự | 14 ký tự | dựng chính xác |
| `summarization` | 120 | 12,371 ký tự | 441 ký tự | model viết (Qwen2.5-14B) |
| `cross_lingual` | 66 | 2,778 ký tự | 2,767 ký tự | template tay + model |
| `agent_tool_calling` | 8 | 567 ký tự | 137 ký tự | template tay |

Phân bố `reference_type`: 300 `human_written_span`, 180 `model_written`, 120
`constructed_exact`, 14 `hand_written_template`.

Mẫu needle điển hình: nhét `"Mã kích hoạt của hệ thống là BẢO MẬT bảy ba hai."`
vào giữa 16,919 ký tự về Hà Nội rồi hỏi lại mã đó. **60/120 mẫu kèm decoy mất
dấu** (`"BAO MAT bảy ba hai"`) để kiểm tra riêng: nếu compressor làm mất dấu thì
model có nhầm sang decoy không.

### 3.3. v2 sửa gì so với v1

v1 có 243 mẫu và **không đo được gì**: `reference_answer` chính là `context` ở
208/243 mẫu (160/160 mẫu QA), tức benchmark đang chấm điểm "tái tạo input" chứ
không phải "làm được task". Median context v1 chỉ ~350 token — không task nào
thực sự dài. v2 nâng context lên 2K–32K token, thay QA bằng span người viết,
mở needle từ 9 mẫu ad-hoc thành grid 4 độ dài × 5 độ sâu × 6 biến thể, và cưỡng
chế **invariant độ dài reference trong CI** để defect này không quay lại.

---

## 4. Kiến trúc phép đo — ba model, ba vai

```
context 28,854 ký tự
        │
        ▼
┌─────────────────────────────────────────────────┐
│ ① SCORER = Qwen3-4B + LoRA  (model TA TRAIN)    │  cho điểm từng token:
│    - perplexity : token này khó đoán không?     │  "quan trọng cỡ nào?"
│    - tone probe : hidden state mã hóa dấu gì?   │  → KHÔNG sinh chữ nào
│    - morphology : từ này thuộc lớp nào?         │
└─────────────────────────────────────────────────┘
        │  giữ B token điểm cao nhất, bỏ phần còn lại
        ▼
context đã nén (≈14,000 ký tự)
        │  + gói vào chat template cùng câu hỏi
        ▼
┌─────────────────────────────────────────────────┐
│ ② GENERATOR = Qwen2.5-7B-Instruct               │  đọc context đã nén,
│    off-the-shelf, KHÔNG train                   │  sinh câu trả lời
└─────────────────────────────────────────────────┘
        │
        ▼
   "Tây Ban Nha"  ──so với── reference_answer  →  token-F1
```

Vai thứ ba nằm ngoài luồng: **Qwen2.5-14B-Instruct** đã viết sẵn
`reference_answer` cho 180/614 mẫu (summarization + phần cross-lingual).

**Generator là con LLM thật sự trả lời câu hỏi.** Nó cố định. Chỉ đổi cách nén
rồi xem generator trả lời tốt hay kém hơn — đó là toàn bộ phép đo. Generator
được chạy hai lần với hai model (Qwen2.5-7B-Instruct và Qwen3-8B, tắt reasoning)
để chắc kết luận không phụ thuộc một con generator.

### Đánh giá lựa chọn model

| Vai | Model | Nhận xét |
|---|---|---|
| Scorer | Qwen3-4B + LoRA | ⚠️ Quá to cho vai này, và vẫn thua scorer 0.5B |
| Generator | Qwen2.5-7B-Instruct | ✅ Hợp lý — tiếng Việt tốt, chạy được 32K context |
| Generator #2 | Qwen3-8B | ✅ Đúng cách làm robustness check |
| Scorer baseline | Qwen2.5-0.5B-Instruct | ✅ Đúng theo paper LLMLingua gốc |
| Viết reference | Qwen2.5-14B-Instruct | ⚠️ Chưa review tay 10%; chỉ so sánh tương đối được |

Chỗ không ổn: cả bài toán tồn tại để **tiết kiệm** chi phí, mà scorer 4B phải nạp
8 GB weights chỉ để chấm điểm token — trong khi baseline dùng scorer 0.5B
off-the-shelf lại đạt QA F1 0.442 so với 0.239 của ta.

---

## 5. Model đã train

### 5.1. Nhóm A — bốn model chính (wave 1)

Tất cả là LoRA + tone probe trên Qwen3:

| Checkpoint | Base | Trainable | % | batch×accum | Peak VRAM |
|---|---|---|---|---|---|
| `trained_qwen3_0.6b` | Qwen3-0.6B | 5,046,272 | 0.84% | 8×4 | 36.8 GB |
| `trained_qwen3_1.7b` | Qwen3-1.7B | 8,716,288 | 0.50% | 8×4 | 50.3 GB |
| `trained_qwen3_4b` | Qwen3-4B | 16,515,072 | 0.41% | 4×8 | 16.8 GB |
| `trained_qwen3_8b` | Qwen3-8B | 21,823,488 | 0.27% | 4×8 | 25.1 GB |

Recipe **giống hệt ở cả bốn rung**: LoRA r=8, α=16, dropout 0.05, 7 target module
(`q,k,v,o,gate,up,down_proj`), bf16, seq 1024, effective batch 32, 1,200
optimizer step, λ_tone=0.1, seed 42. Val split **giống nhau** ở cả bốn rung
(fingerprint `fab97400ef3480e4`, 14,970 text, 2,712,768 token được chấm).

Loss: `L_total = L_LM + 0.1 × L_tone`, trong đó `L_tone` là cross-entropy của
head tuyến tính đoán 7 lớp thanh điệu từ hidden state cuối
(0 ngang, 1 huyền, 2 sắc, 3 hỏi, 4 ngã, 5 nặng, 6 unknown).

### 5.2. Nhóm B — bốn run probe-only (control study), trên Qwen3-4B

`{lora, frozen_base} × {task thật, control task}`. Base đóng băng, chỉ train
probe. 134,723 train / 14,970 val. Control task gán nhãn thanh điệu ngẫu nhiên
cho mỗi **loại** token, giữ nhất quán giữa train và val — nên một probe chỉ ghi
nhớ token identity vẫn giải được.

### 5.3. Nhóm C — legacy, trước wave 1, đã bị thay thế

- `trained_slm/` → `chronopt-research/vietnamese-gpt2-base` (125.6M), r=8, target `c_attn/c_proj/c_fc` — tone acc ~92.9% (Runs 1–3)
- `trained_models_quick/` → `Qwen2.5-0.5B-Instruct`, r=16 α=32 — run smoke

Cả hai đo dưới benchmark **trước khi sửa 7 defect** → không dùng cho paper.

### 5.4. Không train gì

Generator (`Qwen2.5-7B-Instruct`, `Qwen3-8B`), scorer của LLMLingua/SnapKV
(`Qwen2.5-0.5B-Instruct`), và model viết reference (`Qwen2.5-14B-Instruct`) đều
là off-the-shelf.

---

## 6. Cơ chế scorer: chấm từng token, chạy theo cửa sổ

Hai chuyện khác nhau:

- **Đầu ra**: một điểm cho **mỗi token** — vector độ dài `n`. Đúng là per-token.
- **Cách tính**: model **không** chạy n lần. Nó chạy theo **cửa sổ trượt 512
  token**, mỗi cửa sổ một forward pass, từ đó lấy ra 512 điểm cùng lúc.

Cả hai tín hiệu lấy từ **cùng một forward pass**
(`vncompress/compressors/slm_tone_probe.py:229`):

```python
out = self.model(tensor, output_hidden_states=True)
# ① perplexity ← out.logits            : -log P(t_i | context)
# ② tone       ← out.hidden_states[-1]  → probe(h_i) → 7 lớp thanh điệu
```

Mỗi cửa sổ chỉ nhận điểm cho phần đuôi không chồng lấn của nó, nên token nào
cũng có left-context thật. Sau đó điểm token (tokenizer Qwen3-4B) → điểm **ký
tự** → điểm token của generator (tokenizer Qwen2.5-7B) — phải đi qua ký tự vì
scorer và generator tokenize khác nhau.

### 6.1. Số lượt forward

Mặc định `window_size=512, stride=256` (chồng lấn 50%), chạy **tuần tự,
batch=1**:

| Context (token) | Số forward | Vị trí token xử lý | Bội số |
|---|---|---|---|
| 2,000 | 7 | 3,584 | 1.79× |
| 4,000 | 15 | 7,680 | 1.92× |
| **12,000** | **46** | **23,552** | **1.96×** |
| 16,000 | 62 | 31,744 | 1.98× |
| 32,000 | 124 | 63,488 | 1.98× |

Vì overlap 50%, mỗi token bị đưa qua model ~2 lần — chỉ một lần được dùng.

### 6.2. So với LLMLingua

| | Model | Cửa sổ | Lượt @12k | Latency đo được (QA 2×, median) |
|---|---|---|---|---|
| LLMLingua | Qwen2.5-**0.5B** | 2048, overlap 256 | **7** | **59 ms** |
| `slm_scorer_base` | Qwen3-4B (base) | 512, stride 256 | 46 | 922 ms |
| `slm_scorer` | Qwen3-4B + LoRA | 512, stride 256 | 46 | 1,343 ms |
| `slm_tone_probe_rule` | Qwen3-4B + probe | 512, stride 256 | 46 | 3,237 ms |

Ba yếu tố cộng lại: model to hơn 8×, số lượt nhiều hơn 6.6×, cộng một vòng lặp
Python gán từng scalar (`for k in range(512)` × 46 cửa sổ × 2 tín hiệu ≈ 47,000
phép gán/mẫu) → **55× chậm hơn** trên QA, **113×** trên needle (6,675 ms vs
59 ms). Trên needle 32K, `slm_tone_probe_rule` mất tới **18.4 giây một mẫu**.

**Đây là cấu hình tự làm hại mình.** Nâng window lên 2048 và giảm overlap xuống
256 (đúng như LLMLingua) sẽ giảm 46 → 7 lượt, gần như miễn phí, chưa cần đổi
thuật toán. Ở trạng thái hiện tại: nén xong tốn 3.2 giây, còn generator đọc
context đầy đủ chỉ mất ~0.8 giây — tức việc nén đang đắt hơn việc nó tiết kiệm.

Latency của các biến thể không dùng model (cùng cell QA 2×, median):
`tone_aware` 77 ms, `combined` 113 ms, `morphology_aware` 195 ms.

---

## 7. Kết quả — nội tại (train có chạy không)

### 7.1. Thang scorer

| Rung | PPL base | PPL LoRA | ratio | 95% CI của ratio | Δ | Tone acc (marked) | Macro-F1 (marked) |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 26.00 | 16.28 | 0.6259 | [0.6239, 0.6280] | −37.4% | 0.8638 | 0.8626 |
| Qwen3-1.7B | 17.85 | 11.07 | 0.6198 | [0.6177, 0.6218] | −38.0% | 0.9410 | 0.9401 |
| Qwen3-4B | 14.95 | 9.15 | 0.6116 | [0.6098, 0.6136] | −38.8% | 0.9629 | 0.9587 |
| Qwen3-8B | 11.71 | 7.75 | 0.6616 | [0.6597, 0.6633] | −33.8% | 0.9789 | 0.9757 |

Majority baseline 55.0%; chance cho 5 lớp non-`ngang` ~20%.

CI là paired bootstrap trên per-sample `nll_sums` (2,000 resample, seed 42),
tính trong phiên này. Rung 1.7B ra [0.6177, 0.6218], **khớp** con số
[0.6178, 0.6218] paper đang trích → phương pháp tái lập được, và ba rung còn lại
giờ cũng có CI (trước đó không file nào lưu).

Vì cả bốn rung dùng **cùng val split và cùng số token được chấm**, thang này là
một scaling curve so sánh được ngang rung — không chỉ bốn so sánh nội bộ như
caption bảng hiện tại đang nói.

Loss: `lm_loss` 3.20→2.61 (0.6B), 2.94→2.18 (1.7B), 3.29→2.15 (4B),
2.72→2.40 (8B); `tone_loss` 0.22→0.007–0.024 ở mọi rung.

### 7.2. Cấu trúc lỗi của probe

Confusion matrix 7×7, chỉ 4B và 8B có lưu:

| Lớp | Support | Recall 4B | Recall 8B | Nhầm nhiều nhất |
|---|---|---|---|---|
| ngang | 55.0% | 0.996 | 0.997 | — |
| huyền | 11.1% | 0.981 | 0.989 | → ngang 1.1% |
| sắc | 13.8% | 0.961 | 0.979 | → **nặng** 1.8% |
| hỏi | 6.3% | 0.950 | 0.965 | → **ngã** ⇄ |
| **ngã** | **2.7%** | **0.908** | **0.946** | → **hỏi 3.0%**, sắc 2.6% |
| nặng | 11.1% | 0.968 | 0.985 | → sắc 2.1% |
| unknown | 0% | — | — | lớp chết, không bao giờ kích hoạt |

Lỗi không phân bố đều — nó tập trung đúng vào cặp **hỏi ⇄ ngã** (cặp bị nhập
trong tiếng Việt miền Nam/Trung) và cặp **sắc ⇄ nặng** (cùng đặc trưng thanh
cao/tắc thanh hầu). Model học được cấu trúc âm vị học thật, không phải nhớ bảng
tra. `ngã` là lớp khó nhất và cũng thưa nhất; scale 4B→8B mua được nhiều nhất
đúng ở đó (+0.038 recall).

### 7.3. Probe control 2×2 (Qwen3-4B)

Macro-F1 trên token có dấu, 14,970 held-out text:

| Biểu diễn | Task thật | Control task | Selectivity |
|---|---|---|---|
| Frozen base | 0.8601 | 0.7530 | +0.1071 |
| + LoRA (λ_tone=0.1) | **0.9226** | 0.7383 | **+0.1843** |
| *LoRA thêm vào* | **+0.0625** | **−0.0147** | +0.0772 |

Đọc đủ ba số: LoRA thêm cấu trúc **đúng về tone** (tăng task thật, giảm control
task, gần gấp đôi selectivity). Nhưng base đóng băng đã đạt 0.8601 **trước** mọi
huấn luyện tone, và control task đã đạt 0.7530 → phần lớn thứ probe đọc là token
identity. Con số 0.9587 trích một mình là nói quá.

**Caveat cốt lõi:** nhãn thanh điệu là **hàm tất định của token id**. Một
predictor không cần train, chỉ tra token id rồi chạy tone analyzer, đạt **100%**.
Nên probe không đo "model có đoán được thanh điệu không" — nó đo "thông tin
thanh điệu còn sót lại bao nhiêu trong hidden state".

---

## 8. Kết quả — downstream (đem ra dùng thật)

### 8.1. Bảng chất lượng chính (generator Qwen2.5-7B-Instruct)

Mỗi cột là metric headline của task đó: QA = token-F1, Needle = recall,
Cross-lingual = ROUGE-L, Agent = token-F1.

**2× compression**

| Method | Long-doc QA | Needle | Cross-lingual | Agent tools |
|---|---|---|---|---|
| Không nén | 0.634 | 0.993\* | 0.390 | 0.260 |
| **LLMLingua** | **0.442** | **0.742** | **0.326** | 0.262 |
| SnapKV | 0.371 | 0.573 | 0.297 | 0.251 |
| LACC-Morph | 0.277 | 0.296 | 0.249 | **0.280** |
| LACC-Combined | 0.212 | 0.051 | 0.270 | 0.130 |
| LACC-Tone | 0.122 | 0.023 | **0.292** | 0.194 |
| Random dropout | 0.168 | 0.119 | 0.210 | 0.230 |

**4× compression**

| Method | Long-doc QA | Needle | Cross-lingual | Agent tools |
|---|---|---|---|---|
| Không nén | 0.635 | 0.993\* | 0.389 | 0.256 |
| **LLMLingua** | **0.259** | **0.403** | 0.230 | 0.223 |
| SnapKV | 0.222 | 0.353 | **0.237** | 0.104 |
| LACC-Morph | 0.124 | 0.071 | 0.192 | 0.144 |
| LACC-Combined | 0.105 | 0.018 | 0.154 | 0.141 |
| LACC-Tone | 0.027 | 0.000 | 0.186 | 0.149 |
| Random dropout | 0.044 | 0.007 | 0.162 | 0.097 |

**8× compression**

| Method | Long-doc QA | Needle | Cross-lingual | Agent tools |
|---|---|---|---|---|
| Không nén | 0.635 | 0.993\* | 0.389 | 0.256 |
| **LLMLingua** | 0.145 | 0.199 | **0.157** | 0.233 |
| SnapKV | **0.161** | **0.250** | 0.153 | 0.217 |
| LACC-Morph | 0.080 | 0.046 | 0.114 | 0.117 |
| LACC-Combined | 0.063 | 0.010 | 0.112 | 0.182 |
| LACC-Tone | 0.014 | 0.000 | 0.119 | 0.157 |
| Random dropout | 0.026 | 0.000 | 0.109 | 0.136 |

\* Arm `none` fail 28/120 mẫu needle ở cả ba ratio (context 32K + chat wrapper +
256 token sinh ra vượt cửa sổ 32,768 của engine). Mọi method nén fail 0. Nên
0.993 là trung bình trên 92 mẫu vừa cửa sổ — **một subset dễ hơn**. So sánh
method-với-method không bị ảnh hưởng; chỉ dòng baseline không so trực tiếp được
trên task này.

### 8.2. Morphology vs random — nửa đứng được

| Ratio | Task | Morph | Random | Gain |
|---|---|---|---|---|
| 2× | QA | 0.277 | 0.168 | **+65%** |
| 2× | Needle | 0.296 | 0.119 | **+150%** |
| 4× | QA | 0.124 | 0.044 | **+182%** |
| 4× | Needle | 0.071 | 0.007 | +924% |
| 8× | QA | 0.080 | 0.026 | +211% |
| 8× | Agent | 0.117 | 0.136 | −14% |

Trên generator thứ hai (Qwen3-8B) hướng giữ nguyên: QA +29/+37/+56%, needle
+119/+327/+516%, agent âm ở cả ba ratio. **Xếp hạng bất biến theo generator** —
kết luận không phải artifact của một bộ sinh.

### 8.3. Ablation — tách riêng từng tín hiệu (4×)

| Method | Long-doc QA | Needle | Cross-lingual | Agent tools |
|---|---|---|---|---|
| LACC-Combined | 0.118 | 0.214 | 0.145 | **0.244** |
| **w_ppl only** | **0.253** | **0.494** | 0.226 | 0.236 |
| w_tone only | 0.113 | 0.010 | 0.189 | 0.163 |
| w_morph only | 0.184 | 0.216 | **0.226** | 0.177 |

Blend thua tín hiệu perplexity đơn lẻ trên mọi task trừ agent (n=8, không đỡ
được kết luận nào).

### 8.4. TPR và đánh đổi

TPR trung bình trên toàn benchmark:

| Method | 2× | 4× | 8× |
|---|---|---|---|
| LACC-Tone | 0.960 | 0.566 | 0.289 |
| LACC-Combined | 0.777 | 0.503 | 0.256 |

Calibration 30 nhánh trên dev split đặt optimum tại **`tone_weight = 0.0`**,
alpha 0.25, gamma 0.2. Trong sweep sensitivity, quality cao nhất (0.134) đúng ở
điểm `tone_weight=0`; TPR cao nhất (0.493) thì quality thấp nhất (0.050). Metric
mà paper đề xuất bị chính sweep của paper đặt trọng số về 0.

### 8.5. A/B có kiểm soát: probe vs rule

Giữ nguyên SLM (Qwen3-4B), chỉ đổi nguồn tín hiệu thanh điệu. 120 mẫu, paired
bootstrap.

| Ratio | Metric | Δ (probe − rule) | 95% CI | p | win-rate | sig |
|---|---|---|---|---|---|---|
| 2× | TPR | **−0.246** | [−0.251, −0.241] | 0.000 | 0% | ✓ |
| 2× | token-F1 | **+0.086** | [+0.028, +0.146] | 0.003 | 76% | ✓ |
| 2× | ROUGE-L | +0.082 | [+0.024, +0.142] | 0.004 | 75% | ✓ |
| 4× | TPR | −0.136 | [−0.140, −0.133] | 0.000 | 0% | ✓ |
| 4× | token-F1 | +0.016 | [−0.026, +0.060] | 0.459 | 86% | · |

Ở 4× hiệu ứng chất lượng nhỏ và không còn significant, dù win-rate 86% — nhất
quán về dấu nhưng phương sai lớn.

### 8.6. Các arm dùng SLM trên full benchmark (2×)

| Arm | QA | Needle | Cross-ling | Agent |
|---|---|---|---|---|
| `slm_scorer` (4B đã train) | 0.239 | **0.395** | 0.237 | 0.283 |
| `slm_scorer_base` (4B base) | 0.238 | 0.303 | 0.248 | 0.290 |
| `slm_tone_probe_rule` | 0.234 | 0.354 | 0.246 | 0.296 |
| *LLMLingua (scorer 0.5B)* | ***0.442*** | ***0.742*** | ***0.326*** | *0.262* |

Ở 4×: `slm_scorer` 0.062 / 0.038 / 0.149 / 0.188 so với `slm_scorer_base`
0.071 / 0.028 / 0.153 / 0.179.

Đọc ra: train scorer **gần như không đổi gì trên QA** (0.239 vs 0.238), ở 4× còn
kém hơn base. Lợi ích duy nhất có thật là **needle: 0.395 vs 0.303 (+30%)** —
retrieval, không phải QA.

### 8.7. Chi phí (4×, trung bình toàn benchmark)

| Method | Realized CR | Latency | VRAM |
|---|---|---|---|
| Không nén | 1.00× | 2.4 ms | — |
| Random dropout | 4.00× | 3.5 ms | 0 GB |
| LLMLingua | 4.70× | 73.8 ms | ~1 GB (0.5B) |
| SnapKV | 4.00× | 503.8 ms | ~1 GB |
| LACC-Tone | 4.00× | 105.5 ms | **0 GB** |
| LACC-Morph | 4.02× | 234.6 ms | **0 GB** |
| LACC-Combined | 4.00× | 179.2 ms | **0 GB** |

Latency của LACC-Morph bị chi phối bởi word segmenter pure-Python → **không phải
phép đo hiệu năng công bằng**, chưa claim được lợi thế chi phí.

---

## 9. Minh họa: một mẫu gói cả kết luận

Mẫu `viquad_qa_0002`, nén 2×:

| Method | Nén | TPR | Trả lời | token-F1 |
|---|---|---|---|---|
| Không nén | 1.00× | — | "Tây Ban Nha" | **1.000** |
| LLMLingua (scorer 0.5B) | 2.41× | — | "Tây Ban Nha" | **1.000** |
| LACC-Morph | 2.00× | — | "Hoa Kỳ" | 0.000 |
| **LACC-Tone** | 2.00× | **0.932** | **"Không đủ thông tin."** | **0.000** |

Dòng cuối: nó **giữ được 93.2% token có dấu thanh** — gần hoàn hảo đúng cái mục
tiêu paper đặt ra — **và vẫn xóa mất câu chứa đáp án.** Trên 300 mẫu QA có **53
mẫu** cùng dạng: không-nén đúng >0.9, LLMLingua đúng >0.5, tone-aware về 0.

"Giữ được dấu thanh" và "giữ được thông tin cần thiết" là hai chuyện khác nhau,
và wave 1 đã tối ưu chuyện thứ nhất.

---

## 10. Insight

1. **Hai tín hiệu bị gộp dưới một cái tên nhưng hành xử trái ngược.** Morphology
   thắng random 65–182% trên QA ở 0 GB VRAM. Tone-contrast — thành phần được kỳ
   vọng gánh cả paper — giữ dấu gần hoàn hảo và thấp hơn cả random trên mọi task
   định vị thông tin. Tiền đề "language-aware" đúng một nửa, và đúng ở nửa không
   ai đặt cược.

2. **TPR không phải proxy cho chất lượng — nó nghịch biến.** Chính sweep của
   paper đặt trọng số tối ưu của metric mà paper đề xuất về 0.

3. **A/B có kiểm soát tách "học được" khỏi "hữu ích".** Probe học từ LoRA giữ
   tone kém hơn heuristic tra từ điển nhưng trả lời tốt hơn, cả hai delta đều có
   CI loại trừ 0. Phonological consistency loss dạy model một thứ thật, nhưng thứ
   đó không phải "bảo toàn thanh điệu" — và thành phần tối đa hóa TPR chính là
   thành phần trả giá bằng chất lượng.

4. **Phương pháp huấn luyện hoạt động, và đặc hiệu cho tone.** +0.0625 trên task
   thật vs −0.0147 trên control loại trừ giả thuyết "probe chỉ đọc token
   identity". Cộng PPL giảm 34–39% đồng đều ở cả bốn rung → pipeline train là
   đóng góp độc lập, dùng được kể cả khi bỏ hẳn claim về nén.

5. **Scorer tốt hơn ≠ nén tốt hơn.** Scorer Qwen3-4B đã fine-tune cho QA 2× =
   0.239, trong khi LLMLingua với scorer 0.5B = 0.442. Lợi ích nằm ở **thuật
   toán chọn token** (perplexity theo cửa sổ, phân bổ budget lặp), không ở kích
   thước hay chất lượng scorer. Đây là insight đắt nhất của wave: đã đầu tư vào
   nhánh không quyết định kết quả.

6. **Cấu trúc lỗi của probe khôi phục hiện tượng ngôn ngữ học thật.** Lỗi tập
   trung ở cặp hỏi⇄ngã và sắc⇄nặng — đúng các cặp bị nhập trong phương ngữ tiếng
   Việt. Model học âm vị, không nhớ bảng tra.

7. **Có đúng một vùng tone thắng: task mà bề mặt chữ *là* output.** Trên
   cross-lingual 2×, tone_aware (0.292) là biến thể LACC tốt nhất, trên cả morph
   (0.249). Giữ dấu ở đây là phần của task, không phải chi phí.

8. **Xếp hạng bất biến theo generator.** Hai generator khác nhau cho cùng thứ tự.

9. **Nén tiếng Việt long-context chỉ dùng được ở 2×.** Ở 4× method tốt nhất còn
   0.259 QA F1 so với 0.635 không nén (−59%); ở 8× còn 0.145. Không method nào
   thoát khỏi vách này.

10. **Kỷ luật đo lường tạo ra kết quả, không phải method.** Invariant độ dài
    reference và cơ chế đếm failure riêng biệt vừa phát hiện defect v1, vừa bắt
    được caveat `none`-needle. Benchmark cũ *không thể* phát hiện gì vì 86% mẫu
    chấm điểm tái tạo input.

---

## 11. Đối chiếu contribution gốc với kết quả

Paper mở đầu bằng một tiền đề: mọi phương pháp nén hiện có (LLMLingua, SnapKV,
H2O, StreamingLLM, Gist, ICAE) đều **language-blind**. Từ đó suy ra ba vấn đề và
tuyên bố bốn contribution.

### Ba vấn đề

| | Tuyên bố gốc | Kết quả |
|---|---|---|
| **P1** | Mất thông tin thanh điệu — bỏ token có dấu là đổi nghĩa (`má` → `ma`) | ❌ **Tiền đề sai.** Giữ 93.2% dấu vẫn trả lời sai; TPR nghịch biến với quality. Vấn đề không phải "mất dấu" mà là "mất câu chứa đáp án", và dấu thanh không cho biết câu nào chứa đáp án |
| **P2** | Lãng phí budget cho hư từ — 30–40% token là hư từ, bị chấm ngang thực từ | ✅ **Đúng.** Morphology thắng random 65–182% trên QA ở 0 GB VRAM |
| **P3** | Token inflation — tiếng Việt cần 1.5–2.0× token so với tiếng Anh | ⬜ **Chưa đo.** Là citation từ paper khác; không thí nghiệm nào trong wave 1 kiểm chứng hay khai thác |

### Bốn contribution

| | Tuyên bố gốc | Kết quả |
|---|---|---|
| **C1** | LACC — framework blend tone-contrast + morphology + perplexity thành một điểm importance, 3 tier hardware (0 GB → INT4 7B) | ⚠️ **Blend thua từng tín hiệu riêng.** Ablation 4×: ppl-only 0.253 > morph-only 0.184 > combined 0.118. Calibration đặt w_tone = 0 |
| **C2** | *Phonological Consistency Loss* — objective train cùng LoRA để hidden state mã hóa thanh điệu; classifier dùng luôn làm tone probe ("a genuine training-method contribution") | ⚠️ **Nửa đúng, nửa tự phản.** Probe học thật và đặc hiệu, nhưng thiếu ô control λ_tone=0 nên chưa tách được công của λ_tone khỏi công của fine-tune tiếng Việt. Và trong A/B, probe học được thắng **chính vì giữ dấu ít hơn** heuristic |
| **C3** | VCC-Bench — benchmark nén context tiếng Việt đầu tiên, kèm provenance/checksum/reproducibility | ✅ **Đóng góp mạnh nhất còn lại.** v1 243 mẫu → v2 614 mẫu, QA người viết, needle grid có 60 cặp tối thiểu thanh điệu, invariant trong CI |
| **C4** | Báo cáo train/eval từ đầu cho model LoRA + tone loss | ✅ **Hoàn thành và mạnh hơn:** 4 rung 0.6B–8B, cùng split, có CI |

---

## 12. Vấn đề còn lại

Xếp theo mức độ chặn submission.

### Chặn paper

1. **Abstract và Introduction vẫn bảo vệ luận điểm mà Results đã bác bỏ.**
   Abstract **chưa sửa một chữ** so với `HEAD`:

   | Abstract đang viết | Thực tế sau wave 1 |
   |---|---|
   | "~22,200 paragraphs" | 149,693 paragraphs |
   | "tone accuracy 16.6% → **92.9%**" | 0.9629 (4B) — 92.9% là số của model GPT-2 cũ |
   | "perplexity improved **25.2%**" | −38.8% (4B) |
   | "**243-sample** benchmark" | 614 mẫu (v2) |
   | "tone signal **nearly doubling** TPR (0.96 vs 0.51–0.52)" — trình bày như kết quả dương | TPR cao = quality thấp; w_tone tối ưu = 0 |

   Sửa được không cần GPU.

2. **Bảng scorer ladder trong paper thiếu 2 rung.** `paper/tables/slm_ladder.tex`
   (sinh ra) có đủ 4 rung; Table trong `lacc_icisn_en.tex` chỉ có 0.6B và 1.7B,
   **in đậm 1.7B như tốt nhất** — trong khi 4B và 8B đều hơn. Nguyên nhân: 8 bảng
   được copy inline thay vì `\input{paper/tables/…}`, nên
   `make_paper_tables.py` chạy lại không lan vào paper.

3. **Kiểm định thống kê cho các bảng chính.** Chỉ hai so sánh có CI: PPL (4 rung,
   vừa tính trong phiên này) và A/B probe-vs-rule. Bảng benchmark chính, ablation,
   so sánh SLM scorer đều là mean chưa bootstrap.
   `vncompress/evaluation/significance.py` đã có, per-cell dump có per-sample
   values → việc scripting, không cần GPU.

### Chặn kết luận khoa học

4. **Thiếu ô control λ_tone=0.** Probe control so `frozen base` vs
   `base + LoRA(λ=0.1)` → +0.0625. Ô thiếu là `base + LoRA(λ=0)`. Chưa ai tách
   được +0.0625 đó là do λ_tone hay chỉ do fine-tune LM trên tiếng Việt. **Một
   run ~45 phút GPU, và nó quyết định một trong hai kết quả dương của paper.**

5. **λ_tone chưa từng được sweep.** Cả bốn rung dùng đúng λ=0.1. Và `tone_loss`
   rơi xuống 0.020 ở step 300 → nhân λ=0.1, tone term chỉ còn đóng góp **0.09%
   tổng loss** từ step 300 trở đi. Nghĩa là ~75% quá trình train thực chất chỉ
   còn train LM loss trên Wikipedia tiếng Việt. Cái gọi là "tone-aware training"
   chủ yếu là domain adaptation.

6. **Nhiệm vụ phụ vốn quá dễ.** Nhãn thanh điệu là hàm tất định của token id;
   một predictor tra token id đạt 100%. Control task (nhãn ngẫu nhiên theo loại
   token) đã đạt 0.7530 — 3/4 điểm là ghi nhớ token identity.

7. **Lệch train/eval về độ dài.** Train ở seq 1024, eval trên context 2K–32K
   token. Scorer chưa bao giờ thấy context dài lúc train.

8. **Baseline `none` đo trên subset dễ hơn ở needle.** Fail 28/120. Sửa:
   `--max-model-len` 40K cho cell baseline, hoặc cap budget dài nhất ở 24K.

9. **Agent tool-calling chỉ 8 mẫu** — không đỡ được kết luận nào, kể cả cái thú
   vị (morph *trên* uncompressed). Cần ≥60.

### Chặn release

10. **Chưa review tay 10% reference do model viết.** Sheet 18 mẫu đã sinh
    (`vcc_bench_data/reference_review_sheet.md`), chưa ai đọc.
11. **License UIT-ViQuAD2.0 chưa xác nhận** → quyết định VCC-Bench v2 có được
    release cùng paper không. QA split hiện eval-only, không redistribute.
12. **Claim hiệu năng chưa đo được.** Morph 0 GB nhưng 234.6 ms vs LLMLingua
    72.1 ms, gần hết thời gian ở segmenter pure-Python.

### Vận hành

13. **Run `bench_slm` bị cắt giữa nhánh cuối.** Log dừng ở
    `slm_tone_probe / long_document_qa / 2.0x`; chỉ 3/4 nhánh có per-cell dump và
    **không có `vcc_bench_results.json`**. Method headline (probe đã train) không
    có dòng nào trên 614 mẫu — chỉ có A/B 120 mẫu.
14. **Ba trong bốn model đã train chưa từng dùng downstream.** Grep toàn bộ
    `results_cluster/`: 0.6B, 1.7B, 8B chỉ xuất hiện trong log train + 2 file
    eval của chính chúng. Kể cả 8B — model có PPL tốt nhất (7.75) và tone F1 tốt
    nhất (0.9757).
15. **Toàn bộ wave 1 chưa commit** — 17 file modified + 22 untracked (gồm
    `results_cluster/`, `paper/tables/`, `k8s/`, `vcc_bench_v2.json`,
    `HANDOFF.md`). LoRA weights (373 MB) chỉ ở NFS backup và node `dn24z` sẽ bị
    reimage. Trong repo chỉ còn `adapter_config.json` + `tone_probe_meta.json`.
16. **Corpus thơ tiếng Việt bị gate** (thiếu `HF_TOKEN`) → corpus train
    Wikipedia-only.

---

## 13. Việc tiếp theo, theo thứ tự

**Không cần GPU, làm ngay:**

1. Commit wave 1 trước khi node `dn24z` bị reimage (mục 15).
2. Sửa abstract + introduction cho khớp Results (mục 1) — đây là lỗi dễ bị
   reviewer bắt nhất.
3. Đổi 8 bảng inline thành `\input{paper/tables/…}` và regenerate (mục 2).
4. Paired bootstrap trên per-cell dumps cho các bảng chính (mục 3).

**GPU, rẻ, quyết định kết luận:**

5. Run `LoRA(λ_tone=0)` + probe control để tách công của λ_tone (mục 4). ~45 phút.
6. Chạy lại window 2048 / overlap 256 cho các arm SLM (mục §6.2) — giảm 46 → 7
   lượt forward, có thể đổi hoàn toàn bức tranh chi phí.
7. Hoàn thành nhánh `slm_tone_probe` trên full benchmark (mục 13).

**Thí nghiệm mà số liệu đang chỉ vào:**

Hai tín hiệu mạnh nhất nằm ở hai method khác nhau — perplexity (LLMLingua 0.442)
và morphology (Morph 0.277) — và **chưa ai đo chúng cùng nhau**. Perplexity
scorer + phân bổ budget theo tỉ lệ lớp từ là method hiển nhiên tiếp theo, và là
thay đổi nhỏ trong `CombinedCompressor`.

**Nếu giữ hướng tone:** scope về task bề mặt (dịch, chuyển tự, trích dẫn nguyên
văn, sinh danh từ riêng) — nơi duy nhất tone không đứng cuối.

**Không nên:** chạy lại benchmark để lấy số LACC đẹp hơn. Phép đo đã đúng;
method là như vậy.

---

## 14. Tài sản và cách tái lập

| Nội dung | Vị trí |
|---|---|
| Raw results (28 MB, 299 file) | `results_cluster/` — xem `README.md` trong đó |
| Summary máy đọc được | `results_cluster/collected.json` |
| Bảng LaTeX | `paper/tables/` (8 file, regenerate bằng `scripts/make_paper_tables.py`) |
| Báo cáo | `docs/reports/*.html`, `docs/reports/*.md` |
| Backup NFS | `/mnt/hps/anhm-paper/vncompress-wave1-2026-09-04/` — 587 MB, 515 file, md5-verified, gồm LoRA weights |
| Tooling cluster | `k8s/` — xem `README.md` trong đó |

```bash
# đọc số từ raw results
python3 scripts/collect_experiment_results.py --root results_cluster
python3 scripts/make_paper_tables.py --root results_cluster --out paper/tables

# tái lập wave
./k8s/lab.sh up            # 6 runner
./k8s/waves.sh wave2       # SLM eval + sweep chính
./k8s/waves.sh wave3       # ablation, calibration, A/B, probe control
./k8s/backup_to_hps.sh     # copy sang NFS, rồi `verify`
./k8s/lab.sh down
```

Đọc `k8s/README.md` trước — nó ghi năm thứ đã tốn thời gian lần đầu: storage
class mặc định không attach được trong namespace này; CPU (không phải GPU) là
bottleneck scheduling; `pgrep -f` match cả chính process đang đợi; `pgrep` không
thấy được pod khác; và process kết thúc chưa có nghĩa là GPU đã rảnh.

**Tests:** 311 pass, CPU-only, ~4 s (210 test function trên 19 file). Ba suite
canh các fix của wave này: `tests/test_prompting_and_scoring.py` (chat template,
task-aware scoring, strip reasoning block, regression dict-return),
`tests/test_llmlingua_windowing.py` (điểm có windowing phải bằng không
windowing), và hai test invariant độ dài reference trong
`tests/test_dataset_provenance.py`.

---

## Phụ lục — bảy defect đã sửa ở Phase 0

Trước khi sửa, mọi số chất lượng đều vô nghĩa:

1. `reference_answer` chính là `context` ở 208/243 mẫu v1 (160/160 mẫu QA)
2. Prompt sinh văn bản là nối thô token id, không có chat template
3. vLLM 0.28 bỏ `prompt_token_ids=`
4. transformers 5 đổi `apply_chat_template` sang trả về dict — âm thầm cho ra
   một list *string* làm "token ids"
5. LLMLingua tính `log_softmax` trên toàn chuỗi (9.4 GB cho một tensor)
6. SnapKV đòi attention S×S đầy đủ và ghim 512 token bất chấp target ratio
7. `exact_match` là so khớp chuỗi thô, luôn bằng 0, nhưng chiếm 40% trọng số
   xếp hạng
