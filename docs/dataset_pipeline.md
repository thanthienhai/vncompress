# Dataset Pipeline: Teacher Distillation, Training & Evaluation

> **Trạng thái (đọc trước):** Tài liệu này gồm **hai lớp**, đừng lẫn lộn:
>
> 1. **Hợp đồng dữ liệu wave-2 đang dùng thật** — mục §2.5. Đây là thứ các script training đã xây dựng (`E4` relevance probe, `E6` encoder compressor, SLM/tone) **thực sự đọc** hôm nay. Muốn train/eval wave-2 thì làm theo mục này.
> 2. **Roadmap dataset V1 (đề xuất, chưa triển khai)** — mục §4–§5, §11–§13. Đây là teacher-distillation dataset tham vọng hơn (compressed_text, token_labels, preference, hard_negative, …). **Chưa có script nào sinh ra, và không training wave-2 nào tiêu thụ các trường này.** Các khối đó được gắn nhãn `ROADMAP` ở đầu mục.
>
> Điểm mấu chốt: E4 và E6 **tự suy ra nhãn trong code** (span-overlap / perplexity teacher), nên chúng KHÔNG phụ thuộc canonical schema §5. Nhầm hai lớp này sẽ dẫn tới việc dựng một dataset mà script wave-2 không đọc được.

## 1. Mục tiêu

Tài liệu này định nghĩa pipeline xây dựng dataset cho `vncompress` theo hướng **query-conditioned Vietnamese context compression**. Dataset không chỉ chứa văn bản đã rút gọn mà còn chứa tín hiệu supervision để huấn luyện compressor: mức độ quan trọng của span/token, keep/drop label, compressed candidates và preference giữa các candidate.

Mục tiêu cuối cùng:

```text
Long Vietnamese Context + Query
            ↓
       Compressor
            ↓
   Compressed Vietnamese Context
            ↓
      Downstream LLM
            ↓
          Answer
```

Dataset phải phục vụ đồng thời hai mục tiêu:

1. **Training:** học cách giữ lại thông tin cần thiết với budget/ratio xác định.
2. **Evaluation:** đo mức độ nén và khả năng bảo toàn thông tin đối với tác vụ downstream.

> Quy tắc quan trọng: dữ liệu dùng để teacher sinh supervision không được làm rò rỉ sang test/benchmark.

---

## 2. Dataset hiện đang có trong Wave 2

### 2.1. UVW-2026

Là nguồn corpus tổng quát chính cho tiếng Việt. Trong pipeline hiện tại, `scripts/build_training_corpus.py` sử dụng UVW-2026 làm nguồn dữ liệu lớn và lọc chất lượng trước khi đưa vào training corpus.

**Vai trò:**
- General-domain Vietnamese text.
- Tạo context dài/ngắn cho compression.
- Là nguồn chính để mở rộng dataset teacher-distillation.

### 2.2. Vietnamese Poetry

`bigscience-data/roots_vi_vietnamese_poetry` được sử dụng như nguồn augmentation trong corpus hiện tại.

**Vai trò:**
- Bổ sung văn phong và cấu trúc tiếng Việt khác Wikipedia.
- Hỗ trợ kiểm tra khả năng bảo toàn từ ngữ/âm tiết.

Không nên coi poetry là nguồn chính cho query-conditioned factual compression.

### 2.3. VCC-Bench

VCC-Bench là benchmark chính của project, gồm 5 nhóm tác vụ:

- Long-Document QA
- Multi-turn
- Needle-in-a-Haystack
- Agent Tool-Calling
- Cross-lingual

Các metric hiện có gồm ROUGE-L, BLEU, BERTScore, Exact Match, Token-F1, Tone Preservation Rate và Harmonized Score.

Trong Wave 2, benchmark được mở rộng để đánh giá thêm các chiến lược query-conditioned scoring, perplexity và budget.

### 2.4. Wave 2 training data

Wave 2 hiện có hai hướng supervision quan trọng, và điểm chung là **cả hai tự sinh nhãn trong lúc train**, không đọc nhãn dựng sẵn từ file:

- **E4 — Relevance Probe:** học "token này có liên quan tới câu trả lời không" từ hidden states của SLM đóng băng. Nhãn là **weak supervision span-overlap** (`linguistics.build_relevance_labels`): token dương nếu âm tiết đã decode của nó trùng một âm tiết trong `reference_answer`. Không cần teacher, không cần annotation.
- **E6 — Encoder Token Classification Compressor:** encoder (`vinai/phobert-base`) phân loại token keep/drop. Supervision được **distill ngay trong `scripts/train_encoder_compressor.py`**: teacher causal LM (mặc định `Qwen/Qwen2.5-0.5B-Instruct`) chấm perplexity cửa sổ trượt, giữ top-`1/ratio` token làm lớp "keep", rồi chiếu quyết định đó lên encoder qua char-span. Nhãn keep/drop **không** được lưu ra file trung gian.

Chi tiết hợp đồng input xem §2.5.

### 2.5. Hợp đồng dữ liệu wave-2 đang dùng thật

Đây là thứ code training đã build đọc vào **hôm nay**. Mọi thứ ở §4–§13 nằm ngoài mục này là roadmap.

**A. Corpus text thô — cho E6 và SLM/tone/LACC-model** (`vncompress.training.load_training_texts`)

- File mặc định: `data/benchmark/training_corpus_v1.json` (fallback: `wikipedia_vi_raw.json`, rồi corpus demo built-in).
- Shape chấp nhận (một trong ba):
  - `{"paragraphs": [{"text": "..."}, ...]}` — chỉ lấy `text` dài > 200 ký tự (đây là schema `build_training_corpus.py` sinh ra).
  - `{"samples": [{"context": "..."}, ...]}` — lấy `context` > 200 ký tự.
  - `[...]` — list string, hoặc list dict có khóa `text`/`context`.
- Sinh ra corpus này: `scripts/build_training_corpus.py` (UVW-2026 làm nguồn chính + poetry augmentation, xem §2.1/§2.2). Corpus hiện tại `training_corpus_v1.json` có 22.222 paragraph.

**B. Cặp (context, reference_answer) — cho E4 relevance probe** (`vncompress.training.load_relevance_samples`)

- File: VCC-Bench JSON, `--data-path data/benchmark/vcc_bench_v2.json` trong handoff (hiện repo mới có `vcc_bench_v1.json`, cùng schema — xem cảnh báo §12).
- Shape: `{"samples": [{"context", "reference_answer", "task"}]}`.
- **Lọc bắt buộc:** chỉ giữ sample có `task ∈ {long_document_qa, needle_in_haystack}` (task mà câu trả lời là span/needle nằm trong context, để span-overlap có nghĩa), `context` > 100 ký tự, và `reference_answer` khác rỗng. Sample không có token dương nào sau khi gán nhãn sẽ bị bỏ.
- Không cần bất kỳ trường nào khác trong canonical schema §5.

**C. Evaluation — VCC-Bench** (`benchmark.py` → `vncompress.evaluation.VCCBench`)

- File: `data/benchmark/vcc_bench_v1.json` (243 sample, 5 task) hoặc bộ QA thật `build_viquad_eval.py` (UIT-ViQuAD2.0, chỉ split test, eval-only).
- Sample cần: `{task, context, query, reference_answer}`. Eval đo **hành vi downstream** (EM/ROUGE-L/…), không dùng teacher output làm ground truth — khớp nguyên tắc §10.

**Script build dataset đã tồn tại:** `build_training_corpus.py`, `build_vcc_bench.py`, `build_viquad_eval.py`, `fetch_vietnamese_data.py`, `checksum_datasets.py`. (Các script `generate_*/verify/filter/split` ở §13 **chưa được viết** và E4/E6 không cần chúng.)

---

## 3. Dataset bổ sung cho pipeline V1

Các nguồn dưới đây nên được tích hợp theo từng phase, không coi tất cả là dependency bắt buộc của Wave 2.

| Dataset / nguồn | Vai trò | Ưu tiên |
|---|---|---:|
| UVW-2026 | General Vietnamese corpus | P0 |
| UIT-ViQuAD | Query + passage + answer, tạo query-conditioned samples | P0 |
| ViLegalText | Legal long-context compression | P0 |
| ViLegalTF | Legal factual/entailment supervision | P1 |
| VNFinsQA | Financial QA và reasoning | P0 |
| ViSecQA | Securities/legal QA + hard negatives | P0 |
| Vietnamese news | Tin tức, temporal facts, entity/number preservation | P1 |
| Financial reports | Bảng số liệu, tài chính, dài và nhiều số | P1 |
| Government documents | Văn bản hành chính/pháp luật | P1 |
| Scientific/technical text | Domain robustness | P2 |
| OCR/noisy Vietnamese | Robustness với text lỗi | P2 |

### Tỷ lệ gợi ý cho generated samples

Đây là tỷ lệ trên **sample sau khi teacher generation**, không phải tỷ lệ dung lượng raw corpus:

- UVW-2026: 25%
- UIT-ViQuAD: 10%
- ViLegalText: 15%
- ViLegalTF / MCQ / NLI: 5%
- VNFinsQA: 10%
- ViSecQA: 5%
- Vietnamese news: 10%
- Financial reports: 5%
- Government documents: 5%
- Technical/scientific: 5%
- OCR/noisy Vietnamese: 5%

Có thể điều chỉnh tỷ lệ theo phân bố domain thực tế của production workload.

---

## 4. Teacher LLM pipeline

> **`ROADMAP` — chưa triển khai.** Toàn bộ §4 mô tả một teacher-distillation pipeline tham vọng cho V1. **Không có script nào hiện thực hóa nó, và không training wave-2 nào tiêu thụ output của nó.** Đừng nhầm với cách E6 dùng teacher: E6 chỉ dùng teacher để chấm perplexity keep/drop **trong lúc train** (§2.4/§2.5 B), không sinh compressed_text/preference/hard_negative offline như dưới đây.

Teacher LLM được dùng để biến raw context + query thành supervision chất lượng cao cho student compressor.

### 4.1. Input

Mỗi instance nên có tối thiểu:

```json
{
  "source_id": "...",
  "domain": "finance",
  "query": "...",
  "context": "...",
  "target_ratio": 4
}
```

### 4.2. Teacher tasks

Teacher nên thực hiện các nhiệm vụ sau:

1. Xác định thông tin cần thiết để trả lời query.
2. Xác định important spans/sentences.
3. Gán keep/drop hoặc importance score cho sentence/token.
4. Sinh compressed context ở các ratio 2x/4x/8x.
5. Đánh dấu entity, number, date, percentage, condition, negation và relation quan trọng.
6. Sinh hard negative: context nhìn có vẻ liên quan nhưng không đủ thông tin để trả lời.
7. Sinh preference pair giữa compressed candidates.
8. Giải thích ngắn gọn lý do một span được giữ hoặc loại.

### 4.3. Teacher prompt principle

Teacher không nên chỉ được yêu cầu "tóm tắt văn bản". Prompt phải ràng buộc:

- Compression phải phục vụ **query**.
- Không được làm thay đổi fact.
- Không được tự suy diễn thông tin không có trong context.
- Phải giữ entity/number/date/negation/condition có ảnh hưởng đến answer.
- Output phải đạt target token budget.
- Khi không đủ thông tin, phải giữ lại evidence cần thiết thay vì cố tạo câu trả lời.

---

## 5. Canonical dataset schema

> **`ROADMAP` — chưa được tiêu thụ.** Schema giàu trường dưới đây là đích V1. **Không script training wave-2 nào đọc `compressed_text`, `token_labels`, `important_spans`, `entities/numbers/dates`, `hard_negative`, `preference`, `compression_reason`, hay `quality{}`.** Hợp đồng input thực tế của wave-2 chỉ cần `context` (+ `reference_answer`/`task` cho E4) — xem §2.5. Giữ mục này làm mục tiêu thiết kế, không phải định dạng bắt buộc hiện tại.

Dataset cuối cùng nên chuẩn hóa về một schema thống nhất:

```json
{
  "id": "vncomp_000001",
  "source_id": "...",
  "source": "uvw-2026",
  "language": "vi",
  "domain": "finance",
  "task": "context_compression",
  "query": "...",
  "context": "...",
  "compression_ratio": 4,
  "target_tokens": 256,
  "compressed_text": "...",
  "important_spans": [],
  "token_labels": [],
  "tone_sensitive_spans": [],
  "entities": [],
  "numbers": [],
  "dates": [],
  "conditions": [],
  "negations": [],
  "relations": [],
  "removed_spans": [],
  "hard_negative": null,
  "preference": null,
  "compression_reason": "...",
  "teacher": {
    "model": "...",
    "temperature": 0.1
  },
  "quality": {
    "information_preservation": 0.0,
    "semantic_similarity": 0.0,
    "answer_preservation": 0.0,
    "budget_compliance": true
  }
}
```

`token_labels` có thể encode theo sentence/token classification convention, ví dụ `1 = KEEP`, `0 = DROP`.

---

## 6. Verification và filtering

Không đưa toàn bộ teacher output vào training. Mỗi sample phải qua verification.

### 6.1. Deterministic checks

- JSON/schema hợp lệ.
- Context không rỗng.
- Query không rỗng đối với query-conditioned task.
- Compression ratio thực tế nằm trong tolerance.
- Không mất toàn bộ entity quan trọng.
- Number/date/percentage quan trọng được bảo toàn.
- Không xuất hiện text ngoài nguồn nếu task yêu cầu extractive compression.
- Không duplicate với sample khác.

### 6.2. Semantic checks

Có thể dùng một verifier LLM hoặc model đánh giá riêng để kiểm tra:

```text
Original Context + Query
Compressed Context + Query
          ↓
    Can the same answer be recovered?
```

Chấm riêng:

- Information preservation.
- Answer preservation.
- Semantic similarity.
- Factual consistency.
- Budget compliance.

### 6.3. Teacher agreement

Với sample quan trọng hoặc khó, nên dùng teacher thứ hai/verifier độc lập. Nếu hai đánh giá bất đồng lớn, đưa sample vào quarantine thay vì training trực tiếp.

---

## 7. Hard cases bắt buộc

Dataset cần có tỷ lệ đáng kể các trường hợp mà generic summarization thường thất bại:

- Entity gần giống nhau.
- Số liệu: `22.222,00`, phần trăm, tiền tệ.
- Ngày tháng và khoảng thời gian.
- Phủ định: "không", "chưa", "không được".
- Điều kiện: "nếu", "trừ khi", "trong trường hợp".
- So sánh giữa nhiều đối tượng.
- Quan hệ subject-object.
- Multi-hop evidence.
- Hard negative passages.
- Multi-document context.
- Context dài 8K–128K tokens.
- OCR/noisy Vietnamese.

Đặc biệt, dataset finance/legal cần ưu tiên bảo toàn **numbers + entities + conditions + negations** hơn lexical similarity đơn thuần.

---

## 8. Compression ratios

V1 nên tập trung vào:

- 2x
- 4x
- 8x

16x chỉ nên dùng như stress-test hoặc giai đoạn sau.

Nên lưu cả `target_tokens` và `realized_tokens` để đánh giá budget thực tế, không chỉ dựa trên ratio danh nghĩa.

---

## 9. Train / Validation / Test split

Khuyến nghị split theo **source/document**, không random theo từng sample được sinh từ cùng một document.

Ví dụ:

```text
Train      80%
Validation 10%
Test       10%
```

Nếu một document tạo ra nhiều query/ratio thì toàn bộ các instance của document đó phải nằm cùng một split.

### Không được làm

```text
same document
 ├── query A → train
 ├── query B → validation
 └── query C → test
```

Điều này dễ gây leakage.

### Nên làm

```text
Document A → train
Document B → validation
Document C → test
```

Benchmark test nên được giữ độc lập với teacher-generation pipeline.

---

## 10. Training dataset vs evaluation dataset

### Training

Có thể dùng teacher-generated supervision:

- Keep/drop labels.
- Importance scores.
- Compressed text.
- Preference pairs.
- Hard negatives.

### Evaluation

Không dùng teacher output làm ground truth duy nhất. Evaluation nên dựa trên downstream behavior:

```text
Original Context → Downstream LLM → Answer_original
Compressed Context → Downstream LLM → Answer_compressed
```

Sau đó so sánh:

- Exact Match.
- Token-F1.
- Semantic answer similarity.
- BERTScore/ROUGE-L khi phù hợp.
- Tool-call correctness.
- Entity preservation.
- Number preservation.
- Tone preservation.
- Token reduction.
- Latency/cost reduction.

Metric tổng hợp phải phản ánh trade-off:

```text
Compression ↑
       vs
Answer Preservation ↑
```

Không nên tối ưu một metric compression riêng lẻ rồi kết luận model tốt.

---

## 11. Quy mô dataset V1

> **`ROADMAP`.** Con số 60K instance dưới đây là dataset teacher-distilled tương lai. Wave-2 hiện tại **không** dùng một dataset compressed 3-ratio lưu sẵn: E6 nhận **một** `--ratio` và suy ra tập keep ngay lúc train từ perplexity teacher; E4 chỉ cần cặp (context, reference_answer). Quy mô thực tế đang bị chặn bởi số context trong `training_corpus_v1.json` (22.222 paragraph) và `vcc_bench_v1.json` (243 sample), không phải bởi bước teacher generation này.

Mục tiêu ban đầu:

```text
20,000 source contexts
× 3 ratios (2x/4x/8x)
≈ 60,000 compression instances
```

Có thể mở rộng thêm:

- importance/token-label instances;
- hard negatives;
- preference pairs;
- verifier/quarantine samples.

Sau khi pipeline ổn định mới tăng lên 100K–500K+ instances.

Ưu tiên **quality + domain coverage + hard cases** hơn việc tạo hàng triệu sample bằng teacher một cách mù quáng.

---

## 12. Storage layout

**Hiện tại (đang dùng)** — mọi file phẳng trong `data/benchmark/`:

```text
data/benchmark/
├── training_corpus_v1.json        # corpus text thô cho E6/SLM (paragraphs) — build_training_corpus.py
├── wikipedia_vi_raw.json          # fallback corpus
├── vcc_bench_v1.json              # benchmark 5-task + nguồn (context, reference_answer) cho E4
├── vcc_bench_<task>.json          # bản tách theo từng task
├── CHECKSUMS.json / PROVENANCE.md # provenance
└── (vcc_bench_uit_viquad_qa.json) # sinh bởi build_viquad_eval.py khi cần eval QA thật
```

> **Cảnh báo file:** handoff wave-2 và các mục dưới trỏ tới `data/benchmark/vcc_bench_v2.json`, nhưng repo **hiện chỉ có `vcc_bench_v1.json`** (cùng schema). Cho tới khi có v2, hãy chạy E4/benchmark với `--data-path data/benchmark/vcc_bench_v1.json`, hoặc tạo v2 rồi cập nhật lại. `load_relevance_samples`/`VCCBench` không tự đổi tên file — sai tên sẽ rơi về fallback demo (E4) hoặc báo không thấy dataset (benchmark).

**Đề xuất (`ROADMAP`)** — chỉ cần khi triển khai teacher-distillation §4/§5:

```text
data/
├── raw/{uvw2026,legal,finance,news}/
├── teacher/{compression_raw,importance_raw,preference_raw}.jsonl
├── processed/{train,val,test}.jsonl
└── benchmark/{vcc_bench_v2.json, evaluation_sets/}
```

Teacher raw output nên được giữ lại để có thể audit và re-filter mà không phải gọi LLM lại.

---

## 13. Scripts

**Đã tồn tại (dùng được ngay):**

```text
scripts/
├── build_training_corpus.py   # UVW-2026 + poetry -> training_corpus_v1.json (E6/SLM)
├── build_vcc_bench.py         # dựng benchmark VCC-Bench
├── build_viquad_eval.py       # UIT-ViQuAD2.0 -> bộ QA eval-only
├── fetch_vietnamese_data.py   # tải + segment nguồn HF
└── checksum_datasets.py       # provenance/checksum
```

**Đề xuất, chưa viết (`ROADMAP`, chỉ cần cho teacher-distillation §4/§5 — E4/E6 không cần):**

```text
scripts/
├── generate_compression_dataset.py
├── generate_importance_dataset.py
├── generate_preference_dataset.py
├── verify_dataset.py
├── filter_dataset.py
└── split_dataset.py
```

Pipeline tổng thể (`ROADMAP` cho V1; đường đi wave-2 hiện tại ngắn hơn nhiều — xem §2.5):

```text
Raw sources
    ↓
build / normalize / dedup
    ↓
query + context construction
    ↓
large teacher LLM
    ↓
raw teacher outputs
    ↓
verification
    ↓
quality filtering
    ↓
dedup + source-level split
    ↓
train / val / test
    ↓
E4 relevance probe / E6 encoder compressor / future compressor models
```

---

## 14. Recommended teacher strategy

Teacher model nên là model instruction-following mạnh, có context window đủ lớn. Không nên khóa pipeline vào một model cụ thể; model name phải nằm trong metadata để tái lập thí nghiệm.

Khuyến nghị:

- Temperature thấp (`0–0.2`) cho extraction/labeling.
- Sampling đa dạng chỉ dùng khi sinh candidate/preference.
- Cache teacher output.
- Retry khi output invalid.
- Version hóa prompt.
- Lưu model, prompt version, timestamp và generation parameters.

Một pipeline tốt phải cho phép thay teacher mà không thay schema dataset.

---

## 15. Roadmap triển khai

### Phase 1 — Dataset foundation

- Chuẩn hóa UVW-2026 + corpus hiện tại.
- Tích hợp UIT-ViQuAD, ViLegalText, VNFinsQA, ViSecQA.
- Xây query-conditioned sample generator.
- Sinh 20K source contexts.

### Phase 2 — Teacher distillation

- Sinh 2x/4x/8x compression.
- Sinh token/sentence keep-drop labels.
- Sinh hard negatives và preference pairs.
- Chạy verifier + quality filter.

### Phase 3 — Wave 2 training

- E4 relevance probe.
- E6 encoder token-classification compressor.
- So sánh với baseline compression/scoring hiện tại.

### Phase 4 — Evaluation

- Chạy VCC-Bench v2.
- Đo compression ratio thực tế.
- Đo answer preservation.
- Đo entity/number/date/condition/negation preservation.
- Đo latency và token/cost reduction.

### Phase 5 — Hard-case expansion

- Legal.
- Finance.
- News.
- Long context 32K–128K.
- Multi-document.
- OCR/noisy Vietnamese.

---

## 16. Nguyên tắc nghiên cứu

`vncompress` nên được đánh giá như một **information-preserving compressor**, không phải một summarizer thông thường.

Vì vậy, objective chính nên là:

```text
maximize information preserved
subject to token budget
```

hay trực quan:

```text
Compression ↑  ────────────────┐
                               │
                               ├── Optimize jointly
                               │
Answer Preservation ↑ ─────────┘
```

Một compressed context tốt là context **ngắn hơn đáng kể nhưng downstream LLM vẫn có thể đưa ra cùng câu trả lời đúng**, đặc biệt với các thông tin nhạy cảm như entity, số liệu, ngày tháng, phủ định và điều kiện.
