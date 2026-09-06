# VNCompress — Dataset Strategy & LLM Distillation Pipeline

> Thiết kế dữ liệu nền, pipeline dùng LLM lớn để sinh dữ liệu supervision, huấn luyện LACC và đánh giá model/context compression cho tiếng Việt.

## 1. Mục tiêu

VNCompress không chỉ cần một corpus tiếng Việt lớn. Dataset training phải giúp model học được **context compression có ý thức về ngôn ngữ và nhiệm vụ**, đặc biệt trong tiếng Việt:

- Giữ thông tin cần thiết cho câu hỏi/nhiệm vụ downstream.
- Giữ nguyên thực thể, số liệu, ngày tháng, đơn vị, quan hệ và phủ định.
- Giữ thông tin thanh điệu/dấu tiếng Việt khi cần thiết.
- Loại bỏ thông tin dư thừa thay vì chỉ rút ngắn văn bản.
- Hỗ trợ nhiều compression ratio: `2x`, `4x`, `8x`, và có thể mở rộng `16x`.
- Hoạt động trên nhiều domain thực tế: general knowledge, legal, finance, news, government, technical/scientific và dữ liệu noisy/OCR.
- Có evaluation độc lập để chứng minh compression thực sự hữu ích cho downstream LLM/RAG.

Mục tiêu cuối cùng:

```text
Long Vietnamese Context + Query/Task
              |
              v
       VNCompress/LACC
              |
              v
      Compressed Context
              |
              v
       Downstream LLM
              |
              v
          Answer
```

Không tối ưu chỉ cho `text -> shorter text`. Ưu tiên **query-conditioned context compression**.

---

## 2. Dataset nền: đã sử dụng, đang sử dụng và sẽ mở rộng

### 2.1. Dataset hiện đang có trong repository

#### UVW-2026 — nguồn corpus chính

- Repository hiện dùng `undertheseanlp/UVW-2026` làm nguồn bulk corpus.
- Vai trò: general Vietnamese text, đặc biệt phù hợp để tạo các context dài và đa dạng chủ đề.
- Pipeline hiện tại lọc theo `quality_score` và lấy các paragraph phù hợp.
- Đây là nguồn chính cho `scripts/build_training_corpus.py`.

#### Vietnamese poetry — nguồn augmentation

- Repository hiện hỗ trợ `bigscience-data/roots_vi_vietnamese_poetry`.
- Vai trò: tăng mật độ và độ đa dạng thanh điệu tiếng Việt.
- Chỉ dùng như augmentation, không xem đây là nguồn corpus chính cho context compression.
- Dataset gated/access-controlled; phải tuân thủ điều kiện truy cập và license của nguồn.

Pipeline hiện tại:

```text
UVW-2026 (~90%)
       +
Vietnamese Poetry (~10%)
       |
       v
training_corpus_v1.json
       |
       v
SLM / tone-probe training
```

Chi tiết implementation hiện tại nằm trong `scripts/build_training_corpus.py` và `docs/training.md`.

---

### 2.2. Dataset ưu tiên triển khai tiếp theo

Các nguồn dưới đây được dùng làm **nguồn sinh dữ liệu**, không mặc định dùng trực tiếp làm final training set. Cần kiểm tra version, license, provenance và điều kiện redistribution tại thời điểm download.

| Dataset / nguồn | Domain | Vai trò | Ưu tiên |
|---|---|---|---:|
| `undertheseanlp/UVW-2026` | General | Context nền lớn | P0 |
| `UIT-ViQuAD` | QA / Wikipedia | Query-conditioned compression | P0 |
| `ViLegalText` | Legal | Long-context + hard constraints | P0 |
| `ViLegalTF` / legal QA datasets | Legal | Task-oriented legal supervision | P0 |
| `VNFinsQA` | Finance | Financial QA và số liệu | P0 |
| `ViSecQA` | Securities / Legal | Positive-negative / retrieval-aware samples | P0 |
| Vietnamese news corpus | News | Dữ liệu thực tế, thời sự | P1 |
| Financial reports | Finance | Long document + tables/numbers | P1 |
| Government documents | Government / Legal | Điều khoản, quy định, dates/numbers | P1 |
| Scientific / technical documents | Science / Tech | Thuật ngữ và long-context | P1 |
| Medical Vietnamese datasets | Medical | Domain robustness | P2 |
| OCR/noisy Vietnamese corpus | Noisy text | Robustness với text lỗi | P1 |

### 2.3. Nguyên tắc phân bổ

Tỷ trọng bên dưới là **định hướng cho generated samples**, không phải tỷ trọng raw source:

| Nhóm | Tỷ trọng mục tiêu |
|---|---:|
| General / Wikipedia | 25% |
| Legal | 20% |
| Finance / Securities | 15% |
| QA / Knowledge | 10% |
| News | 10% |
| Government | 5% |
| Science / Technical | 5% |
| OCR / noisy text | 5% |
| Other domain | 5% |

Có thể thay đổi theo kết quả benchmark. Không để một dataset lớn như Wikipedia chi phối toàn bộ training distribution.

---

## 3. Tại sao phải dùng nhiều dataset nền?

Một model chỉ train trên Wikipedia có thể học tốt:

```text
entity + factual paragraph -> shorter paragraph
```

nhưng có thể thất bại trên:

```text
legal clause
financial report
numbers
negation
conditional statement
OCR noise
multi-hop reasoning
```

VNCompress cần kiểm tra những lỗi có chi phí thông tin cao. Ví dụ:

```text
"không được phép"   !=   "được phép"
"22.222,00 tỷ đồng" !=   "22.222 tỷ đồng"
"01/07/2025"        !=   "01/07/2024"
"trừ trường hợp A"   !=   "trường hợp A"
```

Do đó dataset phải có **domain diversity + linguistic diversity + task diversity**.

---

# 4. Pipeline xây dựng training dataset bằng LLM lớn

## 4.1. Tổng quan

LLM lớn đóng vai trò **teacher**, không phải nguồn ground truth duy nhất.

```text
                 RAW DATASETS
                      |
                      v
             Normalize / Clean
                      |
                      v
              Document Sampling
                      |
          +-----------+-----------+
          |                       |
          v                       v
     Query/Task source       No-query samples
          |                       |
          +-----------+-----------+
                      |
                      v
              Teacher LLM #1
                      |
       +--------------+---------------+
       |              |               |
       v              v               v
 Importance       Compression      Hard Negative
  Labels           Candidates        Candidates
       |              |               |
       +--------------+---------------+
                      |
                      v
              Teacher Verifier
                      |
                      v
             Rule-based Validator
                      |
                      v
             Quality / Dedup Filter
                      |
                      v
             Dataset Splitter
                      |
       +--------------+---------------+
       |              |               |
       v              v               v
     Train          Validation        Test
       |              |               |
       +--------------+---------------+
                      |
                      v
                Model Training
                      |
                      v
               VCC-Bench Eval
```

---

## 5. Bước 1 — Chuẩn hóa dữ liệu nền

Mỗi source được chuyển về canonical document format:

```json
{
  "source_id": "uvw_000001",
  "source_dataset": "undertheseanlp/UVW-2026",
  "domain": "general",
  "title": "...",
  "text": "...",
  "language": "vi",
  "metadata": {}
}
```

Các bước preprocessing:

1. Unicode normalization.
2. Chuẩn hóa whitespace nhưng **không được làm mất dấu tiếng Việt**.
3. Loại duplicate/exact duplicate.
4. Loại boilerplate/navigation nếu source có HTML.
5. Kiểm tra độ dài.
6. Phát hiện language.
7. Gắn domain/source metadata.
8. Giữ nguyên raw source để truy nguyên.

Không overwrite raw data.

```text
data/
├── raw/
├── normalized/
├── teacher/
├── processed/
└── benchmark/
```

---

# 6. Bước 2 — Tạo Query/Task

Đây là bước quan trọng để chuyển compression từ generic summarization thành **task-aware compression**.

### 6.1. Nếu dataset đã có question

Ví dụ UIT-ViQuAD, VNFinsQA, ViSecQA:

```text
context + existing question
```

được dùng trực tiếp sau khi validate.

### 6.2. Nếu dataset không có question

Dùng teacher LLM tạo query/task:

```text
Document
   |
   v
Teacher
   |
   +--> factual question
   +--> entity question
   +--> numerical question
   +--> temporal question
   +--> comparison question
   +--> multi-hop question
   +--> reasoning question
```

Không nên để teacher chỉ sinh những câu hỏi dễ trả lời từ một câu.

Mỗi source nên có nhiều mức difficulty:

- Single-span.
- Multi-sentence.
- Multi-hop.
- Entity-heavy.
- Number-heavy.
- Negation/condition.
- Long-context.

---

# 7. Bước 3 — Teacher LLM sinh supervision

Dùng LLM lớn để tạo nhiều loại supervision thay vì chỉ sinh một `compressed_text`.

## 7.1. Compression target

Teacher nhận:

```text
Context
Query/Task
Target compression ratio
```

và tạo:

```text
compressed_context
```

cho các ratio:

```text
2x
4x
8x
16x (optional / hard mode)
```

Teacher phải ưu tiên:

1. Thông tin cần để trả lời query.
2. Entity.
3. Number/date/unit.
4. Negation.
5. Condition/exception.
6. Relation.
7. Tone-sensitive information.
8. Supporting context cần thiết.

---

## 7.2. Importance labels

Teacher đánh dấu span quan trọng:

```json
[
  {
    "text": "22.222,00 tỷ đồng",
    "start": 123,
    "end": 138,
    "importance": 1.0,
    "reason": "number_required_for_answer"
  }
]
```

Có thể phân loại:

```text
CRITICAL
IMPORTANT
OPTIONAL
REDUNDANT
```

Hoặc finer-grained:

```text
KEEP_CRITICAL
KEEP_ENTITY
KEEP_NUMBER
KEEP_DATE
KEEP_RELATION
KEEP_CONTEXT
REMOVE_REDUNDANT
```

Mục tiêu là tạo supervision cho **token/span selection**, không phụ thuộc hoàn toàn vào text generation.

---

## 7.3. Removed spans

Teacher cũng ghi nhận những thông tin có thể bỏ:

```json
{
  "removed_spans": [
    {
      "text": "...",
      "reason": "redundant_for_query"
    }
  ]
}
```

Điều này hữu ích cho preference/ranking và phân tích lỗi.

---

## 7.4. Hard negatives

Teacher tạo các compression candidate gần đúng nhưng có lỗi:

```text
Candidate A: giữ đủ thông tin
Candidate B: mất number
Candidate C: mất entity
Candidate D: mất negation
Candidate E: compression quá mạnh
```

Sau đó teacher/verifier xác định:

```text
A > B
A > C
A > D
A > E
```

Đây là dữ liệu rất quan trọng để model học **what not to remove**.

---

# 8. Bước 4 — Teacher verification

Không dùng output của một lần gọi LLM làm ground truth ngay lập tức.

Pipeline verification:

```text
Teacher Generator
       |
       v
Candidate
       |
       +--> Teacher Verifier
       |
       +--> Rule-based checks
       |
       +--> Answer consistency
       |
       +--> Entity check
       |
       +--> Number/date check
       |
       +--> Tone check
       |
       v
     Quality Score
```

Có thể dùng teacher model thứ hai hoặc một verification pass độc lập.

### Các câu hỏi verifier phải trả lời

1. Compression có trả lời được query không?
2. Answer trước/sau compression có tương đương không?
3. Entity quan trọng có còn không?
4. Number/date/unit có còn chính xác không?
5. Negation có bị mất không?
6. Condition/exception có bị thay đổi không?
7. Quan hệ giữa các entity có bị thay đổi không?
8. Vietnamese diacritics/tone có bị phá không?
9. Compression ratio có đạt target không?
10. Văn bản có còn đọc được không?

---

# 9. Quality scoring

Đề xuất score tổng hợp:

```text
Q =
    0.30 * semantic_preservation
  + 0.25 * answer_preservation
  + 0.15 * tone_preservation
  + 0.10 * entity_preservation
  + 0.10 * number_preservation
  + 0.05 * compression_achievement
  + 0.05 * fluency
```

V1 có thể giữ các sample:

```text
Q >= 0.90
```

Sample trong vùng `0.80–0.90` có thể giữ riêng để nghiên cứu hard cases.

Sample `< 0.80` không đưa vào training chính.

Không nên chỉ dùng teacher score; các field quan trọng phải được kiểm tra bằng deterministic rules khi có thể.

---

# 10. Các loại validation bắt buộc

## 10.1. Entity preservation

```text
Original: Nguyễn Văn A ...
Compressed: Nguyễn Văn A ...
```

Không được tự ý đổi tên.

## 10.2. Number preservation

Đặc biệt với:

```text
22.222,00
10.200 tỷ đồng
3,5%
1.234.567
```

Phải kiểm tra cả format và giá trị.

## 10.3. Date preservation

```text
01/07/2025
2025-07-01
quý IV/2025
```

## 10.4. Negation

```text
không
chưa
không được
không phải
trừ khi
```

## 10.5. Condition / exception

```text
nếu
trong trường hợp
trừ trường hợp
ngoại trừ
với điều kiện
```

## 10.6. Vietnamese tone

Không được biến:

```text
mã -> ma
má -> ma
mả -> ma
mạ -> ma
```

hoặc làm mất dấu trong các span có ý nghĩa.

---

# 11. Canonical generated dataset schema

JSONL là format canonical cho generated dataset.

```json
{
  "id": "vncomp_000001",
  "source_id": "uvw_000001",
  "source_dataset": "undertheseanlp/UVW-2026",
  "language": "vi",
  "domain": "finance",
  "task": "query_conditioned_compression",
  "compression_ratio": 4,
  "query": "Doanh thu năm 2025 của công ty là bao nhiêu?",
  "context": "...",
  "target": "...",
  "important_spans": [],
  "removed_spans": [],
  "tone_sensitive_spans": [],
  "entities": [],
  "numbers": [],
  "dates": [],
  "relations": [],
  "hard_negatives": [],
  "quality": {
    "semantic_preservation": 0.96,
    "answer_preservation": 1.0,
    "tone_preservation": 1.0,
    "entity_preservation": 1.0,
    "number_preservation": 1.0,
    "compression_achievement": 0.98,
    "fluency": 0.95,
    "overall": 0.98
  },
  "teacher": {
    "generator_model": "...",
    "verifier_model": "...",
    "temperature": 0.1,
    "generation_version": "v1"
  },
  "provenance": {
    "source_license": "...",
    "source_checksum": "..."
  }
}
```

Raw teacher reasoning/thinking không lưu vào training dataset. Chỉ lưu structured supervision và metadata cần thiết để audit.

---

# 12. Dataset dành cho từng mục tiêu training

Không dùng một dataset duy nhất cho mọi loss.

## 12.1. Compression SFT dataset

```text
context + query + ratio -> compressed_context
```

Mục tiêu: học behavior compression.

## 12.2. Importance dataset

```text
context + query -> token/span importance
```

Mục tiêu: học selection signal.

## 12.3. Preference dataset

```text
context + query + candidate_A + candidate_B
                         |
                         v
                    A preferred
```

Mục tiêu: học ranking giữa compression candidates.

## 12.4. Tone dataset

Các sample tập trung vào:

- tone-bearing tokens
- diacritics
- minimal pairs
- word boundaries
- compounds
- Vietnamese morphology

## 12.5. Hard-case dataset

Tập trung vào:

- number
- date
- entity
- relation
- negation
- condition
- multi-hop
- cross-document
- OCR/noisy text

---

# 13. Dataset size roadmap

## V1 — Proof of concept

```text
20,000 source contexts
× 3 ratios
≈ 60,000 compression instances
```

Cộng thêm importance/preference/hard-negative samples.

Mục tiêu V1: xác nhận teacher pipeline và kiểm tra model có thực sự học được compression tốt hơn baseline.

## V2 — Research scale

```text
100,000 source contexts
× 3 ratios
≈ 300,000 compression instances
```

Có domain balancing và hard-case mining.

## V3 — Production/research scale

```text
300,000 source contexts
× 3 ratios
≈ 900,000 compression instances
```

Có thể mở rộng lên ~1M+ generated samples nếu quality không giảm.

**Ưu tiên quality over quantity.** 1M sample kém chất lượng có thể tệ hơn 100k sample được verifier kỹ.

---

# 14. Train / validation / test split

Không split ngẫu nhiên đơn thuần theo generated samples.

Phải split theo **source/document** trước khi teacher generation hoặc ít nhất theo `source_id`, tránh cùng một document xuất hiện ở train và test dưới các ratio/query khác nhau.

Đề xuất:

```text
Train       80%
Validation  10%
Test        10%
```

Quan trọng hơn tỷ lệ là:

```text
source-level split
+ domain stratification
+ difficulty stratification
```

Benchmark test phải được **đóng băng** sau khi tạo và không được dùng làm teacher-training data.

---

# 15. Benchmark/evaluation dataset

Training dataset và evaluation dataset phải tách biệt.

Evaluation gồm hai lớp.

## 15.1. Intrinsic compression evaluation

Đo:

- realized compression ratio
- token reduction
- character reduction
- semantic similarity
- information preservation
- tone preservation
- entity preservation
- number preservation

## 15.2. Downstream evaluation

```text
Original context -> LLM -> Answer

Compressed context -> LLM -> Answer
```

So sánh answer quality.

Các task chính của VCC-Bench:

1. Long-Document QA.
2. Multi-turn Conversation.
3. Needle-in-a-Haystack.
4. Agent Tool-Calling.
5. Cross-lingual.

Metrics hiện có trong repository:

- ROUGE-L
- BLEU
- BERTScore
- Exact Match
- Token-F1
- Tone Preservation Rate
- Harmonized Score

Cần mở rộng evaluation với các metric chuyên biệt cho:

```text
Entity Preservation
Number Preservation
Date Preservation
Negation Preservation
Answer Consistency
```

---

# 16. Baselines bắt buộc

Mỗi evaluation phải so sánh ít nhất:

```text
Original / No compression
        |
        +-- LLMLingua
        +-- SnapKV (nếu protocol phù hợp)
        +-- Simple token/heuristic baseline
        +-- LACC rule-based
        +-- LACC + SLM scorer
        +-- LACC trained model
```

Mục tiêu không phải chứng minh compressed text đẹp hơn mà chứng minh:

```text
same downstream quality
        hoặc
small quality loss
        với
large token reduction
```

---

# 17. Data provenance và reproducibility

Mỗi dataset phải có:

```text
data/...
├── README.md
├── PROVENANCE.md
├── CHECKSUMS.json
└── metadata.json
```

Record:

- source dataset
- version/revision
- URL/repository identifier
- license
- download date
- preprocessing version
- sample count
- checksum
- teacher model
- verifier model
- prompt version
- generation parameters
- filtering version

Generated data phải truy ngược được:

```text
final sample
    -> generated sample
    -> normalized source
    -> raw source
```

---

# 18. Recommended directory structure

```text
vncompress/
├── data/
│   ├── raw/
│   │   ├── uvw/
│   │   ├── viquad/
│   │   ├── legal/
│   │   ├── finance/
│   │   └── news/
│   │
│   ├── normalized/
│   │
│   ├── teacher/
│   │   ├── compression/
│   │   ├── importance/
│   │   ├── preference/
│   │   └── hard_negative/
│   │
│   ├── processed/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   │
│   └── benchmark/
│
├── scripts/
│   ├── build_training_corpus.py
│   ├── generate_compression_dataset.py
│   ├── generate_importance_dataset.py
│   ├── generate_preference_dataset.py
│   ├── verify_dataset.py
│   ├── filter_dataset.py
│   └── split_dataset.py
│
└── docs/
    ├── training.md
    ├── benchmark.md
    └── dataset_pipeline.md
```

---

# 19. Implementation roadmap

## Phase 1 — Dataset infrastructure

- [x] UVW-2026 corpus builder.
- [x] Vietnamese poetry augmentation.
- [x] Dataset checksum/provenance cho corpus hiện tại.
- [ ] Canonical JSONL schema.
- [ ] Source-level split.
- [ ] Dataset metadata registry.

## Phase 2 — Teacher generation

- [ ] `generate_compression_dataset.py`.
- [ ] Prompt versioning.
- [ ] Batch generation với large LLM.
- [ ] Retry/error handling.
- [ ] Resume từ checkpoint.
- [ ] Structured JSON validation.

## Phase 3 — Verification

- [ ] Teacher verifier.
- [ ] Entity preservation checker.
- [ ] Number/date checker.
- [ ] Negation/condition checker.
- [ ] Tone preservation checker.
- [ ] Compression ratio checker.
- [ ] Semantic/answer consistency checker.

## Phase 4 — Training

- [ ] Compression SFT.
- [ ] Importance supervision.
- [ ] Preference/ranking supervision.
- [ ] Tone-aware auxiliary loss.
- [ ] Multi-task training.

Conceptual loss:

```text
L_total =
    λ1 L_LM
  + λ2 L_importance
  + λ3 L_selection
  + λ4 L_preference
  + λ5 L_tone
```

## Phase 5 — Evaluation

- [ ] VCC-Bench integration.
- [ ] Domain-balanced test set.
- [ ] Hard-case test set.
- [ ] 2x / 4x / 8x / 16x evaluation.
- [ ] Statistical significance.
- [ ] Error taxonomy.
- [ ] Ablation: PPL / tone / morphology / learned importance.

---

# 20. Recommended first experiment

Không nên bắt đầu bằng 1M samples.

### Dataset V1

```text
Source:
  UVW-2026          10,000
  UIT-ViQuAD         3,000
  Legal              3,000
  Finance            2,000
  News               1,000
  Hard cases         1,000
                    ------
                    20,000 contexts
```

Mỗi context tạo:

```text
2x
4x
8x
```

→ khoảng `60k compression samples`.

Mỗi sample có:

```text
context
query
ratio
target
importance
entities
numbers
dates
removed_spans
quality
```

Sau đó train một model nhỏ trước và chạy:

```text
Original
vs
LACC baseline
vs
LACC trained
```

trên cùng một test set.

Nếu `4x` hoặc `8x` đạt quality tốt và giảm token rõ rệt, mới scale teacher generation lên `100k → 300k contexts`.

---

# 21. Nguyên tắc quan trọng

### 1. Không biến bài toán thành summarization

Compression phải trả lời câu hỏi:

> "Thông tin nào cần được giữ lại để downstream task vẫn đúng?"

không chỉ:

> "Đoạn này có thể viết ngắn hơn thế nào?"

### 2. Teacher không phải ground truth tuyệt đối

Teacher output luôn phải qua verifier + deterministic checks + dedup/filter.

### 3. Test set không được sinh từ cùng source với train

Tránh leakage theo document/source.

### 4. Query-conditioned quan trọng hơn generic compression

Một token có thể không quan trọng với query A nhưng cực kỳ quan trọng với query B.

### 5. Số liệu/entity/negation phải được bảo vệ đặc biệt

Đây là những lỗi compression dễ nhìn thấy và có giá trị ứng dụng cao.

### 6. Tone preservation phải được đánh giá độc lập

TPR cao không có ý nghĩa nếu tokenizer gần như không có tone-bearing tokens. Luôn báo cáo tone-bearing token ratio cùng TPR.

### 7. Ưu tiên domain khó

Legal và finance nên được xem là stress-test quan trọng, không chỉ là dữ liệu bổ sung.

### 8. Generated dataset phải reproducible

Cùng source + prompt version + teacher version + seed/config phải có khả năng tái tạo hoặc audit được generated dataset.

---

# 22. Kết luận

Chiến lược dữ liệu của VNCompress nên đi theo hướng:

```text
Vietnamese Foundation Datasets
            |
            v
     Domain + Task Diversity
            |
            v
      Large LLM Teacher
            |
   +--------+---------+
   |        |         |
   v        v         v
Compression Importance Preference
   |        |         |
   +--------+---------+
            v
       Verification
            |
            v
      Quality Filtering
            |
            v
      Train / Val / Test
            |
            v
       LACC Training
            |
            v
        VCC-Bench
            |
            v
      Error Analysis
            |
            +----> Hard-case mining
                         |
                         v
                  Teacher generation
                         |
                         └──> next dataset version
```

Điểm cốt lõi là xây một **feedback loop** thay vì tạo dataset một lần:

```text
Train → Evaluate → Find failures → Mine hard cases → Teacher generate → Verify → Retrain
```

Đây là hướng phù hợp nhất để VNCompress phát triển từ research prototype hiện tại thành một hệ thống **Vietnamese context compression có dataset, training pipeline và benchmark có thể tái lập**.
