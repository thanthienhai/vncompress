# YÊU CẦU TÁI TỔ CHỨC REPOSITORY VNCOMPRESS

## 1. Mục tiêu

Tái tổ chức `thanthienhai/vncompress` thành một **research project tập trung vào training, compression experiment và reproducible results**, không xây dựng thành một Python framework lớn.

Mục tiêu chính:

```text
Training
   ↓
Model / Scorer
   ↓
Compression
   ↓
Evaluation
   ↓
Result
   ↓
Research conclusion
```

Repository phải dễ đọc đối với một nhà nghiên cứu:

* Mở repo → biết ngay training ở đâu.
* Biết model nào được train.
* Biết compression method nào được đề xuất.
* Biết benchmark/evaluation ở đâu.
* Biết kết quả nằm ở đâu.
* Có thể reproduce một experiment bằng một command.

Không ưu tiên:

* abstraction quá sâu;
* quá nhiều package;
* quá nhiều interface/protocol;
* kiến trúc enterprise;
* plugin system phức tạp.

---

# 2. Nguyên tắc tổ chức

Áp dụng 5 nguyên tắc:

```text
1. Ít file.
2. Một file có thể chứa nhiều class nếu cùng một responsibility.
3. Không tạo abstraction chỉ để "đẹp kiến trúc".
4. Logic nghiên cứu phải dễ đọc hơn logic framework.
5. Mọi experiment phải tạo ra result có thể truy nguyên.
```

Đặc biệt:

> Ưu tiên **research clarity > software architecture purity**.

---

# 3. Kiến trúc repository mục tiêu

Repository nên được rút gọn về:

```text
vncompress/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── benchmark.json
│   └── training.json
│
├── vncompress/
│   ├── __init__.py
│   ├── compression.py
│   ├── linguistics.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   └── utils.py
│
├── train.py
├── benchmark.py
├── evaluate.py
│
├── tests/
│   ├── test_compression.py
│   ├── test_linguistics.py
│   ├── test_training.py
│   └── test_evaluation.py
│
├── results/
│   └── ...
│
├── docs/
│   ├── benchmark.md
│   └── training.md
│
├── paper/
│   └── ...
│
└── data/
    └── metadata/
```

Không tạo hàng chục subpackage.

---

# 4. Vai trò của từng file

## `vncompress/compression.py`

Đây là **core algorithm**.

Chứa:

```text
- CompressionResult
- CompressionConfig
- Base compressor
- baseline compressors
- LACC compressor
- token selection
- compression budget
- score combination
```

Tất cả logic liên quan trực tiếp tới:

```text
input tokens
→ importance score
→ select tokens
→ compressed tokens
```

để cùng một file.

Không tách thành:

```text
budget.py
selection.py
reconstruction.py
pipeline.py
```

trừ khi file thực tế trở nên quá lớn.

---

# 5. LACC phải là core research method

Không duy trì quá nhiều class:

```text
ToneAwareCompressor
MorphologyAwareCompressor
CombinedCompressor
EnhancedCompressor
SLMScorerCompressor
SLMToneProbeCompressor
...
```

Mà tổ chức theo:

```text
LACCCompressor
```

và các signal:

```text
perplexity
tone
morphology
tone probe
query
```

Ví dụ:

```python
LACCCompressor(
    use_perplexity=True,
    use_tone=True,
    use_morphology=True,
)
```

Ablation chỉ thay config:

```text
PPL
Tone
Morphology
PPL + Tone
PPL + Morphology
Tone + Morphology
PPL + Tone + Morphology
```

Không tạo class riêng cho từng experiment.

---

# 6. `vncompress/linguistics.py`

Gộp toàn bộ Vietnamese-specific linguistic logic vào một file.

Chứa:

```text
- tone detection
- tone analyzer
- tone weights
- tone preservation
- morphology classification
- function words
- reduplication
- compound handling
- Sino-Vietnamese handling
```

Các dictionary nhỏ có thể vẫn nằm trong file.

Không cần tạo:

```text
tones.py
tone_analyzer.py
tone_matrix.py
classifier.py
reduplication.py
segmentation.py
...
```

vì đây là research project, không phải NLP framework.

Chỉ tách resource thành file riêng khi resource thực sự lớn.

---

# 7. `vncompress/models.py`

Chứa toàn bộ logic model-related:

```text
- load tokenizer
- load generation model
- load SLM scorer
- load tone probe
- model inference helper
- model/device utilities
```

Đặc biệt không để việc load model rải rác trong:

```text
compression.py
training.py
benchmark.py
evaluate.py
```

Ví dụ:

```python
model, tokenizer = load_model(...)
scorer = load_scorer(...)
```

Sau đó truyền model vào nơi cần dùng.

---

# 8. `vncompress/training.py`

Đây phải là một trong những file quan trọng nhất của repository.

Chứa:

```text
- training dataset
- preprocessing
- tone labels
- data collator
- training loop
- loss
- tone consistency loss
- LoRA / QLoRA setup
- checkpoint saving
- validation
```

Có thể chứa trực tiếp:

```python
ToneTrainingDataset
ToneAwareTrainer
PhonologicalConsistencyLoss
```

Không cần tách:

```text
datasets.py
collators.py
losses.py
trainers.py
callbacks.py
pipeline.py
```

trừ khi code thực tế vượt quá khả năng quản lý trong một file.

---

# 9. Training phải là trung tâm của repository

Flow chính:

```text
train.py
   ↓
training.py
   ↓
trained model / LoRA adapter
   ↓
benchmark.py
   ↓
evaluation.py
   ↓
results/
```

Không để benchmark hoặc research utility trở thành phần lớn repository trong khi training method là phần cốt lõi.

---

# 10. `vncompress/evaluation.py`

Gộp:

```text
- CompressionMetrics
- ROUGE
- BLEU
- BERTScore
- Exact Match
- Needle Retrieval
- Tone Preservation Rate
- token metrics
- significance testing
- VCC-Bench task execution
```

vào một file.

Nếu file quá lớn thì chỉ được tách thành:

```text
evaluation.py
evaluation_tasks.py
```

Không tạo cả một cây:

```text
evaluation/
├── metrics/
├── tasks/
├── runners/
├── aggregation/
├── significance/
...
```

Research code cần ưu tiên khả năng đọc toàn bộ pipeline.

---

# 11. Giữ Tone Preservation Rate

Đây là một metric quan trọng của nghiên cứu và phải giữ nguyên.

Evaluation cần có ít nhất:

```text
Compression Ratio
Token Saving
ROUGE-L
BLEU
BERTScore
Exact Match
Needle Retrieval
Tone Preservation Rate
Function Word Keep Rate
Content Word Keep Rate
Latency
```

Metric xử lý tiếng Việt phải bảo toàn dấu.

Không sử dụng tokenizer mặc định có thể làm mất thông tin:

```text
bàn
bán
bạn
```

trong quá trình đánh giá.

---

# 12. `utils.py`

Chỉ chứa utility thật sự dùng chung:

```text
- seed
- save json
- load json
- environment info
- git commit
- logging
- reproducibility
```

Không biến `utils.py` thành nơi chứa mọi thứ không biết đặt ở đâu.

---

# 13. Configuration

Chỉ cần:

```text
configs/
├── training.json
└── benchmark.json
```

Không cần:

```text
config/
├── model.py
├── scoring.py
├── compression.py
├── evaluation.py
├── training.py
...
```

Config cần đủ để reproduce:

```json
{
  "seed": 42,
  "model": "...",
  "dataset": "...",
  "max_length": 512,
  "epochs": 3,
  "batch_size": 1,
  "learning_rate": 0.0002,
  "lambda_tone": 0.1,
  "compression_ratios": [2, 4, 8],
  "methods": ["none", "random", "lacc"]
}
```

---

# 14. Root entrypoints

Chỉ giữ ba entrypoint chính:

```text
train.py
benchmark.py
evaluate.py
```

## `train.py`

```bash
python train.py --config configs/training.json
```

## `benchmark.py`

```bash
python benchmark.py --config configs/benchmark.json
```

## `evaluate.py`

```bash
python evaluate.py --input results/run_xxx/
```

Không duy trì nhiều file:

```text
run_training.py
run_train_slm.py
run_ablation.py
evaluate_slm.py
run_eval.py
run_eval_model.py
run_eval_nomodel.py
run_eval_int4.py
...
```

Các chức năng đó phải được gom vào command/config.

---

# 15. Ablation không cần entrypoint riêng

Ví dụ:

```json
{
  "experiment": "ablation",
  "signals": {
    "perplexity": true,
    "tone": true,
    "morphology": false
  }
}
```

Hoặc CLI:

```bash
python benchmark.py \
  --method lacc \
  --signals perplexity,tone
```

Không cần:

```text
run_ablation.py
run_ablation_v2.py
...
```

---

# 16. SLM training và LACC training

Có thể hỗ trợ hai pipeline:

```text
1. LACC model training
2. SLM scorer / tone probe training
```

nhưng vẫn dùng:

```text
train.py
```

Ví dụ:

```bash
python train.py --mode lacc
```

hoặc:

```bash
python train.py --mode slm
```

Implementation nằm trong:

```text
vncompress/training.py
```

---

# 17. Result structure

Kết quả experiment phải có cấu trúc rõ ràng:

```text
results/
└── 2026-09-05_lacc_qwen/
    ├── config.json
    ├── environment.json
    ├── metrics.json
    ├── predictions.json
    ├── summary.json
    └── README.md
```

Nếu training:

```text
results/
└── training/
    └── 2026-09-05_slm/
        ├── config.json
        ├── environment.json
        ├── metrics.json
        ├── checkpoint/
        └── README.md
```

Mục tiêu là:

> Mỗi experiment là một thư mục độc lập và tự mô tả.

---

# 18. Model artifact

Không commit model nặng trực tiếp vào source nếu không cần.

Không nên duy trì:

```text
trained_models_quick/
trained_slm/
```

ở root.

Đề xuất:

```text
artifacts/
```

hoặc tốt hơn:

```text
models/
```

nhưng mặc định nằm trong `.gitignore`.

Git chỉ lưu metadata:

```text
results/.../config.json
results/.../environment.json
```

để biết model nào đã được dùng.

---

# 19. Dataset

Dataset cần phân biệt:

```text
data/
    raw/
    processed/
    benchmark/
```

Nhưng không cần xây một data framework.

Các script build dataset có thể nằm trong:

```text
scripts/
```

hoặc đơn giản hơn:

```text
prepare_data.py
```

Không cần:

```text
scripts/data/
scripts/data/build/
scripts/data/process/
...
```

---

# 20. Research notes

`latent_space_compression/` hiện đang đóng vai trò research notes.

Không nên giữ nó như một package.

Chuyển thành:

```text
research/
├── ideas.md
├── references.md
├── research_gaps.md
└── experiments.md
```

Chỉ giữ những tài liệu thực sự hữu ích cho nghiên cứu.

---

# 21. Paper

Giữ:

```text
paper/
```

nhưng không để paper artifacts ảnh hưởng vào code.

Có thể:

```text
paper/
├── main.tex
├── figures/
└── supplementary/
```

Các file build tạm thời không commit.

---

# 22. README

README phải cực ngắn.

Chỉ gồm:

```text
1. Project overview
2. Research problem
3. Proposed method
4. Architecture
5. Installation
6. Training
7. Benchmark
8. Results
9. Citation
```

Không đưa toàn bộ technical details vào README.

Các thông tin chi tiết chuyển sang:

```text
docs/training.md
docs/benchmark.md
```

---

# 23. Tests

Không cần quá nhiều test module.

Chỉ cần đảm bảo 4 nhóm:

```text
tests/
├── test_compression.py
├── test_linguistics.py
├── test_training.py
└── test_evaluation.py
```

Các test quan trọng nhất:

```text
- compression không tạo sequence dài hơn input
- token order được giữ
- target ratio hoạt động đúng
- tone detection đúng
- tone preservation đúng
- training loss chạy được
- checkpoint load được
- evaluation tạo result hợp lệ
- cùng seed → reproducible
```

---

# 24. Reproducibility

Đây là yêu cầu bắt buộc của scientific project.

Mỗi run phải ghi:

```text
config.json
environment.json
git commit
seed
model
dataset
package versions
device
dtype
```

Ý tưởng `ExperimentConfig`, seed control và snapshot environment hiện có trong repo là đúng và phải giữ lại, chỉ cần đơn giản hóa implementation.

---

# 25. Một CompressionResult duy nhất

Duy trì:

```python
@dataclass
class CompressionResult:
    compressed_ids: list[int]
    compressed_text: str

    original_length: int
    compressed_length: int

    compression_ratio: float
    token_savings_pct: float

    method_name: str
    processing_time_ms: float

    metadata: dict
```

Không duy trì:

```text
NoModelResult
SLMResult
ToneResult
EnhancedResult
```

Các thông tin riêng đưa vào:

```python
metadata
```

---

# 26. Một budget và selection logic

Không được có nhiều implementation:

```text
target_len
mid_budget
boundary
top-k
```

rải rác khắp repository.

Đặt chúng trong `compression.py`.

Tất cả:

```text
baseline
tone
morphology
LACC
SLM
```

sử dụng cùng quy tắc budget.

---

# 27. Tone + Morphology + PPL chỉ là signals

Đây là thay đổi kiến trúc quan trọng nhất.

Không coi:

```text
Tone
Morphology
PPL
SLM
Query
```

là các compressor độc lập.

Coi chúng là:

```text
signals
```

và:

```text
LACCCompressor
```

là algorithm kết hợp signals.

Ví dụ:

```text
LACC
├── Tone signal
├── Morphology signal
├── PPL signal
└── Query signal
```

Điều này giúp nghiên cứu ablation dễ hơn rất nhiều.

---

# 28. Không cần Registry phức tạp

Không cần:

```text
registry/
factory.py
method taxonomy
plugin architecture
dynamic loading
```

Có thể đơn giản:

```python
METHODS = {
    "none": NoCompressor,
    "random": RandomCompressor,
    "llmlingua": LLMLinguaCompressor,
    "snapkv": SnapKVCompressor,
    "lacc": LACCCompressor,
}
```

Nếu chỉ có khoảng 5–10 method thì dictionary là đủ.

---

# 29. Không cần VRAM framework riêng

Không xây:

```text
runtime/
device/
profiles/
resource manager/
...
```

Nếu cần:

```python
load_model(..., device="cuda")
```

hoặc một helper nhỏ trong:

```text
models.py
```

`VRAMManager` hiện tại có ý nghĩa thực tế cho T4/P100, nhưng không nên trở thành một framework hardware riêng. Có thể giữ logic chuyển model CPU/GPU trong `models.py`.

---

# 30. Không cần Quality module riêng

`SemanticQualityGate` là thành phần hữu ích nhưng nên giữ trực tiếp trong:

```text
compression.py
```

hoặc:

```text
evaluation.py
```

nếu chỉ dùng để đánh giá.

Không cần tạo:

```text
quality/
    base.py
    semantic_gate.py
    tone_gate.py
    policy.py
```

trừ khi research sau này thực sự phát triển nhiều loại quality gate.

---

# 31. Không cần quá nhiều abstraction

Không cần bắt buộc tạo:

```text
Protocol
ABC
Strategy
Factory
Adapter
Provider
Pipeline
Registry
```

cho mọi thành phần.

Chỉ cần abstraction ở ba chỗ:

```text
Compressor
Scorer
Training
```

phần còn lại giữ đơn giản.

---

# 32. Mapping từ repository hiện tại

## Giữ / gộp

```text
vncompress/compressors/base.py
vncompress/compressors/no_model.py
vncompress/compressors/tone_aware.py
vncompress/compressors/external_scorer.py
vncompress/compressors/slm_scorer.py
vncompress/compressors/slm_tone_probe.py
vncompress/compressors/llmlingua.py
vncompress/compressors/snapkv.py
```

→

```text
vncompress/compression.py
```

---

```text
vncompress/tone_aware/*
vncompress/morphology/*
```

→

```text
vncompress/linguistics.py
```

---

```text
vncompress/training/*
run_training.py
run_train_slm.py
```

→

```text
vncompress/training.py
train.py
```

---

```text
vncompress/evaluation/*
run_ablation.py
evaluate_slm.py
scripts/run_eval*.py
```

→

```text
vncompress/evaluation.py
benchmark.py
evaluate.py
```

---

```text
vncompress/config.py
```

→ có thể **giữ nguyên một file**:

```text
vncompress/config.py
```

không cần tách thành nhiều module.

---

# 33. Không nhất thiết chuyển sang `src/`

Trong project này **không bắt buộc** chuyển:

```text
vncompress/
```

sang:

```text
src/vncompress/
```

Lý do:

* repository chủ yếu phục vụ research;
* ít package;
* training script cần đọc dễ;
* giảm migration complexity;
* không mang lại nhiều lợi ích so với chi phí refactor.

Chỉ áp dụng `src` layout khi project sau này trở thành public package/library.

---

# 34. Không nhất thiết xóa ngay các file lịch sử

Các file:

```text
compress.txt
TRAINING_TODO.md
```

hoặc research notes cũ chỉ nên xóa sau khi kiểm tra chúng còn chứa thông tin nghiên cứu cần thiết hay không.

Nhưng các file cá nhân:

```text
ThanNgocThien_TTDN.doc
ThanNgocThien_TTDN_CH_HTTT_K15.2.docx
```

không nên nằm trong repository source code.

---

# 35. Target structure cuối cùng

Đề xuất tối giản:

```text
vncompress/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── train.py
├── benchmark.py
├── evaluate.py
│
├── configs/
│   ├── training.json
│   └── benchmark.json
│
├── vncompress/
│   ├── __init__.py
│   ├── config.py
│   ├── compression.py
│   ├── linguistics.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   └── utils.py
│
├── tests/
│   ├── test_compression.py
│   ├── test_linguistics.py
│   ├── test_training.py
│   └── test_evaluation.py
│
├── data/
│   └── metadata/
│
├── results/
│   └── ...
│
├── docs/
│   ├── training.md
│   └── benchmark.md
│
├── research/
│   ├── references.md
│   ├── research_gaps.md
│   └── experiments.md
│
├── paper/
│   └── ...
│
└── .github/
    └── workflows/
        └── ci.yml
```

Đây là cấu trúc nên hướng tới.

---

# 36. Tiêu chí thành công

Sau refactor:

```text
[ ] Repository nhìn vào hiểu ngay đây là project research.
[ ] Training là thành phần trung tâm.
[ ] LACC algorithm nằm trong một file chính.
[ ] Vietnamese linguistic logic nằm trong một file chính.
[ ] Training logic nằm trong một file chính.
[ ] Evaluation nằm trong một file chính.
[ ] Chỉ có 3 entrypoint: train / benchmark / evaluate.
[ ] Không có hàng chục script run_*.py.
[ ] Không có nhiều loại CompressionResult.
[ ] Không có duplicated budget/selection logic.
[ ] Không có registry/factory framework phức tạp.
[ ] Không có runtime/hardware abstraction riêng.
[ ] Mọi experiment có config + environment + result.
[ ] Có thể reproduce training.
[ ] Có thể reproduce benchmark.
[ ] Existing scientific result không bị mất.
[ ] Existing tests vẫn được giữ và cập nhật.
```

---

# 37. Thứ tự refactor

Thực hiện theo thứ tự:

```text
1. Freeze current behavior/results
2. Gộp Compressor
3. Gộp linguistic modules
4. Gộp model loading
5. Gộp training
6. Gộp evaluation
7. Gộp entrypoints
8. Dọn dataset/artifacts/research files
9. Viết lại README
10. Chạy toàn bộ regression
```

Không rewrite toàn bộ một lần.

---

# 38. Nguyên tắc cuối cùng

VNCompress không cần trở thành:

```text
"framework compression hoàn chỉnh"
```

mà nên trở thành:

```text
"một research codebase sạch để chứng minh LACC"
```

Cấu trúc tối ưu là:

```text
          TRAINING
             │
             ▼
       TRAINED SLM
             │
             ▼
       LACC COMPRESSION
             │
      ┌──────┼──────┐
      ▼      ▼      ▼
    TONE   MORPH    PPL
      └──────┼──────┘
             ▼
          BENCHMARK
             │
             ▼
           RESULT
             │
             ▼
        PAPER / CLAIM
```

Mọi phần không trực tiếp phục vụ chuỗi trên đều phải được xem xét loại bỏ hoặc chuyển sang `research/`, `docs/`, `paper/` hoặc artifact storage.

**Ưu tiên số một: đọc code dễ, chạy experiment dễ, reproduce dễ, và nhìn vào kết quả dễ.**
