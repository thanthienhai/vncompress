# Dataset Pipeline: Teacher Distillation, Training & Evaluation

> **Trạng thái (đọc trước):** Tài liệu này gồm **ba lớp**, đừng lẫn lộn:
>
> 1. **`IMPLEMENTED` — xương sống pipeline, đã chạy trên dữ liệu thật.** Normalize → verify → split theo document (90/10) → consumer. Code: `vncompress/dataset.py` + `scripts/{normalize,verify,split}_dataset.py`. Đây là §2.5, §5 (phần core), §6.1, §9, §12, §13.
> 2. **`IMPLEMENTED, CHƯA CHẠY THẬT` — tầng teacher distillation.** §4 (sinh supervision bằng teacher LLM), §5 (các trường teacher), §6 (filter/quarantine), §14 (prompt versioning, cache, retry). Code: `vncompress/teacher.py`, `vncompress/teacher_prompts.py`, `scripts/generate_teacher_dataset.py`, `scripts/filter_dataset.py`. **Toàn bộ đường đi đã verify end-to-end bằng `--dry-run` (stub offline, không tốn token); chưa gọi endpoint thật lần nào.**
> 3. **`PARTIAL`** — §6.2/§6.3 (verifier LLM + teacher agreement), §8.
> 4. **`ROADMAP` — chưa triển khai:** §3 (các nguồn P0 ngoài UVW/ViQuAD), §7 (hard cases), §11 (quy mô 60K), §15 Phase 3–5. Ngoài ra `generate_importance_dataset.py` / `generate_preference_dataset.py` (token label, preference pair, hard negative) vẫn chưa có.
>
> Điểm mấu chốt: E4 và E6 **tự suy ra nhãn trong code** (span-overlap / perplexity teacher), nên chúng KHÔNG phụ thuộc các trường teacher ở §5 — và **hiện vẫn chưa có training nào đọc `compressed_text`**. Tầng teacher hôm nay *sinh ra* supervision, chưa có student tiêu thụ nó.
>
> **Bắt đầu nhanh:**
>
> ```bash
> # Tầng 1 — bắt buộc trước khi train/eval bất cứ thứ gì
> python scripts/normalize_dataset.py   # raw -> data/processed/records.jsonl (canonical, có doc_id)
> python scripts/verify_dataset.py      # §6.1 deterministic checks
> python scripts/split_dataset.py       # 90/10 theo document, chặn nếu rò rỉ
>
> # Tầng 2 — teacher distillation. Chạy --dry-run trước, luôn luôn.
> cp .env.example .env                  # điền endpoint + key (.env đã gitignore)
> python scripts/generate_teacher_dataset.py --stage queries     --dry-run --limit 20
> python scripts/filter_dataset.py      --stage queries
> python scripts/generate_teacher_dataset.py --stage compression --dry-run --limit 20
> python scripts/filter_dataset.py      --stage compression
> ```

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

Đây là thứ code training đọc vào **hôm nay**.

> **Đã đổi:** mọi consumer giờ đi qua `vncompress/dataset.py`. Loader cũ (`load_training_texts`, `load_relevance_samples`) vẫn giữ nguyên chữ ký và hành vi để tương thích ngược, nhưng đường đi mặc định là các hàm split-aware bên dưới. Không cần chạy script pipeline trước: nếu chưa có `data/processed/`, split theo document được suy ra ngay trong bộ nhớ từ file raw — chỉ khác là nó không được ghi lại để audit.

| Consumer | Hàm | Đọc gì |
|---|---|---|
| SLM/tone, LACC-model, E6 | `training.load_train_eval_texts()` | corpus records, split train/eval theo document |
| E4 relevance probe | `training.load_train_eval_relevance_samples()` | cặp (context, reference_answer) từ benchmark records, split theo document |
| VCC-Bench eval | `benchmark.py` | `data/processed/vcc_bench_eval.json` nếu có, ngược lại `data/benchmark/vcc_bench_v1.json` (kèm cảnh báo) |

**A. Corpus text thô — cho E6 và SLM/tone/LACC-model** (`vncompress.training.load_train_eval_texts`)

- File mặc định: `data/processed/{train,eval}.jsonl` nếu đã build; ngược lại `data/benchmark/training_corpus_v1.json` (fallback: `wikipedia_vi_raw.json`, rồi corpus demo built-in).
- Shape chấp nhận (một trong ba):
  - `{"paragraphs": [{"text": "..."}, ...]}` — chỉ lấy `text` dài > 200 ký tự (đây là schema `build_training_corpus.py` sinh ra).
  - `{"samples": [{"context": "..."}, ...]}` — lấy `context` > 200 ký tự.
  - `[...]` — list string, hoặc list dict có khóa `text`/`context`.
- Sinh ra corpus này: `scripts/build_training_corpus.py` (UVW-2026 làm nguồn chính + poetry augmentation, xem §2.1/§2.2). Corpus hiện tại `training_corpus_v1.json` có 22.222 paragraph.

**B. Cặp (context, reference_answer) — cho E4 relevance probe** (`vncompress.training.load_relevance_samples`)

- File: `data/processed/vcc_bench_train.json` (khuyến nghị — đây là phía train của split 90/10), hoặc bất kỳ VCC-Bench JSON nào. File có `metadata.split` là `train`/`eval` sẽ **không bị split lần hai**.
- Shape: `{"samples": [{"context", "reference_answer", "task"}]}`.
- **Lọc bắt buộc:** chỉ giữ sample có `task ∈ {long_document_qa, needle_in_haystack}` (task mà câu trả lời là span/needle nằm trong context, để span-overlap có nghĩa), `context` > 100 ký tự, và `reference_answer` khác rỗng. Sample không có token dương nào sau khi gán nhãn sẽ bị bỏ.
- Không cần bất kỳ trường nào khác trong canonical schema §5.

**C. Evaluation — VCC-Bench** (`benchmark.py` → `vncompress.evaluation.VCCBench`)

- File mặc định: `data/processed/vcc_bench_eval.json` (24 sample held-out) khi split đã build; ngược lại `data/benchmark/vcc_bench_v1.json` (243 sample). Chấm trên file 243 sample trong khi split tồn tại sẽ in cảnh báo, vì probe E4 đã nhìn thấy 90% số đó.
- Hoặc bộ QA thật `build_viquad_eval.py` (UIT-ViQuAD2.0, chỉ split test, eval-only).
- Sample cần: `{task, context, query, reference_answer}`. Eval đo **hành vi downstream** (EM/ROUGE-L/…), không dùng teacher output làm ground truth — khớp nguyên tắc §10.

> **Đánh đổi cần biết:** bộ eval held-out chỉ có **24 sample**, vì `vcc_bench_v1.json` vốn chỉ có 243 sample trên 105 context duy nhất. Con số đó nhỏ tới mức không nên báo cáo như kết quả chính. Cách thoát đúng là có nguồn train query-conditioned riêng (UIT-ViQuAD train split, §3) rồi chạy `split_dataset.py --eval-only-source vcc_bench --eval-only-source wikipedia` để trả VCC-Bench về đúng vai trò eval-only 243 sample.

**Script dataset đã tồn tại:** `normalize_dataset.py`, `verify_dataset.py`, `split_dataset.py` (pipeline chính) + `build_training_corpus.py`, `build_vcc_bench.py`, `build_viquad_eval.py`, `fetch_vietnamese_data.py`, `checksum_datasets.py` (dựng nguồn raw).

---

## 3. Dataset bổ sung cho pipeline V1

> **`ROADMAP` — chưa tích hợp.** Trong bảng dưới, **chỉ UVW-2026 và UIT-ViQuAD có code**: UVW-2026 qua `build_training_corpus.py`, UIT-ViQuAD qua `build_viquad_eval.py` (và mới chỉ split `test`, dùng để eval). ViLegalText / VNFinsQA / ViSecQA được đánh P0 nhưng **chưa có script nào tải hoặc chuẩn hóa chúng**. Khi thêm, chỉ cần cho ra một trong các shape mà `normalize_file()` nhận là vào được pipeline.

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

> **`IMPLEMENTED`, nhưng CHƯA CHẠY THẬT.** Code: `vncompress/teacher.py` (client + cache + retry), `vncompress/teacher_prompts.py` (prompt có version), `scripts/generate_teacher_dataset.py` (sinh), `scripts/filter_dataset.py` (verify + merge). Toàn bộ đường đi đã chạy end-to-end bằng `--dry-run`; **chưa gọi endpoint thật lần nào**, nên chưa có dữ liệu teacher thật trong repo.
>
> Đừng nhầm với cách E6 dùng teacher: E6 dùng teacher để chấm perplexity keep/drop **trong lúc train** (§2.4/§2.5), còn §4 sinh `compressed_text` + span markup **offline** rồi lưu lại.
>
> **Cấu hình** (`.env`, đã gitignore — xem `.env.example`):
>
> | Biến | Ý nghĩa |
> |---|---|
> | `VNCOMPRESS_TEACHER_BASE_URL` | endpoint OpenAI-compatible, có cả path version (`.../v1`) |
> | `VNCOMPRESS_TEACHER_API_KEY` | key; **không bao giờ được log hay ghi vào metadata** |
> | `VNCOMPRESS_TEACHER_MODEL` | tên model teacher, được ghi vào từng row (§14) |
> | `VNCOMPRESS_TEACHER_TEMPERATURE` | mặc định `0.1` — §14 yêu cầu temperature thấp cho extraction/labeling |
> | `VNCOMPRESS_TEACHER_MAX_TOKENS` / `_TIMEOUT` | tham số sinh |
> | `VNCOMPRESS_TEACHER_MAX_ATTEMPTS` | mặc định `3` — tổng số lần thử cho một request |
> | `VNCOMPRESS_TEACHER_RETRY_DELAY` | mặc định `30` giây — chờ cố định trước khi gửi lại |
>
> Loader cũng nhận alias `baseURL` / `apiKey` / `model_name`, để dán thẳng blob credential nhà cung cấp đưa.
>
> **Ba tính chất đáng chú ý:**
>
> - **`--dry-run` đi hết đường ống mà không tốn token.** `DryRunTeacherClient` trả về JSON đúng cấu trúc suy ra từ chính prompt, nên parse → verify → merge → split đều được kiểm tra trước khi trỏ vào endpoint tính tiền. Nó là phép thử đường ống, **không phải phép thử chất lượng** — phần "nén" chỉ là cắt câu đầu.
> - **Cache trên đĩa** khóa theo (model, prompt version, messages, tham số). Sửa prompt của stage này không làm mất cache của stage kia; chạy lại sau khi crash không mất tiền lần hai.
> - **Mặc định đọc split `train`.** Trỏ vào `eval.jsonl` bị **từ chối** trừ khi truyền `--allow-eval-input` — §10 yêu cầu benchmark độc lập với teacher pipeline, và đây là chỗ ép buộc điều đó.
> - **Dry-run ghi ra file riêng** (`*_raw.dryrun.jsonl`). Row stub và row thật trông giống hệt nhau về cấu trúc, nên nếu chung file thì một lần `--dry-run` lỡ tay sẽ nhiễm bẩn dataset mà không có dấu hiệu gì.

### 4.4. Retry, failure log và song song

**Retry — chờ cố định, không exponential backoff.** Một request lỗi chờ `retry_delay` giây (mặc định 30) rồi gửi lại, tối đa `max_attempts` lần (mặc định 3). Lỗi thường gặp trên endpoint dùng chung là rate limit hoặc restart tạm thời, nên một quãng nghỉ dài và đoán trước được vừa nhẹ tay với endpoint vừa dễ suy luận khi có hàng chục worker đang bay. Ngoại lệ: **4xx không phải rate limit thì fail ngay** — retry một request sai chỉ đốt quota và che lỗi thật.

**Failure log — không bao giờ im lặng bỏ qua.** Hết lượt thử, call đó được ghi vào `data/teacher/failures_<stage>.jsonl`: key, record id, stage, loại lỗi, số lần thử, timestamp. Trong một lần chạy hàng chục nghìn request, một lỗ hổng trong dataset là **vô hình** trừ khi có thứ gì đó ghi lại rằng nó đã xảy ra.

```bash
python scripts/inspect_failures.py --stage queries                  # thống kê theo loại lỗi / HTTP status
python scripts/inspect_failures.py --stage queries --check-output    # lỗi nào còn thiếu thật sự
```

Replay không cần script riêng: **chạy lại đúng lệnh generate cũ**. Call lỗi chưa từng được ghi vào output nên bộ resume sẽ tự lên lịch lại.

**Song song (`--workers N`).** Mỗi task là một request HTTPS blocking, nên thread pool gần như tăng tốc tuyến tính cho tới khi endpoint bão hoà. Đo thật trên GLM-5.2 @ FPT Cloud:

| workers | throughput | ETA stage queries (19.960 đoạn) |
|---:|---:|---:|
| 1 | 0.2 req/s | ~26 giờ |
| 16 | 1.25 req/s | ~4.4 giờ |
| 32 | 2.6 req/s | ~2.1 giờ |
| 64 | 3.4 req/s | ~1.6 giờ |

Bão hoà quanh 3.4 req/s; 64 worker không gây lỗi nào. `ResultWriter` khoá và **flush từng row**, nên crash ở giờ thứ sáu vẫn giữ nguyên phần đã ghi và resume chạy tiếp từ đó. Cache ghi qua file tạm rồi `rename`, nên hai worker đụng cùng một prompt không để lại entry ghi dở.

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

> **`PARTIAL` — sinh được, chưa ai tiêu thụ.**
>
> - **Phần core** (`id`, `source`, `doc_id`, `language`, `domain`, `task`, `query`, `context`, `reference_answer`) đã là schema thật, do `vncompress/dataset.py` định nghĩa và mọi consumer đọc.
> - **Phần teacher** (`compression_ratio`, `target_tokens`, `compressed_text`, `important_spans`, `removed_spans`, `entities`, `numbers`, `dates`, `conditions`, `negations`, `compression_reason`, `quality{}`) **đã được `scripts/filter_dataset.py --stage compression` điền vào** `metadata` của record, sau khi qua §6.
> - **Vẫn chưa có gì:** `token_labels`, `tone_sensitive_spans`, `relations`, `hard_negative`, `preference` — cần `generate_importance_dataset.py` / `generate_preference_dataset.py`, chưa viết.
> - **Quan trọng:** *không training nào đang đọc các trường teacher này.* E4 dùng span-overlap, E6 dùng perplexity teacher trong lúc train. Một student học trực tiếp từ `compressed_text` là bước tiếp theo, chưa làm.
>
> Các trường chưa sinh được **reserve theo tên** trong `dataset.RESERVED_FIELDS`, để phân biệt "chưa sinh" với "gõ sai tên".

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

> **`PARTIAL`.** Hai lớp verification đã có:
>
> - **§6.1 trên dữ liệu nguồn** — `scripts/verify_dataset.py` → `vncompress.dataset.verify_records`.
> - **§6.1 trên teacher output** — `scripts/filter_dataset.py`, chỉ chạy được khi đã có output của §4. Xem bảng check ở §6.1 bên dưới.
>
> Row bị loại **không bị xóa** — chúng đi vào `data/teacher/quarantine_<stage>.jsonl` kèm lý do, đúng tinh thần §6.3. Loại bỏ ở đây không cần gọi lại teacher.
>
> **§6.2 (verifier LLM) và §6.3 (teacher agreement / teacher thứ hai) vẫn chưa có.**

Không đưa toàn bộ teacher output vào training. Mỗi sample phải qua verification.

### 6.1. Deterministic checks

> **`IMPLEMENTED`.** Chạy: `python scripts/verify_dataset.py [--input <file>] [--fail-on-error]`. Nhận cả canonical `.jsonl` lẫn file raw. Kết quả ghi ra `verification_report.json`.

| Check | Mức | Ý nghĩa |
|---|---|---|
| `empty_context` | error | context rỗng |
| `duplicate_id` | error | trùng `id` |
| `missing_query` | error | sample query-conditioned nhưng `query` rỗng |
| `missing_reference` | warning | `reference_answer` rỗng |
| `short_context` / `long_context` | warning | ngoài khoảng 200–50.000 ký tự |
| `duplicate_context` | warning | hai record dùng chung nguyên văn context |
| `degenerate_reference` | warning | `reference_answer` là **bản sao nguyên văn** của `context` |
| `answer_not_in_context` | warning | task span-answer nhưng đáp án không phải span nguyên văn |

Kết quả trên dữ liệu đang commit (22.421 record):

```text
[OK  ] empty_context: 0        [HIT ] long_context: 9
[OK  ] duplicate_id: 0         [HIT ] degenerate_reference: 166
[OK  ] missing_query: 0        [HIT ] duplicate_context: 159
[OK  ] missing_reference: 0    [HIT ] answer_not_in_context: 5
```

> **`degenerate_reference: 166`** là phát hiện đáng chú ý nhất và nó **không phải bug của pipeline mà là tính chất của `vcc_bench_v1.json`**: `build_vcc_bench.py` đặt `reference_answer = text` cho `long_document_qa` và nối toàn bộ turn cho `multi_turn_conversation`. 166/243 sample (68%) có đáp án là bản sao nguyên văn context, nên ROUGE-L/BERTScore trên các sample đó thực chất đo "còn giữ lại bao nhiêu chữ", tức là **phạt mọi mức nén theo định nghĩa**. Chỉ 35/243 sample (needle / agent / cross-lingual) có đáp án thật sự khác context. Đây là lý do `build_viquad_eval.py` tồn tại. Check này không chặn pipeline — nó chỉ bắt buộc con số đó phải hiện ra trong report thay vì lộ ra ở bảng kết quả.

### Check trên teacher output

> **`IMPLEMENTED`.** `scripts/filter_dataset.py`. Ba trong số các check này ban đầu tôi thiết kế sai, và chỉ lộ ra khi soi output thật — ghi lại đây vì cùng một loại sai lầm rất dễ lặp.

| Check | Bắt cái gì |
|---|---|
| `empty_output` | không có gì dùng được |
| `not_extractive` | bản nén chứa từ không có trong nguồn — teacher viết lại/bịa thay vì trích xuất |
| `number_altered` | **con số xuất hiện trong bản nén mà không có trong nguồn** |
| `over_budget` | vượt trần, tolerance = phần trăm **hoặc** một khoảng dư tuyệt đối, lấy cái lớn hơn |
| `too_short` | ngắn tới mức vô nghĩa, tính theo **số từ tuyệt đối** |
| `degenerate` | "bản nén" gần bằng cả ngữ cảnh |
| `answer_not_verbatim`, `degenerate_answer` | cho stage queries |

**Ba sai lầm thiết kế đã sửa** (tỷ lệ chấp nhận đo trên cùng tập dữ liệu thật: **80,6% → 89,9% → 95,5%**):

1. **Coi ngân sách là sàn.** §4.3 định nghĩa nó là **trần** ("vượt quá là lỗi"). Với nén theo câu hỏi, độ dài đúng do câu hỏi quyết định. Check `under_budget` cũ loại đúng output tốt nhất trong tập: một needle task nén 6.973 từ xuống 26 từ chính là cái needle. Thay bằng sàn tuyệt đối (`--min-words`).
2. **Tolerance chỉ theo phần trăm.** Ở 8x, target chỉ ~11 từ nên 25% chưa tới 3 từ — loại nhầm do làm tròn. Thêm khoảng dư tuyệt đối (`--budget-slack`).
3. **`number_dropped`.** Trường `numbers` của teacher liệt kê số tìm thấy trong **nguồn**, không phải số bắt buộc giữ. Bản nén `"Nam Phi biểu quyết trắng để bảo vệ chế độ apartheid"` bị loại vì làm rơi một số `13` mà câu hỏi không hề hỏi tới. Đổi thành `number_altered` — bịa/sửa số mới là lỗi toàn vẹn thật sự. Tỷ lệ số được giữ lại ghi vào `quality.numbers_preserved` để phân tích, **không dùng để loại**.

> Số đo đáng chú ý: `number_altered` nổ **1 lần trên 20.318 bản nén**. Trên dataset này teacher gần như không bao giờ bịa con số.

Row bị loại đi vào `data/teacher/quarantine_<stage>.jsonl` kèm lý do — loại bỏ ở đây không cần gọi lại teacher.

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

## 9. Train / Eval split

> **`IMPLEMENTED`.** `scripts/split_dataset.py` → `vncompress.dataset.split_by_document`. Tỷ lệ **90/10 (train/eval)** thay cho 80/10/10 ba phần: wave-2 hiện chỉ có hai vai trò dữ liệu thật (train và held-out eval), thêm một split thứ ba chỉ làm mỏng đi bộ eval vốn đã nhỏ. Muốn ba phần thì chạy `split_dataset.py` lần hai trên `train.jsonl`.

Split theo **source/document**, không random theo từng sample sinh ra từ cùng một document.

### Không được làm

```text
same document
 ├── query A → train
 ├── query B → eval
 └── query C → eval
```

**Đây chính xác là thứ code cũ đang làm** và là lý do mục này được hiện thực hóa:

| Chỗ | Cách split cũ | Hậu quả |
|---|---|---|
| `run_slm_training` | `random_split` mức paragraph, seed 42 | 1 bài UVW-2026 bị cắt thành tối đa 162 paragraph cùng `topic_id` → ước tính **~737/1136 bài bị chia đôi** giữa train và eval |
| `run_lacc_training` | `Subset(range(0.9*n))` mức record | cùng vấn đề, chỉ khác là cắt theo vị trí |
| E4 relevance probe | không split | train trên **toàn bộ** sample `long_document_qa`/`needle_in_haystack` của `vcc_bench_v1.json` — đúng các sample mà `benchmark.py` chấm điểm |
| E6 encoder | không split | không có held-out nào để đo |

### Đang làm

```text
Document A → train
Document B → eval
```

Thuật toán (`split_by_document`):

1. **Stratify theo `(kind, source)`** — mỗi nguồn đều có mặt ở cả hai bên.
2. Trong mỗi stratum, xếp document theo `blake2b(seed:doc_key)` — **hash thuần, không RNG**: tái lập được across máy/phiên bản Python, và thêm document mới không xáo lại các document đã gán.
3. Lấy document vào eval cho tới khi đủ `eval_ratio × số record` của stratum → tỷ lệ đúng ở **mức record**, mà document vẫn nguyên vẹn.
4. Guard: stratum có ≥2 document thì eval không bao giờ rỗng, và không bao giờ nuốt trọn stratum.

`doc_id` được suy ra ở bước normalize:

| Nguồn | Đơn vị document |
|---|---|
| UVW-2026 | `topic_id` (bài viết gốc) |
| Poetry | mỗi chunk là một document |
| VCC-Bench wiki | `wiki:<article>` — 150 sample gộp về **12 bài** |
| VCC-Bench legal | `law:<law_id>` — các chương của cùng một luật là một document |
| `doc_qa_0007_q{0,1,2}` | `doc_qa_0007` |
| `cross_0000_{vi_to_vi,…}` | `cross_0000` |

Kết quả thực tế trên dữ liệu đang commit (`data/processed/split_manifest.json`):

```text
benchmark/vcc_bench         docs=  42  train=   84  eval=   9  (9.7%)
benchmark/wikipedia         docs=  12  train=  135  eval=  15  (10.0%)
corpus/uvw-2026             docs=1136  train=17960  eval=1996  (10.0%)
corpus/vietnamese-poetry    docs=2222  train= 2000  eval= 222  (10.0%)
-------------------------------------------------------------------
tổng                        docs=3412  train=20179  eval=2242  (10.0%)
Leakage check (§9): CLEAN
```

Bất biến này được **assert trong code** (`check_split_leakage`: trùng document / trùng record id / trùng nguyên văn context), `split_dataset.py` **từ chối ghi** một split bị rò rỉ, và `tests/test_dataset.py` kiểm tra nó trên cả fixture lẫn `vcc_bench_v1.json` thật.

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

> **`IMPLEMENTED` — bộ eval được tách theo nguồn gốc.** Khi dữ liệu teacher được trộn vào, phía eval trở thành **482 sample do teacher sinh so với 24 sample độc lập** (số đo thật ở lần chạy đầu). Gộp chung một file thì mọi con số báo cáo bị chi phối bởi việc chấm một model đã train trên output của teacher bằng chính output của teacher đó, và 24 sample độc lập tan biến trong trung bình.
>
> `split_dataset.py` vì vậy ghi ra hai file:
>
> | file | nội dung | đo cái gì |
> |---|---|---|
> | `vcc_bench_eval.json` | chỉ nguồn độc lập (mặc định của `benchmark.py`) | hành vi trên dữ liệu teacher **chưa từng chạm vào** |
> | `vcc_bench_eval_synthetic.json` | chỉ nguồn teacher sinh | khả năng khái quát sang **tài liệu chưa thấy** |
>
> **Không file nào thay thế file kia, và không được gộp.** Gộp thành một con số giờ là việc phải cố ý làm. `--teacher-source` khai báo nguồn nào tính là teacher sinh; `split_manifest.json` có khối `by_source` cho cả train lẫn eval.
>
> Ghi chú chất lượng, trên đúng trục mà repo này liên tục vấp: trong 24 sample độc lập có **15 sample `reference_answer` là bản sao nguyên văn `context`**; trong 482 sample teacher sinh, con số đó là **0**. Bộ eval độc lập vẫn nhỏ và vẫn nhiều degenerate — cách thoát đúng vẫn là UIT-ViQuAD (§2.5 C), không phải dựa vào dữ liệu teacher.

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

> **`IMPLEMENTED`.** `data/processed/` đã tồn tại và là nơi consumer đọc.

```text
data/
├── benchmark/                          # RAW + benchmark gốc (đầu vào của pipeline)
│   ├── training_corpus_v1.json         #   corpus UVW-2026 + poetry — build_training_corpus.py
│   ├── wikipedia_vi_raw.json           #   fallback corpus — fetch_vietnamese_data.py
│   ├── vcc_bench_v1.json               #   benchmark 5-task — build_vcc_bench.py
│   ├── vcc_bench_<task>.json           #   bản tách theo từng task
│   ├── CHECKSUMS.json / PROVENANCE.md  #   provenance
│   └── (vcc_bench_uit_viquad_qa.json)  #   build_viquad_eval.py, gitignored
│
└── processed/                          # DERIVED (normalize -> verify -> split)
    ├── records.jsonl                   #   canonical stream, có doc_id      [gitignored]
    ├── train.jsonl / eval.jsonl        #   split 90/10 theo document        [gitignored]
    ├── vcc_bench_train.json            #   phía train, legacy shape -> E4    [committed]
    ├── vcc_bench_eval.json             #   eval NGUỒN ĐỘC LẬP -> benchmark   [committed]
    ├── vcc_bench_eval_synthetic.json   #   eval teacher sinh, chấm RIÊNG     [committed]
    ├── split_manifest.json             #   policy/seed/tỷ lệ/eval doc keys   [committed]
    ├── records_meta.json               #   nguồn + sha256 đầu vào            [committed]
    └── verification_report.json        #   kết quả §6.1                      [committed]
```

Ba file `.jsonl` bị gitignore vì chúng là ~58MB dẫn xuất hoàn toàn từ file đã có trong git — dựng lại bằng hai lệnh. Các artifact nhỏ khiến một lần chạy **audit được** thì có commit: `split_manifest.json` ghi policy, seed, tỷ lệ thực tế theo từng stratum, **toàn bộ eval document key**, sha256 của từng output, và kết quả leakage check.

> **Cảnh báo file (vẫn còn):** handoff wave-2 trỏ tới `data/benchmark/vcc_bench_v2.json`, repo **chỉ có `vcc_bench_v1.json`**. Đường mặc định giờ không cần tên đó nữa (benchmark tự lấy `data/processed/vcc_bench_eval.json`), nhưng nếu truyền `--data-path` sai tên thì `load_relevance_samples`/`VCCBench` vẫn rơi về fallback demo (E4) hoặc báo không thấy dataset (benchmark).

**Còn thiếu (`ROADMAP`)** — chỉ cần khi triển khai teacher-distillation §4/§5:

```text
data/teacher/{compression_raw,importance_raw,preference_raw}.jsonl
```

Teacher raw output nên được giữ lại để có thể audit và re-filter mà không phải gọi LLM lại.

---

## 13. Scripts

**Pipeline chính (`IMPLEMENTED`) — chạy theo đúng thứ tự này:**

```text
scripts/
├── normalize_dataset.py       # raw (3 shape) -> data/processed/records.jsonl, canonical + doc_id
├── verify_dataset.py          # §6.1 deterministic checks -> verification_report.json
└── split_dataset.py           # 90/10 theo document -> train/eval + manifest; chặn nếu rò rỉ
```

**Tầng teacher (`IMPLEMENTED`, chưa chạy thật):**

```text
scripts/
├── generate_teacher_dataset.py  # --stage queries|compression -> data/teacher/*_raw.jsonl
└── filter_dataset.py            # --stage queries|compression: verify §6 + merge -> records_*.jsonl
vncompress/
├── teacher.py                   # config .env, client OpenAI-compatible, cache, retry, dry-run
└── teacher_prompts.py           # prompt có version (PROMPT_VERSION), §4.2/§4.3
```

**Dựng nguồn raw (đã có từ trước):**

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
├── generate_importance_dataset.py   # token_labels / importance score cho từng token
└── generate_preference_dataset.py   # preference pair + hard negative
```

(`generate_compression_dataset.py` và `filter_dataset.py` của bản đề xuất ban đầu nay là
`generate_teacher_dataset.py --stage compression` và `filter_dataset.py`.)

Đường đi **hiện tại** (`IMPLEMENTED`):

```text
Raw sources (UVW-2026, poetry, Wikipedia, VCC-Bench templates, UIT-ViQuAD)
    ↓  scripts/normalize_dataset.py
canonical records.jsonl  (schema §5 core + doc_id)
    ↓  scripts/verify_dataset.py
verification_report.json  (§6.1)
    ↓  scripts/split_dataset.py
train.jsonl / eval.jsonl  (90/10 theo document, §9)  +  split_manifest.json
    ↓
E4 relevance probe (train split + held-out metrics)
E6 encoder compressor (train split + held-out metrics + distillation_meta.json)
SLM/tone + LACC-model (train/eval split)
benchmark.py (eval split)
```

Đường đi **teacher distillation** (`IMPLEMENTED`, chưa chạy thật):

```text
data/processed/train.jsonl                     (chỉ train — §10 chặn eval)
    ↓  generate_teacher_dataset.py --stage queries
data/teacher/queries_raw.jsonl                 (raw, giữ lại để re-filter)
    ↓  filter_dataset.py --stage queries
data/processed/records_synthetic_qa.jsonl      (kind=benchmark, doc_id kế thừa)
    ↓  generate_teacher_dataset.py --stage compression   (ratio 2/4/8)
data/teacher/compression_raw.jsonl
    ↓  filter_dataset.py --stage compression   -> quarantine_compression.jsonl
data/processed/records_teacher.jsonl           (canonical + trường teacher §5)
    ↓  split_dataset.py --input <gộp>          (dùng lại nguyên vẹn)
train / eval
```

Còn thiếu để khép kín V1: semantic verification §6.2/§6.3, importance/preference generation,
và **một student thật sự học từ `compressed_text`** — hiện chưa có.

---

## 14. Recommended teacher strategy

Teacher model nên là model instruction-following mạnh, có context window đủ lớn. Không nên khóa pipeline vào một model cụ thể; model name phải nằm trong metadata để tái lập thí nghiệm.

> **`PARTIAL`.** Nguyên tắc "model name phải nằm trong metadata để tái lập" **đã được thực thi** cho hai artifact wave-2:
>
> - `scripts/train_encoder_compressor.py` ghi `distillation_meta.json` cạnh checkpoint: `teacher_model`, tín hiệu teacher, `encoder_id`, `ratio`, `seed`, `max_length`, siêu tham số, provenance split, và metric held-out. Trước đây checkpoint E6 **không ghi gì cả** — nhìn vào một thư mục model không thể biết nó được distill từ teacher nào, ở ratio nào.
> - `relevance_probe_meta.json` (E4) và `val_split.json` (SLM) nay ghi kèm provenance split.
>
> Prompt versioning / caching / retry **cũng đã có** cùng tầng teacher: `teacher_prompts.PROMPT_VERSION` nằm trong cache key và trong mọi row sinh ra; `CachedTeacherClient` cache theo (model, prompt version, messages, tham số); `HTTPTeacherClient` retry 429/5xx với backoff và **không** retry 4xx (retry một request sai chỉ đốt quota và che lỗi thật); `--json-retries` retry khi teacher trả JSON hỏng.
>
> Nguyên tắc "thay teacher mà không đổi schema dataset" được giữ đúng: đổi `VNCOMPRESS_TEACHER_MODEL` trong `.env` là xong, schema §5 không đổi, và `model` cũ vẫn nằm trong các row đã sinh.

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
