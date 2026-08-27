# VCC-Bench Evaluation Protocol

Resolves P1 issue "eval: chuẩn hóa VCC-Bench evaluation protocol". Ties
together the unified experiment config
([`vncompress/config.py`](../vncompress/config.py), P1 #7), dataset
provenance ([`vcc_bench_data/PROVENANCE.md`](../vcc_bench_data/PROVENANCE.md),
P1 #8), and the Tone Preservation Rate metric
([`vncompress/docs/tone_preservation_rate.md`](../vncompress/docs/tone_preservation_rate.md),
P1 #6) into one place: what to run, with what fixed settings, and what
the output means.

## Dataset version

Every run against the "official" VCC-Bench evaluates against
`vcc_bench_data/vcc_bench_v1.json` — version `1.0.0`, snapshot date
`2026-07-06`, 243 samples across 5 tasks. See
[`vcc_bench_data/PROVENANCE.md`](../vcc_bench_data/PROVENANCE.md) for full
source/license/regeneration details and
[`vcc_bench_data/CHECKSUMS.json`](../vcc_bench_data/CHECKSUMS.json) for the
exact file hashes a result should be compared against
(`python scripts/checksum_datasets.py` to verify).

Results are only directly comparable across runs that used the **same**
dataset checksum and the **same** git commit — both are captured
automatically per-run (see "Run metadata" below).

## Fixed split

VCC-Bench v1 is a single fixed evaluation set, not a train/val/test split
— see the "Train / validation / test split" section of
`vcc_bench_data/PROVENANCE.md` for the rationale. `ExperimentConfig.split`
is always `"full"` until (if ever) a future dataset version introduces
held-out splits.

## Fixed compression ratios

The canonical ratios for the main benchmark table are **2.0, 4.0, 8.0**
(`ExperimentConfig.compression_ratios` default). `--quick` (CLI) or
`quick=True` (programmatic) shrinks this to a single ratio (`[2.0]`) for
fast iteration — quick runs are for smoke-testing a change, not for
reporting in a paper table; see "Quick/demo vs. full evaluation" below.

## Fixed seed

`ExperimentConfig.seed` defaults to `42` (matching the seed
`scripts/build_vcc_bench.py` used to build the dataset itself, for
consistency). `run_benchmark.py`'s `main()` calls
`vncompress.config.set_seed(seed)` before anything stochastic runs —
covering Python's `random` (used by `RandomCompressor`, class-budget
sampling in `NoModelMorphCompressor`/`NoModelBaselineCompressor`), NumPy,
and PyTorch (CPU + all CUDA devices). Two runs with the same seed, config,
dataset checksum, and code commit should produce identical
`RandomCompressor`/sampling-dependent numbers.

## Model and tokenizer version

Set via `ExperimentConfig.model` / `.tokenizer` (`.tokenizer` defaults to
`.model` if unset — see `resolved_tokenizer()`). Record the exact
HuggingFace model ID (and revision/commit if you've pinned one) in your
`--config` file; the `transformers` version that resolved it is captured
automatically in `environment.json` (see below), since the same model ID
can resolve to different weights/configs across `transformers` releases.

## Generation and evaluation parameters

`ExperimentConfig.max_new_tokens`, `.temperature`, `.do_sample` control
generation for the quality metrics (ROUGE-L, BLEU, BERTScore, exact
match); defaults are `256` / `0.0` / `False` (greedy decoding) so quality
numbers aren't confounded by sampling variance unless you deliberately
opt into it.

## Run metadata: config + environment snapshot

`run_benchmark.py` writes two files into `--output-dir` **before** the
run starts (via `vncompress.config.save_run_metadata`):

- **`config.json`** — the fully-resolved `ExperimentConfig` actually used
  (after `--config` file + CLI-override precedence — see
  `vncompress/config.py`'s module docstring for the precedence rule).
- **`environment.json`** — `git_commit` (+ `git_dirty` flag), UTC
  timestamp, Python version, and installed versions of the key
  dependencies (`torch`, `transformers`, `accelerate`, `peft`, `numpy`,
  `scipy`, `datasets`, `evaluate`, `sentencepiece`, `rouge_score`,
  `sacrebleu`, `bert-score`, `underthesea`).

Any result directory can be traced back to exactly what produced it by
reading these two files — no more guessing which script defaults were in
effect for a given `results/` folder.

## Result JSON schema

`run_benchmark.py` writes `vcc_bench_results.json` into `--output-dir`,
shaped as:

```jsonc
{
  "<method_name>": {                    // e.g. "none", "tone_aware", "combined"
    "<task_name>": {                    // e.g. "long_document_qa"
      "ratio_<R>": {                    // e.g. "ratio_4.0" -- one per compression_ratios entry
        "mean_compression_ratio": 4.02,
        "mean_token_savings_pct": 75.1,
        "mean_processing_time_ms": 12.3,
        "mean_rouge_l_f1": 0.61,
        "mean_bleu": 0.34,
        "exact_match_rate": 0.10,
        "mean_quality_score": 0.55,
        "mean_efficiency_score": 0.80,
        "num_samples": 40,
        "mean_tone_preservation_rate": 0.88   // only present if the method reports TPR -- see docs/tone_preservation_rate.md
      }
    }
  },
  "summary": { "...": "per-method aggregate across all tasks/ratios, see VCCBench._compute_summary()" }
}
```

Per-sample records (one `CompressionMetrics.to_dict()` per sample, see
`vncompress/evaluation/metrics.py`) are additionally written to
`<method>_<task>_ratio<R>.json` when `VCCBenchConfig.save_predictions` is
`True` (the default) — useful for error analysis beyond the aggregated
means above.

`config.json` and `environment.json` (previous section) live alongside
these result files in the same `--output-dir`, so a results directory is
self-describing: dataset + code + dependencies + aggregated results + raw
per-sample predictions, all in one place.

## ⚠️ `device_map='auto'` segfault trên Windows single-GPU (2026-08-26)

Trên máy Windows single-GPU (vd GTX 1060 6GB) dùng để chạy các benchmark
này, `AutoModelForCausalLM.from_pretrained(..., device_map='auto')` gây
**access violation (segmentation fault)** — không phải Python exception,
process chết ngay lập tức, không có traceback hữu ích trừ khi bật
`PYTHONFAULTHANDLER=1`. Hai điểm crash quan sát được (cả hai đều bên trong
`accelerate`/`transformers`, không phải trong code của dự án):

1. `device_map='auto'` → `accelerate.utils.modeling.get_balanced_memory`
   → `get_max_memory` — crash khi truy vấn bộ nhớ đa thiết bị dù máy chỉ
   có 1 GPU. `device_map={'': 0}` (chỉ định thẳng GPU 0) tránh được crash
   này vì né hẳn logic auto-balance.
2. **Nhưng** `transformers` (`modeling_utils.py`, hàm `_load_pretrained_model`)
   gọi `caching_allocator_warmup(model, expanded_device_map, ...)` bất cứ
   khi nào `device_map is not None` — **không phụ thuộc** giá trị của
   `low_cpu_mem_usage` (đã kiểm chứng đọc source: điều kiện chỉ là
   `load_config.device_map is not None and not is_hqq_or_quark`). Hàm này
   gọi `torch.cuda.mem_get_info()`, và lệnh gọi đó access-violation trên
   máy này. Vì vậy `device_map={'': 0}` **+** `low_cpu_mem_usage=False`
   (thử ban đầu) vẫn crash — chỉ né được crash #1, không né được #2.

**Fix thật sự dùng trong repo**: với các model load **không lượng tử hóa**
(fp16/fp32 thường) — `run_benchmark.py`, `run_ablation.py` (nhánh chính),
`run_training.py` (nhánh không QLoRA), `vncompress/compressors/external_scorer.py`
(nhánh fp16), `vncompress/utils/lightweight.py` (`load_tiny_model` fallback),
`scripts/run_eval*.py` — **không truyền `device_map` chút nào**, load model
xong rồi gọi `.to('cuda')` thủ công. Cách này né hẳn `caching_allocator_warmup`
vì điều kiện gọi nó yêu cầu `device_map is not None`. Đây chính xác là cách
`vncompress/compressors/slm_scorer.py` đã làm từ đầu (không truyền
`device_map`, `.to(device)` sau khi load) — và là lý do `run_train_slm.py`/
`evaluate_slm.py` (cũng không dùng `device_map`) chưa từng gặp lỗi này ở
ba lần train SLM trước đó. Đã kiểm chứng: crash 100% khi còn `device_map`
(dù `{'': 0}` + `low_cpu_mem_usage=False`), ổn định 100% khi bỏ hẳn
`device_map`, lặp lại nhiều lần.

Với các model load **có lượng tử hóa bitsandbytes** (INT4/INT8 qua
`quantization_config`) — nhánh INT4 của `external_scorer.py`,
`vncompress/utils/lightweight.py`'s `load_model_4bit`/`load_model_8bit`,
nhánh QLoRA của `run_training.py` — `device_map` là bắt buộc (không thể
`.to()` một model đã lượng tử hóa), nên các nhánh này **vẫn còn rủi ro
crash residual** trên máy này; đã đổi `'auto'` → `{'': 0}` để né ít nhất
crash #1, nhưng chưa có cách né hoàn toàn crash #2 cho load có lượng tử
hóa. Không nằm trong phạm vi benchmark hiện tại (chỉ dùng model fp16
Qwen2.5-0.5B-Instruct, không lượng tử hóa) nên chưa chặn công việc, nhưng
cần lưu ý nếu sau này chạy lại tầng `full/INT4`.

## ⚠️ ROUGE-L: kết quả trước 2026-08-20 không dùng được

`compute_rouge_l` trước đây gọi `rouge_scorer.RougeScorer(...)` **không
truyền tokenizer**. Tokenizer mặc định của thư viện thay mọi ký tự ngoài
`[a-z0-9]` bằng space — mà **mọi nguyên âm mang thanh của tiếng Việt đều
nằm ngoài khoảng đó** (Latin Extended Additional, U+1EA0–U+1EF9). Hệ quả
đã đo được:

```
'bàn' -> ['b','n']   'bán' -> ['b','n']   'bạn' -> ['b','n']
'Hà Nội là thủ đô của Việt Nam' -> ['h','n','i','l','th','c','a','vi','t','nam']

ROUGE-L('bàn của tôi', 'bạn của tôi') = 1.0000   <-- điểm tuyệt đối cho câu SAI
```

Với một dự án mà toàn bộ luận điểm là nén *có ý thức về thanh điệu*, việc
chấm bằng metric **không nhìn thấy dấu thanh** làm vô hiệu mọi con số nó
sinh ra — và ROUGE-L chiếm **40% của `quality_score`**
(`0.4·ROUGE-L + 0.2·BLEU + 0.4·EM`).

Đã sửa bằng `VietnameseRougeTokenizer` (tách âm tiết theo khoảng trắng,
bỏ dấu câu, giữ nguyên dấu thanh; cùng đơn vị đếm với `compute_token_f1`).
Tiền lệ: LongBench làm đúng như vậy cho tiếng Trung (`rouge_zh_score` chạy
jieba trước khi chấm).

**➜ Mọi `results/` sinh ra trước bản sửa này phải chạy lại.** Kiểm tra
`environment.json.git_commit` để biết thư mục kết quả thuộc bản nào.

## Baseline vs. proposed vs. ablation

See [`vncompress/evaluation/method_taxonomy.py`](../vncompress/evaluation/method_taxonomy.py)
for the canonical classification of every `COMPRESSOR_REGISTRY` /
`run_ablation.py` method into `baseline` / `proposed` / `ablation`, and
`scripts/summarize_results.py` for turning a `results/` directory into a
method × ratio × task comparison table that keeps these three categories
visually distinct. Full detail in
[P1 issue #5's resolution notes below](#baseline-vs-proposed-method-separation).

## Quick/demo vs. full evaluation

- **`--quick`**: full `VCCBench.evaluate()` pipeline, but a single
  compression ratio and (when falling back to the built-in demo samples
  rather than a real dataset file) fewer samples. Fast enough for
  iterating on a code change; not a substitute for a full run when
  reporting numbers.
- **`--demo`**: bypasses `VCCBench` entirely — calls `quick_demo()`,
  which runs every method on one hardcoded sample and prints detailed
  per-token output. For eyeballing what a compressor actually does to a
  specific piece of text, not for producing comparable metrics.
- **Full run** (neither flag): all configured methods × all 3 ratios ×
  all 5 tasks × all 243 samples. This is what should back any reported
  table/figure.

## Reproducing a run

```bash
# 1. Full benchmark with the example config (edit model/device as needed)
python run_benchmark.py --config configs/example_experiment.json

# 2. Verify the dataset you're evaluating against hasn't drifted
python scripts/checksum_datasets.py

# 3. Compare results across runs/commits
python scripts/summarize_results.py results/qwen2.5-7b-vcc-bench-v1
```

Given the same `config.json` (dataset path + checksum, model, seed,
ratios, generation params) and the same `git_commit` in
`environment.json`, a re-run should reproduce the same
`mean_compression_ratio` / `mean_token_savings_pct` numbers exactly
(deterministic compression), and the same `RandomCompressor` /
random-sampling-dependent numbers thanks to `set_seed()`. Generation-based
quality metrics (ROUGE-L, BLEU, BERTScore) are also deterministic under
the default greedy decoding (`do_sample=False`) settings, modulo any
non-determinism in the underlying model/hardware's floating-point kernels.
