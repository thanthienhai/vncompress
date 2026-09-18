# LACC → COLING 2027 — Task list & timeline ngược

> **Mục tiêu paper:** Một *resource + analysis paper* — VCC-Bench v2 (benchmark QA tiếng Việt có kiểm soát rò dữ liệu) là **đóng góp chính**, kèm finding "giữ dấu thanh ≠ giữ thông tin cho QA".
> **Venue:** COLING 2027, Macau (9–14/5/2027). CCF B / CORE B.
> **Nộp qua:** ARR (ACL Rolling Review — hệ review dùng chung của họ ACL), chu kỳ **tháng 10/2026**.
> **HẠN CỨNG:** **12/10/2026**. Hôm nay 18/9 → còn ~3.5 tuần.
> **Lưu ý thực tế:** timeline này rất nén. Nếu 12/10 không kịp chất lượng, bài đã review ở ARR có thể **commit sang NAACL/ACL 2027 sau** mà không viết lại — nên cứ nhắm 12/10, xấu nhất là dời venue chứ không dời việc.

---

## ✅ Hai quyết định Bước 0 (đã chốt)

- [x] **Trục câu chuyện:** benchmark + negative result (resource paper). Hướng query-aware / efficiency → gác sang Future Work.
- [x] **E6 (encoder classifier):** **query-agnostic** (nén-một-lần, không nhìn câu hỏi) — đóng vai *control* cô lập trục cơ chế (B), tách khỏi trục query (A do E4 lo).

---

## 🗓️ Timeline ngược từ 12/10

| Giai đoạn | Cửa sổ | Mốc phải đạt cuối giai đoạn |
|---|---|---|
| G0 — Setup | nay → 21/9 (cuối tuần này) | Harness đã thêm metric mới, sẵn sàng chạy |
| G1 — Có số | 22/9 → 28/9 | `benchmark.py` chạy xong, bảng §5 có số điền |
| G2 — Baseline + viết | 29/9 → 5/10 | Đủ baseline; §3 + §5 bản nháp xong |
| G3 — Hoàn thiện | 6/10 → 11/10 | Full draft + checklist ARR + anonymize |
| Nộp | **12/10** | Submit ARR |

---

## G0 — Setup harness (nay → 21/9) · P0

Đây là việc **chặn mọi thứ**: chưa có metric thì chạy benchmark cũng vô nghĩa.

- [x] **Thêm metric evidence/attribution vào harness VCC-Bench** (chỉ sửa benchmark, *không* train lại) — `vncompress/evaluation.py`, validated bằng benchmark.py chạy thật trên GPU (Qwen2.5-0.5B, VCC-Bench v2, 2 mẫu/task):
  - [x] `evidence_recall` — câu chứa đáp án (xấp xỉ bằng token-overlap cao nhất, VCC-Bench v2 không có gold span) có sống sót sau nén không. Không cần NLI, luôn tính.
  - [x] `source_span_recoverability` — P(compressed context ⊨ evidence sentence), chấm bằng NLI (`--nli-model`, mặc định tắt để không bắt buộc tải model nặng). Đã test thật: entailment true/false case cho điểm 0.97 / 0.001.
  - [x] `unsupported_claim_rate` — tỷ lệ câu trong output không được compressed context chứng thực (NLI, cùng cờ `--nli-model`).
- [x] **Đổi metric E4** trong harness: `vncompress/training.py::evaluate_relevance_probe` giờ trả **PR-AUC** (sklearn `average_precision_score`) + **recall@budget** (2x/4x/8x, per-window ranking) thay vì chỉ argmax F1 (vẫn giữ P/R/F1 để tham khảo).
- [x] Chốt **danh sách arm** cho benchmark: `none, random, llmlingua, llmlingua_contrastive, lacc_ppl_contrastive, lacc_ppl_morph, lacc_cx_morph, lacc_sentence, lacc_classprop, lacc_tone_gated, encoder` — đã khớp `benchmark.py:WAVE2_ARMS`; sửa bug `categorize()` thiếu `lacc_cx_morph`/`lacc_tone` (sẽ crash khi tổng hợp bảng); cập nhật `configs/benchmark.json` và `run_pipeline.sh:BENCH_METHODS` (thiếu `llmlingua`, `lacc_tone_gated`) cho khớp danh sách này.

**Bug chặn G1 phát hiện + đã sửa khi validate (không có trong todo gốc nhưng chặn cứng mọi run):**
- [x] `VCCBench._build_prompt_ids` (evaluation.py): `tokenizer.apply_chat_template(tokenize=True)` trả về `BatchEncoding` trên transformers 5.12.1 (không phải list ids như code cũ giả định) → MỌI generation của MỌI arm crash với `RuntimeError: Could not infer dtype of tokenizers.Encoding`. Đã sửa (`encoded['input_ids']`), verified bằng run thật (generation FAILED biến mất, quality score khác 0).
- [x] `scripts/verify_tone_probe_e2e.py` gắn nhãn cứng "lacc_tone_probe"/"Tone-Probe" dù đang load đúng relevance probe (dùng cho E4 A/B ở G2) → thêm `detect_probe_kind()` tự nhận `probe_kind` từ sidecar meta json, đổi tên arm động (`lacc_relevance_probe` khi cần).
- [x] `run_pipeline.sh` không truyền `--data-path` cho 2 lần gọi `verify_tone_probe_e2e.py` trong `stage_bench` → mặc định rơi về `vcc_bench_v1.json` (script's default) thay vì v2; đã thêm `--data-path "$BENCH_DATA_PATH"`.

---

## G1 — Chạy benchmark, lấy số (22/9 → 28/9) · P0

- [ ] **Chạy `benchmark.py` trên VCC-Bench v2** với đủ arm ở trên → đây là việc-thiếu-lớn-nhất, không có nó thì mọi bảng §5 trống. **Trạng thái:**
  - [x] **Bug nghiêm trọng đã tìm & sửa (2026-09-18):** mọi arm dùng `--scorer-adapter-dir` (tất cả arm `lacc_*`) crash `CUDA device-side assert` trên context ≥1025 token (needle/long_document_qa — tức GẦN NHƯ MỌI SAMPLE). Nguyên nhân: E3 nâng cửa sổ perplexity 512→2048 nhưng SLM mặc định (`chronopt-research/vietnamese-gpt2-base`) chỉ có `n_positions=1024` — **bug 100% tái hiện trên mọi máy, kể cả GPU pod, không phải do GPU yếu**. Đã sửa `models.load_scorer` + `compression.py` (`model_max_positions()`, tự clamp window theo cap thật của model) — verify lại đúng sample từng crash giờ chạy được. Chi tiết: memory `e4-probe-v2-wiring`.
  - [ ] Full sweep (414 mẫu × 13 arm × 3 ratio, + arm `encoder`/`translate_then_compress_llmlingua2` cần checkpoint `models/encoder_compressor` chưa có ở máy local) cần chạy trên GPU pod thật (`./run_pipeline.sh --stages bench`, hoặc `data,slm,probe,encoder,bench` nếu thiếu checkpoint) — máy local (GTX 1060 6GB) không đủ VRAM cho model sinh mặc định + hay bị hệ thống dừng job do thiếu RAM khi chạy nền lâu. Harness đã sẵn sàng, không còn bug đã biết nào chặn.
  - **Quyết định 2026-09-18:** đã ước tính thời lượng thật trên máy local (needle_in_haystack ~60-120s/mẫu × 120 mẫu × 11 arm × 3 ratio ⇒ nhiều giờ-tới-cả-ngày, rủi ro máy tự kill job do thiếu RAM). Đã hỏi người dùng — **chọn tự chạy `./run_pipeline.sh` trên GPU pod (H100)**, không chạy sweep đầy đủ trên máy local. Phiên này chuyển sang làm tiếp phần code/paper không cần GPU trong lúc chờ số liệu thật.
- [ ] Kiểm tra sanity kết quả (arm `none` phải cao nhất, `random` thấp — nếu ngược là bug harness)
- [ ] Xuất bảng thô cho: `tab:compression_efficiency`, `tab:quality`, `tab:tone_preservation`, `tab:ablation`

---

## G2 — Baseline + bắt đầu viết (29/9 → 5/10) · P1

**Baseline bắt buộc (reviewer COLING sẽ hỏi):**
- [ ] **Dịch-rồi-nén** (VN → EN → LLMLingua-2): baseline *sống-còn* — một paper độc lập (*Lost in Compression*) cho thấy nó thắng nén bản-ngữ ở nhiều ngôn ngữ. Không có nó, premise "cần compressor tiếng Việt" bị thủng. **Đã có sẵn trong code** (`TranslateThenCompressCompressor`, arm `translate_then_compress`, mặc định inner=LLMLingua) — không phải viết từ đầu. Đã thêm arm `translate_then_compress_llmlingua2` (inner=encoder/E6, đúng nghĩa "LLMLingua-2") vào `benchmark.py:WAVE2_ARMS` + `run_pipeline.sh:BENCH_METHODS` + `configs/benchmark.json`; validate real GPU run cho biến thể LLMLingua thường (arm `translate_then_compress`, Qwen2.5-0.5B, 1 mẫu/task, không lỗi). **Còn lại:** chưa chạy full trên VCC-Bench v2 (thuộc G1), và biến thể `_llmlingua2` cần checkpoint encoder (`models/encoder_compressor`, chưa có ở máy local) nên chưa smoke-test được ở đây — chạy cùng lúc với arm `encoder` trên GPU pod.
- [x] **LongLLMLingua** (query-aware) + **LLMLingua-2** (encoder) vào bảng so sánh (E11) — đã có sẵn: `llmlingua_contrastive` (LongLLMLingua) và `encoder` (LLMLingua-2/PhoBERT) đều đã trong danh sách arm G0.
- [x] **Báo achieved-rate** (tỷ lệ nén *thực đạt*, không phải đặt) trên **cả 2 tokenizer**: PhoBERT *và* LLM đích (hai cái lệch nhau) — `vncompress/evaluation.py`: thêm `get_phobert_tokenizer()` (lazy, mặc định `vinai/phobert-base`), field `phobert_original_tokens`/`phobert_compressed_tokens`/`phobert_achieved_ratio` trong metadata mỗi mẫu + `mean_phobert_achieved_ratio` khi tổng hợp; cờ `--phobert-tokenizer` trong `benchmark.py` (rỗng để tắt). Phía tokenizer sinh văn bản (`compression_ratio`=achieved, `requested_ratio`=đặt) đã có sẵn từ trước, không cần thêm. **Verify thật** (nhanh, không chạy hết theo yêu cầu): text gốc 304 token (tokenizer Qwen) vs 261 token (PhoBERT) — hai tokenizer lệch nhau đúng như kỳ vọng; `phobert_achieved_ratio=1.0` đúng cho arm `none`.

**Train (song song):**
- [ ] Train lại **probe với focal loss** (asymmetric focal — phạt nặng lớp hiếm để kéo precision), đo lại bằng PR-AUC/recall@budget
- [ ] **A/B `verify_tone_probe_e2e.py`**: relevance-probe vs tone-probe vs rule → đây là phép đo "đảo dấu wave 1" của E4

**Bắt đầu viết (bản nháp):**
- [x] **§3 Phương pháp** viết lại khớp code thật (làm sớm, tận dụng lúc chờ G1 — `paper/lacc_paper.tex`): thêm mục **query-conditioning (E1)** + hệ số query-boost từ-khoá (eq. mới `eq:cond_ppl`/`eq:contrastive_ppl`/`eq:query_boost`), biến thể **blend nhân (E2)** cạnh công thức cộng gốc (`eq:combined_mult`), mục **sentence-level (E5)** mới (`eq:sentence_select`, §3.9), mục **kiến trúc thay thế encoder-classifier (E6)** mới (§3.12), sửa cửa sổ **512 → 2048** ở Thuật toán 1 + văn bản quanh eq:ppl, thêm mục **Relevance Consistency Loss (E4)** trình bày song song Phonological (chưa khẳng định cái nào thắng, chờ A/B), thêm mục **"Định vị E6"** ở Thảo luận (trục A/B độc lập, so với baseline nào) + item Hạn chế nói rõ trần của E6. Đã kiểm tra cross-ref/brace-balance bằng script (không có ref/label lỗi). **Còn thiếu:** không tự dùng LaTeX compiler được (máy không có `pdflatex`) — cần build thử trước khi nộp; §5 vẫn để `---` (chờ G1); §6.1 "Tại sao tín hiệu ngôn ngữ quan trọng?" và toàn bộ Limitations/Kết luận CHƯA viết lại theo hướng negative-result (đúng lịch, đây là việc của G3, cần số thật trước).
- [ ] Điền **§5** bằng số benchmark từ G1

---

## G3 — Hoàn thiện (6/10 → 11/10)

**Định vị & framing (quan trọng cho COLING):**
- [x] **Framing resource paper:** nêu rõ VCC-Bench v2 là đóng góp chính; negative-result là finding đi kèm — viết lại Abstract + đoạn "Đóng góp của bài báo" trong Giới thiệu (`paper/lacc_paper.tex`): VCC-Bench v2 giờ là đóng góp (1), kết quả âm có kiểm soát là (2), khung LACC là (3)/công cụ đo — trước đó abstract còn khẳng định "LACC vượt trội hơn các phương pháp hiện có" (mâu thuẫn trực tiếp với quyết định Bước 0), đã sửa. Cũng đã ẩn danh `\author{}` (yêu cầu ARR double-blind, gộp chung với mục Anonymize bên dưới) và thêm đoạn liên hệ special theme COLING 2027.
- [x] **Related Work — phân biệt với *Lost in Compression*:** họ chứng minh *dữ liệu giám sát tiếng Anh không chuyển ngữ*; ta chứng minh *tín hiệu thanh điệu/hình vị không giúp QA tiếng Việt* + cung cấp benchmark. Khác trục → đồng minh, không trùng lặp. Đã viết vào `paper/lacc_paper.tex` §2.1 (2 đoạn mới + bibitem `lukauskas2026lost`) — tra cứu citation thật qua WebFetch/WebSearch (arXiv:2608.26175, Mantas Lukauskas, "Lost in Compression: A Controlled Cross-Lingual Audit of Extractive Prompt Compressors", 27/7/2026), không bịa tên tác giả. Cũng thêm Bảng `tab:baselines` mới ở §4.3 liệt kê đủ 13 arm (khớp `benchmark.py:WAVE2_ARMS`) + cập nhật §4.4 Độ đo với 3 metric evidence/attribution (G0) và achieved-rate 2 tokenizer.
- [x] **Định vị E6 query-agnostic đúng cách:** nói rõ trục A (query) và B (cơ chế) *độc lập*; E6 nâng cơ chế trong khung query-agnostic. So E6 với **baseline task-agnostic** (LLMLingua-2, Selective Context, LLMLingua) — **KHÔNG** so với LongLLMLingua (khác lớp ngân sách). Viết thành mục riêng "Định vị E6: trục cơ chế, không phải trục truy vấn" trong Thảo luận (`\label{sec:e6_positioning}`), tham chiếu từ §3.7 (Kiến trúc thay thế Encoder).
- [x] **Limitations — viết thẳng ceiling của E6:** query-agnostic không đòi thắng baseline query-aware trên QA; báo phần *hiệu quả* (latency, kích thước model) nơi nó thật sự thắng. Thêm item (4) trong mục Hạn chế.
- [x] **Cập nhật `tab:slm_pilot`** bằng số wave 2 (**94.85%** tone acc, không phải 16.59%) — kèm chú thích: đây là bằng chứng *pipeline chạy được*, **không phải** bằng chứng nén tốt hơn. Giữ nguyên bảng pilot 30-bước/GTX 1060 gốc (vẫn là bằng chứng khả thi phần cứng hợp lệ), thêm đoạn + bảng mới ngay sau nêu số wave-2 (3 epoch, full corpus, H100 — nguồn `results/report/wave_2/WAVE2_TRAINING_REPORT.md`), có đối chứng majority-class (43.48%) và token-id-lookup (100%, trần lý thuyết) để không overclaim.
- [x] Diễn đạt finding **chính xác, không overclaim** (dân ngôn ngữ học đọc): "dấu thanh không tương quan với câu chứa đáp án *cho tác vụ QA*" — không phủ nhận giá trị ngôn ngữ học của thanh điệu. Viết lại toàn bộ §6.1 (đổi tên thành "Một đặc trưng ngôn ngữ học hợp lý không tự động là tín hiệu hữu ích: trường hợp thanh điệu") — trước đó mục này vẫn khẳng định tone giúp ích (mâu thuẫn trực tiếp với Abstract/Giới thiệu vừa sửa); giờ giải thích 3 lý do (không biết câu hỏi/đơn vị token không phải câu/cộng làm loãng — khớp E1/E5/E2), và có đoạn "Điều phát hiện này KHÔNG nói" tường minh để không bị đọc thành "thanh điệu vô giá trị".

**Checklist ARR/COLING (làm trước khi nộp):**
- [x] Đọc yêu cầu format ARR + **Responsible NLP checklist** (data/repro) — leak-control của bạn là điểm cộng ở đây. Tra cứu thật qua WebFetch (2027.coling-iccl.org/calls/main_conference_papers, aclrollingreview.org/responsibleNLPresearch):
  - Hạn: **12/10/2026** (chu kỳ ARR October 2026) — khớp tasklist.
  - Page limit: long paper 8 trang, short 4 trang; reference/appendix không giới hạn.
  - Bắt buộc: mục **Limitations**; **Ethics** là tùy chọn.
  - Checklist ARR có 5 phần: **A** (mọi bài — limitations/risks), **B** (scientific artifacts: cite nguồn, license, PII/nội dung offensive, thống kê split), **C** (computational experiments: số tham số, compute budget, hyperparameter, descriptive stats), **D** (human annotators — có thể N/A nếu VCC-Bench không dùng annotator trả phí), **E** (**bắt buộc khai báo dùng AI assistant** — dự án này dùng Claude Code rất nhiều, **phải khai ở E1**, không khai là vi phạm chính sách ARR).
  - Từ 12/2024: vi phạm checklist lộ liễu (trả lời "yes" hết mà không giải thích) có thể bị desk-reject.
- [x] **Anonymize** bài (ARR double-blind — giấu tên tác giả/repo) — đã ẩn `\author{}` thật thành "Anonymous ACL submission", bỏ `\date{}`. **Rà lần cuối (sau khi viết xong §Công bố dữ liệu và mã nguồn):** grep toàn file cho `thanthien`/`anhalu`/`github.com/`/`huggingface.co/<org>`/`\thanks`/self-citation — sạch, không có rò định danh nào trong `paper/lacc_paper.tex`.
- [x] Viết **Limitations section** (ARR bắt buộc) — 4 mục hạn chế kỹ thuật (thêm mục E6 ceiling) + mục **Rủi ro** riêng (A2: khái quát hoá quá mức, ứng dụng rủi ro cao, contamination benchmark công khai). **Còn lại khi có số G1:** không cần thêm mục mới, chỉ cần đối chiếu §6.1 (đã viết theo hướng "sẽ được điền số ở §5") khớp với số thật.
- [x] Kiểm tra **special theme COLING 2027** — nếu bài khớp theme thì nêu, được cộng điểm. Theme thật (tra WebFetch): **"NLP for Linguistics"** — dùng NLP/LLM cho nghiên cứu ngôn ngữ học, đặc biệt ngôn ngữ đa dạng/low-resource. Bài này khớp trực tiếp (dùng benchmark có kiểm soát để trả lời câu hỏi ngôn ngữ học: thanh điệu có mang thông tin chức năng cho QA không) — đã thêm đoạn liên hệ vào cuối §1.
- [x] Chuẩn bị **data/code release** (link ẩn danh) cho VCC-Bench — resource paper phải có. Thêm mục §"Công bố dữ liệu và mã nguồn" mới trong `paper/lacc_paper.tex`: giấy phép (CC-BY-SA 4.0 Wikipedia + Public Domain pháp luật + MIT synthetic, đúng metadata thật của `vcc_bench_v2.json`), xác nhận **không có PII thật** (rà soát trực tiếp file dữ liệu: `synthetic_pii=true` trên 120/120 mẫu needle, kiểm tra vài payload thật — lưu ý trung thực: giá trị sinh chương trình chứ không phải "rõ ràng giả" như bản rebuild spec kỳ vọng ban đầu, ví dụ số điện thoại/tài khoản nhìn hợp lệ về mặt cú pháp), placeholder `[LIÊN KẾT ẨN DANH -- BỔ SUNG KHI NỘP]`. **Còn lại:** đây mới là bản nháp text trong paper — chưa tạo mirror ẩn danh thật (GitHub/HF ẩn danh) cho lúc nộp, việc đó cần làm ở bước nộp thật, không phải bây giờ.

- [ ] **NỘP ARR — 12/10/2026**

---

## ⏸️ Gác lại (không làm cho vòng này)

- [ ] ~~E6 query-aware~~ — đã chốt query-agnostic; không xây lại
- [ ] ~~Bolt keyword-multiplier vào E6~~ — làm bẩn control, để query cho E4 lo
- [ ] **Regenerate `compression.jsonl`** (prompt v3) — *chỉ cần nếu* muốn báo số E5 train trên gold trích xuất. E6 self-distill đã né được, E5 chạy tạm bằng token-score tổng hợp theo câu → **để Future Work nếu gấp**
- [ ] **Held-out eval cho E6** (validation split) — tốt nếu kịp, nhưng benchmark.py trên VCC-Bench đã cho số eval thật rồi
- [ ] **Distillation confidence-weighted (paper 14–17)** — chưa đọc, chưa kiểm chứng → Future Work

---

## 🎯 Definition of Done tối thiểu (paper nộp được cần)

1. VCC-Bench v2 chạy xong, §5 có số (không còn `---`)
2. Có baseline dịch-rồi-nén + LLMLingua-2 trong bảng
3. §3 khớp code thật; finding phát biểu chính xác
4. Limitations + anonymize + data release link
5. Related Work phân biệt rõ với *Lost in Compression*

> Nếu chỉ kịp 5 điều trên → vẫn là một paper nộp được. Mọi thứ khác là *nâng cấp tùy chọn*.
