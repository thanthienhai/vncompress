# Rà soát phương pháp LACC cho paper + dữ liệu wave 2 — 2026-09-18

Tổng hợp cho việc viết paper: đối chiếu `paper/lacc_paper.tex` (bản thảo hiện tại)
với `research/wave2_proposals.md` (chẩn đoán sau wave 1), code thực tế trong
`vncompress/`, và những gì wave 2 đã chạy được (`results/report/wave_2/WAVE2_TRAINING_REPORT.md`,
`WAVE2_HANDOFF.md`, `WAVE2_DATA_NOTES.md`). Nguồn: đọc trực tiếp các file này,
không suy đoán số liệu.

## 0. Kết luận ngắn

**Bản thảo paper hiện tại mô tả phương pháp wave-1** (tone + morphology +
perplexity, blend cộng, không biết câu hỏi). Nhưng chính nghiên cứu nội bộ
(`wave2_proposals.md`, viết sau khi phân tích số liệu wave 1) đã kết luận trục
đó **thua LLMLingua và bị chính dữ liệu bác bỏ** ("giữ dấu ≠ giữ thông tin"),
và đề ra 11 thay đổi (E1–E11). Tính đến hôm nay, **6/8 thay đổi phần code (E1,
E2, E3, E5, E7, E8) đã có trong `vncompress/compression.py`**, và 2 pipeline cần
GPU (E4 relevance probe, E6 encoder classifier) **đã train xong ở wave 2** —
nhưng **paper chưa được sửa để khớp với bất kỳ thay đổi nào ở trên**. Phần
"Phương pháp đề xuất" (§3) trong `lacc_paper.tex` vẫn là công thức wave-1 gốc,
không có query-conditioning, không nhắc tới relevance probe/encoder classifier/
sentence-level, và mọi bảng kết quả ở §5 vẫn để trống (`---`).

Nói cách khác: **code đã đi trước paper một bước lớn, nhưng chưa có run
benchmark nào (VCC-Bench v2) để paper có số liệu điền vào.**

---

## 1. Vì sao trục "tone-aware" của paper có vấn đề

`wave2_proposals.md` §1 (chẩn đoán gốc rễ, dựa trên số liệu wave 1) chỉ ra 4
nguyên nhân LACC thua LLMLingua, đều **không liên quan gì đến đặc thù tiếng
Việt**:

| # | Nguyên nhân | Bằng chứng wave 1 |
|---|---|---|
| A | Chấm điểm không biết câu hỏi (query-agnostic) | 53/300 mẫu QA: no-compress >0.9, LLMLingua >0.5, LACC-tone → 0 |
| B | Cắt ở mức token, không mức câu | Needle mất câu chứa mã kích hoạt dù giữ dấu thanh |
| C | Blend cộng làm loãng tín hiệu mạnh | ppl-only 0.253 > combined 0.118; calibration tự đặt $w_{\text{tone}}=0$ |
| D | Scorer 4B thua scorer 0.5B off-the-shelf | 0.239 < 0.442 |

Kết luận cứng nhất: **"dấu thanh không tương quan với câu nào chứa đáp
án."** — tức là chính centerpiece của paper (Tone-Aware Scoring, Phonological
Consistency Loss) **không phải là tín hiệu giúp nén tốt hơn cho tác vụ QA**,
dù nó vẫn đúng về mặt kỹ thuật ngôn ngữ học.

`wave2_proposals.md` §0 còn phát hiện thêm hai vấn đề paper-vs-code cụ thể
(đọc code trực tiếp, không có trong báo cáo wave 1):

1. **Toàn bộ 3 tín hiệu LACC được tính chỉ trên context, không nhìn câu hỏi**
   — câu hỏi chỉ vào qua một hệ số nhân thô theo trùng khớp từ khoá, áp *sau*
   khi đã chấm điểm xong. Không có trong công thức §3.1 (eq. `combined`) của
   paper.
2. **"Iterative budget allocation" / "class-proportional budget" mà paper §3.2.3
   mô tả (eq. `budget`) không tồn tại trong code tại thời điểm viết đề xuất**
   (2026-09-05). Điều này đã được sửa ở wave 2 (xem §2 dưới) nhưng cần xác nhận
   text công thức khớp với hàm `_select_class_proportional` thật.

---

## 2. Đối chiếu: paper §3 nói gì vs. code có gì (sau wave 2)

`vncompress/compression.py` (`LACCCompressor.__init__`, dòng ~1092) đã có đủ cờ
cho toàn bộ đề xuất P0/P1/P2, nhưng **không cái nào xuất hiện trong bản thảo
paper**:

| Ký hiệu code | Ứng với đề xuất | Có trong paper §3 không? |
|---|---|---|
| `contrastive_perplexity()` (dòng 419), dùng khi có `question` | E1 — perplexity query-conditioned (LongLLMLingua-style) | ❌ eq. `ppl` (§3.3) chỉ có $-\log P(t_i \mid t_{<i})$, không điều kiện câu hỏi |
| `morph_combine='multiply'` khi ghép ppl+morph | E2 — nhân thay vì cộng để tránh loãng tín hiệu (nguyên nhân C) | ❌ eq. `combined` (§3.1) là tổng có trọng số, không phải tích |
| `DEFAULT_PPL_WINDOW=2048`, overlap 256 | E3 — sửa chi phí tự-hại | Không nhắc trong paper (paper chỉ nói "cửa sổ 512 token" ở Thuật toán 1, dòng "mỗi cửa sổ 512 token" — **đây là số cũ, đã lỗi thời**, thực tế đã đổi sang 2048) |
| `selection_unit='sentence'` | E5 — chọn theo câu thay vì token lẻ | ❌ Thuật toán 1 chỉ có TopK ở mức token |
| `budget_mode='class_proportional'`, `_select_class_proportional()` | E7 — hạn ngạch theo lớp từ (đúng như paper *hứa*) | ⚠️ Có công thức (eq. `budget`) nhưng cần đối chiếu lại text với implementation thật — đề xuất gốc nói công thức này *không tồn tại* lúc viết; nay tồn tại nhưng chưa kiểm chứng khớp 1-1 |
| `tone_task_gate=True` | E8 — tone chỉ bật cho task bề mặt (dịch, trích dẫn nguyên văn), tắt cho QA/needle | ❌ paper coi tone là tín hiệu toàn cục, không route theo task |
| `encoder_compression.py`, `scripts/train_encoder_compressor.py` | E6 — encoder phân loại giữ/bỏ (LLMLingua-2 style, PhoBERT) thay scorer generative | ❌ paper §3.3–3.4 chỉ nói tới mô hình nhỏ *sinh* perplexity (135M–1.5B causal LM), không nhắc kiến trúc encoder-classification |
| `linguistics.RelevanceConsistencyLoss` | E4 — đổi mục tiêu probe từ "thanh điệu" sang "liên-quan-câu-hỏi" | ❌ paper §3.4 hoàn toàn dành cho *Phonological* Consistency Loss (dự đoán thanh điệu); không có một dòng nào về relevance probe |

**Việc cần làm rõ trước khi viết tiếp paper:** đây không phải danh sách "thêm
tính năng" — mà là **paper đang mô tả một phương pháp mà chính nhóm nghiên cứu
đã tự chứng minh là thua baseline**, trong khi phiên bản đã sửa (có
query-conditioning, sentence-level, relevance probe) đã tồn tại trong code
nhưng chưa hề được viết vào bản thảo.

---

## 3. Ba lựa chọn định vị lại paper (từ `wave2_proposals.md` §3)

Nhóm nghiên cứu tự đề xuất 3 trục thay thế cho "tone-aware compression tiếng
Việt" (trục hiện tại của `lacc_paper.tex`), xếp theo độ chắc chắn:

1. **Benchmark + negative result** (chắc nhất, gần xong) — giữ VCC-Bench (đóng
   góp C3) làm trục chính, báo cáo trung thực kết quả âm đã kiểm soát chặt:
   "giữ dấu thanh ≠ giữ thông tin cần cho QA; TPR nghịch biến với chất lượng
   sinh văn bản." Đây là hướng ít rủi ro nhất, không cần thêm thực nghiệm lớn.
2. **Query-aware Vietnamese compression** (nếu E1+E4+E5 cho kết quả dương) —
   xoay trục sang "query-conditioned, morphology-informed", dùng lại đúng hạ
   tầng train wave 1 nhưng cho mục tiêu liên-quan thay vì thanh điệu.
3. **Efficiency angle** (nếu E6+E3 dương) — "encoder-classifier rẻ hơn và tốt
   hơn scorer generative cho tiếng Việt", hẹp nhưng sạch.

**Quan trọng:** đề xuất cảnh báo rõ *"không nên"* (a) đắp thêm tín hiệu tiếng
Việt khác (teencode, dialect, wordnet…) vào blend cộng — dữ liệu cho thấy thêm
tín hiệu vào blend làm loãng chứ không tốt lên; (b) chạy lại benchmark chỉ để
"làm đẹp" số LACC-tone — phép đo wave 1 đã đúng; (c) đầu tư thêm scorer
generative lớn hơn.

---

## 4. Wave 2 đã chạy gì (2026-09-17, cluster `pod-test`, 1× H100)

Nguồn: `results/report/wave_2/WAVE2_TRAINING_REPORT.md`. Ba pipeline train chạy
trên corpus `vncompress-vi-v2` đã checkout trong repo — **chưa có run
`benchmark.py` nào đối chiếu lại VCC-Bench v2** trong báo cáo này (chỉ có
training + validation nội bộ của từng stage).

| Stage | Việc | Kết quả chính |
|---|---|---|
| `slm` (tone-aware LoRA) | 3 epoch, full corpus (56,729 văn bản), LoRA 0.94% tham số | Tone accuracy (token có thanh) **94.85%**, macro-F1 0.9482, val PPL 86.36. Mạnh hơn hẳn con số pilot 16.59% đang nằm trong paper (Bảng `tab:slm_pilot`, §5.1) — **paper cần cập nhật bảng này bằng số wave 2**, có kèm chú thích rằng đây vẫn chỉ là *tone accuracy*, không phải bằng chứng cải thiện chất lượng nén (theo kết luận §1 ở trên) |
| `probe` (E4 relevance probe) | 3 epoch, `qa.jsonl` (5,482 mẫu, 131 tài liệu) | F1 **0.103** (precision 5.6%, recall 63.4%) — yếu, do mất cân bằng lớp nặng (44:1). Báo cáo tự đánh giá: "probe học được *một cái gì đó*" chứ chưa production-ready. **Chưa có A/B đúng khuôn `verify_tone_probe_e2e.py` (probe-relevance vs probe-tone vs rule) mà `wave2_proposals.md` §2/E4 yêu cầu để xác nhận "đảo dấu" kết quả wave 1** — đây là bước còn thiếu quan trọng nhất trước khi paper có thể tuyên bố E4 thành công |
| `encoder` (E6 encoder classifier) | 2 epoch, full corpus, ratio=4, teacher Qwen2.5-0.5B-Instruct (self-distill trực tiếp trên corpus, không cần `compression.jsonl`) | Chỉ có loss curve (0.507 → ~0.29); **không có held-out eval** vì script train chưa có tùy chọn validation split. Paper Bảng `tab:compression_efficiency`/`tab:quality` không có số nào để điền cho arm này |

### Ba lỗi đã sửa trong code (không đổi thuật toán, chỉ đúng đắn/hiệu năng)

1. Cache tra thanh điệu theo token id (O(vocab) thay vì O(total tokens)) —
   dataset build từ "treo nhiều phút" xuống vài giây.
2. `return_offsets_mapping` với PhoBERT (tokenizer chậm, không hỗ trợ fast) —
   thêm fallback decode-and-find dùng chung giữa train và inference.
3. Batch hoá teacher scoring cho E6 labeling (`sliding_window_perplexity_batch`)
   — full-corpus labeling từ dự kiến hàng giờ xuống ~9 phút, GPU util 30%→96-99%.

---

## 5. Trạng thái dữ liệu wave 2 (`WAVE2_DATA_NOTES.md`, `docs/dataset_v2_review.md`)

| Nguồn | Dùng được cho gì | Trạng thái |
|---|---|---|
| `qa.jsonl` (6,000 hàng, `answer_span` xác minh 100%) | E4 relevance probe | ✅ dùng ngay. **Rò dữ liệu đã được chặn đúng cách**: 32/414 mẫu VCC-Bench v2 (bài *Hà Nội*) trùng với `qa.jsonl` split train; `train_relevance_probe.py` holdout các tài liệu này **theo mặc định** (`resolve_holdout_documents`, không cần cờ) — đã xác minh `run_pipeline.sh` không truyền `--no-holdout`, nên wave 2 chạy đúng, không bị nhiễm chéo train/test |
| `compression.jsonl` | E5 (chọn câu), E6 (nếu dùng gold thay vì self-distill) | ❌ **Chưa dùng được**: gold là văn bản model *viết lại*, chỉ 30.0% câu xuất hiện nguyên văn trong context (median 0.038/hàng), 21.2% hàng khẳng định số liệu không có trong context. Prompt v3 (trích xuất + gate cứng) đã sửa trong code nhưng **dữ liệu chưa được sinh lại** — E6 wave 2 né vấn đề này bằng cách tự distill nhãn giữ/bỏ trực tiếp từ corpus + teacher, không đụng tới `compression.jsonl` |
| `eval/test.jsonl` (1,000 dòng) | sanity-check nhanh | ⚠️ Chỉ trải trên **14 tài liệu thực** (4 bài chiếm 534/1000 dòng) — n hiệu dụng là 14. **Không được báo cáo CI/confidence interval tính trên tập này**; kết quả chính phải luôn là VCC-Bench v2 |
| `data/benchmark/vcc_bench_v2.json` (414 mẫu) | eval chính (paper §5) | Chưa có run `benchmark.py` nào trong wave 2 — đây là việc còn thiếu lớn nhất để lấp các bảng trống ở §5 paper |

---

## 6. Việc cần làm trước khi paper có thể viết tiếp (ưu tiên)

1. **Chạy `benchmark.py` trên VCC-Bench v2** với đúng danh sách arm wave 2
   (`none,random,llmlingua,llmlingua_contrastive,lacc_ppl_contrastive,lacc_ppl_morph,
   lacc_cx_morph,lacc_sentence,lacc_classprop,lacc_tone_gated,encoder`) — không
   có bước này thì mọi bảng ở §5 paper (`tab:compression_efficiency`,
   `tab:quality`, `tab:tone_preservation`, `tab:ablation`) không có số để điền.
2. **Chạy A/B relevance-probe vs tone-probe vs rule** (`verify_tone_probe_e2e.py`)
   — là phép đo mà `wave2_proposals.md` coi là "headline mới" của E4; hiện mới
   chỉ có F1 train/val (0.103), chưa có so sánh trực tiếp với cách cũ.
3. **Viết lại §3 (Phương pháp)** để khớp với code thật: thêm query-conditioning
   (E1, eq. mới cho contrastive perplexity), đổi công thức blend sang tích nếu
   dùng `morph_combine='multiply'` (E2), thêm mục sentence-level selection (E5),
   thêm mục encoder classifier (E6) như một nhánh kiến trúc thay thế cho mô
   hình nhỏ causal-LM, sửa số cửa sổ trong Thuật toán 1 (512 → 2048), và quyết
   định giữ/thay Phonological Consistency Loss bằng Relevance Consistency Loss
   (hoặc trình bày cả hai, có A/B).
4. **Quyết định trục câu chuyện** theo §3 tài liệu này (benchmark+negative-result
   / query-aware / efficiency) trước khi viết Discussion — tránh viết Discussion
   §6.1 hiện tại ("Tại sao tín hiệu ngôn ngữ quan trọng?") vốn đang giả định kết
   luận mà chính dữ liệu nội bộ đã bác bỏ.
5. **Cập nhật Bảng `tab:slm_pilot`** (§5.1) bằng số wave 2 (94.85% tone acc,
   không phải 16.59%) — nhưng phải đi kèm khung diễn giải đúng: đây là bằng
   chứng pipeline train hoạt động, **không phải** bằng chứng cải thiện chất
   lượng nén (đã tách hai claim này ra theo đúng tinh thần thận trọng mà chính
   đoạn text hiện tại của §5.1 đang giữ).
6. **Regenerate `compression.jsonl` với prompt v3** trước khi dùng nó để train/
   eval E5 (sentence-level) bằng gold thật — hiện `lacc_sentence` trong code có
   thể chạy (dùng token scores tổng hợp theo câu), nhưng nếu paper muốn báo cáo
   số liệu train trên gold trích xuất, dữ liệu chưa sẵn sàng.
7. **Thêm baseline LongLLMLingua/LLMLingua-2** (E11) vào bảng so sánh §4.3 —
   hiện paper chỉ có LLMLingua gốc + SnapKV, thiếu đúng hai baseline
   query-aware/encoder mạnh nhất mà reviewer sẽ hỏi.

---

## 7. Rủi ro/điểm mở cần quyết định (không tự suy đoán câu trả lời)

- **E4 F1 = 0.103 có đủ để làm headline "đảo dấu kết quả wave 1" không?**
  Cần A/B thật (mục 6.2) trước khi paper tuyên bố bất cứ điều gì về relevance
  probe.
- **E6 chưa có held-out eval** — cần thêm validation split vào
  `train_encoder_compressor.py` hoặc chạy `benchmark.py --methods encoder`
  trực tiếp trên VCC-Bench v2 để có số ROUGE/BLEU/latency thay vì chỉ loss.
- **`compression.jsonl` v3 (trích xuất) chưa được sinh** — nếu timeline paper
  gấp, cần quyết định có đợi hay dùng đường tắt self-distillation (như E6 đã
  làm) cho cả E5.
- **Trục câu chuyện paper chưa chốt** (mục 3) — ảnh hưởng trực tiếp đến việc
  nên viết tiếp Discussion/Limitations theo hướng nào; nên chốt trước khi đầu
  tư viết thêm text.
