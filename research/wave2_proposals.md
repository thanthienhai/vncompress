# Đề xuất cải tiến sau Wave 1 — hướng cho Wave 2

**Ngày:** 2026-09-05 · **Đầu vào:** [`../results/report/wave_1/2026-09-05_lacc-technical-report.md`](../results/report/wave_1/2026-09-05_lacc-technical-report.md) · **Trạng thái:** đề xuất, chưa sửa code

Tài liệu này **không** lặp lại danh sách việc-cần-làm ở §13 của báo cáo wave 1
(commit, sửa abstract, bootstrap CI, run λ=0, window 2048…). Những việc đó đã
đúng và nên làm. Đây là lớp **cải tiến phương pháp** mà số liệu wave 1 đang chỉ
tới nhưng báo cáo chưa nói tường minh, cộng với ba phát hiện mới rút ra khi đọc
lại code hiện tại (`vncompress/compression.py` sau refactor) đối chiếu với những
gì paper tuyên bố.

---

## 0. Ba phát hiện mới (đọc code, không có trong báo cáo)

Ba điều này thay đổi cách diễn giải kết quả, nên đặt lên đầu:

1. **Toàn bộ scoring của LACC là query-agnostic.** Ba tín hiệu (perplexity, tone,
   morphology) được tính **chỉ trên context**, không hề nhìn câu hỏi
   (`compression.py:948-992`). Câu hỏi chỉ vào pipeline ở **một** chỗ: một hệ số
   nhân theo trùng khớp từ khoá thô (`compute_query_relevance_weights`,
   `:670-684`), áp **sau khi** đã chấm điểm xong (`:1000-1002`). Không match mềm,
   không embedding, không contrastive. → Cơ chế thất bại "giữ 93.2% dấu nhưng
   xoá mất câu chứa đáp án" (§9 báo cáo) **chính là hệ quả trực tiếp của việc
   chấm điểm mà không biết câu hỏi.**

2. **Baseline LLMLingua trong repo cũng là query-agnostic** — là LLMLingua *gốc*
   (perplexity thuần theo cửa sổ, `sliding_window_perplexity` `:244-288`), **không
   phải LongLLMLingua** (contrastive perplexity có điều kiện câu hỏi). Nghĩa là:
   phương pháp đang thắng mọi biến thể LACC vẫn chưa hề dùng câu hỏi. Đây là tin
   tốt — **dư địa lớn nhất còn nguyên và chưa ai chạm tới.**

3. **"Iterative budget allocation" và "class-proportional budget" mà paper mô tả
   KHÔNG tồn tại trong code.** Chọn token thực tế là global top-k trong vùng giữa
   có bảo vệ biên (`select_with_boundary` `:167-182`), đồng nhất cho mọi method.
   Morphology chỉ vào dưới dạng **hệ số nhân per-token**, không phải hạn ngạch
   theo lớp từ. → Hoặc phải sửa paper cho khớp code, hoặc thực sự implement điều
   paper hứa (xem E7). Đây là lỗi reviewer dễ bắt thứ hai sau abstract.

---

## 1. Chẩn đoán gốc rễ: tại sao LACC thua

Xâu chuỗi lại: LACC thua LLMLingua **không phải vì tiếng Việt**, mà vì bốn lựa
chọn thiết kế độc lập với ngôn ngữ, cái nào cũng sửa được:

| # | Nguyên nhân | Bằng chứng wave 1 | Vị trí code |
|---|---|---|---|
| A | **Chấm điểm không biết câu hỏi** | 53/300 mẫu QA: no-compress >0.9, LLMLingua >0.5, tone → 0 (§9) | `:948-992` |
| B | **Cắt ở mức token, không mức câu/mệnh đề** | Needle mất câu chứa mã kích hoạt dù giữ dấu | `select_with_boundary :167` |
| C | **Blend cộng làm loãng tín hiệu mạnh** | ppl-only 0.253 > combined 0.118 (§8.3); calibration đặt w_tone=0 (§8.4) | `:994-998` |
| D | **Scorer sai loại và sai kích thước** | 4B đã train QA 0.239 < 0.5B off-the-shelf 0.442 (§8.6) | `LACCScorer :715` |

Tín hiệu tone thất bại vì một lý do sâu hơn, đã được A/B chứng minh: **dấu thanh
không tương quan với "câu nào chứa đáp án".** Giữ dấu là mục tiêu sai. Việc cần
giữ là **token liên quan tới câu hỏi** — và đó chính xác là thứ query-conditioning
đo được.

---

## 2. Đề xuất — xếp theo ưu tiên

Mỗi mục: **giả thuyết → cơ sở tài liệu → thay đổi ở đâu → cách đo → chi phí →
rủi ro.** Đánh dấu 🟢 không cần GPU / 🟡 GPU rẻ / 🔴 GPU đáng kể.

---

### P0 — Rẻ, có thể lật ngược bức tranh (làm trước)

#### E1. 🟡 Query-conditioned perplexity (contrastive) — *đề xuất số 1*

**Giả thuyết:** Phần lớn khoảng cách LACC↔LLMLingua là do không dùng câu hỏi.
Chuyển từ perplexity thuần sang **contrastive perplexity** kiểu LongLLMLingua sẽ
lấy lại phần lớn khoảng cách đó, và có thể **vượt** LLMLingua gốc trên needle/QA.

**Cơ sở:** LongLLMLingua (ACL 2024) — "question-aware coarse-grained + contrastive
perplexity": token quan trọng = token mà phân phối bị dịch chuyển mạnh khi *thêm
điều kiện câu hỏi*. `importance(t) = PPL(t | context) − PPL(t | question, context)`.
Cải thiện tới +21.4% ở ~4× trên NaturalQuestions. QCFuse (2026, đã có trong
`compression_landscape_notes.md`) cũng là query-aware selector cho RAG.

**Thay đổi ở đâu:** `sliding_window_perplexity` (`:244-288`) hiện là
`f(model, input_ids)`. Thêm biến thể có điều kiện: chạy hai lượt (có/không prepend
question tokens) và lấy hiệu, hoặc rẻ hơn — chỉ tính reweight `r_k` theo
LongLLMLingua. Áp cho cả `LLMLinguaCompressor._compute_token_importance` và
nhánh perplexity của LACC (`:949-952`). *Chưa cần train lại gì.*

**Cách đo:** thêm arm `llmlingua_contrastive` và `lacc_ppl_contrastive`; so trên
VCC-Bench QA + needle ở 2×/4×/8× với paired bootstrap. Kỳ vọng dương rõ nhất ở
needle (task định vị thông tin).

**Chi phí:** ~gấp đôi số forward pass (nhưng xem E3 để bù). Không train.
**Rủi ro:** thấp; đây là kỹ thuật đã được chứng minh, chỉ là port.

---

#### E2. 🟢 Hợp nhất perplexity + morphology (tín hiệu tốt của hai method) — *báo cáo đã gợi ý §13*

**Giả thuyết:** Hai tín hiệu đứng được nằm ở hai method khác nhau — perplexity
(LLMLingua 0.442) và morphology (Morph thắng random 65–182%) — **chưa ai đo cùng
nhau.** Perplexity chọn token bất ngờ/nhiều thông tin; morphology hạ ưu tiên hư
từ ở 0 GB. Ghép lại có thể cộng dồn.

**Thay đổi ở đâu:** đây là thay đổi nhỏ trong `LACCCompressor` — đặt
`use_tone=False`, `weights = ScoreWeights(perplexity=…, morphology=…, tone=0)`
(`:692`). Nhưng **quan trọng:** đừng dùng blend cộng đã bị chứng minh làm loãng
(nguyên nhân C). Thử hai cách kết hợp:
- (a) morphology làm **hệ số nhân** lên điểm perplexity thay vì cộng có trọng số;
- (b) morphology làm **hạn ngạch budget theo lớp từ** (nối sang E7).

**Cách đo:** arm `lacc_ppl_morph` vs `llmlingua` vs `morph_only`. Cần thắng
LLMLingua để biện minh việc thêm morphology.
**Chi phí:** ~0 (morphology 0 GB). **Rủi ro:** có thể chỉ ngang perplexity đơn lẻ
— nếu vậy, kết luận "morphology không cộng thêm khi đã có ppl" cũng là kết quả
sạch, đáng báo cáo.

---

#### E3. 🟢 Sửa cấu hình cửa sổ tự-hại (window 2048, overlap 256) — *báo cáo §6.2*

**Giả thuyết:** LACC đang chậm 55–113× **không phải do thuật toán** mà do cấu
hình: window 512 / stride 256 → mỗi token qua model ~2 lần × 46 cửa sổ @12k.
Đổi sang 2048/overlap-256 (đúng như LLMLingua) giảm 46→7 lượt.

**Thay đổi ở đâu:** default `window_size`, `stride` trong `sliding_window_perplexity`
(`:259`) và cấu hình scorer. Không đổi logic.
**Vì sao P0:** claim chi phí ("0 GB VRAM") hiện **không đo được công bằng** — LACC
nén tốn 3.2s trong khi generator đọc full context chỉ 0.8s. Không sửa cái này thì
không có luận điểm hiệu năng nào đứng được, kể cả khi chất lượng tốt lên.
**Chi phí:** ~0. **Rủi ro:** window lớn hơn có thể làm mượt điểm perplexity — đo
lại chất lượng sau khi đổi (test `test_llmlingua_windowing.py` đã canh invariant).

---

### P1 — Thay đổi phương pháp, tạo kết quả *dương* mới cho paper

#### E4. 🔴 Chuyển mục tiêu probe: từ "dự đoán thanh điệu" sang "dự đoán liên quan-câu-hỏi" (Sentinel-style) — *đề xuất tạo đột phá*

**Đây là đề xuất cứu C2 và biến "phương pháp train hoạt động" thành đóng góp có ích.**

**Giả thuyết:** Hạ tầng train của wave 1 (LoRA + probe trên hidden state) *hoạt
động* — PPL giảm 34–39%, probe học đặc hiệu (+0.0625 real vs −0.0147 control).
Vấn đề là probe đang học **nhãn sai** (thanh điệu, vốn là hàm tất định của token
id → vô dụng cho việc chọn token). Nếu **giữ nguyên toàn bộ máy móc** nhưng đổi
nhãn probe thành **"token này có liên quan tới câu hỏi không"**, ta có một scorer
liên-quan học được, chạy trên chính hạ tầng đã có.

**Cơ sở:** *Sentinel* (2505.23277, 2025) — dùng SLM 0.5B off-the-shelf, **probe
tín hiệu attention** (query-aware qua QA prompt template), train một logistic
regression nhẹ trên ~6k mẫu weak-supervision (SQuAD/HotpotQA) để dự đoán độ liên
quan **mức câu**. LongBench F1 47.89 vs LLMLingua-2 39.1, nhanh hơn 1.13×. Điểm
mấu chốt: *"learning explicit relevance outperforms heuristic scoring"* — đúng
thứ LACC đang thiếu. EXIT/RECOMP cũng đặt nén = phân loại nhị phân liên-quan có
giám sát bởi generator.

**Thay đổi ở đâu:** `PhonologicalConsistencyLoss` (`linguistics.py:1172`) và trainer
đổi target: thay vì 7 lớp thanh điệu, probe dự đoán nhãn liên-quan (nhị phân/hồi
quy). Nhãn train sinh **weak-supervision**: token/câu chứa reference answer = 1
(VCC-Bench đã có `reference_answer` và needle dựng chính xác — nguồn nhãn sẵn).
Ở inference, `score_importance` (`linguistics.py:1218`) trả điểm liên-quan thay
vì độ tự tin thanh điệu.

**Cách đo:** A/B đúng khuôn wave 1 (`verify_tone_probe_e2e.py`): probe-liên-quan
vs probe-tone vs rule, cùng SLM. Kỳ vọng probe-liên-quan **đảo dấu** kết quả A/B
(wave 1: probe-tone thua rule). Đây là headline mới, dương, và tái dùng đúng
đóng góp "train được".

**Chi phí:** một chu kỳ train probe (rẻ, base đóng băng — job 3 trong
`experiments.md` chỉ ~45 phút loại) + một run A/B.
**Rủi ro:** trung bình. Nhãn weak-supervision có nhiễu; cần kiểm tra probe không
chỉ học "token là danh từ riêng". Nhưng kể cả kết quả trung tính vẫn có ý nghĩa
khoa học hơn probe-tone.

---

#### E5. 🔴 Nén mức câu/mệnh đề (extractive) thay vì mức token — cho task định vị

**Giả thuyết:** Needle và QA fail vì **mất nguyên câu chứa đáp án**. Cắt token lẻ
phá vỡ câu; giữ/bỏ theo **đơn vị câu** bảo toàn được "đường trả lời". Báo cáo cho
thấy LLMLingua đã có một bước lọc câu thô (`_sentence_level_filter :355-391`) và
điều đó có thể là lý do nó thắng — LACC không có bước này.

**Cơ sở:** EXIT (ACL 2025 findings), RECOMP, CORE (2606.20571, edge, sentence-level,
+30.19% accuracy), CPC (sentence encoder xếp hạng theo query). Xu hướng 2025 rõ:
**query-aware + sentence-level** cho RAG/QA.

**Thay đổi ở đâu:** thêm chế độ chọn theo câu vào `LACCCompressor` — chấm điểm câu
= tổng hợp điểm token (đã query-conditioned từ E1) rồi chọn câu theo budget, thay
vì `select_with_boundary` mức token. Giữ token-level cho task bề mặt (dịch).

**Cách đo:** arm `lacc_sentence` vs `lacc_token`; kỳ vọng cách biệt lớn nhất ở
needle và long-doc QA, ít/không đổi ở summarization.
**Chi phí:** trung bình (cần sentence segmentation tiếng Việt — dùng luôn
underthesea/pyvi). **Rủi ro:** ở ratio cao (8×) một câu có thể vượt budget → cần
fallback token-level trong câu.

---

#### E6. 🟡 Thu nhỏ / đổi loại scorer: encoder classifier (LLMLingua-2) hoặc 0.5B

**Giả thuyết:** Bài toán tồn tại để *tiết kiệm*, mà scorer 4B nạp 8 GB chỉ để chấm
điểm token lại **thua** scorer 0.5B (§8.6). Insight §5 của báo cáo: "scorer tốt
hơn ≠ nén tốt hơn". Nên bỏ hẳn scorer generative to.

**Cơ sở:** LLMLingua-2 (ACL 2024 findings) — đặt nén = **token classification**
bằng một encoder cỡ BERT (XLM-RoBERTa), distill nhãn từ GPT-4; nhanh hơn 3–6×,
tốt hơn trên out-of-domain, và **entropy/perplexity một chiều vốn không thẳng
hàng với mục tiêu nén**. Với tiếng Việt có sẵn **PhoBERT** (đã nằm trong
`ideas.md` cải tiến 7) và XLM-R — encoder mạnh, rẻ.

**Thay đổi ở đâu:** thêm một `Compressor` mới dùng encoder phân loại token
"giữ/bỏ" (song song, một forward pass, không sliding window tuần tự). Có thể
distill nhãn từ chính Qwen2.5-14B đã dùng viết reference.

**Cách đo:** so `lacc_phobert_cls` vs LLMLingua vs LACC-4B ở cả chất lượng **và**
latency/VRAM (đây là arm để *thắng về chi phí*).
**Chi phí:** cần một vòng distill nhãn + train encoder (rẻ hơn LoRA 4B nhiều).
**Rủi ro:** trung bình; là hướng đầu tư mới nhưng khả năng đổi cả câu chuyện chi
phí lẫn chất lượng là cao nhất trong nhóm P1.

---

#### E7. 🟢/🟡 Thực sự implement class-proportional / iterative budget allocation

**Giả thuyết:** Điều paper *hứa* (phân bổ budget theo lớp từ, lặp) có thể chính là
thứ làm morphology mạnh hơn — nhưng code chưa có (phát hiện #3). Biến morphology
từ hệ số nhân thành **hạn ngạch**: ví dụ dành X% budget cho thực từ, cắt mạnh hư
từ trước, rồi mới top-k trong mỗi lớp.

**Cơ sở:** LLMLingua budget controller (coarse-to-fine + iterative token pruning);
PyramidKV/ReasonAlloc (budget không đồng đều). Đây là cách chuẩn để một tín hiệu
"loại token" tác động lên chọn lọc.

**Thay đổi ở đâu:** `select_with_boundary` (`:167`) → thêm biến thể phân bổ theo
nhóm morphology trước khi top-k. Đồng thời **sửa mô tả paper cho khớp** dù có
implement hay không.
**Chi phí:** thấp (thuật toán chọn, không train). **Rủi ro:** thấp.

---

### P2 — Cứu hướng tiếng Việt một cách có kiểm soát

#### E8. 🟢 Tone như tín hiệu *có điều kiện theo task*, không phải tín hiệu toàn cục

**Giả thuyết:** Có đúng một vùng tone thắng: task mà **bề mặt chữ chính là output**
— cross-lingual 2× tone_aware 0.292 > morph 0.249 (§8.7, insight 7). Ở đó giữ dấu
là *phần của task*, không phải chi phí.

**Đề xuất:** không bỏ tone, mà **route** nó: bật tone cho dịch/chuyển tự/trích dẫn
nguyên văn/sinh danh từ riêng; tắt cho QA/needle/summarization. Nối với ý tưởng
adaptive rate (Gap 3 trong `research_gaps.md`).
**Chi phí:** ~0 (chọn cờ theo task_type, đã có trong VCC-Bench).
**Rủi ro:** thấp; đây là cách trung thực để giữ một đóng góp tiếng Việt hẹp nhưng
thật.

#### E9. 🟡 Đo P3 (token inflation) — mệnh đề duy nhất chưa ai kiểm chứng

Báo cáo để P3 ⬜ "chưa đo". Đây là mệnh đề *dễ đo nhất* và có thể thành động lực
định lượng cho toàn bộ hướng tiếng Việt: đo tỉ lệ token VI/EN trên cùng nội dung
song ngữ (VCC-Bench cross-lingual đã có cặp), và nối sang **adaptive compression
rate** — nén tiếng Việt cần rate khác tiếng Anh. Rẻ, và lấp một ô trong bảng
contribution.

---

### P3 — Củng cố khoa học (bổ sung cho §12 báo cáo)

Ngoài các mục §12 (λ=0 control, λ sweep, agent n≥60, subset needle `none`…), thêm:

- **E10. 🟡 Sửa nhiệm vụ probe cho hết "quá dễ".** Nhãn thanh điệu là hàm tất
  định của token id (predictor tra bảng đạt 100%; control task đã 0.7530 chỉ nhờ
  token identity — §7.3, §12.6). Nếu vẫn giữ probe chẩn đoán, hãy **mask token id
  đang xét** (dự đoán thanh điệu của vị trí *bị che*) để buộc probe dùng ngữ cảnh,
  không tra bảng. Đây là điều kiện cần để bất kỳ tuyên bố "probe học được cấu trúc"
  nào đứng vững. (Nếu theo E4 thì vấn đề này tự biến mất — liên-quan-câu-hỏi không
  phải hàm tất định của token id.)
- **E11. 🟢 Bổ sung LongLLMLingua & LLMLingua-2 làm baseline chính thức.** Hiện chỉ
  có LLMLingua gốc + SnapKV. Thiếu hai baseline query-aware/encoder mạnh nhất →
  reviewer sẽ hỏi. Thêm chúng cũng đặt E1/E6 vào đúng khung so sánh.

---

## 3. Định vị lại câu chuyện paper (nếu muốn publishable)

Trục cũ — *"tone-aware compression cho tiếng Việt"* — đã bị chính dữ liệu bác bỏ,
đừng cố cứu. Ba trục thay thế, xếp theo độ chắc:

1. **Benchmark + negative result (chắc nhất, gần như xong).** VCC-Bench (C3) là
   đóng góp mạnh nhất còn lại; cộng với kết quả âm được kiểm soát chặt ("giữ dấu
   ≠ giữ thông tin; TPR nghịch biến chất lượng") là một paper
   *benchmark & analysis* hoàn toàn đứng được (đúng như Giai đoạn 1 trong
   `research_gaps.md`). Negative result + benchmark rất publishable ở venue tốt.
2. **Query-aware Vietnamese compression (nếu E1+E4+E5 dương).** Xoay từ "tone-aware"
   sang "query-conditioned, morphology-informed" — dùng chính hạ tầng train của
   wave 1 nhưng cho mục tiêu liên-quan. Đây là con đường có đóng góp *method* dương.
3. **Efficiency angle (nếu E6+E3 dương).** "Encoder-classifier 0 GB rẻ hơn và tốt
   hơn scorer generative cho tiếng Việt" — hẹp nhưng sạch.

---

## 4. Bảng ưu tiên tổng hợp

Cột **Code** cập nhật 2026-09-05: ✅ đã implement + test (CPU), sẵn cho đội training chạy.

| ID | Đề xuất | Impact | Effort | GPU | Code |
|---|---|---|---|---|---|
| E1 | Query-conditioned (contrastive) perplexity | 🔴 Rất cao | Thấp | 🟡 | ✅ `contrastive_perplexity`; arm `llmlingua_contrastive`, `lacc_ppl_contrastive` |
| E2 | Hợp nhất ppl + morphology | 🟡 Cao | Rất thấp | 🟢 | ✅ `morph_combine='multiply'`; arm `lacc_ppl_morph` |
| E3 | Sửa window 2048 (chi phí) | 🟡 Cao | Rất thấp | 🟢 | ✅ `DEFAULT_PPL_WINDOW=2048`, overlap 256 |
| E4 | Probe → dự đoán liên-quan (Sentinel) | 🔴 Rất cao | Trung bình | 🔴 | ✅ `RelevanceConsistencyLoss` + `scripts/train_relevance_probe.py` |
| E5 | Nén mức câu/mệnh đề | 🔴 Cao | Trung bình | 🔴 | ✅ `selection_unit='sentence'`; arm `lacc_sentence` |
| E6 | Encoder classifier (LLMLingua-2/PhoBERT) | 🔴 Cao | Trung bình-cao | 🟡 | ✅ `encoder_compression.py` + `scripts/train_encoder_compressor.py`; arm `encoder` |
| E7 | Class-proportional budget (khớp paper) | 🟡 Trung bình | Thấp | 🟢 | ✅ `budget_mode='class_proportional'`; arm `lacc_classprop` |
| E8 | Tone có điều kiện theo task | 🟢 Thấp-TB | Rất thấp | 🟢 | ✅ `tone_task_gate=True`; arm `lacc_tone_gated` |
| E9 | Đo P3 token inflation | 🟢 Thấp-TB | Thấp | 🟡 | ✅ `scripts/measure_token_inflation.py` |
| E10 | Sửa probe task (mask token) | 🟢 (khoa học) | Thấp | 🟡 | ⚠️ E4 thay thế (relevance ≠ hàm tất định của token id) |
| E11 | Thêm LongLLMLingua/LLMLingua-2 baseline | 🟡 (bắt buộc cho paper) | Thấp | 🟡 | ✅ = arm `llmlingua_contrastive` (E1) + `encoder` (E6) |

**Cách chạy (đội training):** `python benchmark.py --list-methods` liệt kê mọi arm mới.
Ví dụ: `python benchmark.py --model Qwen/Qwen2.5-7B-Instruct --ratios 2,4,8 --methods none,llmlingua,llmlingua_contrastive,lacc_ppl_contrastive,lacc_ppl_morph,lacc_sentence,lacc_classprop`.
Chi tiết đầy đủ ở `WAVE2_HANDOFF.md`.

**Đường đi khuyến nghị:** E3 + E2 + E7 (một buổi, không GPU, sửa chi phí và blend)
→ E1 (contrastive, quyết định phần lớn khoảng cách) → nếu E1 dương thì E5 + E4
(mức câu + probe liên-quan, đây là nơi có method mới) → E6 song song cho góc chi
phí → E11 trước khi submit.

**Không nên:** (1) đắp thêm tín hiệu tiếng Việt (teencode, dialect, wordnet…) lên
blend cộng — dữ liệu cho thấy thêm tín hiệu vào blend làm *loãng*, không làm tốt
lên; (2) chạy lại benchmark để "làm đẹp" số LACC — phép đo đã đúng; (3) tiếp tục
đầu tư scorer generative lớn hơn — §8.6 đã chứng minh vô ích.

---

## Tài liệu tham khảo mới (ngoài `references.md`)

- LongLLMLingua — Question-aware coarse-grained + contrastive perplexity, ACL 2024. [arxiv:2310.06839](https://arxiv.org/abs/2310.06839)
- LLMLingua-2 — Token classification bằng encoder, data distillation, ACL 2024 findings. [arxiv:2403.12968](https://arxiv.org/abs/2403.12968)
- Sentinel — Attention probing of a 0.5B proxy cho query-aware sentence compression, 2025. [arxiv:2505.23277](https://arxiv.org/abs/2505.23277)
- EXIT — Context-aware extractive compression cho RAG, ACL 2025 findings. [arxiv:2412.12559](https://arxiv.org/abs/2412.12559)
- CORE — Two-stage sentence-level compression, edge, 2026. [arxiv:2606.20571](https://arxiv.org/abs/2606.20571)
- RDRsegmenter / VnCoreNLP — word segmentation tiếng Việt nhanh (62k words/s). [arxiv:1709.06307](https://arxiv.org/abs/1709.06307)
