# Kết quả research — nguyên liệu viết paper LACC/VNCompress (COLING 2027 / ARR 12-10-2026)

> Trả lời cho `docs/agent_research.md`. 3 nhánh sub-agent đọc THẬT paper (WebFetch alphaxiv/arxiv),
> tổng hợp + đối chiếu chéo với code hiện có. Quy tắc: `[từ paper]` (kèm section) vs `[suy luận]`;
> mỗi kỹ thuật gắn `[Kịp 3.5 tuần]` / `[Future Work]` + công sức.
>
> **Đính chính trung thực (đọc trước):**
> - `2505.13518` KHÔNG phải paper class-imbalance có công thức loss — nó là survey resampling. Chỉ dùng
>   để chống lưng "F1/accuracy đơn ngưỡng gây hiểu lầm khi mất cân bằng".
> - LLMLingua-2 dùng **cross-entropy thường** (nhãn của họ ~cân bằng). Focal/ASL bên dưới là `[suy luận]`
>   từ Lin 2017 / Ridnik 2021 — khi viết phải cite hai paper đó, KHÔNG cite paper nén.
> - Tác giả *Lost in Compression* = **Mantas Lukauskas (Kaunas)** — đã có bibitem `lukauskas2026lost`.
>   (Không phải "Chen et al.".)

---

## 1. Bảng paper × kỹ thuật rút ra

| Paper (ID) | Kỹ thuật rút ra | Dùng §nào | Kịp? | Công sức |
|---|---|---|---|---|
| LLMLingua-2 (2403.12968) | Prompt distill extractive + chunk ≤512 cắt câu; Algorithm 1 align (sliding-window bidirectional, lemmatize); cổng **VR** & **AG**; infer = top-k theo budget | §Method-nhãn (C), §Method-E4 (A threshold) | Có | 3–5 ngày |
| Attribution Frontier (2609.14245) | EF1, grounded recall Rg, citation precision Pg + cổng joint-entail, unsupported-claim rate, **emitted−grounded gap** (control laundering), re-attribution | §Benchmark (B) | Có (lõi) | 3–5 ngày |
| Does Accuracy=Evidence (2608.01631) | **answer–evidence gap**, answer-chain consistency (RWAC), acc↔evidence Spearman ρ; (perturbation faithfulness, fixed-trace = Future Work) | §Benchmark (B), §Discussion | Có (lõi) | 3–4 ngày |
| Lost in Compression (2608.26175) | Setup 10 ngôn ngữ (KHÔNG có VN); "normalized context utilization"; transfer gap rate-dependent; **translate-then-compress thắng 3/5 ngôn ngữ hình thái giàu ở nửa token cost** | §Baseline (D), §Related Work (E) | Có | (đã có arm) |
| LongLLMLingua (2310.06839) | Ranh giới query-aware: perplexity câu hỏi \|context + phân bổ budget theo độ liên quan → **KHÔNG so với E6** | §Limitations (F) | Có | — |
| Survey (2410.12388) | Taxonomy chính = hard/soft, trong hard = filtering(extractive)/paraphrasing(abstractive). KHÔNG gán cho survey trục query-aware/agnostic (nó không tuyên bố tường minh) | §Related Work | Có | — |
| PhoBERT (2003.00744) | RoBERTa-base, pretrain 20GB/~3B word tokens, RDRSegmenter word-seg trước fastBPE | §Method-student | Có | — |

---

## 2. Bolt-on theo thành phần (đòn bẩy giảm dần) + trạng thái code

### E4 — probe (mục A). Bối cảnh: R=63.4% / P=5.6% / F1=0.103 @0.5, giữ:bỏ ≈ 1:44.

**Đòn bẩy #1 — Đổi cách BÁO/CHỌN ngưỡng, không phải model.** `[LLMLingua-2 §5]` Quyết định thực khi nén
là **top-k theo ngân sách** (`Ñ = τ·N`, giữ top-Ñ theo p, giữ thứ tự gốc), KHÔNG phải threshold 0.5.
→ F1=0.103 @0.5 gần như vô nghĩa; calibrate ngưỡng bằng **QA downstream trên VCC-Bench**, không bằng F1 probe.
Đây chính là luận điểm negative-result: *"F1 probe kém ≠ nén kém"*.
- Metric thay F1: **PR-AUC** (chính) + **Recall@budget** (τ∈{.05,.1,.2,.3}) + **MCC** (chọn ngưỡng vận hành). Bỏ F1@0.5.
- **Trạng thái code:** `evaluate_relevance_probe` ĐÃ trả PR-AUC + recall@budget. **GAP:** chưa có vòng
  calibrate τ theo QA downstream (nối probe→benchmark). `[Kịp 3.5 tuần, ~1 ngày]`

**Đòn bẩy #2 — Loss cho 44:1.** Thứ tự chạy: **weighted-CE `[từ 2409.03238 §3]` → Focal → ASL.**
- Weighted-CE: `w_c = 1 − N_c/N` (giữ≈0.978, bỏ≈0.022). Sàn bắt buộc.
- Focal `[suy luận, Lin 2017]`: `FL = −α(1−p)^γ y·log p − (1−α)p^γ(1−y)·log(1−p)`, γ=2, α≈0.9.
- **ASL (chính) `[suy luận, Ridnik 2021]`**: `L+=(1−p)^{γ+}·log p`, `L−=p_m^{γ−}·log(1−p_m)`, `p_m=max(p−m,0)`,
  γ+=0, γ−=4, m=0.05 — vứt gradient của token "bỏ" dễ, đúng thứ cần khi 44/45 token là "bỏ".
- **Trạng thái code:** `training.py` ĐÃ có `focal_gamma` + weighted-CE (cap 50). **GAP:** ASL chưa có;
  chưa CHẠY tuning (item G2 chưa tick, GPU segfault theo memory). `[Kịp 3.5 tuần, ~1–1.5 ngày + GPU]`
- **Bỏ:** dice/LDAM/resampling nặng — không paper nào ở vòng này hậu thuẫn.

### E6 — encoder-classifier query-agnostic (mục F). GIỮ query-agnostic.
- So công bằng CHỈ với task-agnostic: truncation, random drop, BM25/TF-IDF salience (query-agnostic),
  LLMLingua-2 (XLM-R chạy thẳng tiếng Việt). **KHÔNG** so LongLLMLingua (query-aware — khác lớp budget).
- Nơi E6 thắng: **nén-một-lần cache được** (query-aware phải nén lại mỗi câu hỏi); model nhẹ (PhoBERT-base);
  amortized cost trên corpus tĩnh nhiều truy vấn. Báo E6-gap = (query-aware ceiling − E6) như *đại lượng
  giới hạn*, không phải thất bại.

### Benchmark — VCC-Bench (mục B). Đóng góp chính.

**Insight framing quan trọng nhất (đối chiếu chéo 2 agent):** với nén **extractive trích nguyên văn**,
unit truy được provenance → `emitted = grounded`, `Pe = Pg`, `Us = Um = Ur`, gap ≈ 0 `[Attribution Frontier §3.3, Table 1]`.
→ Một nửa bộ metric attribution của paper đó trở thành **CONTROL/trivial cho extractive**, chỉ "sống lại"
khi so với baseline abstractive/translate. **Đây là điểm bán hàng:** extractive tiếng Việt *không launder
attribution* (gap≈0), tương phản translate-then-compress.

Lõi 3.5 tuần (theo đòn bẩy):
1. **span_recoverability_hard = 1 − VR** `[LLMLingua-2 + adapt]` — deterministic, KHÔNG NLI → không dính
   circularity. Arm `none`/`lacc_*` ≈1.0; `translate_then_compress` <1.0. **✅ ĐÃ IMPLEMENT + test** (`evaluation.py`,
   `tests/test_evaluation.py::test_hard_recoverability_*`). Đây là hiện thân code của "extractive không launder".
2. **emitted−grounded gap = Pe−Pg** `[Attribution Frontier §3.2]` — control, ~free khi có Pe,Pg.
3. **Grounded evidence recall Rg** `[§3.1]`: `1[NLI(⊕cited spans ⊨ statement)=entail]`. **Cần NLI VN.**
   Trạng thái: `source_span_recoverability` (NLI soft) ĐÃ có; **GAP:** chưa có second-evaluator sanity + human κ spot-check.
4. **unsupported-claim rate** `[§3.3]` (1 câu = 1 claim). **✅ ĐÃ có** (`compute_unsupported_claim_rate`, NLI).
5. **answer–evidence gap** `[Does Accuracy=Evidence §1,§4.2, adapt]`: `Gap(ρ)=Acc(ρ)/Acc(full) − Evid(ρ)/Evid(full)`
   — trục negative-result chính. ~free khi có thành phần.
6. **acc↔faithfulness Spearman ρ** `[§4.3]` — 1 câu chốt mạnh ("chỉ nhìn accuracy sẽ chọn nhầm compressor").
   Cần ≥5–6 arm. ~free.
- **EF1 (declared-evidence F1)** `[Attribution Frontier §3.4]`: set-F1(selected vs gold spans). **HÓA RA GẦN NHƯ SẴN SÀNG
  (đo dữ liệu thật):** `qa.jsonl` v2 có `answer_span` = **offset ký tự chính xác** 6000/6000 (`context[start:end]==answer`),
  và `reference_answer` verbatim trong context 6000/6000. → **KHÔNG cần chiến dịch annotation**; chỉ cần map offset→câu chứa
  span (deterministic). **✅ ĐÃ nâng `find_evidence_sentence`** ưu tiên câu chứa answer *verbatim* (thay heuristic token-overlap)
  → evidence_recall/recoverability lên mức ground-truth cho đa số mẫu. Còn lại (nếu muốn EF1 set-F1 đầy đủ): mang `answer_span`
  offset vào `vcc_bench_v2.json` khi build (benchmark hiện chỉ có `reference_answer` text) — nhỏ, ~0.5 ngày.
- **Future Work:** perturbation faithfulness (đo robustness *reader* chứ không phải compressor, ~1 tuần);
  fixed-trace replay (KV-cache, không áp cho nén-text); claim-decomposition thật (FActScore VN).

**Nút cổ chai chung:** NLI/entailment tiếng Việt (mDeBERTa-v3 XNLI đa ngữ). BẮT BUỘC: second-evaluator
sanity + human spot-check vài chục ví dụ, nếu không reviewer CL bác vì circularity `[Attribution Frontier §9]`.

---

## 3. Nhãn keep/drop (mục C) + phát hiện "bịa số"

Thuật toán LLMLingua-2 (đủ tái hiện) `[2403.12968 §data/annotation]`: (1) teacher nén extractive bằng prompt
cấm đổi/thêm/đảo từ, chunk ≤512 cắt câu; (2) Algorithm 1 gán nhãn token gốc = True nếu khớp từ trong compressed
qua **sliding-window bidirectional** + fuzzy lemmatize; (3) cổng **VR** (`= (1/|S_comp|)Σ 1[w∉S_ori]`, loại 5% cao nhất)
và **AG = HR − MR** (loại 10% cao nhất).

Áp tiếng Việt (teacher Qwen2.5-0.5B-vi): thay lemmatize spaCy bằng **word-seg (VnCoreNLP/underthesea) + NFC + lowercase,
bỏ lemmatize** (tiếng Việt không biến hình); map nhãn từ→subword PhoBERT, trung bình xác suất subword.
Cảnh báo: teacher 0.5B tuân prompt kém GPT-4 → nhiều abstractive hơn → có thể phải lọc mạnh hơn 5%/10%.

**Cổng VR/AG có bắt "bịa số" (~21% của `compression.jsonl`) không? → KHÔNG (đủ):**
- VR bắt được bịa *tràn lan* nhưng bịa *lẻ tẻ* chỉ tạo `VR≈1/|S_comp|` → lọt dưới ngưỡng top-5%.
- AG *giảm* khi có số bịa (hạ HR) → cổng loại 10% AG *cao nhất* nên KHÔNG bắt.
- 30% dòng trùng nguyên văn: VR≈0, AG≈0 → hai cổng cho qua *đúng ý đồ* (verbatim không phải lỗi extractive;
  vấn đề trùng câu là **ratio**, xử lý bằng ràng buộc budget khi distill, không bằng cổng lọc).

**Vá deterministic: NumGate — ✅ HÓA RA ĐÃ CÓ VÀ ĐANG CHẶN (đo dữ liệu thật 2026-09-18).**
`dataset_build.unsupported_numbers()` (regex số + chuẩn hóa, đúng đề xuất) là **gate loại cứng** ở
`generate_compression_pairs.py:505-508` (drop mẫu bịa số), cộng gate sentence-extractive ≥ 0.80.
Đo `data/vncompress_vi_v2/compression.jsonl` (486 hàng): **0/486 hàng bịa số**, sentence_extractive mean=0.981
min=0.800. Nghĩa là "21% bịa số / 30% trùng / gold viết lại" là trạng thái **v1**; **v2 đã sạch**. KHÔNG re-implement.

---

## 4. Baseline bắt buộc (mục D)

| Baseline | Vì sao BẮT BUỘC | Trạng thái | Kịp? | Công sức |
|---|---|---|---|---|
| **Translate-then-compress (VI→EN→LLMLingua-2)** | Đe doạ trực tiếp premise "cần compressor VN"; Lost-in-Compression: thắng 3/5 ngôn ngữ hình thái giàu ở ½ token cost `[từ paper]`. Không chạy = reject | arm `translate_then_compress` + `_llmlingua2` ĐÃ có | Có | 2–3 ngày (chạy full) |
| Truncation (head/tail) | Sàn tuyệt đối, query-agnostic thuần, fair với E6 | arm có | Có | 0.5 ngày |
| Random token/sentence drop | Sàn ngẫu nhiên (E6 hơn "chỉ nhờ ít token"?) | arm `random` | Có | 0.5 ngày |
| BM25/TF-IDF salience (query-agnostic) | Baseline extractive không-học mạnh nhất cùng điều kiện E6 | **GAP: kiểm arm** | Có | 1 ngày |
| LLMLingua-2 (query-agnostic, chạy thẳng VN qua XLM-R) | SOTA extractive query-agnostic = "nén bản ngữ", đối thủ chính E6 | arm `encoder`/`llmlingua` | Có | 1–1.5 ngày |
| Full context / No context (closed-book) | Trần trên + sàn dưới; no-context = leak-probe (mẫu số utilization) | arm `none` (+ cần no-context) | Có | 0.5 ngày |
| LongLLMLingua (query-aware) | CHỈ định vị trần query-aware — ghi "reference, KHÔNG phải comparator của E6" | arm `llmlingua_contrastive` | Có (nếu dư) | 1.5–2 ngày |

Tổng nhóm bắt buộc ≈ **6–7 ngày công** — kịp. LongLLMLingua đẩy `[Future Work]` nếu GPU segfault chưa gỡ.

---

## 5. Related Work — đoạn viết sẵn (mục E), dùng ngay

> Our work is closest in spirit to Lukauskas's cross-lingual audit (*Lost in Compression*, 2026), which
> shows that extractive prompt compressors supervised on English (e.g., LLMLingua-2) transfer poorly to
> other languages, that this transfer gap is strongly rate-dependent, and that a translate-then-compress
> pipeline can match or beat native-language compression at roughly half the token cost in three of five
> morphologically rich languages tested. We differ on both axis and artifact. That audit spans ten
> languages across five scripts — none of them Vietnamese or any tonal, Southeast-Asian language — and
> measures *normalized context utilization* of off-the-shelf compressors without a leak-controlled QA
> benchmark. Our contribution is instead (i) VCC-Bench, a Vietnamese extractive-QA benchmark with explicit
> leak control, and (ii) a negative result: training a Vietnamese-specific extractive compressor (with a
> PhoBERT student) and finding that Vietnamese-specific surface signals such as tone and morpheme
> structure do **not** yield measurable gains for extractive QA over language-agnostic selection. We make
> no claim about the linguistic importance of tone in general — only that, within the scope of
> query-agnostic extractive compression for QA, it does not translate into downstream answer accuracy.

(Overclaim-guard đã cài ở câu cuối: giới hạn phạm vi "query-agnostic extractive compression for QA".)

---

## 6. Mâu thuẫn & rủi ro (mục G)

1. **Loss E4 không có nguồn từ paper nén.** Focal/ASL là `[suy luận]` (Lin/Ridnik) — với reviewer CL phải cite
   đúng hai paper gốc, KHÔNG cite LLMLingua-2/2409 cho công thức focal. Weighted-CE thì cite 2409.03238 hợp lệ.
2. **NLI VN = circularity risk** cho Rg/Pg/unsupported/soft-recoverability. Không có second-evaluator + human κ
   thì các số faithfulness bị bác. → span_recoverability_hard (đã làm, no-NLI) và EF1 (string) là "phao" sạch nhất.
3. **EF1 — KHÔNG còn bị chặn (đo dữ liệu thật đã lật giả định cũ):** memory `dataset-v2-review` ("gold viết lại")
   phản ánh trạng thái v1/cũ. Đo v2 thật: NumGate đang chặn (0/486 bịa số), gold 98.1% extractive, `answer_span` là
   offset chính xác 100%. → verbatim span ĐÃ có; EF1 chỉ cần map offset→câu (đã nâng `find_evidence_sentence`).
   *Bài học: đo dữ liệu thật trước khi thiết kế annotation — tránh được cả một chiến dịch annotation thừa.*
4. **LongLLMLingua là query-aware** — tuyệt đối không dùng làm comparator của E6 (đã chốt; arm `llmlingua_contrastive`
   phải ghi nhãn "reference ceiling"). Không mâu thuẫn với ràng buộc, chỉ là kỷ luật ghi bảng.
5. **Perturbation faithfulness / fixed-trace** đo *reader*/KV-cache, không phải compressor → lệch trọng tâm → Future Work.
6. **Báo achieved ratio, không nominal** `[Attribution Frontier §4.1: nominal phóng đại tới 32.8×]` — extractive bám budget
   sát nhưng vẫn phải báo. Repo đã có dual-tokenizer achieved-rate.

---

## 7. KẾT LUẬN — 3–5 hướng ưu tiên LÀM ĐƯỢC trong 3.5 tuần

1. **[ĐÃ LÀM] span_recoverability_hard (= 1−VR)** — control "extractive không launder", no-NLI, đã test. Đưa vào §Benchmark
   như control chính; ghép với emitted−grounded gap khi có baseline abstractive.
2. **[ĐÃ CÓ] NumGate** — `unsupported_numbers()` đã là gate loại cứng trong build v2; đo thật 0/486 bịa số. Không phải làm gì
   thêm ngoài xác nhận v2 là bản đang dùng để train/eval. **[ĐÃ LÀM] find_evidence_sentence** ưu tiên câu chứa answer verbatim
   (dùng offset span 100% chính xác của v2) → evidence metric lên mức ground-truth.
3. **[Kịp, ~1–1.5 ngày + GPU] Chạy loss E4** theo thứ tự weighted-CE→ASL, báo PR-AUC/recall@budget; calibrate ngưỡng
   top-k bằng QA downstream → dựng luận điểm "F1 kém ≠ nén kém".
4. **[Kịp, lõi] Bộ metric faithfulness §Benchmark:** Rg + unsupported (đã có NLI) + answer–evidence gap + Spearman ρ,
   KÈM second-evaluator sanity + human κ spot-check. Framing extractive-triviality (Pe=Pg) làm điểm bán hàng.
5. **[Kịp, 6–7 ngày] Chạy đủ baseline** (translate-then-compress là sống-còn) trên VCC-Bench v2 (G1, GPU pod).

**Ngã rẽ EF1 đã giải quyết:** user chọn "thêm annotation span verbatim", nhưng đo dữ liệu thật cho thấy verbatim span
ĐÃ CÓ (offset chính xác 100%) → không cần annotation. Đã nâng `find_evidence_sentence` dùng verbatim. Nếu muốn EF1 set-F1
đầy đủ, việc còn lại duy nhất là mang `answer_span` offset vào `vcc_bench_v2.json` khi build (~0.5 ngày) — quyết định khi cần.

**Future Work:** perturbation faithfulness, fixed-trace replay, claim-decomposition VN, LongLLMLingua (nếu GPU kẹt),
distillation confidence-weighted (paper vòng sau).
