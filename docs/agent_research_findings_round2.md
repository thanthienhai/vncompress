# Research vòng 2 — ĐỌC THÊM paper, tìm hướng KHẮC PHỤC điểm yếu của đề xuất LACC/VNCompress

> Nối tiếp `docs/agent_research_findings.md`. 4 nhánh sub-agent đọc THẬT paper (arxiv/aclanthology/alphaxiv),
> mỗi nhánh nhắm 1 điểm yếu. Quy tắc: `[từ paper]` (venue+section) / `[suy luận]`; `[Kịp 3.5 tuần]`/`[Future Work]` + công sức.
>
> **Verify trước khi cite (agent đưa nhưng cần đối chiếu ID):** Focal Loss = Lin et al. 2017 **arXiv:1708.02002**
> (không phải id agent ghi). Selective Context (EMNLP 2023) = **2310.06201**. Các ID khác (RECOMP 2310.04408,
> Provence 2501.16214, EXIT 2412.12559, LLMLingua-2 2403.12968, ViNLI 2022.coling-1.339, Hewitt&Liang D19-1275,
> Voita&Titov 2020.emnlp-main.14, Card et al. 2020.emnlp-main.745, ALCE 2023.emnlp-main.398, AttrScore
> 2023.findings-emnlp.307) khớp và kiểm được.
>
> **Chốt bối cảnh (đã đọc code):** nhiều "fix" hóa ra ĐÃ CÓ khung trong repo → đừng làm lại, chỉ bolt-on phần thiếu.

---

## Điểm yếu #1 (chí mạng) — Negative result bị confound "method quá yếu" (E4 F1=0.103)

Reviewer sẽ bác: "tone/hình vị không giúp CHỈ vì compressor của các bạn không bắt được gì." Đây là mối đe doạ số 1
cho một negative-result paper. **Đã có trong repo:** `scripts/train_probe_control.py` cài sẵn **control task + selectivity
(Hewitt & Liang 2019)** cho tone probe (`apply_control_labels`); `scripts/train_relevance_probe.py` (E4) đã dùng
**nhãn answer-span sạch** (verified 6000/6000) + focal-gamma + class-weight + query-conditioning + holdout leak-control;
`evaluation.py` có `paired_bootstrap_delta`. Vậy gap là **bộ máy làm negative result ĐÁNG TIN**, không phải probe.

| Fix | Nguồn | Trạng thái | Kịp? | Công sức |
|---|---|---|---|---|
| **POSITIVE CONTROL** — chạy tín hiệu overlap câu-hỏi↔câu (BM25/lexical, đã có arm `SentenceRetrieval(BM25)` `compression.py:1481`) qua CHÍNH pipeline yếu. Nếu overlap thắng no-context/random mà tone/hình vị không thêm gì → negative result bất khả phản biện "method yếu" | Hewitt&Liang; Belinkov 2022 §Comparisons-and-Controls | infra có (BM25 arm); cần đóng khung "positive control" trong bảng | **Kịp** | ~1.5 ngày |
| **Selectivity cho E4 relevance probe** (tái dùng `apply_control_labels`), báo ở ≥2 dung lượng probe (chặn phản biện Pimentel "cố tình làm probe yếu") | Hewitt&Liang; Pimentel 2020 | control task có cho tone, **thiếu cho relevance** | **Kịp** | ~0.5 ngày |
| **TOST equivalence test + SESOI đăng ký trước (=1 điểm EM/F1)** → phát biểu "tương đương trong ±1 EM", KHÔNG phải "p>0.05" | Lakens 2017 | **✅ ĐÃ IMPLEMENT** `tost_equivalence()` + test (`evaluation.py`) | **Kịp** | xong |
| **Power analysis + bootstrap CI cho mọi Δ**; nêu thẳng subset E8 n=18 underpowered → hạ cấp phát biểu ở đó | Card et al. 2020; Dror 2020 | **✅ ĐÃ IMPLEMENT** `paired_power_analysis()` (power + MDE, chuẩn Card et al.) + test; bootstrap có sẵn | **Kịp** | xong |
| **MDL online-code (codelength) cho tone+relevance** — kết luận độc lập probe capacity ("tone không giảm codelength quá control") | Voita&Titov 2020 | thiếu | Kịp (online-code) / variational [Future Work] | ~1–1.5 ngày |
| **Feature-ablation matrix end-to-end** {+overlap(positive), +tone, +morph, none} × EM/F1 reader | Belinkov "correlation vs causation" | tái dùng eval | Kịp (tái dùng) / full sweep [Future Work] | ~2 ngày |

**Câu chốt phương pháp luận (Belinkov):** chỉ phát biểu mạnh khi ≥3-4 method đồng thuận (selectivity + positive control
+ TOST + power/MDL); nếu chỉ probe yếu đơn độc → phát biểu "không thấy bằng chứng DƯỚI các method này" + ghi scope ở Limitations.

---

## Điểm yếu #2 — Compressor yếu (E4/E6) làm confound + kết quả nghèo

**Ràng buộc:** E6 GIỮ query-agnostic. RECOMP/Provence/EXIT/CPC đều **query-aware → chỉ E4**; chỉ LLMLingua-2 /
Selective Context / distill-improve an toàn cho E6.

| Fix | Nguồn | Cho | Kịp? | Công sức |
|---|---|---|---|---|
| **E4: focal/weighted loss + ngưỡng top-k/document (thay 0.5) + calibration** | Lin 2017; LLMLingua-2/Provence | E4 (đã có focal_gamma) | **Kịp** | rất thấp |
| **E6: thay nhãn self-distill Qwen-0.5B bằng pipeline LLMLingua-2** (teacher nén chunk-wise ≤512 + align sliding-window → nhãn keep/discard), giữ token-classification query-agnostic; nâng teacher 7B+ | LLMLingua-2 (ACL 2024 Findings) | E6 | **Kịp** | trung bình |
| **E4: silver-label Provence (LLM trích dẫn câu `[i]`) + FilCo STRINC/LEXICAL từ đáp án gold** | Provence ICLR25; FilCo 2023 | E4 (nhãn answer-span STRINC **đã có**) | **Kịp** | thấp-TB |
| **Baseline mạnh chống "method yếu":** query-agnostic = Selective Context + LLMLingua-2 off-the-shelf (E6); query-aware = Provence/XProvence + EXIT (E4) | các paper trên | baseline | **Kịp** | trung bình |
| **E4 head query-conditioned cross-attention (thay linear) + E6 soft-label/teacher-ensembling/confidence-weighting** | probing chuẩn; KD chuẩn | nâng trần | Kịp (head) / ensembling thấp | TB |

**Hội tụ quan trọng:** FilCo STRINC (nhãn = câu chứa answer-span) = chính nhãn E4 đang dùng = positive-control của #1 =
`find_evidence_sentence` (vòng 1). Một nguồn answer-span verified phục vụ cả 3.

---

## Điểm yếu #3 — Premise "cần compressor tiếng Việt" bị đe doạ bởi translate-then-compress

**Đòn phản công mạnh nhất (từ chính paper đe doạ):** *Lost in Compression* (Lukauskas, 2608.26175 §6) kết luận
**"supervision-data language, not architecture, drives the gap"** — compressor multilingual-native (XProvence trên 16 ngôn ngữ
native) **KHÔNG có transfer gap**. → dịch-rồi-nén chỉ thắng khi compressor native THIẾU dữ liệu bản ngữ; VCC-Bench + dataset
tiếng Việt của ta lấp đúng chỗ đó. Thêm: nó **không test ngôn ngữ thanh điệu/đơn lập nào** (thắng toàn Baltic/Finnic biến hình);
XProvence-v2 train trên dữ liệu **dịch** trả context RỖNG 92% cho tiếng Trung (pipeline dịch giòn, lỗi im lặng).

| Luận cứ bảo vệ premise | Nguồn | Kịp? | Công sức |
|---|---|---|---|
| Chạy baseline translate-then-compress trên subset cross_lingual (n=18) + **đo entity/number preservation + answer-span EM** giữa native vs dịch-rồi-nén (native thắng ở đây dù thua token) | XOR-QA 2020; Lingua Franca 2023 (MT phá entity/số) | **Kịp** | ~3 ngày |
| **Đo lại leak-control sau MT round-trip** (dịch máy phá cổng chống rò) | contamination survey | **Kịp** | ~0.5 ngày |
| Định vị: đơn vị nén (âm tiết≠từ, segmentation còn tranh cãi — Nguyen et al. AAAI 2025) + thanh điệu mang nghĩa = quyết định RIÊNG tiếng Việt | AAAI 2025; PhoBERT/VnCoreNLP | **Kịp** (viết) | ~0.5 ngày |
| Full "safe-budget vs token-premium" cho tiếng Việt kiểu Lukauskas Fig 6 | — | [Future Work] | — |

Câu Related Work/Intro dùng ngay (agent C): xem cuối file.

---

## Điểm yếu #4 — Metric faithfulness phụ thuộc 1 NLI tiếng Việt (circularity) → reviewer bác số

**Đã có:** `source_span_recoverability`/`unsupported_claim_rate` (NLI mDeBERTa-XNLI), `span_recoverability_hard` (deterministic, vòng 1).

| Fix | Nguồn | Kịp? | Công sức + bộ chấm |
|---|---|---|---|
| **Ensemble ≥2 evaluator KHÁC HỌ**: mDeBERTa-XNLI + **NLI bản ngữ fine-tune ViNLI/ViANLI** (CafeBERT/XLM-R) + họ QA/QG; báo agreement + **correlation với `span_recoverability_hard`** (anchor no-NLI) | TRUE (NAACL22); AttrScore; ViNLI COLING22 | **Kịp** | ~1 GPU-ngày fine-tune ViNLI |
| **Human calibration 150–200 mẫu, nhãn fine-grained (attributable/extrapolatory/contradictory), báo Cohen's κ + ROC-AUC per-metric** (chuẩn ALCE §6; AttrScore: NLI hỏng nhất ở contradictory/partial) | ALCE; AttrScore; FActScore | **Kịp** | ~2–3 ngày gán nhãn, 2 annotator |
| **Bảng "Contamination Audit":** closed-book vs open-book gap + 13-gram overlap với Wikipedia/OSCAR/C4/CulturaX-vi + perturbation drop + provenance dates + canary GUID trong release | contamination survey 2502.14425 | **Kịp** | ~1–1.5 ngày |
| **Báo threshold-free (ROC-AUC)** thay vì accuracy tại 1 ngưỡng entailment; câu Limitations chủ động "NLI chưa calibrate factual VN → dùng ensemble + human κ" | TRUE | **Kịp** | ~0.5 ngày |
| Nếu dùng LLM-judge: method-anonymized + randomize order + báo κ VÀ exact-match (κ deflation 33–41 điểm); nếu KHÔNG dùng → tuyên bố tường minh (là điểm mạnh) | Zheng NeurIPS23; Reliability-without-Validity | **Kịp** | ~0.5 ngày |

---

## MASTER LIST — ưu tiên khắc phục trong 3.5 tuần (đòn bẩy giảm dần)

1. **[ĐÃ LÀM] TOST equivalence + [ĐÃ LÀM] Power analysis (`paired_power_analysis`) + [Kịp, ~1.5 ngày] POSITIVE CONTROL
   (BM25 overlap qua pipeline) + [Kịp, ~0.5 ngày code + GPU run] selectivity E4.** Cụm này cứu điểm yếu #1 (chí mạng) —
   biến negative result thành khẳng định có kiểm soát. Bộ ba thống kê (TOST + power + bootstrap) giờ đã đủ, chỉ còn positive
   control (analysis) và selectivity (cần 1 train run GPU).
3. **[Kịp, TB] E6 thay nhãn self-distill bằng pipeline LLMLingua-2 (teacher 7B, align sliding-window).** Sửa gốc "E6 nhãn kém".
4. **[Kịp, ~1 GPU-ngày + 2–3 ngày người] Eval ensemble ≥2 NLI khác họ + human κ 150–200 mẫu + bảng Contamination Audit.**
   Cứu điểm yếu #4, biến leak-control (đóng góp chính) thành bằng chứng định lượng.
5. **[Kịp, ~3.5 ngày] Baseline translate-then-compress + đo entity/number/span-EM + leak sau MT.** Cứu premise (#3).

**Future Work:** MDL variational-code; full feature-ablation sweep; safe-budget/token-premium chart; XProvence multilingual; head cross-attention nếu GPU rảnh.

---

## Câu Related Work / Intro dùng ngay (điểm yếu #3, agent C — dùng Lukauskas, không "Chen")

1. "Recent cross-lingual audits show extractive prompt compressors trained on English transfer unevenly: translate-then-compress
   can match native compression at ~half the token cost for several morphologically rich Baltic/Finnic languages (Lukauskas, 2026),
   **but that audit covers no isolating, tonal, or Southeast-Asian language and no Vietnamese**, leaving the regime we study unexamined."
2. "Crucially, the same audit finds the transfer gap is driven by **supervision-data language, not architecture** — a
   natively-multilingual pruner shows no gap — which motivates building a Vietnamese-native compression resource rather than
   defaulting to a translate-then-compress pipeline."
3. "Translate-then-process pipelines are brittle: full pre-translation of prompts can degrade accuracy substantially (Gupta et al.,
   2025 — verify số −32% trước khi cite), and machine translation systematically corrupts named entities and numbers unless copied
   verbatim (XOR-QA; Lingua Franca) — precisely the spans extractive Vietnamese QA must preserve."
4. "Vietnamese further separates syllables, not words, in orthography (Nguyen et al., AAAI 2025) and encodes meaning in tone
   diacritics, so the unit of compression is itself a language-specific design choice absent from English compressors."
