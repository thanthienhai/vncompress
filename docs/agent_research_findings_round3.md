# Research vòng 3 — NÂNG CẤP ĐỘT PHÁ cho LACC/VNCompress (COLING 2027, top-tier, theme "NLP for Linguistics")

> Nối tiếp `agent_research_findings.md` + `_round2.md`. 4 nhánh sub-agent đọc CHI TIẾT paper (arxiv/aclanthology).
> Bối cảnh: snapshot training ĐÓNG BĂNG → mọi đề xuất chạy trên biểu diễn/inference đã có, KHÔNG train lại.
> Quy tắc: `[từ paper]` (venue+section) / `[suy luận]`; `[Kịp 3.5 tuần]` / `[Future Work]` + công sức.
>
> **Verify ID trước khi cite (một số future-dated, plausible vì hôm nay 09/2026 nhưng phải mở đọc lại):**
> 2603.27653 (diacriticity), 2608.10414 (VialectBench), 2604.27712 (52.8% collision), 2506.11673 (LEACE-vs-INLP),
> 2605.22462 (five-stage), ACL 2026 long-472 (HuTieuBERT). ID đã kiểm chắc: LEACE 2306.03819 (NeurIPS 2023),
> Amnesic Probing TACL 2021, INLP ACL 2020, RLACE ICML 2022, Kernelized EMNLP 2022 (2201.12191),
> Kumar NeurIPS 2022 (2207.04153), "Don't Touch My Diacritics" 2410.24140, HANS P19-1334, Show-Your-Work 1909.03004.

---

## 0. LUẬN ĐỀ ĐỘT PHÁ (4 nhánh hội tụ thành MỘT câu chuyện)

Hiện negative-result đứng trên probe **tương quan** (E4 F1=0.103) → reviewer bác "probe fail ≠ tín hiệu vô dụng".
Nâng cấp: chuyển thành một phát hiện **ngôn ngữ học có kiểm chứng ba mặt (triangulation)**, đóng gói thành **protocol
tái dùng** — vượt khỏi "case study tiếng Việt".

> **Research question (đóng khung cả bài):** Ở các ngôn ngữ có thanh điệu và ranh giới hình vị không tường minh,
> các đặc trưng phụ-âm-vị (thanh điệu, hình vị) có mang thông tin **bổ sung, KHÔNG dư thừa** cho việc chọn-giữ nội dung
> khi nén extractive phục vụ QA hay không — và ta đo được điều đó một cách **kiểm-soát-rò-rỉ, tái-lặp, chuyển-ngôn-ngữ** đến đâu?

Trả lời "không" trở thành phát hiện có giá trị vì được chứng minh bằng **3 mặt độc lập + 1 giải thích cơ chế**:

| Mặt chứng minh | Công cụ | Chạy trên snapshot? |
|---|---|---|
| **Nhân quả (representation)** | LEACE erasure xoá thanh điệu khỏi hidden states + Rand baseline + control concept | ✅ không train lại |
| **Hành vi (behavioral, input)** | Minimal-pair {có dấu / mất dấu / **sai dấu**} → reader; + control ngữ nghĩa | ✅ inference-only |
| **Thông tin (information-theoretic)** | Diacritic-restorability của câu evidence (91–99% char) → tone ≈ hàm của ngữ cảnh | ✅ rẻ |
| **Cơ chế (typology)** | hình vị = tín hiệu *cú pháp* không phải *relevance ngữ nghĩa*; VN đơn lập ≠ ngôn ngữ giàu hình thái | phân tích |

Ba mặt cùng chỉ một hướng ⇒ claim "thanh điệu/hình vị không mang tín hiệu functional bổ sung cho SELECTION" trở nên **bất khả phản biện "method yếu"**.

---

## 1. Framing đưa lên top-tier (nhánh framing)

- **Đóng khung ở tầng câu hỏi NGÔN NGỮ HỌC** (chức năng/độ dư thừa thông tin của thanh điệu), tiếng Việt là ca kiểm định — đúng theme "NLP phục vụ ngôn ngữ học" (không phải "cải thiện nén tiếng Việt" → lệch theme). `[COLING 2027 CfP]`
- **VCC-Bench = HANS**: benchmark chẩn đoán leak-control là *di sản*; negative-result là "kết quả đầu tiên nó bóc ra". `[McCoy et al. ACL 2019]`
- **`lacc_tone` (E8) = control task kiểu Hewitt&Liang**: không nói "thanh điệu vô dụng", mà "đây là một quy trình control để kiểm feature có mang tín hiệu không". `[Hewitt&Liang EMNLP 2019]`
- **Đóng gói protocol tái dùng + tool + lời mời** mở rộng sang ngôn ngữ thanh điệu khác (Thái/Lào/Quan Thoại) → tầm rộng, kiểu Show-Your-Work. `[Dodge et al. EMNLP 2019]`
- ARR chấm **Soundness/Excitement tách rời** → negative result chết ở Excitement; framing này kéo Excitement, positive control giữ Soundness. Tự khai nhiều **contribution type** (resource + analysis + low-resource) để không bị chấm như "engineering experiment thua SOTA". `[ARR guidelines]`
- Cấu trúc abstract: câu 1 = câu hỏi ngôn ngữ học phổ quát (chưa nhắc "tiếng Việt"); câu 2 = khoảng trống phương pháp; câu 3 = 3 đóng góp khai đúng type; câu 4 = kết quả bất ngờ CÓ SCOPE + "positive control xác nhận pipeline bắt được tín hiệu khác"; câu cuối = lời mời mở rộng.

---

## 2. Đóng góp TÍCH CỰC thứ hai (nhánh contribution) — nâng khỏi "case study"

**FUSE — Functional-Utility Surface-feature Equivalence test** `[đề xuất mới, novelty đã check]`: đóng gói 4 chân thành
một phương pháp luận CÓ TÊN để khẳng định "đặc trưng bề mặt X không mang tín hiệu functional cho task Y":
1. positive control (BM25 overlap qua chính pipeline yếu — arm `compression.py:1481`);
2. causal erasure input-space (strip/swap dấu, resegment — deterministic, không train lại);
3. **TOST equivalence + SESOI đăng ký trước** `[✅ tost_equivalence() đã có]`;
4. **power analysis + bootstrap CI** `[✅ paired_power_analysis() đã có]`.
- Novelty: khác **amnesic probing** (xoá ở biểu diễn nội tại, đo trên MLM, KHÔNG equivalence test) và **five-stage
  methodology** (không equivalence, không surface-input downstream). Chưa ai ghép 4 chân này thành protocol khẳng-định-vắng-mặt.

**Đóng góp phụ (miễn phí, code đã có): anchor deterministic bắt NLI-laundering.** Nâng `span_recoverability_hard`
(= 1−VR, no-NLI) thành chẩn đoán tái dùng: `L = P(NLI=entail | hard_recoverable=0)` — lấp đúng lỗ **circularity mà
Attribution-Compression Frontier (2609.14245) TỰ THỪA NHẬN** ("recovery relies entirely on the same NLI model... acknowledged
circularity", "human calibration remains absent"). Reusable cho mọi harness faithfulness-NLI.

---

## 3. Kỹ thuật nhân quả — chi tiết implement (nhánh causal)

**Công cụ chính = LEACE** (Belrose et al., NeurIPS 2023, 2306.03819): eraser affine **đóng-form một bước**, chặn mọi
classifier tuyến tính đọc concept mà biến dạng biểu diễn ÍT NHẤT → collateral damage cực thấp (cosine 0.80–0.95 với gốc),
**không lặp, không train lại** → hợp snapshot. Thư viện `concept-erasure` của chính tác giả.
- Công thức (Eq.1): `r(x) = x − W⁺·P_{WΣxz}·W·(x − E[X])`, `W=(Σxx^{1/2})⁺`, ràng buộc `Cov(Px, Z)=0`.
- Quy trình amnesic (Elazar TACL 2021): erase Z khỏi hidden states → đưa qua phần còn lại → đo Δ hành vi; **Rand baseline**
  (xoá cùng số chiều ngẫu nhiên) để chống nhầm "mất chiều = mất năng lực".
- **Concept Z**: thanh điệu = nhãn categorical per-token {ngang, huyền, sắc, hỏi, ngã, nặng}; hình vị = ranh giới âm tiết/ghép.
- **Control concept (BẮT BUỘC)**: đặc trưng KỲ VỌNG hữu ích (lexical-overlap với câu hỏi / có answer-span) — xoá cái này PHẢI
  làm sập downstream (positive control cho chính erasure).
- **Downstream đo**: sentence-selection (câu chọn có chứa answer-span?), QA EM/F1, relevance ranking.
- Cross-check: **RLACE** (ICML 2022, cơ chế khác, rank thấp) chạy song song; nếu cùng kết luận → rất chắc.

**Rào overclaim (bắt buộc ở Limitations):** (a) LEACE/RLACE chỉ xoá thông tin **tuyến tính** — trích Kernelized Erasure
(EMNLP 2022): bảo vệ KHÔNG chuyển sang adversary phi tuyến; (b) trích **Kumar et al. NeurIPS 2022**: removal có thể kết luận
nhầm → phòng bằng positive control + Rand + kiểm concept thật sự bị xoá; (c) dùng LEACE thay INLP để tránh collateral damage
(2506.11673). Đúng bộ ba control này mới nâng negative result từ "probe fail" lên "can thiệp nhân quả kiểm chứng được".

---

## 4. Cơ chế "TẠI SAO" (nhánh mechanism) — để negative result có giải thích

- **Tuyến chính (dư thừa):** dấu thanh khôi phục được từ ngữ cảnh không dấu (char 91–99%, word ~74%) `[1709.07104; 2603.27653]`
  + encoder PhoBERT đã mã hoá sẵn dấu → **giá trị biên cho SELECTION ≈ 0**. Money experiment (rẻ nhất): reader trên context
  {có dấu / mất toàn bộ / mất chỉ câu evidence} `[Kịp, dùng strip_tone() sẵn có]`; + đo diacritic-restorability của câu evidence.
- **Tuyến phụ (typology):** hình vị là tín hiệu **cú pháp** (POS/NER), không phải **relevance ngữ nghĩa** `[HuTieuBERT ACL 2026]`;
  tiếng Việt đơn lập KHÁC ngôn ngữ giàu hình thái trong *Lost in Compression* (ở đó nén phá vai nghĩa vì case-ending; VN nghèo
  hình thái nên không) → **typology TIÊN ĐOÁN feature nào giúp** = đóng góp "NLP for Linguistics".
- **Rào:** trích "Don't Touch My Diacritics" (2410.24140) — ta KHÔNG khuyên xoá dấu; chỉ chứng minh tone-AWARENESS không thêm
  giá trị cho SELECTION khi biểu diễn đã có dấu. Cẩn thận confound: mất dấu tạo đồng âm hàng loạt → dùng **"sai dấu" (swap_tone)**
  để cô lập riêng thanh điệu khỏi nhập nhằng từ vựng.

---

## 5. Đã IMPLEMENT vòng này (fix nhỏ, local, không GPU, có test)

- **`swap_tone()` trong `linguistics.py`** — toán tử "sai dấu" (derangement 5-vòng của tone mark), giữ nguyên chữ nền + vowel
  quality, đổi CHỈ thanh điệu. Bất biến chứng minh đúng: `strip_tone(swap_tone(x)) == strip_tone(x)`. Là control SẠCH hơn
  strip_tone cho minimal-pair (không thêm đồng âm). Test: `tests/test_linguistics.py::test_swap_tone_*` (3 test, pass).
  → cùng với `strip_tone` sẵn có, bộ toán tử counterfactual input-level cho §3/§4 đã đủ.

Tổng code local đã có sau 3 vòng (đều có test, không cần GPU): `span_recoverability_hard`, `find_evidence_sentence` verbatim,
`tost_equivalence`, `paired_power_analysis`, `swap_tone`. Đây là bộ công cụ để khi snapshot train xong, chạy inference là ra
số kể được câu chuyện đột phá — không phải chờ train thêm.

---

## 6. KẾ HOẠCH ĐỘT PHÁ ưu tiên (3.5 tuần, trên snapshot đóng băng)

1. **[Kịp, ~4–6 ngày, không train] LEACE amnesic + Rand + control** trên hidden states đã trích → claim nhân quả. ⭐ đòn bẩy cao nhất.
2. **[Kịp, ~5–7 ngày, inference] Minimal-pair {có/mất/sai dấu} + control ngữ nghĩa** → claim behavioral (toán tử đã có: strip_tone/swap_tone). Ưu tiên "sai dấu".
3. **[Kịp, ~1–2 ngày] Diacritic-restorability câu evidence** → claim information-theoretic. Rẻ nhất, làm trước để có tín hiệu sớm.
4. **[Kịp, viết + đã có code] Đóng gói FUSE protocol + anchor NLI-laundering** → đóng góp tích cực thứ hai + phụ.
5. **[Kịp, viết] Reframe abstract/intro về RQ ngôn ngữ học + typology (§1, §4)** → khớp theme, kéo Excitement.

**Future Work:** kernelized/phi tuyến erasure; activation-patching/causal-tracing định vị lớp; mở rộng sang ngôn ngữ thanh điệu
khác (Thái/Quan Thoại) để chứng minh protocol tổng quát; "law/frontier" chỉ gọi "characterization" (n=18 quá nhỏ).

**Rào overclaim tổng:** phát biểu scope chính xác ("cho QA extractive, các chế độ nén đã test, thông tin tuyến tính + họ phi
tuyến đã kiểm"); không nói "thanh điệu vô nghĩa"; audit contamination tường minh (vì bán điểm mạnh leak-control).
