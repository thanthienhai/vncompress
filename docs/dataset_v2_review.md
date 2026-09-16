# Rà soát dataset `anhalu/vncompress-vi-v2`

**Ngày:** 2026-09-15 · **Revision đã tải:** `75e1670` (lastModified 2026-09-15T08:10Z)
**Đối chiếu với:** [`docs/dataset_rebuild_spec.md`](dataset_rebuild_spec.md), [`WAVE2_HANDOFF.md`](../WAVE2_HANDOFF.md), [`research/wave2_proposals.md`](../research/wave2_proposals.md)
**Mẫu clone:** [`docs/samples/vncompress_vi_v2_18_samples.md`](samples/vncompress_vi_v2_18_samples.md) (18 mẫu)

---

## 0. Kết luận ngắn

**Chưa đúng, chưa đủ, và chưa phù hợp để train — nhưng sai ở chỗ khác với những gì dataset card tự nhận.**

Card chỉ nêu hai hạn chế về *khối lượng* (mới 5% tiến độ, chưa verify độc lập). Rà soát dữ liệu thật cho thấy ba vấn đề nặng hơn, **không tự khỏi khi sinh nốt 95% còn lại**:

1. **`gold_compression` không phải bản nén trích xuất mà là bản viết lại** — chỉ 30% câu trong gold xuất hiện nguyên văn trong context, và **21% số hàng chứa con số không có trong context** (teacher lấy từ kiến thức nội tại). Mọi arm wave 2 đều là *chọn lọc* (E1 ppl, E5 câu, E7 hạn ngạch, E6 phân loại token) — không arm nào học được từ nhãn sinh kiểu này.
2. **Cổng answerability loại nhầm ~50% dữ liệu tốt** — 295/585 hàng trong `extras/` có **đáp án người gán nằm nguyên văn trong `gold_compression`** nhưng vẫn bị loại vì token-F1 < 0.6. Bộ sống sót vì thế lệch về câu hỏi có đáp án là chuỗi ngắn, khớp mặt chữ.
3. **Tập test 1.000 dòng chỉ trải trên 14 tài liệu** (4 bài Wikipedia chiếm 534/1000). n hiệu dụng là 14, không phải 1000 — spec E1 đòi ≥300 mẫu *tách theo document*, và mọi khoảng tin cậy tính trên 1.000 dòng này sẽ hẹp giả tạo.

Ngoài ra có **một đường rò train→eval** cụ thể với VCC-Bench v2 (mục §5.3).

---

## 1. Phạm vi & phương pháp

Tải toàn bộ 4 config + `extras/` + `provenance/` từ HF, kiểm tra bằng script đọc trực tiếp jsonl (không qua datasets-server):

| File | Rows | Docs (`doc_id`) | Ghi chú |
|---|---|---|---|
| `compression.jsonl` | 973 | 143 | 143 context duy nhất, 602 query duy nhất |
| `extras/compression_unanswerable.jsonl` | 585 | — | hàng bị cổng answerability loại |
| `qa.jsonl` | 6.000 | 143 | 146 context duy nhất |
| `eval/test.jsonl` | 1.000 | **14** | 68 context duy nhất |
| `corpus.jsonl` | 72.301 | 4.010 | là **đoạn văn**, không phải tài liệu (median 103 token) |

Clone 18 mẫu trải theo ratio 2×/4×/8×, train/validation, `extras/`, `test`, `qa` → `docs/samples/vncompress_vi_v2_18_samples.md`.

---

## 2. Những chỗ spec đã được làm đúng (giữ nguyên)

Đây là phần tiến bộ thật so với v1, không nên đụng vào:

- **B1/P1 — ratio yêu cầu không còn là nhãn.** Có đủ `target_tokens`, `realized_tokens`, `realized_ratio`, `budget_compliance`; tính lại tay: **0/973 hàng vi phạm** ngưỡng 0.15. `requested_ratio` chỉ còn là trường tra cứu. ✅
- **B2/P2 — hết trùng lặp.** 0 cặp `(doc_id, gold_compression)` trùng; 973/973 `gold_compression` là duy nhất. ✅
- **B3/P7 — tách config.** `corpus` không mang nhãn task; `compression` / `qa` tách bạch; thơ đã bị loại (`split_manifest.poetry = {}`). ✅
- **B4/P6 — không còn cột rỗng.** Cột rỗng cao nhất là `numbers_preserved` 1,5% (context không có số) và `wikidata_type` 9% ở corpus. ✅
- **P5 — `is_long_document` tính đúng theo token.** 3.187/6.000 hàng `true`; ước lượng lại bằng tỉ lệ char/token của chính corpus (4,61) cho 3.196 hàng ≥ 4.000 token → lệch 9 hàng. ✅
- **Span QA hợp lệ tuyệt đối.** `context[answer_span] == answer` đúng **7.000/7.000** hàng (qa + test). ✅
- **Split theo document.** compression: 131 doc train / 12 doc validation, **overlap 0**; test tách khỏi train theo document, leak-check CLEAN. ✅
- **P11 — card khớp nguồn.** Khai 31 văn bản pháp luật → đếm được đúng 31 hàng `source=legal` trong corpus. ✅
- **Card trung thực về tiến độ** (WIP, 5%, "đừng dùng bản này để báo cáo kết quả benchmark"). ✅

---

## 3. Vấn đề chặn việc training

### 3.1 🔴 `gold_compression` là văn bản *viết lại*, không phải *nén trích xuất*

| Phép đo | Kết quả |
|---|---|
| gold là substring nguyên văn của context | **61 / 973** (6,3%) |
| câu trong gold xuất hiện nguyên văn trong context | **7.329 / 24.424 = 30,0%** |
| `extractive_ratio` (overlap unigram) | median 0,933 · min 0,497 |
| hàng có **số trong gold không có trong context** | **206 / 973 = 21,2%** |

Ví dụ kiểm chứng tận tay (doc `viquad:Phạm_Văn_Đồng`, 4×, context 4.712 ký tự): gold khẳng định *"Năm 1925, tham gia phong trào để tang Phan Châu Trinh. Năm 1926 sang Quảng Châu…"* và *"Đại hội Đảng toàn quốc lần thứ VIII (1996)"* — **cả ba con số 1925, 1926, 1996 đều không có trong context**. Đây không phải lỗi cắt context (chỉ 16/206 hàng có context chạm trần 24.000 ký tự); đây là teacher viết bằng kiến thức nội tại.

Mẫu #9 trong file clone (`François_Mitterrand`, 8×) còn rõ hơn: gold là đoạn tường thuật *dài dòng hơn nguồn* về các cuộc tuần hành 1935, văn phong giải thích ("Sự kiện này đánh dấu một bước ngoặt quan trọng…") — nội dung đã bị **khuếch trương**, chỉ ngắn hơn full context vì bỏ phần còn lại của bài. Mẫu #6 (`Albert_Einstein`, 4×) tương tự: gold là một tiểu luận về hiệu ứng quang điện, `extractive_ratio` 0,808.

**Vì sao chặn:** `realized_ratio` đo đúng, nhưng thứ được đo không phải "nén". Hệ quả trực tiếp lên wave 2:

- **E6 (encoder token-classification)** cần nhãn giữ/bỏ **mức token, căn được về context**. Với 30% câu khớp nguyên văn và 21% hàng có số ngoài context, distill nhãn từ gold sẽ sinh nhãn nhiễu hoặc không căn được.
- **E5 (chọn mức câu)** cần gold là **tập con các câu của context**. Hiện không phải.
- **E1/E7** không dùng gold để train, nhưng nếu lấy `compression.jsonl` làm tập đánh giá reference-based thì mọi method trích xuất sẽ bị phạt oan vì không bao giờ sinh được văn bản viết lại.

**Việc cần làm:** tách hai loại nhãn. Muốn train compressor trích xuất → prompt teacher **chỉ được chọn nguyên văn câu/mệnh đề**, rồi verify bằng máy `gold ⊆ sentences(context)` và loại thẳng hàng trượt. Muốn giữ nhánh abstractive → để config riêng, không trộn.

### 3.2 🔴 Cổng answerability là token-F1 ≥ 0,6, loại nhầm ~50%

Ngưỡng đo được chính xác: hàng được giữ có `answerability_f1` min = **0,600**; hàng trong `extras/` có max = **0,5977**.

Đếm trên `extras/`: **295 / 585 = 50,4%** hàng bị loại **vẫn chứa nguyên văn đáp án người gán** trong `gold_compression`. Ba ví dụ có trong file clone:

| Mẫu | Đáp án gốc | Judge trả lời | F1 | Phán quyết |
|---|---|---|---|---|
| #13 `Cuba` | `biển Caribe` | `Quần đảo Đại Antilles.` | 0,00 | loại — *gold có đủ cả hai* |
| #14 `Ả_Rập_Xê_Út` | `quân chủ chuyên chế` | `Quân chủ chuyên chế, chế độ độc tài thế tập do hoàng tộc cai trị.` | 0,42 | loại — *đúng, chỉ dài hơn* |
| #12 `Lưu_vực` | `biển Caspian, biển Aral và nhiều hồ nhỏ hơn` | `Biển Caspian` | 0,36 | loại — *đúng, chỉ ngắn hơn* |

Đồng thời cổng này **cho lọt** chiều ngược lại: 156/973 hàng được giữ **không** chứa đáp án gốc (chỉ paraphrase đủ trùng từ), trong đó 5 hàng có F1 = 1,0.

**Hệ quả:** không chỉ mất 50% năng suất sinh (đắt), mà bộ sống sót bị **lệch có hệ thống** về câu hỏi mà đáp án là chuỗi ngắn khớp mặt chữ — đúng loại câu hỏi mà một compressor query-agnostic cũng làm được, tức là làm *nhỏ đi* chính khoảng cách mà E1 (query-conditioned) sinh ra để đo.

**Việc cần làm:** đổi F1 cứng bằng judge chấm đúng/sai về ngữ nghĩa (LLM-as-judge có rubric, `contains_gold_span` làm tín hiệu phụ), rồi **chạy lại trên 585 hàng `extras/` đã có** — đó là ~295 mẫu tốt lấy lại được gần như miễn phí, bằng 1/3 bộ hiện tại.

### 3.3 🔴 Tập test: 1.000 dòng nhưng chỉ 14 tài liệu

```
Hà_Nam(TQ) 134 · Michael_Jackson 134 · Nhà_Hán 133 · Texas 133 · Iran 110 ·
Tiếng_Anh 71 · Nội_chiến_Anh 40 · Phần_mềm_giáo_dục 40 · Cholesterol 38 ·
Voyager_1 37 · Dương_cầm 36 · Lệch_lạc 35 · Nhu_cầu 30 · Montréal 29
```

`provenance/leak_check.json` tự ghi `"eval_documents": 14`. Spec E1 đòi "≥ 300 mẫu, **tách riêng theo document** khỏi train" — ý của điều kiện đó là đa dạng document, không phải 1.000 câu hỏi trên 14 bài. Bốn bài đầu đã chiếm 534/1.000 dòng.

Kèm theo: `verification_method` = `none` cho **100%** hàng (spec E1/B5 đòi người kiểm), và `gold_compression` của test là `answer-sentence (human span)` — tức **một câu duy nhất** chứa span đáp án. Đó là gold hợp lý cho *recall đáp án*, nhưng không phải "bản nén do người xác nhận là đủ để trả lời" như spec E2 mô tả, và ở ratio thấp (2×) không dùng làm reference được.

**Việc cần làm:** mở rộng lên ≥ 150–200 document (giữ ≤ 10 câu hỏi/document), rồi mới nói tới con số 1.000 dòng.

---

## 4. Vấn đề mức trung bình

### 4.1 🟡 Toàn bộ supervision nằm trên **143 tài liệu**

`qa.jsonl` 6.000 dòng và `compression.jsonl` 973 dòng đều chỉ trải trên **143 doc / 146 context**. Phân bố số hàng nén trên mỗi context: median 6, max 15. Nghĩa là khi sinh nốt 18.000 cặp như kế hoạch, ta sẽ có **18.000 cặp trên vẫn 143 tài liệu** (~126 cặp/tài liệu).

Với E4 (relevance probe) và E6 (encoder) đây là rủi ro overfit theo document rõ rệt: probe hoàn toàn có thể học "đoạn nào của bài *Nhà Minh* thường chứa đáp án" thay vì học quan hệ query↔token. Chính `research/wave2_proposals.md` §E4 đã cảnh báo phải kiểm tra "probe không chỉ học *token là danh từ riêng*" — với 143 document thì phép kiểm đó rất khó qua.

**Việc cần làm:** trước khi scale số cặp, scale **số tài liệu**. Nguồn đã có sẵn: `corpus.jsonl` có 4.010 document chưa hề được dùng cho QA/nén.

### 4.2 🟡 Một nguồn, một loại câu hỏi — không phủ được task nào của wave 2

`question_type` = `extractive` **100%**, `source` = `uit-viquad-2.0` **100%** cho cả qa/test/compression (card đã thừa nhận).

Đối chiếu spec §4 **T1** (`agent / needle / multi_turn / cross_lingual`: hoặc ≥ 300 mẫu/task, hoặc tách `extras/` có ghi chú): **cả bốn task đều vắng mặt hoàn toàn**, kể cả trong `extras/`. Hệ quả lên wave 2:

| Arm | Cần gì | v2 có? |
|---|---|---|
| `lacc_sentence` (E5) | needle — nơi kỳ vọng cách biệt lớn nhất | ❌ |
| `lacc_tone_gated` (E8) | cross_lingual để bật tone, QA/needle để tắt | ❌ (không có cross_lingual) |
| `measure_token_inflation` (E9) | cặp song ngữ VI/EN | ❌ |
| E4 probe | nhãn liên-quan đa dạng task | chỉ extractive 1-hop |

Nói cách khác: **v2 không thay thế được VCC-Bench v2** cho phần eval của wave 2; nó chỉ bổ sung nguồn *train*. Nên ghi thẳng điều này vào card (spec E4 đòi "card ghi rõ tập nào dùng để báo cáo kết quả").

### 4.3 🟡 `domain` 89% là `other` — nhãn tồn tại nhưng vô dụng

`compression`: other 888 / admin 42 / history 24 / science 19. `qa`: other 5.331/6.000. Giá trị đều nằm trong bộ kiểm soát của spec S1 nên **về hình thức là đạt**, nhưng `domain_source` cho thấy 5.167/6.000 hàng là `none` (không suy được từ đâu), chỉ 833 hàng có `uvw-category`. Không thể phân tích hay cân bằng theo domain.

### 4.4 🟡 `answer_lang` sai ở ~17% hàng

983/6.000 hàng qa và 171/1.000 hàng test gán `answer_lang = "en"`. Kiểm mẫu: đáp án `"gan"` (test `qa_uit_006281`, câu hỏi về cholesterol) bị gán `en`. Đây là langdetect chạy trên span 1–3 từ — không tin được. Spec S2/P9 đòi `*_lang` khớp nội dung; `context_lang` đúng 100% `vi`, `answer_lang` thì không.

**Việc cần làm:** với span ngắn hơn ~5 từ thì kế thừa `context_lang`, hoặc bỏ hẳn trường này (B4 cho phép loại trường không điền được tin cậy).

### 4.5 🟡 `stable_across_ratios` là trường rỗng nghĩa

Spec §3.2 đòi: nếu 2 mức ratio cho gold trùng nhau → gộp 1 hàng, ghi `stable_across_ratios=[...]`. Thực tế **973/973 hàng có `stable_across_ratios` là singleton bằng đúng `requested_ratio` của chính nó** — chưa lần nào gộp. Đúng thôi, vì văn bản viết lại thì hai mức không bao giờ trùng tuyệt đối. Trường này đang không mang thông tin; muốn nó có nghĩa thì phải so ở mức *tập câu được chọn*, không phải so chuỗi.

Liên quan: trong 134 cặp query có cả 2× lẫn 8×, tỉ lệ token của bản 8× xuất hiện trong bản 2× chỉ **median 0,79, min 0,15** — bản nén mạnh hơn không phải tập con của bản nén nhẹ hơn. Với một compressor thật (top-k theo budget) thì tính lồng nhau này là bất biến; gold hiện tại không có.

---

## 5. Vấn đề nhỏ / cần ghi chú

### 5.1 `corpus.jsonl` là index đoạn văn, card gọi là "tài liệu"

72.301 hàng nhưng chỉ **4.010 `doc_id`**; median 103 token, max 3.847 token (0 hàng ≥ 4.000 token). Card ghi "72.270 tài liệu" — sai đơn vị. Ngoài ra 2.410 hàng (3,3%) bị gán `text_lang=en`, kiểm mẫu cho thấy đó là **rác markup MediaWiki** còn sót (`value:rgb(...)`, `barset:Presidents from:1789.40 till:1797.17…`), không phải tiếng Anh. 568 hàng (0,8%) chứa dấu vết markup; 169 hàng trùng text.

### 5.2 Schema `compression` thiếu `verification_method`

Card viết "`verification_method` của phần lớn row vẫn là `none`", nhưng **`compression.jsonl` không có trường này** — nó chỉ tồn tại trong `eval/test.jsonl`. Cần thêm cho khớp card (và để stage 4 có chỗ ghi). Cũng nên ghi **tên model judge** vào `gen_config`: hiện chỉ có `base_url`, `temperature`, `prompt_version`, `max_reprompts`, nên không biết judge có phải chính GLM-5.2 hay không — nếu phải thì đó là teacher tự chấm bài mình và phải nói rõ.

### 5.3 ⚠️ Một đường rò train → VCC-Bench v2

Bài **Hà Nội** xuất hiện ở cả hai phía:
- `qa.jsonl`: 46 hàng, **toàn bộ ở split `train`**;
- `data/benchmark/vcc_bench_v2.json`: **32/414 mẫu (7,7%)**.

Đếm câu trùng nguyên văn giữa hai bên: **54 câu**. `WAVE2_HANDOFF.md` §4 đã dặn không train E4 trên vcc_bench — nhưng ô nhiễm ở đây đến từ chiều ngược lại (train trên `qa.jsonl` → eval trên VCC-Bench). Không lớn, nhưng phải loại `viquad:Hà_Nội` khỏi train trước khi chạy E4/E6, hoặc loại 32 mẫu đó khỏi VCC-Bench khi báo cáo.

### 5.4 Lệch độ dài mà card đã nêu — xác nhận và nói thêm

Card nói tỉ lệ sống sót giảm theo độ dài context. Số liệu hiện tại: context của `compression` có median 8.426 ký tự trong khi `qa` median 19.453 ký tự — bộ nén **ngắn hơn hẳn** bộ nguồn. Đáng nói hơn: trần cứng là 24.000 ký tự (~5.200 token), nên với một dự án *long-context compression* thì không có mẫu nào thực sự dài — VCC-Bench wave 1 chạy tới 12k **token**.

### 5.5 `task` vẫn dùng tên cũ `long_document_qa`

Spec §2.3 nói "đổi tên task cũ `long_document_qa`". Hiện `task ∈ {long_document_qa, short_context_extractive_qa}`, ánh xạ 1-1 với `is_long_document`. Không sai ngữ nghĩa, nhưng nếu spec muốn xoá tên cũ thì chưa xoá.

---

## 6. Đối chiếu checklist nghiệm thu (spec §7)

| # | Tiêu chí | Kết quả |
|---|---|---|
| 1 | 100% có `realized_tokens`/`realized_ratio`; ≥95% `budget_compliance` | ✅ 100% / 100% (tính lại tay: 0 vi phạm) |
| 2 | 0 hàng trùng `(doc_id, gold_compression)` | ✅ 0 |
| 3 | 100% mẫu có query đã tính `answerable_from_compression`; train chỉ gồm `true` | ⚠️ đã tính 100%, train 100% `true` — **nhưng cổng sai ~50%** (§3.2) |
| 4 | 0 hàng corpus mang nhãn task; 0 gold rỗng | ✅ |
| 5 | Không cột rỗng > 50% | ✅ (cao nhất 9%) |
| 6 | `qa` gán đúng `is_long_document`; multi-turn có đáp án riêng | ✅ / ➖ không có multi-turn |
| 7 | `domain` trong bộ kiểm soát; `*_lang` khớp nội dung | ⚠️ domain hợp lệ nhưng 89% `other`; **`answer_lang` sai ~17%** |
| 8 | Test ≥300 mẫu, người kiểm, leak CLEAN, không teacher output | ❌ 1.000 dòng / **14 document**; `verification_method=none` 100%; leak CLEAN ✅; không teacher ✅ |
| 9 | PII giả + `synthetic_pii=true` | ➖ không có needle/PII nên không áp dụng |
| 10 | Card khớp `source`, nêu rõ tập báo cáo vs sanity-check | ⚠️ source khớp; **chưa nêu tập nào dùng báo cáo**, và gọi sai "72.270 tài liệu" |

Về bàn giao (spec §8): **`report_v2.md` chưa tồn tại** (phân bố task/domain/source/lang, tỉ lệ loại theo từng bước lọc). `provenance/` có `split_manifest`, `leak_check`, `CHECKSUMS`, `batch_manifest` — nhưng **`CHECKSUMS.json` chỉ phủ 3 file** (`corpus`, `qa`, `eval/test`), thiếu đúng hai file mới sinh là `compression.jsonl` và `extras/`. Chưa có `verification_report.json`.

---

## 7. Trả lời trực tiếp câu hỏi

> **Dữ liệu đã được sinh ra đúng, đủ và phù hợp với yêu cầu dùng để training chưa?**

**Đúng?** — Hạ tầng đúng (budget, dedup, split, span, provenance). **Nội dung nhãn thì chưa**: gold là văn bản viết lại có pha kiến thức ngoài context (§3.1), và cổng lọc phán quyết sai ~50% (§3.2).

**Đủ?** — Không. 973/18.000 cặp là vấn đề nhỏ nhất; vấn đề thật là **143 tài liệu** (§4.1), **1 loại câu hỏi** (§4.2), **14 tài liệu cho test** (§3.3). Sinh nốt 95% còn lại không sửa được bất kỳ điều nào trong ba điều đó.

**Phù hợp với wave 2?** — Một phần:
- Dùng được ngay làm nguồn **weak-supervision cho E4** (relevance probe): `qa.jsonl` có `answer_span` chính xác 100%, tốt hơn `reference_answer` của VCC-Bench. Cần viết converter sang shape `{context, query, reference_answer, task}` và **loại `viquad:Hà_Nội`** (§5.3).
- **Chưa** dùng được cho **E6** (thiếu nhãn căn được về context) và **E5** (gold không phải tập con câu).
- **Không** phủ eval của **E5/E8/E9** (không có needle/cross_lingual) — vẫn phải dùng VCC-Bench v2.

---

## 8. Thứ tự việc nên làm (đề xuất)

**Trước khi sinh thêm bất kỳ cặp nào** — ba việc này đều rẻ và quyết định giá trị của 17.000 cặp còn lại:

1. **Sửa cổng answerability** (F1 cứng → judge ngữ nghĩa) rồi **chạy lại trên 585 hàng `extras/` sẵn có**. Kỳ vọng lấy lại ~295 mẫu. Rẻ nhất, hiệu quả ngay.
2. **Đổi prompt teacher sang chế độ trích xuất** cho nhánh train compressor: chỉ được chọn nguyên văn câu, verify bằng máy `gold ⊆ sentences(context)`, trượt thì loại. Giữ nhánh viết-lại ở config riêng nếu vẫn muốn.
3. **Mở rộng số tài liệu** — lấy từ 4.010 doc trong `corpus.jsonl` thay vì bơm thêm câu hỏi cho 143 bài cũ. Test set: ≥150 document, ≤10 câu/document.

**Sau đó:** thêm `verification_method` + tên model judge vào `compression`; sửa `answer_lang`; bổ sung checksum cho `compression.jsonl`/`extras/`; viết `report_v2.md`; sửa card ("72.270 đoạn văn", nêu rõ tập nào dùng báo cáo, ghi rõ v2 **không** thay thế VCC-Bench cho needle/cross-lingual).
