# Cải tiến LACC từ Nghiên cứu Tiếng Việt

**Ngày:** 28/06/2026

---

## Tổng quan

Qua khảo sát ~44 paper arXiv + 20 repo GitHub về NLP tiếng Việt, xác định **7 cải tiến** cho thuật toán LACC hiện tại:

---

## Cải tiến 1: Tích hợp Word Segmentation (RDRsegmenter)

### Vấn đề hiện tại
LACC phân tích thanh điệu và hình thái ở mức **subword token** (BPE). Nhưng tiếng Việt là ngôn ngữ đơn lập — mỗi âm tiết là một từ riêng. Tokenizer BPE thường tách từ ghép thành nhiều subword token, làm sai lệch phân tích.

Ví dụ: `hợp_tác_xã` → BPE tokenizer → [`hợp`, `_tác`, `_xã`]
→ Mỗi subword được phân tích riêng → `hợp` bị coi là hư từ (vì "hợp" cũng là từ đơn)

### Giải pháp
Tích hợp **RDRsegmenter** (LREC 2018, Dat Quoc Nguyen) — word segmenter nhanh và chính xác cho tiếng Việt — để nhóm subword token thành từ hoàn chỉnh **trước khi** phân tích morphology.

```
Trước: [hợp, _tác, _xã] → mỗi token phân tích riêng → sai class
Sau:   [hợp_tác_xã]     → phân tích như 1 COMPOUND → đúng class
```

### Implementation
```python
# Thêm word_segmentation.py
class VietnameseWordSegmenter:
    def segment(self, text: str) -> List[str]:
        """Ghép subword tokens thành từ hoàn chỉnh"""
        # Dùng RDRsegmenter hoặc underthesea
        ...
    
    def align_tokens_to_words(
        self, subword_tokens: List[str]
    ) -> List[Tuple[int, int]]:
        """Map subword index → word span"""
        ...
```

---

## Cải tiến 2: Phân tích Thanh điệu ở mức Âm tiết (Syllable-Level)

### Vấn đề hiện tại
LACC phân tích thanh điệu ở mức **ký tự** (character-level). Nhưng tiếng Việt, thanh điệu là thuộc tính của **âm tiết** (syllable), không phải ký tự riêng lẻ.

Ví dụ: token `chuyện` → BPE tokenizer → có thể thành [`chuy`, `ện`]
→ `chuy` không có thanh điệu, `ện` có thanh nặng
→ Phân tích sai: token gốc `chuyện` có thanh nặng toàn bộ âm tiết

### Giải pháp
Phân tích thanh điệu ở mức âm tiết, dùng **bảng âm tiết tiếng Việt** (~6,500 âm tiết hợp lệ) để xác định thanh điệu chính xác.

### Implementation
```python
# Thêm syllable_tones.py
VIETNAMESE_SYLLABLES = {
    'chuyện': 'nặng',
    'chuyền': 'huyền',
    'chuyến': 'sắc',
    # ... ~6,500 entries
}

def get_syllable_tone(syllable: str) -> str:
    """Tra cứu thanh điệu ở mức âm tiết (chính xác hơn character-level)"""
    return VIETNAMESE_SYLLABLES.get(syllable.lower(), 'unknown')
```

---

## Cải tiến 3: Từ điển Mở rộng — Teencode, Từ địa phương, Từ Hán-Việt

### Vấn đề hiện tại
Từ điển hư từ hiện tại (~200 từ) chỉ cover từ phổ thông. Thiếu:
- **Teencode**: `ko` (không), `dc` (được), `vs` (với), `cx` (cũng)...
- **Từ địa phương**: `mô` (nào — Trung), `hông` (không — Nam), `chi` (gì — Trung)...
- **Từ Hán-Việt thông dụng**: `quốc_gia`, `xã_hội`, `chính_phủ`...

### Giải pháp
Mở rộng từ điển với 3 nguồn:
1. **Behitek/vietnam-sensitive-words**: ~5,000 từ nhạy cảm + teencode
2. **Duyet/vietnamese-wordlist**: ~10,000 từ tiếng Việt phổ biến nhất
3. **Zeloru/vietnamese-wordnet**: WordNet tiếng Việt

### Implementation
```python
# Mở rộng trong merge_policy.py
TEENCODE_MAP = {
    'ko': 'không', 'k': 'không', 'dc': 'được', 'đc': 'được',
    'vs': 'với', 'cx': 'cũng', 'ng': 'người', 'đc': 'được',
    'vc': 'vợ_chồng', 'nc': 'nói_chuyện', 'nt': 'nhắn_tin',
    'tk': 'tài_khoản', 'sp': 'sản_phẩm', 'đt': 'điện_thoại',
}

DIALECT_MAP = {
    # Trung
    'mô': 'nào', 'tê': 'kia', 'răng': 'sao', 'rứa': 'thế',
    'chi': 'gì', 'nờ': 'nào', 'chừ': 'giờ',
    # Nam
    'hông': 'không', 'hổng': 'không', 'hen': 'nhé',
    'ghe': 'thuyền', 'mắc': 'đắt', 'bông': 'hoa',
    'trái': 'quả', 'thơm': 'dứa', 'đậu_phộng': 'lạc',
}

SINO_VIETNAMESE_MORPHEMES = {
    # Common Sino-Vietnamese morphemes (academic/formal words)
    'quốc': 'nation', 'gia': 'family/home',
    'xã': 'society', 'hội': 'gather',
    'chính': 'main/government', 'phủ': 'government',
    'học': 'study', 'sinh': 'birth/student',
    # ... ~500 common morphemes
}
```

---

## Cải tiến 4: Trọng số Dựa trên Tần suất Từ (Frequency-Based Weighting)

### Vấn đề hiện tại
LACC đối xử mọi thực từ như nhau (cùng hệ số 1.20). Nhưng từ hiếm mang nhiều thông tin hơn từ phổ biến.

### Giải pháp
Tích hợp **danh sách tần suất từ tiếng Việt** (Duyet/vietnamese-wordlist) để điều chỉnh trọng số bảo toàn:

$$f_{\text{freq}}(t) = 1 + \delta \cdot \left(1 - \frac{\text{rank}(t)}{\text{max\_rank}}\right)$$

Từ càng hiếm → $f_{\text{freq}}$ càng cao → càng được ưu tiên giữ.

```python
# Thêm frequency_weight.py
def load_vietnamese_word_frequencies():
    """Tải danh sách tần suất từ tiếng Việt"""
    # Từ duyet/vietnamese-wordlist
    ...

def compute_frequency_weight(word: str) -> float:
    """Trọng số dựa trên độ hiếm của từ"""
    rank = word_freq_rank.get(word, max_rank)
    return 1.0 + delta * (1.0 - rank / max_rank)
```

---

## Cải tiến 5: Phát hiện và Bảo toàn Từ Nhạy cảm / Quan trọng

### Vấn đề hiện tại
LACC không phân biệt từ thông thường với từ mang thông tin nhạy cảm/quan trọng (tên riêng, số liệu, thuật ngữ chuyên ngành).

### Giải pháp
Thêm lớp `CRITICAL` — từ không bao giờ được nén:

```python
CRITICAL_PATTERNS = {
    # Numbers (statistics, dates, money)
    'numbers': r'\d+[\.,\d]*\s*(tỷ|triệu|nghìn|đồng|USD|%)',
    # Proper names (capitalized in Vietnamese text)
    'proper_names': r'[A-ZĐ][a-zà-ỹ]+(?:\s+[A-ZĐ][a-zà-ỹ]+)*',
    # Legal/technical terms
    'legal_terms': ['điều', 'khoản', 'nghị_định', 'thông_tư'],
    # Medical terms
    'medical_terms': ['bệnh', 'triệu_chứng', 'điều_trị', 'thuốc'],
}
```

Từ thuộc lớp CRITICAL → `f_class = 3.0` → **luôn được giữ**.

---

## Cải tiến 6: WordNet Semantic Similarity cho Token Merging

### Vấn đề hiện tại
LACC merge token dựa trên vị trí hoặc class, không dựa trên ngữ nghĩa.

### Giải pháp  
Dùng **Vietnamese WordNet** (zeloru/vietnamese-wordnet) để tính độ tương đồng ngữ nghĩa:

$$S_{\text{semantic}}(t_i, t_j) = \text{Wu-Palmer-Similarity}(t_i, t_j)$$

Khi merge token, ưu tiên merge các token có semantic similarity cao (cùng chủ đề) thay vì merge theo vị trí.

---

## Cải tiến 7: Context-Aware Function Word Detection

### Vấn đề hiện tại
Từ điển hư từ là tĩnh — không xét ngữ cảnh. Từ "là" có thể là hư từ (động từ liên kết) hoặc thực từ (động từ "ủi").

### Giải pháp
Dùng **PhoBERT** (encoder nhẹ, pre-trained) để kiểm tra ngữ cảnh:

```python
def is_function_word_in_context(word: str, context: str) -> bool:
    """Dùng PhoBERT để xác định từ có phải hư từ trong ngữ cảnh cụ thể"""
    # Encode context
    # Classify word's syntactic role
    ...
```

---

## Kế hoạch tích hợp

| Cải tiến | Độ khó | Impact | Thời gian | Trạng thái |
|----------|--------|--------|-----------|------------|
| 1. Word Segmentation | 🟢 Thấp | 🔴 Cao | 1 tuần | Chưa làm |
| 2. Syllable-Level Tone | 🟢 Thấp | 🔴 Cao | 3 ngày | Chưa làm |
| 3. Từ điển Mở rộng | 🟢 Thấp | 🟡 Trung bình | 2 ngày | Chưa làm |
| 4. Frequency Weighting | 🟢 Thấp | 🟡 Trung bình | 1 ngày | Chưa làm |
| 5. Critical Word Detection | 🟢 Thấp | 🟡 Trung bình | 2 ngày | Chưa làm |
| 6. Semantic Similarity | 🟡 Trung bình | 🟢 Thấp | 1 tuần | Chưa làm |
| 7. Context-Aware Detection | 🔴 Cao | 🟢 Thấp | 2 tuần | Chưa làm |

**Khuyến nghị:** Làm 1→2→3→5→4 trước (impact cao, dễ làm), 6→7 sau (phức tạp hơn).

---

## References

1. Dat Quoc Nguyen et al. "RDRsegmenter: A Fast and Accurate Vietnamese Word Segmenter." LREC 2018. [GitHub](https://github.com/datquocnguyen/RDRsegmenter)
2. Duyet Le. "vietnamese-wordlist" — 10K common Vietnamese words. [GitHub](https://github.com/duyet/vietnamese-wordlist)
3. Behitek. "vietnam-sensitive-words" — 5K sensitive + teencode words. [GitHub](https://github.com/behitek/vietnam-sensitive-words)
4. Zeloru. "vietnamese-wordnet" — Vietnamese WordNet. [GitHub](https://github.com/zeloru/vietnamese-wordnet)
5. James Vo. "Vi-Mistral-X: Building a Vietnamese Language Model with Advanced Continual Pre-training." 2024. [arxiv:2403.15470](https://arxiv.org/abs/2403.15470)
6. Ta et al. "ViDia2Std: A Parallel Corpus and Methods for Vietnamese Dialect-to-Standard Translation." AAAI 2026 Oral. [arxiv:2603.10211](https://arxiv.org/abs/2603.10211)
7. Nguyen & Nguyen. "PhoBERT: Pre-trained Language Models for Vietnamese." EMNLP 2020 Findings. [arxiv:2003.00744](https://arxiv.org/abs/2003.00744). [GitHub](https://github.com/VinAIResearch/PhoBERT)
8. Phan et al. "ViT5: Pretrained Transformer-based Models for Vietnamese." 2022. [arxiv:2205.06457](https://arxiv.org/abs/2205.06457)
9. Son VX. "VietSentiWordNet" — Vietnamese sentiment lexicon. [GitHub](https://github.com/sonvx/VietSentiWordNet)
10. Nguyen et al. "DSC2025 ViHallu Challenge: Detecting Hallucination in Vietnamese LLMs." 2026. [arxiv:2601.04711](https://arxiv.org/abs/2601.04711)

Xem thêm danh sách đầy đủ tại: [references.md](references.md)
