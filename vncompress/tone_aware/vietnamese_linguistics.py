"""
Vietnamese Word Segmentation & Syllable Analysis
==================================================
Integrates Vietnamese NLP tools to improve LACC token analysis:

1. WORD SEGMENTATION: Groups BPE subword tokens into complete Vietnamese words
   - Uses RDRsegmenter (LREC 2018) or underthesea as backend
   - Falls back to dictionary-based segmentation if no external lib

2. SYLLABLE-LEVEL TONE: Analyzes tone at syllable level (not character level)
   - Vietnamese has ~6,500 valid syllables, each with exactly one tone
   - Much more accurate than character-level analysis

3. EXTENDED DICTIONARIES: Teencode, dialect words, Sino-Vietnamese morphemes
   - Teencode: ko→không, dc→được, vs→với...
   - Dialect: mô→nào (Central), hông→không (Southern)...
   - Sino-Vietnamese: academic/formal word detection

Key improvements over current LACC:
  - Word segmentation → correct compound word classification
  - Syllable tone → correct tone for BPE-split tokens
  - Extended dict → better function word detection for informal text

References:
  - RDRsegmenter: Dat Quoc Nguyen et al., LREC 2018
  - Duyet/vietnamese-wordlist: 10K common Vietnamese words
  - Behitek/vietnam-sensitive-words: 5K sensitive + teencode words
  - ViDia2Std: AAAI 2026 Oral, Vietnamese dialect normalization
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Set


# ============================================================================
# 1. VIETNAMESE SYLLABLE DATABASE (~6,500 valid syllables with tones)
# ============================================================================

# Each Vietnamese syllable = initial_consonant + rhyme + tone
# This is a representative subset covering the most common syllables
# Full database would have ~6,500 entries

VIETNAMESE_SYLLABLE_TONES: Dict[str, str] = {}

# Generate common syllables programmatically
_INITIALS = [
    '', 'b', 'c', 'ch', 'd', 'đ', 'g', 'gh', 'gi', 'h', 'k', 'kh',
    'l', 'm', 'n', 'ng', 'ngh', 'nh', 'p', 'ph', 'qu', 'r', 's',
    't', 'th', 'tr', 'v', 'x',
]

_RHYME_BASES = [
    # Simple vowels
    'a', 'e', 'ê', 'i', 'o', 'ô', 'ơ', 'u', 'ư', 'y',
    # Diphthongs
    'ai', 'ao', 'au', 'ay', 'âu', 'ây',
    'eo', 'êu',
    'ia', 'iê', 'iu',
    'oa', 'oe', 'oi', 'oo', 'oă', 'ôi', 'ơi',
    'ua', 'uâ', 'uê', 'ui', 'uô', 'uơ', 'uy', 'ưa', 'ươi', 'ươu', 'ưu',
    # Triphthongs
    'iêu', 'oai', 'oay', 'uây', 'uya', 'uyên', 'uyêt', 'ươi',
    # Ending consonants
    'ac', 'ach', 'am', 'an', 'ang', 'anh', 'ap', 'at',
    'ăc', 'ăm', 'ăn', 'ăng', 'ăp', 'ăt',
    'âc', 'âm', 'ân', 'âng', 'âp', 'ât',
    'ec', 'em', 'en', 'eng', 'ep', 'et',
    'êch', 'êm', 'ên', 'ênh', 'êp', 'êt',
    'ich', 'iêm', 'iên', 'iêng', 'iêp', 'iêt', 'im', 'in', 'inh', 'ip', 'it',
    'oac', 'oach', 'oam', 'oan', 'oang', 'oanh', 'oap', 'oat',
    'oăc', 'oăm', 'oăn', 'oăt',
    'oc', 'om', 'on', 'ong', 'op', 'ot',
    'ôc', 'ôm', 'ôn', 'ông', 'ôp', 'ôt',
    'ơm', 'ơn',
    'uân', 'uât', 'uc', 'uch', 'um', 'un', 'ung', 'up', 'ut',
    'uôc', 'uôn', 'uông', 'uôt',
    'uyên', 'uyêt', 'uynh', 'uyt', 'uych',
    'ưc', 'ưng', 'ươc', 'ươm', 'ươn', 'ương', 'ươp', 'ươt', 'ưt',
]

# Tone marks for each rhyme base
_TONE_MARKS = {
    'ngang': '',    # no diacritic
    'huyền': '\u0300',  # combining grave
    'sắc': '\u0301',    # combining acute
    'hỏi': '\u0309',    # combining hook above
    'ngã': '\u0303',    # combining tilde
    'nặng': '\u0323',   # combining dot below
}

# Build syllable database
for initial in _INITIALS:
    for rhyme in _RHYME_BASES:
        for tone_name, tone_mark in _TONE_MARKS.items():
            syllable = initial + rhyme + tone_mark
            # Normalize to NFC
            syllable = unicodedata.normalize('NFC', syllable)
            VIETNAMESE_SYLLABLE_TONES[syllable] = tone_name

# Manual overrides for common syllables that don't follow rules
_MANUAL_SYLLABLE_TONES = {
    # Irregular cases
    'gì': 'huyền', 'bị': 'nặng', 'đã': 'ngã',
    # Common words
    'và': 'huyền', 'mà': 'huyền', 'đã': 'ngã',
    'sẽ': 'ngã', 'cũng': 'ngã', 'vẫn': 'ngã',
    'được': 'nặng', 'phải': 'hỏi', 'mới': 'sắc',
    'cũ': 'ngã', 'cả': 'hỏi', 'những': 'ngã',
    'các': 'sắc', 'mọi': 'nặng', 'mỗi': 'ngã',
    'này': 'huyền', 'đó': 'sắc', 'kia': 'ngang',
    'đây': 'ngang', 'ấy': 'sắc', 'nọ': 'nặng',
    'rất': 'sắc', 'quá': 'sắc', 'lắm': 'sắc',
    'hơi': 'ngang', 'khá': 'sắc', 'cực': 'nặng',
    'luôn': 'ngang', 'cũng': 'ngã', 'vẫn': 'ngã',
    'cứ': 'sắc', 'chỉ': 'hỏi', 'đều': 'huyền',
    # Add more common Vietnamese words
    'tôi': 'ngang', 'anh': 'ngang', 'chị': 'nặng',
    'em': 'ngang', 'mình': 'huyền', 'họ': 'nặng',
    'nó': 'sắc', 'ta': 'ngang', 'chúng': 'sắc',
    'có': 'sắc', 'không': 'ngang', 'chưa': 'ngang',
    'là': 'huyền', 'thì': 'huyền', 'nên': 'ngang',
    'vì': 'huyền', 'tại': 'nặng', 'bởi': 'hỏi',
    'cho': 'ngang', 'để': 'hỏi', 'với': 'sắc',
    'về': 'huyền', 'đến': 'sắc', 'từ': 'huyền',
    'đi': 'ngang', 'lại': 'nặng', 'ra': 'ngang',
    'vào': 'huyền', 'lên': 'ngang', 'xuống': 'sắc',
}

VIETNAMESE_SYLLABLE_TONES.update(_MANUAL_SYLLABLE_TONES)


def get_syllable_tone(syllable: str) -> Optional[str]:
    """
    Get the tone of a Vietnamese syllable.

    Vietnamese syllable structure: (C1)(w)V(C2) + tone
    Each syllable has exactly ONE tone.

    Args:
        syllable: A Vietnamese syllable string

    Returns:
        Tone name ('ngang', 'huyền', 'sắc', 'hỏi', 'ngã', 'nặng')
        or None if not a valid Vietnamese syllable
    """
    syllable = syllable.lower().strip()
    
    # Direct lookup
    if syllable in VIETNAMESE_SYLLABLE_TONES:
        return VIETNAMESE_SYLLABLE_TONES[syllable]
    
    # Try without diacritics (character-level fallback)
    decomposed = unicodedata.normalize('NFD', syllable)
    for char in decomposed:
        if char in ['\u0300', '\u0301', '\u0303', '\u0309', '\u0323']:
            tone_map = {
                '\u0300': 'huyền', '\u0301': 'sắc',
                '\u0303': 'ngã', '\u0309': 'hỏi',
                '\u0323': 'nặng',
            }
            return tone_map.get(char, 'ngang')
    
    return 'ngang'


# ============================================================================
# 2. WORD SEGMENTATION (BPE subword → Complete Vietnamese word)
# ============================================================================

class VietnameseWordSegmenter:
    """
    Groups BPE subword tokens into complete Vietnamese words.
    
    BPE tokenizers (Llama, Qwen) split Vietnamese words into subword pieces:
      'hợp_tác_xã' → ['hợp', '_tác', '_xã'] or ['hợp', 'tác', 'xã']
    
    This makes morphology analysis inaccurate because:
      - Individual subwords may look like different word classes
      - Compound words lose their identity
      - Reduplicative pairs may be split
    
    This segmenter re-groups subwords into complete words using:
      1. Vietnamese word list lookup
      2. Syllable boundary heuristics  
      3. External segmenter (RDRsegmenter/underthesea) if available
    
    Usage:
        segmenter = VietnameseWordSegmenter()
        words = segmenter.group_tokens(['hợp', '_tác', '_xã'])
        # → ['hợp_tác_xã']
    """
    
    def __init__(self, use_external: bool = True):
        self.use_external = use_external
        self._external_segmenter = None
        self._word_list: Set[str] = set()
        self._load_word_list()
        
        if use_external:
            self._init_external()
    
    def _load_word_list(self):
        """Load Vietnamese word list for dictionary-based segmentation."""
        # Common Vietnamese compound words
        compounds = [
            'hợp_tác_xã', 'máy_tính', 'điện_thoại', 'học_sinh',
            'giáo_viên', 'sinh_viên', 'nhà_trường', 'bệnh_viện',
            'sân_bay', 'nhà_ga', 'xe_buýt', 'tàu_hỏa',
            'máy_bay', 'ô_tô', 'xe_máy', 'xe_đạp',
            'công_ty', 'doanh_nghiệp', 'cửa_hàng', 'siêu_thị',
            'ngân_hàng', 'bưu_điện', 'thư_viện', 'nhà_sách',
            'công_viên', 'bảo_tàng', 'rạp_chiếu_phim', 'nhà_hát',
            'đất_nước', 'con_người', 'xã_hội', 'cộng_đồng',
            'môi_trường', 'kinh_tế', 'chính_trị', 'văn_hóa',
            'giáo_dục', 'y_tế', 'khoa_học', 'công_nghệ',
            'phát_triển', 'bảo_vệ', 'xây_dựng', 'quản_lý',
            'nghiên_cứu', 'đào_tạo', 'sản_xuất', 'kinh_doanh',
            'dịch_vụ', 'thương_mại', 'xuất_khẩu', 'nhập_khẩu',
            'đầu_tư', 'tài_chính', 'kế_toán', 'kiểm_toán',
            'luật_sư', 'bác_sĩ', 'kỹ_sư', 'kiến_trúc_sư',
            'nhà_báo', 'ca_sĩ', 'diễn_viên', 'vận_động_viên',
            'bóng_đá', 'cầu_lông', 'bơi_lội', 'điền_kinh',
            'âm_nhạc', 'hội_họa', 'điện_ảnh', 'nhiếp_ảnh',
            # Multi-word compounds
            'hợp_tác_xã_nông_nghiệp', 'ủy_ban_nhân_dân',
            'hội_đồng_nhân_dân', 'tòa_án_nhân_dân',
            'viện_kiểm_sát_nhân_dân', 'mặt_trận_tổ_quốc',
            # Reduplicative pairs as single units
            'xinh_xắn', 'đẹp_đẽ', 'mạnh_mẽ', 'nhẹ_nhàng',
            'vội_vàng', 'chậm_chạp', 'sạch_sẽ', 'dơ_dáy',
            'sáng_sủa', 'tối_tăm', 'khó_khăn', 'dễ_dàng',
            'ngoan_ngoãn', 'hư_hỏng', 'buồn_bã', 'vui_vẻ',
        ]
        self._word_list.update(compounds)
        
        # Add single words from common wordlist
        common_words = [
            'tôi', 'anh', 'chị', 'em', 'mình', 'họ', 'nó', 'ta',
            'chúng_tôi', 'chúng_ta', 'các_bạn', 'mọi_người',
            'có', 'không', 'chưa', 'đã', 'sẽ', 'đang', 'vừa', 'mới',
            'là', 'thì', 'nên', 'vì', 'tại', 'bởi', 'cho', 'để',
            'với', 'về', 'đến', 'từ', 'ở', 'trong', 'ngoài',
            'trên', 'dưới', 'trước', 'sau', 'giữa', 'bên',
            'này', 'đó', 'kia', 'đây', 'ấy', 'nọ', 'đâu',
            'rất', 'quá', 'lắm', 'hơi', 'khá', 'cực', 'cực_kỳ',
        ]
        self._word_list.update(common_words)
    
    def _init_external(self):
        """Try to load external Vietnamese word segmenter."""
        # Try underthesea first (lighter)
        try:
            from underthesea import word_tokenize
            self._external_segmenter = word_tokenize
            print("[WordSegmenter] Using underthesea for word segmentation")
        except ImportError:
            # Try pyvi
            try:
                from pyvi import ViTokenizer
                self._external_segmenter = ViTokenizer.tokenize
                print("[WordSegmenter] Using pyvi for word segmentation")
            except ImportError:
                print("[WordSegmenter] Using dictionary-based segmentation "
                      "(install underthesea or pyvi for better results)")
    
    def segment_text(self, text: str) -> List[str]:
        """Segment Vietnamese text into words (with external tool if available)."""
        if self._external_segmenter:
            result = self._external_segmenter(text)
            if isinstance(result, str):
                return result.split()
            return result
        
        # Fallback: simple space-based + common compounds
        return self._dictionary_segment(text)
    
    def _dictionary_segment(self, text: str) -> List[str]:
        """Simple dictionary-based word segmentation."""
        words = text.strip().split()
        result = []
        i = 0
        while i < len(words):
            # Try to match longest compound
            matched = False
            for length in range(min(5, len(words) - i), 0, -1):
                candidate = '_'.join(words[i:i + length]).lower()
                if candidate in self._word_list:
                    result.append(candidate)
                    i += length
                    matched = True
                    break
            
            if not matched:
                result.append(words[i].lower())
                i += 1
        
        return result
    
    def group_subword_tokens(self, tokens: List[str]) -> List[str]:
        """
        Group BPE subword tokens into complete Vietnamese words.
        
        BPE tokenizer patterns to handle:
          - Llama  : 'hợp', 'tác', 'xã'  (no prefix for continuation)
          - Qwen   : 'hợp', '_tác', '_xã' (underscore prefix)
          - Others : 'hợp', '##tác', '##xã' (## prefix)
        
        Strategy:
          1. Remove BPE prefixes (Ġ, _, ##) to get clean syllables
          2. Try to match groups against known compounds
          3. Use syllable-level heuristics as fallback
        """
        # Clean BPE artifacts
        def clean_token(t: str) -> str:
            t = t.strip()
            # Remove common BPE prefixes
            for prefix in ['\u2581', 'Ġ', '##', '_']:
                if t.startswith(prefix):
                    t = t[len(prefix):]
            return t.strip().lower()
        
        clean = [clean_token(t) for t in tokens]
        clean = [c for c in clean if c]  # Remove empty
        
        if not clean:
            return []
        
        # Greedy grouping: match longest known words
        result = []
        i = 0
        n = len(clean)
        
        while i < n:
            best_len = 1
            best_word = clean[i]
            
            # Try progressively longer combinations
            for length in range(min(5, n - i), 0, -1):
                candidate = '_'.join(clean[i:i + length])
                if candidate in self._word_list:
                    best_len = length
                    best_word = candidate
                    break
            
            result.append(best_word)
            i += best_len
        
        return result
    
    def align_to_original(
        self,
        groups: List[str],
        original_tokens: List[str],
        group_token_ids: List[List[int]] = None,
    ) -> List[Tuple[int, int]]:
        """
        Map word groups back to original token indices.
        
        Returns list of (start_idx, end_idx) for each word in groups.
        Useful for applying word-level analysis to original token sequence.
        """
        # Simple implementation: assume 1-to-1 mapping for now
        # In practice, need to track which original tokens formed each group
        spans = []
        idx = 0
        for group in groups:
            parts = group.split('_')
            start = idx
            end = idx + len(parts)
            spans.append((start, min(end, len(original_tokens))))
            idx = end
        return spans


# ============================================================================
# 3. EXTENDED DICTIONARIES
# ============================================================================

# 3a. TEENCODE / Internet Slang
TEENCODE_MAP: Dict[str, str] = {
    # Negations
    'ko': 'không', 'k': 'không', 'kh': 'không',
    'hong': 'không', 'hông': 'không', 'hổng': 'không',
    'chẳng': 'không', 'chả': 'không',
    # Affirmative/Modal
    'dc': 'được', 'đc': 'được', 'đk': 'được',
    'dx': 'được', 'ok': 'được', 'oke': 'được',
    # Pronouns/People
    'ng': 'người', 'ngta': 'người_ta', 'ah': 'anh',
    'e': 'em', 'a': 'anh', 'c': 'chị', 'm': 'mày',
    't': 'tao', 'bn': 'bạn', 'mn': 'mọi_người',
    'ae': 'anh_em', 'ace': 'anh_chị_em',
    # Conjunctions/Prepositions
    'vs': 'với', 'w': 'với', 'vs': 'với',
    'cx': 'cũng', 'cxn': 'cũng', 'cug': 'cũng',
    'vc': 'vợ_chồng', 'ck': 'chồng', 'vk': 'vợ',
    'ny': 'người_yêu', 'bn': 'bạn', 'bb': 'bạn_bè',
    # Communication
    'nc': 'nói_chuyện', 'nt': 'nhắn_tin',
    'ib': 'nhắn_tin', 'inb': 'nhắn_tin',
    'cmt': 'bình_luận', 'cm': 'bình_luận',
    'rep': 'trả_lời', 'fb': 'facebook',
    # Time
    'h': 'giờ', 'p': 'phút', 's': 'giây',
    'tn': 'tuần', 'th': 'tháng', 'n': 'năm',
    # Common abbreviations
    'tk': 'tài_khoản', 'sp': 'sản_phẩm',
    'đt': 'điện_thoại', 'dt': 'điện_thoại',
    'mt': 'máy_tính', 'lh': 'liên_hệ',
    'stt': 'số_thứ_tự', 'st': 'số_thứ_tự',
    'cl': 'chất_lượng', 'sl': 'số_lượng',
    'bh': 'bảo_hành', 'km': 'khuyến_mãi',
    # Emotions
    'vui': 'vui', 'bùn': 'buồn', 'giận': 'giận',
    'thưn': 'thương', 'thik': 'thích', 'thix': 'thích',
    'ghét': 'ghét', 'ghen': 'ghen', 'nhớ': 'nhớ',
    # Actions
    'bt': 'biết', 'bít': 'biết', 'hiu': 'hiểu',
    'nghĩ': 'nghĩ', 'nghix': 'nghĩ', 'nghj': 'nghĩ',
    'lm': 'làm', 'lam': 'làm', 'lèm': 'làm',
    # Questions
    'j': 'gì', 'chi': 'gì', 'z': 'vậy', 'zị': 'vậy',
    'sao': 'sao', 'seo': 'sao', 's': 'sao',
    # Quantifiers
    'nhìu': 'nhiều', 'nhiu': 'nhiều', 'ít': 'ít',
    'hơi': 'hơi', 'khá': 'khá', 'cực': 'cực',
    # Intensifiers
    'quá': 'quá', 'lắm': 'lắm', 'cực': 'rất',
    'siêu': 'rất', 'cực_kỳ': 'rất',
}

# 3b. DIALECT WORDS (North/Central/South variations)
DIALECT_MAP: Dict[str, Dict[str, str]] = {
    'central': {
        'mô': 'nào', 'tê': 'kia', 'răng': 'sao',
        'rứa': 'thế', 'chi': 'gì', 'nờ': 'nào',
        'chừ': 'giờ', 'mi': 'mày', 'tau': 'tao',
        'hắn': 'nó', 'bọn_hắn': 'bọn_nó',
        'eng': 'em', 'nác': 'nước', 'đọi': 'bát',
        'trốc': 'đầu', 'tru': 'trâu', 'cươi': 'sân',
        'nốc': 'uống', 'bổ': 'ngã', 'trốc_tru': 'ngu_ngốc',
    },
    'southern': {
        'hông': 'không', 'hổng': 'không', 'hen': 'nhé',
        'ghe': 'thuyền', 'mắc': 'đắt', 'bông': 'hoa',
        'trái': 'quả', 'thơm': 'dứa', 'đậu_phộng': 'lạc',
        'má': 'mẹ', 'tía': 'bố', 'ngoại': 'bà_ngoại',
        'nội': 'ông_bà_nội', 'cưng': 'yêu', 'ghê': 'nhiều',
        'dữ': 'nhiều', 'dễ_sợ': 'nhiều', 'dữ_thần': 'nhiều',
        'chén': 'ăn', 'nhậu': 'ăn_nhậu', 'dzô': 'uống',
        'xe_hơi': 'ô_tô', 'vi_tính': 'máy_tính',
        'quần_gin': 'quần_bò', 'áo_thun': 'áo_phông',
        'chả_lụ': 'chả_lụa', 'bánh_mì': 'bánh_mì',
        'nước_đá': 'đá', 'đá_lạnh': 'đá',
    },
    'northern': {
        'ngô': 'bắp', 'dứa': 'thơm', 'lợn': 'heo',
        'quả': 'trái', 'hoa_quả': 'trái_cây',
        'bát': 'chén', 'đũa': 'đũa', 'thìa': 'muỗng',
        'muôi': 'vá', 'rổ': 'rổ', 'rá': 'rá',
        'chăn': 'mền', 'gối': 'gối', 'đệm': 'nệm',
        'phích': 'bình_thủy', 'ấm': 'ấm', 'tích': 'bình_trà',
    },
}

# Merge all dialects into single lookup
ALL_DIALECT_MAP: Dict[str, str] = {}
for region, mapping in DIALECT_MAP.items():
    ALL_DIALECT_MAP.update(mapping)

# 3c. SINO-VIETNAMESE MORPHEMES (academic/formal vocabulary)
SINO_VIETNAMESE_MORPHEMES: Dict[str, str] = {
    # High-frequency Sino-Vietnamese morphemes
    'quốc': 'quốc_gia', 'gia': 'gia_đình',
    'xã': 'xã_hội', 'hội': 'hội_nghị',
    'chính': 'chính_trị', 'phủ': 'chính_phủ',
    'học': 'học_tập', 'sinh': 'sinh_viên',
    'giáo': 'giáo_dục', 'viên': 'giáo_viên',
    'công': 'công_nghiệp', 'nghiệp': 'công_nghiệp',
    'thương': 'thương_mại', 'mại': 'thương_mại',
    'nông': 'nông_nghiệp', 'lâm': 'lâm_nghiệp',
    'ngư': 'ngư_nghiệp', 'nghiệp': 'sự_nghiệp',
    'khoa': 'khoa_học', 'kỹ': 'kỹ_thuật', 'thuật': 'kỹ_thuật',
    'văn': 'văn_hóa', 'hóa': 'văn_hóa',
    'nghệ': 'nghệ_thuật', 'mỹ': 'mỹ_thuật',
    'y': 'y_tế', 'tế': 'y_tế', 'dược': 'dược_phẩm',
    'luật': 'luật_pháp', 'pháp': 'pháp_luật',
    'kinh': 'kinh_tế', 'tài': 'tài_chính', 'chính': 'tài_chính',
    'điện': 'điện_tử', 'tử': 'điện_tử',
    'cơ': 'cơ_khí', 'khí': 'cơ_khí',
    'kiến': 'kiến_trúc', 'trúc': 'kiến_trúc',
    'xây': 'xây_dựng', 'dựng': 'xây_dựng',
    'phát': 'phát_triển', 'triển': 'phát_triển',
    'bảo': 'bảo_vệ', 'vệ': 'bảo_vệ',
    'quản': 'quản_lý', 'lý': 'quản_lý',
    'nghiên': 'nghiên_cứu', 'cứu': 'nghiên_cứu',
    'đào': 'đào_tạo', 'tạo': 'đào_tạo',
    'sản': 'sản_xuất', 'xuất': 'sản_xuất',
    'doanh': 'kinh_doanh',
    'đầu': 'đầu_tư', 'tư': 'đầu_tư',
    'thông': 'thông_tin', 'tin': 'thông_tin',
    'truyền': 'truyền_thông', 'viễn': 'viễn_thông',
    'giao': 'giao_thông', 'vận': 'vận_tải', 'tải': 'vận_tải',
    'hàng': 'hàng_không', 'không': 'không_gian',
    'hải': 'hàng_hải', 'đường': 'đường_bộ',
    'thủy': 'đường_thủy', 'lợi': 'thủy_lợi',
    'thanh': 'thanh_tra', 'tra': 'kiểm_tra',
    'kiểm': 'kiểm_soát', 'soát': 'kiểm_soát',
    'thẩm': 'thẩm_định', 'định': 'đánh_giá',
    'tổ': 'tổ_chức', 'chức': 'tổ_chức',
    'đoàn': 'đoàn_thể', 'thể': 'tập_thể',
    'hợp': 'hợp_tác', 'tác': 'hợp_tác',
    'liên': 'liên_kết', 'kết': 'kết_nối',
    'thống': 'thống_nhất', 'nhất': 'thống_nhất',
    'độc': 'độc_lập', 'lập': 'độc_lập',
    'tự': 'tự_do', 'do': 'tự_do',
    'dân': 'dân_chủ', 'chủ': 'dân_chủ',
    'cộng': 'cộng_đồng', 'đồng': 'cộng_đồng',
    'hòa': 'hòa_bình', 'bình': 'hòa_bình',
}

# Words containing Sino-Vietnamese morphemes → likely formal/academic → preserve more
SINO_VIETNAMESE_COMPOUNDS: Set[str] = set()
for morph1 in SINO_VIETNAMESE_MORPHEMES:
    for morph2 in SINO_VIETNAMESE_MORPHEMES:
        if morph1 != morph2:
            compound = f"{morph1}_{morph2}"
            if len(compound) >= 6:  # Reasonable compound length
                SINO_VIETNAMESE_COMPOUNDS.add(compound)

# 3d. CRITICAL PATTERNS (must never be compressed)

CRITICAL_PATTERNS = {
    'numbers': re.compile(
        r'\d+[\.,\d]*\s*(?:tỷ|triệu|nghìn|ngàn|trăm|đồng|USD|VND|%|phần_trăm)?',
        re.IGNORECASE
    ),
    'dates': re.compile(
        r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',
    ),
    'proper_names': re.compile(
        r'[A-ZĐ][a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]+(?:\s+[A-ZĐ][a-zà-ỹ]+)*',
    ),
    'legal_refs': re.compile(
        r'(?:Điều|Khoản|Mục|Chương|Phần)\s+\d+',
        re.IGNORECASE
    ),
    'emails': re.compile(r'[\w\.-]+@[\w\.-]+\.\w+'),
    'urls': re.compile(r'https?://[^\s]+'),
}


def is_critical_pattern(token: str) -> bool:
    """Check if a token matches any critical pattern (never compress)."""
    for name, pattern in CRITICAL_PATTERNS.items():
        if pattern.search(token):
            return True
    return False


# ============================================================================
# 4. EXTENDED FUNCTION WORD DETECTION
# ============================================================================

def is_vietnamese_function_word_extended(word: str) -> bool:
    """
    Check if a word is a Vietnamese function word.
    Extended version with teencode, dialect, and context awareness.
    
    Args:
        word: A Vietnamese word (after word segmentation)
    
    Returns:
        True if the word is a function word
    """
    from vncompress.morphology.merge_policy import VIETNAMESE_FUNCTION_WORDS
    
    word_lower = word.lower().strip()
    
    # Check base dictionary
    if word_lower in VIETNAMESE_FUNCTION_WORDS:
        return True
    
    # Check teencode → standard mapping
    if word_lower in TEENCODE_MAP:
        standard = TEENCODE_MAP[word_lower]
        if standard in VIETNAMESE_FUNCTION_WORDS:
            return True
    
    # Check dialect mapping
    if word_lower in ALL_DIALECT_MAP:
        standard = ALL_DIALECT_MAP[word_lower]
        if standard in VIETNAMESE_FUNCTION_WORDS:
            return True
    
    return False


def normalize_vietnamese_word(word: str) -> str:
    """
    Normalize Vietnamese word: teencode → standard, dialect → standard.
    
    Returns the standard form if mapping exists, otherwise the original.
    """
    word_lower = word.lower().strip()
    
    if word_lower in TEENCODE_MAP:
        return TEENCODE_MAP[word_lower]
    
    if word_lower in ALL_DIALECT_MAP:
        return ALL_DIALECT_MAP[word_lower]
    
    return word_lower


# ============================================================================
# 5. COMBINED ANALYZER
# ============================================================================

class EnhancedVietnameseAnalyzer:
    """
    Combined Vietnamese linguistic analyzer for LACC v2.
    
    Integrates:
      1. Word segmentation (BPE → complete words)
      2. Syllable-level tone analysis
      3. Extended dictionaries (teencode, dialect, Sino-Vietnamese)
      4. Critical pattern detection
    
    Usage:
        analyzer = EnhancedVietnameseAnalyzer()
        words = analyzer.segment_and_analyze(tokens, raw_text)
        for word in words:
            print(f"{word.text}: tone={word.tone}, class={word.word_class}")
    """
    
    def __init__(self, use_external_segmenter: bool = True):
        self.segmenter = VietnameseWordSegmenter(use_external=use_external_segmenter)
    
    def segment_and_analyze(
        self,
        bpe_tokens: List[str],
        raw_text: Optional[str] = None,
    ) -> List[Dict]:
        """
        Full pipeline: segment BPE tokens → analyze each word.
        
        Args:
            bpe_tokens: List of BPE subword token strings
            raw_text: Original raw text (for re-segmentation if available)
        
        Returns:
            List of dicts with per-word analysis
        """
        # Step 1: Group into words
        words = self.segmenter.group_subword_tokens(bpe_tokens)
        
        # Step 2: Analyze each word
        results = []
        for word in words:
            # Normalize
            word_norm = normalize_vietnamese_word(word)
            
            # Tone (syllable level)
            tone = get_syllable_tone(word_norm)
            
            # Check critical
            is_critical = is_critical_pattern(word)
            
            # Check function word (extended)
            is_func = is_vietnamese_function_word_extended(word_norm)
            
            # Check Sino-Vietnamese (formal/academic)
            is_sino = word_norm in SINO_VIETNAMESE_COMPOUNDS
            
            # Word class
            if is_critical:
                word_class = 'CRITICAL'
            elif is_func:
                word_class = 'FUNC'
            elif is_sino:
                word_class = 'SINO'  # Sino-Vietnamese → preserve more
            else:
                word_class = 'CONTENT'
            
            results.append({
                'word': word_norm,
                'original': word,
                'tone': tone,
                'word_class': word_class,
                'is_critical': is_critical,
                'is_function': is_func,
                'is_sino_vietnamese': is_sino,
                'preservation_multiplier': self._get_preservation_multiplier(
                    word_class, is_critical
                ),
            })
        
        return results
    
    def _get_preservation_multiplier(
        self,
        word_class: str,
        is_critical: bool = False,
    ) -> float:
        """Get preservation multiplier based on word class."""
        multipliers = {
            'CRITICAL': 3.00,   # Never compress
            'FUNC': 0.40,       # Compress aggressively
            'SINO': 1.50,       # Preserve strongly (formal/academic)
            'CONTENT': 1.20,    # Preserve
            'REDUP': 0.60,      # Moderate (redundant pair)
            'COMPOUND': 1.50,   # Preserve as unit
            'OTHER': 1.00,      # Neutral
        }
        return multipliers.get(word_class, 1.0)
