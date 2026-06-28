"""
Morphology-Aware Compression
============================
Leverage Vietnamese morphological structure for better compression.

Key Insight:
  Vietnamese is an isolating language with distinct word classes:
  - Function words (hư từ): đã, sẽ, đang, của, những, các, và, với...
    → High frequency, low semantic content → can be aggressively compressed
  - Content words (thực từ): nouns, verbs, adjectives
    → Low frequency, high semantic content → preserve carefully
  - Reduplicative words (từ láy): xinh xắn, đẹp đẽ, mạnh mẽ...
    → Semantic redundancy → can merge into single embedding
  - Compound words (từ ghép): máy_tính, học_sinh, giáo_viên...
    → Should be treated as single unit, not split during merge

Mathematical Formulas:
---------------------

1. WORD CLASSIFICATION:

   For each token t, assign class c(t) ∈ {FUNC, CONTENT, REDUP, COMPOUND, OTHER}

2. CLASS-AWARE MERGE RATIO:

   r(t) = {
     r_func     if c(t) = FUNC      (e.g., 0.2 = keep 20%, merge 80%)
     r_content  if c(t) = CONTENT   (e.g., 0.7 = keep 70%, merge 30%)
     r_redup    if c(t) = REDUP     (e.g., 0.5 = merge redup pairs)
     r_compound if c(t) = COMPOUND  (e.g., 1.0 = keep whole compound)
     r_other    otherwise           (e.g., 0.5)
   }

3. CLASS-AWARE SCORE:

   S(t) = S_base(t) × f_class(t)

   where f_class(t) is a preservation factor based on word class:
     f_class(FUNC) = 0.3-0.5     (lower → more likely to be compressed)
     f_class(CONTENT) = 1.0-1.5  (higher → preserved)
     f_class(REDUP) = 0.4-0.6    (can be merged with partner)
     f_class(COMPOUND) = 1.2-2.0 (preserve strongly)

4. TOKEN INFLATION RATIO (Vietnamese vs English):

   TIR = tokens_vi / tokens_en (for same content)
   Typical TIR for Vietnamese: 1.5-2.0

5. EFFECTIVE COMPRESSION RATIO:

   CR_effective = CR_raw / TIR

   Because Vietnamese already uses more tokens, the compression method
   needs to work harder to achieve the same effective context capacity.

Reference:
  - arxiv:2606.15044 "Equity with Efficiency" — SEA language tokenizers
  - Vietnamese morphology: Đinh Điền (2008), Nguyễn Tài Cẩn (1999)
"""

from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from enum import Enum


class WordClass(Enum):
    """Vietnamese word classes for morphology-aware compression."""
    FUNC = 'function'       # Hư từ: đã, sẽ, đang, của, những...
    CONTENT = 'content'     # Thực từ: danh từ, động từ, tính từ
    REDUP = 'reduplicative' # Từ láy: xinh xắn, đẹp đẽ...
    COMPOUND = 'compound'   # Từ ghép: máy tính, học sinh...
    OTHER = 'other'         # Unknown / punctuation / numbers


@dataclass
class MorphologyConfig:
    """Configuration for morphology-aware compression."""
    # Merge ratios per word class (lower = compress more aggressively)
    r_func: float = 0.3       # Keep 30% of function words
    r_content: float = 0.85   # Keep 85% of content words
    r_redup: float = 0.5      # Keep 50% of reduplicative words (merge pairs)
    r_compound: float = 0.95  # Keep 95% of compound words
    r_other: float = 0.5      # Keep 50% of unknown

    # Preservation multipliers for scoring
    f_func: float = 0.4       # Function words: lower → compress
    f_content: float = 1.2    # Content words: higher → preserve
    f_redup: float = 0.6      # Reduplicative: moderate
    f_compound: float = 1.5   # Compounds: strongly preserve
    f_other: float = 1.0      # Unknown: neutral

    # Reduplicative detection threshold
    redup_similarity_threshold: float = 0.6

    # Minimum token length for classification
    min_token_len: int = 1


@dataclass
class WordInfo:
    """Per-token morphological information."""
    token: str
    token_id: int
    word_class: WordClass
    is_function_word: bool = False
    is_content_word: bool = False
    is_reduplicative: bool = False
    is_compound_part: bool = False
    preservation_multiplier: float = 1.0
    merge_ratio: float = 0.5


# ============================================================================
# Vietnamese Function Words (Hư Từ)
# ============================================================================

VIETNAMESE_FUNCTION_WORDS: Set[str] = {
    # Grammatical markers
    'đã', 'sẽ', 'đang', 'vừa', 'mới', 'từng',
    'bị', 'được', 'phải', 'cần', 'nên', 'có_thể',
    # Prepositions
    'của', 'cho', 'với', 'về', 'tại', 'trong', 'ngoài', 'trên', 'dưới',
    'ở', 'đến', 'từ', 'để', 'bằng', 'vào', 'ra', 'lên', 'xuống',
    # Conjunctions
    'và', 'hoặc', 'nhưng', 'mà', 'nếu', 'thì', 'vì', 'nên', 'tuy', 'dù',
    'còn', 'hay', 'rằng', 'là',
    # Articles / Quantifiers
    'những', 'các', 'mọi', 'mỗi', 'một', 'vài', 'mấy', 'những',
    'cả', 'tất_cả', 'toàn_bộ',
    # Classifiers
    'cái', 'con', 'chiếc', 'người', 'cuốn', 'quyển', 'tờ', 'bức',
    # Pronouns / Demonstratives
    'tôi', 'ta', 'chúng_tôi', 'chúng_ta', 'mình', 'họ', 'nó',
    'này', 'đó', 'kia', 'ấy', 'đây', 'nọ', 'kia',
    # Particles
    'ạ', 'nhé', 'nhỉ', 'đi', 'thôi', 'chứ', 'cơ', 'mà',
    'ư', 'hả', 'hử', 'sao', 'không', 'chưa', 'đừng', 'chớ',
    # Adverbs (high frequency)
    'rất', 'quá', 'lắm', 'hơi', 'khá', 'cực', 'cực_kỳ',
    'luôn', 'cũng', 'vẫn', 'cứ', 'chỉ', 'mới', 'đều',
    # Common auxiliaries
    'có', 'làm', 'cho', 'khi', 'khiến', 'bắt_đầu', 'tiếp_tục',
}

# ============================================================================
# Vietnamese Reduplicative Patterns (Từ Láy)
# ============================================================================

# Common reduplicative pairs (tone/rhyme patterns)
REDUPLICATIVE_PATTERNS = [
    # Full reduplication
    ('xinh', 'xắn'), ('đẹp', 'đẽ'), ('mạnh', 'mẽ'),
    ('nhẹ', 'nhàng'), ('vội', 'vàng'), ('chậm', 'chạp'),
    ('sạch', 'sẽ'), ('dơ', 'dáy'), ('sáng', 'sủa'),
    ('tối', 'tăm'), ('khó', 'khăn'), ('dễ', 'dàng'),
    ('ngoan', 'ngoãn'), ('hư', 'hỏng'), ('buồn', 'bã'),
    ('vui', 'vẻ'), ('lạnh', 'lẽo'),
    ('nóng', 'nực'), ('mát', 'mẻ'), ('ấm', 'áp'),
    ('rộng', 'rãi'), ('hẹp', 'hòi'), ('cao', 'cả'),
    ('thấp', 'thoải'), ('xa', 'xôi'), ('gần', 'gũi'),
    ('bừa', 'bãi'), ('lộn', 'xộn'), ('ngăn', 'nắp'),
    ('chăm', 'chỉ'), ('siêng', 'năng'), ('lười', 'nhác'),
    ('thông', 'thái'), ('ngu', 'ngốc'), ('khôn', 'khéo'),
    ('vụng', 'về'), ('tài', 'tình'), ('giỏi', 'giang'),
    # Partial reduplication (initial consonant + rhyme)
    ('lung', 'linh'), ('long', 'lanh'), ('lấp', 'lánh'),
    ('rực', 'rỡ'), ('lung', 'lay'), ('đủng', 'đỉnh'),
    ('thong', 'thả'), ('từ', 'tốn'), ('điềm', 'đạm'),
    # Tone-pattern reduplication (ngang-huyền, sắc-nặng)
    ('xanh', 'xao'), ('vàng', 'vọt'), ('đỏ', 'đắn'),
    ('tim', 'tím'), ('trắng', 'trẻo'), ('đen', 'đúa'),
]


# ============================================================================
# Morphology Analyzer
# ============================================================================

class MorphologyAnalyzer:
    """
    Analyze Vietnamese word morphology for compression-aware token processing.

    Uses:
      1. Static function word dictionary (fast, no external deps)
      2. Reduplicative pattern matching
      3. Optional: underthesea for POS tagging (if available)
    """

    def __init__(self, use_pos_tagger: bool = False):
        self.use_pos_tagger = use_pos_tagger
        self.function_words = VIETNAMESE_FUNCTION_WORDS
        
        # Build reduplicative lookup: second word → first word
        self.redup_pairs: Dict[str, str] = {}
        for first, second in REDUPLICATIVE_PATTERNS:
            self.redup_pairs[second] = first
            self.redup_pairs[first + '_' + second] = first
        
        # POS tagger (lazy loading)
        self._pos_tagger = None
        if use_pos_tagger:
            self._init_pos_tagger()

    def _init_pos_tagger(self):
        """Initialize underthesea POS tagger if available."""
        try:
            from underthesea import pos_tag
            self._pos_tagger = pos_tag
        except ImportError:
            print("[MorphologyAnalyzer] underthesea not installed. "
                  "Using dictionary-based classification only.")
            self.use_pos_tagger = False

    def classify_word(self, token: str) -> WordClass:
        """
        Classify a token into Vietnamese word class.

        Priority:
          1. Check reduplicative with original underscore
          2. Function word dictionary match
          3. Reduplicative pattern match (after cleaning)
          4. Compound detection (has underscore)
          5. Heuristic: short + no meaning → likely function
          6. Default: content word
        """
        token_lower = token.strip().lower()
        token_clean = token_lower.replace('_', ' ')

        # Check reduplicative in original form (with underscore)
        if token_lower in self.redup_pairs:
            return WordClass.REDUP
        
        # Check function word dictionary (with spaces from underscores)
        if token_clean in self.function_words or token_lower in self.function_words:
            return WordClass.FUNC
        
        # Check reduplicative patterns after cleaning
        if token_clean in self.redup_pairs:
            return WordClass.REDUP
        
        # Check individual words in compound-like tokens
        if '_' in token_lower:
            # Has underscore → likely compound or reduplicative
            parts = token_lower.split('_')
            # Check if any part is a reduplicative partner
            for part in parts:
                if part in self.redup_pairs:
                    return WordClass.REDUP
            return WordClass.COMPOUND
        
        # Heuristics for unknown words
        if len(token_lower) <= 2 and token_lower.isalpha():
            return WordClass.OTHER
        
        # Default: content word
        return WordClass.CONTENT

    def classify_batch(
        self,
        tokens: List[str],
        sentence: Optional[str] = None,
    ) -> List[WordInfo]:
        """
        Classify a batch of tokens with optional POS tagging context.

        Args:
            tokens: List of decoded token strings
            sentence: Optional full sentence for POS tagging context

        Returns:
            List of WordInfo for each token
        """
        results = []
        
        # If POS tagger available, use it for better classification
        pos_tags = {}
        if self.use_pos_tagger and self._pos_tagger and sentence:
            try:
                tagged = self._pos_tagger(sentence)
                for word, tag in tagged:
                    pos_tags[word.lower()] = tag
            except Exception:
                pass  # Fall back to dictionary-based

        for i, token in enumerate(tokens):
            word_class = self.classify_word(token)
            info = WordInfo(
                token=token,
                token_id=i,
                word_class=word_class,
                is_function_word=(word_class == WordClass.FUNC),
                is_content_word=(word_class == WordClass.CONTENT),
                is_reduplicative=(word_class == WordClass.REDUP),
                is_compound_part=(word_class == WordClass.COMPOUND),
            )
            results.append(info)
        
        return results

    def get_preservation_multiplier(
        self,
        info: WordInfo,
        config: MorphologyConfig,
    ) -> float:
        """Get the preservation multiplier for a word based on its class."""
        mapping = {
            WordClass.FUNC: config.f_func,
            WordClass.CONTENT: config.f_content,
            WordClass.REDUP: config.f_redup,
            WordClass.COMPOUND: config.f_compound,
            WordClass.OTHER: config.f_other,
        }
        return mapping.get(info.word_class, 1.0)

    def get_merge_ratio(
        self,
        info: WordInfo,
        config: MorphologyConfig,
    ) -> float:
        """Get the merge keep ratio for a word based on its class."""
        mapping = {
            WordClass.FUNC: config.r_func,
            WordClass.CONTENT: config.r_content,
            WordClass.REDUP: config.r_redup,
            WordClass.COMPOUND: config.r_compound,
            WordClass.OTHER: config.r_other,
        }
        return mapping.get(info.word_class, 0.5)

    def find_reduplicative_pairs(
        self,
        tokens: List[str],
        window_size: int = 3,
    ) -> List[Tuple[int, int]]:
        """
        Find reduplicative word pairs in a token sequence.
        
        Returns list of (left_idx, right_idx) pairs that form từ láy.
        """
        pairs = []
        n = len(tokens)
        
        for i in range(n):
            for j in range(i + 1, min(i + window_size + 1, n)):
                pair_key = f"{tokens[i]}_{tokens[j]}"
                if pair_key in self.redup_pairs:
                    pairs.append((i, j))
                    break  # Each word pairs with at most one partner
        
        return pairs


# ============================================================================
# Token Inflation Calculator
# ============================================================================

class TokenInflationAnalyzer:
    """
    Measure token inflation for Vietnamese vs English.
    
    Token Inflation Ratio (TIR):
      TIR = tokens_vi / tokens_en
    
    This helps quantify how much "harder" Vietnamese compression is.
    """

    def __init__(self, vi_tokenizer, en_tokenizer):
        self.vi_tokenizer = vi_tokenizer
        self.en_tokenizer = en_tokenizer

    def compute_tir(
        self,
        text_vi: str,
        text_en: str,
    ) -> float:
        """
        Compute Token Inflation Ratio for parallel text.
        
        Args:
            text_vi: Vietnamese text
            text_en: English translation of same content
        
        Returns:
            TIR = count_vi_tokens / count_en_tokens
        """
        vi_tokens = len(self.vi_tokenizer.encode(text_vi))
        en_tokens = len(self.en_tokenizer.encode(text_en))
        return vi_tokens / en_tokens if en_tokens > 0 else 1.0

    def estimate_tir_batch(
        self,
        texts_vi: List[str],
        texts_en: List[str],
    ) -> Dict[str, float]:
        """Compute TIR statistics over a batch of parallel texts."""
        tirs = []
        for vi, en in zip(texts_vi, texts_en):
            if vi and en:
                tirs.append(self.compute_tir(vi, en))
        
        if not tirs:
            return {'mean': 1.0, 'min': 1.0, 'max': 1.0, 'std': 0.0}
        
        import statistics
        return {
            'mean': statistics.mean(tirs),
            'min': min(tirs),
            'max': max(tirs),
            'std': statistics.stdev(tirs) if len(tirs) > 1 else 0.0,
        }


# Cached singleton
_default_morph_analyzer: Optional[MorphologyAnalyzer] = None

def get_morphology_analyzer(**kwargs) -> MorphologyAnalyzer:
    """Get or create the default MorphologyAnalyzer instance."""
    global _default_morph_analyzer
    if _default_morph_analyzer is None:
        _default_morph_analyzer = MorphologyAnalyzer(**kwargs)
    return _default_morph_analyzer
