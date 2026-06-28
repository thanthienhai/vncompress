"""
Vietnamese Tone Utilities
==========================
Detect and analyze tones in Vietnamese text. Vietnamese has 6 tones:
  - ngang (level)     : a, e, i, o, u, y...
  - huyền (falling)   : à, è, ì, ò, ù, ỳ
  - sắc (rising)      : á, é, í, ó, ú, ý
  - hỏi (dipping)     : ả, ẻ, ỉ, ỏ, ủ, ỷ
  - ngã (broken)      : ã, ẽ, ĩ, õ, ũ, ỹ
  - nặng (heavy)       : ạ, ẹ, ị, ọ, ụ, ỵ

Tone-Aware Compression Key Insight:
  Token-level compression methods (LLMLingua, Selective Context) risk deleting
  diacritic-bearing characters that carry tone information. For Vietnamese,
  removing a tone mark can change word meaning entirely (ma ≠ má ≠ mà ≠ mả ≠ mã ≠ mạ).
  Our tone-aware compression adds a preservation factor for tone-carrying tokens.

Reference:
  - Vietnamese phonology: 6 tones with contrastive function
  - arxiv:2606.15044 "Equity with Efficiency: Tokenizers for Multilingual LLMs"
  - arxiv:2606.03618 "Cross-Lingual Token Arbitrage"
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# Vietnamese Tone Constants
# ============================================================================

# Map combining diacritics to tone names
TONE_MARK_TO_NAME = {
    '\u0300': 'huyền',  # combining grave accent
    '\u0301': 'sắc',     # combining acute accent
    '\u0309': 'hỏi',     # combining hook above
    '\u0303': 'ngã',     # combining tilde
    '\u0323': 'nặng',    # combining dot below
}

TONE_NAME_TO_MARK = {v: k for k, v in TONE_MARK_TO_NAME.items()}

# Precomposed vowel+tone combinations (common)
PRECOMPOSED_TONES: Dict[str, str] = {}
for base_vowel in 'aeiouyAEIOUY':
    for base_d in ['', '\u0306', '\u0302', '\u031B']:  # breve, circumflex, horn
        base_char = unicodedata.normalize('NFC', base_vowel + base_d)
        for tone_mark, tone_name in TONE_MARK_TO_NAME.items():
            combined = unicodedata.normalize('NFC', base_char + tone_mark)
            PRECOMPOSED_TONES[combined] = tone_name

# Extended: manually add common Vietnamese vowels with tones
MANUAL_TONE_MAP = {
    'a': 'ngang', 'à': 'huyền', 'á': 'sắc', 'ả': 'hỏi', 'ã': 'ngã', 'ạ': 'nặng',
    'ă': 'ngang', 'ằ': 'huyền', 'ắ': 'sắc', 'ẳ': 'hỏi', 'ẵ': 'ngã', 'ặ': 'nặng',
    'â': 'ngang', 'ầ': 'huyền', 'ấ': 'sắc', 'ẩ': 'hỏi', 'ẫ': 'ngã', 'ậ': 'nặng',
    'e': 'ngang', 'è': 'huyền', 'é': 'sắc', 'ẻ': 'hỏi', 'ẽ': 'ngã', 'ẹ': 'nặng',
    'ê': 'ngang', 'ề': 'huyền', 'ế': 'sắc', 'ể': 'hỏi', 'ễ': 'ngã', 'ệ': 'nặng',
    'i': 'ngang', 'ì': 'huyền', 'í': 'sắc', 'ỉ': 'hỏi', 'ĩ': 'ngã', 'ị': 'nặng',
    'o': 'ngang', 'ò': 'huyền', 'ó': 'sắc', 'ỏ': 'hỏi', 'õ': 'ngã', 'ọ': 'nặng',
    'ô': 'ngang', 'ồ': 'huyền', 'ố': 'sắc', 'ổ': 'hỏi', 'ỗ': 'ngã', 'ộ': 'nặng',
    'ơ': 'ngang', 'ờ': 'huyền', 'ớ': 'sắc', 'ở': 'hỏi', 'ỡ': 'ngã', 'ợ': 'nặng',
    'u': 'ngang', 'ù': 'huyền', 'ú': 'sắc', 'ủ': 'hỏi', 'ũ': 'ngã', 'ụ': 'nặng',
    'ư': 'ngang', 'ừ': 'huyền', 'ứ': 'sắc', 'ử': 'hỏi', 'ữ': 'ngã', 'ự': 'nặng',
    'y': 'ngang', 'ỳ': 'huyền', 'ý': 'sắc', 'ỷ': 'hỏi', 'ỹ': 'ngã', 'ỵ': 'nặng',
    # Uppercase
    'A': 'ngang', 'À': 'huyền', 'Á': 'sắc', 'Ả': 'hỏi', 'Ã': 'ngã', 'Ạ': 'nặng',
    'Ă': 'ngang', 'Ằ': 'huyền', 'Ắ': 'sắc', 'Ẳ': 'hỏi', 'Ẵ': 'ngã', 'Ặ': 'nặng',
    'Â': 'ngang', 'Ầ': 'huyền', 'Ấ': 'sắc', 'Ẩ': 'hỏi', 'Ẫ': 'ngã', 'Ậ': 'nặng',
    'E': 'ngang', 'È': 'huyền', 'É': 'sắc', 'Ẻ': 'hỏi', 'Ẽ': 'ngã', 'Ẹ': 'nặng',
    'Ê': 'ngang', 'Ề': 'huyền', 'Ế': 'sắc', 'Ể': 'hỏi', 'Ễ': 'ngã', 'Ệ': 'nặng',
    'I': 'ngang', 'Ì': 'huyền', 'Í': 'sắc', 'Ỉ': 'hỏi', 'Ĩ': 'ngã', 'Ị': 'nặng',
    'O': 'ngang', 'Ò': 'huyền', 'Ó': 'sắc', 'Ỏ': 'hỏi', 'Õ': 'ngã', 'Ọ': 'nặng',
    'Ô': 'ngang', 'Ồ': 'huyền', 'Ố': 'sắc', 'Ổ': 'hỏi', 'Ỗ': 'ngã', 'Ộ': 'nặng',
    'Ơ': 'ngang', 'Ờ': 'huyền', 'Ớ': 'sắc', 'Ở': 'hỏi', 'Ỡ': 'ngã', 'Ợ': 'nặng',
    'U': 'ngang', 'Ù': 'huyền', 'Ú': 'sắc', 'Ủ': 'hỏi', 'Ũ': 'ngã', 'Ụ': 'nặng',
    'Ư': 'ngang', 'Ừ': 'huyền', 'Ứ': 'sắc', 'Ử': 'hỏi', 'Ữ': 'ngã', 'Ự': 'nặng',
    'Y': 'ngang', 'Ỳ': 'huyền', 'Ý': 'sắc', 'Ỷ': 'hỏi', 'Ỹ': 'ngã', 'Ỵ': 'nặng',
}

TONE_ID_TO_NAME = {
    0: 'ngang',
    1: 'huyền',
    2: 'sắc',
    3: 'hỏi',
    4: 'ngã',
    5: 'nặng',
}

TONE_NAME_TO_ID = {v: k for k, v in TONE_ID_TO_NAME.items()}

# Tone contrast matrix: how "different" two tones are
# Based on phonetic features (register, contour)
# Higher value = more acoustically distinct
TONE_CONTRAST = {
    ('ngang', 'ngang'): 0.0,   ('ngang', 'huyền'): 0.5,
    ('ngang', 'sắc'): 0.7,     ('ngang', 'hỏi'): 0.8,
    ('ngang', 'ngã'): 0.9,     ('ngang', 'nặng'): 0.6,
    ('huyền', 'huyền'): 0.0,   ('huyền', 'sắc'): 0.9,
    ('huyền', 'hỏi'): 0.6,     ('huyền', 'ngã'): 0.8,
    ('huyền', 'nặng'): 0.4,    ('sắc', 'sắc'): 0.0,
    ('sắc', 'hỏi'): 0.7,       ('sắc', 'ngã'): 0.4,
    ('sắc', 'nặng'): 0.8,      ('hỏi', 'hỏi'): 0.0,
    ('hỏi', 'ngã'): 0.7,       ('hỏi', 'nặng'): 0.8,
    ('ngã', 'ngã'): 0.0,       ('ngã', 'nặng'): 0.9,
    ('nặng', 'nặng'): 0.0,
}

@dataclass
class ToneInfo:
    """Information about tone in a character or token."""
    has_tone: bool
    tone_name: Optional[str] = None
    tone_id: Optional[int] = None
    base_char: Optional[str] = None

@dataclass
class TokenToneInfo:
    """Per-token tone analysis result."""
    token: str
    token_id: int
    tones_present: List[str]
    dominant_tone: Optional[str]
    tone_density: float          # fraction of characters carrying tone
    tone_variety: int            # number of distinct tones in token
    preservation_weight: float   # computed weight for compression scoring


class VietnameseToneAnalyzer:
    """
    Analyze Vietnamese tones in text and tokens.

    Mathematical Foundation:
    ------------------------
    For a token t consisting of characters c_1, c_2, ..., c_n:

    Tone density ρ(t) = (1/n) × Σ_i I[c_i has tone mark]

    Tone variety ν(t) = |unique({tone(c_i) : I[c_i has tone mark]})|

    Preservation weight w_tone(t):
      w_tone(t) = 1.0 + α × ρ(t) × (1 + β × ν(t))
      where α controls base tone importance, β controls variety bonus

    Contrast factor f_contrast(t, neighbors):
      f_contrast(t) = 1 + γ × mean_{n ∈ neighbors} ToneContrast(tone(t), tone(n))
      where γ amplifies contrast importance

    The final tone-aware score multiplier for compression:
      s_tone(t) = w_tone(t) × f_contrast(t, context)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.4):
        """
        Args:
            alpha: Base importance of tone information (0-1)
            beta: Bonus for tone variety within a token (0-1)
            gamma: Amplification for tonal contrast with neighbors (0-1)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._build_lookup()

    def _build_lookup(self):
        """Build fast lookup tables for tone detection."""
        # Character -> tone name (fast dict lookup)
        self.char_to_tone: Dict[str, str] = {}
        for char, tone in MANUAL_TONE_MAP.items():
            self.char_to_tone[char] = tone
    
    def get_char_tone(self, char: str) -> ToneInfo:
        """Get tone information for a single character."""
        tone = self.char_to_tone.get(char)
        if tone and tone != 'ngang':
            return ToneInfo(
                has_tone=True,
                tone_name=tone,
                tone_id=TONE_NAME_TO_ID.get(tone, 0)
            )
        elif tone == 'ngang':
            return ToneInfo(has_tone=False, tone_name='ngang', tone_id=0)
        else:
            return ToneInfo(has_tone=False)

    def detect_tones(self, text: str) -> List[ToneInfo]:
        """Detect tones for all characters in a text string."""
        return [self.get_char_tone(c) for c in text]
    
    def get_tone_sequence(self, text: str) -> List[int]:
        """Get tone sequence as integer IDs (0=ngang, 1=huyền, ..., 5=nặng)."""
        tones = []
        for c in text:
            info = self.get_char_tone(c)
            tones.append(info.tone_id if info.tone_id is not None else 0)
        return tones

    def compute_tone_density(self, token: str) -> float:
        """
        Compute tone density ρ(t) for a token.

        ρ(t) = (count of tone-carrying chars) / (token length)
        """
        if not token:
            return 0.0
        tone_count = sum(1 for c in token if self.char_to_tone.get(c, 'ngang') != 'ngang')
        return tone_count / len(token)

    def compute_tone_variety(self, token: str) -> int:
        """Count distinct non-ngang tones in a token."""
        tones = set()
        for c in token:
            t = self.char_to_tone.get(c, 'ngang')
            if t != 'ngang':
                tones.add(t)
        return len(tones)

    def get_dominant_tone(self, token: str) -> Optional[str]:
        """Get the most frequent non-ngang tone in a token."""
        tone_counts: Dict[str, int] = {}
        for c in token:
            t = self.char_to_tone.get(c, 'ngang')
            if t != 'ngang':
                tone_counts[t] = tone_counts.get(t, 0) + 1
        if not tone_counts:
            return 'ngang'
        return max(tone_counts, key=tone_counts.get)

    def compute_preservation_weight(self, token: str) -> float:
        """
        Compute tone preservation weight w_tone(t).

        Formula:
          w_tone(t) = 1.0 + α × ρ(t) × (1 + β × ν(t) / 6)

        where:
          ρ(t) = tone density
          ν(t) = tone variety (number of distinct tones)
          α = base importance (default 0.5)
          β = variety bonus (default 0.3)
          6 = max possible tone varieties

        Returns weight in range [1.0, 1.0 + α × (1 + β)]
        """
        if not token:
            return 1.0
        rho = self.compute_tone_density(token)
        nu = self.compute_tone_variety(token)
        return 1.0 + self.alpha * rho * (1.0 + self.beta * nu / 6.0)

    def compute_contrast_factor(
        self,
        token: str,
        neighbor_tokens: List[str],
    ) -> float:
        """
        Compute tonal contrast factor f_contrast(t).

        Formula:
          f_contrast(t) = 1 + γ × mean_{n ∈ neighbors} ToneContrast(tone(t), tone(n))

        Higher when token's tone differs from neighbors — these tokens are
        more important because tone changes may signal semantic boundaries.

        Args:
            token: The target token
            neighbor_tokens: Nearby tokens (window of ±2 typically)

        Returns:
            Contrast factor >= 1.0
        """
        if not neighbor_tokens:
            return 1.0
        
        my_tone = self.get_dominant_tone(token) or 'ngang'
        contrasts = []
        
        for neighbor in neighbor_tokens:
            neighbor_tone = self.get_dominant_tone(neighbor) or 'ngang'
            contrast = TONE_CONTRAST.get((my_tone, neighbor_tone), 0.0)
            # Symmetric lookup
            if contrast == 0.0 and my_tone != neighbor_tone:
                contrast = TONE_CONTRAST.get((neighbor_tone, my_tone), 0.5)
            contrasts.append(contrast)
        
        mean_contrast = sum(contrasts) / len(contrasts)
        return 1.0 + self.gamma * mean_contrast

    def analyze_token(
        self,
        token: str,
        token_id: int,
        neighbor_tokens: Optional[List[str]] = None,
    ) -> TokenToneInfo:
        """
        Full tone analysis for a single token.

        Returns TokenToneInfo with computed preservation_weight that can be used
        as a multiplier in compression scoring.

        The preservation_weight combines:
          - Tone density: how many characters carry tones
          - Tone variety: how many distinct tones in token
          - Contrast: how different from neighbor tokens' tones
        """
        tones_present = []
        for c in token:
            t = self.char_to_tone.get(c, 'ngang')
            if t != 'ngang':
                tones_present.append(t)
        
        dominant = self.get_dominant_tone(token)
        density = self.compute_tone_density(token)
        variety = self.compute_tone_variety(token)
        
        w_base = self.compute_preservation_weight(token)
        
        neighbors = neighbor_tokens or []
        f_contrast = self.compute_contrast_factor(token, neighbors)
        
        preservation_weight = w_base * f_contrast
        
        return TokenToneInfo(
            token=token,
            token_id=token_id,
            tones_present=tones_present,
            dominant_tone=dominant,
            tone_density=density,
            tone_variety=variety,
            preservation_weight=preservation_weight,
        )

    def analyze_tokens(
        self,
        tokens: List[str],
        window_size: int = 2,
    ) -> List[TokenToneInfo]:
        """
        Analyze tones for a list of tokens with context window.

        For each token at position i, neighbors are tokens in [i-w, i+w] excluding i.

        Args:
            tokens: List of token strings
            window_size: Half-window size for contrast computation

        Returns:
            List of TokenToneInfo for each token
        """
        results = []
        n = len(tokens)
        
        for i, token in enumerate(tokens):
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            neighbors = [tokens[j] for j in range(start, end) if j != i]
            
            info = self.analyze_token(token, i, neighbors)
            results.append(info)
        
        return results

    def build_tone_embedding_weights(self, embed_dim: int = 64) -> 'torch.Tensor':
        """
        Build learnable tone embedding lookup table.

        Creates a 7 × embed_dim matrix:
          Row 0: no-tone / ngang
          Rows 1-6: huyền, sắc, hỏi, ngã, nặng, unknown

        These can be concatenated with word embeddings to provide tone information
        to the compression model during training.
        """
        import torch
        return torch.randn(7, embed_dim) * 0.02


# ============================================================================
# Utility Functions
# ============================================================================

def is_vietnamese(text: str, threshold: float = 0.10) -> bool:
    """
    Quick heuristic to check if text is Vietnamese.
    Based on presence of Vietnamese-specific characters.

    Returns True if ratio of Vietnamese-specific chars exceeds threshold.
    """
    if not text:
        return False
    
    vi_chars = set('àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ'
                    'ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ'
                    'ăâêôơưĂÂÊÔƠƯđĐ')
    
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False
    
    vi_count = sum(1 for c in alpha_chars if c in vi_chars)
    return (vi_count / len(alpha_chars)) >= threshold


def strip_tone(text: str) -> str:
    """
    Remove tone marks from Vietnamese text.
    Useful for ablation studies: comparing compression with/without tone info.
    """
    result = []
    for c in text:
        # Decompose character
        decomposed = unicodedata.normalize('NFD', c)
        # Remove tone marks (combining diacritics)
        without_tone = ''.join(ch for ch in decomposed if ch not in TONE_MARK_TO_NAME)
        result.append(unicodedata.normalize('NFC', without_tone))
    return ''.join(result)


def extract_tone_marks(text: str) -> List[str]:
    """Extract the sequence of tone marks from Vietnamese text."""
    marks = []
    for c in text:
        tone = MANUAL_TONE_MAP.get(c, 'ngang')
        marks.append(tone)
    return marks


# Singleton instance for reuse
_default_analyzer: Optional[VietnameseToneAnalyzer] = None

def get_tone_analyzer(**kwargs) -> VietnameseToneAnalyzer:
    """Get or create the default VietnameseToneAnalyzer instance."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = VietnameseToneAnalyzer(**kwargs)
    return _default_analyzer
