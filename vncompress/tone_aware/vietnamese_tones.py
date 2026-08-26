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

import unicodedata
from typing import TYPE_CHECKING, List, Dict, Tuple, Optional, Callable, Sequence
from dataclasses import dataclass

if TYPE_CHECKING:
    import torch


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


def _nfc(text: str) -> str:
    """Compose to NFC before any per-character tone lookup.

    Vietnamese tone marks exist in two Unicode forms: composed (NFC, 'á' =
    U+00E1, one codepoint) and decomposed (NFD, 'á' = 'a' + U+0301, two).
    MANUAL_TONE_MAP contains only composed characters, so iterating an NFD
    string finds no tone-bearing character at all.

    That failure is silent and inverts the project's headline metric rather
    than raising. Measured on decomposed text, dropping *every* token:

        NFC: tone-bearing 4/4 -> TPR 0.0   (correct: all tone info lost)
        NFD: tone-bearing 0/4 -> TPR 1.0   (claims perfect preservation)

    `is_vietnamese()` likewise returned False for decomposed Vietnamese,
    which switches the tone-aware path off in run_benchmark.py and
    compressors/tone_aware.py without any warning. The committed datasets
    happen to be NFC, so this is latent -- until one new source is not.
    """
    return unicodedata.normalize('NFC', text)


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

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.4,
        tone_contrast: Optional[Dict[Tuple[str, str], float]] = None,
    ):
        """
        Args:
            alpha: Base importance of tone information (0-1)
            beta: Bonus for tone variety within a token (0-1)
            gamma: Amplification for tonal contrast with neighbors (0-1)
            tone_contrast: Optional tone x tone contrast matrix to use instead
                of the hand-picked default TONE_CONTRAST. Pass the output of
                `estimate_tone_contrast_matrix()` to use an embedding-derived,
                data-driven matrix instead.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tone_contrast = tone_contrast if tone_contrast is not None else TONE_CONTRAST
        self._build_lookup()

    def _build_lookup(self):
        """Build fast lookup tables for tone detection."""
        # Character -> tone name (fast dict lookup)
        self.char_to_tone: Dict[str, str] = {}
        for char, tone in MANUAL_TONE_MAP.items():
            self.char_to_tone[_nfc(char)] = tone
    
    def get_char_tone(self, char: str) -> ToneInfo:
        """Get tone information for a single character."""
        # MANUAL_TONE_MAP holds precomposed (NFC) characters only. In NFD,
        # 'á' is 'a' + U+0301 and matches nothing, so every tone silently
        # reads as ngang -- see _nfc() for why that is dangerous.
        tone = self.char_to_tone.get(_nfc(char))
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
        return [self.get_char_tone(c) for c in _nfc(text)]
    
    def get_tone_sequence(self, text: str) -> List[int]:
        """Get tone sequence as integer IDs (0=ngang, 1=huyền, ..., 5=nặng)."""
        tones = []
        for c in _nfc(text):
            info = self.get_char_tone(c)
            tones.append(info.tone_id if info.tone_id is not None else 0)
        return tones

    def compute_tone_density(self, token: str) -> float:
        """
        Compute tone density ρ(t) for a token.

        ρ(t) = (count of tone-carrying chars) / (token length)
        """
        token = _nfc(token)
        if not token:
            return 0.0
        tone_count = sum(1 for c in token if self.char_to_tone.get(c, 'ngang') != 'ngang')
        return tone_count / len(token)

    def compute_tone_variety(self, token: str) -> int:
        """Count distinct non-ngang tones in a token."""
        tones = set()
        for c in _nfc(token):
            t = self.char_to_tone.get(c, 'ngang')
            if t != 'ngang':
                tones.add(t)
        return len(tones)

    def get_dominant_tone(self, token: str) -> Optional[str]:
        """Get the most frequent non-ngang tone in a token."""
        tone_counts: Dict[str, int] = {}
        for c in _nfc(token):
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
            contrast = self.tone_contrast.get((my_tone, neighbor_tone), 0.0)
            # Symmetric lookup
            if contrast == 0.0 and my_tone != neighbor_tone:
                contrast = self.tone_contrast.get((neighbor_tone, my_tone), 0.5)
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
        # `tones_present` is what compute_tone_preservation_rate() counts, so
        # a missed normalisation here makes TPR report 1.0 on text where every
        # tone was in fact dropped. See _nfc().
        tones_present = []
        for c in _nfc(token):
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
# Tone Preservation Rate (TPR)
# ============================================================================
#
# Formal definition -- see docs/tone_preservation_rate.md for the full
# writeup (edge cases, limitations, interpretation guidance):
#
#   TPR = |{i in tone_bearing : i in retained}| / |tone_bearing|
#
#   tone_bearing = {i : tone_infos[i].tones_present is non-empty}
#
# i.e. of the original token positions that carry at least one
# non-'ngang' tone mark, what fraction survive compression by index.
# Tokens with no tone mark (ngang/level tone, punctuation, digits,
# non-Vietnamese text) are excluded from both numerator and
# denominator -- TPR only measures whether *tone-bearing* information
# survives, not overall token retention (that's `compression_ratio`).
#
# Edge case: if the input has no tone-bearing tokens at all (denominator
# = 0), TPR is defined as 1.0 by convention -- there is nothing to lose,
# so nothing was lost.
#
# This is token-level (per decoded tokenizer output, not per Vietnamese
# syllable/word): if a tokenizer ever splits a syllable's base vowel and
# its diacritic across two tokens, each half is scored independently.
# This is intentional -- it mirrors what a compressor can actually see
# and drop (token IDs), not an idealized linguistic unit it has no
# access to.

def compute_tone_preservation_rate(
    tone_infos: Sequence['TokenToneInfo'],
    retained_indices: Sequence[int],
) -> float:
    """Canonical Tone Preservation Rate over one sequence's tone analysis.

    Args:
        tone_infos: per-token TokenToneInfo, in original sequence order
            (e.g. from VietnameseToneAnalyzer.analyze_tokens()).
        retained_indices: original-sequence indices that survived
            compression (any container supporting `in`; a set is fastest
            for repeated membership checks).

    Returns:
        TPR in [0.0, 1.0]. 1.0 if there are no tone-bearing tokens (see
        module docs above for the edge-case rationale).
    """
    retained_set = retained_indices if isinstance(retained_indices, (set, frozenset)) else set(retained_indices)
    tone_bearing = [i for i, info in enumerate(tone_infos) if info.tones_present]
    if not tone_bearing:
        return 1.0
    preserved = sum(1 for i in tone_bearing if i in retained_set)
    return preserved / len(tone_bearing)


def majority_tone_baseline_rate(tone_infos: Sequence['TokenToneInfo']) -> float:
    """TPR a compressor gets 'for free' by construction, independent of
    which tokens it actually keeps -- the fraction of ALL tokens that are
    NOT tone-bearing (majority-class rate for the binary
    tone-bearing/not-tone-bearing label).

    Used as a sanity baseline: TPR alone can look inflated on
    tone-sparse text just because few tokens carry tones at all. A
    compressor's TPR should be interpreted relative to this floor, and a
    NoCompressor/keep-everything baseline always scores 1.0 (trivially,
    since nothing is dropped) -- compare both when judging whether a
    method's TPR reflects real tone-aware selection versus text that
    happened to be easy.
    """
    n = len(tone_infos)
    if n == 0:
        return 1.0
    non_tone_bearing = sum(1 for info in tone_infos if not info.tones_present)
    return non_tone_bearing / n


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

    # Composed form required: vi_chars below holds precomposed characters, so
    # decomposed Vietnamese would score 0 and read as non-Vietnamese, silently
    # disabling the tone-aware path in run_benchmark.py / compressors.
    text = _nfc(text)

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
    # Normalise first so NFC and NFD inputs give byte-identical output --
    # otherwise the ablation's stripped text depends on the source encoding.
    decomposed = unicodedata.normalize('NFD', _nfc(text))
    without_tone = ''.join(ch for ch in decomposed if ch not in TONE_MARK_TO_NAME)
    return unicodedata.normalize('NFC', without_tone)


def extract_tone_marks(text: str) -> List[str]:
    """Extract the sequence of tone marks from Vietnamese text."""
    marks = []
    for c in _nfc(text):
        tone = MANUAL_TONE_MAP.get(c, 'ngang')
        marks.append(tone)
    return marks


# ============================================================================
# Data-Driven Tone Contrast Estimation (LACC improvement B4)
# ============================================================================
#
# TONE_CONTRAST above encodes phonetic-feature guesses about how "different"
# two tones sound (e.g. ngang vs ngã = 0.9). This is a reasonable prior but
# it's hand-picked, not measured. The function below instead estimates
# contrast empirically: take minimal-pair syllable families (same consonant +
# vowel, differing only by tone — "ma"/"má"/"mà"/"mả"/"mã"/"mạ"), embed each
# variant with a trained model, and use the embedding distance between tone
# variants as the contrast signal. Tones a model actually represents as very
# different get a high contrast score; tones it treats as similar get a low
# one — grounded in what the model learned rather than a linguist's guess.

# Common Vietnamese consonant + vowel-tone-group minimal-pair families.
# Each dict maps tone_name -> syllable. Not every entry is necessarily a
# common dictionary word (Vietnamese doesn't fill every tone slot for every
# syllable), but all six variants are valid to embed and compare.
DEFAULT_SYLLABLE_FAMILIES: List[Dict[str, str]] = [
    {'ngang': 'ma', 'huyền': 'mà', 'sắc': 'má', 'hỏi': 'mả', 'ngã': 'mã', 'nặng': 'mạ'},
    {'ngang': 'la', 'huyền': 'là', 'sắc': 'lá', 'hỏi': 'lả', 'ngã': 'lã', 'nặng': 'lạ'},
    {'ngang': 'ba', 'huyền': 'bà', 'sắc': 'bá', 'hỏi': 'bả', 'ngã': 'bã', 'nặng': 'bạ'},
    {'ngang': 'ca', 'huyền': 'cà', 'sắc': 'cá', 'hỏi': 'cả', 'ngã': 'cã', 'nặng': 'cạ'},
    {'ngang': 'da', 'huyền': 'dà', 'sắc': 'dá', 'hỏi': 'dả', 'ngã': 'dã', 'nặng': 'dạ'},
    {'ngang': 'ha', 'huyền': 'hà', 'sắc': 'há', 'hỏi': 'hả', 'ngã': 'hã', 'nặng': 'hạ'},
    {'ngang': 'ta', 'huyền': 'tà', 'sắc': 'tá', 'hỏi': 'tả', 'ngã': 'tã', 'nặng': 'tạ'},
    {'ngang': 'sa', 'huyền': 'sà', 'sắc': 'sá', 'hỏi': 'sả', 'ngã': 'sã', 'nặng': 'sạ'},
    {'ngang': 'ra', 'huyền': 'rà', 'sắc': 'rá', 'hỏi': 'rả', 'ngã': 'rã', 'nặng': 'rạ'},
    {'ngang': 'xa', 'huyền': 'xà', 'sắc': 'xá', 'hỏi': 'xả', 'ngã': 'xã', 'nặng': 'xạ'},
]


def _euclidean(u: Sequence[float], v: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(u, v)) ** 0.5


def estimate_tone_contrast_matrix(
    get_embedding: Callable[[str], Sequence[float]],
    syllable_families: Optional[List[Dict[str, str]]] = None,
    distance_fn: Callable[[Sequence[float], Sequence[float]], float] = _euclidean,
) -> Dict[Tuple[str, str], float]:
    """
    Estimate a TONE_CONTRAST matrix from embedding distances instead of
    hand-picked phonetic-feature guesses.

    Args:
        get_embedding: token string -> fixed-size embedding vector. This is
            kept generic (no torch/model dependency here) so any embedding
            source works, e.g.:

                tone_analyzer_model = ...  # HF model
                def get_embedding(token):
                    ids = tokenizer.encode(token, add_special_tokens=False)
                    vecs = model.get_input_embeddings()(torch.tensor(ids))
                    return vecs.mean(dim=0).tolist()

        syllable_families: list of {tone_name: syllable} dicts (minimal
            pairs). Defaults to DEFAULT_SYLLABLE_FAMILIES.
        distance_fn: vector distance function, defaults to Euclidean.

    Returns:
        Dict[(tone_a, tone_b), float] with the same key shape as the module
        TONE_CONTRAST (only pairs with tone_a <= tone_b by TONE_NAME_TO_ID
        order are populated, diagonal = 0.0), min-max normalized to [0, 1]
        so it plugs into the same w_tone/f_contrast formulas unchanged.
    """
    families = syllable_families or DEFAULT_SYLLABLE_FAMILIES
    tone_names = list(TONE_ID_TO_NAME.values())

    pair_distances: Dict[Tuple[str, str], List[float]] = {}
    for family in families:
        embeddings = {}
        for tone in tone_names:
            syllable = family.get(tone)
            if not syllable:
                continue
            try:
                embeddings[tone] = get_embedding(syllable)
            except Exception:
                continue

        present_tones = list(embeddings.keys())
        for i, tone_a in enumerate(present_tones):
            for tone_b in present_tones[i + 1:]:
                dist = distance_fn(embeddings[tone_a], embeddings[tone_b])
                key = tuple(sorted((tone_a, tone_b), key=tone_names.index))
                pair_distances.setdefault(key, []).append(dist)

    if not pair_distances:
        # No families produced usable embeddings — fall back to the
        # hand-picked matrix rather than returning an empty one.
        return dict(TONE_CONTRAST)

    mean_distances = {k: sum(v) / len(v) for k, v in pair_distances.items()}
    lo = min(mean_distances.values())
    hi = max(mean_distances.values())
    spread = hi - lo

    result: Dict[Tuple[str, str], float] = {(t, t): 0.0 for t in tone_names}
    for key, dist in mean_distances.items():
        result[key] = (dist - lo) / spread if spread > 1e-8 else 0.5

    return result


# Singleton instance for reuse
_default_analyzer: Optional[VietnameseToneAnalyzer] = None

def get_tone_analyzer(**kwargs) -> VietnameseToneAnalyzer:
    """Get or create the default VietnameseToneAnalyzer instance."""
    global _default_analyzer
    if _default_analyzer is None or kwargs:
        _default_analyzer = VietnameseToneAnalyzer(**kwargs)
    return _default_analyzer
