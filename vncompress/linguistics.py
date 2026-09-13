"""
Vietnamese linguistic logic — tone, morphology, word segmentation.
====================================================================
Everything about Vietnamese-as-a-language that LACC's scoring signals depend
on lives in this one file: tone detection/analysis, the Tone Preservation
Rate metric, morphology classification (function/content/reduplicative/
compound/Sino-Vietnamese words), word segmentation for BPE-split compounds,
and the torch-dependent tone probe used both at training time (auxiliary
loss) and inference time (LACC's trained-tone-probe signal).

Sections:
  1. Tone constants & VietnameseToneAnalyzer   (no torch)
  2. Tone Preservation Rate                     (no torch)
  3. Word segmentation & extended dictionaries  (teencode/dialect/Sino, no torch)
  4. Morphology (WordClass, MorphologyAnalyzer) (no torch)
  5. Torch-dependent tone probe / trainer       (PhonologicalConsistencyLoss, ...)

Reference papers:
  - arxiv:2606.15044 "Equity with Efficiency: Tokenizers for Multilingual LLMs"
  - arxiv:2606.03618 "Cross-Lingual Token Arbitrage"
  - Vietnamese morphology: Đinh Điền (2008), Nguyễn Tài Cẩn (1999)
  - RDRsegmenter: Dat Quoc Nguyen et al., LREC 2018
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. Tone constants & VietnameseToneAnalyzer
# ============================================================================
#
# Vietnamese has 6 tones: ngang (level), huyền (falling), sắc (rising),
# hỏi (dipping), ngã (broken), nặng (heavy). Token-level compression risks
# deleting diacritic-bearing characters that carry this tone information --
# for Vietnamese, that can change word meaning entirely
# (ma != má != mà != mả != mã != mạ). The analyzer below quantifies how much
# tone information a token/sequence carries so compression can preserve it.

TONE_MARK_TO_NAME = {
    '̀': 'huyền',  # combining grave accent
    '́': 'sắc',     # combining acute accent
    '̉': 'hỏi',     # combining hook above
    '̃': 'ngã',     # combining tilde
    '̣': 'nặng',    # combining dot below
}
TONE_NAME_TO_MARK = {v: k for k, v in TONE_MARK_TO_NAME.items()}

PRECOMPOSED_TONES: Dict[str, str] = {}
for _base_vowel in 'aeiouyAEIOUY':
    for _base_d in ['', '̆', '̂', '̛']:  # breve, circumflex, horn
        _base_char = unicodedata.normalize('NFC', _base_vowel + _base_d)
        for _tone_mark, _tone_name in TONE_MARK_TO_NAME.items():
            _combined = unicodedata.normalize('NFC', _base_char + _tone_mark)
            PRECOMPOSED_TONES[_combined] = _tone_name

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

TONE_ID_TO_NAME = {0: 'ngang', 1: 'huyền', 2: 'sắc', 3: 'hỏi', 4: 'ngã', 5: 'nặng'}
TONE_NAME_TO_ID = {v: k for k, v in TONE_ID_TO_NAME.items()}

# Tone contrast matrix: how acoustically distinct two tones are (0=identical).
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

    Vietnamese tone marks exist composed (NFC, 'á' = one codepoint) and
    decomposed (NFD, 'á' = 'a' + U+0301, two codepoints). MANUAL_TONE_MAP only
    has composed characters, so an un-normalized NFD string would silently
    read every tone as ngang, inverting the project's headline metric
    (Tone Preservation Rate) instead of raising.
    """
    return unicodedata.normalize('NFC', text)


@dataclass
class ToneInfo:
    """Tone information for a single character."""
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
    preservation_weight: float   # multiplier for compression scoring


class VietnameseToneAnalyzer:
    """
    Analyze Vietnamese tones in text and tokens.

    For a token t of characters c_1..c_n:
      tone density   rho(t) = (1/n) * sum_i [c_i has tone mark]
      tone variety   nu(t)  = |unique tones among tone-bearing c_i|
      preservation weight:
        w_tone(t) = 1.0 + alpha * rho(t) * (1 + beta * nu(t) / 6)
      contrast factor with neighbor tokens N:
        f_contrast(t) = 1 + gamma * mean_{n in N} ToneContrast(tone(t), tone(n))
      final score multiplier:
        s_tone(t) = w_tone(t) * f_contrast(t, context)
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
            alpha: base importance of tone information (0-1)
            beta: bonus for tone variety within a token (0-1)
            gamma: amplification for tonal contrast with neighbors (0-1)
            tone_contrast: optional tone x tone matrix overriding the
                hand-picked TONE_CONTRAST default -- pass the output of
                estimate_tone_contrast_matrix() for a data-driven one.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tone_contrast = tone_contrast if tone_contrast is not None else TONE_CONTRAST
        self.char_to_tone: Dict[str, str] = {_nfc(c): t for c, t in MANUAL_TONE_MAP.items()}

    def get_char_tone(self, char: str) -> ToneInfo:
        tone = self.char_to_tone.get(_nfc(char))
        if tone and tone != 'ngang':
            return ToneInfo(has_tone=True, tone_name=tone, tone_id=TONE_NAME_TO_ID.get(tone, 0))
        if tone == 'ngang':
            return ToneInfo(has_tone=False, tone_name='ngang', tone_id=0)
        return ToneInfo(has_tone=False)

    def detect_tones(self, text: str) -> List[ToneInfo]:
        return [self.get_char_tone(c) for c in _nfc(text)]

    def get_tone_sequence(self, text: str) -> List[int]:
        return [self.get_char_tone(c).tone_id or 0 for c in _nfc(text)]

    def compute_tone_density(self, token: str) -> float:
        token = _nfc(token)
        if not token:
            return 0.0
        tone_count = sum(1 for c in token if self.char_to_tone.get(c, 'ngang') != 'ngang')
        return tone_count / len(token)

    def compute_tone_variety(self, token: str) -> int:
        tones = {self.char_to_tone.get(c, 'ngang') for c in _nfc(token)}
        tones.discard('ngang')
        return len(tones)

    def get_dominant_tone(self, token: str) -> Optional[str]:
        counts: Dict[str, int] = {}
        for c in _nfc(token):
            t = self.char_to_tone.get(c, 'ngang')
            if t != 'ngang':
                counts[t] = counts.get(t, 0) + 1
        return max(counts, key=counts.get) if counts else 'ngang'

    def compute_preservation_weight(self, token: str) -> float:
        """w_tone(t) = 1.0 + alpha * rho(t) * (1 + beta * nu(t) / 6)."""
        if not token:
            return 1.0
        rho = self.compute_tone_density(token)
        nu = self.compute_tone_variety(token)
        return 1.0 + self.alpha * rho * (1.0 + self.beta * nu / 6.0)

    def compute_contrast_factor(self, token: str, neighbor_tokens: List[str]) -> float:
        """f_contrast(t) = 1 + gamma * mean_n ToneContrast(tone(t), tone(n))."""
        if not neighbor_tokens:
            return 1.0
        my_tone = self.get_dominant_tone(token) or 'ngang'
        contrasts = []
        for neighbor in neighbor_tokens:
            neighbor_tone = self.get_dominant_tone(neighbor) or 'ngang'
            contrast = self.tone_contrast.get((my_tone, neighbor_tone), 0.0)
            if contrast == 0.0 and my_tone != neighbor_tone:
                contrast = self.tone_contrast.get((neighbor_tone, my_tone), 0.5)
            contrasts.append(contrast)
        return 1.0 + self.gamma * (sum(contrasts) / len(contrasts))

    def analyze_token(
        self, token: str, token_id: int, neighbor_tokens: Optional[List[str]] = None,
    ) -> TokenToneInfo:
        # tones_present drives compute_tone_preservation_rate(); it must go
        # through _nfc() the same way get_char_tone() does (see _nfc docstring).
        tones_present = [
            t for c in _nfc(token)
            if (t := self.char_to_tone.get(c, 'ngang')) != 'ngang'
        ]
        dominant = self.get_dominant_tone(token)
        density = self.compute_tone_density(token)
        variety = self.compute_tone_variety(token)
        w_base = self.compute_preservation_weight(token)
        f_contrast = self.compute_contrast_factor(token, neighbor_tokens or [])
        return TokenToneInfo(
            token=token, token_id=token_id, tones_present=tones_present,
            dominant_tone=dominant, tone_density=density, tone_variety=variety,
            preservation_weight=w_base * f_contrast,
        )

    def analyze_tokens(self, tokens: List[str], window_size: int = 2) -> List[TokenToneInfo]:
        """Per-token analysis; neighbors for token i are tokens in [i-w, i+w]."""
        n = len(tokens)
        results = []
        for i, token in enumerate(tokens):
            start, end = max(0, i - window_size), min(n, i + window_size + 1)
            neighbors = [tokens[j] for j in range(start, end) if j != i]
            results.append(self.analyze_token(token, i, neighbors))
        return results

    def build_tone_embedding_weights(self, embed_dim: int = 64) -> torch.Tensor:
        """7 x embed_dim lookup table (row 0 = ngang/no-tone, rows 1-6 = tones)."""
        return torch.randn(7, embed_dim) * 0.02


# ============================================================================
# 2. Tone Preservation Rate (TPR)
# ============================================================================
#
#   TPR = |{i in tone_bearing : i in retained}| / |tone_bearing|
#   tone_bearing = {i : tone_infos[i].tones_present is non-empty}
#
# i.e. of the original token positions carrying at least one non-'ngang' tone
# mark, what fraction survive compression by index. Tokens with no tone mark
# (ngang, punctuation, digits, non-Vietnamese text) are excluded from both
# numerator and denominator -- TPR measures only whether tone-bearing
# information survives, not overall token retention (compression_ratio does
# that). If there are no tone-bearing tokens at all, TPR is 1.0 by convention.
#
# This is token-level (per decoded tokenizer output, not per Vietnamese
# syllable): if a tokenizer splits a syllable's base vowel and its diacritic
# across two tokens, each half is scored independently -- intentional, since
# it mirrors what a compressor can actually see and drop.


def compute_tone_preservation_rate(
    tone_infos: Sequence[TokenToneInfo],
    retained_indices: Sequence[int],
) -> float:
    """Canonical Tone Preservation Rate over one sequence's tone analysis."""
    retained_set = retained_indices if isinstance(retained_indices, (set, frozenset)) else set(retained_indices)
    tone_bearing = [i for i, info in enumerate(tone_infos) if info.tones_present]
    if not tone_bearing:
        return 1.0
    preserved = sum(1 for i in tone_bearing if i in retained_set)
    return preserved / len(tone_bearing)


def majority_tone_baseline_rate(tone_infos: Sequence[TokenToneInfo]) -> float:
    """TPR a compressor gets 'for free': the fraction of ALL tokens that are
    NOT tone-bearing. A NoCompressor/keep-everything baseline always scores
    TPR=1.0 trivially; compare a method's TPR against this floor to judge
    whether it reflects real tone-aware selection or just tone-sparse text."""
    n = len(tone_infos)
    if n == 0:
        return 1.0
    return sum(1 for info in tone_infos if not info.tones_present) / n


def is_vietnamese(text: str, threshold: float = 0.10) -> bool:
    """Heuristic: True if the ratio of Vietnamese-specific characters exceeds
    `threshold`. NFC-composes first (see _nfc) or decomposed input scores 0."""
    if not text:
        return False
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
    """Remove tone marks (for ablation: compression with/without tone info)."""
    decomposed = unicodedata.normalize('NFD', _nfc(text))
    without_tone = ''.join(ch for ch in decomposed if ch not in TONE_MARK_TO_NAME)
    return unicodedata.normalize('NFC', without_tone)


def extract_tone_marks(text: str) -> List[str]:
    """Extract the sequence of tone marks (by name) from Vietnamese text."""
    return [MANUAL_TONE_MAP.get(c, 'ngang') for c in _nfc(text)]


# Data-driven tone contrast estimation (embedding-distance based, replacing
# the hand-picked TONE_CONTRAST matrix with a measured one).
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
    """Estimate a TONE_CONTRAST matrix from embedding distances between
    minimal-pair syllables (same consonant+vowel, differing only by tone),
    instead of the hand-picked phonetic-feature guesses in TONE_CONTRAST.

    `get_embedding` is any token-string -> fixed-size-vector callable (e.g. a
    HF model's mean-pooled input embedding for the syllable's token ids), kept
    generic so this module itself has no torch/model dependency.
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
        present = list(embeddings.keys())
        for i, tone_a in enumerate(present):
            for tone_b in present[i + 1:]:
                dist = distance_fn(embeddings[tone_a], embeddings[tone_b])
                key = tuple(sorted((tone_a, tone_b), key=tone_names.index))
                pair_distances.setdefault(key, []).append(dist)

    if not pair_distances:
        return dict(TONE_CONTRAST)

    mean_distances = {k: sum(v) / len(v) for k, v in pair_distances.items()}
    lo, hi = min(mean_distances.values()), max(mean_distances.values())
    spread = hi - lo
    result: Dict[Tuple[str, str], float] = {(t, t): 0.0 for t in tone_names}
    for key, dist in mean_distances.items():
        result[key] = (dist - lo) / spread if spread > 1e-8 else 0.5
    return result


_default_analyzer: Optional[VietnameseToneAnalyzer] = None


def get_tone_analyzer(**kwargs) -> VietnameseToneAnalyzer:
    """Get or create the default VietnameseToneAnalyzer instance."""
    global _default_analyzer
    if _default_analyzer is None or kwargs:
        _default_analyzer = VietnameseToneAnalyzer(**kwargs)
    return _default_analyzer


# ============================================================================
# 3. Word segmentation & extended dictionaries
# ============================================================================
#
# BPE tokenizers (Llama, Qwen) split Vietnamese words into subword pieces
# ('hợp_tác_xã' -> ['hợp', '_tác', '_xã']), which makes per-token morphology
# classification miss compound words. VietnameseWordSegmenter re-groups
# decoded BPE tokens into known multi-syllable words so MorphologyAnalyzer can
# classify them correctly (see COMPOUND handling below).

VIETNAMESE_SYLLABLE_TONES: Dict[str, str] = {}
_INITIALS = [
    '', 'b', 'c', 'ch', 'd', 'đ', 'g', 'gh', 'gi', 'h', 'k', 'kh',
    'l', 'm', 'n', 'ng', 'ngh', 'nh', 'p', 'ph', 'qu', 'r', 's',
    't', 'th', 'tr', 'v', 'x',
]
_RHYME_BASES = [
    'a', 'e', 'ê', 'i', 'o', 'ô', 'ơ', 'u', 'ư', 'y',
    'ai', 'ao', 'au', 'ay', 'âu', 'ây', 'eo', 'êu', 'ia', 'iê', 'iu',
    'oa', 'oe', 'oi', 'oo', 'oă', 'ôi', 'ơi',
    'ua', 'uâ', 'uê', 'ui', 'uô', 'uơ', 'uy', 'ưa', 'ươi', 'ươu', 'ưu',
    'iêu', 'oai', 'oay', 'uây', 'uya', 'uyên', 'uyêt',
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
    'uôc', 'uôn', 'uông', 'uôt', 'uyên', 'uyêt', 'uynh', 'uyt', 'uych',
    'ưc', 'ưng', 'ươc', 'ươm', 'ươn', 'ương', 'ươp', 'ươt', 'ưt',
]
_TONE_MARKS = {
    'ngang': '', 'huyền': '̀', 'sắc': '́',
    'hỏi': '̉', 'ngã': '̃', 'nặng': '̣',
}
for _initial in _INITIALS:
    for _rhyme in _RHYME_BASES:
        for _tone_name, _tone_mark in _TONE_MARKS.items():
            _syllable = unicodedata.normalize('NFC', _initial + _rhyme + _tone_mark)
            VIETNAMESE_SYLLABLE_TONES[_syllable] = _tone_name

_MANUAL_SYLLABLE_TONES = {
    'gì': 'huyền', 'bị': 'nặng', 'đã': 'ngã', 'và': 'huyền', 'mà': 'huyền',
    'sẽ': 'ngã', 'cũng': 'ngã', 'vẫn': 'ngã', 'được': 'nặng', 'phải': 'hỏi',
    'mới': 'sắc', 'cũ': 'ngã', 'cả': 'hỏi', 'những': 'ngã', 'các': 'sắc',
    'mọi': 'nặng', 'mỗi': 'ngã', 'này': 'huyền', 'đó': 'sắc', 'kia': 'ngang',
    'đây': 'ngang', 'ấy': 'sắc', 'nọ': 'nặng', 'rất': 'sắc', 'quá': 'sắc',
    'lắm': 'sắc', 'hơi': 'ngang', 'khá': 'sắc', 'cực': 'nặng', 'luôn': 'ngang',
    'cứ': 'sắc', 'chỉ': 'hỏi', 'đều': 'huyền', 'tôi': 'ngang', 'anh': 'ngang',
    'chị': 'nặng', 'em': 'ngang', 'mình': 'huyền', 'họ': 'nặng', 'nó': 'sắc',
    'ta': 'ngang', 'chúng': 'sắc', 'có': 'sắc', 'không': 'ngang', 'chưa': 'ngang',
    'là': 'huyền', 'thì': 'huyền', 'nên': 'ngang', 'vì': 'huyền', 'tại': 'nặng',
    'bởi': 'hỏi', 'cho': 'ngang', 'để': 'hỏi', 'với': 'sắc', 'về': 'huyền',
    'đến': 'sắc', 'từ': 'huyền', 'đi': 'ngang', 'lại': 'nặng', 'ra': 'ngang',
    'vào': 'huyền', 'lên': 'ngang', 'xuống': 'sắc',
}
VIETNAMESE_SYLLABLE_TONES.update(_MANUAL_SYLLABLE_TONES)


def get_syllable_tone(syllable: str) -> Optional[str]:
    """Tone of a Vietnamese syllable (each syllable has exactly one)."""
    syllable = syllable.lower().strip()
    if syllable in VIETNAMESE_SYLLABLE_TONES:
        return VIETNAMESE_SYLLABLE_TONES[syllable]
    decomposed = unicodedata.normalize('NFD', syllable)
    tone_map = {'̀': 'huyền', '́': 'sắc', '̃': 'ngã', '̉': 'hỏi', '̣': 'nặng'}
    for char in decomposed:
        if char in tone_map:
            return tone_map[char]
    return 'ngang'


class VietnameseWordSegmenter:
    """Groups decoded BPE subword tokens into complete Vietnamese words, using
    (in order of preference) underthesea/pyvi if installed, else a small
    dictionary of common compounds/words plus greedy longest-match grouping."""

    def __init__(self, use_external: bool = True):
        self.use_external = use_external
        self._external_segmenter = None
        self._word_list: Set[str] = set()
        self._load_word_list()
        if use_external:
            self._init_external()

    def _load_word_list(self):
        compounds = [
            'hợp_tác_xã', 'máy_tính', 'điện_thoại', 'học_sinh', 'giáo_viên', 'sinh_viên',
            'nhà_trường', 'bệnh_viện', 'sân_bay', 'nhà_ga', 'xe_buýt', 'tàu_hỏa',
            'máy_bay', 'ô_tô', 'xe_máy', 'xe_đạp', 'công_ty', 'doanh_nghiệp',
            'cửa_hàng', 'siêu_thị', 'ngân_hàng', 'bưu_điện', 'thư_viện', 'nhà_sách',
            'công_viên', 'bảo_tàng', 'rạp_chiếu_phim', 'nhà_hát', 'đất_nước', 'con_người',
            'xã_hội', 'cộng_đồng', 'môi_trường', 'kinh_tế', 'chính_trị', 'văn_hóa',
            'giáo_dục', 'y_tế', 'khoa_học', 'công_nghệ', 'phát_triển', 'bảo_vệ',
            'xây_dựng', 'quản_lý', 'nghiên_cứu', 'đào_tạo', 'sản_xuất', 'kinh_doanh',
            'dịch_vụ', 'thương_mại', 'xuất_khẩu', 'nhập_khẩu', 'đầu_tư', 'tài_chính',
            'kế_toán', 'kiểm_toán', 'luật_sư', 'bác_sĩ', 'kỹ_sư', 'kiến_trúc_sư',
            'nhà_báo', 'ca_sĩ', 'diễn_viên', 'vận_động_viên', 'bóng_đá', 'cầu_lông',
            'bơi_lội', 'điền_kinh', 'âm_nhạc', 'hội_họa', 'điện_ảnh', 'nhiếp_ảnh',
            'hợp_tác_xã_nông_nghiệp', 'ủy_ban_nhân_dân', 'hội_đồng_nhân_dân',
            'tòa_án_nhân_dân', 'viện_kiểm_sát_nhân_dân', 'mặt_trận_tổ_quốc',
            'xinh_xắn', 'đẹp_đẽ', 'mạnh_mẽ', 'nhẹ_nhàng', 'vội_vàng', 'chậm_chạp',
            'sạch_sẽ', 'dơ_dáy', 'sáng_sủa', 'tối_tăm', 'khó_khăn', 'dễ_dàng',
            'ngoan_ngoãn', 'hư_hỏng', 'buồn_bã', 'vui_vẻ',
        ]
        self._word_list.update(compounds)
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
        try:
            from underthesea import word_tokenize
            self._external_segmenter = word_tokenize
        except ImportError:
            try:
                from pyvi import ViTokenizer
                self._external_segmenter = ViTokenizer.tokenize
            except ImportError:
                pass  # dictionary-based segmentation only

    def segment_text(self, text: str) -> List[str]:
        if self._external_segmenter:
            result = self._external_segmenter(text)
            return result.split() if isinstance(result, str) else result
        return self._dictionary_segment(text)

    def _dictionary_segment(self, text: str) -> List[str]:
        words = text.strip().split()
        result, i = [], 0
        while i < len(words):
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

    def is_known_compound(self, word: str) -> bool:
        return '_' in word and word in self._word_list

    def group_subword_tokens_with_spans(self, tokens: List[str]) -> List[Tuple[str, List[int]]]:
        """Greedy word grouping that also returns which original token indices
        contributed to each group, so callers can map words back onto the
        original token sequence (e.g. compound-word classification)."""

        def clean_token(t: str) -> str:
            t = t.strip()
            for prefix in ('▁', 'Ġ', '##', '_'):
                if t.startswith(prefix):
                    t = t[len(prefix):]
            return t.strip().lower()

        indexed_clean = [(i, clean_token(t)) for i, t in enumerate(tokens)]
        indexed_clean = [(i, c) for i, c in indexed_clean if c]
        if not indexed_clean:
            return []

        result: List[Tuple[str, List[int]]] = []
        pos, n = 0, len(indexed_clean)
        while pos < n:
            best_len, best_word = 1, indexed_clean[pos][1]
            for length in range(min(5, n - pos), 0, -1):
                candidate = '_'.join(c for _, c in indexed_clean[pos:pos + length])
                if candidate in self._word_list:
                    best_len, best_word = length, candidate
                    break
            span_indices = [idx for idx, _ in indexed_clean[pos:pos + best_len]]
            result.append((best_word, span_indices))
            pos += best_len
        return result

    def group_subword_tokens(self, tokens: List[str]) -> List[str]:
        def clean_token(t: str) -> str:
            t = t.strip()
            for prefix in ('▁', 'Ġ', '##', '_'):
                if t.startswith(prefix):
                    t = t[len(prefix):]
            return t.strip().lower()

        clean = [c for c in (clean_token(t) for t in tokens) if c]
        if not clean:
            return []
        result, i, n = [], 0, len(clean)
        while i < n:
            best_len, best_word = 1, clean[i]
            for length in range(min(5, n - i), 0, -1):
                candidate = '_'.join(clean[i:i + length])
                if candidate in self._word_list:
                    best_len, best_word = length, candidate
                    break
            result.append(best_word)
            i += best_len
        return result


# --- Teencode / dialect / Sino-Vietnamese dictionaries ----------------------

TEENCODE_MAP: Dict[str, str] = {
    'ko': 'không', 'k': 'không', 'kh': 'không', 'hong': 'không', 'hông': 'không',
    'hổng': 'không', 'chẳng': 'không', 'chả': 'không',
    'dc': 'được', 'đc': 'được', 'đk': 'được', 'dx': 'được', 'ok': 'được', 'oke': 'được',
    'ng': 'người', 'ngta': 'người_ta', 'ah': 'anh', 'e': 'em', 'a': 'anh', 'c': 'chị',
    'm': 'mày', 't': 'tao', 'bn': 'bạn', 'mn': 'mọi_người', 'ae': 'anh_em', 'ace': 'anh_chị_em',
    'vs': 'với', 'w': 'với', 'cx': 'cũng', 'cxn': 'cũng', 'cug': 'cũng',
    'vc': 'vợ_chồng', 'ck': 'chồng', 'vk': 'vợ', 'ny': 'người_yêu', 'bb': 'bạn_bè',
    'nc': 'nói_chuyện', 'nt': 'nhắn_tin', 'ib': 'nhắn_tin', 'inb': 'nhắn_tin',
    'cmt': 'bình_luận', 'cm': 'bình_luận', 'rep': 'trả_lời', 'fb': 'facebook',
    'h': 'giờ', 'p': 'phút', 'tn': 'tuần', 'th': 'tháng', 'n': 'năm',
    'tk': 'tài_khoản', 'sp': 'sản_phẩm', 'đt': 'điện_thoại', 'dt': 'điện_thoại',
    'mt': 'máy_tính', 'lh': 'liên_hệ', 'stt': 'số_thứ_tự', 'st': 'số_thứ_tự',
    'cl': 'chất_lượng', 'sl': 'số_lượng', 'bh': 'bảo_hành', 'km': 'khuyến_mãi',
    'vui': 'vui', 'bùn': 'buồn', 'giận': 'giận', 'thưn': 'thương', 'thik': 'thích',
    'thix': 'thích', 'ghét': 'ghét', 'ghen': 'ghen', 'nhớ': 'nhớ',
    'bt': 'biết', 'bít': 'biết', 'hiu': 'hiểu', 'nghĩ': 'nghĩ', 'nghix': 'nghĩ',
    'nghj': 'nghĩ', 'lm': 'làm', 'lam': 'làm', 'lèm': 'làm',
    'j': 'gì', 'chi': 'gì', 'z': 'vậy', 'zị': 'vậy', 'sao': 'sao', 'seo': 'sao', 's': 'sao',
    'nhìu': 'nhiều', 'nhiu': 'nhiều', 'ít': 'ít', 'hơi': 'hơi', 'khá': 'khá',
    'quá': 'quá', 'lắm': 'lắm', 'cực': 'rất', 'siêu': 'rất', 'cực_kỳ': 'rất',
}

DIALECT_MAP: Dict[str, Dict[str, str]] = {
    'central': {
        'mô': 'nào', 'tê': 'kia', 'răng': 'sao', 'rứa': 'thế', 'chi': 'gì', 'nờ': 'nào',
        'chừ': 'giờ', 'mi': 'mày', 'tau': 'tao', 'hắn': 'nó', 'bọn_hắn': 'bọn_nó',
        'eng': 'em', 'nác': 'nước', 'đọi': 'bát', 'trốc': 'đầu', 'tru': 'trâu',
        'cươi': 'sân', 'nốc': 'uống', 'bổ': 'ngã', 'trốc_tru': 'ngu_ngốc',
    },
    'southern': {
        'hông': 'không', 'hổng': 'không', 'hen': 'nhé', 'ghe': 'thuyền', 'mắc': 'đắt',
        'bông': 'hoa', 'trái': 'quả', 'thơm': 'dứa', 'đậu_phộng': 'lạc', 'má': 'mẹ',
        'tía': 'bố', 'ngoại': 'bà_ngoại', 'nội': 'ông_bà_nội', 'cưng': 'yêu', 'ghê': 'nhiều',
        'dữ': 'nhiều', 'dễ_sợ': 'nhiều', 'dữ_thần': 'nhiều', 'chén': 'ăn', 'nhậu': 'ăn_nhậu',
        'dzô': 'uống', 'xe_hơi': 'ô_tô', 'vi_tính': 'máy_tính', 'quần_gin': 'quần_bò',
        'áo_thun': 'áo_phông', 'chả_lụ': 'chả_lụa', 'bánh_mì': 'bánh_mì', 'nước_đá': 'đá',
        'đá_lạnh': 'đá',
    },
    'northern': {
        'ngô': 'bắp', 'dứa': 'thơm', 'lợn': 'heo', 'quả': 'trái', 'hoa_quả': 'trái_cây',
        'bát': 'chén', 'đũa': 'đũa', 'thìa': 'muỗng', 'muôi': 'vá', 'rổ': 'rổ', 'rá': 'rá',
        'chăn': 'mền', 'gối': 'gối', 'đệm': 'nệm', 'phích': 'bình_thủy', 'ấm': 'ấm', 'tích': 'bình_trà',
    },
}
ALL_DIALECT_MAP: Dict[str, str] = {}
for _region, _mapping in DIALECT_MAP.items():
    ALL_DIALECT_MAP.update(_mapping)

SINO_VIETNAMESE_MORPHEMES: Dict[str, str] = {
    'quốc': 'quốc_gia', 'gia': 'gia_đình', 'xã': 'xã_hội', 'hội': 'hội_nghị', 'phủ': 'chính_phủ',
    'học': 'học_tập', 'sinh': 'sinh_viên', 'giáo': 'giáo_dục', 'viên': 'giáo_viên', 'công': 'công_nghiệp',
    'thương': 'thương_mại', 'mại': 'thương_mại', 'nông': 'nông_nghiệp', 'lâm': 'lâm_nghiệp',
    'ngư': 'ngư_nghiệp', 'nghiệp': 'sự_nghiệp', 'khoa': 'khoa_học', 'kỹ': 'kỹ_thuật',
    'thuật': 'kỹ_thuật', 'văn': 'văn_hóa', 'hóa': 'văn_hóa', 'nghệ': 'nghệ_thuật', 'mỹ': 'mỹ_thuật',
    'y': 'y_tế', 'tế': 'y_tế', 'dược': 'dược_phẩm', 'luật': 'luật_pháp', 'pháp': 'pháp_luật',
    'kinh': 'kinh_tế', 'tài': 'tài_chính', 'chính': 'tài_chính', 'điện': 'điện_tử', 'tử': 'điện_tử',
    'cơ': 'cơ_khí', 'khí': 'cơ_khí', 'kiến': 'kiến_trúc', 'trúc': 'kiến_trúc', 'xây': 'xây_dựng',
    'dựng': 'xây_dựng', 'phát': 'phát_triển', 'triển': 'phát_triển', 'bảo': 'bảo_vệ', 'vệ': 'bảo_vệ',
    'quản': 'quản_lý', 'lý': 'quản_lý', 'nghiên': 'nghiên_cứu', 'cứu': 'nghiên_cứu', 'đào': 'đào_tạo',
    'tạo': 'đào_tạo', 'sản': 'sản_xuất', 'xuất': 'sản_xuất', 'doanh': 'kinh_doanh', 'đầu': 'đầu_tư',
    'tư': 'đầu_tư', 'thông': 'thông_tin', 'tin': 'thông_tin', 'truyền': 'truyền_thông',
    'viễn': 'viễn_thông', 'giao': 'giao_thông', 'vận': 'vận_tải', 'tải': 'vận_tải', 'hàng': 'hàng_không',
    'không': 'không_gian', 'hải': 'hàng_hải', 'đường': 'đường_bộ', 'thủy': 'đường_thủy',
    'lợi': 'thủy_lợi', 'thanh': 'thanh_tra', 'tra': 'kiểm_tra', 'kiểm': 'kiểm_soát',
    'soát': 'kiểm_soát', 'thẩm': 'thẩm_định', 'định': 'đánh_giá', 'tổ': 'tổ_chức', 'chức': 'tổ_chức',
    'đoàn': 'đoàn_thể', 'thể': 'tập_thể', 'hợp': 'hợp_tác', 'tác': 'hợp_tác', 'liên': 'liên_kết',
    'kết': 'kết_nối', 'thống': 'thống_nhất', 'nhất': 'thống_nhất', 'độc': 'độc_lập', 'lập': 'độc_lập',
    'tự': 'tự_do', 'do': 'tự_do', 'dân': 'dân_chủ', 'chủ': 'dân_chủ', 'cộng': 'cộng_đồng',
    'đồng': 'cộng_đồng', 'hòa': 'hòa_bình', 'bình': 'hòa_bình',
}
SINO_VIETNAMESE_COMPOUNDS: Set[str] = {
    f"{m1}_{m2}"
    for m1 in SINO_VIETNAMESE_MORPHEMES for m2 in SINO_VIETNAMESE_MORPHEMES
    if m1 != m2 and len(f"{m1}_{m2}") >= 6
}

CRITICAL_PATTERNS = {
    'numbers': re.compile(
        r'\d+[\.,\d]*\s*(?:tỷ|triệu|nghìn|ngàn|trăm|đồng|USD|VND|%|phần_trăm)?', re.IGNORECASE),
    'dates': re.compile(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}'),
    'proper_names': re.compile(
        r'[A-ZĐ][a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]+(?:\s+[A-ZĐ][a-zà-ỹ]+)*'),
    'legal_refs': re.compile(r'(?:Điều|Khoản|Mục|Chương|Phần)\s+\d+', re.IGNORECASE),
    'emails': re.compile(r'[\w\.-]+@[\w\.-]+\.\w+'),
    'urls': re.compile(r'https?://[^\s]+'),
}


def is_critical_pattern(token: str) -> bool:
    """True if `token` matches a pattern that should never be compressed
    (numbers, dates, proper names, legal references, emails, URLs)."""
    return any(pattern.search(token) for pattern in CRITICAL_PATTERNS.values())


def is_vietnamese_function_word_extended(word: str) -> bool:
    """Function-word check extended with teencode/dialect normalization."""
    word_lower = word.lower().strip()
    if word_lower in VIETNAMESE_FUNCTION_WORDS:
        return True
    standard = TEENCODE_MAP.get(word_lower) or ALL_DIALECT_MAP.get(word_lower)
    return bool(standard and standard in VIETNAMESE_FUNCTION_WORDS)


def normalize_vietnamese_word(word: str) -> str:
    """teencode/dialect -> standard form, else the word unchanged (lowercased)."""
    word_lower = word.lower().strip()
    return TEENCODE_MAP.get(word_lower) or ALL_DIALECT_MAP.get(word_lower) or word_lower


# ============================================================================
# 4. Morphology (word classification)
# ============================================================================
#
# Vietnamese is an isolating language with distinct word classes that make
# very different compression targets:
#   FUNC (hu tu: da, se, cua, nhung...) -- high frequency, low semantic
#     content -> compress aggressively.
#   CONTENT (thuc tu: nouns/verbs/adjectives) -- preserve carefully.
#   REDUP (tu lay: xinh xan, dep de...) -- semantic redundancy -> can merge.
#   COMPOUND (tu ghep: may_tinh, hoc_sinh...) -- must not be split.
#   SINO (Sino-Vietnamese morphemes) -- formal/academic -> preserve.


class WordClass(Enum):
    FUNC = 'function'
    CONTENT = 'content'
    REDUP = 'reduplicative'
    COMPOUND = 'compound'
    SINO = 'sino'
    OTHER = 'other'


@dataclass
class MorphologyConfig:
    """Per-class merge (keep) ratios and preservation-score multipliers."""
    r_func: float = 0.3
    r_content: float = 0.85
    r_redup: float = 0.5
    r_compound: float = 0.95
    r_sino: float = 0.90
    r_other: float = 0.5

    f_func: float = 0.4
    f_content: float = 1.2
    f_redup: float = 0.6
    f_compound: float = 1.5
    f_sino: float = 1.5
    f_other: float = 1.0

    redup_similarity_threshold: float = 0.6
    use_phonetic_redup: bool = True
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


VIETNAMESE_FUNCTION_WORDS: Set[str] = {
    'đã', 'sẽ', 'đang', 'vừa', 'mới', 'từng', 'bị', 'được', 'phải', 'cần', 'nên', 'có_thể',
    'của', 'cho', 'với', 'về', 'tại', 'trong', 'ngoài', 'trên', 'dưới',
    'ở', 'đến', 'từ', 'để', 'bằng', 'vào', 'ra', 'lên', 'xuống',
    'và', 'hoặc', 'nhưng', 'mà', 'nếu', 'thì', 'vì', 'tuy', 'dù', 'còn', 'hay', 'rằng', 'là',
    'những', 'các', 'mọi', 'mỗi', 'một', 'vài', 'mấy', 'cả', 'tất_cả', 'toàn_bộ',
    'cái', 'con', 'chiếc', 'người', 'cuốn', 'quyển', 'tờ', 'bức',
    'tôi', 'ta', 'chúng_tôi', 'chúng_ta', 'mình', 'họ', 'nó',
    'này', 'đó', 'kia', 'ấy', 'đây', 'nọ',
    'ạ', 'nhé', 'nhỉ', 'đi', 'thôi', 'chứ', 'cơ', 'ư', 'hả', 'hử', 'sao', 'không', 'chưa',
    'đừng', 'chớ', 'rất', 'quá', 'lắm', 'hơi', 'khá', 'cực', 'cực_kỳ',
    'luôn', 'cũng', 'vẫn', 'cứ', 'chỉ', 'đều',
    'có', 'làm', 'khi', 'khiến', 'bắt_đầu', 'tiếp_tục',
}

REDUPLICATIVE_PATTERNS = [
    ('xinh', 'xắn'), ('đẹp', 'đẽ'), ('mạnh', 'mẽ'), ('nhẹ', 'nhàng'), ('vội', 'vàng'),
    ('chậm', 'chạp'), ('sạch', 'sẽ'), ('dơ', 'dáy'), ('sáng', 'sủa'), ('tối', 'tăm'),
    ('khó', 'khăn'), ('dễ', 'dàng'), ('ngoan', 'ngoãn'), ('hư', 'hỏng'), ('buồn', 'bã'),
    ('vui', 'vẻ'), ('lạnh', 'lẽo'), ('nóng', 'nực'), ('mát', 'mẻ'), ('ấm', 'áp'),
    ('rộng', 'rãi'), ('hẹp', 'hòi'), ('cao', 'cả'), ('thấp', 'thoải'), ('xa', 'xôi'),
    ('gần', 'gũi'), ('bừa', 'bãi'), ('lộn', 'xộn'), ('ngăn', 'nắp'), ('chăm', 'chỉ'),
    ('siêng', 'năng'), ('lười', 'nhác'), ('thông', 'thái'), ('ngu', 'ngốc'), ('khôn', 'khéo'),
    ('vụng', 'về'), ('tài', 'tình'), ('giỏi', 'giang'), ('lung', 'linh'), ('long', 'lanh'),
    ('lấp', 'lánh'), ('rực', 'rỡ'), ('lung', 'lay'), ('đủng', 'đỉnh'), ('thong', 'thả'),
    ('từ', 'tốn'), ('điềm', 'đạm'), ('xanh', 'xao'), ('vàng', 'vọt'), ('đỏ', 'đắn'),
    ('tim', 'tím'), ('trắng', 'trẻo'), ('đen', 'đúa'),
]


class MorphologyAnalyzer:
    """Classify Vietnamese tokens by word class using a static function-word
    dictionary, reduplicative pattern matching, compound-span detection (via
    VietnameseWordSegmenter), Sino-Vietnamese morpheme detection, and
    (optionally) underthesea POS tagging."""

    def __init__(self, use_pos_tagger: bool = False, use_word_segmentation: bool = True):
        self.use_pos_tagger = use_pos_tagger
        self.function_words = VIETNAMESE_FUNCTION_WORDS

        self.redup_pairs: Dict[str, str] = {}
        for first, second in REDUPLICATIVE_PATTERNS:
            self.redup_pairs[second] = first
            self.redup_pairs[first + '_' + second] = first

        self._sino_morphemes: Set[str] = set(SINO_VIETNAMESE_MORPHEMES.keys())

        self._pos_tagger = None
        if use_pos_tagger:
            self._init_pos_tagger()

        # Regroups decoded BPE tokens ('máy', 'tính') into known multi-syllable
        # compounds ('máy_tính') so COMPOUND fires on real tokenizer output,
        # not just tokens that already contain a literal '_'.
        self.use_word_segmentation = use_word_segmentation
        self._segmenter = None
        if use_word_segmentation:
            try:
                self._segmenter = VietnameseWordSegmenter(use_external=True)
            except Exception:
                self._segmenter = None
                self.use_word_segmentation = False

    def _init_pos_tagger(self):
        try:
            from underthesea import pos_tag
            self._pos_tagger = pos_tag
        except ImportError:
            self.use_pos_tagger = False

    @staticmethod
    def _tone_strip(syllable: str) -> str:
        decomposed = unicodedata.normalize('NFD', syllable.lower())
        no_tone = ''.join(ch for ch in decomposed if ch not in ('̀', '́', '̃', '̉', '̣'))
        return unicodedata.normalize('NFC', no_tone)

    # Vietnamese tone registers ("luật hài thanh"): reduplicated syllables take
    # both tones from the same register; crossing registers is a minimal pair
    # (bàn/bán), not reduplication (nhè/nhẹ). See _phonetic_similarity.
    _TONE_REGISTER = {'': 'bong', '̉': 'bong', '́': 'bong', '̀': 'tram', '̃': 'tram', '̣': 'tram'}

    @staticmethod
    def _tone_register(syllable: str) -> str:
        for ch in unicodedata.normalize('NFD', syllable.lower()):
            if ch in MorphologyAnalyzer._TONE_REGISTER and ch != '':
                return MorphologyAnalyzer._TONE_REGISTER[ch]
        return 'bong'

    @staticmethod
    def _phonetic_similarity(a: str, b: str) -> float:
        """Shared initial consonant + shared rhyme similarity in [0, 1]."""
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        a_stripped, b_stripped = MorphologyAnalyzer._tone_strip(a), MorphologyAnalyzer._tone_strip(b)
        if a_stripped == b_stripped:
            # Same segments, different tone: only a same-register tone change
            # (luật hài thanh) is real reduplication -- a cross-register pair
            # (bàn/bán) is a minimal pair, which must NOT be scored as redup
            # (it would otherwise get merged away, destroying tone contrast).
            return 1.0 if MorphologyAnalyzer._tone_register(a) == MorphologyAnalyzer._tone_register(b) else 0.0
        a, b = a_stripped, b_stripped
        initials = r'^(b|c|ch|d|đ|g|gh|gi|h|k|kh|l|m|n|ng|ngh|nh|p|ph|qu|r|s|t|th|tr|v|x)?'
        a_init, b_init = re.match(initials, a), re.match(initials, b)
        a_rhyme = a[a_init.end():] if a_init else a
        b_rhyme = b[b_init.end():] if b_init else b
        score = 0.0
        if a_init and b_init and a_init.group(1) == b_init.group(1):
            score += 0.4
        if a_rhyme == b_rhyme:
            score += 0.6
        elif len(a_rhyme) >= 2 and len(b_rhyme) >= 2:
            if sum(1 for ca, cb in zip(a_rhyme, b_rhyme) if ca == cb) >= 2:
                score += 0.3
        return score

    def classify_word(self, token: str) -> WordClass:
        """Priority: function word > reduplicative pair > compound (has '_')
        > Sino-Vietnamese morpheme > content word. Punctuation/numbers/
        whitespace classify as OTHER."""
        token_lower = token.strip().lower()
        if not token_lower or not any(c.isalpha() for c in token_lower):
            return WordClass.OTHER
        if token_lower in self.function_words:
            return WordClass.FUNC
        # Not `token_lower in self.redup_pairs`: that dict also maps every
        # second syllable back to its first, so a standalone content word that
        # happens to BE some word's second syllable (e.g. 'vàng') would
        # misclassify as REDUP. Reduplication is a property of an adjacent
        # PAIR (handled by find_reduplicative_pairs), not a lone token.
        if '_' in token_lower and token_lower in self.redup_pairs:
            return WordClass.REDUP
        if '_' in token_lower:
            for part in token_lower.split('_'):
                if part in self.redup_pairs:
                    return WordClass.REDUP
                if part in self._sino_morphemes:
                    return WordClass.SINO
            return WordClass.COMPOUND
        if token_lower in self._sino_morphemes:
            return WordClass.SINO
        return WordClass.CONTENT

    def _find_compound_spans(self, tokens: List[str]) -> Set[int]:
        """Token indices that fall inside a known multi-token compound, from
        regrouping `tokens` with the word segmenter -- lets COMPOUND fire on
        real BPE-decoded output, not just literal-underscore tokens."""
        if not self._segmenter:
            return set()
        try:
            groups = self._segmenter.group_subword_tokens_with_spans(tokens)
        except Exception:
            return set()
        return {
            idx
            for word, indices in groups
            if len(indices) > 1 and self._segmenter.is_known_compound(word)
            for idx in indices
        }

    def classify_batch(self, tokens: List[str], sentence: Optional[str] = None) -> List[WordInfo]:
        pos_tags = {}
        if self.use_pos_tagger and self._pos_tagger and sentence:
            try:
                for word, tag in self._pos_tagger(sentence):
                    pos_tags[word.lower()] = tag
            except Exception:
                pass

        compound_indices = self._find_compound_spans(tokens) if self.use_word_segmentation else set()
        results = []
        for i, token in enumerate(tokens):
            word_class = WordClass.COMPOUND if i in compound_indices else self.classify_word(token)
            results.append(WordInfo(
                token=token, token_id=i, word_class=word_class,
                is_function_word=(word_class == WordClass.FUNC),
                is_content_word=(word_class == WordClass.CONTENT),
                is_reduplicative=(word_class == WordClass.REDUP),
                is_compound_part=(word_class == WordClass.COMPOUND),
            ))
        return results

    def get_preservation_multiplier(self, info: WordInfo, config: MorphologyConfig) -> float:
        mapping = {
            WordClass.FUNC: config.f_func, WordClass.CONTENT: config.f_content,
            WordClass.REDUP: config.f_redup, WordClass.COMPOUND: config.f_compound,
            WordClass.SINO: config.f_sino, WordClass.OTHER: config.f_other,
        }
        return mapping.get(info.word_class, 1.0)

    def get_merge_ratio(self, info: WordInfo, config: MorphologyConfig) -> float:
        mapping = {
            WordClass.FUNC: config.r_func, WordClass.CONTENT: config.r_content,
            WordClass.REDUP: config.r_redup, WordClass.COMPOUND: config.r_compound,
            WordClass.SINO: config.r_sino, WordClass.OTHER: config.r_other,
        }
        return mapping.get(info.word_class, 0.5)

    def find_reduplicative_pairs(
        self, tokens: List[str], window_size: int = 3, config: Optional[MorphologyConfig] = None,
    ) -> List[Tuple[int, int, float]]:
        """(left_idx, right_idx, confidence) for reduplicative pairs, found by
        dictionary match first, then phonetic similarity fallback."""
        pairs = []
        n = len(tokens)
        for i in range(n):
            left = tokens[i].strip().lower()
            for j in range(i + 1, min(i + window_size + 1, n)):
                right = tokens[j].strip().lower()
                pair_key = f"{left}_{right}"
                # Require the actual (left, right) pair in order -- redup_pairs
                # also maps every second syllable back to its first, so
                # accepting `right in self.redup_pairs` alone matched ANY
                # preceding token, not just the true left partner.
                if pair_key in self.redup_pairs or self.redup_pairs.get(right) == left:
                    pairs.append((i, j, 1.0))
                    break
                if config and config.use_phonetic_redup:
                    sim = self._phonetic_similarity(left, right)
                    if sim >= config.redup_similarity_threshold:
                        pairs.append((i, j, sim))
                        break
        return pairs


class TokenInflationAnalyzer:
    """Token Inflation Ratio (TIR) = tokens_vi / tokens_en for parallel text --
    quantifies how much "harder" Vietnamese compression is (typical TIR: 1.5-2.0)."""

    def __init__(self, vi_tokenizer, en_tokenizer):
        self.vi_tokenizer = vi_tokenizer
        self.en_tokenizer = en_tokenizer

    def compute_tir(self, text_vi: str, text_en: str) -> float:
        vi_tokens = len(self.vi_tokenizer.encode(text_vi))
        en_tokens = len(self.en_tokenizer.encode(text_en))
        return vi_tokens / en_tokens if en_tokens > 0 else 1.0

    def estimate_tir_batch(self, texts_vi: List[str], texts_en: List[str]) -> Dict[str, float]:
        tirs = [self.compute_tir(vi, en) for vi, en in zip(texts_vi, texts_en) if vi and en]
        if not tirs:
            return {'mean': 1.0, 'min': 1.0, 'max': 1.0, 'std': 0.0}
        return {
            'mean': statistics.mean(tirs), 'min': min(tirs), 'max': max(tirs),
            'std': statistics.stdev(tirs) if len(tirs) > 1 else 0.0,
        }


_default_morph_analyzer: Optional[MorphologyAnalyzer] = None


def get_morphology_analyzer(**kwargs) -> MorphologyAnalyzer:
    """Get or create the default MorphologyAnalyzer instance."""
    global _default_morph_analyzer
    if _default_morph_analyzer is None:
        _default_morph_analyzer = MorphologyAnalyzer(**kwargs)
    return _default_morph_analyzer


# ============================================================================
# 5. Torch-dependent tone probe (training + inference bridge)
# ============================================================================
#
#   Phonological Consistency Loss:  L_tone = mean CE(linear(h_i), tone_id(t_i))
#   Combined training objective:    L = L_LM + lambda_tone * L_tone
#
# The same classifier trained here is reused at inference time as LACC's
# trained-tone-probe signal (see compression.LACCScorer / models.load_scorer):
# tokens the model is tonally confident about are scored as more important to
# keep. This is the training -> inference bridge the paper's method section
# describes.


@dataclass
class ToneAwareConfig:
    """Configuration for the (legacy) standalone ToneAwareScorer helper."""
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.4
    window_size: int = 2
    tone_embed_dim: int = 64
    lambda_tone: float = 0.1
    use_tone_embedding: bool = True
    use_contrast: bool = True
    min_preservation_weight: float = 1.0
    max_preservation_weight: float = 3.0


class ToneAwareScorer:
    """Combine tone-aware weights with an externally computed base score --
    a thin convenience wrapper around VietnameseToneAnalyzer for callers that
    already have their own base (perplexity/attention) scores."""

    def __init__(self, tokenizer, config: Optional[ToneAwareConfig] = None):
        self.tokenizer = tokenizer
        self.config = config or ToneAwareConfig()
        self.tone_analyzer = get_tone_analyzer(
            alpha=self.config.alpha, beta=self.config.beta, gamma=self.config.gamma,
        )

    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        tokens = []
        for tid in token_ids:
            t = self.tokenizer.decode([tid]).replace('▁', ' ').replace('Ġ', ' ').strip()
            tokens.append(t)
        return tokens

    def compute_tone_scores(self, token_ids: List[int], base_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.decode_tokens(token_ids)
        tone_infos = self.tone_analyzer.analyze_tokens(tokens, window_size=self.config.window_size)
        tone_weights = torch.tensor([
            max(self.config.min_preservation_weight,
                min(self.config.max_preservation_weight, info.preservation_weight))
            for info in tone_infos
        ]) if tone_infos else torch.ones(len(tokens))

        if base_scores is None:
            return tone_weights
        base_norm = base_scores
        if base_scores.max() > base_scores.min():
            base_norm = (base_scores - base_scores.min()) / (base_scores.max() - base_scores.min())
        return base_norm * tone_weights

    def select_tokens(
        self, token_ids: List[int], budget: int, base_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], torch.Tensor]:
        tone_scores = self.compute_tone_scores(token_ids, base_scores)
        n = len(token_ids)
        if budget >= n:
            return token_ids, tone_scores
        mid_scores = tone_scores[1:-1].clone()
        mid_budget = budget - 2
        if mid_budget <= 0:
            keep_indices = [0, n - 1] if n > 2 else list(range(n))
        else:
            _, mid_top = torch.topk(mid_scores, min(mid_budget, len(mid_scores)))
            keep_indices = sorted([0] + (mid_top + 1).tolist() + [n - 1])
        return [token_ids[i] for i in keep_indices if i < n], tone_scores


class ToneEmbeddingAugmentation(nn.Module):
    """Concatenate a learnable tone embedding onto token embeddings:
    e'_t = e_t (+) W_tone[tone_id(t)] -- for training a compression-aware model."""

    def __init__(self, d_model: int, tone_embed_dim: int = 64, num_tones: int = 7):
        super().__init__()
        self.d_model = d_model
        self.tone_embed_dim = tone_embed_dim
        self.num_tones = num_tones
        self.tone_embedding = nn.Embedding(num_tones, tone_embed_dim)
        nn.init.normal_(self.tone_embedding.weight, std=0.02)
        self.projection = nn.Linear(d_model + tone_embed_dim, d_model, bias=False)
        self.use_projection = False

    def forward(self, token_embeddings: torch.Tensor, tone_ids: torch.Tensor, project: bool = False) -> torch.Tensor:
        tone_embeds = self.tone_embedding(tone_ids)
        augmented = torch.cat([token_embeddings, tone_embeds], dim=-1)
        if project or self.use_projection:
            augmented = self.projection(augmented)
        return augmented


class PhonologicalConsistencyLoss(nn.Module):
    """Auxiliary loss + inference-time tone probe.

    L_tone = mean CE(linear(h_i), tone_label_i); combined L = L_LM + lambda*L_tone.
    `score_importance` reuses the trained classifier as LACC's model-tone
    signal: higher classifier confidence = tone info is well encoded in this
    hidden state = keep this token.
    """

    def __init__(self, hidden_dim: int, num_tones: int = 7, lambda_tone: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.num_tones = num_tones
        self.lambda_tone = lambda_tone
        self.ignore_index = ignore_index
        self.tone_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, num_tones),
        )

    def forward(self, hidden_states: torch.Tensor, tone_labels: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = hidden_states.shape
        Bt, St = tone_labels.shape
        if S != St:
            raise ValueError(
                f"Sequence length mismatch: hidden_states has S={S} but tone_labels has "
                f"S={St}. Tone labels must align with input (compressed) positions."
            )
        logits = self.tone_classifier(hidden_states)
        if mask is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_tones), tone_labels.view(-1),
                reduction='none', ignore_index=self.ignore_index,
            )
            loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-8)
        else:
            loss = F.cross_entropy(
                logits.view(-1, self.num_tones), tone_labels.view(-1),
                reduction='mean', ignore_index=self.ignore_index,
            )
        return self.lambda_tone * loss

    def predict_tones(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.tone_classifier(hidden_states).argmax(dim=-1)

    def score_importance(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """1.0 + max softmax probability, clamped to [0.5, 3.0] -- higher
        confidence in the tone prediction means preserve this token more."""
        probs = F.softmax(self.tone_classifier(hidden_states), dim=-1)
        max_prob, _ = probs.max(dim=-1)
        return torch.clamp(1.0 + max_prob, 0.5, 3.0)


# --- Wave-2 E4: query-relevance probe (Sentinel / EXIT-style learned relevance) --
#
# Wave 1 refuted tone as a compression signal, but the training method itself
# works (LoRA + a probe on hidden states genuinely learns structure). E4
# keeps that machinery and swaps the LABEL: instead of predicting a token's
# tone (a deterministic function of the token id, and useless for choosing
# which tokens to keep), the probe predicts whether a token is RELEVANT to the
# answer -- exactly the signal LACC lacked. Because RelevanceConsistencyLoss
# mirrors PhonologicalConsistencyLoss's public surface (a `.tone_classifier`
# Sequential, `.num_tones`, `forward`, `score_importance`), it drops into the
# existing LACCScorer / tone_source='model' path with no change to compression.py.


def _normalize_for_overlap(text: str) -> List[str]:
    """Lowercase, NFC, strip punctuation, split on whitespace -- the same
    syllable-level tokenization evaluation._normalize_answer uses, replicated
    here to keep linguistics.py free of an evaluation.py import (and the tone
    marks intact, which the default [a-z0-9] tokenizers would destroy)."""
    text = unicodedata.normalize('NFC', text).lower()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return text.split()


def build_relevance_labels(
    tokenizer, input_ids: Sequence[int], answer_text: str, min_token_len: int = 2, ignore_index: int = -100,
) -> List[int]:
    """Weak-supervision per-token relevance labels for a (context, answer) pair.

    A context token is labelled RELEVANT (1) iff its decoded, normalized
    surface form is a non-trivial syllable (>= `min_token_len` chars after
    stripping) that appears among the answer's syllables; NOT-RELEVANT (0)
    otherwise. Punctuation / whitespace / sub-`min_token_len` pieces are set to
    `ignore_index` so they neither train nor score (they are neither reliably
    relevant nor a clean negative). This is deliberately noisy span-overlap
    supervision (RECOMP/EXIT-style), not gold labels -- enough to teach a probe
    "does this token carry answer content", cheap to produce from VCC-Bench's
    existing reference answers and constructed needles.
    """
    answer_syllables = set(_normalize_for_overlap(answer_text or ''))
    labels: List[int] = []
    for tid in input_ids:
        piece = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        norm = _normalize_for_overlap(piece)
        # A token often decodes to one syllable (possibly with a leading space);
        # treat the whole decoded piece as relevant if ANY of its syllables is
        # a long-enough answer syllable.
        keep = [s for s in norm if len(s) >= min_token_len]
        if not keep:
            labels.append(ignore_index)
        elif any(s in answer_syllables for s in keep):
            labels.append(1)
        else:
            labels.append(0)
    return labels


class RelevanceConsistencyLoss(nn.Module):
    """Binary query-relevance probe: a drop-in, interface-compatible sibling of
    PhonologicalConsistencyLoss (E4).

    L_rel = lambda * CE(linear(h_i), relevant_i);  `score_importance` reuses the
    trained classifier as LACC's model signal -- a token the probe judges more
    likely to be answer-relevant is scored as more important to keep. The
    classifier is stored under the attribute name `tone_classifier` and
    `num_tones` aliases `num_classes` (=2) purely so the existing loader
    (models.load_scorer) and scorer (compression.LACCScorer, which reads
    `.tone_classifier[0].weight` and calls `.score_importance`) work unchanged.
    """

    def __init__(self, hidden_dim: int, num_classes: int = 2, lambda_relevance: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.num_classes = num_classes
        self.num_tones = num_classes  # alias: keeps the tone-probe interface/loader working
        self.lambda_relevance = lambda_relevance
        self.ignore_index = ignore_index
        self.tone_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = hidden_states.shape
        Bt, St = labels.shape
        if S != St:
            raise ValueError(
                f"Sequence length mismatch: hidden_states has S={S} but relevance labels "
                f"have S={St}. Labels must align with input positions."
            )
        logits = self.tone_classifier(hidden_states)
        if mask is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_classes), labels.view(-1),
                reduction='none', ignore_index=self.ignore_index,
            )
            loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-8)
        else:
            loss = F.cross_entropy(
                logits.view(-1, self.num_classes), labels.view(-1),
                reduction='mean', ignore_index=self.ignore_index,
            )
        return self.lambda_relevance * loss

    def predict_relevance(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.tone_classifier(hidden_states).argmax(dim=-1)

    def score_importance(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """1.0 + P(relevant), clamped to [0.5, 3.0] -- a token the probe judges
        more likely to be answer-relevant is preserved more. Unlike the tone
        probe, this reads a signal (relevance) that is NOT a deterministic
        function of the token id, so it can actually help token selection."""
        probs = F.softmax(self.tone_classifier(hidden_states), dim=-1)
        relevant_prob = probs[..., 1]  # class 1 = relevant
        return torch.clamp(1.0 + relevant_prob, 0.5, 3.0)


class TonePreservationProbe(nn.Module):
    """Lightweight probe trained on frozen hidden states to *measure* how well
    tonal information survives after compression (evaluation only, not a
    training objective)."""

    def __init__(self, hidden_dim: int, num_tones: int = 7):
        super().__init__()
        self.probe = nn.Linear(hidden_dim, num_tones)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.probe(hidden_states)

    def train_probe(self, hidden_states, tone_labels, mask=None, lr: float = 1e-3, steps: int = 200):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.train()
        for _ in range(steps):
            optimizer.zero_grad()
            logits = self(hidden_states)
            if mask is not None:
                loss = F.cross_entropy(logits.view(-1, self.probe.out_features), tone_labels.view(-1), reduction='none')
                loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-8)
            else:
                loss = F.cross_entropy(logits.view(-1, self.probe.out_features), tone_labels.view(-1))
            loss.backward()
            optimizer.step()

    def score_preservation(self, hidden_states, tone_labels, mask=None) -> float:
        self.eval()
        with torch.no_grad():
            preds = self(hidden_states).argmax(dim=-1)
            correct = (preds == tone_labels).float()
            if mask is not None:
                correct = correct * mask
                return (correct.sum() / (mask.sum() + 1e-8)).item()
            return correct.mean().item()


class ToneAugmentedTrainer:
    """Training helper bundling tone embedding injection, the phonological
    consistency loss, and tone-preservation-rate tracking via a probe."""

    def __init__(self, model, tokenizer, config: ToneAwareConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.tone_analyzer = get_tone_analyzer()
        hidden_dim = model.config.hidden_size
        self.tone_augmentation = ToneEmbeddingAugmentation(d_model=hidden_dim, tone_embed_dim=config.tone_embed_dim)
        self.tone_loss = PhonologicalConsistencyLoss(hidden_dim=hidden_dim, lambda_tone=config.lambda_tone)
        self.tone_probe = TonePreservationProbe(hidden_dim=hidden_dim)

    def compute_tone_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        tone_labels = torch.full((B, S), -100, dtype=torch.long)
        for b in range(B):
            for s in range(S):
                tid = input_ids[b, s].item()
                if tid == self.tokenizer.pad_token_id:
                    continue
                t = self.tokenizer.decode([tid]).replace('▁', ' ').replace('Ġ', ' ').strip()
                dominant = self.tone_analyzer.get_dominant_tone(t[:20])
                tone_labels[b, s] = TONE_NAME_TO_ID.get(dominant or 'ngang', 0)
        return tone_labels

    def train_step(self, input_ids, attention_mask, labels, optimizer) -> Dict[str, float]:
        self.model.train()
        optimizer.zero_grad()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        lm_loss = outputs.loss
        tone_labels = self.compute_tone_labels(input_ids).to(input_ids.device)
        tone_l = self.tone_loss(outputs.hidden_states[-1], tone_labels, attention_mask)
        total_loss = lm_loss + tone_l
        total_loss.backward()
        optimizer.step()
        return {'lm_loss': lm_loss.item(), 'tone_loss': tone_l.item(), 'total_loss': total_loss.item()}

    def compute_tone_preservation_rate(self, input_ids, attention_mask=None, train_steps: int = 100) -> float:
        """Train a lightweight probe on this model's hidden states and report
        tone-prediction accuracy -- higher means more tone info retained."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1].detach().clone()
        tone_labels = self.compute_tone_labels(input_ids).to(hidden_states.device)
        if self.tone_probe.probe.in_features != hidden_states.shape[-1]:
            self.tone_probe = TonePreservationProbe(hidden_dim=hidden_states.shape[-1]).to(hidden_states.device)
        self.tone_probe.train_probe(hidden_states, tone_labels, attention_mask, steps=train_steps)
        return self.tone_probe.score_preservation(hidden_states, tone_labels, attention_mask)
