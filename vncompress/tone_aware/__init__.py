"""
Tone-aware module for vncompress.

Core utilities (no torch needed):
  - VietnameseToneAnalyzer
  - is_vietnamese, strip_tone, extract_tone_marks

Advanced (requires torch):
  - ToneAwareScorer, ToneEmbeddingAugmentation
  - PhonologicalConsistencyLoss, ToneAugmentedTrainer
"""
from .vietnamese_tones import (
    VietnameseToneAnalyzer,
    ToneInfo,
    TokenToneInfo,
    TONE_NAME_TO_ID,
    TONE_ID_TO_NAME,
    TONE_CONTRAST,
    is_vietnamese,
    strip_tone,
    extract_tone_marks,
    get_tone_analyzer,
)

# Lazy import for torch-dependent classes
_tone_scoring_available = False
ToneAwareConfig = None
ToneAwareScorer = None
ToneEmbeddingAugmentation = None
PhonologicalConsistencyLoss = None
ToneAugmentedTrainer = None

def _ensure_tone_scoring():
    global ToneAwareConfig, ToneAwareScorer, ToneEmbeddingAugmentation
    global PhonologicalConsistencyLoss, ToneAugmentedTrainer, _tone_scoring_available
    if not _tone_scoring_available:
        from .tone_scoring import (
            ToneAwareConfig as _TAC,
            ToneAwareScorer as _TAS,
            ToneEmbeddingAugmentation as _TEA,
            PhonologicalConsistencyLoss as _PCL,
            ToneAugmentedTrainer as _TAT,
        )
        ToneAwareConfig = _TAC
        ToneAwareScorer = _TAS
        ToneEmbeddingAugmentation = _TEA
        PhonologicalConsistencyLoss = _PCL
        ToneAugmentedTrainer = _TAT
        _tone_scoring_available = True

__all__ = [
    "VietnameseToneAnalyzer",
    "ToneInfo",
    "TokenToneInfo",
    "TONE_NAME_TO_ID",
    "TONE_ID_TO_NAME",
    "TONE_CONTRAST",
    "is_vietnamese",
    "strip_tone",
    "extract_tone_marks",
    "get_tone_analyzer",
]
