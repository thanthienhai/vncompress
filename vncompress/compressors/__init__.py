"""
Compressors module — all compression methods.

No-model compressors (pure Python, 0 VRAM):
  - NoModelToneCompressor, NoModelMorphCompressor
  - NoModelCombinedCompressor, NoModelBaselineCompressor

External model scoring (tiny model, 0.3GB VRAM):
  - TinyModelScorer, EnhancedCompressor, create_tiny_scorer
  - ScoreWeights, VRAMManager

Torch-dependent compressors (lazy load, requires GPU):
  - LLMLinguaCompressor, SnapKVCompressor
  - ToneAwareCompressor, MorphologyAwareCompressor, CombinedCompressor

For T4/P100 (16GB):
  0 VRAM:   no_model compressors (tone + morphology only)
  0.3GB:    add external tiny model scoring (tone + morph + perplexity)
  5GB:      add INT4 7B for generation
  7.8GB:    all three together ✓ fits in 16GB
"""

# No-model compressors: always available (pure Python)
from .no_model import (
    NoModelResult,
    NoModelCompressor,
    NoModelToneCompressor,
    NoModelMorphCompressor,
    NoModelCombinedCompressor,
    NoModelBaselineCompressor,
    evaluate_no_model,
)

# External scorer (requires torch + transformers, but tiny model)
try:
    from .external_scorer import (
        ScoreWeights,
        TinyModelScorer,
        VRAMManager,
        EnhancedCompressor,
        create_tiny_scorer,
        TINY_MODEL_IDS,
    )
    _has_external_scorer = True
except ImportError:
    _has_external_scorer = False
    ScoreWeights = None
    TinyModelScorer = None
    VRAMManager = None
    EnhancedCompressor = None
    create_tiny_scorer = None
    TINY_MODEL_IDS = {}

# Lazy imports for torch-dependent compressors — NOT pre-assigned here
# so that __getattr__ triggers on first access.
_compressors_loaded = False

def _ensure_compressors():
    global _compressors_loaded, BaseCompressor, CompressionResult, CompressionConfig
    global NoCompressor, RandomCompressor
    global LLMLinguaCompressor, LLMLinguaWithSmallModel
    global SnapKVCompressor, SelectiveContextCompressor
    global ToneAwareCompressor, MorphologyAwareCompressor, CombinedCompressor
    global COMPRESSOR_REGISTRY, create_compressor

    if _compressors_loaded:
        return

    try:
        from .base import (
            BaseCompressor as _BC, CompressionResult as _CR, CompressionConfig as _CC,
            NoCompressor as _NC, RandomCompressor as _RC,
        )
        from .llmlingua import (
            LLMLinguaCompressor as _LL, LLMLinguaWithSmallModel as _LLS,
        )
        from .snapkv import (
            SnapKVCompressor as _SKV, SelectiveContextCompressor as _SCC,
        )
        from .tone_aware import (
            ToneAwareCompressor as _TAC, MorphologyAwareCompressor as _MAC,
            CombinedCompressor as _CCB,
        )
    except ImportError as e:
        raise ImportError(
            "Cannot load torch-dependent compressors. "
            "Install torch + transformers or use no_model compressors only. "
            f"Original error: {e}"
        )

    BaseCompressor = _BC
    CompressionResult = _CR
    CompressionConfig = _CC
    NoCompressor = _NC
    RandomCompressor = _RC
    LLMLinguaCompressor = _LL
    LLMLinguaWithSmallModel = _LLS
    SnapKVCompressor = _SKV
    SelectiveContextCompressor = _SCC
    ToneAwareCompressor = _TAC
    MorphologyAwareCompressor = _MAC
    CombinedCompressor = _CCB

    COMPRESSOR_REGISTRY = {
        'none': NoCompressor,
        'random': RandomCompressor,
        'llmlingua': LLMLinguaCompressor,
        'llmlingua_small': LLMLinguaWithSmallModel,
        'snapkv': SnapKVCompressor,
        'selective': SelectiveContextCompressor,
        'tone_aware': ToneAwareCompressor,
        'morphology_aware': MorphologyAwareCompressor,
        'combined': CombinedCompressor,
    }

    def _create_compressor(method, tokenizer, model=None, config=None, device='cuda', **kwargs):
        if method not in COMPRESSOR_REGISTRY:
            raise ValueError(f"Unknown method: {method}. Available: {list(COMPRESSOR_REGISTRY.keys())}")
        cls = COMPRESSOR_REGISTRY[method]
        if method == 'tone_aware':
            return cls(tokenizer, model, config, device, **kwargs)
        elif method == 'morphology_aware':
            return cls(tokenizer, model, config, device, **kwargs)
        elif method == 'combined':
            return cls(tokenizer, model, config, device, **kwargs)
        elif method in ('snapkv',):
            return cls(tokenizer, model, config, device, **kwargs)
        else:
            return cls(tokenizer, model, config)

    create_compressor = _create_compressor
    _compressors_loaded = True


def __getattr__(name):
    if name in (
        'BaseCompressor', 'CompressionResult', 'CompressionConfig',
        'NoCompressor', 'RandomCompressor',
        'LLMLinguaCompressor', 'LLMLinguaWithSmallModel',
        'SnapKVCompressor', 'SelectiveContextCompressor',
        'ToneAwareCompressor', 'MorphologyAwareCompressor', 'CombinedCompressor',
        'COMPRESSOR_REGISTRY', 'create_compressor',
    ):
        _ensure_compressors()
        return globals()[name]
    raise AttributeError(f"module 'vncompress.compressors' has no attribute '{name}'")

__all__ = [
    # No-model (always available)
    "NoModelResult", "NoModelCompressor",
    "NoModelToneCompressor", "NoModelMorphCompressor",
    "NoModelCombinedCompressor", "NoModelBaselineCompressor",
    "evaluate_no_model",
    # External scorer
    "ScoreWeights", "TinyModelScorer", "VRAMManager",
    "EnhancedCompressor", "create_tiny_scorer", "TINY_MODEL_IDS",
    # Torch-dependent (lazy load via _ensure_compressors())
    "BaseCompressor", "CompressionResult", "CompressionConfig",
    "NoCompressor", "RandomCompressor",
    "LLMLinguaCompressor", "LLMLinguaWithSmallModel",
    "SnapKVCompressor", "SelectiveContextCompressor",
    "ToneAwareCompressor", "MorphologyAwareCompressor", "CombinedCompressor",
    "COMPRESSOR_REGISTRY", "create_compressor",
]
