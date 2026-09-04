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
    global SLMScorerCompressor, SLMPerplexityScorer
    global SLMToneProbeCompressor, SLMToneProbeScorer
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
        from .slm_scorer import (
            SLMScorerCompressor as _SLM, SLMPerplexityScorer as _SLMS,
        )
        from .slm_tone_probe import (
            SLMToneProbeCompressor as _STP, SLMToneProbeScorer as _STPS,
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
    SLMScorerCompressor = _SLM
    SLMPerplexityScorer = _SLMS
    SLMToneProbeCompressor = _STP
    SLMToneProbeScorer = _STPS

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
        # LACC lightweight: real SLM as the perplexity scorer. `_base` is the
        # same scorer WITHOUT the LoRA adapter -- the ablation that isolates
        # what fine-tuning contributed. Both need --scorer-adapter-dir.
        'slm_scorer': SLMScorerCompressor,
        'slm_scorer_base': SLMScorerCompressor,
        # LACC full/INT4 tier: the trained tone probe (Sect. 3.4) as the live
        # tone signal. `_rule` is the controlled ablation -- same SLM, same
        # perplexity/morphology signals, tone from the dictionary heuristic
        # instead of the probe -- so the two isolate exactly what the probe adds.
        'slm_tone_probe': SLMToneProbeCompressor,
        'slm_tone_probe_rule': SLMToneProbeCompressor,
    }

    # These share one class, so the class alone can't say which is which.
    _SLM_METHOD_KWARGS = {
        'slm_scorer': {'use_adapter': True, 'name': 'slm_scorer'},
        'slm_scorer_base': {'use_adapter': False, 'name': 'slm_scorer_base'},
    }
    _TONE_PROBE_METHOD_KWARGS = {
        'slm_tone_probe': {'tone_source': 'model', 'name': 'slm_tone_probe'},
        'slm_tone_probe_rule': {'tone_source': 'rule', 'name': 'slm_tone_probe_rule'},
    }

    def _create_compressor(method, tokenizer, model=None, config=None, device='cuda', **kwargs):
        if method not in COMPRESSOR_REGISTRY:
            raise ValueError(f"Unknown method: {method}. Available: {list(COMPRESSOR_REGISTRY.keys())}")
        cls = COMPRESSOR_REGISTRY[method]
        if method in _SLM_METHOD_KWARGS:
            # slm_scorer needs no tone probe; drop the tone-probe-only kwarg.
            sk = {k: v for k, v in kwargs.items() if k != 'tone_probe_path'}
            return cls(tokenizer, model, config, device, **{**sk, **_SLM_METHOD_KWARGS[method]})
        if method in _TONE_PROBE_METHOD_KWARGS:
            return cls(tokenizer, model, config, device, **{**kwargs, **_TONE_PROBE_METHOD_KWARGS[method]})
        # Only the SLM methods understand scorer kwargs; drop them elsewhere so
        # a single --scorer-adapter-dir / --tone-probe-path flag can be passed
        # for every method.
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in ('scorer_adapter_dir', 'use_adapter', 'scorer', 'tone_probe_path')}
        if method in ('tone_aware', 'morphology_aware', 'combined', 'snapkv'):
            return cls(tokenizer, model, config, device, **kwargs)
        if method in ('llmlingua', 'llmlingua_small', 'selective'):
            # Keyword args, not positional: LLMLinguaCompressor's third
            # parameter is `small_model`, so `cls(tokenizer, model, config)`
            # bound the CompressionConfig into the small-model slot and left
            # config None. It only avoided crashing because run_benchmark.py
            # happened to pass config=None.
            return cls(tokenizer, model=model, config=config, device=device, **kwargs)
        return cls(tokenizer, model=model, config=config)

    create_compressor = _create_compressor
    _compressors_loaded = True


def __getattr__(name):
    if name in (
        'BaseCompressor', 'CompressionResult', 'CompressionConfig',
        'NoCompressor', 'RandomCompressor',
        'LLMLinguaCompressor', 'LLMLinguaWithSmallModel',
        'SnapKVCompressor', 'SelectiveContextCompressor',
        'ToneAwareCompressor', 'MorphologyAwareCompressor', 'CombinedCompressor',
        'SLMScorerCompressor', 'SLMPerplexityScorer',
        'SLMToneProbeCompressor', 'SLMToneProbeScorer',
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
    "SLMScorerCompressor", "SLMPerplexityScorer",
    "SLMToneProbeCompressor", "SLMToneProbeScorer",
    "COMPRESSOR_REGISTRY", "create_compressor",
]
