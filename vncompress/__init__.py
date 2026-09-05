"""
vncompress — LACC (Language-Aware Context Compression) research codebase.

Modules:
  config       : ExperimentConfig, seed control, environment snapshotting
  compression  : CompressionResult/Config, BaseCompressor, baselines, LACCCompressor
  linguistics  : Vietnamese tone analysis, Tone Preservation Rate, morphology
  models       : model/tokenizer/scorer loading, device utilities
  training     : LACC model training + SLM/tone-probe training + validation
  evaluation   : VCC-Bench, metrics, significance testing, method taxonomy
  utils        : small shared helpers (JSON I/O)

Quick start (no GPU needed):
    >>> from vncompress.linguistics import get_tone_analyzer, is_vietnamese
    >>> get_tone_analyzer().analyze_tokens(['xin', 'chào', 'các', 'bạn'])

    >>> from vncompress.compression import LACCCompressor, CompressionConfig
    >>> comp = LACCCompressor(tokenizer, model=None, use_perplexity=False)  # 0 VRAM
    >>> comp.compress(input_ids)
"""

__version__ = "0.2.0"
