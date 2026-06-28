"""
Morphology module for vncompress.
"""
from .merge_policy import (
    MorphologyConfig,
    MorphologyAnalyzer,
    WordInfo,
    WordClass,
    TokenInflationAnalyzer,
    VIETNAMESE_FUNCTION_WORDS,
    get_morphology_analyzer,
)

__all__ = [
    "MorphologyConfig",
    "MorphologyAnalyzer",
    "WordInfo",
    "WordClass",
    "TokenInflationAnalyzer",
    "VIETNAMESE_FUNCTION_WORDS",
    "get_morphology_analyzer",
]
