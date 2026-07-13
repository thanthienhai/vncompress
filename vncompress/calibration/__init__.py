"""
Calibration — search/learn scoring weights instead of hand-picked constants.
"""

from .weight_search import (
    CalibrationObjective,
    SampleResult,
    grid_search,
    coordinate_ascent,
    calibrate_combined_compressor,
    calibrate_score_weights,
    DEFAULT_PARAM_CANDIDATES,
    DEFAULT_SCORE_WEIGHT_CANDIDATES,
)

__all__ = [
    "CalibrationObjective",
    "SampleResult",
    "grid_search",
    "coordinate_ascent",
    "calibrate_combined_compressor",
    "calibrate_score_weights",
    "DEFAULT_PARAM_CANDIDATES",
    "DEFAULT_SCORE_WEIGHT_CANDIDATES",
]
