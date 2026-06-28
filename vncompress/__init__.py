"""
vncompress — Context Compression Toolkit
=========================================
Research framework for tone-aware and morphology-aware context compression
for Vietnamese and other low-resource languages.

Key modules:
  - tone_aware  : Vietnamese tone detection and tone-aware scoring (no torch needed)
  - morphology  : POS-based word classification and merge policies (no torch needed)
  - compressors : Compression methods (requires torch + transformers)
  - evaluation  : VCC-Bench benchmark and evaluation metrics (requires torch)

Quick start (no GPU needed):
  >>> from vncompress.tone_aware import VietnameseToneAnalyzer, is_vietnamese
  >>> analyzer = VietnameseToneAnalyzer()
  >>> analyzer.analyze_tokens(['xin', 'chào', 'các', 'bạn'])

Author: Thanthien
Date: 2026-06-28
"""

__version__ = "0.1.0"
__author__ = "Thanthien"

# Lazy imports for modules requiring torch
# Users should import submodules directly:
#   from vncompress.tone_aware import ...
#   from vncompress.compressors import ...
