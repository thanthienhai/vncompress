"""
init utils
"""
from .lightweight import (
    detect_gpu,
    load_model_4bit,
    load_model_8bit,
    load_tiny_model,
    TINY_MODELS,
    clear_gpu_memory,
    get_vram_info,
    print_vram_status,
    benchmark_friendly_generate,
    auto_setup,
)

__all__ = [
    "detect_gpu",
    "load_model_4bit",
    "load_model_8bit",
    "load_tiny_model",
    "TINY_MODELS",
    "clear_gpu_memory",
    "get_vram_info",
    "print_vram_status",
    "benchmark_friendly_generate",
    "auto_setup",
]
