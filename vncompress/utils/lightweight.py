"""
Lightweight Model Loading — optimized for T4/P100 (16GB VRAM)
==============================================================

Supports:
  1. INT4 quantization (bitsandbytes) — fits 7B model in ~5GB VRAM
  2. INT8 quantization — fits 7B in ~8GB
  3. Tiny model scoring (<500M params, <1GB VRAM)
  4. CPU-only compression (no GPU needed for tone/morphology)
  5. Memory-efficient generation (flash_attention, min cache)

VRAM Budget (T4 16GB):
  ┌─────────────────────────────────┐
  │ Strategy           │ VRAM Used  │
  ├─────────────────────────────────┤
  │ INT4 7B            │ ~5.0 GB    │
  │ + KV cache (2K ctx)│ ~0.8 GB    │
  │ + Tiny scorer 0.5B │ ~0.5 GB    │
  │ + Overhead         │ ~2.0 GB    │
  │ TOTAL              │ ~8.3 GB ✓  │
  ├─────────────────────────────────┤
  │ No-model (CPU)     │ ~0.0 GB    │
  │ + INT4 7B          │ ~5.0 GB    │
  │ + KV cache         │ ~0.8 GB    │
  │ TOTAL              │ ~5.8 GB ✓✓ │
  └─────────────────────────────────┘

Usage:
  from vncompress.utils.lightweight import load_model_4bit, load_tiny_model

  model, tok = load_model_4bit('Qwen/Qwen2.5-7B-Instruct')
  tiny, _   = load_tiny_model('HuggingFaceTB/SmolLM2-135M-Instruct')
"""

import os
import gc
from typing import Tuple, Optional

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None


# ============================================================================
# Hardware Detection
# ============================================================================

def detect_gpu() -> dict:
    """Detect GPU and return info for auto-configuration."""
    info = {
        'has_cuda': False,
        'gpu_name': 'cpu',
        'vram_gb': 0,
        'num_gpus': 0,
        'recommended_bits': 4,
        'recommended_model_size': '0.5B',
        'colab_environment': False,
        'kaggle_environment': False,
    }

    if not _HAS_TORCH or not torch.cuda.is_available():
        return info
    
    info['has_cuda'] = True
    info['num_gpus'] = torch.cuda.device_count()
    info['gpu_name'] = torch.cuda.get_device_name(0)
    
    # VRAM detection
    props = torch.cuda.get_device_properties(0)
    info['vram_gb'] = props.total_memory / (1024**3)
    
    # Detect environment
    if 'COLAB_GPU' in os.environ:
        info['colab_environment'] = True
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        info['kaggle_environment'] = True
    
    # Auto-recommend settings
    vram = info['vram_gb']
    if vram >= 80:
        info['recommended_bits'] = 16
        info['recommended_model_size'] = '70B'
    elif vram >= 40:
        info['recommended_bits'] = 8
        info['recommended_model_size'] = '13B'
    elif vram >= 24:
        info['recommended_bits'] = 4
        info['recommended_model_size'] = '7B'
    elif vram >= 15:
        info['recommended_bits'] = 4
        info['recommended_model_size'] = '7B'
    elif vram >= 10:
        info['recommended_bits'] = 4
        info['recommended_model_size'] = '3B'
    else:
        info['recommended_bits'] = 4
        info['recommended_model_size'] = '1.5B'
    
    return info


# ============================================================================
# INT4 Quantized Model Loading
# ============================================================================

def load_model_4bit(
    model_name: str,
    device_map: str = 'auto',
    max_memory: Optional[dict] = None,
    trust_remote_code: bool = True,
    use_flash_attention: bool = False,
) -> Tuple[any, any]:
    """
    Load a model in INT4 quantization — fits 7B model in ~5GB VRAM.

    Args:
        model_name: HuggingFace model ID
        device_map: 'auto', 'cpu', or custom
        max_memory: Optional dict like {0: '12GB', 'cpu': '16GB'}
        trust_remote_code: Trust model code
        use_flash_attention: Enable Flash Attention 2 if available

    Returns:
        (model, tokenizer) tuple
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"[LIGHTWEIGHT] Loading {model_name} in INT4...")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB" if torch.cuda.is_available() else "")
    
    # INT4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',  # NormalFloat4 — best for LLMs
    )
    
    # Model kwargs
    model_kwargs = {
        'quantization_config': bnb_config,
        'device_map': device_map,
        'trust_remote_code': trust_remote_code,
        'torch_dtype': torch.float16,
        'low_cpu_mem_usage': True,
    }
    
    if max_memory:
        model_kwargs['max_memory'] = max_memory
    
    # Flash Attention 2 (optional, requires compatible GPU + flash-attn package)
    if use_flash_attention:
        try:
            import flash_attn
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            print("  Flash Attention 2 enabled")
        except (ImportError, ModuleNotFoundError):
            print("  Flash Attention 2 not available, using default")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    
    # Memory report
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"  GPU memory: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")
        print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory/(1024**3) - reserved):.1f} GB")
    
    return model, tokenizer


def load_model_8bit(
    model_name: str,
    device_map: str = 'auto',
    trust_remote_code: bool = True,
) -> Tuple[any, any]:
    """Load INT8 quantized model (~8GB for 7B)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


# ============================================================================
# Tiny Model for Scoring (Perplexity-based compression)
# ============================================================================

TINY_MODELS = {
    'smollm2-135m': 'HuggingFaceTB/SmolLM2-135M-Instruct',
    'smollm2-360m': 'HuggingFaceTB/SmolLM2-360M-Instruct',
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'gemma-2b': 'google/gemma-2-2b-it',
}

def load_tiny_model(
    model_id: str = 'smollm2-135m',
    quantize: bool = True,
    device: str = 'cuda',
) -> Tuple[any, any]:
    """
    Load a tiny model (<500M params) for fast perplexity scoring.

    Models:
      - smollm2-135m: 135M params, ~300MB VRAM (INT4)
      - smollm2-360m: 360M params, ~750MB VRAM (INT4)
      - qwen2.5-0.5b: 500M params, ~1GB VRAM (INT4)

    These are used for LLMLingua-style scoring:
      Instead of running the full 7B model for importance scoring,
      use this tiny model to estimate token importance quickly.

    Returns:
        (model, tokenizer) — note: tokenizer matches the tiny model,
        so embeddings won't match the main model. But for perplexity
        scoring, this is sufficient (we only need relative importance).
    """
    import torch
    model_name = TINY_MODELS.get(model_id, model_id)
    
    print(f"[LIGHTWEIGHT] Loading tiny scorer: {model_name} ({model_id})")
    
    if quantize and device == 'cuda':
        try:
            return load_model_4bit(model_name)
        except Exception as e:
            print(f"  INT4 failed ({e}), falling back to FP16")
    
    # Fallback: FP16 or CPU
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_kwargs = {'trust_remote_code': True}
    if device == 'cuda':
        model_kwargs['torch_dtype'] = torch.float16
        model_kwargs['device_map'] = 'auto'
    else:
        model_kwargs['device_map'] = 'cpu'
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


# ============================================================================
# Memory Management
# ============================================================================

def clear_gpu_memory():
    """Free GPU memory between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_vram_info() -> dict:
    """Get current VRAM status."""
    if not torch.cuda.is_available():
        return {'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': 0, 'total_gb': 0}
    
    alloc = torch.cuda.memory_allocated(0) / (1024**3)
    reserv = torch.cuda.memory_reserved(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return {
        'allocated_gb': round(alloc, 2),
        'reserved_gb': round(reserv, 2),
        'free_gb': round(total - alloc, 2),
        'total_gb': round(total, 2),
    }


def print_vram_status(prefix: str = ''):
    """Print current VRAM usage."""
    info = get_vram_info()
    print(f"{prefix}VRAM: {info['allocated_gb']}GB used / "
          f"{info['total_gb']}GB total ({info['free_gb']}GB free)")


# ============================================================================
# Benchmark helper for limited hardware
# ============================================================================

def benchmark_friendly_generate(
    model,
    tokenizer,
    input_ids: list,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    """
    Memory-efficient generation for T4/P100.

    Uses:
      - Min cache (no KV cache accumulation)
      - No gradient tracking
      - Immediate CPU offload of result
    """
    input_tensor = torch.tensor([input_ids], device=model.device)
    
    with torch.no_grad():
        # Use model.generate with memory-efficient settings
        outputs = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,  # Greedy → less memory
        )
    
    generated = outputs[0][len(input_ids):].cpu()
    return tokenizer.decode(generated, skip_special_tokens=True)


# ============================================================================
# Auto-Setup for Environment
# ============================================================================

def auto_setup(
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct',
    use_tiny_scorer: bool = True,
    tiny_scorer_id: str = 'smollm2-135m',
) -> dict:
    """
    Auto-configure everything for the current hardware.

    Returns dict with:
      - model, tokenizer: Main generation model (INT4 7B)
      - tiny_model, tiny_tokenizer: Tiny scoring model (optional)
      - hw_info: Hardware detection info
    """
    hw = detect_gpu()
    
    print("="*60)
    print("AUTO-SETUP for VNCOMPRESS")
    print("="*60)
    print(f"GPU: {hw['gpu_name']} ({hw['vram_gb']:.0f} GB VRAM)")
    print(f"Recommended: INT{hw['recommended_bits']}, {hw['recommended_model_size']} model")
    print(f"Env: {'Colab' if hw['colab_environment'] else 'Kaggle' if hw['kaggle_environment'] else 'Local'}")
    print("="*60)
    
    result = {'hw_info': hw}
    
    # Load main model
    if hw['has_cuda'] and hw['vram_gb'] >= 14:
        if hw['recommended_bits'] == 4:
            model, tokenizer = load_model_4bit(model_name)
        else:
            model, tokenizer = load_model_8bit(model_name)
    elif hw['has_cuda'] and hw['vram_gb'] >= 6:
        # Very small GPU — use smaller model
        print(f"\n  ⚠ VRAM limited ({hw['vram_gb']:.0f}GB), switching to 1.5B model")
        model, tokenizer = load_model_4bit('Qwen/Qwen2.5-1.5B-Instruct')
    else:
        # CPU only
        print("\n  ⚠ No GPU detected, loading small model on CPU")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-1.5B-Instruct',
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen2.5-1.5B-Instruct',
            trust_remote_code=True,
        )
    
    result['model'] = model
    result['tokenizer'] = tokenizer
    
    # Load tiny scorer
    if use_tiny_scorer and hw['has_cuda']:
        print(f"\n  Loading tiny scorer: {tiny_scorer_id}")
        try:
            tiny_model, tiny_tok = load_tiny_model(tiny_scorer_id, quantize=True)
            result['tiny_model'] = tiny_model
            result['tiny_tokenizer'] = tiny_tok
            print("  ✓ Tiny scorer loaded")
        except Exception as e:
            print(f"  ⚠ Tiny scorer failed ({e}), using main model for scoring")
            result['tiny_model'] = model
            result['tiny_tokenizer'] = tokenizer
    
    print_vram_status("\n  Final ")
    return result
