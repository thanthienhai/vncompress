"""
Model loading & device utilities — the single place LACC loads a model.
=========================================================================
Every script that needs a model (train.py, benchmark.py, evaluate.py) calls
into this module instead of duplicating `AutoModelForCausalLM.from_pretrained`
calls with their own quantization/device_map logic. That used to be copy-pasted
across run_benchmark.py, run_ablation.py, run_training.py, run_train_slm.py,
evaluate_slm.py and vncompress/compressors/{external_scorer,slm_scorer,
slm_tone_probe}.py, each with slightly different (and occasionally buggy)
device_map handling.

Windows/CUDA note: on this single-GPU Windows/CUDA dev machine, any non-None
`device_map` (`'auto'` or `{'': 0}`) makes transformers run
`caching_allocator_warmup()` -> `torch.cuda.mem_get_info()`, which has been
observed to segfault (access violation). Quantized (4-bit/8-bit) loads need a
device_map at load time and can't avoid this; unquantized loads are loaded
plain and moved with `.to(device)` afterward, which sidesteps the crash
entirely. See docs/training.md for the full writeup.

Contents:
  - detect_gpu / clear_gpu_memory / get_vram_info / print_vram_status
  - load_model, load_model_4bit, load_model_8bit  (generation / any causal LM)
  - load_tiny_model                                (small scorer models)
  - resize_embeddings_if_needed                     (embedding/tokenizer mismatch)
  - load_scorer                                      (LACC perplexity/tone-probe scorer)
  - VRAMManager                                      (swap scorer <-> generator on <16GB)
"""

from __future__ import annotations

import gc
import os
from typing import Optional, Tuple

import torch

# ============================================================================
# Hardware detection & memory utilities
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
    if not torch.cuda.is_available():
        return info

    info['has_cuda'] = True
    info['num_gpus'] = torch.cuda.device_count()
    info['gpu_name'] = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    info['vram_gb'] = props.total_memory / (1024**3)
    info['colab_environment'] = 'COLAB_GPU' in os.environ
    info['kaggle_environment'] = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

    vram = info['vram_gb']
    if vram >= 80:
        info['recommended_bits'], info['recommended_model_size'] = 16, '70B'
    elif vram >= 40:
        info['recommended_bits'], info['recommended_model_size'] = 8, '13B'
    elif vram >= 15:
        info['recommended_bits'], info['recommended_model_size'] = 4, '7B'
    elif vram >= 10:
        info['recommended_bits'], info['recommended_model_size'] = 4, '3B'
    else:
        info['recommended_bits'], info['recommended_model_size'] = 4, '1.5B'
    return info


def clear_gpu_memory() -> None:
    """Free GPU memory between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_vram_info() -> dict:
    """Current VRAM status (all zero on CPU)."""
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


def print_vram_status(prefix: str = '') -> None:
    info = get_vram_info()
    print(f"{prefix}VRAM: {info['allocated_gb']}GB used / "
          f"{info['total_gb']}GB total ({info['free_gb']}GB free)")


class VRAMManager:
    """Swap a scorer and a generator on/off GPU so they never coexist --
    useful on a 16GB card where both would not fit at once."""

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.active_model = None

    def load_scorer(self, scorer) -> None:
        if self.active_model:
            self._unload_current()
        scorer.to(self.device)
        self.active_model = 'scorer'

    def load_generator(self, model) -> None:
        if self.active_model:
            self._unload_current()
        if hasattr(model, 'to'):
            model.to(self.device)
        self.active_model = 'generator'

    def _unload_current(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.active_model = None

    vram_info = staticmethod(get_vram_info)


# ============================================================================
# Generation model loading
# ============================================================================


def load_tokenizer(model_name: str, trust_remote_code: bool = True):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name: str,
    device: str = 'cuda',
    dtype: str = 'float16',
    trust_remote_code: bool = True,
) -> Tuple[object, object]:
    """Load a plain (unquantized) causal LM + tokenizer for generation or scoring.

    Deliberately no `device_map` -- see module docstring for why. Loads on CPU
    then moves with `.to(device)`, which is the pattern every entrypoint in
    this repo now shares.
    """
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model_name, trust_remote_code)
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype,
    )
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to('cuda')
    model.eval()
    return model, tokenizer


def load_model_4bit(
    model_name: str,
    device_map=None,
    max_memory: Optional[dict] = None,
    trust_remote_code: bool = True,
    use_flash_attention: bool = False,
) -> Tuple[object, object]:
    """Load a model in INT4 (bitsandbytes NF4) -- fits a 7B model in ~5GB VRAM.

    Quantized loads must specify device_map at load time. `{'': 0}` (pin to
    GPU 0) avoids accelerate's multi-device `get_balanced_memory` crash, but
    the `caching_allocator_warmup()` segfault mentioned in the module
    docstring is otherwise unavoidable for a quantized load.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if device_map is None:
        device_map = {'': 0} if torch.cuda.is_available() else 'cpu'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    model_kwargs = {
        'quantization_config': bnb_config,
        'device_map': device_map,
        'trust_remote_code': trust_remote_code,
        'torch_dtype': torch.float16,
        'low_cpu_mem_usage': False,  # see module docstring
    }
    if max_memory:
        model_kwargs['max_memory'] = max_memory
    if use_flash_attention:
        try:
            import flash_attn  # noqa: F401 -- import itself is the availability probe
            model_kwargs['attn_implementation'] = 'flash_attention_2'
        except (ImportError, ModuleNotFoundError):
            pass

    tokenizer = load_tokenizer(model_name, trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model, tokenizer


def load_model_8bit(
    model_name: str,
    device_map=None,
    trust_remote_code: bool = True,
) -> Tuple[object, object]:
    """Load an INT8-quantized model (~8GB for 7B)."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if device_map is None:
        device_map = {'': 0} if torch.cuda.is_available() else 'cpu'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    )
    tokenizer = load_tokenizer(model_name, trust_remote_code)
    return model, tokenizer


TINY_MODEL_IDS = {
    'smollm2-135m': 'HuggingFaceTB/SmolLM2-135M-Instruct',
    'smollm2-360m': 'HuggingFaceTB/SmolLM2-360M-Instruct',
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'gemma-2b': 'google/gemma-2-2b-it',
    'llama-3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct',
}


def load_tiny_model(
    model_id: str = 'smollm2-135m',
    quantize: bool = True,
    device: str = 'cuda',
) -> Tuple[object, object]:
    """Load a tiny (<500M param) model for fast perplexity scoring.

    Falls back to plain FP16/CPU if INT4 loading fails for any reason.
    """
    model_name = TINY_MODEL_IDS.get(model_id, model_id)
    if quantize and device == 'cuda':
        try:
            return load_model_4bit(model_name)
        except Exception:
            pass
    return load_model(model_name, device=device, dtype='float16' if device == 'cuda' else 'float32')


def auto_setup(
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct',
    use_tiny_scorer: bool = True,
    tiny_scorer_id: str = 'smollm2-135m',
) -> dict:
    """Auto-configure a generation model (+ optional tiny scorer) for the
    detected hardware. Returns a dict with model/tokenizer/hw_info and,
    if requested, tiny_model/tiny_tokenizer."""
    hw = detect_gpu()
    result: dict = {'hw_info': hw}

    if hw['has_cuda'] and hw['vram_gb'] >= 14:
        model, tokenizer = (
            load_model_4bit(model_name) if hw['recommended_bits'] == 4
            else load_model_8bit(model_name)
        )
    elif hw['has_cuda'] and hw['vram_gb'] >= 6:
        model, tokenizer = load_model_4bit('Qwen/Qwen2.5-1.5B-Instruct')
    else:
        model, tokenizer = load_model('Qwen/Qwen2.5-1.5B-Instruct', device='cpu', dtype='float32')

    result['model'], result['tokenizer'] = model, tokenizer

    if use_tiny_scorer and hw['has_cuda']:
        try:
            result['tiny_model'], result['tiny_tokenizer'] = load_tiny_model(tiny_scorer_id, quantize=True)
        except Exception:
            result['tiny_model'], result['tiny_tokenizer'] = model, tokenizer

    return result


# ============================================================================
# Embedding resize (tokenizer/checkpoint mismatch)
# ============================================================================


def resize_embeddings_if_needed(model, tokenizer) -> bool:
    """Grow the embedding matrix to cover every tokenizer id, deterministically.

    Some community checkpoints (e.g. chronopt-research/vietnamese-gpt2-base)
    ship a tokenizer with more ids than the checkpoint has embedding rows.
    Left un-resized, an id past the last row crashes with a CUDA
    `srcIndex < srcSelectDimSize` assertion the first time it's padded into a
    batch (it is usually also `pad_token_id`, so this fires almost immediately).

    New rows are zeroed rather than left to `resize_token_embeddings`'s random
    mean-resizing: the row is never a prediction target (padding is masked to
    -100), and a deterministic row means an adapter trained afterward does not
    need to ship the whole embedding matrix to be reloadable (PEFT turns on
    `save_embedding_layers` as soon as embeddings are resized).

    Returns True if a resize happened.
    """
    if len(tokenizer) == model.get_input_embeddings().weight.shape[0]:
        return False
    old_rows = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        model.get_input_embeddings().weight[old_rows:].zero_()
        out = model.get_output_embeddings()
        if out is not None and out.weight.shape[0] >= len(tokenizer):
            out.weight[old_rows:].zero_()
    return True


def lora_target_modules(model) -> list:
    """PEFT target module names by architecture (GPT-2 SLM vs Qwen/LLaMA-style)."""
    model_type = getattr(model.config, 'model_type', '')
    if model_type in {'gpt2', 'gpt_neo', 'gptj'}:
        return ['c_attn', 'c_proj', 'c_fc']
    if model_type.startswith('qwen') or model_type in {'llama', 'mistral'}:
        return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    raise ValueError(f"Unsupported model_type={model_type!r}. Add its PEFT target modules to lora_target_modules().")


# ============================================================================
# LACC scorer loading (the SLM behind the perplexity / trained tone-probe signals)
# ============================================================================


def load_scorer(
    adapter_dir: str,
    tone_probe_path: Optional[str] = None,
    use_adapter: bool = True,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    load_4bit: bool = False,
):
    """Load the SLM that supplies LACC's perplexity signal, and -- if
    `tone_probe_path` is given -- the trained tone probe (see
    linguistics.PhonologicalConsistencyLoss) that supplies its trained-model
    tone signal (compression.LACCScorer wraps the result).

    `adapter_dir` is a LoRA adapter directory produced by
    `train.py --mode slm` (e.g. `models/slm/final`), or a plain HuggingFace
    model id / local model dir. `use_adapter=False` loads only the adapter's
    base model -- the ablation that isolates what fine-tuning contributed.
    """
    import os as _os

    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_adapter = False
    base_name = adapter_dir
    try:
        from peft import PeftConfig

        peft_config = PeftConfig.from_pretrained(adapter_dir)
        base_name = peft_config.base_model_name_or_path
        is_adapter = True
    except Exception:
        pass  # a plain HF model id / local model dir

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_4bit:
        from transformers import BitsAndBytesConfig

        dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            device_map={'': 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)

    resize_embeddings_if_needed(model, tokenizer)
    if is_adapter and use_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model = (model if load_4bit else model.to(device)).eval()
    model.config.use_cache = False

    tone_probe = None
    if tone_probe_path:
        from .linguistics import PhonologicalConsistencyLoss

        meta_path = _os.path.join(_os.path.dirname(_os.path.abspath(tone_probe_path)), 'tone_probe_meta.json')
        if _os.path.exists(meta_path):
            import json as _json

            with open(meta_path, encoding='utf-8') as _f:
                meta = _json.load(_f)
            if meta.get('base_model') and base_name and meta['base_model'] != base_name:
                raise ValueError(
                    f"tone_probe_meta.json says this probe was trained on "
                    f"{meta['base_model']!r}, but adapter_dir resolves to base model "
                    f"{base_name!r}. Pair the probe with its own adapter."
                )
        tone_probe = PhonologicalConsistencyLoss(hidden_dim=model.config.hidden_size, lambda_tone=0.0)
        state = torch.load(tone_probe_path, map_location='cpu', weights_only=True)
        probe_dim = state['tone_classifier.0.weight'].shape[1]
        if probe_dim != model.config.hidden_size:
            raise ValueError(
                f"Tone probe hidden dim ({probe_dim}) does not match the SLM's hidden "
                f"size ({model.config.hidden_size}). This probe was trained on a "
                f"different base model -- pass the tone_probe.pt that belongs to {base_name}."
            )
        tone_probe.load_state_dict(state)
        tone_probe = tone_probe.to(device=device, dtype=dtype).eval()

    from .compression import LACCScorer

    return LACCScorer(model, tokenizer, tone_probe=tone_probe)
