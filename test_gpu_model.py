#!/usr/bin/env python3
"""
GPU test for GTX 1060 6GB with Qwen2.5-0.5B-Instruct.
Tests: model loading, compression methods, generation quality.
"""
import sys, os, time, io, gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch

PASS = 0
FAIL = 0

def check(desc, condition):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {desc}")
        PASS += 1
    else:
        print(f"  [FAIL] {desc}")
        FAIL += 1

print("=" * 60)
print("  VNCOMPRESS -- GPU Test (Qwen 0.5B)")
print("=" * 60)

# --- GPU info ---
print("\n--- GPU Info ---")
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  VRAM: {vram_gb:.1f} GB")
print(f"  CUDA: {torch.version.cuda}")
print(f"  PyTorch: {torch.__version__}")
print(f"  VRAM used: {torch.cuda.memory_allocated(0)/(1024**3):.2f} GB")

# --- Load model ---
print("\n--- Loading Qwen2.5-0.5B-Instruct ---")
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
USE_INT4 = vram_gb < 8

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if USE_INT4:
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        quantization_config=bnb_cfg, device_map='auto',
    )
    print(f"  Model loaded in INT4")
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        torch_dtype=torch.float16, device_map='auto',
    )
    print(f"  Model loaded in FP16")
model.eval()
load_time = time.time() - t0

params = sum(p.numel() for p in model.parameters()) / 1e6
vram_used = torch.cuda.memory_allocated(0) / (1024**3)
print(f"  Params: {params:.0f}M | Load time: {load_time:.1f}s | VRAM: {vram_used:.2f} GB")
check("Model loaded", True)

# --- Pure Python compressors (CPU, 0 VRAM extra) ---
print("\n--- Test 1: No-Model Compressors ---")
from vncompress.compressors.no_model import (
    NoModelToneCompressor, NoModelMorphCompressor,
    NoModelCombinedCompressor, NoModelBaselineCompressor,
)

# Vietnamese sample
sample = (
    "Hom nay troi dep, chung toi di dao trong cong vien. "
    "Buoi chieu co mua rao nhe, nhung khong anh huong nhieu den cuoc di choi. "
    "Chung toi da chup nhieu anh dep va an nhieu mon ngon."
) * 3
input_ids = tokenizer.encode(sample, add_special_tokens=False)
n = len(input_ids)
print(f"  Input: {n} tokens | 4x compression target: {n//4} tokens")

compressors = [
    ('baseline_first',   NoModelBaselineCompressor(tokenizer, mode='first')),
    ('baseline_random',  NoModelBaselineCompressor(tokenizer, mode='random')),
    ('tone_only',        NoModelToneCompressor(tokenizer)),
    ('morph_only',       NoModelMorphCompressor(tokenizer)),
    ('combined',         NoModelCombinedCompressor(tokenizer)),
]

for name, comp in compressors:
    t0 = time.time()
    r = comp.compress(input_ids, target_ratio=4.0)
    elapsed = (time.time() - t0) * 1000
    check(f"  {name}: {r.original_length}->{r.compressed_length} ({r.compression_ratio:.1f}x) {elapsed:.1f}ms",
          r.compressed_length < r.original_length)

# --- ToneAwareCompressor (needs model for base scoring) ---
print("\n--- Test 2: ToneAwareCompressor ---")
from vncompress.compressors.tone_aware import ToneAwareCompressor
from vncompress.compressors.base import CompressionConfig

cfg = CompressionConfig(target_ratio=4.0)

t0 = time.time()
tac = ToneAwareCompressor(tokenizer, model, config=cfg, device='cuda',
                          alpha=0.5, beta=0.3, gamma=0.4, base_method='llmlingua')
result = tac.compress(input_ids)
elapsed = (time.time() - t0) * 1000
check(f"  ToneAware: {result.original_length}->{result.compressed_length} ({result.compression_ratio:.1f}x)",
      result.compressed_length < result.original_length)
check(f"  Processing time: {elapsed:.0f}ms", elapsed > 0)
if 'tone_preservation_rate' in result.metadata:
    check(f"  Tone preservation rate: {result.metadata['tone_preservation_rate']:.3f}",
          result.metadata['tone_preservation_rate'] >= 0)

# --- MorphologyAwareCompressor ---
print("\n--- Test 3: MorphologyAwareCompressor ---")
from vncompress.compressors.tone_aware import MorphologyAwareCompressor

t0 = time.time()
mac = MorphologyAwareCompressor(tokenizer, model, config=cfg, device='cuda',
                                 f_func=0.4, f_content=1.2, f_redup=0.6, f_compound=1.5)
result = mac.compress(input_ids)
elapsed = (time.time() - t0) * 1000
check(f"  MorphAware: {result.original_length}->{result.compressed_length} ({result.compression_ratio:.1f}x)",
      result.compressed_length < result.original_length)
check(f"  Has class distribution", 'original_class_distribution' in result.metadata)

# --- CombinedCompressor ---
print("\n--- Test 4: CombinedCompressor ---")
from vncompress.compressors.tone_aware import CombinedCompressor

t0 = time.time()
cc = CombinedCompressor(tokenizer, model, config=cfg, device='cuda',
                        alpha=0.5, beta=0.3, gamma=0.4,
                        f_func=0.4, f_content=1.2, tone_weight=0.6)
result = cc.compress(input_ids)
elapsed = (time.time() - t0) * 1000
check(f"  Combined: {result.original_length}->{result.compressed_length} ({result.compression_ratio:.1f}x)",
      result.compressed_length < result.original_length)
check(f"  Has mean tone & morph multipliers",
      'mean_tone_multiplier' in result.metadata and 'mean_morph_multiplier' in result.metadata)

# --- Generation test ---
print("\n--- Test 5: Generation with Compression ---")
query = "Thoi tiet hom nay nhu the nao?"
query_ids = tokenizer.encode(query, add_special_tokens=False)

for name, comp in compressors[:3]:
    r = comp.compress(input_ids, target_ratio=4.0)
    full_input = r.compressed_ids + query_ids
    input_t = torch.tensor([full_input]).to('cuda')

    with torch.no_grad():
        outputs = model.generate(input_t, max_new_tokens=32, temperature=0.0,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
    gen_text = tokenizer.decode(outputs[0][len(full_input):], skip_special_tokens=True)
    check(f"  {name}: generated '{gen_text[:50]}...'", len(gen_text) > 0)

# --- Memory check ---
print(f"\n--- VRAM Final ---")
vram_end = torch.cuda.memory_allocated(0) / (1024**3)
print(f"  VRAM used: {vram_end:.2f} GB / {vram_gb:.1f} GB")
check(f"  VRAM well within limits (< 5GB)", vram_end < 5.0)

# --- VRAM saved by compression ---
print("\n--- Compression Savings ---")
original_ids = tokenizer.encode(sample, add_special_tokens=False)
r = NoModelCombinedCompressor(tokenizer).compress(original_ids, target_ratio=4.0)
saved_pct = (len(original_ids) - len(r.compressed_ids)) / len(original_ids) * 100
check(f"  Combined 4x: saved {saved_pct:.0f}% tokens", saved_pct > 40)

# --- Summary ---
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed  ({PASS+FAIL} total)")
print(f"{'='*60}")

if FAIL:
    print(f"\n  !! {FAIL} FAILURES !!")
    sys.exit(1)
else:
    print(f"\n  All clear on GTX 1060 6GB!")
    print(f"  Model: Qwen2.5-0.5B (~{params:.0f}M, {vram_used:.1f}GB VRAM)")
    sys.exit(0)
