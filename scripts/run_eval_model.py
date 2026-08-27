#!/usr/bin/env python3
"""
Model-Based Compression Evaluation — Qwen2.5-0.5B-Instruct (cached, fits 6GB)
Compares: no-model vs model-based (perplexity + tone + attention) compressors
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vncompress.evaluation import VCCBench, VCCBenchConfig
from vncompress.compressors.no_model import (
    NoModelToneCompressor, NoModelCombinedCompressor,
    NoModelBaselineCompressor,
)
from vncompress.tone_aware.vietnamese_tones import TONE_NAME_TO_ID, get_tone_analyzer
from vncompress.tone_aware.tone_scoring import PhonologicalConsistencyLoss

tone_analyzer = get_tone_analyzer()

def compute_tpr(original_ids, compressed_ids, tokenizer):
    def get_tones(ids):
        tones = []
        for tid in ids:
            ts = tokenizer.decode([tid])
            ts = ts.replace('\u2581', ' ').replace('Ġ', ' ').strip()
            tone_name = tone_analyzer.get_dominant_tone(ts[:20])
            tones.append(TONE_NAME_TO_ID.get(tone_name or 'ngang', 0))
        return tones
    o = get_tones(original_ids)
    c = get_tones(compressed_ids)
    oc = sum(1 for t in o if t > 0)
    cc = sum(1 for t in c if t > 0)
    return cc / max(oc, 1) if oc > 0 else 1.0

print("=" * 64)
print("VCC-Bench: Model-Based Compression")
print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
print("=" * 64)

# ── Load model ──
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"\n[1/3] Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16,
)  # no device_map: see docs/benchmark.md (caching_allocator_warmup segfault)
model = model.to('cuda')
model.eval()
model.config.output_hidden_states = True
mem_used = torch.cuda.max_memory_allocated() / 1024**3
print(f"  Model loaded. VRAM used: {mem_used:.1f} GB")

# ── Init model-based tone probe ──
hidden_dim = model.config.hidden_size
tone_probe = PhonologicalConsistencyLoss(hidden_dim=hidden_dim, lambda_tone=0.0)
tone_probe.eval()
tone_probe.to(model.device)

# ── Load dataset ──
print("\n[2/3] Loading VCC-Bench...")
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vcc_bench_data", "vcc_bench_v1.json")
config = VCCBenchConfig(methods=["none","random","tone_aware","combined"], compression_ratios=[2.0, 4.0], output_dir="./results_model", device="cuda")
bench = VCCBench.load_from_json(data_path, config)

# ── Define compressors ──
def model_tone_compress(input_ids, target_ratio):
    """Model-based tone compression using hidden states + probe."""
    n = len(input_ids)
    target_len = max(int(n / target_ratio), 4)
    k = 2
    input_t = torch.tensor([input_ids]).to(model.device)
    with torch.no_grad():
        outputs = model(input_t, output_hidden_states=True)
        hs = outputs.hidden_states[-1]
    tone_probe.to(model.device)
    weights = tone_probe.score_importance(hs).squeeze(0).cpu()  # [S]
    weights = torch.clamp(weights, 0.5, 3.0)
    mid_weights = weights[k:n - k] if n > 2 * k else weights
    mid_budget = max(0, target_len - 2 * k)
    if mid_budget > 0 and len(mid_weights) > 0:
        _, top = torch.topk(mid_weights, min(mid_budget, len(mid_weights)))
        top = sorted(top.tolist())
        keep = list(range(k)) + [k + i for i in top] + list(range(n - k, n))
    else:
        keep = list(range(n))
    return [input_ids[i] for i in sorted(set(keep)) if i < n]

def model_perplexity_compress(input_ids, target_ratio):
    """LLMLingua-style: keep tokens with highest perplexity."""
    n = len(input_ids)
    target_len = max(int(n / target_ratio), 4)
    k = 2
    input_t = torch.tensor([input_ids]).to(model.device)
    with torch.no_grad():
        outputs = model(input_t, labels=input_t)
    logits = outputs.logits.squeeze(0)  # [S, V]
    shift_logits = logits[:-1]
    shift_labels = input_t[0, 1:]
    losses = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, reduction='none'
    )
    ppl = torch.cat([losses[:1], losses])  # pad first token
    ppl = 1.0 + ppl.cpu()
    ppl = torch.clamp(ppl, 0.5, 3.0)
    mid_ppl = ppl[k:n - k] if n > 2 * k else ppl
    mid_budget = max(0, target_len - 2 * k)
    if mid_budget > 0 and len(mid_ppl) > 0:
        _, top = torch.topk(mid_ppl, min(mid_budget, len(mid_ppl)))
        top = sorted(top.tolist())
        keep = list(range(k)) + [k + i for i in top] + list(range(n - k, n))
    else:
        keep = list(range(n))
    return [input_ids[i] for i in sorted(set(keep)) if i < n]

# ── Evaluate ──
print("\n[3/3] Running evaluation...")
methods_desc = {
    "none": ("No Compression (first-N)", NoModelBaselineCompressor(tokenizer, mode="first")),
    "random": ("Random Baseline", NoModelBaselineCompressor(tokenizer, mode="random")),
    "tone_aware": ("Tone-Aware (no-model, rule)", NoModelToneCompressor(tokenizer)),
    "combined": ("Combined (no-model, rule)", NoModelCombinedCompressor(tokenizer)),
}
model_methods = {
    "model_tone": ("Tone-Aware (model hidden states)", model_tone_compress),
    "model_ppl": ("Perplexity-Based (model perplexity)", model_perplexity_compress),
}

all_results = {}
total_start = time.time()

for task_name, samples in bench.samples.items():
    if task_name not in config.tasks or not samples:
        continue
    n_eval = min(3, len(samples))
    print(f"\n--- {task_name} ({n_eval} samples) ---")
    all_results[task_name] = {}

    for ratio in config.compression_ratios:
        print(f"  Ratio {ratio}x:")

        # No-model methods
        for method_name, (desc, compressor) in methods_desc.items():
            crs, tprs, times = [], [], []
            for sample in samples[:n_eval]:
                ids = tokenizer.encode(sample.context, add_special_tokens=False)
                if len(ids) < 10:
                    continue
                try:
                    t0 = time.time()
                    result = compressor.compress(ids, target_ratio=ratio)
                    dt = (time.time() - t0) * 1000
                    crs.append(len(ids) / max(len(result.compressed_ids), 1))
                    tprs.append(compute_tpr(ids, result.compressed_ids, tokenizer))
                    times.append(dt)
                except Exception:
                    pass
            if crs:
                avg_cr = sum(crs)/len(crs); avg_tpr = sum(tprs)/len(tprs); avg_t = sum(times)/len(times)
                all_results[task_name][f"{method_name}_r{ratio}"] = {"cr":avg_cr,"tpr":avg_tpr,"time_ms":avg_t,"n":len(crs)}
                print(f"    {desc[:42]:<42s} CR={avg_cr:.1f}x TPR={avg_tpr:.3f} {avg_t:.0f}ms")

        # Model-based methods
        for method_name, (desc, compress_fn) in model_methods.items():
            crs, tprs, times = [], [], []
            for sample in samples[:n_eval]:
                ids = tokenizer.encode(sample.context, add_special_tokens=False)
                if len(ids) < 10:
                    continue
                try:
                    t0 = time.time()
                    compressed = compress_fn(ids, target_ratio=ratio)
                    dt = (time.time() - t0) * 1000
                    crs.append(len(ids) / max(len(compressed), 1))
                    tprs.append(compute_tpr(ids, compressed, tokenizer))
                    times.append(dt)
                except Exception as e:
                    print(f"      ERROR: {str(e)[:80]}")
            if crs:
                avg_cr = sum(crs)/len(crs); avg_tpr = sum(tprs)/len(tprs); avg_t = sum(times)/len(times)
                all_results[task_name][f"{method_name}_r{ratio}"] = {"cr":avg_cr,"tpr":avg_tpr,"time_ms":avg_t,"n":len(crs)}
                print(f"    {desc[:42]:<42s} CR={avg_cr:.1f}x TPR={avg_tpr:.3f} {avg_t:.0f}ms (GPU)")

# Summary
total_elapsed = time.time() - total_start
print("\n" + "=" * 72)
print("MODEL-BASED vs NO-MODEL — COMPARISON")
print(f"Total time: {total_elapsed:.1f}s | Model: {MODEL_ID}")
print("=" * 72)

all_methods = ["none","random","tone_aware","combined","model_tone","model_ppl"]
display_names = {
    "none":"No Compression", "random":"Random",
    "tone_aware":"Tone-Aware (no-model)", "combined":"Combined (no-model)",
    "model_tone":"Tone-Aware (model)", "model_ppl":"Perplexity-Based (model)",
}

for ratio_str in ["2.0", "4.0"]:
    print(f"\n  COMPRESSION {ratio_str}x:")
    print(f"  {'Method':<35s} {'CR':>6s} {'TPR':>7s} {'Time':>8s}")
    print(f"  {'-'*35} {'-'*6} {'-'*7} {'-'*8}")
    for mn in all_methods:
        crs, tprs, times = [], [], []
        for tn in all_results:
            r = all_results[tn].get(f"{mn}_r{ratio_str}", {})
            if r.get("n",0) > 0:
                crs.append(r["cr"]); tprs.append(r["tpr"]); times.append(r["time_ms"])
        if crs:
            name = display_names.get(mn, mn)
            print(f"  {name:<35s} {sum(crs)/len(crs):>6.2f} {sum(tprs)/len(tprs):>7.3f} {sum(times)/len(times):>7.0f}ms")

mem_peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"\nPeak GPU memory: {mem_peak:.1f} GB")
os.makedirs("./results_model", exist_ok=True)
with open("./results_model/comparison.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print("Results saved: ./results_model/comparison.json")
print("Done!")
