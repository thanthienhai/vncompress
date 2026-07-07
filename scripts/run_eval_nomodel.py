#!/usr/bin/env python3
"""VCC-Bench no-model evaluation with proper TPR computation."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vncompress.evaluation import VCCBench, VCCBenchConfig
from vncompress.compressors.no_model import (
    NoModelToneCompressor, NoModelMorphCompressor,
    NoModelCombinedCompressor, NoModelBaselineCompressor,
)
from vncompress.tone_aware.vietnamese_tones import VietnameseToneAnalyzer, TONE_NAME_TO_ID, get_tone_analyzer
from transformers import AutoTokenizer

tone_analyzer = get_tone_analyzer()

def compute_tpr(original_ids, compressed_ids, tokenizer, tone_analyzer):
    """Compute Tone Preservation Rate: fraction of non-ngang tones preserved."""
    def get_tone_ids(ids):
        tones = []
        for tid in ids:
            token_str = tokenizer.decode([tid])
            token_str = token_str.replace('\u2581', ' ').replace('Ġ', ' ').strip()
            tone_name = tone_analyzer.get_dominant_tone(token_str[:20])
            tone_id = TONE_NAME_TO_ID.get(tone_name or 'ngang', 0)
            tones.append(tone_id)
        return tones

    orig_tones = get_tone_ids(original_ids)
    comp_tones = get_tone_ids(compressed_ids)

    orig_has_tone = [1 for t in orig_tones if t > 0]
    comp_has_tone = [1 for t in comp_tones if t > 0]

    orig_count = sum(orig_has_tone)
    comp_count = sum(comp_has_tone)

    if orig_count == 0:
        return 1.0
    return comp_count / orig_count


print("=" * 60)
print("VCC-Bench: No-Model Compressors (0 VRAM, CPU)")
print("=" * 60)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading VCC-Bench dataset...")
config = VCCBenchConfig(
    methods=["none", "random", "tone_aware", "morphology_aware", "combined"],
    compression_ratios=[2.0, 4.0],
    output_dir="./results_no_model",
    device="cpu",
)
data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vcc_bench_data", "vcc_bench_v1.json"
)
bench = VCCBench.load_from_json(data_path, config)

compressors = {
    "none": NoModelBaselineCompressor(tokenizer, mode="first"),
    "random": NoModelBaselineCompressor(tokenizer, mode="random"),
    "tone_aware": NoModelToneCompressor(tokenizer),
    "morphology_aware": NoModelMorphCompressor(tokenizer),
    "combined": NoModelCombinedCompressor(tokenizer),
}

task_names_display = {
    "long_document_qa": "Long-Doc QA",
    "multi_turn_conversation": "Multi-turn Conv",
    "needle_in_haystack": "Needle-in-Haystack",
    "agent_tool_calling": "Agent Tool-Calling",
    "cross_lingual": "Cross-lingual",
}

method_names_display = {
    "none": "No Compression",
    "random": "Random",
    "tone_aware": "Tone-Aware",
    "morphology_aware": "Morph-Aware",
    "combined": "Combined",
}

all_results = {}
total_start = time.time()

for task_name, samples in bench.samples.items():
    if task_name not in config.tasks or not samples:
        continue
    n_eval = min(5, len(samples))
    print(f"\n--- {task_names_display.get(task_name, task_name)} ({n_eval} samples) ---")
    all_results[task_name] = {}

    for ratio in config.compression_ratios:
        print(f"  Ratio {ratio}x:")

        for method_name, compressor in compressors.items():
            crs, tprs, times_ms = [], [], []

            for sample in samples[:n_eval]:
                ids = tokenizer.encode(sample.context, add_special_tokens=False)
                if len(ids) < 10:
                    continue
                try:
                    t0 = time.time()
                    r = compressor.compress(ids, target_ratio=ratio)
                    dt = (time.time() - t0) * 1000

                    cr = len(ids) / max(len(r.compressed_ids), 1)
                    tpr = compute_tpr(ids, r.compressed_ids, tokenizer, tone_analyzer)

                    crs.append(cr)
                    tprs.append(tpr)
                    times_ms.append(dt)
                except Exception as e:
                    pass

            if crs:
                avg_cr = sum(crs) / len(crs)
                avg_tpr = sum(tprs) / len(tprs) if tprs else 0
                avg_time = sum(times_ms) / len(times_ms)
                all_results[task_name][f"{method_name}_r{ratio}"] = {
                    "avg_cr": avg_cr, "avg_tpr": avg_tpr, "avg_time_ms": avg_time, "n": len(crs)
                }
                print(f"    {method_names_display.get(method_name, method_name):<15s} CR={avg_cr:.1f}x  TPR={avg_tpr:.3f}  {avg_time:.0f}ms")

total_elapsed = time.time() - total_start

# Summary table
print("\n" + "=" * 64)
print("VCC-BENCH v1.0 — RESULTS")
print(f"Total time: {total_elapsed:.1f}s | 0 VRAM | CPU only")
print("=" * 64)

for task_name in sorted(all_results.keys()):
    print(f"\n{task_names_display.get(task_name, task_name)}:")
    header = f"  {'Method':<15s} {'CR(2x)':>8s} {'TPR(2x)':>8s} {'CR(4x)':>8s} {'TPR(4x)':>8s}"
    sep = f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}"
    print(header)
    print(sep)
    for method_name in ["none", "random", "tone_aware", "morphology_aware", "combined"]:
        r2 = all_results[task_name].get(f"{method_name}_r2.0", {})
        r4 = all_results[task_name].get(f"{method_name}_r4.0", {})
        name = method_names_display.get(method_name, method_name)
        row = f"  {name:<15s} {r2.get('avg_cr',0):>8.2f} {r2.get('avg_tpr',0):>8.3f} {r4.get('avg_cr',0):>8.2f} {r4.get('avg_tpr',0):>8.3f}"
        print(row)

# Overall averages across all tasks
print("\n--- OVERALL AVERAGE (all 5 tasks) ---")
print(f"  {'Method':<15s} {'2x CR':>8s} {'2x TPR':>8s} {'4x CR':>8s} {'4x TPR':>8s} {'Time':>7s}")
print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
for method_name in ["none", "random", "tone_aware", "morphology_aware", "combined"]:
    for ratio_str in ["2.0", "4.0"]:
        crs, tprs, times = [], [], []
        for task_name in all_results:
            r = all_results[task_name].get(f"{method_name}_r{ratio_str}", {})
            if r.get("n", 0) > 0:
                crs.append(r["avg_cr"])
                if r.get("avg_tpr", 0) > 0:
                    tprs.append(r["avg_tpr"])
                times.append(r.get("avg_time_ms", 0))
        avg_cr = sum(crs) / len(crs) if crs else 0
        avg_tpr = sum(tprs) / len(tprs) if tprs else 0
        avg_time = sum(times) / len(times) if times else 0

    cr2 = sum(r.get("avg_cr",0) for t in all_results for r in [all_results[t].get(f"{method_name}_r2.0",{})] if r.get("n",0)>0)
    n2 = sum(1 for t in all_results if all_results[t].get(f"{method_name}_r2.0",{}).get("n",0)>0)
    tpr2 = sum(r.get("avg_tpr",0) for t in all_results for r in [all_results[t].get(f"{method_name}_r2.0",{})] if r.get("n",0)>0)
    cr4 = sum(r.get("avg_cr",0) for t in all_results for r in [all_results[t].get(f"{method_name}_r4.0",{})] if r.get("n",0)>0)
    n4 = sum(1 for t in all_results if all_results[t].get(f"{method_name}_r4.0",{}).get("n",0)>0)
    tpr4 = sum(r.get("avg_tpr",0) for t in all_results for r in [all_results[t].get(f"{method_name}_r4.0",{})] if r.get("n",0)>0)
    avg_time = sum(r.get("avg_time_ms",0) for t in all_results for r in [all_results[t].get(f"{method_name}_r2.0",{}), all_results[t].get(f"{method_name}_r4.0",{})] if r.get("n",0)>0)

    name = method_names_display.get(method_name, method_name)
    print(f"  {name:<15s} {cr2/max(n2,1):>8.2f} {tpr2/max(n2,1):>8.3f} {cr4/max(n4,1):>8.2f} {tpr4/max(n4,1):>8.3f} {avg_time:>6.0f}ms")

# Save
os.makedirs("./results_no_model", exist_ok=True)
out = {
    "model": "no-model (CPU only, 0 VRAM)",
    "gpu": "None (CPU evaluation)",
    "total_time_s": total_elapsed,
    "compressors": list(compressors.keys()),
    "results": all_results,
}
with open("./results_no_model/summary.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n\nResults saved: ./results_no_model/summary.json")
print("Done!")
