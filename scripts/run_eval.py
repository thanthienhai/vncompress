#!/usr/bin/env python3
"""
VCC-Bench Evaluation — Qwen2.5-1.5B-Instruct (fits GTX 1060 6GB)
Tests compression efficiency + tone preservation + generation quality.
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from vncompress.evaluation import VCCBench, VCCBenchConfig
from vncompress.evaluation.metrics import compute_rouge_l, compute_bleu
from vncompress.compressors import create_compressor
from vncompress.tone_aware import get_tone_analyzer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0

print("=" * 64)
print("VCC-Bench v1.0 — Evaluation Report")
print(f"Model: {MODEL_ID}")
print(f"GPU: {GPU_NAME} ({VRAM_GB:.1f} GB)")
print("=" * 64)

# ═══ 1. Load Tokenizer ═══
print("\n[1/5] Loading tokenizer...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"  Vocab: {len(tokenizer)}")

# ═══ 2. Load Model ═══
print(f"\n[2/5] Loading {MODEL_ID}...")
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
mem_model = torch.cuda.max_memory_allocated() / 1024**3
print(f"  Model loaded. GPU mem: {mem_model:.1f} GB")
torch.cuda.reset_peak_memory_stats()

# ═══ 3. Load VCC-Bench ═══
print("\n[3/5] Loading VCC-Bench dataset...")
data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "vcc_bench_data", "vcc_bench_v1.json"
)

config = VCCBenchConfig(
    methods=["none", "random", "tone_aware", "morphology_aware", "combined"],
    compression_ratios=[2.0, 4.0],
    output_dir="./results_qwen15",
    device="cuda",
    max_new_tokens=96,
    save_predictions=False,
)
bench = VCCBench.load_from_json(data_path, config)

NUM_EVAL_PER_TASK = 5
all_task_samples = {}
for task, samples in bench.samples.items():
    if task not in config.tasks:
        continue
    selected = samples[:NUM_EVAL_PER_TASK]
    all_task_samples[task] = selected

total_eval = sum(len(v) for v in all_task_samples.values())
print(f"  Evaluating {total_eval} samples across {len(all_task_samples)} tasks")

# ═══ 4. Run Evaluation ═══
print("\n[4/5] Running evaluation...")
print(f"  Methods: {config.methods}")
print(f"  Ratios: {config.compression_ratios}")

METHOD_NAMES = {
    "none": "No Compression",
    "random": "Random Baseline",
    "tone_aware": "Tone-Aware",
    "morphology_aware": "Morphology-Aware",
    "combined": "Combined",
}

all_results = {}
tone_analyzer = get_tone_analyzer()

total_start = time.time()

for ratio in config.compression_ratios:
    print(f"\n  ── Compression Ratio: {ratio}x ──")
    all_results[f"ratio_{ratio}"] = {}

    for method_name in config.methods:
        print(f"\n    {METHOD_NAMES.get(method_name, method_name)}:")
        method_results = {
            "compression_ratios": [],
            "token_savings": [],
            "times_ms": [],
            "tone_preservation_rates": [],
            "rouge_l": [],
            "bleu": [],
            "samples": 0,
        }

        try:
            compressor = create_compressor(method_name, tokenizer, model, device="cuda")
        except Exception as e:
            print(f"      SKIP: Cannot create compressor — {e}")
            all_results[f"ratio_{ratio}"][method_name] = {"error": str(e)}
            continue

        compressor.config.target_ratio = ratio

        for task_name, samples in all_task_samples.items():
            for sample in samples:
                input_ids = tokenizer.encode(sample.context, add_special_tokens=False)
                orig_len = len(input_ids)
                if orig_len < 10:
                    continue

                try:
                    t0 = time.time()
                    result = compressor.compress(input_ids)
                    comp_time = (time.time() - t0) * 1000

                    comp_len = len(result.compressed_ids)
                    cr = orig_len / max(comp_len, 1)
                    savings = (orig_len - comp_len) / max(orig_len, 1) * 100
                    tpr = result.metadata.get("tone_preservation_rate")

                    method_results["compression_ratios"].append(cr)
                    method_results["token_savings"].append(savings)
                    method_results["times_ms"].append(comp_time)
                    if tpr is not None:
                        method_results["tone_preservation_rates"].append(tpr)

                    # Generation (lightweight, short output)
                    try:
                        query_ids = tokenizer.encode(sample.query, add_special_tokens=False)
                        full_input = result.compressed_ids + query_ids
                        if len(full_input) > 1024:
                            full_input = full_input[-1024:]

                        input_t = torch.tensor([full_input]).to(model.device)
                        with torch.no_grad():
                            outputs = model.generate(
                                input_t,
                                max_new_tokens=96,
                                temperature=0.0,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                            )
                        gen_ids = outputs[0][len(full_input):]
                        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                        if gen_text.strip() and sample.reference_answer.strip():
                            rouge = compute_rouge_l([gen_text], [sample.reference_answer])
                            method_results["rouge_l"].append(rouge.get("rougeL_f1", 0))
                            bleu = compute_bleu([gen_text], [sample.reference_answer])
                            method_results["bleu"].append(bleu or 0)
                    except Exception:
                        pass

                    method_results["samples"] += 1

                except Exception as e:
                    print(f"      ERROR [{task_name}/{sample.get('sample_id','?')[:20]}]: {str(e)[:100]}")

        # Compute averages
        for key in ["compression_ratios", "token_savings", "times_ms", "tone_preservation_rates", "rouge_l", "bleu"]:
            v = method_results[key]
            method_results[f"avg_{key.rstrip('s')}"] = sum(v) / len(v) if v else 0

        all_results[f"ratio_{ratio}"][method_name] = method_results

        n = method_results["samples"]
        avg_cr = method_results["avg_compression_ratio"]
        avg_tpr = method_results["avg_tone_preservation_rate"]
        avg_rouge = method_results["avg_rouge_l"]
        avg_bleu = method_results["avg_bleu"]
        avg_time = method_results["avg_time_ms"]
        print(f"      {n} samples | CR: {avg_cr:.1f}x | TPR: {avg_tpr:.3f} | ROUGE-L: {avg_rouge:.4f} | BLEU: {avg_bleu:.4f} | {avg_time:.0f}ms/sample")

total_elapsed = time.time() - total_start

# ═══ 5. Summary Table ═══
print("\n" + "=" * 64)
print("RESULTS SUMMARY")
print(f"Total time: {total_elapsed:.0f}s")
print("=" * 64)

for ratio_key in sorted(all_results.keys()):
    ratio = ratio_key.replace("ratio_", "")
    print(f"\n{'─' * 64}")
    print(f"  Compression Ratio: {ratio}x")
    print(f"  {'Method':<20s} {'CR':>6s} {'TPR':>7s} {'ROUGE-L':>8s} {'BLEU':>7s} {'Time':>7s}")
    print(f"  {'-'*20} {'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")

    for method_name in config.methods:
        r = all_results[ratio_key].get(method_name, {})
        if "error" in r:
            print(f"  {METHOD_NAMES.get(method_name, method_name):<20s} {'ERROR':>6s}")
            continue

        cr = r.get("avg_compression_ratio", 0)
        tpr = r.get("avg_tone_preservation_rate", 0)
        rouge = r.get("avg_rouge_l", 0)
        bleu = r.get("avg_bleu", 0)
        t = r.get("avg_time_ms", 0)
        name = METHOD_NAMES.get(method_name, method_name)
        print(f"  {name:<20s} {cr:>6.2f} {tpr:>7.3f} {rouge:>8.4f} {bleu:>7.4f} {t:>6.0f}ms")

# Save results
os.makedirs("./results_qwen15", exist_ok=True)
output = {
    "model": MODEL_ID,
    "gpu": GPU_NAME,
    "vram_gb": VRAM_GB,
    "total_time_s": total_elapsed,
    "num_samples": total_eval,
    "methods": config.methods,
    "ratios": config.compression_ratios,
    "results": all_results,
}
with open("./results_qwen15/vcc_bench_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

mem_peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"\nPeak GPU memory: {mem_peak:.1f} GB")
print("Results: ./results_qwen15/vcc_bench_results.json")
print("Done!")
