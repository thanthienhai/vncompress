#!/usr/bin/env python3
"""End-to-end verification of the trained tone probe inside the live compressor.

This is the measurement the paper (Sect. 6, limitation #2) lists as pending:
the Vietnamese SLM tone probe is trained and evaluated in isolation, but its
effect on end-to-end compression *quality* had never been measured, because
nothing wired it into the compressor. `LACCCompressor(tone_source='model')`
(see vncompress/compression.py) closes that wiring; this script produces the number.

It runs a controlled A/B on VCC-Bench:

  * lacc_tone_probe   tone term = trained probe   1 + max softmax(MLP(h_i))
  * lacc_tone_rule    tone term = dictionary heuristic (preservation weight)

Everything else is held fixed -- the SAME fine-tuned SLM (loaded once and shared
between both arms), the same perplexity signal, the same morphology signal, the
same selection, the same generation model, the same decoding. So the difference
between the two arms is attributable to the probe alone. `none` and `random`
are included as references.

The metric set follows what prompt/context-compression papers report
(LLMLingua / LongLLMLingua, LongBench, the NAACL 2025 prompt-compression survey,
and recent empirical studies):

  * task-appropriate quality, reported PER TASK (LongBench convention):
      QA / agent      -> token-level F1
      summarization / multi-turn / cross-lingual -> ROUGE-L F1
      needle-in-haystack -> needle-retrieval recall (RULER-style)
    plus BLEU, Exact Match, and optional multilingual BERTScore for every task;
  * Tone Preservation Rate (the Vietnamese-specific metric, paper P1);
  * compression: realized ratio + token savings;
  * efficiency: compression overhead, generation latency, and speedup vs the
    uncompressed baseline;
  * performance retention: quality kept relative to no compression;
  * statistical rigor: the headline probe - rule delta is reported with a
    paired 95% bootstrap CI, a p-value, and a per-sample win rate, so it is
    distinguishable from noise (cf. scripts/compare_slm_runs.py).

Usage:
  python scripts/verify_tone_probe_e2e.py \
      --generation-model Qwen/Qwen2.5-0.5B-Instruct \
      --scorer-adapter-dir models/slm/final \
      --tone-probe-path   models/slm/tone_probe.pt \
      --ratios 2,4 --max-samples 40 --bertscore --output-dir results/tone_probe_e2e

No training is performed; the SLM adapter + tone probe must already exist
(see `train.py --mode slm`).
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LongBench-style task -> primary quality metric (the number that heads each
# task's row). Every metric is still recorded for every task; this only picks
# which one is "primary" for retention and the paired significance test.
PRIMARY_METRIC = {
    "long_document_qa": "token_f1",
    "agent_tool_calling": "token_f1",
    "multi_turn_conversation": "rouge_l_f1",
    "cross_lingual": "rouge_l_f1",
    "needle_in_haystack": "needle_recall",
}
DEFAULT_PRIMARY = "token_f1"


def _decode_tokens(tokenizer, ids):
    return [tokenizer.decode([t]).replace("▁", " ").replace("Ġ", " ").strip() for t in ids]


def tone_preservation_rate_multiset(tokenizer, orig_ids, comp_ids, tone_analyzer):
    """Cross-arm-consistent TPR: fraction of original tone-bearing tokens (by
    identity, not position) still present after compression. Used uniformly for
    every arm -- including none/random -- so the table compares like with like.
    """
    orig_tokens = _decode_tokens(tokenizer, orig_ids)
    comp_tokens = _decode_tokens(tokenizer, comp_ids)
    orig_infos = tone_analyzer.analyze_tokens(orig_tokens)
    bearing = [t for t, info in zip(orig_tokens, orig_infos) if info.tones_present]
    if not bearing:
        return 1.0
    comp_infos = tone_analyzer.analyze_tokens(comp_tokens)
    remaining = Counter(t for t, info in zip(comp_tokens, comp_infos) if info.tones_present)
    preserved = 0
    for t in bearing:
        if remaining.get(t, 0) > 0:
            preserved += 1
            remaining[t] -= 1
    return preserved / len(bearing)


def timed_generate(model, tokenizer, compressed_ids, query, max_new_tokens):
    """Greedy generation from compressed context + query. Returns (text, ms)."""
    import torch

    t0 = time.time()
    try:
        query_ids = tokenizer.encode(query, add_special_tokens=False)
        full = compressed_ids + query_ids
        inp = torch.tensor([full]).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inp, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][len(full):], skip_special_tokens=True)
    except Exception:
        text = None
    return text, (time.time() - t0) * 1000


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--generation-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--scorer-adapter-dir", default="models/slm/final")
    ap.add_argument("--tone-probe-path", default="models/slm/tone_probe.pt")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--ratios", default="2,4")
    ap.add_argument("--max-samples", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scorer-dtype", choices=["float32", "bfloat16"], default="float32",
                    help="SLM scorer weight dtype; use bfloat16 for a ~4B scorer (Qwen3-4B).")
    ap.add_argument("--scorer-4bit", action="store_true",
                    help="Load the SLM scorer in 4-bit NF4 to fit a large scorer alongside "
                         "the generation model.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="results/tone_probe_e2e")
    ap.add_argument("--bertscore", action="store_true",
                    help="Also compute multilingual BERTScore (downloads a BERT "
                         "model; slower). Off by default so the script is light.")
    ap.add_argument("--bootstrap", type=int, default=10000,
                    help="Bootstrap resamples for the paired probe-vs-rule CI.")
    ap.add_argument("--no-generation", action="store_true",
                    help="Skip generation/quality metrics; report CR + TPR only.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vncompress.compression import create_compressor, LACCCompressor
    from vncompress.config import ExperimentConfig, save_run_metadata, set_seed
    from vncompress.evaluation import (
        VCCBench,
        VCCBenchConfig,
        compute_bert_score,
        compute_bleu,
        compute_exact_match,
        compute_needle_recall,
        compute_rouge_l,
        compute_token_f1,
        paired_bootstrap_delta,
    )
    from vncompress.linguistics import get_tone_analyzer
    from vncompress.models import load_scorer

    ratios = [float(r) for r in args.ratios.split(",")]
    arms = ["none", "random", "lacc_tone_rule", "lacc_tone_probe"]
    os.makedirs(args.output_dir, exist_ok=True)
    save_run_metadata(args.output_dir, ExperimentConfig(
        model=args.generation_model, device=args.device, seed=args.seed,
        compression_ratios=ratios, output_dir=args.output_dir, methods=arms,
    ))

    # --- generation model ------------------------------------------------------
    gen_model = None
    print(f"Loading generation model: {args.generation_model}")
    gen_tokenizer = AutoTokenizer.from_pretrained(args.generation_model, trust_remote_code=True)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    if not args.no_generation:
        gen_model = AutoModelForCausalLM.from_pretrained(
            args.generation_model, trust_remote_code=True, torch_dtype=torch.float16,
        )
        if args.device == "cuda":
            gen_model = gen_model.to("cuda")  # no device_map, see docs/benchmark.md
        gen_model.eval()
    set_seed(args.seed)

    # --- one shared SLM scorer for both tone arms (controlled A/B) -------------
    print(f"Loading SLM tone-probe scorer once: {args.scorer_adapter_dir} + {args.tone_probe_path}")
    shared_scorer = load_scorer(
        args.scorer_adapter_dir, tone_probe_path=args.tone_probe_path, use_adapter=True, device=args.device,
        dtype=torch.bfloat16 if args.scorer_dtype == "bfloat16" else torch.float32,
        load_4bit=args.scorer_4bit,
    )

    def make_arm(method):
        if method in ("none", "random"):
            return create_compressor(method, gen_tokenizer, None, config=None, device=args.device)
        tone_source = "model" if method == "lacc_tone_probe" else "rule"
        return LACCCompressor(
            gen_tokenizer, model=None, config=None, device=args.device,
            scorer=shared_scorer, tone_source=tone_source, name=method,
        )

    compressors = {m: make_arm(m) for m in arms}
    tone_analyzer = get_tone_analyzer()

    # --- samples ---------------------------------------------------------------
    default_data = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "benchmark", "vcc_bench_v1.json",
    )
    bench = VCCBench.load_from_json(args.data_path or default_data,
                                    VCCBenchConfig(output_dir=args.output_dir))
    samples = [s for task in bench.samples.values() for s in task][: args.max_samples]
    print(f"Evaluating {len(samples)} samples x {len(ratios)} ratios x {len(arms)} arms"
          f"{' (+BERTScore)' if args.bertscore else ''}")

    def quality_metrics(out, ref):
        m = {
            "rouge_l_f1": compute_rouge_l([out], [ref])["rougeL_f1"],
            "token_f1": compute_token_f1([out], [ref]),
            "bleu": compute_bleu([out], [ref]),
            "exact_match": float(compute_exact_match([out], [ref])),
            "needle_recall": compute_needle_recall([out], [ref]),
        }
        if args.bertscore:
            m["bertscore_f1"] = compute_bert_score([out], [ref])
        return m

    # records[arm][ratio] -> list of {sample_id, task, ...metrics}
    records = defaultdict(lambda: defaultdict(list))
    t0 = time.time()
    for sid, sample in enumerate(samples):
        input_ids = gen_tokenizer.encode(sample.context, add_special_tokens=False)
        if len(input_ids) < 8:
            continue
        for ratio in ratios:
            for method, comp in compressors.items():
                comp.config.target_ratio = ratio
                ct0 = time.time()
                result = comp.compress(list(input_ids), query=sample.query)
                comp_ms = (time.time() - ct0) * 1000
                rec = {
                    "sample_id": sid,
                    "task": sample.task,
                    "compression_ratio": result.compression_ratio,
                    "token_savings_pct": result.token_savings_pct,
                    "compression_ms": comp_ms,
                    "tpr": tone_preservation_rate_multiset(
                        gen_tokenizer, input_ids, result.compressed_ids, tone_analyzer),
                    "tpr_reported": result.metadata.get("tone_preservation_rate"),
                }
                if not args.no_generation:
                    out, gen_ms = timed_generate(
                        gen_model, gen_tokenizer, result.compressed_ids,
                        sample.query, args.max_new_tokens)
                    rec["generation_ms"] = gen_ms
                    if out is not None:
                        rec.update(quality_metrics(out, sample.reference_answer))
                        rec["primary"] = rec.get(
                            PRIMARY_METRIC.get(sample.task, DEFAULT_PRIMARY))
                records[method][ratio].append(rec)
        if (sid + 1) % 5 == 0:
            print(f"  {sid + 1}/{len(samples)} samples ({time.time() - t0:.0f}s)")

    # --- aggregation helpers ---------------------------------------------------
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    quality_keys = ["rouge_l_f1", "token_f1", "bleu", "exact_match", "needle_recall"]
    if args.bertscore:
        quality_keys.append("bertscore_f1")

    def agg_cell(rows):
        cell = {
            "n": len(rows),
            "realized_cr": mean([r["compression_ratio"] for r in rows]),
            "token_savings_pct": mean([r["token_savings_pct"] for r in rows]),
            "tpr": mean([r["tpr"] for r in rows]),
            "compression_ms": mean([r.get("compression_ms") for r in rows]),
            "generation_ms": mean([r.get("generation_ms") for r in rows]),
            "primary": mean([r.get("primary") for r in rows]),
        }
        for k in quality_keys:
            cell[k] = mean([r.get(k) for r in rows])
        return cell

    # Overall (all tasks) and per-task, per arm per ratio.
    summary = {}
    per_task = {}
    for method in arms:
        summary[method] = {}
        per_task[method] = {}
        for ratio in ratios:
            rows = records[method][ratio]
            if not rows:
                continue
            summary[method][f"ratio_{ratio}"] = agg_cell(rows)
            by_task = defaultdict(list)
            for r in rows:
                by_task[r["task"]].append(r)
            per_task[method][f"ratio_{ratio}"] = {
                task: agg_cell(trows) for task, trows in by_task.items()
            }

    # Performance retention vs no compression (quality kept at each ratio).
    retention = {}
    for method in arms:
        retention[method] = {}
        for ratio in ratios:
            key = f"ratio_{ratio}"
            m = summary.get(method, {}).get(key)
            base = summary.get("none", {}).get(key)
            if m and base and m["primary"] is not None and base["primary"]:
                retention[method][key] = m["primary"] / base["primary"]

    # --- headline: paired probe - rule with bootstrap CI + significance --------
    def paired_series(metric_key, ratio):
        """Align probe vs rule on the SAME sample ids at this ratio."""
        rule = {r["sample_id"]: r.get(metric_key) for r in records["lacc_tone_rule"][ratio]}
        probe = {r["sample_id"]: r.get(metric_key) for r in records["lacc_tone_probe"][ratio]}
        ids = sorted(set(rule) & set(probe))
        return ([probe[i] for i in ids], [rule[i] for i in ids])

    significance = {}
    for ratio in ratios:
        key = f"ratio_{ratio}"
        significance[key] = {}
        metric_list = ["tpr", "primary", "rouge_l_f1", "token_f1"]
        if args.bertscore:
            metric_list.append("bertscore_f1")
        for metric_key in metric_list:
            a, b = paired_series(metric_key, ratio)
            res = paired_bootstrap_delta(a, b, n_boot=args.bootstrap, seed=args.seed)
            if res is not None:
                significance[key][metric_key] = res.to_dict()

    out = {
        "summary": summary,
        "per_task": per_task,
        "retention_vs_none": retention,
        "probe_vs_rule_significance": significance,
        "config": {"ratios": ratios, "n_samples": len(samples),
                   "generation_model": args.generation_model,
                   "bertscore": args.bertscore, "no_generation": args.no_generation,
                   "primary_metric_by_task": PRIMARY_METRIC},
    }
    json_path = os.path.join(args.output_dir, "tone_probe_e2e_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # --- markdown report -------------------------------------------------------
    def fmt(v, pct=False):
        if v is None:
            return "-"
        return f"{v*100:.1f}%" if pct else f"{v:.3f}"

    lines = ["# Tone-Probe End-to-End Verification\n",
             f"Generation model: `{args.generation_model}`  |  samples: {len(samples)}",
             f"SLM: `{args.scorer_adapter_dir}` + `{args.tone_probe_path}`\n",
             "## Overall (all tasks, sample-weighted)\n",
             "| Method | Ratio | Real CR | TPR | ROUGE-L | Token-F1 | BLEU | EM | "
             + ("BERTScore | " if args.bertscore else "") + "Retain | Gen ms |",
             "|--------|-------|---------|-----|---------|----------|------|----|"
             + ("-----------|" if args.bertscore else "") + "--------|--------|"]
    for method in arms:
        for ratio in ratios:
            c = summary.get(method, {}).get(f"ratio_{ratio}")
            if not c:
                continue
            row = (f"| {method} | {ratio}x | {fmt(c['realized_cr'])} | {fmt(c['tpr'])} | "
                   f"{fmt(c['rouge_l_f1'])} | {fmt(c['token_f1'])} | {fmt(c['bleu'])} | "
                   f"{fmt(c['exact_match'])} | ")
            if args.bertscore:
                row += f"{fmt(c.get('bertscore_f1'))} | "
            ret = retention.get(method, {}).get(f"ratio_{ratio}")
            row += f"{fmt(ret, pct=True)} | {fmt(c['generation_ms'])} |"
            lines.append(row)

    lines.append("\n## Probe contribution: lacc_tone_probe - lacc_tone_rule (paired)\n")
    lines.append("Δ with 95% bootstrap CI; ✓ = CI excludes 0 (significant).\n")
    lines.append("| Ratio | Metric | Δ (probe−rule) | 95% CI | p | win-rate | sig |")
    lines.append("|-------|--------|----------------|--------|---|----------|-----|")
    for ratio in ratios:
        for metric_key, res in significance.get(f"ratio_{ratio}", {}).items():
            lines.append(
                f"| {ratio}x | {metric_key} | {res['mean_delta']:+.3f} | "
                f"[{res['ci_low']:+.3f}, {res['ci_high']:+.3f}] | {res['p_value']:.3f} | "
                f"{res['win_rate']*100:.0f}% | {'✓' if res['significant'] else '·'} |")

    lines.append("\n## Per-task primary quality (task-appropriate metric)\n")
    lines.append("| Task | Metric | Ratio | none | rule | probe | Δ probe−rule |")
    lines.append("|------|--------|-------|------|------|-------|--------------|")
    tasks_seen = sorted({r["task"] for ratio in ratios for r in records["lacc_tone_probe"][ratio]}) \
        if not args.no_generation else []
    for task in tasks_seen:
        pm = PRIMARY_METRIC.get(task, DEFAULT_PRIMARY)
        for ratio in ratios:
            def tv(method):
                cell = per_task.get(method, {}).get(f"ratio_{ratio}", {}).get(task)
                return cell["primary"] if cell else None
            n, r, p = tv("none"), tv("lacc_tone_rule"), tv("lacc_tone_probe")
            d = (p - r) if (p is not None and r is not None) else None
            lines.append(f"| {task} | {pm} | {ratio}x | {fmt(n)} | {fmt(r)} | {fmt(p)} | "
                         f"{('%+.3f' % d) if d is not None else '-'} |")

    md_path = os.path.join(args.output_dir, "tone_probe_e2e_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved: {json_path}\n       {md_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
