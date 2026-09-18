#!/usr/bin/env python3
"""analyze_sweep.py -- turn a VCC-Bench sweep's per-sample files into the table
the paper needs: per-arm means with bootstrap CIs, and a PAIRED bootstrap of
every arm against a reference arm, broken down per task and per ratio.

Why this exists as a separate script: benchmark.py prints arm means only, so a
sweep that costs GPU-days yields a table whose gaps ("0.225 vs 0.223") cannot be
read as a result. The per-sample rows it writes alongside
(`{method}_{task}_ratio{r}_seed{s}.json`, `save_predictions=True` by default)
carry everything needed to add significance afterwards -- no GPU, no re-run.
So COPY THOSE FILES BACK from the pod; the .md/aggregate reports alone cannot be
re-analysed.

Per-task output is not a nicety. On VCC-Bench v2 the tasks do not measure the
same thing: `needle_in_haystack` references are the answer string (median 3
words) so exact-match is meaningful, while `long_document_qa` references are
verbatim passages (median 347 words, up to 1449) that no arm can reproduce under
a 256-token generation cap -- and those are 53% of the benchmark, which the
sample-count-weighted headline number is dominated by. Read arms per task, then
decide what the pooled number is allowed to claim.

    python scripts/analyze_sweep.py results/bench_wave2/sweep
    python scripts/analyze_sweep.py <dir> --metric token_f1 --reference none
    python scripts/analyze_sweep.py <dir> --metric needle_recall --tasks needle_in_haystack
    python scripts/analyze_sweep.py <dir> --tasks cross_lingual \
        --reference lacc_tone --arms lacc_tone_gated        # the E8 read-out
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from vncompress.evaluation import paired_bootstrap_delta  # noqa: E402

# Tasks whose v2 reference is the answer itself rather than a passage lifted out
# of the context. Only these support exact-match/needle-recall readings.
ANSWER_SHAPED_TASKS = ('needle_in_haystack', 'agent_tool_calling', 'cross_lingual')


def load_rows(sweep_dir, tasks=None, ratios=None):
    """-> {(task, ratio): {method: {sample_key: metric_dict}}}"""
    known_tasks = ('long_document_qa', 'multi_turn_conversation', 'needle_in_haystack',
                   'agent_tool_calling', 'cross_lingual')
    out = defaultdict(lambda: defaultdict(dict))
    for name in sorted(os.listdir(sweep_dir)):
        if not name.endswith('.json'):
            continue
        # Split on the longest known task name rather than the regex's greedy
        # method group: arm names contain underscores too.
        task = next((t for t in known_tasks if f'_{t}_ratio' in name), None)
        if task is None:
            continue
        method = name.split(f'_{task}_ratio')[0]
        m = re.search(r'_ratio([\d.]+)_seed(\d+)\.json$', name)
        if not m:
            continue
        ratio = float(m.group(1))
        if tasks and task not in tasks:
            continue
        if ratios and ratio not in ratios:
            continue
        with open(os.path.join(sweep_dir, name), 'r', encoding='utf-8') as f:
            rows = json.load(f)
        for i, row in enumerate(rows):
            meta = row.get('metadata') or {}
            # sample_id pairs arms robustly; older sweeps (pre 2026-09-18) have
            # none, so fall back to position and say so.
            key = meta.get('sample_id') or f'#idx{i}'
            out[(task, ratio)][method][key] = row
    return out


def metric_of(row, metric):
    if metric in row:
        value = row.get(metric)
    else:
        value = (row.get('metadata') or {}).get(metric)
    if isinstance(value, bool):
        return float(value)
    return float(value) if isinstance(value, (int, float)) else None


def bootstrap_mean_ci(values, n_boot=10000, ci=0.95, seed=42):
    vals = np.array([v for v in values if v is not None], dtype=float)
    if len(vals) < 2:
        return (float(vals.mean()) if len(vals) else 0.0), None, None
    rng = np.random.default_rng(seed)
    boot = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(vals.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description='Per-task, per-ratio significance tables for a VCC-Bench sweep.')
    ap.add_argument('sweep_dir', help='Directory holding {method}_{task}_ratio{r}_seed{s}.json files.')
    ap.add_argument('--metric', default='token_f1',
                    help="Per-sample field to analyse (default: token_f1 -- the standard extractive-QA "
                         "metric, and the one the probe A/Bs already report). Others: quality_score "
                         "(benchmark.py's headline: 0.4*rougeL + 0.2*bleu + 0.4*exact_match, so 40%% of "
                         "it is dead weight wherever references are passages), rouge_l_f1, exact_match, "
                         "needle_recall, evidence_recall, unsupported_claim_rate, compression_ratio.")
    ap.add_argument('--reference', default='none',
                    help="Arm every other arm is paired against (default: none = uncompressed).")
    ap.add_argument('--arms', default=None, help='Comma-separated subset of arms to report.')
    ap.add_argument('--tasks', default=None, help='Comma-separated subset of tasks.')
    ap.add_argument('--ratios', default=None, help='Comma-separated subset of ratios, e.g. 2,4,8.')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None, help='Write the markdown here as well as stdout.')
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(',')] if args.tasks else None
    ratios = [float(r) for r in args.ratios.split(',')] if args.ratios else None
    arms = [a.strip() for a in args.arms.split(',')] if args.arms else None

    data = load_rows(args.sweep_dir, tasks, ratios)
    if not data:
        raise SystemExit(f"No per-sample files found in {args.sweep_dir}. The sweep must run with "
                         "save_predictions=True (the default) and those files must be copied back.")

    lines = [f"# Sweep analysis: `{args.sweep_dir}`", "",
             f"Metric: **{args.metric}** | reference arm: **{args.reference}** | "
             f"paired bootstrap, {args.n_boot:,} resamples, 95% CI.", ""]

    positional = any(k.startswith('#idx') for per_method in data.values()
                     for rows in per_method.values() for k in rows)
    if positional:
        lines += ["> **Warning:** some rows carry no `sample_id`, so arms are paired BY POSITION. "
                  "That is only valid if every arm scored the same samples in the same order "
                  "(sweeps before 2026-09-18 did not record the id).", ""]

    pooled = defaultdict(list)          # method -> values, across everything reported
    pooled_paired = defaultdict(list)   # method -> (value, reference value) pairs

    for (task, ratio) in sorted(data):
        per_method = data[(task, ratio)]
        names = [m for m in sorted(per_method) if not arms or m in arms or m == args.reference]
        if not names:
            continue
        note = "" if task in ANSWER_SHAPED_TASKS else \
            "  \n*Reference answers here are verbatim passages, not answers -- read as passage recovery.*"
        lines += [f"## {task} @ {ratio:g}x{note}", "",
                  f"| Arm | n | mean {args.metric} | 95% CI | Δ vs {args.reference} | Δ 95% CI | p | sig |",
                  "|---|---:|---:|---|---:|---|---:|:--:|"]
        ref_rows = per_method.get(args.reference, {})
        for method in names:
            rows = per_method[method]
            keys = sorted(rows)
            values = [metric_of(rows[k], args.metric) for k in keys]
            mean, lo, hi = bootstrap_mean_ci(values, args.n_boot, seed=args.seed)
            ci_txt = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "--"
            pooled[method] += values

            delta_txt = ci_delta = p_txt = sig = "--"
            if ref_rows and method != args.reference:
                shared = [k for k in keys if k in ref_rows]
                a = [metric_of(rows[k], args.metric) for k in shared]
                b = [metric_of(ref_rows[k], args.metric) for k in shared]
                pooled_paired[method] += list(zip(a, b))
                cmp = paired_bootstrap_delta(a, b, n_boot=args.n_boot, seed=args.seed)
                if cmp:
                    delta_txt = f"{cmp.mean_delta:+.3f}"
                    ci_delta = f"[{cmp.ci_low:+.3f}, {cmp.ci_high:+.3f}]"
                    p_txt = f"{cmp.p_value:.3f}"
                    sig = "**yes**" if cmp.significant else "no"
            lines.append(f"| {method} | {len([v for v in values if v is not None])} | {mean:.3f} | "
                         f"{ci_txt} | {delta_txt} | {ci_delta} | {p_txt} | {sig} |")
        lines.append("")

    lines += ["## Pooled over everything reported above", "",
              "Pooled across tasks that do not measure the same thing -- keep the per-task rows "
              "as the real result.", "",
              f"| Arm | n | mean {args.metric} | 95% CI | Δ vs {args.reference} | Δ 95% CI | sig |",
              "|---|---:|---:|---|---:|---|:--:|"]
    for method in sorted(pooled, key=lambda m: -np.mean([v for v in pooled[m] if v is not None] or [0])):
        values = pooled[method]
        mean, lo, hi = bootstrap_mean_ci(values, args.n_boot, seed=args.seed)
        ci_txt = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "--"
        delta_txt = ci_delta = sig = "--"
        if pooled_paired.get(method):
            a = [x for x, _ in pooled_paired[method]]
            b = [y for _, y in pooled_paired[method]]
            cmp = paired_bootstrap_delta(a, b, n_boot=args.n_boot, seed=args.seed)
            if cmp:
                delta_txt = f"{cmp.mean_delta:+.3f}"
                ci_delta = f"[{cmp.ci_low:+.3f}, {cmp.ci_high:+.3f}]"
                sig = "**yes**" if cmp.significant else "no"
        lines.append(f"| {method} | {len([v for v in values if v is not None])} | {mean:.3f} | "
                     f"{ci_txt} | {delta_txt} | {ci_delta} | {sig} |")

    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(report + "\n")
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == '__main__':
    main()
