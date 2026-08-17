#!/usr/bin/env python3
"""Turn a run_benchmark.py / run_ablation.py results directory into a
method x compression-ratio x task comparison table, tagged with each
method's baseline / proposed / ablation category.

Resolves P1 issue "eval: tach baseline va proposed method trong
benchmark": reads vcc_bench_results.json (written by VCCBench.evaluate(),
see docs/benchmark.md for the schema), classifies every method via
vncompress.evaluation.method_taxonomy, and prints (or writes) one row per
(method, task, ratio) -- keeping baselines, the proposed method, and
ablation arms visually distinct rather than lumped into one flat list.

Usage:
    python scripts/summarize_results.py results/
    python scripts/summarize_results.py results_ablation/ --context ablation
    python scripts/summarize_results.py results/ --format csv --out summary.csv
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vncompress.evaluation.method_taxonomy import categorize

RESULTS_FILENAME = "vcc_bench_results.json"

# Columns pulled from each 'ratio_<R>' aggregate dict (see
# VCCBench._aggregate_metrics in vncompress/evaluation/metrics.py).
METRIC_COLUMNS = [
    "mean_compression_ratio",
    "mean_token_savings_pct",
    "mean_quality_score",
    "mean_efficiency_score",
    "mean_tone_preservation_rate",  # not present for methods with no tone signal
    "num_samples",
]


def load_results(results_dir: str) -> dict:
    path = os.path.join(results_dir, RESULTS_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Point --results-dir at the --output-dir a "
            f"run_benchmark.py / run_ablation.py run wrote to."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_rows(results: dict, context: str) -> list:
    rows = []
    for method_name, method_results in results.items():
        if method_name == "summary":
            continue
        try:
            category = categorize(method_name, context=context).value
        except ValueError:
            category = "unclassified"

        for task_name, task_results in method_results.items():
            for ratio_key, agg in task_results.items():
                if not ratio_key.startswith("ratio_"):
                    continue
                ratio = ratio_key[len("ratio_"):]
                row = {
                    "category": category,
                    "method": method_name,
                    "task": task_name,
                    "ratio": ratio,
                }
                for col in METRIC_COLUMNS:
                    row[col] = agg.get(col)
                rows.append(row)

    category_order = {"baseline": 0, "proposed": 1, "ablation": 2, "unclassified": 3}
    rows.sort(key=lambda r: (
        category_order.get(r["category"], 9), r["method"], r["task"], float(r["ratio"]),
    ))
    return rows


def format_markdown(rows: list) -> str:
    if not rows:
        return "(no results)\n"
    headers = ["category", "method", "task", "ratio"] + METRIC_COLUMNS
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    last_category = None
    for row in rows:
        if row["category"] != last_category:
            lines.append(f"| **{row['category']}** " + "| " * (len(headers) - 1) + "|")
            last_category = row["category"]
        cells = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append("-" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def format_csv(rows: list) -> str:
    import io
    buf = io.StringIO()
    headers = ["category", "method", "task", "ratio"] + METRIC_COLUMNS
    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", help="Directory containing vcc_bench_results.json")
    parser.add_argument(
        "--context", choices=["registry", "ablation"], default="registry",
        help="'registry' for run_benchmark.py results (default), 'ablation' for run_ablation.py results",
    )
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument("--out", default=None, help="Write to this file instead of stdout")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    rows = build_rows(results, context=args.context)
    output = format_csv(rows) if args.format == "csv" else format_markdown(rows)

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            f.write(output)
        print(f"Wrote {len(rows)} rows to {args.out}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
