#!/usr/bin/env python3
"""analyze_gate1.py -- turn a benchmark.py sweep into the Gate-1 evidence.

    python scripts/analyze_gate1.py --runs results/gate1/qwen2.5-7b
    python scripts/analyze_gate1.py --runs results/gate1/gen1 results/gate1/gen2 \
        --out results/report/gate1

Reads the PER-SAMPLE files benchmark.py writes
(`<arm>_<task>_ratio<R>_seed<S>.json`), not the aggregate `vcc_bench_results.json`:
every analysis docs/eval_sweep_gate1.md SS5 asks for needs sample-level rows, and
the aggregate has already thrown them away.

Produces:
  1. Table C1 -- needle results split by `needle_group` (A/B/C), with a PAIRED
     bootstrap of A vs B inside each arm. VCC-Bench v2 places one A, one B and
     one C needle in the same haystack at the same position, so the groups are
     matched samples; pairing controls for haystack difficulty.
  2. The same table at matched REALIZED compression, not configured ratio.
  3. insert_position (lost-in-the-middle) and TPR-vs-token-F1.
  4. Cost: compression latency, generation latency, peak VRAM.
  5. Cross-generator rank inversions, when given more than one run.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from vncompress.evaluation import paired_bootstrap_delta

RESULT_RE = re.compile(r'^(?P<arm>.+)_(?P<task>[a-z_]+)_ratio(?P<ratio>[\d.]+)_seed(?P<seed>\d+)\.json$')
NEEDLE_PAIR_RE = re.compile(r'needle_v2_(\d+)_([ABC])')

# Realized-compression buckets. docs/eval_sweep_gate1.md SS4 rule 1: wave 1 saw
# LLMLingua realize 4.70x when configured for 4x, so comparing arms at their
# configured ratio compares them at different operating points.
CR_BUCKETS = [(1.5, 2.5, '~2x'), (3.5, 5.5, '~4x'), (7.0, 10.0, '~8x')]


def load_runs(run_dirs):
    """Per-sample rows from every result file, tagged with run/arm/task/ratio."""
    rows = []
    for run_dir in run_dirs:
        run = os.path.basename(os.path.normpath(run_dir))
        files = sorted(glob.glob(os.path.join(run_dir, '*.json')))
        if not files:
            print(f"[WARN] no result files in {run_dir}")
        for path in files:
            m = RESULT_RE.match(os.path.basename(path))
            if not m:
                continue  # config.json / environment.json / vcc_bench_results.json
            with open(path, encoding='utf-8') as f:
                records = json.load(f)
            for rec in records:
                rec.setdefault('method', m.group('arm'))
                rec.setdefault('task', m.group('task'))
                rec.setdefault('requested_ratio', float(m.group('ratio')))
                rec['run'] = run
                rows.append(rec)
    return rows


def collapse_random_seeds(arm: str) -> str:
    """random_s1/random_s2/... are one arm measured three times."""
    return 'random' if re.fullmatch(r'random_s\d+', arm) else arm


def _mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _fmt(value, width=8, nd=3):
    return f"{value:>{width}.{nd}f}" if isinstance(value, float) else f"{'--':>{width}}"


def report_generation_health(rows, out_lines):
    """Failed generations score 0 and are indistinguishable from a bad arm."""
    failed = defaultdict(int)
    total = defaultdict(int)
    for r in rows:
        key = (r['run'], collapse_random_seeds(r['method']))
        total[key] += 1
        if r.get('generation_error'):
            failed[key] += 1
    bad = {k: v for k, v in failed.items() if v}
    out_lines.append("\n## 0. Run health\n")
    if not bad:
        out_lines.append(f"All {sum(total.values())} generations succeeded.\n")
        return True
    out_lines.append("**Generation failures found. These score 0 and are NOT evidence "
                     "about the compressor -- fix before reading any table below.**\n")
    out_lines.append("| run | arm | failed / total |")
    out_lines.append("|---|---|---:|")
    for key in sorted(bad):
        out_lines.append(f"| {key[0]} | {key[1]} | {bad[key]} / {total[key]} |")
    out_lines.append("")
    return False


def needle_group_table(rows, out_lines, metric='needle_recall', cr_bucket=None):
    """Table C1: the hypothesis the whole sweep exists to test.

    docs/eval_sweep_gate1.md SS5 -- `lacc_tone` is predicted to collapse on
    group A (no diacritics) but hold on group B (diacritics), while `llmlingua`
    and `random` show no group difference. If that holds, wave 1's 2.3% is an
    artifact of needle design, not a finding about tone.
    """
    label = f" at realized CR {cr_bucket}" if cr_bucket else ""
    out_lines.append(f"\n## Table C1 -- needle `{metric}` by needle_group{label}\n")

    # arm -> group -> {pair_index: value}
    by_arm = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r.get('task') != 'needle_in_haystack' or r.get(metric) is None:
            continue
        m = NEEDLE_PAIR_RE.match(str(r.get('sample_id', '')))
        if not m:
            continue
        arm = collapse_random_seeds(r['method'])
        # Average the seeds of `random` on the same sample before pairing.
        by_arm[arm][m.group(2)].setdefault(m.group(1), []).append(r[metric])

    if not by_arm:
        out_lines.append("_No needle rows with this metric._\n")
        return

    out_lines.append("| arm | A (no diacritics) | B (diacritics) | C (mixed) | B - A | 95% CI | p | n pairs |")
    out_lines.append("|---|---:|---:|---:|---:|---|---:|---:|")
    for arm in sorted(by_arm):
        groups = by_arm[arm]
        means = {g: _mean([np.mean(v) for v in groups.get(g, {}).values()]) for g in 'ABC'}
        shared = sorted(set(groups.get('A', {})) & set(groups.get('B', {})))
        cmp_txt, ci_txt, p_txt = '--', '--', '--'
        if shared:
            a = [float(np.mean(groups['A'][k])) for k in shared]
            b = [float(np.mean(groups['B'][k])) for k in shared]
            pc = paired_bootstrap_delta(b, a)
            if pc:
                star = ' *' if pc.significant else ''
                cmp_txt = f"{pc.mean_delta:+.3f}{star}"
                ci_txt = f"[{pc.ci_low:+.3f}, {pc.ci_high:+.3f}]"
                p_txt = f"{pc.p_value:.3f}"
        out_lines.append(
            f"| `{arm}` |{_fmt(means['A'])} |{_fmt(means['B'])} |{_fmt(means['C'])} "
            f"| {cmp_txt} | {ci_txt} | {p_txt} | {len(shared)} |"
        )
    out_lines.append("\n`B - A` is a PAIRED bootstrap over shared haystacks (`*` = 95% CI excludes 0). "
                     "A large positive delta for a tone-driven arm, and ~0 for `llmlingua`/`random`, "
                     "is the C1 result.\n")


def realized_cr_tables(rows, out_lines):
    out_lines.append("\n## 1. Quality at matched REALIZED compression\n")
    out_lines.append("Configured ratio is not an operating point -- these rows group samples by "
                     "the compression actually achieved.\n")
    out_lines.append("| arm | task | CR bucket | n | realized CR | token-F1 | needle recall | EM |")
    out_lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    buckets = defaultdict(list)
    for r in rows:
        cr = r.get('compression_ratio')
        if cr is None:
            continue
        for lo, hi, name in CR_BUCKETS:
            if lo <= cr < hi:
                buckets[(collapse_random_seeds(r['method']), r.get('task'), name)].append(r)
                break
    for key in sorted(buckets):
        rs = buckets[key]
        out_lines.append(
            f"| `{key[0]}` | {key[1]} | {key[2]} | {len(rs)} "
            f"|{_fmt(_mean([r.get('compression_ratio') for r in rs]), 6, 2)} "
            f"|{_fmt(_mean([r.get('token_f1') for r in rs]))} "
            f"|{_fmt(_mean([r.get('needle_recall') for r in rs]))} "
            f"|{_fmt(_mean([float(r.get('exact_match', 0)) for r in rs]))} |"
        )
    out_lines.append("")

    # Configured-vs-realized drift: name the arms whose knob does not mean what it says.
    drift = defaultdict(list)
    for r in rows:
        cr, req = r.get('compression_ratio'), r.get('requested_ratio')
        if cr and req:
            drift[(collapse_random_seeds(r['method']), req)].append(cr / req)
    bad = {k: _mean(v) for k, v in drift.items() if abs((_mean(v) or 1) - 1) > 0.15}
    if bad:
        out_lines.append("**Arms whose realized compression misses the request by >15%:**\n")
        out_lines.append("| arm | requested | realized / requested |")
        out_lines.append("|---|---:|---:|")
        for (arm, req), v in sorted(bad.items()):
            out_lines.append(f"| `{arm}` | {req:g}x | {v:.2f} |")
        out_lines.append("\nComparing these at their configured ratio compares different budgets.\n")


def position_table(rows, out_lines):
    out_lines.append("\n## 2. Needle recall by insert_position (lost-in-the-middle)\n")
    cells = defaultdict(list)
    positions = []
    for r in rows:
        if r.get('task') != 'needle_in_haystack' or r.get('needle_recall') is None:
            continue
        pos = r.get('insert_position')
        cells[(collapse_random_seeds(r['method']), pos)].append(r['needle_recall'])
        if pos not in positions:
            positions.append(pos)
    if not cells:
        out_lines.append("_No needle rows._\n")
        return
    positions = [p for p in ('beginning', 'middle', 'end') if p in positions]
    out_lines.append("| arm | " + " | ".join(positions) + " |")
    out_lines.append("|---|" + "---:|" * len(positions))
    for arm in sorted({k[0] for k in cells}):
        vals = "".join(_fmt(_mean(cells.get((arm, p), []))) + " |" for p in positions)
        out_lines.append(f"| `{arm}` |{vals}")
    out_lines.append("")


def tpr_correlation(rows, out_lines):
    out_lines.append("\n## 3. Tone preservation vs answer quality\n")
    pairs = [(r['tone_preservation_rate'], r['token_f1']) for r in rows
             if r.get('tone_preservation_rate') is not None and r.get('token_f1') is not None]
    if len(pairs) < 3:
        out_lines.append("_Not enough rows._\n")
        return
    tpr = np.array([p[0] for p in pairs])
    f1 = np.array([p[1] for p in pairs])
    pearson = float(np.corrcoef(tpr, f1)[0, 1]) if tpr.std() and f1.std() else float('nan')
    rank = lambda x: np.argsort(np.argsort(x))  # noqa: E731
    spearman = float(np.corrcoef(rank(tpr), rank(f1))[0, 1])
    out_lines.append(f"n = {len(pairs)} · Pearson r = {pearson:.3f} · Spearman rho = {spearman:.3f}\n")
    out_lines.append("Wave 1 reported this as INVERSE. A negative r here reproduces that; "
                     "a positive one retires it.\n")


def cost_table(rows, out_lines):
    out_lines.append("\n## 4. Cost after the 2048/256 perplexity window\n")
    out_lines.append("The wave-1 cost table predates this window and cannot be cited.\n")
    out_lines.append("| arm | compress ms | generate ms | peak VRAM (GB) | n |")
    out_lines.append("|---|---:|---:|---:|---:|")
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[collapse_random_seeds(r['method'])].append(r)
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        vram = [r['peak_vram_bytes'] for r in rs if r.get('peak_vram_bytes')]
        out_lines.append(
            f"| `{arm}` |{_fmt(_mean([r.get('processing_time_ms') for r in rs]), 10, 1)} "
            f"|{_fmt(_mean([r.get('generation_time_ms') for r in rs]), 10, 1)} "
            f"|{_fmt(max(vram) / 1e9 if vram else None, 10, 2)} | {len(rs)} |"
        )
    out_lines.append("")


def cross_generator_check(rows, out_lines):
    runs = sorted({r['run'] for r in rows})
    if len(runs) < 2:
        return
    out_lines.append("\n## 5. Robustness across generators\n")
    scores = defaultdict(dict)
    for r in rows:
        if r.get('token_f1') is None:
            continue
        scores[r['run']].setdefault(collapse_random_seeds(r['method']), []).append(r['token_f1'])
    ranked = {run: [a for a, _ in sorted(((a, float(np.mean(v))) for a, v in arms.items()),
                                         key=lambda kv: kv[1], reverse=True)]
              for run, arms in scores.items()}
    for run in runs:
        out_lines.append(f"- **{run}**: {' > '.join(f'`{a}`' for a in ranked.get(run, []))}")
    shared = set.intersection(*(set(v) for v in ranked.values())) if ranked else set()
    inversions = []
    for a in sorted(shared):
        for b in sorted(shared):
            if a >= b:
                continue
            orders = {ranked[run].index(a) < ranked[run].index(b) for run in runs if a in ranked[run] and b in ranked[run]}
            if len(orders) > 1:
                inversions.append((a, b))
    out_lines.append("")
    if inversions:
        out_lines.append("**Rank inversions between generators** -- a conclusion that flips is a "
                         "conclusion about the generator, not the compressor:\n")
        for a, b in inversions:
            out_lines.append(f"- `{a}` vs `{b}`")
    else:
        out_lines.append("No rank inversions: the ordering is stable across generators.")
    out_lines.append("")


def write_figure(rows, out_dir):
    """Grouped bar of needle recall by arm x needle_group -- figure C1."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not installed -- no figure written (tables are unaffected)")
        return None
    cells = defaultdict(list)
    for r in rows:
        if r.get('task') == 'needle_in_haystack' and r.get('needle_recall') is not None:
            cells[(collapse_random_seeds(r['method']), r.get('needle_group'))].append(r['needle_recall'])
    arms = sorted({k[0] for k in cells})
    if not arms:
        return None
    groups = ['A', 'B', 'C']
    width, x = 0.26, np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(arms)), 4.2))
    for i, g in enumerate(groups):
        vals = [_mean(cells.get((a, g), [])) or 0.0 for a in arms]
        ax.bar(x + (i - 1) * width, vals, width, label=f"Group {g}")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=30, ha='right')
    ax.set_ylabel('Needle recall')
    ax.set_title('C1 -- needle recall by diacritic group')
    ax.legend()
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ('png', 'pdf'):
        path = os.path.join(out_dir, f'fig_c1.{ext}')
        fig.savefig(path, dpi=200)
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='benchmark.py --output-dir directories (one per generator)')
    ap.add_argument('--out', default='results/report/gate1', help='Directory for gate1_analysis.md + figures')
    ap.add_argument('--metric', default='needle_recall', help='Metric for table C1 (default: needle_recall)')
    args = ap.parse_args()

    rows = load_runs(args.runs)
    if not rows:
        ap.error(f"No per-sample result files found under {args.runs}. "
                 "Point --runs at a benchmark.py --output-dir.")
    print(f"Loaded {len(rows)} per-sample rows from {len(args.runs)} run(s)")

    lines = [
        "# Gate-1 analysis",
        "",
        f"Runs: {', '.join(args.runs)} · {len(rows)} per-sample rows",
        "",
        "Generated by `scripts/analyze_gate1.py`. See docs/eval_sweep_gate1.md.",
    ]
    healthy = report_generation_health(rows, lines)
    needle_group_table(rows, lines, metric=args.metric)
    for lo, hi, name in CR_BUCKETS:
        subset = [r for r in rows if r.get('compression_ratio') and lo <= r['compression_ratio'] < hi]
        if subset:
            needle_group_table(subset, lines, metric=args.metric, cr_bucket=name)
    realized_cr_tables(rows, lines)
    position_table(rows, lines)
    tpr_correlation(rows, lines)
    cost_table(rows, lines)
    cross_generator_check(rows, lines)

    os.makedirs(args.out, exist_ok=True)
    report_path = os.path.join(args.out, 'gate1_analysis.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report: {report_path}")
    for path in (write_figure(rows, args.out) or []):
        print(f"Figure: {path}")
    if not healthy:
        print("\n[WARN] the run had generation failures -- see section 0 before reading any table.")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
