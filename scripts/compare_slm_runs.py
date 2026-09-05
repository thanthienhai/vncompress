#!/usr/bin/env python3
"""Paired significance test between two `train.py --mode slm --validate` runs.

"LoRA perplexity 110.76 vs base 148.10" is only a claim once you know the
gap survives resampling. Both runs score the *same* held-out texts, so the
right frame is paired: resample texts (not tokens), recompute both corpus
NLLs on the same resampled set, and look at the distribution of the gap.

Refuses to compare runs whose split_fingerprint differs -- every training
run overwrites models/slm/final/val_split.json, so a stale dump silently
compares different validation sets.

Usage:
    python train.py --mode slm --validate --adapter-dir models/slm/final --no-adapter \
        --dump-per-sample results/training/base.json
    python train.py --mode slm --validate --adapter-dir models/slm/final \
        --tone-probe models/slm/tone_probe.pt --dump-per-sample results/training/lora.json
    python scripts/compare_slm_runs.py results/training/base.json results/training/lora.json
"""
import argparse
import json
import math

import numpy as np


def corpus_nll(nll_sums, n_tokens, idx):
    """Token-weighted corpus NLL over the texts selected by `idx`."""
    return nll_sums[idx].sum() / max(n_tokens[idx].sum(), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", help="per-sample dump of the reference run (e.g. --no-adapter)")
    ap.add_argument("candidate", help="per-sample dump of the run being tested (e.g. the LoRA adapter)")
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples (default 10000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05, help="1-alpha confidence interval (default 0.05 -> 95%%)")
    args = ap.parse_args()

    with open(args.baseline, encoding="utf-8") as f:
        a = json.load(f)
    with open(args.candidate, encoding="utf-8") as f:
        b = json.load(f)

    if a["split_fingerprint"] != b["split_fingerprint"]:
        raise SystemExit(
            f"Refusing to compare: the two runs scored different validation splits\n"
            f"  {args.baseline}: fingerprint={a['split_fingerprint']} n={a['num_samples']}\n"
            f"  {args.candidate}: fingerprint={b['split_fingerprint']} n={b['num_samples']}\n"
            f"Re-run both dumps against the same models/slm/final/val_split.json "
            f"(it is overwritten by every training run)."
        )

    a_nll = np.asarray(a["nll_sums"], dtype=np.float64)
    b_nll = np.asarray(b["nll_sums"], dtype=np.float64)
    a_tok = np.asarray(a["n_tokens"], dtype=np.float64)
    b_tok = np.asarray(b["n_tokens"], dtype=np.float64)
    if not np.array_equal(a_tok, b_tok):
        raise SystemExit("Token counts differ per sample despite matching fingerprints -- refusing to compare.")

    n = len(a_nll)
    full = np.arange(n)
    nll_a, nll_b = corpus_nll(a_nll, a_tok, full), corpus_nll(b_nll, b_tok, full)
    ppl_a, ppl_b = math.exp(nll_a), math.exp(nll_b)

    print(f"Validation texts: {n} (split fingerprint {a['split_fingerprint']})")
    print(f"\n{'':<12}{'NLL':>10}{'Perplexity':>14}")
    print(f"{'baseline':<12}{nll_a:>10.4f}{ppl_a:>14.2f}   ({a['adapter_dir']}"
          f"{', no-adapter' if a.get('no_adapter') else ''})")
    print(f"{'candidate':<12}{nll_b:>10.4f}{ppl_b:>14.2f}   ({b['adapter_dir']}"
          f"{', no-adapter' if b.get('no_adapter') else ''})")
    print(f"{'delta':<12}{nll_b - nll_a:>10.4f}{ppl_b - ppl_a:>14.2f}"
          f"   ({(ppl_b / ppl_a - 1) * 100:+.1f}% perplexity)")

    rng = np.random.default_rng(args.seed)
    deltas = np.empty(args.bootstrap)
    ratios = np.empty(args.bootstrap)
    for i in range(args.bootstrap):
        idx = rng.integers(0, n, n)  # paired: same texts for both runs
        d_a, d_b = corpus_nll(a_nll, a_tok, idx), corpus_nll(b_nll, b_tok, idx)
        deltas[i] = d_b - d_a
        ratios[i] = math.exp(d_b) / math.exp(d_a)
    lo, hi = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)
    conf = int(round((1 - args.alpha) * 100))
    print(f"\nPaired bootstrap ({args.bootstrap} resamples over texts, seed={args.seed}):")
    print(f"  delta NLL        {conf}% CI: [{np.percentile(deltas, lo):+.4f}, {np.percentile(deltas, hi):+.4f}]")
    print(f"  perplexity ratio {conf}% CI: [{np.percentile(ratios, lo):.4f}, {np.percentile(ratios, hi):.4f}]"
          f"  (1.0 = no change)")
    share = float((deltas < 0).mean())
    print(f"  resamples where the candidate is better: {share:.2%}")

    try:
        from scipy.stats import wilcoxon
        per_a, per_b = a_nll / np.maximum(a_tok, 1), b_nll / np.maximum(b_tok, 1)
        stat, p = wilcoxon(per_b, per_a)
        print(f"\nWilcoxon signed-rank on per-text mean NLL: statistic={stat:.0f}, p={p:.3g}")
        better = int((per_b < per_a).sum())
        print(f"  candidate has lower NLL on {better}/{n} texts ({better / n:.1%})")
    except ImportError:
        print("\n(scipy not installed -- skipping Wilcoxon signed-rank test)")

    if np.percentile(ratios, hi) < 1.0:
        print(f"\n=> The candidate is better, and the {conf}% CI excludes 'no change'.")
    elif np.percentile(ratios, lo) > 1.0:
        print(f"\n=> The candidate is WORSE, and the {conf}% CI excludes 'no change'.")
    else:
        print(f"\n=> Inconclusive: the {conf}% CI includes 'no change' (ratio 1.0).")


if __name__ == "__main__":
    main()
