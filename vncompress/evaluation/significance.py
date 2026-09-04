"""Paired significance testing for compression A/B comparisons.

Compression papers increasingly report not just a mean quality delta but its
uncertainty -- a delta of +0.02 ROUGE means little without knowing whether it
is distinguishable from noise. This module provides a dependency-light paired
bootstrap (numpy only), matching the philosophy of scripts/compare_slm_runs.py
(which bootstraps paired per-sample NLL for the SLM), so the end-to-end
tone-probe comparison can say "the probe adds X, 95% CI [lo, hi], p=..." rather
than a bare point estimate.

The comparison is *paired*: arm A and arm B are evaluated on the SAME samples,
so we resample sample indices (not the two arms independently), which controls
for sample difficulty and is far more powerful than an unpaired test.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class PairedComparison:
    """Result of comparing arm A (treatment) vs arm B (control) on paired data."""
    n: int
    mean_a: float
    mean_b: float
    mean_delta: float           # mean(a - b)
    ci_low: float               # 95% (by default) bootstrap CI on mean_delta
    ci_high: float
    p_value: float              # two-sided bootstrap p for H0: mean_delta = 0
    win_rate: float             # fraction of paired samples with a >= b
    significant: bool           # CI excludes 0 at the chosen level

    def to_dict(self) -> Dict:
        return asdict(self)


def paired_bootstrap_delta(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Optional[PairedComparison]:
    """Bootstrap the paired mean difference mean(a - b).

    Args:
        a, b: per-sample metric for the treatment (a) and control (b) arms, in
            the SAME sample order. Pairs where either value is None are dropped.
        n_boot: bootstrap resamples.
        ci: central interval mass (0.95 -> 2.5%/97.5% percentiles).
        seed: RNG seed for reproducibility.

    Returns:
        PairedComparison, or None if fewer than 2 usable pairs remain.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    av = np.array([p[0] for p in pairs], dtype=float)
    bv = np.array([p[1] for p in pairs], dtype=float)
    diff = av - bv
    n = len(diff)

    rng = np.random.default_rng(seed)
    # Resample sample indices with replacement; recompute the paired mean delta.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)

    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    ci_low, ci_high = np.quantile(boot_means, [lo_q, hi_q])

    # Two-sided bootstrap p: how often the resampled delta crosses 0, doubled.
    # (Standard percentile-bootstrap p-value; symmetric, clamped to [0, 1].)
    share_le0 = float((boot_means <= 0).mean())
    share_ge0 = float((boot_means >= 0).mean())
    p_value = min(1.0, 2 * min(share_le0, share_ge0))

    return PairedComparison(
        n=n,
        mean_a=float(av.mean()),
        mean_b=float(bv.mean()),
        mean_delta=float(diff.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=p_value,
        win_rate=float((diff >= 0).mean()),
        significant=bool(ci_low > 0 or ci_high < 0),
    )
