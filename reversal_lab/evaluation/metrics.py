"""Performance statistics, all in R (multiples of the initial risk per trade).

A trade's R is its net profit divided by the amount risked (entry to stop). Using R
makes trades with different stop sizes comparable and keeps position sizing out of it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def wilson_ci(wins: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Better than the textbook "p ± 1.96·sqrt(p(1-p)/n)" (Wald) interval, which
    misbehaves for small n or p near 0/1. It is the same interval recommended
    for, e.g., a diagnostic test's sensitivity.
    """
    if n == 0:
        return (math.nan, math.nan)
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(x: np.ndarray, reps: int = 5000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean R (resample trades with replacement)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def max_drawdown(r: np.ndarray) -> float:
    """Largest peak-to-trough fall of cumulative R (starting from 0). Returned as a positive number."""
    if len(r) == 0:
        return 0.0
    eq = np.concatenate([[0.0], np.cumsum(r)])
    return float((np.maximum.accumulate(eq) - eq).max())


def longest_losing_streak(r: np.ndarray) -> int:
    best = cur = 0
    for v in r:
        cur = cur + 1 if v <= 0 else 0
        best = max(best, cur)
    return best


def summarize(trades: pd.DataFrame, min_trades: int = 100, bootstrap_reps: int = 5000) -> dict:
    """Headline statistics for a set of trades (in chronological order)."""
    r = trades["R"].to_numpy(dtype=float) if len(trades) else np.array([])
    n = len(r)
    wins = int((r > 0).sum())
    lo, hi = wilson_ci(wins, n)
    gains, losses = r[r > 0].sum(), -r[r <= 0].sum()
    blo, bhi = bootstrap_mean_ci(r, reps=bootstrap_reps)
    out = {
        "trades": n,
        "win_rate": wins / n if n else math.nan,
        "win_rate_ci_low": lo,
        "win_rate_ci_high": hi,
        "avg_win_R": float(r[r > 0].mean()) if wins else math.nan,
        "avg_loss_R": float(r[r <= 0].mean()) if n - wins else math.nan,
        "expectancy_R": float(r.mean()) if n else math.nan,
        "expectancy_ci_low": blo,
        "expectancy_ci_high": bhi,
        "total_R": float(r.sum()),
        "profit_factor": float(gains / losses) if losses > 0 else (math.inf if gains > 0 else math.nan),
        "max_drawdown_R": max_drawdown(r),
        "longest_losing_streak": longest_losing_streak(r),
        "avg_cost_R": float((trades["commission_R"] + trades["slippage_R"]).mean()) if n else math.nan,
        "expectancy_before_costs_R": float(trades["R_frictionless"].mean()) if n else math.nan,
        "sufficient_evidence": n >= min_trades,
    }
    return out


def summarize_by_setup(trades: pd.DataFrame, setups: list[str], min_trades: int = 100,
                       bootstrap_reps: int = 5000) -> pd.DataFrame:
    rows = {}
    for s in setups:
        sub = trades[trades["setup"] == s] if len(trades) else trades
        rows[s] = summarize(sub, min_trades, bootstrap_reps)
    rows["combined"] = summarize(trades, min_trades, bootstrap_reps)
    return pd.DataFrame(rows).T
