"""Builders for hand-crafted bar sequences used in the setup tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reversal_lab.config import load_config

BARS_PER_DAY = 288


def cfg(**overrides):
    base = {"costs.slippage.value": 0.0, "costs.commission.value": 0.0}
    base.update(overrides)
    return load_config(None, base)


def bars_from_path(closes, start="2024-01-01", wick=0.3, volume=1.0) -> pd.DataFrame:
    """OHLC bars whose open is the previous close and whose wicks extend ``wick`` beyond the body."""
    c = np.asarray(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    df = pd.DataFrame({
        "open": o, "high": np.maximum(o, c) + wick, "low": np.minimum(o, c) - wick, "close": c,
        "volume": np.full(len(c), volume, dtype=float),
    }, index=pd.date_range(start, periods=len(c), freq="5min", tz="UTC", name="time"))
    return df


def ramp(a, b, n):
    """n values moving from a (exclusive) to b (inclusive)."""
    return list(np.linspace(a, b, n + 1)[1:])


def double_top_day2(pdh=110.0, peak1=109.9, peak2=109.95, neck=107.0, day1_level=100.0, after_break=None):
    """Day 1: quiet around ``day1_level`` with a single high at ``pdh`` and low at 90.
    Day 2: ramp up to a first peak, pull back to ``neck``, second peak, then fall through the neckline.

    Returns (df, info) with the bar positions of the peaks in ``info``.
    """
    rng = np.random.default_rng(0)
    day1 = list(day1_level + rng.normal(0, 0.05, BARS_PER_DAY))
    path = day1.copy()
    s2 = len(path)
    path += [105.0] * 20                     # bars 0-19 of day 2
    path += ramp(105.0, peak1 - 0.3, 10)     # up to peak 1 (bar 29)
    p1 = len(path) - 1
    path += ramp(peak1 - 0.3, neck + 0.3, 8)  # pullback
    path += ramp(neck + 0.3, peak2 - 0.3, 8)  # second push
    p2 = len(path) - 1
    path += ramp(peak2 - 0.3, neck - 2.0, 10) if after_break is None else after_break
    path += [path[-1]] * (2 * BARS_PER_DAY - len(path))
    df = bars_from_path(path)
    hi = df.columns.get_loc("high")
    lo = df.columns.get_loc("low")
    df.iloc[100, hi] = pdh
    df.iloc[150, lo] = 90.0
    df.iloc[p1, hi] = peak1
    df.iloc[p2, hi] = peak2
    return df, {"p1": p1, "p2": p2, "session2_start": s2}


def mirror(df: pd.DataFrame, around: float = 200.0) -> pd.DataFrame:
    """Reflect prices (x -> around - x): turns a double top into a double bottom."""
    out = df.copy()
    out["open"], out["close"] = around - df["open"], around - df["close"]
    out["high"], out["low"] = around - df["low"], around - df["high"]
    return out


def scan(fn, feats, cfg_):
    """Run a setup function on every bar; return [(bar_index, candidates)] where it fired."""
    from reversal_lab.indicators.features import FeatureWindow, to_arrays

    arrays, times = to_arrays(feats)
    hits = []
    for i in range(len(times)):
        out = fn(FeatureWindow(arrays, times, i), cfg_)
        if out:
            hits.append((i, out))
    return hits
