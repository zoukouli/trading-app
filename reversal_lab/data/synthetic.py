"""Synthetic 24/7 bars with realistic texture but **no edge by construction**.

Log returns are drawn independently of the past price path (fat-tailed
Student-t shocks, slowly varying volatility, an intraday volatility
seasonality). Because tomorrow's return is independent of yesterday's high,
any "edge" a strategy shows on this data is noise or a bug. That makes it
a useful null test for the evaluation code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_bars(days: int = 800, seed: int = 7, start_price: float = 30000.0,
                  annual_vol: float = 0.6, timeframe: str = "1min",
                  start: str = "2023-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    step = pd.Timedelta(timeframe)
    n = int(pd.Timedelta(days=days) / step)
    idx = pd.date_range(start, periods=n, freq=step, tz="UTC")

    per_year = pd.Timedelta(days=365) / step
    base_sigma = annual_vol / np.sqrt(per_year)

    # Slowly varying log-volatility (AR(1), half-life about 2 days).
    phi = 0.5 ** (1 / (2 * 24 * 60 * pd.Timedelta("1min") / step))
    eps = rng.standard_normal(n) * 0.35 * np.sqrt(1 - phi**2)
    logv = np.empty(n)
    logv[0] = 0.0
    for i in range(1, n):  # small loop, but fast enough for ~1e6 points
        logv[i] = phi * logv[i - 1] + eps[i]
    # Intraday seasonality: busier 13:00-17:00 UTC (US/EU overlap).
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    season = 1 + 0.35 * np.exp(-0.5 * ((hour - 15) / 2.5) ** 2)
    sigma = base_sigma * np.exp(logv - 0.5 * 0.35**2) * season

    dof = 4
    shocks = rng.standard_t(dof, n) / np.sqrt(dof / (dof - 2))
    rets = sigma * shocks
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    # Intrabar extremes: extend beyond open/close by a half-normal fraction of sigma.
    wick_hi = np.abs(rng.standard_normal(n)) * sigma * 0.6
    wick_lo = np.abs(rng.standard_normal(n)) * sigma * 0.6
    high = np.maximum(open_, close) * np.exp(wick_hi)
    low = np.minimum(open_, close) * np.exp(-wick_lo)
    volume = np.exp(rng.normal(0, 0.5, n)) * (1 + np.abs(shocks)) * season * 5.0
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                        index=pd.DatetimeIndex(idx, name="time"))
