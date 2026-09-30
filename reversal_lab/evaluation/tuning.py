"""In-sample / out-of-sample split, grid search and walk-forward analysis.

Clinical-research analogy: the in-sample period is the derivation cohort
(explore and tune freely), the out-of-sample period is the validation cohort
(analysed once, with the rules frozen beforehand), and walk-forward repeats
derive-then-validate many times through time.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ..backtest.engine import run_backtest
from ..config import strategy_hash, with_overrides
from ..indicators.features import build_features, feature_key
from .ledger import Ledger

# ------------------------------------------------------------------ split
def split_dates(index: pd.DatetimeIndex, cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """(start, in_sample_end, end). Out-of-sample = [in_sample_end, end)."""
    ev = cfg["evaluation"]
    start, end = index[0], index[-1] + pd.Timedelta(cfg["timeframe"])
    if ev["in_sample_end"]:
        cut = pd.Timestamp(ev["in_sample_end"], tz="UTC")
    else:
        cut = start + (end - start) * ev["in_sample_fraction"]
        cut = cut.floor("D")  # split on a session boundary (UTC day)
    return start, cut, end


# ------------------------------------------------------------------ grid
def expand_grid(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


_feature_key = feature_key


def objective_value(trades: pd.DataFrame, cfg: dict) -> float:
    ev = cfg["evaluation"]
    if len(trades) < ev["min_trades_for_selection"]:
        return -np.inf
    r = trades["R"].to_numpy()
    if ev["objective"] == "expectancy":
        return float(r.mean())
    if ev["objective"] == "total_R":
        return float(r.sum())
    if ev["objective"] == "profit_factor":
        loss = -r[r <= 0].sum()
        return float(r[r > 0].sum() / loss) if loss > 0 else np.inf
    raise ValueError(ev["objective"])


_FEATURES: dict[str, pd.DataFrame] = {}


def _features_for(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    key = _feature_key(cfg)
    if key not in _FEATURES:
        _FEATURES[key] = build_features(df, cfg)
    return _FEATURES[key]


def _run_combo(args):
    cfg, start, end = args
    feats = _FEATURES[_feature_key(cfg)]
    res = run_backtest(feats, cfg, start, end)
    return res.trades


def evaluate_grid(df: pd.DataFrame, base_cfg: dict, start, end, grid: dict | None = None,
                  ledger: Ledger | None = None, data_fp: str = "", segment: str = "in_sample",
                  workers: int | None = None) -> pd.DataFrame:
    """Backtest every grid combination on [start, end). Returns one row per combination."""
    combos = expand_grid(base_cfg["evaluation"]["grid"] if grid is None else grid)
    cfgs = [with_overrides(base_cfg, c) for c in combos]
    for c in cfgs:
        _features_for(df, c)  # build in the parent so forked workers inherit them
    workers = workers or min(len(cfgs), os.cpu_count() or 1)
    jobs = [(c, start, end) for c in cfgs]
    if workers > 1 and len(cfgs) > 1:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            results = list(ex.map(_run_combo, jobs))
    else:
        results = [_run_combo(j) for j in jobs]
    rows = []
    for combo, c, trades in zip(combos, cfgs, results):
        obj = objective_value(trades, c)
        row = {**combo, "trades": len(trades),
               "expectancy_R": float(trades["R"].mean()) if len(trades) else np.nan,
               "total_R": float(trades["R"].sum()) if len(trades) else 0.0,
               "objective": obj, "strategy_hash": strategy_hash(c)}
        rows.append(row)
        if ledger is not None:
            ledger.record(data_fp=data_fp, segment=segment, strategy_hash=row["strategy_hash"],
                          params=combo, start=start, end=end, trades=len(trades),
                          expectancy_R=row["expectancy_R"])
    return pd.DataFrame(rows).sort_values("objective", ascending=False, kind="stable").reset_index(drop=True)


def best_overrides(table: pd.DataFrame, grid_keys: list[str]) -> dict:
    top = table.iloc[0]
    return {k: (top[k].item() if hasattr(top[k], "item") else top[k]) for k in grid_keys}


# ------------------------------------------------------------------ walk-forward
def walk_forward(df: pd.DataFrame, cfg: dict, span_start, span_end, ledger: Ledger | None = None,
                 data_fp: str = "", workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling re-optimisation. Returns (fold table, stitched out-of-sample trades).

    For each fold: pick the best grid combination on the training window, then
    trade it unchanged on the following test window. The concatenated test-window
    trades estimate how the *whole procedure* (strategy + tuning) performs on
    data it has not seen.
    """
    wf = cfg["evaluation"]["walk_forward"]
    grid = cfg["evaluation"]["grid"]
    keys = list(grid)
    train, test, step = (pd.Timedelta(days=wf[k]) for k in ("train_days", "test_days", "step_days"))
    folds, oos = [], []
    a = pd.Timestamp(span_start).ceil("D")
    while a + train + test <= pd.Timestamp(span_end) + pd.Timedelta(days=1):
        tr_end, te_end = a + train, min(a + train + test, pd.Timestamp(span_end))
        table = evaluate_grid(df, cfg, a, tr_end, grid, ledger, data_fp, "wf_train", workers)
        chosen = best_overrides(table, keys) if np.isfinite(table.iloc[0]["objective"]) else {}
        ccfg = with_overrides(cfg, chosen)
        trades = run_backtest(_features_for(df, ccfg), ccfg, tr_end, te_end).trades
        if ledger is not None:
            ledger.record(data_fp=data_fp, segment="wf_test", strategy_hash=strategy_hash(ccfg),
                          params=chosen, start=tr_end, end=te_end, trades=len(trades),
                          expectancy_R=float(trades["R"].mean()) if len(trades) else None)
        folds.append({"train_start": a.date(), "test_start": tr_end.date(), "test_end": te_end.date(),
                      **{f"chosen:{k}": chosen.get(k) for k in keys},
                      "train_trades": int(table.iloc[0]["trades"]),
                      "train_expectancy_R": table.iloc[0]["expectancy_R"],
                      "test_trades": len(trades),
                      "test_expectancy_R": float(trades["R"].mean()) if len(trades) else np.nan})
        if len(trades):
            trades = trades.assign(fold=len(folds))
            oos.append(trades)
        a += step
    stitched = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame(columns=["setup", "R"])
    return pd.DataFrame(folds), stitched
