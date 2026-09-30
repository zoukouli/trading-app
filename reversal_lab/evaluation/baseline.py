"""Random-entry baseline (a Monte Carlo permutation-style test).

Question: *could entries at random times, with the same risk management, have
done as well?*

For each of ``reps`` repetitions we build a fake strategy with exactly as many
trades as the real one. Each real trade is paired with a random-entry trade that
has
  * the same direction (long/short),
  * the same stop distance in ATR units,
  * the same target distance in R,
  * the same costs, fill rules and session-end exit,
but a random entry bar drawn from the same period and trading hours. The mean R
of each fake strategy gives a null distribution. The p-value is the share of
random strategies that did at least as well as the real one:

    p = (1 + #{random >= real}) / (1 + reps)

(The +1s count the real result as one draw from the null, so p is never
exactly 0.) A small p (e.g. < 0.05) says the entry *timing* added something
beyond what random timing with the same exits achieves. It does not prove a
future edge, particularly after trying many parameter combinations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import to_np_time
from ..indicators.features import to_arrays


def _pct_or_fixed(spec: dict, price: np.ndarray, tick: float) -> np.ndarray:
    if spec["mode"] == "pct":
        return spec["value"] * price
    if spec["mode"] == "ticks":
        return np.full_like(price, spec["value"] * tick)
    return np.full_like(price, float(spec["value"]))


class RandomBaseline:
    def __init__(self, features: pd.DataFrame, cfg: dict, start=None, end=None):
        self.cfg = cfg
        a, times = to_arrays(features)
        self.o, self.h, self.l, self.c = (a[k].astype(float) for k in ("open", "high", "low", "close"))
        self.atr = a["atr"].astype(float)
        sess = a["session_id"]
        n = len(times)
        pos = np.arange(n)
        # last index of each session in the data (used to cap the holding period)
        last_of = pd.Series(pos).groupby(sess).transform("max").to_numpy()
        self.sess_end = last_of
        ok = a["in_entry_window"].astype(bool) & ~a["is_last_bar"].astype(bool)
        ok &= a["bar_in_session"] >= cfg["session"]["min_bars_into_session"]
        ok &= np.isfinite(self.atr)
        nxt_same = np.zeros(n, dtype=bool)
        nxt_same[:-1] = sess[1:] == sess[:-1]
        ok &= nxt_same
        if start is not None:
            ok &= times >= to_np_time(start)
        if end is not None:
            ok &= times < to_np_time(end)
        self.eligible = np.flatnonzero(ok)
        if not cfg["session"]["flatten_at_session_end"]:
            self.sess_end = np.full(n, n - 1)
        mh = cfg["exits"]["max_hold_bars"]
        self.max_hold = int(mh) if mh else None
        self.tick = cfg["instrument"]["tick_size"]
        self.through = self.tick if cfg["exits"]["target_fill"] == "through" else 0.0
        self.stop_first = cfg["exits"]["intrabar_priority"] == "stop_first"

    def simulate(self, sig_idx: np.ndarray, direction: int, stop_atr: float, target_R: float) -> np.ndarray:
        """Net R of trades signalled at bars ``sig_idx`` (vectorised over the array)."""
        cfg = self.cfg
        f = sig_idx + 1
        entry_raw = self.o[f]
        d = direction
        fill = entry_raw + d * _pct_or_fixed(cfg["costs"]["slippage"], entry_raw, self.tick)
        risk = stop_atr * self.atr[sig_idx]
        stop = fill - d * risk
        target = fill + d * target_R * risk
        last = self.sess_end[f]
        if self.max_hold:
            last = np.minimum(last, f + self.max_hold - 1)
        H = int((last - f).max()) + 1
        idx = f[:, None] + np.arange(H)[None, :]
        valid = idx <= last[:, None]
        idx = np.minimum(idx, len(self.o) - 1)
        o, h, lo = self.o[idx], self.h[idx], self.l[idx]
        st, tg = stop[:, None], target[:, None]
        if d == 1:
            gap_s, gap_t = o <= st, o >= tg + self.through
            s_hit, t_hit = lo <= st, h >= tg + self.through
        else:
            gap_s, gap_t = o >= st, o <= tg - self.through
            s_hit, t_hit = h >= st, lo <= tg - self.through
        if self.stop_first:
            ev_s = gap_s | (s_hit & ~gap_t)
            ev_t = ~gap_s & (gap_t | (t_hit & ~s_hit))
        else:
            ev_s = gap_s | (s_hit & ~gap_t & ~t_hit)
            ev_t = ~gap_s & (gap_t | t_hit)
        ev_s &= valid
        ev_t &= valid
        any_ev = ev_s | ev_t
        has = any_ev.any(axis=1)
        j = np.where(has, any_ev.argmax(axis=1), 0)
        rows = np.arange(len(f))
        is_stop = has & ev_s[rows, j]
        exit_raw = np.where(is_stop, np.where(gap_s[rows, j], o[rows, j], stop), target)
        time_exit = ~has
        exit_raw = np.where(time_exit, self.c[last], exit_raw)
        market = is_stop | time_exit
        slip = np.where(market, _pct_or_fixed(cfg["costs"]["slippage"], exit_raw, self.tick), 0.0)
        exit_fill = exit_raw - d * slip
        comm = _pct_or_fixed(cfg["costs"]["commission"], fill, self.tick) + \
            _pct_or_fixed(cfg["costs"]["commission"], exit_fill, self.tick)
        return ((exit_fill - fill) * d - comm) / risk

    def distribution(self, trades: pd.DataFrame, reps: int = 1000, seed: int = 42) -> np.ndarray:
        """Mean R of ``reps`` random strategies paired trade-by-trade with ``trades``."""
        if len(trades) == 0 or len(self.eligible) == 0:
            return np.array([])
        rng = np.random.default_rng(seed)
        total = np.zeros(reps)
        for _, t in trades.iterrows():
            s = self.eligible[rng.integers(0, len(self.eligible), reps)]
            total += self.simulate(s, int(t["dir"]), float(t["stop_atr"]), float(t["target_R"]))
        return total / len(trades)


def compare_to_baseline(actual_mean: float, dist: np.ndarray) -> dict:
    if len(dist) == 0 or not np.isfinite(actual_mean):
        return {"baseline_mean": np.nan, "baseline_p05": np.nan, "baseline_p95": np.nan,
                "percentile": np.nan, "p_value": np.nan}
    return {
        "baseline_mean": float(dist.mean()),
        "baseline_p05": float(np.quantile(dist, 0.05)),
        "baseline_p95": float(np.quantile(dist, 0.95)),
        "percentile": float((dist < actual_mean).mean() * 100),
        "p_value": float((1 + (dist >= actual_mean).sum()) / (1 + len(dist))),
    }
