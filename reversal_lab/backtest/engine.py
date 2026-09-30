"""Bar-by-bar event loop, shared by the backtester and the paper trader.

Order of events for each bar ``i`` (its open, then its range, then its close):

1. **Open**: a signal generated at the close of bar ``i-1`` fills at ``open[i]``
   plus adverse slippage, provided we are still in the same session.
2. **Intrabar**: the position's stop and target are checked against bar
   ``i``'s high and low (see ``exit_on_bar`` for gaps and same-bar ambiguity).
3. **Close**: time-based exits (session end, max hold) at ``close[i]``.
4. **Decision**: if flat, the strategy sees the window of bars ``0..i``
   (bar ``i`` has now closed) and may emit a signal for bar ``i+1``.

The strategy never sees bar ``i+1`` when deciding, and fills never use the
signal bar's own prices.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import spec_to_price
from ..indicators.features import FeatureWindow, feature_key, to_arrays
from ..strategy.strategy import Signal, Strategy


# ------------------------------------------------------------------ costs
def slippage_amount(cfg: dict, price: float) -> float:
    s = cfg["costs"]["slippage"]
    if s["mode"] == "pct":
        return s["value"] * price
    if s["mode"] == "ticks":
        return s["value"] * cfg["instrument"]["tick_size"]
    return float(s["value"])


def commission_amount(cfg: dict, price: float) -> float:
    """Commission per unit, per side, in price units."""
    c = cfg["costs"]["commission"]
    return c["value"] * price if c["mode"] == "pct" else float(c["value"])


def exit_on_bar(direction: int, o: float, h: float, lo: float, stop: float, target: float,
                through: float, stop_first: bool) -> tuple[str, float] | None:
    """Did a stop or target fill during this bar? Returns (reason, raw fill price) or None.

    * Gap through the stop at the open: filled at the open (worse than the stop).
    * Gap through the target at the open: filled at the target (a resting limit; conservative).
    * Both touched inside one bar: we cannot know which came first from OHLC,
      so ``stop_first`` (the default) assumes the loss.
    * Targets need price to trade ``through`` beyond them (default 1 tick) to count as filled.
    """
    if direction == 1:
        if o <= stop:
            return "stop", o
        if o >= target + through:
            return "target", target
        s_hit, t_hit = lo <= stop, h >= target + through
    else:
        if o >= stop:
            return "stop", o
        if o <= target - through:
            return "target", target
        s_hit, t_hit = h >= stop, lo <= target - through
    if s_hit and t_hit:
        return ("stop", stop) if stop_first else ("target", target)
    if s_hit:
        return "stop", stop
    if t_hit:
        return "target", target
    return None


@dataclass
class Position:
    signal: Signal
    entry_time: object
    entry_raw: float
    entry_fill: float
    stop: float
    target: float
    risk: float
    session_id: int
    bars_held: int = 0


class Engine:
    def __init__(self, cfg: dict, trade_start=None, trade_end=None, strategy: Strategy | None = None):
        self.cfg = cfg
        self.strategy = strategy or Strategy(cfg)
        self.trade_start = to_np_time(trade_start)
        self.trade_end = to_np_time(trade_end)
        self.pending: Signal | None = None
        self.position: Position | None = None
        self.trades: list[dict] = []
        self.skipped: Counter = Counter()
        self.day_counts: Counter = Counter()
        self.used_patterns: set = set()
        tick = cfg["instrument"]["tick_size"]
        self.through = tick if cfg["exits"]["target_fill"] == "through" else 0.0
        self.stop_first = cfg["exits"]["intrabar_priority"] == "stop_first"
        self.on_event = None  # optional callback(kind: str, payload: dict), used by paper trading

    # -------------------------------------------------------------- helpers
    def _emit(self, kind: str, payload: dict) -> None:
        if self.on_event is not None:
            self.on_event(kind, payload)

    def _open(self, sig: Signal, t, o: float) -> None:
        d = sig.direction
        slip = slippage_amount(self.cfg, o)
        fill = o + d * slip
        risk = (fill - sig.stop) * d
        if risk <= 0:
            self.skipped[(sig.setup, "gap_through_stop")] += 1
            return
        if sig.target_price is None:
            target = fill + d * sig.r_multiple * risk
        else:
            target = sig.target_price
            if (target - fill) * d / risk < self.cfg["exits"]["min_reward_risk"]:
                self.skipped[(sig.setup, "target_too_close_at_fill")] += 1
                return
        self.position = Position(sig, t, o, fill, sig.stop, target, risk, sig.session_id)
        self.day_counts[sig.session_id] += 1
        self._emit("fill", {"time": t, "setup": sig.setup, "direction": d, "price": fill,
                            "stop": sig.stop, "target": target})

    def _close(self, t, raw: float, reason: str, market: bool) -> None:
        p, cfg = self.position, self.cfg
        d = p.signal.direction
        slip_exit = slippage_amount(cfg, raw) if market else 0.0
        fill = raw - d * slip_exit
        comm = commission_amount(cfg, p.entry_fill) + commission_amount(cfg, fill)
        gross = (fill - p.entry_fill) * d
        slip_total = (p.entry_fill - p.entry_raw) * d + slip_exit
        s = p.signal
        rec = {
            "setup": s.setup, "direction": "long" if d == 1 else "short", "dir": d,
            "signal_time": s.signal_time, "entry_time": p.entry_time, "exit_time": t,
            "entry_price": p.entry_fill, "stop": p.stop, "target": p.target, "exit_price": fill,
            "exit_reason": reason, "risk": p.risk, "R": (gross - comm) / p.risk,
            "R_frictionless": (gross + slip_total) / p.risk, "commission_R": comm / p.risk,
            "slippage_R": slip_total / p.risk, "bars_held": p.bars_held, "session_id": p.session_id,
            "level_name": s.level_name, "level": s.level, "extreme": s.extreme, "neckline": s.neckline,
            "atr": s.atr, "stop_atr": p.risk / s.atr if s.atr else np.nan,
            "target_R": (p.target - p.entry_fill) * d / p.risk,
            "pivot1_time": s.pivot_times[0], "pivot2_time": s.pivot_times[-1],
            "confirm_time": s.confirm_time, "also_matches": ",".join(s.also_matches),
        }
        self.trades.append(rec)
        self.position = None
        self._emit("exit", rec)

    def _may_signal(self, w: FeatureWindow) -> bool:
        t = w.times[-1]
        if self.trade_start is not None and t < self.trade_start:
            return False
        if self.trade_end is not None and t >= self.trade_end:
            return False
        if w.at("is_last_bar") or not w.at("in_entry_window"):
            return False
        if w.at("bar_in_session") < self.cfg["session"]["min_bars_into_session"]:
            return False
        return self.day_counts[int(w.at("session_id"))] < self.cfg["risk"]["max_trades_per_day"]

    # -------------------------------------------------------------- the loop body
    def step(self, w: FeatureWindow) -> None:
        """Process the newest bar in ``w`` (bar ``w.i``, which has just closed)."""
        t = w.times[-1]
        o, h, lo, c = (float(w.at(k)) for k in ("open", "high", "low", "close"))
        sess = int(w.at("session_id"))

        # 1. fill the pending order at this bar's open
        if self.pending is not None:
            sig, self.pending = self.pending, None
            if sig.session_id != sess:
                self.skipped[(sig.setup, "session_changed_before_fill")] += 1
            else:
                self._open(sig, t, o)

        # 2. manage the open position through this bar
        if self.position is not None and self.position.session_id != sess:
            self._close(t, o, "session_gap", market=True)  # missing bars across session end
        if self.position is not None:
            p = self.position
            hit = exit_on_bar(p.signal.direction, o, h, lo, p.stop, p.target, self.through, self.stop_first)
            if hit is not None:
                reason, raw = hit
                self._close(t, raw, reason, market=(reason == "stop"))

        # 3. time exits at this bar's close
        if self.position is not None:
            self.position.bars_held += 1
            mh = self.cfg["exits"]["max_hold_bars"]
            if self.cfg["session"]["flatten_at_session_end"] and w.at("is_last_bar"):
                self._close(t, c, "session_end", market=True)
            elif mh is not None and self.position.bars_held >= mh:
                self._close(t, c, "max_hold", market=True)

        # 4. decide (flat only; one position at a time)
        if self.position is None and self.pending is None and self._may_signal(w):
            sig = self.strategy.evaluate(w)
            if sig is not None and sig.pattern_key not in self.used_patterns:
                self.used_patterns.add(sig.pattern_key)
                self.pending = sig
                self._emit("signal", {"time": t, "setup": sig.setup, "direction": sig.direction,
                                      "stop": sig.stop, "level": sig.level_name})

    def finish(self, w: FeatureWindow) -> None:
        """Close anything still open at the last bar (end of data)."""
        if self.position is not None:
            self._close(w.times[-1], float(w.at("close")), "data_end", market=True)
        self.pending = None

    def trades_frame(self) -> pd.DataFrame:
        return trades_to_frame(self.trades)


def trades_to_frame(trades: list[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["setup", "direction", "signal_time", "entry_time", "exit_time", "R"])
    df = pd.DataFrame(trades)
    for col in ("signal_time", "entry_time", "exit_time", "pivot1_time", "pivot2_time", "confirm_time"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    skipped: Counter
    rejections: Counter
    n_bars: int


def to_np_time(ts) -> np.datetime64 | None:
    """Timestamp-like -> naive-UTC numpy datetime64 (the representation used in feature arrays)."""
    if ts is None:
        return None
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return np.datetime64(ts.tz_localize(None), "ns")


def run_backtest(features: pd.DataFrame, cfg: dict, start=None, end=None) -> BacktestResult:
    """Run the event loop over ``features``; new signals only when signal time is in [start, end).

    Indicators were computed on all bars up to each point, so bars before ``start``
    act as warm-up. After ``end`` the loop continues only until any open position
    or pending order is resolved.
    """
    key = features.attrs.get("feature_key")
    if key is not None and key != feature_key(cfg):
        raise ValueError("Features were built with different timeframe/session/indicator settings "
                         "than this config; rebuild them with build_features(df, cfg).")
    arrays, times = to_arrays(features)
    eng = Engine(cfg, trade_start=start, trade_end=end)
    first = 0 if start is None else int(np.searchsorted(times, to_np_time(start), "left"))
    end_np = to_np_time(end)
    w = None
    last = first
    for i in range(first, len(times)):
        if end_np is not None and times[i] >= end_np and eng.position is None and eng.pending is None:
            break
        w = FeatureWindow(arrays, times, i)
        eng.step(w)
        last = i
    if w is not None:
        eng.finish(w)
    return BacktestResult(eng.trades_frame(), eng.skipped, eng.strategy.rejections, last - first + 1)
