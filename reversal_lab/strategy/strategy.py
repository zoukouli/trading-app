"""Combine the setups into one decision per bar: priority, stops, targets and filters."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..config import spec_to_price
from ..indicators.features import FeatureWindow
from .setups import SETUP_FUNCS, Candidate, double_patterns


@dataclass(frozen=True)
class Signal:
    setup: str
    direction: int              # +1 long, -1 short
    signal_time: object
    session_id: int
    level_name: str
    level: float
    extreme: float
    neckline: float
    stop: float
    target_mode: str
    target_price: float | None  # fixed target (vwap / opposite_level); None for r_multiple
    r_multiple: float
    atr: float
    entry_est: float            # close of the signal bar (actual fill is next bar's open)
    pivot_times: tuple
    confirm_time: object
    also_matches: tuple = field(default_factory=tuple)

    @property
    def pattern_key(self) -> tuple:
        return (self.direction, self.pivot_times[-1])


def _round_out(price: float, tick: float, direction: int) -> float:
    """Round a stop price away from the entry, to the tick grid."""
    if tick <= 0:
        return price
    q = price / tick
    return (math.floor(q + 1e-9) if direction == 1 else math.ceil(q - 1e-9)) * tick


class Strategy:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = [s for s in cfg["setups"]["priority"] if s in cfg["setups"]["enabled"]]
        self.rejections: Counter = Counter()
        self._needs_double = any(s in ("pdh_pdl", "session_extreme") for s in self.enabled) or \
            cfg["setups"]["reversal"]["pattern"] == "double"

    def candidates(self, w: FeatureWindow) -> list[Candidate]:
        pats = double_patterns(w, self.cfg) if self._needs_double else {}
        out: list[Candidate] = []
        for name in self.enabled:
            out.extend(SETUP_FUNCS[name](w, self.cfg, pats))
        return out

    def _target(self, cand: Candidate, w: FeatureWindow) -> float | None:
        mode = self.cfg["exits"]["target"]["mode"]
        if mode == "r_multiple":
            return None
        if mode == "vwap":
            return float(w.at("vwap"))
        if cand.setup == "pdh_pdl":
            return float(w.at("pdl" if cand.direction == -1 else "pdh"))
        return float(w.at("sess_low" if cand.direction == -1 else "sess_high"))

    def build_signal(self, cand: Candidate, w: FeatureWindow, also: tuple) -> Signal | None:
        cfg, ex = self.cfg, self.cfg["exits"]
        tick = cfg["instrument"]["tick_size"]
        atr, close = float(w.at("atr")), float(w.at("close"))
        d = cand.direction
        buf = spec_to_price(ex["stop_buffer"], atr, close, tick)
        stop = _round_out(cand.pattern.extreme - d * buf, tick, d)
        risk = (close - stop) * d
        if risk <= 0:
            self.rejections[(cand.setup, "stop_wrong_side")] += 1
            return None
        if risk < spec_to_price(ex["min_stop"], atr, close, tick):
            self.rejections[(cand.setup, "stop_too_tight")] += 1
            return None
        if risk > spec_to_price(ex["max_stop"], atr, close, tick):
            self.rejections[(cand.setup, "stop_too_wide")] += 1
            return None
        target = self._target(cand, w)
        if target is not None:
            if not np.isfinite(target) or (target - close) * d / risk < ex["min_reward_risk"]:
                self.rejections[(cand.setup, "target_too_close")] += 1
                return None
        times = w.times
        return Signal(
            setup=cand.setup, direction=d, signal_time=times[-1], session_id=int(w.at("session_id")),
            level_name=cand.level_name, level=cand.level, extreme=cand.pattern.extreme,
            neckline=cand.pattern.neckline, stop=stop, target_mode=ex["target"]["mode"],
            target_price=target, r_multiple=float(ex["target"]["r_multiple"]), atr=atr, entry_est=close,
            pivot_times=tuple(times[p] for p in cand.pattern.pivots),
            confirm_time=times[cand.pattern.confirm_idx], also_matches=also,
        )

    def evaluate(self, w: FeatureWindow) -> Signal | None:
        """Return the highest-priority signal whose trigger fires on this (closed) bar, or None."""
        cands = self.candidates(w)
        for cand in cands:
            also = tuple(sorted({c.setup for c in cands if c.pattern == cand.pattern and c.setup != cand.setup}))
            sig = self.build_signal(cand, w, also)
            if sig is not None:
                return sig
        return None
