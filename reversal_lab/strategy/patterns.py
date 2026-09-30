"""Pattern primitives: double tops/bottoms, single-swing rejections, and entry triggers.

Everything takes a ``FeatureWindow`` that ends at the current bar ``i``, so it
cannot see the future. Swing points are read from the ``sh_conf``/``sl_conf``
flags, which are set only on the bar that confirms a swing (pivot + ``right``
bars), so a swing is never "known" before it could be known in real time.

Conventions: ``side = "top"`` means swing highs / short trades (direction -1).
``side = "bottom"`` means swing lows / long trades (direction +1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import spec_to_price
from ..indicators.features import FeatureWindow


@dataclass(frozen=True)
class Pattern:
    kind: str            # "double" | "rejection"
    side: str            # "top" | "bottom"
    pivots: tuple        # bar indices of the peak(s)/trough(s)
    extreme: float       # highest peak (top) / lowest trough (bottom): the stop anchor
    neckline: float      # low between two peaks (top) / high between troughs; pivot-bar opposite extreme for rejection
    confirm_idx: int     # bar on which the last pivot was confirmed

    @property
    def direction(self) -> int:
        return -1 if self.side == "top" else 1


def _side_arrays(w: FeatureWindow, side: str):
    if side == "top":
        return w["high"], w["low"], w["sh_conf"]
    return w["low"], w["high"], w["sl_conf"]


def confirmed_pivots(w: FeatureWindow, side: str, right: int, lookback: int) -> np.ndarray:
    """Pivot indices of swings confirmed within the last ``lookback`` bars (oldest first)."""
    _, _, conf = _side_arrays(w, side)
    start = max(0, w.i + 1 - lookback)
    return np.flatnonzero(conf[start:]) + start - right


def invalidated(w: FeatureWindow, pat: Pattern) -> bool:
    """True if price traded beyond the pattern extreme after the last pivot."""
    last = pat.pivots[-1]
    if last + 1 > w.i:
        return False
    if pat.side == "top":
        return bool(w["high"][last + 1:].max() > pat.extreme)
    return bool(w["low"][last + 1:].min() < pat.extreme)


def find_double(w: FeatureWindow, side: str, cfg: dict) -> Pattern | None:
    """Most recent valid double top (side="top") or double bottom (side="bottom") ending at the
    latest confirmed swing, or None.

    Definition (all thresholds from ``cfg['setups']['double']``):
      * p2 = most recent confirmed swing; it must have been confirmed at most
        ``entry.max_bars_to_trigger`` bars ago.
      * p1 = an earlier confirmed swing in the same session, with
        ``min_bars_apart <= p2 - p1 <= max_bars_apart``.
      * |price(p1) - price(p2)| <= peak_tolerance.
      * No bar between p1 and p2 exceeds the pattern extreme.
      * Pullback: |extreme - neckline| >= min_pullback, where the neckline is the
        lowest low (top) / highest high (bottom) strictly between the pivots.
      * The closest qualifying p1 is used.
    """
    d = cfg["setups"]["double"]
    right = cfg["indicators"]["swing"]["right"]
    max_trig = cfg["entry"]["max_bars_to_trigger"]
    lookback = d["max_bars_apart"] + right + max_trig + 1
    pivots = confirmed_pivots(w, side, right, lookback)
    if len(pivots) < 2:
        return None
    p2 = int(pivots[-1])
    confirm = p2 + right
    if w.i - confirm > max_trig:
        return None
    ext_arr, opp_arr, _ = _side_arrays(w, side)
    sess = w["session_id"]
    atr = w.at("atr")
    if not np.isfinite(atr) or sess[p2] != sess[w.i]:
        return None
    tick = cfg["instrument"]["tick_size"]
    price = w.at("close")
    tol = spec_to_price(d["peak_tolerance"], atr, price, tick)
    min_pb = spec_to_price(d["min_pullback"], atr, price, tick)
    sign = 1 if side == "top" else -1
    tried = 0
    for p1 in pivots[-2::-1]:
        p1 = int(p1)
        if tried >= d["max_candidates"]:
            break
        tried += 1
        gap = p2 - p1
        if gap < d["min_bars_apart"]:
            continue
        if gap > d["max_bars_apart"] or sess[p1] != sess[w.i]:
            break
        a, b = ext_arr[p1], ext_arr[p2]
        if abs(a - b) > tol:
            continue
        extreme = max(a, b) if side == "top" else min(a, b)
        between = ext_arr[p1:p2 + 1]
        if sign * (between.max() if side == "top" else between.min()) > sign * extreme:
            continue
        inner = opp_arr[p1 + 1:p2]
        if len(inner) == 0:
            continue
        neckline = inner.min() if side == "top" else inner.max()
        if sign * (extreme - neckline) < min_pb:
            continue
        pat = Pattern("double", side, (p1, p2), float(extreme), float(neckline), confirm)
        if invalidated(w, pat):
            continue
        return pat
    return None


def find_rejection(w: FeatureWindow, side: str, level: np.ndarray, cfg: dict) -> Pattern | None:
    """Single swing that tagged a moving level after approaching it from the far side.

    Long (side="bottom"): the latest confirmed swing low p has
      * |low[p] - level[p]| <= near_tolerance,
      * at least ``approach_min_frac`` of the ``approach_bars`` closes before p above level,
      * (optional) close[p] >= level[p] - near_tolerance, i.e. the bar closed back near/above the level.
    Short mirrors this with swing highs.
    """
    r = cfg["setups"]["reversal"]
    right = cfg["indicators"]["swing"]["right"]
    max_trig = cfg["entry"]["max_bars_to_trigger"]
    pivots = confirmed_pivots(w, side, right, right + max_trig + 1)
    if len(pivots) == 0:
        return None
    p = int(pivots[-1])
    confirm = p + right
    if w.i - confirm > max_trig or p - r["approach_bars"] < 0:
        return None
    atr = w.at("atr")
    lv = level[p]
    if not (np.isfinite(atr) and np.isfinite(lv)):
        return None
    tick = cfg["instrument"]["tick_size"]
    near = spec_to_price(cfg["setups"]["near_tolerance"], atr, w.at("close"), tick)
    ext_arr, opp_arr, _ = _side_arrays(w, side)
    if abs(ext_arr[p] - lv) > near:
        return None
    closes = w["close"][p - r["approach_bars"]:p]
    lvls = level[p - r["approach_bars"]:p]
    frac = np.mean(closes > lvls) if side == "bottom" else np.mean(closes < lvls)
    if not frac >= r["approach_min_frac"]:
        return None
    c = w["close"][p]
    if r["require_close_back"]:
        if side == "bottom" and c < lv - near:
            return None
        if side == "top" and c > lv + near:
            return None
    pat = Pattern("rejection", side, (p,), float(ext_arr[p]), float(opp_arr[p]), confirm)
    return None if invalidated(w, pat) else pat


def trigger_fires(w: FeatureWindow, pat: Pattern, level: np.ndarray | float, cfg: dict) -> bool:
    """True if the entry trigger fires on the current bar for the *first* time since confirmation.

    * ``neckline_break``: close beyond the neckline (below it for tops).
    * ``close_back_through_level``: close back on the reversal side of the level.
    * ``confirmation``: enter on the confirmation bar itself.
    ``level`` may be a scalar (PDH) or a per-bar array (VWAP/MA/session extreme).
    """
    e = cfg["entry"]
    i, c0 = w.i, pat.confirm_idx
    if i < c0 or i - c0 > e["max_bars_to_trigger"]:
        return False
    if e["trigger"] == "confirmation":
        return i == c0
    atr = w.at("atr")
    off = spec_to_price(e["trigger_offset"], atr, w.at("close"), cfg["instrument"]["tick_size"])
    closes = w["close"][c0:i + 1]
    if e["trigger"] == "neckline_break":
        ref = np.full(len(closes), pat.neckline)
    else:
        ref = level[c0:i + 1] if isinstance(level, np.ndarray) else np.full(len(closes), level)
    cond = closes < ref - off if pat.side == "top" else closes > ref + off
    return bool(cond[-1] and not cond[:-1].any())
