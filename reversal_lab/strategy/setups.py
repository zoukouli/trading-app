"""The four setups, each an independent, unit-tested signal function.

Each ``setup_*`` function looks at a ``FeatureWindow`` ending at the current
(closed) bar and returns a list of ``Candidate``s whose entry trigger fires on
*this* bar. They do not handle stops, targets, filters or position state;
``Strategy`` does that, so the setups stay small and testable.

A+       ``pdh_pdl``          double top near previous-session high -> short;
                              double bottom near previous-session low -> long
Second.  ``session_extreme``  same pattern at the current session's running high/low
Tertiary ``vwap`` / ``ma``    reversal off session VWAP / the 200-period MA
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import spec_to_price
from ..indicators.features import FeatureWindow
from .patterns import Pattern, find_double, find_rejection, trigger_fires

SIDES = ("top", "bottom")


@dataclass(frozen=True)
class Candidate:
    setup: str
    pattern: Pattern
    level_name: str
    level: float

    @property
    def direction(self) -> int:
        return self.pattern.direction


def _near(w: FeatureWindow, cfg: dict) -> float:
    return spec_to_price(cfg["setups"]["near_tolerance"], w.at("atr"), w.at("close"),
                         cfg["instrument"]["tick_size"])


def double_patterns(w: FeatureWindow, cfg: dict) -> dict:
    """Latest double top and double bottom (or None) at this bar. Shared across setups."""
    return {side: find_double(w, side, cfg) for side in SIDES}


def setup_pdh_pdl(w: FeatureWindow, cfg: dict, patterns: dict | None = None) -> list[Candidate]:
    """A+ setup: double top whose extreme is within ``near_tolerance`` of PDH (short),
    or double bottom within tolerance of PDL (long)."""
    patterns = patterns if patterns is not None else double_patterns(w, cfg)
    atr = w.at("atr")
    if not np.isfinite(atr):
        return []
    near = _near(w, cfg)
    out = []
    for side, name in (("top", "pdh"), ("bottom", "pdl")):
        pat = patterns.get(side)
        level = w.at(name)
        if pat is None or not np.isfinite(level):
            continue
        if abs(pat.extreme - level) > near:
            continue
        if trigger_fires(w, pat, float(level), cfg):
            out.append(Candidate("pdh_pdl", pat, name.upper(), float(level)))
    return out


def setup_session_extreme(w: FeatureWindow, cfg: dict, patterns: dict | None = None) -> list[Candidate]:
    """Secondary setup: double top at the session's running high (short) or double bottom at
    the running low (long). "At" means the running extreme, which already includes every
    bar up to now, is within ``near_tolerance`` of the pattern extreme."""
    patterns = patterns if patterns is not None else double_patterns(w, cfg)
    atr = w.at("atr")
    if not np.isfinite(atr):
        return []
    near = _near(w, cfg)
    out = []
    for side, name in (("top", "sess_high"), ("bottom", "sess_low")):
        pat = patterns.get(side)
        if pat is None:
            continue
        level = float(w.at(name))
        if abs(level - pat.extreme) > near:
            continue
        if trigger_fires(w, pat, level, cfg):
            out.append(Candidate("session_extreme", pat, "SESSION_HIGH" if side == "top" else "SESSION_LOW", level))
    return out


def _level_reversal(w: FeatureWindow, cfg: dict, setup: str, col: str, label: str,
                    patterns: dict | None) -> list[Candidate]:
    level = w[col]
    if not np.isfinite(w.at("atr")) or not np.isfinite(level[-1]):
        return []
    r = cfg["setups"]["reversal"]
    out = []
    for side in SIDES:
        if r["pattern"] == "rejection":
            pat = find_rejection(w, side, level, cfg)
        else:
            patterns = patterns if patterns is not None else double_patterns(w, cfg)
            pat = patterns.get(side)
            if pat is not None:
                p1, p2 = pat.pivots
                lv = level[p2]
                a0 = p1 - r["approach_bars"]
                if not np.isfinite(lv) or a0 < 0 or abs(pat.extreme - lv) > _near(w, cfg):
                    pat = None
                else:
                    closes, lvls = w["close"][a0:p1], level[a0:p1]
                    frac = np.mean(closes < lvls) if side == "top" else np.mean(closes > lvls)
                    if not frac >= r["approach_min_frac"]:
                        pat = None
        if pat is not None and trigger_fires(w, pat, level, cfg):
            out.append(Candidate(setup, pat, label, float(level[-1])))
    return out


def setup_vwap(w: FeatureWindow, cfg: dict, patterns: dict | None = None) -> list[Candidate]:
    """Tertiary: reversal off session VWAP (support when approached from above -> long;
    resistance when approached from below -> short)."""
    return _level_reversal(w, cfg, "vwap", "vwap", "VWAP", patterns)


def setup_ma(w: FeatureWindow, cfg: dict, patterns: dict | None = None) -> list[Candidate]:
    """Tertiary: reversal off the configured SMA/EMA (default 200-period)."""
    ma = cfg["indicators"]["ma"]
    label = f"{ma['type'].upper()}{ma['period']}@{ma['timeframe']}"
    return _level_reversal(w, cfg, "ma", "ma", label, patterns)


SETUP_FUNCS = {
    "pdh_pdl": setup_pdh_pdl,
    "session_extreme": setup_session_extreme,
    "vwap": setup_vwap,
    "ma": setup_ma,
}
