"""Assemble all indicators into one feature table, and a look-ahead-proof window over it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import core

FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume", "session_id", "bar_in_session", "is_last_bar",
    "in_entry_window", "atr", "vwap", "ma", "pdh", "pdl", "sess_high", "sess_low", "sh_conf", "sl_conf",
]


def feature_key(cfg: dict) -> str:
    """Identifies the config sections that change the feature table."""
    import json

    return json.dumps({k: cfg[k] for k in ("timeframe", "session", "indicators")}, sort_keys=True)


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s, ind, tf = cfg["session"], cfg["indicators"], cfg["timeframe"]
    sess = core.session_frame(df.index, tf, s["timezone"], s["start"], s["entry_start"], s["entry_end"])
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out = out.join(sess)
    out["atr"] = core.atr(df, ind["atr_period"])
    out["vwap"] = core.session_vwap(df, sess["session_id"], ind["vwap"]["price"])
    ma = ind["ma"]
    out["ma"] = core.moving_average(df, ma["type"], ma["period"], ma["timeframe"], tf)
    expected = int(pd.Timedelta(days=1) / pd.Timedelta(tf))
    out = out.join(core.previous_session_levels(df, sess["session_id"], expected, s["min_prev_session_coverage"]))
    out = out.join(core.session_extremes(df, sess["session_id"]))
    out = out.join(core.swing_confirmations(df, ind["swing"]["left"], ind["swing"]["right"]))
    out = out[FEATURE_COLUMNS]
    out.attrs["feature_key"] = feature_key(cfg)
    return out


class FeatureWindow:
    """Read-only view of the feature arrays *up to and including* bar ``i``.

    Strategy code receives one of these per bar. ``w["close"]`` returns a
    numpy view that ends at the current bar, so future bars are physically
    out of reach: indexing past the end raises IndexError instead of
    silently peeking ahead.
    """

    __slots__ = ("_arrays", "_end", "times")

    def __init__(self, arrays: dict[str, np.ndarray], times: np.ndarray, i: int):
        self._arrays = arrays
        self._end = i + 1
        self.times = times[: self._end]

    @property
    def i(self) -> int:
        return self._end - 1

    def __len__(self) -> int:
        return self._end

    def __getitem__(self, name: str) -> np.ndarray:
        return self._arrays[name][: self._end]

    def at(self, name: str, k: int = -1):
        """Value of column ``name`` at bar ``k`` (default: the current bar)."""
        if k >= self._end:
            raise IndexError("look-ahead: bar not yet closed")
        return self._arrays[name][: self._end][k]


def to_arrays(features: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays = {c: features[c].to_numpy() for c in features.columns}
    idx = features.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return arrays, idx.to_numpy(dtype="datetime64[ns]")
