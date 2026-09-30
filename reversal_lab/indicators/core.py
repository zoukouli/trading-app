"""Causal indicators.

Rule for every function here: the value at bar *i* may depend only on bars
``0..i`` (bar *i* having closed). The truncation tests in
``tests/test_no_lookahead.py`` enforce this: computing on ``df[:k]`` must
give exactly the same values as computing on the full ``df`` and slicing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- sessions
def _hhmm_to_minutes(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def session_frame(index: pd.DatetimeIndex, timeframe: str, tz: str = "UTC", start: str = "00:00",
                  entry_start: str = "00:00", entry_end: str = "23:30") -> pd.DataFrame:
    """Per-bar session bookkeeping, computed from the clock alone (never from later bars).

    * ``session_id``: integer day number of the trading session the bar belongs to
    * ``bar_in_session``: 0 for the first bar slot after the session start
    * ``is_last_bar``: the bar *closes* at or after the session end (known in advance)
    * ``in_entry_window``: the bar's close falls inside [entry_start, entry_end)
    """
    tf = pd.Timedelta(timeframe)
    offset = pd.Timedelta(minutes=_hhmm_to_minutes(start))
    local = index.tz_convert(tz).tz_localize(None)
    shifted = local - offset
    sdate = shifted.floor("D")
    sess_start = sdate + offset
    sess_end = sess_start + pd.Timedelta(days=1)
    close_local = local + tf
    mins_since_start_close = (close_local - sess_start) / pd.Timedelta(minutes=1)
    es = (_hhmm_to_minutes(entry_start) - _hhmm_to_minutes(start)) % 1440
    ee = (_hhmm_to_minutes(entry_end) - _hhmm_to_minutes(start)) % 1440
    if ee == es:
        in_window = np.ones(len(index), dtype=bool)
    elif es < ee:
        in_window = (mins_since_start_close > es) & (mins_since_start_close <= ee)
    else:
        in_window = (mins_since_start_close > es) | (mins_since_start_close <= ee)
    epoch = pd.Timestamp("1970-01-01")
    return pd.DataFrame({
        "session_id": ((sdate - epoch) // pd.Timedelta(days=1)).astype("int64"),
        "bar_in_session": ((local - sess_start) // tf).astype("int64"),
        "is_last_bar": np.asarray(close_local >= sess_end),
        "in_entry_window": np.asarray(in_window),
    }, index=index)


# ---------------------------------------------------------------- ATR
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------- VWAP
def session_vwap(df: pd.DataFrame, session_id: pd.Series, price: str = "typical") -> pd.Series:
    """VWAP that resets at each session start. Falls back to an equal-weight mean if volume is 0."""
    px = (df["high"] + df["low"] + df["close"]) / 3 if price == "typical" else df["close"]
    vol = df["volume"].clip(lower=0)
    g = session_id.to_numpy()
    cum_pv = (px * vol).groupby(g).cumsum()
    cum_v = vol.groupby(g).cumsum()
    cum_p = px.groupby(g).cumsum()
    cnt = px.groupby(g).cumcount() + 1
    vwap = cum_pv / cum_v.replace(0, np.nan)
    return vwap.fillna(cum_p / cnt)


# ---------------------------------------------------------------- moving averages
def _ma(series: pd.Series, kind: str, period: int) -> pd.Series:
    if kind == "sma":
        return series.rolling(period, min_periods=period).mean()
    if kind == "ema":
        return series.ewm(span=period, adjust=False, min_periods=period).mean()
    raise ValueError(kind)


def moving_average(df: pd.DataFrame, kind: str, period: int, timeframe: str, base_timeframe: str) -> pd.Series:
    """SMA/EMA of closes on ``timeframe`` bars, aligned to the base bars.

    For a higher timeframe the value of a higher-timeframe bar becomes available
    only on the base bar whose close is at or after that bar's close. An hourly
    bar that is still forming is never used.
    """
    base_tf = pd.Timedelta(base_timeframe)
    tf = pd.Timedelta(timeframe)
    if tf == base_tf:
        return _ma(df["close"], kind, period)
    if tf < base_tf:
        raise ValueError("MA timeframe must be >= bar timeframe")
    htf_close = df["close"].resample(tf, label="left", closed="left").last().dropna()
    ma = _ma(htf_close, kind, period)
    avail_at = ma.index + tf  # time the higher-timeframe bar closes
    left = pd.DataFrame({"t": df.index + base_tf})
    right = pd.DataFrame({"t": avail_at, "ma": ma.to_numpy()})
    merged = pd.merge_asof(left, right, on="t", direction="backward")
    return pd.Series(merged["ma"].to_numpy(), index=df.index)


# ---------------------------------------------------------------- levels
def previous_session_levels(df: pd.DataFrame, session_id: pd.Series, expected_bars: int,
                            min_coverage: float = 0.9) -> pd.DataFrame:
    """PDH / PDL: high and low of the previous session present in the data.

    If the previous session has fewer than ``min_coverage * expected_bars``
    bars (data gap, or the first partial day), the levels are NaN.
    """
    g = session_id.to_numpy()
    per = df.groupby(g).agg(h=("high", "max"), l=("low", "min"), n=("close", "size"))
    ok = per["n"] >= min_coverage * expected_bars
    prev = pd.DataFrame({"pdh": per["h"].where(ok).shift(1), "pdl": per["l"].where(ok).shift(1)})
    out = prev.reindex(g)
    out.index = df.index
    return out


def session_extremes(df: pd.DataFrame, session_id: pd.Series) -> pd.DataFrame:
    """Running session high / low up to and including the current bar."""
    g = session_id.to_numpy()
    return pd.DataFrame({"sess_high": df["high"].groupby(g).cummax(),
                         "sess_low": df["low"].groupby(g).cummin()}, index=df.index)


# ---------------------------------------------------------------- swings
def swing_confirmations(df: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    """Fractal swing points, flagged on the bar that *confirms* them.

    Bar *j* is a swing high when ``high[j]`` is strictly greater than the
    ``left`` highs before it and greater than or equal to the ``right`` highs
    after it. That can only be known once bar ``j + right`` has closed, so the
    flag ``sh_conf`` is set on bar ``i = j + right``, never on bar *j*. Swing
    lows mirror this.
    """
    h, lo = df["high"], df["low"]
    piv_h = h.shift(right)
    piv_l = lo.shift(right)
    left_max = h.shift(right + 1).rolling(left, min_periods=left).max()
    left_min = lo.shift(right + 1).rolling(left, min_periods=left).min()
    right_max = h.rolling(right, min_periods=right).max()
    right_min = lo.rolling(right, min_periods=right).min()
    sh = (piv_h > left_max) & (piv_h >= right_max)
    sl = (piv_l < left_min) & (piv_l <= right_min)
    return pd.DataFrame({"sh_conf": sh.fillna(False).astype(bool),
                         "sl_conf": sl.fillna(False).astype(bool)}, index=df.index)
