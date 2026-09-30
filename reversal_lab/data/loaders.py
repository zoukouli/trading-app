"""Bar-data loaders.

All loaders return a DataFrame indexed by the bar *open* time (tz-aware UTC)
with float columns ``open, high, low, close, volume``.

Supported inputs:
* TradingView "Export chart data" CSV (``time`` column as UNIX seconds or ISO-8601).
* Generic OHLCV CSV (``timestamp`` / ``datetime`` / ``date`` / ``time`` column).
* Binance kline CSVs (see ``binance.py``; they are converted to the generic format).
* Several files at once (e.g. many TradingView exports): they are stitched,
  sorted and de-duplicated.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]
_TIME_CANDIDATES = ("time", "timestamp", "datetime", "date", "open_time", "open time")


def _parse_time(col: pd.Series, tz: str) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(col):
        v = col.astype("int64")
        # Pick the unit from magnitude: s (~1e9), ms (~1e12), us (~1e15).
        mx = v.abs().max()
        unit = "s" if mx < 1e11 else "ms" if mx < 1e14 else "us"
        return pd.DatetimeIndex(pd.to_datetime(v, unit=unit, utc=True)).as_unit("ns")
    parsed = pd.to_datetime(col, utc=False, format="mixed")
    idx = pd.DatetimeIndex(parsed)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    return idx.tz_convert("UTC").as_unit("ns")


def read_ohlcv_csv(path: str | Path, tz: str = "UTC") -> pd.DataFrame:
    raw = pd.read_csv(path)
    cols = {c: c.strip().lower() for c in raw.columns}
    raw = raw.rename(columns=cols)
    tcol = next((c for c in _TIME_CANDIDATES if c in raw.columns), None)
    if tcol is None:
        raise ValueError(f"{path}: no time column (looked for {_TIME_CANDIDATES})")
    missing = [c for c in OHLCV[:4] if c not in raw.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = pd.DataFrame(index=_parse_time(raw[tcol], tz))
    for c in OHLCV[:4]:
        df[c] = pd.to_numeric(raw[c], errors="coerce").to_numpy()
    # TradingView FX exports may have no volume; crypto exports call it "Volume".
    df["volume"] = pd.to_numeric(raw["volume"], errors="coerce").to_numpy() if "volume" in raw else np.nan
    df.index.name = "time"
    return df


def clean_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sort, de-duplicate and drop impossible bars. Returns (clean_df, report)."""
    report = {"rows_in": len(df)}
    df = df.sort_index()
    dup = df.index.duplicated(keep="last")
    report["duplicates_dropped"] = int(dup.sum())
    df = df[~dup]
    bad = df[OHLCV[:4]].isna().any(axis=1)
    bad |= df["high"] < df[["open", "close", "low"]].max(axis=1)
    bad |= df["low"] > df[["open", "close", "high"]].min(axis=1)
    bad |= (df[OHLCV[:4]] <= 0).any(axis=1)
    report["invalid_bars_dropped"] = int(bad.sum())
    df = df[~bad].copy()
    df["volume"] = df["volume"].fillna(0.0)
    report["rows_out"] = len(df)
    return df, report


def gap_report(df: pd.DataFrame, timeframe: str) -> dict:
    step = pd.Timedelta(timeframe)
    diffs = df.index.to_series().diff().dropna()
    gaps = diffs[diffs > step]
    return {
        "n_gaps": int(len(gaps)),
        "missing_bars_est": int(((gaps / step) - 1).sum()) if len(gaps) else 0,
        "largest_gap": str(gaps.max()) if len(gaps) else "0",
    }


def infer_timeframe(df: pd.DataFrame) -> pd.Timedelta:
    return df.index.to_series().diff().dropna().mode().iloc[0]


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate finer bars into ``timeframe`` bars labelled by their open time.

    The final bar is dropped if the source data ends before it would close,
    because a partially-formed bar would never exist in real time.
    """
    tf = pd.Timedelta(timeframe)
    base = infer_timeframe(df)
    if base == tf:
        return df
    if base > tf:
        raise ValueError(f"Data timeframe {base} is coarser than requested {tf}")
    out = df.resample(tf, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])
    last_src_close = df.index[-1] + base
    if len(out) and out.index[-1] + tf > last_src_close:
        out = out.iloc[:-1]
    return out


def load_csv(paths: str | Path | Iterable[str | Path], tz: str = "UTC", timeframe: str | None = None,
             start: str | None = None, end: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Load one or many CSV files (globs allowed), clean and optionally resample."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files: list[str] = []
    for p in paths:
        hits = sorted(glob.glob(str(p)))
        files.extend(hits if hits else [str(p)])
    frames = [read_ohlcv_csv(f, tz) for f in files]
    df, report = clean_bars(pd.concat(frames))
    report["files"] = files
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]
    if timeframe:
        df = resample(df, timeframe)
        report.update(gap_report(df, timeframe))
    return df, report


class BrokerDataAPI:
    """Stub for a broker / vendor historical-data API.

    Implement ``fetch`` to return a DataFrame in the standard format. It is
    deliberately read-only: nothing in this project can place orders.
    """

    def fetch(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError("Plug in your data vendor here (e.g. ccxt, Databento, Polygon).")


def load_data(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Load bars according to ``cfg['data']`` and resample to ``cfg['timeframe']``."""
    d = cfg["data"]
    if d["source"] == "synthetic":
        from .synthetic import generate_bars

        s = d["synthetic"]
        df = generate_bars(days=s["days"], seed=s["seed"], start_price=s["start_price"],
                           annual_vol=s["annual_vol"], timeframe="1min")
        df = resample(df, cfg["timeframe"])
        return df, {"source": "synthetic", **s, "rows_out": len(df)}
    if d["source"] == "csv":
        return load_csv(d["path"], tz=d["timezone"], timeframe=cfg["timeframe"],
                        start=d["start"], end=d["end"])
    raise ValueError(f"Unknown data.source {d['source']}")
