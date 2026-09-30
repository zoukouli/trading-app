"""Free long-history BTCUSDT bars from Binance's public data archive.

https://data.binance.vision publishes monthly ZIP files of klines for every
spot symbol, with no API key. This module downloads them, converts them to
the project's standard CSV (``time,open,high,low,close,volume``) and stitches
them into one file.

It is market data only: there is no order code here or anywhere else in the project.

Note: from 2025-01-01 Binance spot archive timestamps are in microseconds
(earlier files use milliseconds). ``loaders._parse_time`` detects the unit.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ARCHIVE = "https://data.binance.vision/data/spot/monthly/klines/{sym}/{tf}/{sym}-{tf}-{ym}.zip"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "trades", "taker_base", "taker_quote", "ignore"]
_TF_MAP = {"1min": "1m", "5min": "5m", "15min": "15m", "1h": "1h"}


def _fetch(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed https host)
        return resp.read()


def parse_kline_zip(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            raw = pd.read_csv(fh, header=None)
    # Some archive files ship with a header row; drop it if present.
    if not str(raw.iloc[0, 0]).isdigit():
        raw = raw.iloc[1:]
    raw.columns = KLINE_COLS[: raw.shape[1]]
    t = raw["open_time"].astype("int64")
    unit = "us" if t.max() > 1e14 else "ms"
    out = pd.DataFrame({
        "time": pd.to_datetime(t, unit=unit, utc=True).astype("datetime64[ns, UTC]"),
        "open": raw["open"].astype(float), "high": raw["high"].astype(float),
        "low": raw["low"].astype(float), "close": raw["close"].astype(float),
        "volume": raw["volume"].astype(float),
    })
    return out


def download_binance(symbol: str = "BTCUSDT", timeframe: str = "5min", start: str = "2022-01",
                     end: str | None = None, out_path: str | Path = "data/raw/BTCUSDT_5m.csv",
                     fetch=_fetch) -> Path:
    """Download monthly archives [start, end] (YYYY-MM) and write one CSV."""
    tf = _TF_MAP.get(timeframe, timeframe)
    months = pd.period_range(start, end or (pd.Timestamp.utcnow() - pd.DateOffset(months=1)).strftime("%Y-%m"),
                             freq="M")
    frames = []
    for m in months:
        url = ARCHIVE.format(sym=symbol, tf=tf, ym=m.strftime("%Y-%m"))
        try:
            frames.append(parse_kline_zip(fetch(url)))
            print(f"  got {m}")
        except urllib.error.HTTPError as err:
            print(f"  skip {m}: HTTP {err.code}")
    if not frames:
        raise RuntimeError("No data downloaded (network blocked or symbol/timeframe wrong?)")
    df = pd.concat(frames).drop_duplicates("time").sort_values("time")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    return out_path
