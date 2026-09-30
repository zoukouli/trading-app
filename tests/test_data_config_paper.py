import io
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helpers import cfg
from reversal_lab.backtest import run_backtest
from reversal_lab.config import ConfigError, load_config, with_overrides
from reversal_lab.data import generate_bars, load_csv, resample
from reversal_lab.data.binance import download_binance, parse_kline_zip
from reversal_lab.indicators import build_features, session_frame
from reversal_lab.paper import PaperTrader, ReplayFeed

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- config
def test_config_yaml_matches_defaults():
    assert load_config(ROOT / "config.yaml") == load_config()


def test_unknown_keys_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("setups:\n  double:\n    max_bars_apar: 10\n")
    with pytest.raises(ConfigError):
        load_config(p)
    with pytest.raises(ConfigError):
        with_overrides(load_config(), {"exits.target.r_mult": 2})
    with pytest.raises(ConfigError):
        with_overrides(load_config(), {"entry.trigger": "vibes"})


# ---------------------------------------------------------------- loaders
def _bars_csv(n=30):
    df = resample(generate_bars(days=1, seed=1), "5min").iloc[:n]
    return df


def test_tradingview_unix_export(tmp_path):
    df = _bars_csv()
    tv = pd.DataFrame({"time": (df.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s"), "open": df.open, "high": df.high,
                       "low": df.low, "close": df.close, "Volume": df.volume, "EMA": 1.0})
    tv.to_csv(tmp_path / "BINANCE_BTCUSDT, 5.csv", index=False)
    out, rep = load_csv(tmp_path / "*.csv")
    assert len(out) == len(df) and out.index.tz is not None
    np.testing.assert_allclose(out["close"], df["close"])


def test_iso_with_offset_and_stitching(tmp_path):
    df = _bars_csv(40)
    a = df.iloc[:25].reset_index()
    b = df.iloc[20:].reset_index()  # overlaps the first file
    a["time"] = a["time"].dt.tz_convert("Europe/Athens").astype(str)
    a.to_csv(tmp_path / "a.csv", index=False)
    b.to_csv(tmp_path / "b.csv", index=False)
    out, rep = load_csv([tmp_path / "a.csv", tmp_path / "b.csv"])
    assert len(out) == 40 and rep["duplicates_dropped"] == 5
    assert out.index.equals(df.index)


def test_bad_bars_dropped(tmp_path):
    df = _bars_csv().reset_index()
    df.loc[3, "high"] = df.loc[3, "low"] - 1
    df.to_csv(tmp_path / "x.csv", index=False)
    out, rep = load_csv(tmp_path / "x.csv")
    assert rep["invalid_bars_dropped"] == 1


def test_resample_drops_partial_last_bar():
    one = generate_bars(days=1, seed=1).iloc[:23]  # 23 one-minute bars = 4 full 5-min bars + 3 minutes
    five = resample(one, "5min")
    assert len(five) == 4
    assert five["high"].iloc[0] == one["high"].iloc[:5].max()


def _zip(rows):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("k.csv", "\n".join(",".join(map(str, r)) for r in rows))
    return buf.getvalue()


def test_binance_zip_ms_and_us(tmp_path):
    t_ms = 1704067200000  # 2024-01-01
    row = lambda t: [t, 1, 2, 0.5, 1.5, 10, t + 299999, 0, 0, 0, 0, 0]
    ms = parse_kline_zip(_zip([row(t_ms), row(t_ms + 300000)]))
    us = parse_kline_zip(_zip([row(t_ms * 1000), row((t_ms + 300000) * 1000)]))
    assert ms["time"].iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert us["time"].equals(ms["time"])

    calls = []

    def fake_fetch(url):
        calls.append(url)
        return _zip([row(t_ms + len(calls) * 10**9)])

    p = download_binance("BTCUSDT", "5min", "2024-01", "2024-03", tmp_path / "b.csv", fetch=fake_fetch)
    assert len(calls) == 3 and "BTCUSDT-5m-2024-02.zip" in calls[1]
    out, _ = load_csv(p)
    assert len(out) == 3


def test_session_frame_fx_style_boundary():
    idx = pd.date_range("2024-03-04 21:50", periods=4, freq="5min", tz="UTC")  # 16:50-17:05 New York
    s = session_frame(idx, "5min", tz="America/New_York", start="17:00")
    assert s["is_last_bar"].tolist() == [False, True, False, False]
    assert s["bar_in_session"].tolist()[2:] == [0, 1]
    assert s["session_id"].iloc[1] + 1 == s["session_id"].iloc[2]


# ---------------------------------------------------------------- paper trading
def test_paper_replay_equals_backtest(tmp_path):
    c = cfg(**{"costs.commission.value": 0.0005, "indicators.ma.period": 50})
    df = resample(generate_bars(days=4, seed=9), "5min")
    bt = run_backtest(build_features(df, c), c).trades
    trader = PaperTrader(c, tmp_path, buffer_bars=0)
    trader.run(ReplayFeed(df))
    trader.engine.finish(trader._last_w)
    pt = trader.engine.trades_frame()
    assert len(bt) > 3
    cols = ["setup", "signal_time", "entry_time", "exit_time", "entry_price", "exit_price", "R"]
    pd.testing.assert_frame_equal(bt[cols].reset_index(drop=True), pt[cols].reset_index(drop=True))
    assert (tmp_path / "paper_trades.csv").exists() and (tmp_path / "events.jsonl").exists()


def test_live_flag_refused():
    r = subprocess.run([sys.executable, "-m", "reversal_lab", "paper", "--live"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "NOT implemented" in r.stderr


def test_no_order_code_in_package():
    """Guard rail: nothing in the package talks to an order endpoint."""
    src = "\n".join(p.read_text() for p in (ROOT / "reversal_lab").rglob("*.py"))
    for needle in ("/api/v3/order", "create_order", "place_order", "fapi/v1/order", "X-MBX-APIKEY"):
        assert needle not in src
