"""Paper trading: the exact backtest engine, fed one closed bar at a time.

SAFETY: this module only *simulates* fills. There is no broker connection or
order-routing code anywhere in this project. The ``--live`` CLI flag
deliberately refuses to run (see ``cli.py``). Real-money trading would need
new code, a broker adapter and an explicit confirmation step.

Each time a bar closes, the trader:
1. appends it to a rolling buffer (``paper.buffer_bars``);
2. recomputes the indicators on the buffer (the same ``build_features`` as the backtest);
3. calls ``Engine.step`` on the newest bar, the same method the backtest loop calls.

So paper results on replayed history should equal backtest results on the
same bars (``tests/test_paper.py`` checks this).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from ..backtest.engine import Engine
from ..indicators.features import FeatureWindow, build_features, to_arrays

Bar = tuple[pd.Timestamp, dict]


class ReplayFeed:
    """Yield historical bars one at a time (optionally with a delay, to watch it run)."""

    def __init__(self, df: pd.DataFrame, delay_seconds: float = 0.0):
        self.df, self.delay = df, delay_seconds

    def __iter__(self) -> Iterator[Bar]:
        for t, row in self.df.iterrows():
            yield t, row[["open", "high", "low", "close", "volume"]].to_dict()
            if self.delay:
                time.sleep(self.delay)


class BinancePollFeed:
    """Poll Binance's *public market-data* klines endpoint and yield newly closed bars.

    Read-only: no API key, no account access, no orders.
    """

    URL = "https://api.binance.com/api/v3/klines?symbol={sym}&interval={iv}&limit=3"

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "5m", poll_seconds: int = 20):
        self.symbol, self.interval, self.poll = symbol, interval, poll_seconds
        self.last_open = None

    def _fetch(self) -> list:
        import urllib.request

        with urllib.request.urlopen(self.URL.format(sym=self.symbol, iv=self.interval), timeout=15) as r:  # noqa: S310
            return json.loads(r.read())

    def __iter__(self) -> Iterator[Bar]:  # pragma: no cover (needs network)
        while True:
            now_ms = time.time() * 1000
            for k in self._fetch():
                open_ms, close_ms = int(k[0]), int(k[6])
                if close_ms >= now_ms:
                    continue  # bar still forming: never use it
                if self.last_open is not None and open_ms <= self.last_open:
                    continue
                self.last_open = open_ms
                t = pd.Timestamp(open_ms, unit="ms", tz="UTC")
                yield t, {"open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                          "close": float(k[4]), "volume": float(k[5])}
            time.sleep(self.poll)


class PaperTrader:
    def __init__(self, cfg: dict, log_dir: str | Path | None = None, warmup: pd.DataFrame | None = None,
                 buffer_bars: int | None = None, verbose: bool = False):
        self.cfg = cfg
        self.engine = Engine(cfg)
        self.buffer = warmup[["open", "high", "low", "close", "volume"]].copy() if warmup is not None \
            else pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype=float)
        self.buffer_bars = buffer_bars if buffer_bars is not None else cfg["paper"]["buffer_bars"]
        self.verbose = verbose
        self.log_dir = Path(log_dir or cfg["paper"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / "events.jsonl"
        self.trades_path = self.log_dir / "paper_trades.csv"
        self.engine.on_event = self._log
        self._last_w: FeatureWindow | None = None

    def _log(self, kind: str, payload: dict) -> None:
        with open(self.events_path, "a") as fh:
            fh.write(json.dumps({"event": kind, **payload}, default=str) + "\n")
        if kind == "exit":
            new = not self.trades_path.exists()
            with open(self.trades_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(payload))
                if new:
                    w.writeheader()
                w.writerow(payload)
        if self.verbose:
            print(f"[{kind}] " + ", ".join(f"{k}={v}" for k, v in list(payload.items())[:8]))

    def on_bar(self, t: pd.Timestamp, bar: dict) -> None:
        """Handle one *closed* bar."""
        t = pd.Timestamp(t)
        t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
        if len(self.buffer) and t <= self.buffer.index[-1]:
            return  # duplicate or out-of-order bar
        row = pd.DataFrame([bar], index=pd.DatetimeIndex([t], name="time"))
        self.buffer = pd.concat([self.buffer, row]) if len(self.buffer) else row
        if self.buffer_bars and len(self.buffer) > self.buffer_bars:
            self.buffer = self.buffer.iloc[-self.buffer_bars:]
        feats = build_features(self.buffer.astype(float), self.cfg)
        arrays, times = to_arrays(feats)
        w = FeatureWindow(arrays, times, len(times) - 1)
        self.engine.step(w)
        self._last_w = w

    def run(self, feed: Iterable[Bar]) -> pd.DataFrame:
        for t, bar in feed:
            self.on_bar(t, bar)
        return self.engine.trades_frame()
