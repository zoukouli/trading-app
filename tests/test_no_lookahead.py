"""No-look-ahead tests: the most important tests in the project.

Look-ahead bias means using information in a backtest that was not yet available at
the time of the decision. It is the trading equivalent of immortal-time bias or
outcome leakage: results look great and cannot be reproduced.

Four independent checks:
1. **Truncation invariance**: every indicator computed on bars[:k] equals the same
   indicator computed on all bars, sliced to [:k]. If any value changed when the
   future was removed, it was using the future.
2. **Future perturbation**: scramble every bar after time T and re-run the whole
   backtest. Every decision made before T must be unchanged.
3. **Structural guards**: the strategy receives a window that ends at the current
   bar and cannot index past it; fills happen on the *next* bar's open.
4. **Test of the test**: a deliberately leaky indicator must fail check 1.
"""

import numpy as np
import pandas as pd
import pytest

from helpers import cfg
from reversal_lab.backtest import run_backtest
from reversal_lab.backtest.engine import Engine
from reversal_lab.data import generate_bars, resample
from reversal_lab.indicators import build_features
from reversal_lab.indicators.features import FeatureWindow, to_arrays
from reversal_lab.strategy import Strategy


@pytest.fixture(scope="module")
def bars():
    return resample(generate_bars(days=12, seed=11), "5min")


CONFIGS = [
    {},
    {"indicators.ma.type": "ema", "indicators.ma.timeframe": "1h", "indicators.ma.period": 20},
    {"indicators.vwap.price": "close", "indicators.swing.left": 5, "indicators.swing.right": 2},
    {"session.start": "17:00", "session.timezone": "America/New_York"},
]


def truncation_mismatches(fn, df, cuts) -> list[tuple[int, str]]:
    full = fn(df)
    bad = []
    for k in cuts:
        part = fn(df.iloc[:k])
        ref = full.iloc[:k]
        for col in ref.columns:
            a, b = part[col].to_numpy(), ref[col].to_numpy()
            if a.dtype.kind == "f" or b.dtype.kind == "f":
                same = np.allclose(a.astype(float), b.astype(float), rtol=0, atol=1e-9, equal_nan=True)
            else:
                same = np.array_equal(a, b)
            if not same:
                bad.append((k, col))
    return bad


@pytest.mark.parametrize("overrides", CONFIGS)
def test_features_truncation_invariant(bars, overrides):
    c = cfg(**overrides)
    rng = np.random.default_rng(0)
    # include cut points mid-session, on session boundaries and on hour boundaries
    cuts = sorted(set(rng.integers(300, len(bars), 12).tolist() + [288 * 3, 288 * 3 + 1, 288 * 5 + 12, 288 * 7 - 1]))
    assert truncation_mismatches(lambda d: build_features(d, c), bars, cuts) == []


def test_truncation_check_catches_a_leak(bars):
    """Test of the test: an indicator that peeks one bar ahead must be caught."""
    c = cfg()

    def leaky(d):
        f = build_features(d, c)
        f["next_close"] = d["close"].shift(-1)  # uses the future
        return f

    bad = truncation_mismatches(leaky, bars, [1000, 2000])
    assert ("next_close" in {col for _, col in bad})


def test_centered_swing_would_be_caught(bars):
    """A 'swing' flagged on the pivot bar itself (not on the confirmation bar) is a leak."""
    def centred(d):
        h = d["high"]
        piv = (h > h.shift(1).rolling(3).max()) & (h >= h[::-1].shift(1).rolling(3).max()[::-1])
        return pd.DataFrame({"sh": piv})

    assert truncation_mismatches(centred, bars, [500, 1500, 2500])


def test_swing_flag_appears_only_after_right_bars(bars):
    c = cfg()
    f = build_features(bars, c)
    right = c["indicators"]["swing"]["right"]
    for i in np.flatnonzero(f["sh_conf"].to_numpy())[:50]:
        j = i - right
        h = bars["high"].to_numpy()
        assert h[j] >= h[j + 1:i + 1].max()  # pivot beats the bars *after* it, which had to close first


@pytest.mark.parametrize("overrides", [{}, {"entry.trigger": "confirmation", "setups.reversal.pattern": "double"}])
def test_future_perturbation_does_not_change_past_decisions(bars, overrides):
    c = cfg(**{"costs.slippage.value": 0.0001, "costs.commission.value": 0.0005, **overrides})
    base = run_backtest(build_features(bars, c), c).trades
    rng = np.random.default_rng(5)
    for cut in (288 * 4 + 37, 288 * 8 + 150):
        t_cut = bars.index[cut]
        alt = bars.copy()
        noise = np.exp(np.cumsum(rng.normal(0, 0.004, len(alt) - cut)))
        for col in ("open", "high", "low", "close"):
            alt.iloc[cut:, alt.columns.get_loc(col)] *= noise
        alt.iloc[cut:, alt.columns.get_loc("volume")] *= rng.uniform(0.1, 10, len(alt) - cut)
        pert = run_backtest(build_features(alt, c), c).trades
        # trades fully closed before the cut are identical
        a = base[base["exit_time"] < t_cut].reset_index(drop=True)
        b = pert[pert["exit_time"] < t_cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        # every signal generated before the cut is identical (entry may differ if it fills after the cut)
        cols = ["signal_time", "setup", "direction", "stop", "level"]
        a = base[base["signal_time"] < t_cut][cols].reset_index(drop=True)
        b = pert[pert["signal_time"] < t_cut][cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        assert len(base) > 0


def test_window_cannot_see_future(bars):
    c = cfg()
    arrays, times = to_arrays(build_features(bars, c))
    w = FeatureWindow(arrays, times, 500)
    assert len(w["close"]) == 501 and w.times[-1] == times[500]
    with pytest.raises(IndexError):
        w.at("close", 501)
    with pytest.raises(IndexError):
        w["close"][501]


class SpyStrategy(Strategy):
    """Records what the strategy was shown and checks it never extends past the current bar."""

    def __init__(self, cfg_, times):
        super().__init__(cfg_)
        self.all_times = times
        self.seen = []

    def evaluate(self, w):
        assert w.times[-1] == self.all_times[w.i]
        assert len(w["close"]) == w.i + 1
        self.seen.append(w.i)
        return super().evaluate(w)


def test_engine_feeds_strategy_only_closed_bars_and_fills_next_open(bars):
    c = cfg(**{"costs.slippage.value": 0.0002})
    feats = build_features(bars, c)
    arrays, times = to_arrays(feats)
    spy = SpyStrategy(c, times)
    eng = Engine(c, strategy=spy)
    for i in range(len(times)):
        eng.step(FeatureWindow(arrays, times, i))
    trades = eng.trades_frame()
    assert len(trades) > 5
    pos = {t: k for k, t in enumerate(feats.index)}
    for _, t in trades.iterrows():
        k = pos[t["signal_time"]]
        assert t["entry_time"] == feats.index[k + 1]  # the very next bar
        o = feats["open"].iloc[k + 1]
        assert t["entry_price"] == pytest.approx(o * (1 + 0.0002 * t["dir"]))
        assert t["exit_time"] >= t["entry_time"]
