import numpy as np
import pandas as pd
import pytest

from helpers import cfg, double_top_day2, ramp
from reversal_lab.backtest import exit_on_bar, run_backtest
from reversal_lab.data import generate_bars, resample
from reversal_lab.indicators import build_features


# ---------------------------------------------------------------- exit_on_bar
@pytest.mark.parametrize("args, expected", [
    # long: stop 95, target 110, through 0
    ((1, 100, 101, 99, 95, 110, 0.0, True), None),
    ((1, 100, 101, 94, 95, 110, 0.0, True), ("stop", 95)),
    ((1, 100, 111, 99, 95, 110, 0.0, True), ("target", 110)),
    ((1, 100, 111, 94, 95, 110, 0.0, True), ("stop", 95)),       # both in one bar: stop first
    ((1, 100, 111, 94, 95, 110, 0.0, False), ("target", 110)),   # ...unless target_first
    ((1, 93, 96, 92, 95, 110, 0.0, True), ("stop", 93)),         # gap through stop: fill at open
    ((1, 112, 113, 94, 95, 110, 0.0, True), ("target", 110)),    # gap through target: fill at target
    ((1, 100, 110, 99, 95, 110, 0.01, True), None),              # touched target but did not trade through
    # short mirror: stop 105, target 90
    ((-1, 100, 106, 99, 105, 90, 0.0, True), ("stop", 105)),
    ((-1, 100, 101, 89, 105, 90, 0.0, True), ("target", 90)),
    ((-1, 107, 108, 100, 105, 90, 0.0, True), ("stop", 107)),
])
def test_exit_on_bar(args, expected):
    assert exit_on_bar(*args) == expected


# ---------------------------------------------------------------- full trade on the hand-built double top
def test_double_top_trade_end_to_end():
    c = cfg(**{"costs.commission.value": 0.0005, "costs.slippage.value": 0.0001})
    df, info = double_top_day2(after_break=ramp(109.65, 96.0, 30))
    res = run_backtest(build_features(df, c), c)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["setup"] == "pdh_pdl" and t["direction"] == "short"
    sig_bar = df.index.get_loc(t["signal_time"])
    assert t["entry_time"] == df.index[sig_bar + 1]
    assert t["entry_price"] == pytest.approx(df["open"].iloc[sig_bar + 1] * (1 - 0.0001))
    # price keeps falling: 2R target hit
    assert t["exit_reason"] == "target"
    assert t["target"] == pytest.approx(t["entry_price"] - 2 * t["risk"])
    comm = 0.0005 * (t["entry_price"] + t["exit_price"])
    assert t["R"] == pytest.approx(((t["entry_price"] - t["exit_price"]) - comm) / t["risk"])
    assert t["R"] == pytest.approx(2 - t["commission_R"])


def test_opposite_level_target_uses_pdl():
    c = cfg(**{"exits.target.mode": "opposite_level", "exits.min_reward_risk": 0.5})
    df, _ = double_top_day2()
    t = run_backtest(build_features(df, c), c).trades
    assert len(t) == 1 and t.iloc[0]["target"] == pytest.approx(90.0)


# ---------------------------------------------------------------- portfolio rules on random data
@pytest.fixture(scope="module")
def rnd():
    c = cfg(**{"costs.commission.value": 0.0005})
    df = resample(generate_bars(days=40, seed=2), "5min")
    return c, df, build_features(df, c)


def test_one_position_at_a_time(rnd):
    c, _, f = rnd
    t = run_backtest(f, c).trades.sort_values("entry_time")
    assert len(t) > 20
    assert (t["entry_time"].iloc[1:].to_numpy() >= t["exit_time"].iloc[:-1].to_numpy()).all()


def test_max_trades_per_day(rnd):
    c, _, f = rnd
    c1 = cfg(**{"risk.max_trades_per_day": 1, "costs.commission.value": 0.0005})
    t = run_backtest(f, c1).trades
    assert t.groupby("session_id").size().max() == 1


def test_positions_flat_at_session_end(rnd):
    c, _, f = rnd
    t = run_backtest(f, c).trades
    assert (t["session_id"] == (t["exit_time"].dt.tz_convert("UTC").dt.floor("D") - pd.Timestamp("1970-01-01", tz="UTC")).dt.days).all()
    assert set(t["exit_reason"]) <= {"stop", "target", "session_end"}


def test_entry_window_and_min_bars_respected(rnd):
    c, _, f = rnd
    t0 = run_backtest(f, c).trades
    assert (f.loc[t0["signal_time"], "bar_in_session"] >= c["session"]["min_bars_into_session"]).all()
    c1 = cfg(**{"session.entry_start": "08:00", "session.entry_end": "12:00", "costs.commission.value": 0.0005})
    with pytest.raises(ValueError):
        run_backtest(f, c1)  # features built for another session config must be rejected
    t = run_backtest(build_features(_, c1), c1).trades
    close_time = t["signal_time"] + pd.Timedelta("5min")
    mins = close_time.dt.hour * 60 + close_time.dt.minute
    assert len(t) and ((mins > 8 * 60) & (mins <= 12 * 60)).all()


def test_costs_reduce_R(rnd):
    _, _, f = rnd
    free = run_backtest(f, cfg()).trades
    c = cfg(**{"costs.commission.value": 0.001, "costs.slippage.value": 0.0005})
    costly = run_backtest(f, c).trades
    assert costly["R"].mean() < free["R"].mean()
    assert np.allclose(free["R"], free["R_frictionless"])


def test_trade_window_restricts_signals(rnd):
    c, df, f = rnd
    start, end = df.index[288 * 10], df.index[288 * 20]
    t = run_backtest(f, c, start, end).trades
    assert len(t) and (t["signal_time"] >= start).all() and (t["signal_time"] < end).all()
    full = run_backtest(f, c).trades
    inside = full[(full["signal_time"] >= start) & (full["signal_time"] < end)]
    # identical except possibly the first trade (the full run may still hold a position at `start`)
    assert len(t) >= len(inside) - 1
