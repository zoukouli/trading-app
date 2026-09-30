import numpy as np
import pandas as pd
import pytest

from helpers import cfg
from reversal_lab.backtest import exit_on_bar, run_backtest
from reversal_lab.backtest.engine import commission_amount, slippage_amount
from reversal_lab.data import generate_bars, resample
from reversal_lab.evaluation import (Ledger, RandomBaseline, compare_to_baseline, expand_grid, longest_losing_streak,
                                     max_drawdown, split_dates, summarize, walk_forward, wilson_ci)
from reversal_lab.evaluation.tuning import evaluate_grid
from reversal_lab.indicators import build_features


# ---------------------------------------------------------------- metrics
def test_wilson_known_values():
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4) and hi == pytest.approx(0.5962, abs=1e-4)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0 and hi == pytest.approx(0.2775, abs=1e-4)


def test_drawdown_and_streak():
    r = np.array([1, -1, -1, -1, 2, -1, 3])
    assert max_drawdown(r) == 3
    assert longest_losing_streak(r) == 3
    assert max_drawdown(np.array([-1.0, -1.0])) == 2  # drawdown counted from the starting equity


def test_summarize():
    t = pd.DataFrame({"R": [2.0, -1.0, -1.0, 2.0], "commission_R": 0.1, "slippage_R": 0.05,
                      "R_frictionless": [2.15, -0.85, -0.85, 2.15]})
    s = summarize(t, min_trades=100)
    assert s["trades"] == 4 and s["win_rate"] == 0.5
    assert s["expectancy_R"] == pytest.approx(0.5)
    assert s["profit_factor"] == pytest.approx(2.0)
    assert s["avg_cost_R"] == pytest.approx(0.15)
    assert not s["sufficient_evidence"]


# ---------------------------------------------------------------- random baseline
@pytest.fixture(scope="module")
def setup_data():
    c = cfg(**{"costs.commission.value": 0.0005, "costs.slippage.value": 0.0001})
    df = resample(generate_bars(days=30, seed=4), "5min")
    return c, df, build_features(df, c)


def _scalar_trade(f, c, s, d, stop_atr, target_R):
    """Reference implementation using the engine's own exit_on_bar, bar by bar."""
    o, h, lo, cl = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
    sess, last_bar = f["session_id"].to_numpy(), f["is_last_bar"].to_numpy()
    k = s + 1
    fill = o[k] + d * slippage_amount(c, o[k])
    risk = stop_atr * f["atr"].iloc[s]
    stop, target = fill - d * risk, fill + d * target_R * risk
    through = c["instrument"]["tick_size"]
    while True:
        hit = exit_on_bar(d, o[k], h[k], lo[k], stop, target, through, True)
        if hit:
            raw = hit[1]
            ex = raw - d * slippage_amount(c, raw) if hit[0] == "stop" else raw
            break
        if last_bar[k] or k + 1 >= len(o) or sess[k + 1] != sess[k]:
            ex = cl[k] - d * slippage_amount(c, cl[k])
            break
        k += 1
    comm = commission_amount(c, fill) + commission_amount(c, ex)
    return ((ex - fill) * d - comm) / risk


def test_vectorised_baseline_matches_engine_exit_rules(setup_data):
    c, _, f = setup_data
    rb = RandomBaseline(f, c)
    rng = np.random.default_rng(1)
    s = rb.eligible[rng.integers(0, len(rb.eligible), 300)]
    for d in (1, -1):
        vec = rb.simulate(s, d, 1.5, 2.0)
        ref = np.array([_scalar_trade(f, c, int(x), d, 1.5, 2.0) for x in s])
        np.testing.assert_allclose(vec, ref, rtol=1e-9, atol=1e-9)


def test_baseline_distribution_and_p_value(setup_data):
    c, _, f = setup_data
    trades = run_backtest(f, c).trades
    rb = RandomBaseline(f, c)
    dist = rb.distribution(trades, reps=200, seed=3)
    assert dist.shape == (200,)
    # random entries with costs lose roughly the costs on average
    assert dist.mean() < 0.1
    cmp = compare_to_baseline(float(dist.max()) + 1, dist)
    assert cmp["p_value"] == pytest.approx(1 / 201)
    cmp = compare_to_baseline(float(dist.min()) - 1, dist)
    assert cmp["p_value"] == pytest.approx(1.0)


# ---------------------------------------------------------------- tuning and ledger
def test_expand_grid():
    g = expand_grid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(g) == 6 and {"a": 2, "b": "z"} in g


def test_split_dates_chronological(setup_data):
    c, df, _ = setup_data
    start, cut, end = split_dates(df.index, c)
    assert start < cut < end and cut == cut.floor("D")
    c2 = cfg(**{"evaluation.in_sample_end": "2023-01-20"})
    assert split_dates(df.index, c2)[1] == pd.Timestamp("2023-01-20", tz="UTC")


def test_grid_ledger_and_walk_forward(setup_data, tmp_path):
    c, df, _ = setup_data
    c = cfg(**{"costs.commission.value": 0.0005, "evaluation.min_trades_for_selection": 1,
               "evaluation.grid": {"exits.target.r_multiple": [1.0, 2.0]},
               "evaluation.walk_forward.train_days": 10, "evaluation.walk_forward.test_days": 5,
               "evaluation.walk_forward.step_days": 5})
    ledger = Ledger(tmp_path / "ledger.jsonl")
    start, cut, end = split_dates(df.index, c)
    table = evaluate_grid(df, c, start, cut, None, ledger, "fp1", "in_sample", workers=1)
    assert len(table) == 2 and table["objective"].is_monotonic_decreasing
    assert ledger.combos_tried("fp1") == 2
    evaluate_grid(df, c, start, cut, None, ledger, "fp1", "in_sample", workers=1)
    assert ledger.combos_tried("fp1") == 2  # same configs again do not inflate the count
    folds, oos = walk_forward(df, c, start, end, ledger, "fp1", workers=1)
    assert len(folds) >= 2
    assert (pd.to_datetime(folds["test_start"]) > pd.to_datetime(folds["train_start"])).all()
    if len(oos):
        for k, fold in folds.iterrows():
            sub = oos[oos["fold"] == k + 1]
            assert (sub["signal_time"].dt.date >= fold["test_start"]).all()
    assert ledger.oos_touches("fp1") == 0
