"""Each setup tested on hand-built bar sequences with a known answer."""

import numpy as np
import pytest

from helpers import bars_from_path, cfg, double_top_day2, mirror, ramp, scan
from reversal_lab.indicators import build_features
from reversal_lab.indicators.features import FeatureWindow, to_arrays
from reversal_lab.strategy import (Strategy, find_double, setup_ma, setup_pdh_pdl, setup_session_extreme,
                                   setup_vwap)


@pytest.fixture(scope="module")
def dt():
    c = cfg()
    df, info = double_top_day2()
    return c, df, build_features(df, c), info


# ---------------------------------------------------------------- A+: PDH / PDL
def test_pdh_double_top_fires_short_after_confirmation(dt):
    c, _, f, info = dt
    hits = scan(setup_pdh_pdl, f, c)
    assert len(hits) == 1
    i, cands = hits[0]
    cand = cands[0]
    assert cand.setup == "pdh_pdl" and cand.direction == -1 and cand.level_name == "PDH"
    assert cand.pattern.pivots == (info["p1"], info["p2"])
    # the second peak can only be known 3 bars (swing.right) after it happened
    assert cand.pattern.confirm_idx == info["p2"] + c["indicators"]["swing"]["right"]
    assert i >= cand.pattern.confirm_idx
    # neckline break: this bar closes below the neckline, the previous ones did not
    assert f["close"].iloc[i] < cand.pattern.neckline
    assert (f["close"].iloc[cand.pattern.confirm_idx:i] >= cand.pattern.neckline).all()


def test_pdl_double_bottom_fires_long():
    c = cfg()
    df, info = double_top_day2()
    f = build_features(mirror(df), c)
    hits = scan(setup_pdh_pdl, f, c)
    assert len(hits) == 1
    cand = hits[0][1][0]
    assert cand.direction == 1 and cand.level_name == "PDL"
    assert cand.pattern.pivots == (info["p1"], info["p2"])


def test_pdh_not_near_level_does_not_fire():
    c = cfg()
    df, _ = double_top_day2(pdh=115.0)  # peaks at ~110 are 5 ATR away from PDH
    assert scan(setup_pdh_pdl, build_features(df, c), c) == []


def test_peaks_too_different_rejected():
    c = cfg(**{"setups.double.peak_tolerance.value": 0.01})
    df, _ = double_top_day2(peak1=109.3, peak2=109.95)
    assert scan(setup_pdh_pdl, build_features(df, c), c) == []


def test_peaks_too_far_apart_rejected():
    c = cfg(**{"setups.double.max_bars_apart": 10})
    df, _ = double_top_day2()
    assert scan(setup_pdh_pdl, build_features(df, c), c) == []


def test_pullback_too_shallow_rejected():
    c = cfg(**{"setups.double.min_pullback.value": 5.0})
    df, _ = double_top_day2()
    assert scan(setup_pdh_pdl, build_features(df, c), c) == []


def test_pattern_invalidated_if_price_exceeds_extreme_before_trigger():
    c = cfg()
    # after peak 2: drift sideways, spike above the double top, then fall through the neckline
    after = ramp(109.65, 108.5, 3) + [108.5, 110.3, 108.0] + ramp(108.0, 105.0, 8)
    df, info = double_top_day2(after_break=after)
    f = build_features(df, c)
    assert f["high"].iloc[info["p2"] + 5] > 110.0  # the spike really exceeds the double top
    hits = scan(setup_pdh_pdl, f, c)
    assert not any(h[1][0].pattern.pivots == (info["p1"], info["p2"]) for h in hits)


def test_trigger_expires():
    c = cfg(**{"entry.max_bars_to_trigger": 1})
    df, _ = double_top_day2()  # neckline break happens ~3 bars after confirmation
    assert scan(setup_pdh_pdl, build_features(df, c), c) == []


def test_confirmation_trigger_fires_on_confirmation_bar(dt):
    c = cfg(**{"entry.trigger": "confirmation"})
    _, df, _, info = dt
    hits = scan(setup_pdh_pdl, build_features(df, c), c)
    assert [h[0] for h in hits] == [info["p2"] + 3]


# ---------------------------------------------------------------- secondary: session extreme
def test_session_high_double_top(dt):
    c = cfg()
    df, info = double_top_day2(pdh=130.0)
    hits = scan(setup_session_extreme, build_features(df, c), c)
    assert len(hits) == 1
    cand = hits[0][1][0]
    assert cand.setup == "session_extreme" and cand.direction == -1
    assert cand.level == pytest.approx(109.95)


def test_priority_prefers_pdh_and_records_overlap(dt):
    c, _, f, _ = dt
    arrays, times = to_arrays(f)
    strat = Strategy(c)
    sigs = [s for i in range(len(times)) if (s := strat.evaluate(FeatureWindow(arrays, times, i)))]
    assert len(sigs) == 1
    s = sigs[0]
    assert s.setup == "pdh_pdl" and s.also_matches == ("session_extreme",)
    assert s.stop == pytest.approx(109.95 + 0.1 * s.atr, abs=0.011)  # extreme + 0.1 ATR buffer, tick-rounded
    assert s.stop > s.extreme


def test_find_double_ignores_unconfirmed_second_peak(dt):
    c, _, f, info = dt
    arrays, times = to_arrays(f)
    # on the bars before confirmation, the second peak is not yet a known swing
    for i in range(info["p2"], info["p2"] + 3):
        pat = find_double(FeatureWindow(arrays, times, i), "top", c)
        assert pat is None or pat.pivots[-1] != info["p2"]


# ---------------------------------------------------------------- tertiary: VWAP and MA
def _vwap_pullback_df():
    """Heavy volume at 100 pins VWAP near 100; price rises to 106, then dips back to tag VWAP."""
    day1 = [100.0] * 288
    day2 = [100.0] * 10 + ramp(100.0, 106.0, 30) + [106.0] * 10 + ramp(106.0, 100.3, 12) + ramp(100.3, 104.0, 10)
    path = day1 + day2 + [104.0] * (576 - len(day1) - len(day2))
    df = bars_from_path(path)
    df.iloc[288:298, df.columns.get_loc("volume")] = 1e6
    return df


def test_vwap_rejection_long():
    c = cfg()
    df = _vwap_pullback_df()
    f = build_features(df, c)
    hits = scan(setup_vwap, f, c)
    assert hits, "expected a VWAP reversal"
    i, cands = hits[0]
    cand = cands[0]
    assert cand.direction == 1 and cand.setup == "vwap"
    p = cand.pattern.pivots[0]
    assert abs(f["low"].iloc[p] - f["vwap"].iloc[p]) <= 0.5 * f["atr"].iloc[i] + 1e-9
    assert i >= p + 3


def test_vwap_rejection_short_mirror():
    c = cfg()
    f = build_features(mirror(_vwap_pullback_df()), c)
    hits = scan(setup_vwap, f, c)
    assert hits and hits[0][1][0].direction == -1


def test_ma_rejection_long():
    c = cfg(**{"indicators.ma.period": 50})
    day1 = [100.0] * 288
    # trend up well above the 50-bar SMA, then a shallow dip whose low tags the (rising) SMA
    day2 = ramp(100.0, 104.0, 40) + [104.0] * 20 + ramp(104.0, 103.6, 8) + ramp(103.6, 104.5, 10)
    df = bars_from_path(day1 + day2 + [104.5] * (576 - 288 - len(day2)))
    f = build_features(df, c)
    p = 288 + 40 + 20 + 8 - 1  # bottom of the dip
    hits = scan(setup_ma, f, c)
    assert hits and hits[0][1][0].direction == 1
    i, cands = hits[0]
    assert cands[0].pattern.pivots == (p,)
    assert abs(f["low"].iloc[p] - f["ma"].iloc[p]) <= 0.5 * f["atr"].iloc[i]


def test_tertiary_double_mode():
    c = cfg(**{"setups.reversal.pattern": "double", "setups.reversal.approach_bars": 5})
    df = _vwap_pullback_df()
    f = build_features(df, c)
    # a single dip is not a double bottom
    assert scan(setup_vwap, f, c) == []
