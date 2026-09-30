# reversal_lab: honest evaluation of an intraday double-top/bottom reversal strategy

A research and paper-trading toolkit for a discretionary reversal strategy taken
from a trading video, turned into explicit, testable rules. It is built for
**honest evaluation**, not for trading.

> This repository also contains an unrelated Streamlit charting app (`app.py`,
> `requirements.txt`). The research toolkit lives in `reversal_lab/` and uses
> `requirements-research.txt`.

> **Safety.** Nothing in this project can place a real order. There is no
> broker adapter and no order-routing code (a test scans the package for order
> endpoints). `python -m reversal_lab paper --live` exits with an error on
> purpose. Paper trading only *simulates* fills, using public market data.

Default instrument: **BTCUSDT, 5-minute bars, 24-hour UTC session**
(PDH/PDL, session high/low and VWAP all reset at 00:00 UTC). Everything is in
`config.yaml`.

---

## 1. Quick start

```bash
pip install -r requirements-research.txt
python -m pytest                        # 65 tests, including the no-look-ahead suite

# Data: pick one
python -m reversal_lab download-binance --start 2022-01 --end 2025-08   # free, years of history
#   or put TradingView "Export chart data" CSVs in data/raw/ and set data.path to a glob,
#   e.g. data/raw/BINANCE_BTCUSDT*.csv (overlapping exports are stitched & de-duplicated)
#   or: python -m reversal_lab synth --days 800   (random walk, no edge by construction)

# Full evaluation → reports/run1/report.md (+ PNGs, CSVs, results.json)
python -m reversal_lab report --out reports/run1
python -m reversal_lab report --synthetic --out reports/sample --ledger reports/sample/ledger.jsonl

# Pieces
python -m reversal_lab backtest --start 2024-01-01 --end 2024-07-01
python -m reversal_lab tune                     # grid search on in-sample only (logged to the ledger)
python -m reversal_lab walkforward
python -m reversal_lab oos --params '{"exits.target.r_multiple": 2.0}'   # refuses a 2nd look unless --again
python -m reversal_lab paper --replay data/raw/BTCUSDT_5m.csv --warmup-bars 3000 --max-bars 2000
python -m reversal_lab paper                    # live Binance public data, simulated fills only
```

Any config value can be overridden on the command line with dotted keys:
`--params '{"setups.near_tolerance.value": 0.3, "entry.trigger": "confirmation"}'`.

## 2. Layout

```
reversal_lab/
  config.py            defaults, YAML loading, strict validation (unknown keys are errors)
  data/                loaders.py (TradingView/generic CSV, stitching, cleaning, resampling),
                       binance.py (free monthly archives), synthetic.py, BrokerDataAPI stub
  indicators/          core.py (sessions, ATR, VWAP, SMA/EMA on any timeframe, PDH/PDL,
                       session high/low, swing points), features.py (FeatureWindow)
  strategy/            patterns.py (double top/bottom, rejection, triggers),
                       setups.py (one function per setup), strategy.py (priority, stops, targets, filters)
  backtest/engine.py   bar-by-bar event loop + fill/cost model (shared with paper trading)
  evaluation/          metrics.py, baseline.py (random entries), tuning.py (split, grid, walk-forward),
                       ledger.py (every configuration ever tried), plots.py, report.py
  paper/trader.py      same Engine fed one closed bar at a time; replay or Binance feed
  cli.py
config.yaml            every parameter, commented
tests/                 pytest suite
reports/sample/        sample report on synthetic data
experiments/ledger.jsonl  created on first run: append-only record of every configuration evaluated
```

## 3. The rules, precisely

The video's rules are vague. Below is the exact version implemented, with the
config key for every threshold. "ATR" means Wilder's 14-bar ATR on the bar
timeframe, read at the bar where the decision is made.

### 3.1 Bars and sessions
* Strategy bars: `timeframe` (default 5min). Finer data is resampled; a bar that
  is still forming at the end of the file is dropped.
* Session ("trading day"): starts at `session.start` in `session.timezone`
  (default 00:00 UTC) and lasts 24h. For FX you would use `17:00` / `America/New_York`.
* `is_last_bar`: the bar whose close reaches the session end. This is known from
  the clock, not from seeing the next bar.
* New signals only when the signal bar closes inside `[entry_start, entry_end]`
  (default 00:00–23:30), and not before bar `min_bars_into_session` (6 = 30 min).
* `flatten_at_session_end: true`: any open trade is closed at the close of the last bar.

### 3.2 Levels and indicators
| Item | Definition |
|---|---|
| **PDH / PDL** | High / low of the previous session in the data. NaN if that session has < `min_prev_session_coverage` (90%) of its bars. |
| **Session high / low** | Running max / min from the session start up to and including the current bar. |
| **VWAP** | Σ(price·volume)/Σvolume from the session start; `price` = typical (H+L+C)/3 or close. Resets every session. If volume is zero, it falls back to an equal-weight average. |
| **200 MA** | SMA or EMA (`indicators.ma.type`) of closes with `period` 200 on `ma.timeframe`. For a higher timeframe (e.g. `1h`), a bar's value is used only after that hourly bar has closed. |
| **Swing high** | Bar *j* whose high is **strictly greater** than the `left` (3) previous highs and **≥** the `right` (3) following highs. It is *confirmed* (becomes usable) only on bar *j*+`right`, after those bars close. Swing lows mirror this. |

### 3.3 Double top (double bottom is the mirror image)
At the current bar *i*, a double top exists if:
1. **p2** is the most recent confirmed swing high, confirmed no more than
   `entry.max_bars_to_trigger` (12) bars ago, in the current session.
2. **p1** is an earlier confirmed swing high in the same session, with
   `min_bars_apart` (5) ≤ p2 − p1 ≤ `max_bars_apart` (60). Up to `max_candidates` (5)
   earlier swings are tried, nearest first.
3. |high(p1) − high(p2)| ≤ `peak_tolerance` (0.25 ATR).
4. No bar between p1 and p2 is higher than the **pattern extreme** = max(high(p1), high(p2)).
5. **Neckline** = lowest low strictly between p1 and p2, and extreme − neckline ≥ `min_pullback` (1.0 ATR).
6. **Invalidation:** if any bar after p2 has traded above the extreme, the pattern is dead.

### 3.4 The setups
| Setup | Config name | Rule | Direction |
|---|---|---|---|
| **A+** | `pdh_pdl` | Double top whose extreme is within `near_tolerance` (0.5 ATR) of PDH; double bottom within tolerance of PDL | top → short, bottom → long |
| **Secondary** | `session_extreme` | Double top whose extreme is within `near_tolerance` of the *current* running session high (so it is still the day's high); mirror for lows | top → short, bottom → long |
| **Tertiary: VWAP** | `vwap` | `reversal.pattern: rejection` (default): the latest confirmed swing low tagged VWAP (\|low − VWAP\| ≤ near_tolerance) after ≥ `approach_min_frac` (80%) of the previous `approach_bars` (12) closes were *above* VWAP, and the swing bar closed back near/above VWAP. Mirror for shorts from below. `pattern: double` instead requires a double bottom/top at VWAP with the same approach condition. | approached from above → long; from below → short |
| **Tertiary: MA** | `ma` | Same as VWAP, with the configured SMA/EMA | same |

If several setups fire on the same bar, the first in `setups.priority` wins
(A+ > session > VWAP > MA). The trade log records the other matches in
`also_matches`, because an A+ double top at PDH is very often also the session high.

### 3.5 Entry
The trigger must fire for the **first time** on bar *i*, between the confirmation bar and
`max_bars_to_trigger` bars after it:
* `neckline_break` (default): close below the neckline (tops) / above it (bottoms).
  For a rejection pattern, the "neckline" is the opposite extreme of the swing bar.
* `close_back_through_level`: close back below the level (PDH, session high, VWAP, MA) for shorts.
* `confirmation`: enter on the confirmation bar itself (most aggressive).

`trigger_offset` requires the close to go this much beyond the reference. **The order is
filled at the next bar's open**, plus slippage.

### 3.6 Stop, target, filters
* **Stop** = pattern extreme ± `stop_buffer` (0.1 ATR), rounded to the tick, away from the entry.
* **Risk (1R)** = |fill price − stop|, measured from the actual fill.
* **Target** (`exits.target.mode`):
  * `r_multiple` (default 2.0): fill ± 2R;
  * `vwap`: session VWAP at the signal bar;
  * `opposite_level`: PDL for a PDH short (PDH for a PDL long); for the other setups,
    the opposite session extreme.
  * Fixed targets closer than `min_reward_risk` (1.0R) skip the trade.
* **Filters:** skip if risk < `min_stop` (0.5 ATR; costs would dominate) or > `max_stop` (4 ATR).
* Optional time stop `max_hold_bars`.
* **One position at a time** (always), and at most `risk.max_trades_per_day` (3) fills per session.
* Each pattern (identified by its last pivot) can only be traded once.

### 3.7 Fills and costs
* Entries, stop exits and time exits are market orders: adverse `slippage`
  (default 1 bp) on each.
* Targets are resting limit orders: no slippage, but by default price must
  trade **1 tick through** the target (`target_fill: through`).
* If the bar **gaps through the stop** at its open, the fill is at the open (worse).
* If one bar touches **both** stop and target, we cannot know which came first
  from OHLC data, so the default assumes **the stop** (`intrabar_priority: stop_first`).
* Commission `costs.commission` per side (default 5 bp of notional, a typical
  perpetual-futures taker fee; Binance spot is ~10 bp).
* Every trade reports `R` (net), `R_frictionless`, `commission_R` and `slippage_R`.

## 4. Assumptions I had to make

1. **"Near" the level** means the pattern *extreme* (the higher peak) is within
   0.5 ATR of the level, on either side. Peaks can poke slightly above PDH or stop just short of it.
2. **"Double top"** uses 3/3 fractal swings. Other swing definitions (ZigZag, % retrace)
   would find different patterns. `swing.left/right` is tunable.
3. The **second peak** may be lower or higher than the first (within tolerance). The stop
   goes beyond whichever is higher.
4. For the **secondary setup**, "at the session high" means the pattern extreme is
   still (within tolerance) the session's running high when the trigger fires.
5. The **tertiary "reversal off VWAP/MA"** is ambiguous in the video. I implemented a
   single-swing *rejection* by default and a double-pattern option. "Off" is read
   as: price approached the level from one side and bounced back.
6. For crypto the **"previous day"** is the previous UTC calendar day (00:00–24:00).
   The video presumably traded US stocks or futures with RTH sessions.
7. **One trade at a time across all setups** (not one per setup).
8. A pattern whose trigger bar arrives while a position is open is **not queued**; it is missed.
9. When the day ends, the trade is closed: **no overnight holds** (it is intraday).
10. **Costs** default to 5 bp commission and 1 bp slippage per side. Change them to match your venue.
11. **Wick-based levels**: PDH/PDL and swing points use highs/lows, not closes.
12. **Missing data**: PDH/PDL are disabled after an incomplete day. A position open
    across a data gap at the session boundary is closed at the next available open.

## 5. How look-ahead bias is prevented (and tested)

*Look-ahead bias* is using information that was not available when the decision was
made. It is the backtesting equivalent of immortal-time bias or letting the outcome
leak into the predictors: results look spectacular and cannot be reproduced.

* Every indicator is **causal**: its value at bar *i* uses bars 0..*i* only.
* Swing points are flagged on the **confirmation bar** (pivot + `right`), never on the pivot itself.
* Higher-timeframe MA values appear only after the higher-timeframe bar has closed.
* The strategy receives a `FeatureWindow` whose arrays *end at the current bar*.
  Indexing past it raises `IndexError`.
* Orders fill at the **next** bar's open, never at the signal bar's close.
* "Last bar of the session" comes from the clock, not from seeing the next bar.

`tests/test_no_lookahead.py` checks this four ways:
1. **Truncation invariance:** every feature computed on `bars[:k]` must equal the
   full-data feature sliced to `[:k]`. This is checked at 16 cut points under 4
   configurations (EMA on 1h, New York 17:00 sessions, and others).
2. **Future perturbation:** scramble all prices and volumes after time T. Every
   trade closed before T, and every signal before T, must be identical.
3. **Structural:** a spy strategy checks it only ever sees closed bars, and every
   fill happens exactly one bar after its signal, at that bar's open ± slippage.
4. **Testing the test:** a deliberately leaky indicator (`close.shift(-1)`) and a
   "centred" swing detector must be *caught* by check 1.

`tests/test_data_config_paper.py` also checks that paper trading on replayed
bars produces **exactly** the backtest's trades.

## 6. Evaluation methodology

All results are in **R**: net profit ÷ initial risk. A +2R trade made twice what it risked.

| Statistic | Definition / why |
|---|---|
| Win rate + **95% Wilson CI** | Share of trades with R > 0. Wilson rather than the "± 1.96·SE" (Wald) interval, which misbehaves at small n. |
| **Expectancy** | Mean R per trade = win% × avg win + loss% × avg loss. The number that matters. Its 95% CI is a percentile bootstrap (5,000 resamples of trades). |
| Profit factor | Σ winning R ÷ \|Σ losing R\|. Above 1 means profitable. |
| Max drawdown | Largest peak-to-trough fall in cumulative R. |
| Longest losing streak | Consecutive trades with R ≤ 0. Useful to prepare yourself psychologically. |
| Costs per trade | Commission + slippage in R. With 5-min stops this is often 0.1–0.3R per trade. |
| Evidence flag | **⚠ INSUFFICIENT** if fewer than `min_trades_for_evidence` (100) trades. |

**Random-entry baseline (permutation-style test).** Each real trade is paired with
a random-time trade with the same direction, stop distance (in ATR), target (in R),
costs and exit rules. Repeating this 1,000 times gives the distribution of results
that random timing produces. Reported p-value = (1 + #random ≥ strategy) / (1 + 1000).
It answers: *did the setup's timing add anything beyond the exits?*

**Chronological split.** The first `in_sample_fraction` (60%) of the calendar span is
**in-sample**: explore and tune freely. The rest is the **holdout**. Once the
configuration is frozen, `report`/`oos` evaluates it on the holdout. The ledger
records every look, `oos` refuses a second look without `--again`, and the report
prints a warning if the holdout has already been used.

**Walk-forward.** Rolling windows (train 180 days → test 60 days, step 60). On each
train window, re-run the whole grid, pick the best combination, then trade it
unchanged on the next test window. The stitched test windows estimate how the
*procedure* (strategy + tuning) would have done going forward. By default this runs
inside the in-sample period, so the holdout stays untouched.

**Multiple-testing bookkeeping.** `experiments/ledger.jsonl` appends every
configuration ever evaluated (with a fingerprint of the dataset). The report
states how many distinct configurations were tried on this data. The more
you try, the more the best in-sample result is inflated by luck.

### Statistics notes (for a clinical researcher)
* **In-sample / out-of-sample** = derivation / validation cohort. Tuning on the
  validation data is like choosing your cut-off on the test set: the reported
  accuracy is biased upwards.
* **Parameter grid** = multiple comparisons. If 18 combinations were independent
  tests of a strategy with no edge, at least one would be "significant" at α = 0.05
  with probability 1 − 0.95¹⁸ ≈ 60%. Neighbouring parameter values are correlated,
  so the real inflation is smaller, but it points the same way.
* **Random-entry baseline** = a permutation test / placebo arm. It holds everything
  constant except the thing being tested (entry timing).
* **Trades are not independent patients.** Clustering by volatility regime means CIs
  computed as if trades were i.i.d. are, if anything, too narrow.
* **~100 trades per setup** is a floor, not a guarantee. The CI width of a win rate at
  n = 100 is about ±10 percentage points.

## 7. Sample report (synthetic data, a negative control)

`reports/sample/report.md` (with charts, trade logs, grid and fold CSVs) was generated with

```bash
python -m reversal_lab report --synthetic --out reports/sample --ledger reports/sample/ledger.jsonl
```

on **800 days (230,400 five-minute bars) of synthetic BTC-like data**: a fat-tailed
random walk with volatility clustering. Tomorrow's price is independent of
yesterday's high, so **there is no edge to find**. A correct evaluation must
report none. It took ~2.5 minutes on 4 cores.

| Stage | Trades | Expectancy (R) [95% CI] | Before costs | Random-entry p |
|---|---|---|---|---|
| In-sample, default parameters (combined) | 1,268 | −0.154 [−0.234, −0.077] | +0.047 | 0.12 |
| Best of 18 grid combinations, in-sample | 1,230 | −0.078 | – | – |
| Walk-forward, 5 stitched test windows | 811 | −0.069 [−0.185, +0.049] | +0.164 | – |
| **Holdout, chosen parameters (touched once)** | **825** | **−0.217 [−0.330, −0.103]** | +0.001 | 0.58 |

What it demonstrates:
* **Costs dominate at 5-minute scale.** Commission + slippage average ~0.2R per
  trade. The "before costs" column is roughly zero, as a random walk should be;
  after costs every stage loses.
* **Selection bias is visible.** The best grid combination improved in-sample
  expectancy from −0.154 to −0.078R. On the holdout it fell to −0.217R.
* **A false positive, caught.** In-sample, the VWAP setup alone had random-entry
  p = 0.02 (it would have "passed"). On the holdout it was p = 0.34. With 4 setups ×
  several stages, some small p-values are expected by chance, which is why the
  holdout, the ledger and the walk-forward exist.
* **The A+ setup is rare**: 36 in-sample and 42 holdout trades in ~26 months, so it
  is flagged **insufficient evidence** everywhere. On real data you will likely need
  several years of history, or a looser `near_tolerance`, to judge it at all.

## 8. Known limitations
* OHLC bars cannot resolve the order of events inside a bar. The stop-first
  assumption is conservative, but it also means targets hit in the same bar as the
  stop are counted as losses.
* No market-impact model, funding rates (perpetuals) or borrow costs.
* TradingView exports contain only the bars loaded on the chart (often months, not
  years); use the Binance downloader for long history.
* Forex VWAP would rely on tick volume only (no centralised volume).
* The walk-forward grid is small (18 combinations) to keep run time down. Widen it
  in `evaluation.grid` knowing that the ledger counts every combination.
