"""End-to-end evaluation: in-sample, tuning, walk-forward, a one-time out-of-sample test, and the report."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import run_backtest
from ..config import SETUP_NAMES, config_hash, strategy_hash, with_overrides
from ..data.loaders import load_data
from . import plots
from .baseline import RandomBaseline, compare_to_baseline
from .ledger import Ledger, data_fingerprint
from .metrics import summarize_by_setup
from .tuning import _features_for, best_overrides, evaluate_grid, split_dates, walk_forward


# ------------------------------------------------------------------ formatting helpers
def _fmt(v, kind: str = "f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "r":
        return f"{v:+.3f}"
    if kind == "i":
        return f"{int(v)}"
    return f"{v:.2f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def metrics_table(summary: pd.DataFrame, base: dict[str, dict]) -> str:
    rows = []
    for name, s in summary.iterrows():
        b = base.get(name, {})
        rows.append({
            "Setup": plots.LABELS.get(name, name),
            "Trades": _fmt(s["trades"], "i"),
            "Win rate [95% CI]": f"{_fmt(s['win_rate'], 'pct')} [{_fmt(s['win_rate_ci_low'], 'pct')}–{_fmt(s['win_rate_ci_high'], 'pct')}]",
            "Expectancy R [95% CI]": f"{_fmt(s['expectancy_R'], 'r')} [{_fmt(s['expectancy_ci_low'], 'r')}, {_fmt(s['expectancy_ci_high'], 'r')}]",
            "Avg win / loss R": f"{_fmt(s['avg_win_R'], 'r')} / {_fmt(s['avg_loss_R'], 'r')}",
            "Profit factor": _fmt(s["profit_factor"]),
            "Max DD (R)": _fmt(s["max_drawdown_R"]),
            "Longest losing streak": _fmt(s["longest_losing_streak"], "i"),
            "Costs per trade (R)": _fmt(s["avg_cost_R"]),
            "Expectancy before costs": _fmt(s["expectancy_before_costs_R"], "r"),
            "Random-entry p": _fmt(b.get("p_value"), "f") if b else "–",
            "Evidence": "OK (n≥min)" if s["sufficient_evidence"] else "⚠ INSUFFICIENT",
        })
    return md_table(pd.DataFrame(rows))


def baseline_for(features, cfg, trades, start, end, setups, out_dir: Path, tag: str) -> dict[str, dict]:
    ev = cfg["evaluation"]["baseline"]
    rb = RandomBaseline(features, cfg, start, end)
    res = {}
    for name in list(setups) + ["combined"]:
        sub = trades if name == "combined" else trades[trades["setup"] == name]
        if len(sub) == 0:
            continue
        dist = rb.distribution(sub, reps=ev["reps"], seed=ev["seed"])
        actual = float(sub["R"].mean())
        res[name] = compare_to_baseline(actual, dist)
        if name == "combined":
            plots.baseline_hist(dist, actual, res[name]["p_value"], str(out_dir / f"baseline_{tag}.png"),
                                f"Random-entry baseline ({tag}, {ev['reps']} reps): combined")
    return res


def _counter_table(c) -> str:
    if not c:
        return "_none_"
    rows = [{"Setup": k[0], "Reason": k[1], "Count": v} for k, v in sorted(c.items())]
    return md_table(pd.DataFrame(rows))


# ------------------------------------------------------------------ main pipeline
def run_full_evaluation(cfg: dict, out_dir: str | Path, ledger_path: str | Path | None = None,
                        workers: int | None = None, df: pd.DataFrame | None = None,
                        data_report: dict | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if df is None:
        df, data_report = load_data(cfg)
    ledger = Ledger(ledger_path or cfg["evaluation"]["ledger_path"])
    fp = data_fingerprint(df)
    setups = [s for s in SETUP_NAMES if s in cfg["setups"]["enabled"]]
    min_n = cfg["evaluation"]["min_trades_for_evidence"]
    bs = cfg["evaluation"]["bootstrap_reps"]
    start, cut, end = split_dates(df.index, cfg)
    grid = cfg["evaluation"]["grid"]
    keys = list(grid)
    print(f"Data {df.index[0]} → {df.index[-1]} ({len(df)} bars); IS < {cut} ≤ OOS")

    # 1) In-sample with the configuration as written (pre-registered defaults)
    feats = _features_for(df, cfg)
    is_trades = run_backtest(feats, cfg, start, cut)
    ledger.record(data_fp=fp, segment="in_sample", strategy_hash=strategy_hash(cfg), params={},
                  start=start, end=cut, trades=len(is_trades.trades),
                  expectancy_R=float(is_trades.trades["R"].mean()) if len(is_trades.trades) else None)
    is_sum = summarize_by_setup(is_trades.trades, setups, min_n, bs)
    is_base = baseline_for(feats, cfg, is_trades.trades, start, cut, setups, out, "in_sample_default")
    plots.equity_curves(is_trades.trades, str(out / "equity_in_sample_default.png"),
                        "In-sample, default parameters")
    is_trades.trades.to_csv(out / "trades_in_sample_default.csv", index=False)
    print("  in-sample default done")

    # 2) Grid search on in-sample only
    table = evaluate_grid(df, cfg, start, cut, grid, ledger, fp, "in_sample", workers)
    table.to_csv(out / "grid_in_sample.csv", index=False)
    finite = np.isfinite(table["objective"]).any()
    chosen = best_overrides(table, keys) if finite else {}
    ccfg = with_overrides(cfg, chosen)
    print(f"  grid done: {len(table)} combos; chosen {chosen}")

    # 3) Walk-forward (by default inside the in-sample period so the holdout stays untouched)
    wf_end = cut if cfg["evaluation"]["walk_forward"]["span"] == "in_sample" else end
    folds, wf_trades = walk_forward(df, cfg, start, wf_end, ledger, fp, workers)
    folds.to_csv(out / "walk_forward_folds.csv", index=False)
    wf_sum = summarize_by_setup(wf_trades, setups, min_n, bs) if len(wf_trades) else None
    plots.equity_curves(wf_trades, str(out / "equity_walk_forward.png"),
                        "Walk-forward: stitched test windows")
    print(f"  walk-forward done: {len(folds)} folds")

    # 4) Out-of-sample: touched once, with the chosen parameters frozen
    prior_touches = ledger.oos_touches(fp)
    cfeats = _features_for(df, ccfg)
    oos = run_backtest(cfeats, ccfg, cut, end)
    ledger.record(data_fp=fp, segment="oos", strategy_hash=strategy_hash(ccfg), params=chosen,
                  start=cut, end=end, trades=len(oos.trades),
                  expectancy_R=float(oos.trades["R"].mean()) if len(oos.trades) else None)
    oos_sum = summarize_by_setup(oos.trades, setups, min_n, bs)
    oos_base = baseline_for(cfeats, ccfg, oos.trades, cut, end, setups, out, "out_of_sample")
    plots.equity_curves(oos.trades, str(out / "equity_out_of_sample.png"),
                        "Out-of-sample (holdout), chosen parameters")
    oos.trades.to_csv(out / "trades_out_of_sample.csv", index=False)
    print("  out-of-sample done")

    # chosen config: in-sample stats too (to show the optimism of the best-of-grid pick)
    chosen_is = table.iloc[0].to_dict() if len(table) else {}
    total_combos = ledger.combos_tried(fp)

    results = {
        "data_fingerprint": fp, "config_hash": config_hash(cfg), "chosen": chosen,
        "split": {"start": str(start), "in_sample_end": str(cut), "end": str(end)},
        "in_sample_default": is_sum.to_dict(orient="index"), "in_sample_default_baseline": is_base,
        "out_of_sample": oos_sum.to_dict(orient="index"), "out_of_sample_baseline": oos_base,
        "walk_forward": wf_sum.to_dict(orient="index") if wf_sum is not None else None,
        "grid_combos_this_run": len(table), "combos_tried_on_this_data_total": total_combos,
        "oos_touches_before_this_run": prior_touches,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # ------------------------------------------------------------------ markdown
    L = []
    L.append(f"# Reversal strategy evaluation: {cfg['instrument']['symbol']} {cfg['timeframe']}\n")
    if (data_report or {}).get("source") == "synthetic":
        L.append("> **Synthetic data.** These bars are a random walk with no exploitable pattern "
                 "by construction. Any apparent edge below is noise. The point of this report is to show "
                 "the evaluation machinery, and that it correctly finds *nothing*.\n")
    if prior_touches:
        L.append(f"> ⚠ **The out-of-sample period had already been evaluated {prior_touches} time(s) on this "
                 "dataset.** It is no longer a clean holdout; treat its results as in-sample.\n")
    L.append("## Data and design\n")
    L.append(f"- Bars: {len(df):,} × {cfg['timeframe']}, {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d} "
             f"(data fingerprint `{fp}`, config `{config_hash(cfg)}`)")
    L.append(f"- Session: {cfg['session']['start']} {cfg['session']['timezone']} day boundary; entries "
             f"{cfg['session']['entry_start']}–{cfg['session']['entry_end']}; flatten at session end: "
             f"{cfg['session']['flatten_at_session_end']}")
    L.append(f"- **In-sample (tuning):** {start:%Y-%m-%d} → {cut:%Y-%m-%d}.  **Out-of-sample (holdout):** "
             f"{cut:%Y-%m-%d} → {end:%Y-%m-%d}")
    L.append(f"- Costs: slippage {cfg['costs']['slippage']}, commission {cfg['costs']['commission']} per side; "
             "fills at next bar open; stop-first when a bar touches both stop and target")
    L.append(f"- Evidence threshold: {min_n} trades. Random baseline: {cfg['evaluation']['baseline']['reps']} "
             "repetitions.\n")

    L.append("## 1. In-sample, default (pre-specified) parameters\n")
    L.append(metrics_table(is_sum, is_base) + "\n")
    L.append("![](equity_in_sample_default.png)\n")
    L.append("![](baseline_in_sample_default.png)\n")
    L.append("Signals rejected by filters (stop too tight/wide, target too close):\n")
    L.append(_counter_table(is_trades.rejections) + "\n")
    L.append("Orders cancelled between signal and fill:\n")
    L.append(_counter_table(is_trades.skipped) + "\n")

    L.append("## 2. Parameter search (in-sample only)\n")
    L.append(f"- Combinations evaluated in this run's grid: **{len(table)}**")
    L.append(f"- Distinct configurations ever evaluated on this dataset's tuning data (ledger, incl. "
             f"walk-forward): **{total_combos}**")
    L.append("- Selection objective: "
             f"`{cfg['evaluation']['objective']}` with ≥{cfg['evaluation']['min_trades_for_selection']} trades\n")
    top = table.head(10).copy()
    for c in ("expectancy_R", "total_R", "objective"):
        top[c] = top[c].map(lambda v: _fmt(float(v), "r"))
    L.append(md_table(top.drop(columns=["strategy_hash"])) + "\n")
    L.append(f"**Chosen:** `{chosen}`. Its in-sample expectancy ({_fmt(chosen_is.get('expectancy_R'), 'r')}R) is the "
             "*best of many tries*, so expect it to be optimistic (selection bias / winner's curse).\n")

    L.append("## 3. Walk-forward analysis\n")
    L.append(f"Train {cfg['evaluation']['walk_forward']['train_days']}d → test "
             f"{cfg['evaluation']['walk_forward']['test_days']}d, step {cfg['evaluation']['walk_forward']['step_days']}d, "
             f"span: {cfg['evaluation']['walk_forward']['span']}. Each fold re-runs the grid on its training "
             "window and trades the winner on the next window.\n")
    if len(folds):
        f2 = folds.copy()
        for c in ("train_expectancy_R", "test_expectancy_R"):
            f2[c] = f2[c].map(lambda v: _fmt(float(v), "r"))
        L.append(md_table(f2) + "\n")
    if wf_sum is not None:
        L.append("Stitched test-window results:\n")
        L.append(metrics_table(wf_sum, {}) + "\n")
        L.append("![](equity_walk_forward.png)\n")
    else:
        L.append("_Not enough data for a single walk-forward fold, or no trades._\n")

    L.append("## 4. Out-of-sample holdout (evaluated once, parameters frozen)\n")
    L.append(metrics_table(oos_sum, oos_base) + "\n")
    L.append("![](equity_out_of_sample.png)\n")
    L.append("![](baseline_out_of_sample.png)\n")
    if oos_base:
        brows = [{"Setup": plots.LABELS.get(k, k), "Strategy mean R": _fmt(float(oos_sum.loc[k, "expectancy_R"]), "r"),
                  "Random mean R": _fmt(v["baseline_mean"], "r"),
                  "Random 5–95%": f"{_fmt(v['baseline_p05'], 'r')} to {_fmt(v['baseline_p95'], 'r')}",
                  "Percentile": _fmt(v["percentile"]), "p-value": _fmt(v["p_value"])}
                 for k, v in oos_base.items()]
        L.append("Where the strategy falls in the random-entry distribution:\n")
        L.append(md_table(pd.DataFrame(brows)) + "\n")

    L.append("## 5. How to read this\n")
    L.append(interpretation(oos_sum, oos_base, min_n, len(table), total_combos))
    (out / "report.md").write_text("\n".join(L))
    print(f"Report written to {out / 'report.md'}")
    return results


def interpretation(oos_sum: pd.DataFrame, oos_base: dict, min_n: int, n_grid: int, n_total: int) -> str:
    lines = []
    comb = oos_sum.loc["combined"]
    p = oos_base.get("combined", {}).get("p_value", float("nan"))
    lines.append(f"- Out-of-sample combined: {int(comb['trades'])} trades, expectancy "
                 f"{_fmt(comb['expectancy_R'], 'r')}R (95% CI {_fmt(comb['expectancy_ci_low'], 'r')} to "
                 f"{_fmt(comb['expectancy_ci_high'], 'r')}), random-entry p = {_fmt(p)}.")
    if np.isfinite(comb["expectancy_ci_low"]) and comb["expectancy_ci_low"] > 0:
        lines.append("- The CI for expectancy excludes zero. That is encouraging, but check it survives the "
                     "number of combinations tried and holds up on more data.")
    else:
        lines.append("- The CI for expectancy includes zero (or is below it): **no evidence of an edge**.")
    thin = [plots.LABELS.get(k, k) for k, r in oos_sum.iterrows() if not r["sufficient_evidence"]]
    if thin:
        lines.append(f"- Fewer than {min_n} trades (insufficient evidence, do not draw conclusions): {', '.join(thin)}.")
    lines.append(f"- {n_total} configurations have been tried on this dataset. With that many tries, "
                 "a best-of-grid in-sample result is expected to look better than it really is; only the "
                 "walk-forward and holdout sections are honest estimates.")
    lines.append("- 'Expectancy before costs' vs 'Expectancy' shows how much of the result fees and slippage "
                 "consume. With 5-minute stops, costs are a large fraction of 1R.")
    return "\n".join(lines) + "\n"
