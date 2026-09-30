"""Command-line interface.

    python -m reversal_lab report     --config config.yaml --out reports/run1
    python -m reversal_lab backtest   --config config.yaml [--start 2024-01-01 --end 2024-06-01]
    python -m reversal_lab tune       --config config.yaml           # in-sample grid only
    python -m reversal_lab walkforward --config config.yaml
    python -m reversal_lab oos        --config config.yaml --params '{"exits.target.r_multiple": 2.0}'
    python -m reversal_lab paper      --config config.yaml --replay data/raw/BTCUSDT_5m.csv --warmup-bars 3000
    python -m reversal_lab download-binance --start 2022-01 --end 2025-08
    python -m reversal_lab synth      --days 800 --out data/raw/synthetic_5m.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .backtest.engine import run_backtest
from .config import SETUP_NAMES, load_config, strategy_hash, with_overrides
from .data.loaders import load_data

LIVE_REFUSAL = (
    "Live trading is NOT implemented, by design. This project only simulates fills.\n"
    "Placing real orders would need a broker adapter, risk limits and an explicit\n"
    "confirmation step, none of which exist here."
)


def _cfg(args) -> dict:
    overrides = {}
    if getattr(args, "synthetic", False):
        overrides["data.source"] = "synthetic"
    if getattr(args, "data", None):
        overrides["data.source"] = "csv"
        overrides["data.path"] = args.data
    if getattr(args, "params", None):
        overrides.update(json.loads(args.params))
    return load_config(args.config, overrides)


def _print_summary(trades: pd.DataFrame, cfg: dict) -> None:
    from .evaluation.metrics import summarize_by_setup

    setups = [s for s in SETUP_NAMES if s in cfg["setups"]["enabled"]]
    s = summarize_by_setup(trades, setups, cfg["evaluation"]["min_trades_for_evidence"], 2000)
    cols = ["trades", "win_rate", "win_rate_ci_low", "win_rate_ci_high", "expectancy_R", "profit_factor",
            "max_drawdown_R", "longest_losing_streak", "sufficient_evidence"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(s[cols])


def cmd_backtest(args) -> None:
    from .evaluation.plots import equity_curves
    from .indicators.features import build_features

    cfg = _cfg(args)
    df, rep = load_data(cfg)
    print(f"Loaded {len(df)} bars: {rep}")
    res = run_backtest(build_features(df, cfg), cfg, args.start, args.end)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out / "trades.csv", index=False)
    equity_curves(res.trades, str(out / "equity.png"), "Backtest equity (R)")
    _print_summary(res.trades, cfg)
    print(f"Rejected signals: {dict(res.rejections)}\nCancelled orders: {dict(res.skipped)}")
    print(f"Trades → {out / 'trades.csv'}")


def cmd_tune(args) -> None:
    from .evaluation.ledger import Ledger, data_fingerprint
    from .evaluation.tuning import evaluate_grid, split_dates

    cfg = _cfg(args)
    df, _ = load_data(cfg)
    start, cut, _ = split_dates(df.index, cfg)
    ledger = Ledger(args.ledger or cfg["evaluation"]["ledger_path"])
    fp = data_fingerprint(df)
    table = evaluate_grid(df, cfg, start, cut, None, ledger, fp, "in_sample", args.workers)
    with pd.option_context("display.width", 200):
        print(table.drop(columns=["strategy_hash"]).head(20))
    print(f"\nIn-sample {start:%Y-%m-%d} → {cut:%Y-%m-%d}. Combos this run: {len(table)}; "
          f"distinct configs ever tried on this data: {ledger.combos_tried(fp)}")


def cmd_walkforward(args) -> None:
    from .evaluation.ledger import Ledger, data_fingerprint
    from .evaluation.tuning import split_dates, walk_forward

    cfg = _cfg(args)
    df, _ = load_data(cfg)
    start, cut, end = split_dates(df.index, cfg)
    wf_end = cut if cfg["evaluation"]["walk_forward"]["span"] == "in_sample" else end
    ledger = Ledger(args.ledger or cfg["evaluation"]["ledger_path"])
    folds, trades = walk_forward(df, cfg, start, wf_end, ledger, data_fingerprint(df), args.workers)
    with pd.option_context("display.width", 250):
        print(folds)
    _print_summary(trades, cfg)


def cmd_oos(args) -> None:
    from .evaluation.ledger import Ledger, data_fingerprint
    from .evaluation.tuning import split_dates
    from .indicators.features import build_features

    cfg = _cfg(args)
    df, _ = load_data(cfg)
    _, cut, end = split_dates(df.index, cfg)
    ledger = Ledger(args.ledger or cfg["evaluation"]["ledger_path"])
    fp = data_fingerprint(df)
    n = ledger.oos_touches(fp)
    if n and not args.again:
        sys.exit(f"The out-of-sample period of this dataset has already been evaluated {n} time(s).\n"
                 "Re-using it turns it into in-sample data. Re-run with --again if you accept that; "
                 "the ledger will record it.")
    res = run_backtest(build_features(df, cfg), cfg, cut, end)
    ledger.record(data_fp=fp, segment="oos", strategy_hash=strategy_hash(cfg),
                  params=json.loads(args.params or "{}"), start=cut, end=end, trades=len(res.trades),
                  expectancy_R=float(res.trades["R"].mean()) if len(res.trades) else None)
    print(f"OUT-OF-SAMPLE {cut:%Y-%m-%d} → {end:%Y-%m-%d} (touch #{n + 1})")
    _print_summary(res.trades, cfg)


def cmd_report(args) -> None:
    from .evaluation.report import run_full_evaluation

    cfg = _cfg(args)
    run_full_evaluation(cfg, args.out, args.ledger, args.workers)


def cmd_paper(args) -> None:
    from .paper.trader import BinancePollFeed, PaperTrader, ReplayFeed

    if args.live:
        sys.exit(LIVE_REFUSAL)
    cfg = _cfg(args)
    if args.replay:
        from .data.loaders import load_csv

        df, _ = load_csv(args.replay, tz=cfg["data"]["timezone"], timeframe=cfg["timeframe"])
        warm = df.iloc[: args.warmup_bars] if args.warmup_bars else None
        rest = df.iloc[args.warmup_bars:] if args.warmup_bars else df
        if args.max_bars:
            rest = rest.iloc[: args.max_bars]
        trader = PaperTrader(cfg, args.log_dir, warmup=warm, verbose=True)
        trades = trader.run(ReplayFeed(rest, args.delay))
    else:
        print("Polling Binance public market data (read-only). Ctrl+C to stop.")
        iv = {"1min": "1m", "5min": "5m", "15min": "15m"}[cfg["timeframe"]]
        trader = PaperTrader(cfg, args.log_dir, verbose=True)
        try:
            trades = trader.run(BinancePollFeed(cfg["instrument"]["symbol"], iv, cfg["paper"]["poll_seconds"]))
        except KeyboardInterrupt:
            trades = trader.engine.trades_frame()
    print(f"{len(trades)} simulated trades; log in {trader.log_dir}")


def cmd_download(args) -> None:
    from .data.binance import download_binance

    p = download_binance(args.symbol, args.timeframe, args.start, args.end, args.out)
    print(f"Wrote {p}")


def cmd_synth(args) -> None:
    from .data.synthetic import generate_bars
    from .data.loaders import resample

    df = resample(generate_bars(days=args.days, seed=args.seed), args.timeframe)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(args.out, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    print(f"Wrote {len(df)} bars to {args.out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="reversal_lab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, data=True):
        sp.add_argument("--config", default="config.yaml")
        sp.add_argument("--params", help="JSON dict of dotted-key overrides")
        if data:
            sp.add_argument("--data", help="CSV path/glob (overrides config data.path)")
            sp.add_argument("--synthetic", action="store_true", help="use synthetic data")
        sp.add_argument("--ledger", help="experiment ledger path (default from config)")
        sp.add_argument("--workers", type=int, default=None)

    sp = sub.add_parser("backtest"); common(sp)
    sp.add_argument("--start"); sp.add_argument("--end"); sp.add_argument("--out", default="reports/backtest")
    sp.set_defaults(fn=cmd_backtest)
    sp = sub.add_parser("tune"); common(sp); sp.set_defaults(fn=cmd_tune)
    sp = sub.add_parser("walkforward"); common(sp); sp.set_defaults(fn=cmd_walkforward)
    sp = sub.add_parser("oos"); common(sp)
    sp.add_argument("--again", action="store_true", help="allow re-using an already-touched holdout")
    sp.set_defaults(fn=cmd_oos)
    sp = sub.add_parser("report"); common(sp); sp.add_argument("--out", default="reports/latest")
    sp.set_defaults(fn=cmd_report)
    sp = sub.add_parser("paper"); common(sp, data=False)
    sp.add_argument("--replay", help="CSV to replay bar-by-bar (omit for Binance live data)")
    sp.add_argument("--warmup-bars", type=int, default=0)
    sp.add_argument("--max-bars", type=int, default=0)
    sp.add_argument("--delay", type=float, default=0.0)
    sp.add_argument("--log-dir", default=None)
    sp.add_argument("--live", action="store_true", help="refused: live order routing is not implemented")
    sp.set_defaults(fn=cmd_paper)
    sp = sub.add_parser("download-binance")
    sp.add_argument("--symbol", default="BTCUSDT"); sp.add_argument("--timeframe", default="5min")
    sp.add_argument("--start", default="2022-01"); sp.add_argument("--end", default=None)
    sp.add_argument("--out", default="data/raw/BTCUSDT_5m.csv")
    sp.set_defaults(fn=cmd_download)
    sp = sub.add_parser("synth")
    sp.add_argument("--days", type=int, default=800); sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--timeframe", default="5min"); sp.add_argument("--out", default="data/raw/synthetic_5m.csv")
    sp.set_defaults(fn=cmd_synth)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
