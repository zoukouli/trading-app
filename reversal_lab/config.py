"""Configuration: defaults, YAML loading, validation and dotted-key overrides.

Every threshold the strategy uses lives here. ``config.yaml`` at the repo root
mirrors these defaults with comments; a test keeps the two in sync.

Unknown keys raise an error on purpose: a typo such as ``max_bars_apar`` would
otherwise be silently ignored and you would be testing a different strategy
from the one you think you are testing.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

# A "spec" is {mode, value}; see ``spec_to_price``.
SPEC_MODES = ("ticks", "atr", "pct", "price")

DEFAULTS: dict[str, Any] = {
    "instrument": {
        "symbol": "BTCUSDT",
        "tick_size": 0.01,
        "point_value": 1.0,
    },
    "data": {
        "source": "csv",  # csv | synthetic
        "path": "data/raw/BTCUSDT_5m.csv",
        "timezone": "UTC",  # timezone of naive timestamps in the file
        "start": None,
        "end": None,
        "synthetic": {
            "days": 800,
            "seed": 7,
            "start_price": 30000.0,
            "annual_vol": 0.6,
        },
    },
    "timeframe": "5min",
    "session": {
        "timezone": "UTC",
        "start": "00:00",
        "entry_start": "00:00",
        "entry_end": "23:30",
        "min_bars_into_session": 6,
        "min_prev_session_coverage": 0.9,
        "flatten_at_session_end": True,
    },
    "indicators": {
        "atr_period": 14,
        "swing": {"left": 3, "right": 3},
        "ma": {"type": "sma", "period": 200, "timeframe": "5min"},
        "vwap": {"price": "typical"},
    },
    "setups": {
        "enabled": ["pdh_pdl", "session_extreme", "vwap", "ma"],
        "priority": ["pdh_pdl", "session_extreme", "vwap", "ma"],
        "near_tolerance": {"mode": "atr", "value": 0.5},
        "double": {
            "peak_tolerance": {"mode": "atr", "value": 0.25},
            "min_bars_apart": 5,
            "max_bars_apart": 60,
            "min_pullback": {"mode": "atr", "value": 1.0},
            "max_candidates": 5,
        },
        "reversal": {
            "pattern": "rejection",  # rejection | double
            "approach_bars": 12,
            "approach_min_frac": 0.8,
            "require_close_back": True,
        },
    },
    "entry": {
        "trigger": "neckline_break",  # neckline_break | close_back_through_level | confirmation
        "trigger_offset": {"mode": "ticks", "value": 0},
        "max_bars_to_trigger": 12,
    },
    "exits": {
        "stop_buffer": {"mode": "atr", "value": 0.1},
        "min_stop": {"mode": "atr", "value": 0.5},
        "max_stop": {"mode": "atr", "value": 4.0},
        "target": {"mode": "r_multiple", "r_multiple": 2.0},  # r_multiple | vwap | opposite_level
        "min_reward_risk": 1.0,
        "target_fill": "through",  # through | touch
        "intrabar_priority": "stop_first",  # stop_first | target_first
        "max_hold_bars": None,
    },
    "risk": {
        "max_trades_per_day": 3,
    },
    "costs": {
        "slippage": {"mode": "pct", "value": 0.0001},
        "commission": {"mode": "pct", "value": 0.0005},
    },
    "evaluation": {
        "in_sample_fraction": 0.6,
        "in_sample_end": None,
        "min_trades_for_evidence": 100,
        "ledger_path": "experiments/ledger.jsonl",
        "baseline": {"reps": 1000, "seed": 42},
        "bootstrap_reps": 5000,
        "objective": "expectancy",
        "min_trades_for_selection": 30,
        "walk_forward": {
            "span": "in_sample",  # in_sample | all
            "train_days": 180,
            "test_days": 60,
            "step_days": 60,
        },
        "grid": {
            "setups.double.peak_tolerance.value": [0.15, 0.25, 0.4],
            "entry.trigger": ["neckline_break", "close_back_through_level"],
            "exits.target.r_multiple": [1.5, 2.0, 3.0],
        },
    },
    "paper": {
        "feed": "replay",  # replay | binance
        "log_dir": "paper_logs",
        "buffer_bars": 5000,
        "poll_seconds": 20,
    },
}

# Keys whose value is a free-form mapping (not validated key-by-key).
_FREEFORM = {("evaluation", "grid")}

SETUP_NAMES = ("pdh_pdl", "session_extreme", "vwap", "ma")


class ConfigError(ValueError):
    pass


def _merge(base: dict, override: dict, path: tuple = ()) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        here = path + (key,)
        if key not in base:
            raise ConfigError(f"Unknown config key: {'.'.join(here)}")
        if isinstance(base[key], dict) and here not in _FREEFORM:
            if not isinstance(val, dict):
                raise ConfigError(f"Config key {'.'.join(here)} must be a mapping")
            out[key] = _merge(base[key], val, here)
        else:
            out[key] = copy.deepcopy(val)
    return out


def validate(cfg: dict) -> dict:
    """Sanity-check values that would otherwise fail deep inside a backtest."""
    def check_spec(path: str) -> None:
        spec = get(cfg, path)
        if spec["mode"] not in SPEC_MODES:
            raise ConfigError(f"{path}.mode must be one of {SPEC_MODES}")
        if spec["value"] < 0:
            raise ConfigError(f"{path}.value must be >= 0")

    for p in (
        "setups.near_tolerance", "setups.double.peak_tolerance", "setups.double.min_pullback",
        "entry.trigger_offset", "exits.stop_buffer", "exits.min_stop", "exits.max_stop",
        "costs.slippage", "costs.commission",
    ):
        check_spec(p)
    for s in cfg["setups"]["enabled"] + cfg["setups"]["priority"]:
        if s not in SETUP_NAMES:
            raise ConfigError(f"Unknown setup '{s}'; choose from {SETUP_NAMES}")
    if cfg["entry"]["trigger"] not in ("neckline_break", "close_back_through_level", "confirmation"):
        raise ConfigError("entry.trigger invalid")
    if cfg["exits"]["target"]["mode"] not in ("r_multiple", "vwap", "opposite_level"):
        raise ConfigError("exits.target.mode invalid")
    if cfg["setups"]["reversal"]["pattern"] not in ("rejection", "double"):
        raise ConfigError("setups.reversal.pattern invalid")
    if cfg["indicators"]["ma"]["type"] not in ("sma", "ema"):
        raise ConfigError("indicators.ma.type must be sma or ema")
    d = cfg["setups"]["double"]
    if d["min_bars_apart"] > d["max_bars_apart"]:
        raise ConfigError("setups.double.min_bars_apart > max_bars_apart")
    if cfg["exits"]["intrabar_priority"] not in ("stop_first", "target_first"):
        raise ConfigError("exits.intrabar_priority invalid")
    if cfg["exits"]["target_fill"] not in ("through", "touch"):
        raise ConfigError("exits.target_fill invalid")
    if cfg["costs"]["commission"]["mode"] not in ("pct", "price"):
        raise ConfigError("costs.commission.mode must be pct or price")
    if cfg["costs"]["slippage"]["mode"] not in ("pct", "price", "ticks"):
        raise ConfigError("costs.slippage.mode must be pct, price or ticks")
    return cfg


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path) as fh:
            user = yaml.safe_load(fh) or {}
        cfg = _merge(cfg, user)
    if overrides:
        cfg = with_overrides(cfg, overrides)
    return validate(cfg)


def get(cfg: dict, dotted: str) -> Any:
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def with_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """Return a copy of ``cfg`` with dotted keys replaced, e.g. {"exits.target.r_multiple": 3}."""
    out = copy.deepcopy(cfg)
    for dotted, val in overrides.items():
        parts = dotted.split(".")
        node = out
        for part in parts[:-1]:
            if part not in node:
                raise ConfigError(f"Unknown config key: {dotted}")
            node = node[part]
        if parts[-1] not in node:
            raise ConfigError(f"Unknown config key: {dotted}")
        node[parts[-1]] = copy.deepcopy(val)
    return validate(out)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def strategy_hash(cfg: dict) -> str:
    """Hash of everything that changes which trades are taken (excludes evaluation/paper/data)."""
    keep = {k: cfg[k] for k in ("instrument", "timeframe", "session", "indicators", "setups",
                                "entry", "exits", "risk", "costs")}
    return config_hash(keep)


def spec_to_price(spec: dict, atr: float, price: float, tick: float) -> float:
    """Convert a {mode, value} tolerance spec into a price distance."""
    mode, value = spec["mode"], float(spec["value"])
    if mode == "ticks":
        return value * tick
    if mode == "atr":
        return value * atr
    if mode == "pct":
        return value * price
    if mode == "price":
        return value
    raise ConfigError(f"Unknown spec mode {mode}")
