"""Experiment ledger: an append-only record of every parameter set ever evaluated.

Why it matters: if you try 200 parameter combinations, the best one looks
good partly by luck, just as in a study that tests 200 outcomes and reports
the one with p < 0.05. The ledger keeps an honest count of how many
combinations were tried on each dataset and how many times the
out-of-sample (holdout) data has been looked at.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd


def data_fingerprint(df: pd.DataFrame) -> str:
    """Short hash identifying a dataset (range, size and a checksum of closes)."""
    blob = f"{df.index[0]}|{df.index[-1]}|{len(df)}|{float(df['close'].sum()):.6f}"
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(self, **entry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **entry}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def combos_tried(self, data_fp: str) -> int:
        """Distinct strategy configurations evaluated on this dataset's tuning data."""
        seen = {e["strategy_hash"] for e in self.entries()
                if e.get("data_fp") == data_fp and e.get("segment") in ("in_sample", "wf_train")}
        return len(seen)

    def oos_touches(self, data_fp: str) -> int:
        return sum(1 for e in self.entries() if e.get("data_fp") == data_fp and e.get("segment") == "oos")
