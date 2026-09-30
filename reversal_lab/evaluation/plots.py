"""Matplotlib figures for the report (PNG)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed categorical order (colour follows the setup, never its rank).
SETUP_COLORS = {"pdh_pdl": "#2a78d6", "session_extreme": "#eb6834", "vwap": "#1baf7a", "ma": "#eda100"}
COMBINED = "#0b0b0b"
TEXT2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {"pdh_pdl": "A+ PDH/PDL", "session_extreme": "Session high/low", "vwap": "VWAP", "ma": "200 MA",
          "combined": "Combined"}


def _style(ax, title: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=9)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def equity_curves(trades: pd.DataFrame, path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=120, facecolor=SURFACE)
    _style(ax, title, "Cumulative R (net of costs)")
    ax.axhline(0, color=TEXT2, linewidth=0.8)
    if len(trades):
        t = trades.sort_values("exit_time")
        for s, color in SETUP_COLORS.items():
            sub = t[t["setup"] == s]
            if len(sub):
                ax.plot(sub["exit_time"], sub["R"].cumsum(), color=color, linewidth=1.6,
                        label=f"{LABELS[s]} (n={len(sub)})")
        ax.plot(t["exit_time"], t["R"].cumsum(), color=COMBINED, linewidth=2.2,
                label=f"{LABELS['combined']} (n={len(t)})")
        ax.legend(frameon=False, fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "No trades", transform=ax.transAxes, ha="center", color=TEXT2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def baseline_hist(dist: np.ndarray, actual: float, p: float, path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=120, facecolor=SURFACE)
    _style(ax, title, "Random strategies")
    if len(dist):
        ax.hist(dist, bins=40, color="#2a78d6", alpha=0.85, edgecolor=SURFACE, linewidth=0.8)
        ax.axvline(actual, color="#e34948", linewidth=2)
        ax.text(actual, ax.get_ylim()[1] * 0.92, f"  strategy {actual:+.3f}R\n  p = {p:.3f}",
                color="#0b0b0b", fontsize=8, va="top",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9, "pad": 2})
    ax.set_xlabel("Mean R per trade", color=TEXT2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
