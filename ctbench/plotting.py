"""Figures for comparison and ablation tables (matplotlib + seaborn).

Deliberately minimal and headless (``Agg`` backend): horizontal bars for
per-method comparison and grouped bars for the joint-vs-single ablation. Ours is
highlighted so the reader's eye lands on the contrast that matters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # backend must be set before pyplot

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

_OURS = "#c0392b"
_OTHER = "#8a8a8a"


def bar_comparison(
    metrics: pd.DataFrame,
    value_col: str,
    out_path: Path,
    *,
    ascending: bool = True,
) -> Path:
    """Horizontal bar chart of ``value_col`` per method; 'OURS'/'joint' highlighted."""
    s = metrics[value_col].sort_values(ascending=ascending)
    colors = [_OURS if _is_ours(m) else _OTHER for m in s.index]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(s) + 1))
    ax.barh([str(m) for m in s.index], s.to_numpy(), color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(value_col)
    for i, v in enumerate(s.to_numpy()):
        ax.text(v, i, f" {v:.3g}", va="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def ablation_bars(metrics: pd.DataFrame, value_cols: list[str], out_path: Path) -> Path:
    """Grouped bars: one group per metric, one bar per variant (joint highlighted)."""
    variants = list(metrics.index)
    fig, axes = plt.subplots(
        1,
        len(value_cols),
        figsize=(3.2 * len(value_cols), 3.4),
        squeeze=False,
    )
    for ax, col in zip(axes[0], value_cols, strict=True):
        vals = metrics[col]
        colors = [_OURS if _is_ours(v) else _OTHER for v in variants]
        ax.bar([str(v) for v in variants], vals.to_numpy(), color=colors)
        ax.set_title(col, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _is_ours(name: object) -> bool:
    s = str(name).lower()
    return "our" in s or s == "joint"
