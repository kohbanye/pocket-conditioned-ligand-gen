"""The per-complex dump schema, and validated readers and writers for it.

Every method -- ours and each baseline -- writes one row per CASF core complex
in this schema, so the metric layer never has to know which model produced a
number. Column names match the source repo's own CSVs so historical dumps can
be dropped into ``results/`` and read unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

#: pdbid, the measured pK, the CASF cluster, the MLM pseudo-log-likelihood
#: (recorded as a zero-shot reference point, not used by the tables), and the
#: head's prediction.
AFFINITY_COLUMNS = ("pdbid", "logka", "cluster", "pll", "head")
_REQUIRED = ("pdbid", "logka")


def _require(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        msg = f"{path}: missing required columns {missing}; have {list(df.columns)}"
        raise ValueError(msg)


def write_affinity(df: pd.DataFrame, path: Path) -> None:
    """Write a per-complex dump in the canonical column order."""
    _require(df, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in AFFINITY_COLUMNS if c in df.columns]
    df.to_csv(path, columns=cols, index=False)
