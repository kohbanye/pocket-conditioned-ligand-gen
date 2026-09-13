"""Baseline affinity predictions: collect existing dumps, or regenerate them.

The collectors parse the sibling repos' per-complex outputs into this bench's
schema. The rerun wrappers regenerate them under the same protocol, each in the
backend's own environment -- GenScore pins a DGL build that cannot share this
one, so it is driven as a subprocess rather than imported.

All three baselines are scored on the SAME crystal poses as our head. That is
what makes the comparison a comparison; it is also why the reproduced numbers
differ slightly from each paper's own, which each used its own inputs.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


def collect_genscore(baselines_repo: Path) -> pd.DataFrame:
    """GenScore's scoring-power CSV -> the canonical schema."""
    df = pd.read_csv(baselines_repo / "casf_work" / "scoring_power_genscore.csv")
    return df.rename(columns={"score": "head"})[["pdbid", "logka", "cluster", "head"]]


def collect_vina(source_repo: Path) -> pd.DataFrame:
    """Vina's per-complex score -> the canonical schema.

    ``vina_score`` is already oriented so that higher means stronger binding in
    this dump, matching the sign of a pK.
    """
    df = pd.read_csv(source_repo / "outputs" / "casf" / "vina_scoring.csv")
    return df.rename(columns={"vina_score": "head"})[
        ["pdbid", "logka", "cluster", "head"]
    ]


def collect_boltz2(
    source_repo: Path,
    labels: dict[str, tuple[float, str]] | None = None,
) -> pd.DataFrame:
    """Boltz-2's CASF affinity predictions -> the canonical schema.

    Boltz-2 runs in its native mode: it is given the protein sequence and the
    ligand SMILES and predicts a structure and an affinity, so unlike every
    other method here it never sees the crystal pose. That makes it the least
    like-for-like column in the table, and the most interesting one -- a
    lightweight head reading a crystal pose against a large structure predictor
    reading a sequence.

    The model reports ``affinity_pred_value`` as log10(IC50 in uM), so
    ``pK = 6 - value``. One JSON per complex, under the run tree rather than in
    a single CSV, because the predictions come from an array job.
    """
    from affinity_bench.inference import load_coreset_labels  # noqa: PLC0415

    pred_root = source_repo / "outputs" / "boltz_casf" / "predict"
    if not pred_root.exists():
        msg = f"no Boltz-2 predictions under {pred_root}"
        raise FileNotFoundError(msg)
    if labels is None:
        labels = load_coreset_labels(
            source_repo / "data" / "casf2016" / "power_scoring" / "CoreSet.dat",
        )
    rows = []
    for path in sorted(pred_root.glob("boltz_results_*/predictions/*/affinity_*.json")):
        pdbid = path.stem.replace("affinity_", "").lower()
        if pdbid not in labels:
            continue
        try:
            value = json.loads(path.read_text())["affinity_pred_value"]
        except (KeyError, json.JSONDecodeError):
            continue
        logka, cluster = labels[pdbid]
        rows.append(
            {
                "pdbid": pdbid,
                "logka": logka,
                "cluster": cluster,
                "head": 6.0 - float(value),
            },
        )
    return pd.DataFrame(rows, columns=["pdbid", "logka", "cluster", "head"])  # ty: ignore[invalid-argument-type]  # pandas stub: list[str] columns


def rerun_genscore(baselines_repo: Path, extra_args: list[str] | None = None) -> None:
    """Re-run GenScore's scoring power in its own micromamba environment."""
    cmd = [
        "micromamba",
        "run",
        "-p",
        str(baselines_repo / "envs" / "genscore"),
        "python",
        str(baselines_repo / "run_casf_scoring.py"),
        "--backend",
        "genscore",
        *(extra_args or []),
    ]
    subprocess.run(cmd, check=True, cwd=str(baselines_repo))  # noqa: S603
