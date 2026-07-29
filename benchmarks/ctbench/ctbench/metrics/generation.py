"""Generation metrics: molecule-level aggregation + light RDKit quality/diversity.

The *authoritative* generation metrics (Vina docking, PoseBusters validity, SA,
the composite hit-rate) are computed by the sbdd-bench harness on the generated
SDFs — see :mod:`ctbench.baselines.sbdd_gen` — and this repo consumes its
per-molecule dump. The functions here are (a) a straightforward molecule-level
aggregation for quick diagnostics on a per-molecule frame, and (b) RDKit-only
validity/uniqueness/diversity that need no docking.

NOTE: :func:`aggregate_molecules` is molecule-pooled; the paper's per-model
numbers are per-target-then-averaged with metric-specific subsets, so treat this
as a diagnostic, not a bit-exact reproduction (use the seeded per_model.csv for
headline numbers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

_MORGAN_RADIUS = 2
_MORGAN_BITS = 2048


def aggregate_molecules(
    df: pd.DataFrame,
    by: Sequence[str] = ("model",),
) -> pd.DataFrame:
    """Molecule-pooled aggregation of a per-molecule dump (diagnostic)."""
    by = list(by)

    def _agg(g: pd.DataFrame) -> pd.Series:
        valid = (
            g["valid"].astype(bool)
            if "valid" in g
            else pd.Series(data=True, index=g.index)
        )
        out = {
            "n": len(g),
            "validity": float(valid.mean()),
            "vina_score_mean": float(g["vina_score"].dropna().mean())
            if "vina_score" in g
            else float("nan"),
            "vina_min_mean": float(g["vina_min"].dropna().mean())
            if "vina_min" in g
            else float("nan"),
            "qed_mean": float(g.loc[valid, "qed"].mean())
            if "qed" in g
            else float("nan"),
            "sa_mean": float(g.loc[valid, "sa"].mean()) if "sa" in g else float("nan"),
            "pb_valid_rate": float(g["pb_valid"].astype(bool).mean())
            if "pb_valid" in g
            else float("nan"),
            "clash_free_rate": float((g["clash_count"] == 0).mean())
            if "clash_count" in g
            else float("nan"),
        }
        if "smiles" in g:
            smis = [s for s in g.loc[valid, "smiles"].tolist() if isinstance(s, str)]
            out["uniqueness"] = uniqueness(smis)
            out["scaffold_diversity"] = scaffold_diversity(smis)
        return pd.Series(out)

    return df.groupby(by, group_keys=True).apply(_agg, include_groups=False)


def uniqueness(smiles: Sequence[str]) -> float:
    """Fraction of distinct canonical SMILES among the (valid) molecules."""
    if not smiles:
        return float("nan")
    return len(set(smiles)) / len(smiles)


def scaffold_diversity(smiles: Sequence[str]) -> float:
    """Fraction of distinct Bemis-Murcko scaffolds among the molecules."""
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: PLC0415

    scaffolds = set()
    n = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n += 1
        scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
    return len(scaffolds) / n if n else float("nan")


def validity_from_smiles(smiles: Sequence[str]) -> float:
    """Fraction of SMILES that RDKit can parse into a molecule."""
    from rdkit import Chem  # noqa: PLC0415

    if not smiles:
        return float("nan")
    ok = sum(1 for s in smiles if Chem.MolFromSmiles(s) is not None)
    return ok / len(smiles)


def internal_diversity(smiles: Sequence[str]) -> float:
    """Mean pairwise (1 - Tanimoto) over Morgan fingerprints of the molecules."""
    from rdkit import Chem, DataStructs  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    fps = []
    make_gen = AllChem.GetMorganGenerator  # ty: ignore[unresolved-attribute]  # rdkit C-ext
    gen = make_gen(radius=_MORGAN_RADIUS, fpSize=_MORGAN_BITS)
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(gen.GetFingerprint(mol))
    if len(fps) < 2:  # noqa: PLR2004
        return float("nan")
    sims = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :]))
    return float(1.0 - np.mean(sims)) if sims else float("nan")
