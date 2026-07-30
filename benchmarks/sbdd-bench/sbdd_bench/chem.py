"""Per-molecule chemical-validity and drug-likeness metrics (category 1).

Pure RDKit, computed on the *topology* of the sanitized molecule (the ``mol``
field of :class:`sbdd_bench.molio.GenMol`). These are the 2D-graph-flavoured
metrics; treat them as supporting evidence, not as the headline for a
3D target-aware model (cf. the note in the benchmark README).
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache


# RDKit's SA scorer lives in the contrib tree, not the top-level package.
@lru_cache(maxsize=1)
def _sascorer():
    from rdkit.Chem import RDConfig

    sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_dir not in sys.path:
        sys.path.append(sa_dir)
    import sascorer  # type: ignore

    return sascorer


@lru_cache(maxsize=1)
def _pains_catalog():
    from rdkit.Chem import FilterCatalog

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)


def sa_score(mol) -> float | None:
    """Synthetic accessibility (Ertl & Schuffenhauer), 1 (easy) – 10 (hard)."""
    try:
        return float(_sascorer().calculateScore(mol))
    except Exception:  # noqa: BLE001
        return None


def qed(mol) -> float | None:
    from rdkit.Chem import QED

    try:
        return float(QED.qed(mol))
    except Exception:  # noqa: BLE001
        return None


def lipinski_pass(mol) -> bool | None:
    """Lipinski Ro5: MW≤500, logP≤5, HBD≤5, HBA≤10 (≤1 violation allowed)."""
    from rdkit.Chem import Crippen, Descriptors, Lipinski

    try:
        violations = sum(
            [
                Descriptors.MolWt(mol) > 500,
                Crippen.MolLogP(mol) > 5,
                Lipinski.NumHDonors(mol) > 5,
                Lipinski.NumHAcceptors(mol) > 10,
            ]
        )
        return violations <= 1
    except Exception:  # noqa: BLE001
        return None


def veber_pass(mol) -> bool | None:
    """Veber oral-bioavailability rule: rotatable bonds ≤10 and TPSA ≤140."""
    from rdkit.Chem import Descriptors, Lipinski

    try:
        return (
            Lipinski.NumRotatableBonds(mol) <= 10
            and Descriptors.TPSA(mol) <= 140
        )
    except Exception:  # noqa: BLE001
        return None


def pains_free(mol) -> bool | None:
    try:
        return not _pains_catalog().HasMatch(mol)
    except Exception:  # noqa: BLE001
        return None


def n_fragments(mol) -> int:
    from rdkit import Chem

    return len(Chem.GetMolFrags(mol))


def molecule_metrics(mol) -> dict:
    """Full per-molecule chemistry metric set. ``mol`` may be ``None`` (invalid).

    ``valid`` is the single most important field: True iff the molecule passed
    RDKit sanitization at load time. Everything else is only meaningful when
    ``valid`` is True.
    """
    from rdkit.Chem import Descriptors

    if mol is None:
        return {
            "valid": False, "connected": None, "n_atoms": None, "mol_wt": None,
            "qed": None, "sa": None, "logp": None,
            "lipinski": None, "veber": None, "pains_free": None,
        }
    nfrag = n_fragments(mol)
    return {
        "valid": True,
        "connected": nfrag == 1,
        "n_atoms": mol.GetNumHeavyAtoms(),
        "mol_wt": float(Descriptors.MolWt(mol)),
        "qed": qed(mol),
        "sa": sa_score(mol),
        "logp": float(Descriptors.MolLogP(mol)),
        "lipinski": lipinski_pass(mol),
        "veber": veber_pass(mol),
        "pains_free": pains_free(mol),
    }
