"""Diversity / novelty metrics (category 5), computed over a *set* of valid
molecules (per target, or pooled).

Caveat baked into the API: these are sensitive to library size. A model that
emits 100× more molecules and reports best-k looks artificially diverse/novel.
The runner therefore always records the set size alongside these ratios, and
comparisons should hold the number of generated molecules per pocket fixed.
"""

from __future__ import annotations

import numpy as np


def _morgan_fps(smiles_list, radius: int = 2, n_bits: int = 2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps, smis = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else smi
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
        smis.append(smi)
    return fps, smis


def uniqueness(smiles_list) -> float | None:
    smis = [s for s in smiles_list if s]
    if not smis:
        return None
    return len(set(smis)) / len(smis)


def internal_diversity(smiles_list) -> float | None:
    """1 − mean pairwise Tanimoto over Morgan fingerprints (Polykovskiy MOSES)."""
    from rdkit import DataStructs

    fps, _ = _morgan_fps(smiles_list)
    if len(fps) < 2:
        return None
    sims = []
    for i in range(len(fps)):
        row = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        sims.extend(row)
    return float(1.0 - np.mean(sims)) if sims else None


def novelty(smiles_list, train_smiles: set[str] | None) -> float | None:
    """Fraction of (canonical) molecules not present in the training set."""
    if not train_smiles:
        return None
    from rdkit import Chem

    canon, n = 0, 0
    for smi in smiles_list:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n += 1
        if Chem.MolToSmiles(mol) not in train_smiles:
            canon += 1
    return canon / n if n else None


def bemis_murcko_scaffold(smi: str) -> str | None:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:  # noqa: BLE001
        return None


def scaffold_diversity(smiles_list) -> float | None:
    """Unique Bemis–Murcko scaffolds / number of valid molecules."""
    scaffolds, n = set(), 0
    for smi in smiles_list:
        s = bemis_murcko_scaffold(smi) if smi else None
        if s is None:
            continue
        n += 1
        scaffolds.add(s)
    return len(scaffolds) / n if n else None


def load_train_smiles(path) -> set[str]:
    """Load a canonical-SMILES set (one per line) for novelty scoring."""
    from rdkit import Chem

    out: set[str] = set()
    for line in open(path):
        smi = line.strip().split()[0] if line.strip() else ""
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        out.add(Chem.MolToSmiles(mol) if mol is not None else smi)
    return out


def diversity_metrics(smiles_list, train_smiles: set[str] | None = None) -> dict:
    """Set-level diversity metrics + the set size they depend on."""
    valid = [s for s in smiles_list if s]
    return {
        "n_valid": len(valid),
        "uniqueness": uniqueness(valid),
        "internal_diversity": internal_diversity(valid),
        "scaffold_diversity": scaffold_diversity(valid),
        "novelty": novelty(valid, train_smiles),
    }
