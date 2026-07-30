"""Evaluation-set builders.

* ``casp16`` — the CASP16 pharma-ligand complexes (protein + ligand), prepared by
  ``scripts/prepare_casp.py``. These experimental targets are out of every
  model's training data, so reconstruction here is a clean held-out test. The
  own model reconstructs the pocket + ligand; ESM3 / FoldToken reconstruct the
  protein backbone (full chain, or the own model's pocket — see the runner's
  ``protein_scope``).
* ``pdb_folder`` — any directory of protein PDBs (protein-backbone-only
  comparison of ESM3 vs FoldToken on structures you drop in).
"""

from __future__ import annotations

import json
from pathlib import Path

from recon_bench import paths
from recon_bench.types import Sample

CASP_INDEX = paths.DATA_DIR / "casp16" / "index.json"


def casp16(
    index_path: str | Path = CASP_INDEX,
    limit: int | None = None,
    targets: list[str] | None = None,
) -> list[Sample]:
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"CASP index missing: {index_path}. Run scripts/prepare_casp.py first."
        )
    records = json.loads(index_path.read_text())
    if targets:
        wanted = set(targets)
        records = [r for r in records if r["target"] in wanted]
    if limit is not None:
        records = records[:limit]
    return [
        Sample(
            sample_id=r["sample_id"],
            protein_pdb=Path(r["protein_pdb"]),
            ligand_sdf=Path(r["ligand_sdf"]),
            meta={"target": r["target"], "ligand_pdb": r.get("ligand_pdb")},
        )
        for r in records
    ]


def pdb_folder(pdb_dir: str | Path, limit: int | None = None) -> list[Sample]:
    pdb_dir = Path(pdb_dir)
    pdbs = sorted([*pdb_dir.glob("*.pdb"), *pdb_dir.glob("*.cif")])
    if limit is not None:
        pdbs = pdbs[:limit]
    return [Sample(sample_id=p.stem, protein_pdb=p) for p in pdbs]


def build(name: str, **kwargs) -> list[Sample]:
    if name == "casp16":
        return casp16(
            index_path=kwargs.get("index_path", CASP_INDEX),
            limit=kwargs.get("limit"),
            targets=kwargs.get("targets"),
        )
    if name in ("pdb-folder", "pdb_folder"):
        return pdb_folder(kwargs["pdb_dir"], kwargs.get("limit"))
    raise KeyError(f"unknown dataset {name!r}")
