"""Evaluation-set builders.

* ``posebusters`` — the PoseBusters Benchmark set, prepared by
  ``scripts/prepare_posebusters.py``. 428 complexes over 426 distinct proteins
  and 427 distinct ligands, all released to the PDB in 2021 or later, so none
  of them can be in PDBbind v2020 or CrossDocked2020 — which is what the
  all-atom VQ-VAE trains on. This is the default reconstruction set.
* ``astex`` — the Astex Diverse set (85 complexes, PDB entries up to 2007),
  bundled in the same archive. Its complexes *are* in PDBbind/CrossDocked, so
  it reads as a seen-data contrast to ``posebusters`` rather than a held-out
  test.
* ``casp16`` — the CASP16 pharma-ligand complexes, prepared by
  ``scripts/prepare_casp.py``. Held out from every model's training data, but
  its 303 complexes cover only **four** proteins (autotaxin 227, Mpro 57,
  chymase 17, cathepsin G 2) because CASP's pharma-ligand category is a
  super-target design. Read it as "one pocket, 267 ligands", never as n=303:
  always group by ``target`` prefix before aggregating.
* ``pdb_folder`` — any directory of protein PDBs (protein-backbone-only
  comparison of ESM3 vs FoldToken on structures you drop in).

The own model reconstructs the pocket + ligand; ESM3 / FoldToken reconstruct
the protein backbone (full chain, or the own model's pocket — see the runner's
``protein_scope``).
"""

from __future__ import annotations

import json
from pathlib import Path

from recon_bench import paths
from recon_bench.types import Sample

CASP_INDEX = paths.DATA_DIR / "casp16" / "index.json"
POSEBUSTERS_INDEX = paths.DATA_DIR / "posebusters" / "posebusters_benchmark_set_index.json"
ASTEX_INDEX = paths.DATA_DIR / "posebusters" / "astex_diverse_set_index.json"

#: Record keys that are metadata rather than the sample itself.
_META_KEYS = ("target", "ligand_pdb", "ccd")


def _from_index(
    index_path: str | Path,
    prepare_hint: str,
    limit: int | None = None,
    targets: list[str] | None = None,
) -> list[Sample]:
    """Load samples from a ``{sample_id, protein_pdb, ligand_sdf, ...}`` index.

    Every set here is a directory of protein PDBs plus ligand SDFs, so they
    differ only in the index they read and what to say when it is missing.
    """
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"index missing: {index_path}. Run {prepare_hint} first.")
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
            meta={k: r[k] for k in _META_KEYS if k in r},
        )
        for r in records
    ]


def posebusters(
    index_path: str | Path = POSEBUSTERS_INDEX,
    limit: int | None = None,
    targets: list[str] | None = None,
) -> list[Sample]:
    return _from_index(index_path, "scripts/prepare_posebusters.py", limit, targets)


def astex(
    index_path: str | Path = ASTEX_INDEX,
    limit: int | None = None,
    targets: list[str] | None = None,
) -> list[Sample]:
    return _from_index(index_path, "scripts/prepare_posebusters.py", limit, targets)


def casp16(
    index_path: str | Path = CASP_INDEX,
    limit: int | None = None,
    targets: list[str] | None = None,
) -> list[Sample]:
    return _from_index(index_path, "scripts/prepare_casp.py", limit, targets)


def pdb_folder(pdb_dir: str | Path, limit: int | None = None) -> list[Sample]:
    pdb_dir = Path(pdb_dir)
    pdbs = sorted([*pdb_dir.glob("*.pdb"), *pdb_dir.glob("*.cif")])
    if limit is not None:
        pdbs = pdbs[:limit]
    return [Sample(sample_id=p.stem, protein_pdb=p) for p in pdbs]


_INDEXED = {
    "posebusters": (posebusters, POSEBUSTERS_INDEX),
    "astex": (astex, ASTEX_INDEX),
    "casp16": (casp16, CASP_INDEX),
}


def build(name: str, **kwargs) -> list[Sample]:
    if name in _INDEXED:
        builder, default_index = _INDEXED[name]
        return builder(
            index_path=kwargs.get("index_path", default_index),
            limit=kwargs.get("limit"),
            targets=kwargs.get("targets"),
        )
    if name in ("pdb-folder", "pdb_folder"):
        return pdb_folder(kwargs["pdb_dir"], kwargs.get("limit"))
    raise KeyError(f"unknown dataset {name!r}")
