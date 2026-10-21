"""Prepare the PoseBusters Benchmark set (and the bundled Astex Diverse set).

CASP16 is 303 complexes over **four** proteins -- autotaxin carries 75% of them,
Mpro another 19% -- because CASP's pharma-ligand category is a *super-target*
design: one protein, many ligands. That is the right shape for scoring pose
prediction and the wrong shape for measuring how a tokenizer reconstructs
protein geometry, and it makes the n=303 paired tests in
``docs/results/casp16_significance.md`` an n=4 claim wearing a large number.

The PoseBusters Benchmark set is the field's standard answer: 428 complexes,
426 distinct proteins, 427 distinct ligands, PDB releases from 2021 on, so it
cannot overlap PDBbind v2020 or CrossDocked2020 -- which is what the all-atom
VQ-VAE trains on (``--source-types cdonly``). Drug-like ligands only; solvents
and ions were dropped when the set was curated.

The same archive ships the **Astex Diverse set** (85 complexes, PDB entries up
to 2007). Those *are* inside PDBbind/CrossDocked, which is exactly what makes
them useful here: a seen-data contrast arm for the same tokenizer.

Ligands arrive as SDF with correct bond orders, so unlike ``prepare_casp.py``
there is no PDB->SDF conversion step and no OpenBabel dependency.

    uv run python scripts/prepare_posebusters.py

Cofactors, glycerol, ethylene glycol and DMSO remain as ``HETATM`` in the
protein PDBs. They are not a problem: ``precompute_pocket_atom_candidates``
keeps only residues in ``AA_3TO1`` that have a CA, so none of them can enter a
pocket as a protein atom. They *do* count towards the clash metrics, which is
what they should do -- they are part of the wall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recon_bench import paths  # noqa: E402

#: Zenodo record 8278563, the archive published with the PoseBusters paper.
ARCHIVE_URL = "https://zenodo.org/records/8278563/files/posebusters_paper_data.zip"
ARCHIVE_SHA256 = "495a8f432ee5612c0dfa3cc582829f112bfca3c29dddc2db2c3a8dc7609e721c"

#: Set name -> number of complexes it must contain, so a truncated download or a
#: reshuffled archive fails here instead of silently shrinking the benchmark.
EXPECTED_SIZE = {"posebusters_benchmark_set": 428, "astex_diverse_set": 85}


def download(archive: Path) -> None:
    """Fetch the archive unless it is already present and intact."""
    if archive.exists() and _sha256(archive) == ARCHIVE_SHA256:
        print(f"[pb] archive already present -> {archive}")
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pb] downloading {ARCHIVE_URL}")
    urllib.request.urlretrieve(ARCHIVE_URL, archive)  # noqa: S310
    digest = _sha256(archive)
    if digest != ARCHIVE_SHA256:
        msg = f"checksum mismatch for {archive}: {digest} != {ARCHIVE_SHA256}"
        raise SystemExit(msg)
    print(f"[pb] downloaded {archive.stat().st_size / 1e6:.1f} MB, checksum ok")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_dir)  # noqa: S202 - Zenodo archive, checksum verified
    print(f"[pb] extracted -> {out_dir}")


def build_index(set_dir: Path) -> list[dict]:
    """Index one set's ``{PDB}_{CCD}/`` folders.

    ``sample_id`` is the folder name (``5S8I_2LY``) and ``target`` is the PDB
    id, so the runner can group by target the way it does for CASP -- here that
    grouping is almost the identity, which is the point.
    """
    records: list[dict] = []
    missing: list[str] = []
    for entry in sorted(p for p in set_dir.iterdir() if p.is_dir()):
        protein = entry / f"{entry.name}_protein.pdb"
        ligand = entry / f"{entry.name}_ligand.sdf"
        if not (protein.exists() and ligand.exists()):
            missing.append(entry.name)
            continue
        pdb_id, _, ccd = entry.name.partition("_")
        records.append(
            {
                "sample_id": entry.name,
                "target": pdb_id,
                "protein_pdb": str(protein),
                "ligand_sdf": str(ligand),
                "ccd": ccd,
            }
        )
    if missing:
        print(f"[pb] WARNING {len(missing)} folders lack protein/ligand: {missing[:5]}")
    expected = EXPECTED_SIZE.get(set_dir.name)
    if expected is not None and len(records) != expected:
        msg = f"{set_dir.name}: indexed {len(records)} complexes, expected {expected}"
        raise SystemExit(msg)
    print(f"[pb] indexed {len(records)} complexes in {set_dir.name}")
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=paths.DATA_DIR / "posebusters")
    p.add_argument(
        "--sets",
        nargs="+",
        default=list(EXPECTED_SIZE),
        choices=list(EXPECTED_SIZE),
        help="which bundled sets to index (both live in the one archive)",
    )
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    args = p.parse_args()

    archive = args.data_dir / "posebusters_paper_data.zip"
    if not args.skip_download:
        download(archive)
    if not args.skip_extract:
        extract(archive, args.data_dir)

    for name in args.sets:
        set_dir = args.data_dir / name
        if not set_dir.exists():
            msg = f"{set_dir} missing; re-run without --skip-extract"
            raise SystemExit(msg)
        index = args.data_dir / f"{name}_index.json"
        index.write_text(json.dumps(build_index(set_dir), indent=2))
        print(f"[pb] wrote index -> {index}")


if __name__ == "__main__":
    main()
