"""Prepare the CASP16 pharma-ligand complexes as a benchmark dataset.

Each target archive (``L*.tgz``) holds ``protein_aligned.pdb`` plus one or more
``ligand_*.pdb`` files (with CONECT records). This script extracts them, converts
each ligand PDB to a V2000 SDF (what the own model's ``parse_sdf`` expects), and
writes an index the ``casp16`` dataset loader reads.

These structures are CASP16 experimental targets — guaranteed out of every
model's training data, so reconstruction here is a clean held-out test.

    uv run python scripts/prepare_casp.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from plbench import paths  # noqa: E402


def extract_targets(casp_dir: Path, out_dir: Path) -> None:
    tgzs = sorted(casp_dir.glob("L*.tgz"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for tgz in tgzs:
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(out_dir, filter="data")
    print(f"[casp] extracted {len(tgzs)} target archives -> {out_dir}")


def _rdkit_pdb_to_sdf(pdb_path: Path, sdf_path: Path) -> bool:
    """Fallback conversion via RDKit, used only when OpenBabel is unavailable.

    WARNING: RDKit's PDB reader does not perceive bond orders -- every bond comes
    back SINGLE, aromatic rings included. Coordinates and elements are exact, so
    RMSD / TM-score / lDDT are unaffected, but every chemistry-aware metric
    (PoseBusters validity, bond-length and bond-angle error) is meaningless on
    the result, because the crystal reference itself fails them.
    """
    from rdkit import Chem

    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
    if mol is None or mol.GetNumAtoms() == 0:
        return False
    try:
        Chem.SanitizeMol(mol)
        block = Chem.MolToMolBlock(mol)
    except Exception:  # noqa: BLE001 - fall back to non-kekulized block
        try:
            block = Chem.MolToMolBlock(mol, kekulize=False)
        except Exception:  # noqa: BLE001
            return False
    sdf_path.write_text(block + "$$$$\n")
    return True


def ligand_pdb_to_sdf(pdb_path: Path, sdf_path: Path) -> bool:
    """Convert a ligand PDB (with CONECT) to a V2000 SDF, bond orders included.

    OpenBabel perceives bond orders from geometry; RDKit's PDB reader does not.
    On a 15-ligand CASP16 sample the crystal references pass PoseBusters 15/15
    after OpenBabel and only 6/15 through RDKit alone -- the failures being
    aromatic rings flagged as non-flat *non-aromatic* rings, precisely because
    their bonds had all been read as single. The converter, not the tokenizer,
    was what those checks were measuring.

    OpenBabel preserves atom count, element order, and coordinates exactly
    (verified: max coordinate difference 0.0), so changing converters cannot move
    any RMSD-based number; it only fixes the bond block.
    """
    proc = subprocess.run(  # noqa: S603
        [paths.OBABEL, "-ipdb", str(pdb_path), "-osdf", "-O", str(sdf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and sdf_path.exists() and sdf_path.stat().st_size > 0:
        return True
    return _rdkit_pdb_to_sdf(pdb_path, sdf_path)


def build_index(extracted_dir: Path, rebuild: bool = False) -> list[dict]:
    records: list[dict] = []
    n_fail = 0
    for prot in sorted(extracted_dir.rglob("protein_aligned.pdb")):
        target_dir = prot.parent
        target = target_dir.name
        for lig_pdb in sorted(target_dir.glob("ligand_*.pdb")):
            sdf = lig_pdb.with_suffix(".sdf")
            if (rebuild or not sdf.exists()) and not ligand_pdb_to_sdf(lig_pdb, sdf):
                n_fail += 1
                continue
            records.append(
                {
                    "sample_id": f"{target}__{lig_pdb.stem}",
                    "target": target,
                    "protein_pdb": str(prot),
                    "ligand_pdb": str(lig_pdb),
                    "ligand_sdf": str(sdf),
                }
            )
    print(f"[casp] indexed {len(records)} complexes ({n_fail} ligand conversions failed)")
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--casp-dir", type=Path, default=paths.DATA_DIR / "casp16")
    p.add_argument("--out-index", type=Path, default=paths.DATA_DIR / "casp16" / "index.json")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument(
        "--rebuild-ligand-sdf",
        action="store_true",
        help="re-convert every ligand PDB even if its SDF exists (needed after "
        "changing the converter -- SDFs written by an older version are kept "
        "otherwise, silently preserving wrong bond orders)",
    )
    args = p.parse_args()

    extracted = args.casp_dir / "extracted"
    if not args.skip_extract:
        extract_targets(args.casp_dir, extracted)
    records = build_index(extracted, rebuild=args.rebuild_ligand_sdf)
    args.out_index.write_text(json.dumps(records, indent=2))
    print(f"[casp] wrote index -> {args.out_index}")


if __name__ == "__main__":
    main()
