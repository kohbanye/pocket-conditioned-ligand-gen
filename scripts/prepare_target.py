"""Prepare a docking target from a PDB entry (e.g. EGFR 2ITY).

Splits a crystal structure into a protein-only receptor and its co-crystal
ligand, converts both to the AutoDock ``.pdbqt`` format used by Vina, and
derives a docking box centred on the reference ligand. The reference ligand
doubles as (a) the pocket definition that conditions ligand generation and
(b) a positive-control pose for docking ("does the known inhibitor score
well in its own crystal pose?").

Outputs under ``--out-dir``::

    {tag}_raw.pdb            verbatim download / input
    {tag}_receptor.pdb       protein ATOM records only (no ligand / water)
    {tag}_receptor.pdbqt     receptor prepared for Vina
    {tag}_ref_ligand.sdf     reference ligand, bonds perceived by Open Babel
    {tag}_ref_ligand.pdbqt   reference ligand prepared for Vina
    {tag}_box.json           {"center": [x,y,z], "size": [sx,sy,sz]}

Example::

    uv run python scripts/prepare_target.py --pdb-id 2ITY --ligand-resname IRE \
        --out-dir data/targets/2ity --tag 2ity
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# System tools (overridable via CLI). These live outside the uv venv.
DEFAULT_OBABEL = "/home/5/uq02055/usr/app/babel/bin/obabel"
DEFAULT_PREPARE_RECEPTOR = "/home/5/uq02055/usr/app/ADFRsuite/bin/prepare_receptor"

# Standard amino-acid residue names kept in the receptor (drop everything else
# that arrives as ATOM, e.g. modified residues we cannot prep cleanly).
_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}
_WATER = {"HOH", "WAT", "DOD"}


def _download_pdb(pdb_id: str, dest: Path) -> None:
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    logger.info("Downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        dest.write_bytes(resp.read())


def _auto_ligand_resname(het_lines: list[str]) -> str:
    """Pick the most common non-water HETATM residue as the reference ligand."""
    counts = Counter(ln[17:20].strip() for ln in het_lines)
    for water in _WATER:
        counts.pop(water, None)
    if not counts:
        msg = "No non-water HETATM residue found to use as reference ligand."
        raise ValueError(msg)
    return counts.most_common(1)[0][0]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    logger.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb-id", type=str, help="RCSB PDB id to download, e.g. 2ITY")
    src.add_argument("--pdb-file", type=Path, help="Local PDB file instead of download")
    parser.add_argument(
        "--ligand-resname",
        type=str,
        default=None,
        help="Co-crystal ligand residue name (auto-detected if omitted).",
    )
    parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Restrict receptor + ligand to this chain id (default: all chains).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument(
        "--box-padding",
        type=float,
        default=6.0,
        help="Angstrom padding added to each side of the ligand bounding box.",
    )
    parser.add_argument(
        "--min-box",
        type=float,
        default=22.5,
        help="Lower bound on each box edge (Angstrom).",
    )
    parser.add_argument("--obabel", type=str, default=DEFAULT_OBABEL)
    parser.add_argument(
        "--prepare-receptor", type=str, default=DEFAULT_PREPARE_RECEPTOR
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / f"{args.tag}_raw.pdb"
    if args.pdb_file is not None:
        raw_path.write_bytes(Path(args.pdb_file).read_bytes())
    else:
        _download_pdb(args.pdb_id, raw_path)

    lines = raw_path.read_text().splitlines()
    atom_lines = [ln for ln in lines if ln[:6].strip() == "ATOM"]
    het_lines = [ln for ln in lines if ln[:6].strip() == "HETATM"]
    if args.chain:
        atom_lines = [ln for ln in atom_lines if ln[21] == args.chain]
        het_lines = [ln for ln in het_lines if ln[21] == args.chain]

    resname = args.ligand_resname or _auto_ligand_resname(het_lines)
    lig_lines = [ln for ln in het_lines if ln[17:20].strip() == resname]
    if not lig_lines:
        msg = f"Ligand residue {resname!r} not found in {raw_path}."
        raise SystemExit(msg)
    # If the ligand occupies multiple chains/copies, keep the first copy.
    first_chain = lig_lines[0][21]
    first_resseq = lig_lines[0][22:26]
    lig_lines = [
        ln for ln in lig_lines if ln[21] == first_chain and ln[22:26] == first_resseq
    ]

    # Receptor: protein ATOM records for standard amino acids only.
    rec_lines = [ln for ln in atom_lines if ln[17:20].strip() in _STANDARD_AA]
    logger.info(
        "Receptor: %d atoms | reference ligand %s: %d atoms (chain %s res %s)",
        len(rec_lines),
        resname,
        len(lig_lines),
        first_chain,
        first_resseq.strip(),
    )

    receptor_pdb = args.out_dir / f"{args.tag}_receptor.pdb"
    receptor_pdb.write_text("\n".join(rec_lines) + "\nTER\nEND\n")
    ligand_pdb = args.out_dir / f"{args.tag}_ref_ligand_raw.pdb"
    ligand_pdb.write_text("\n".join(lig_lines) + "\nEND\n")

    coords = np.array(
        [[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])] for ln in lig_lines]
    )
    center = coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0)
    size = np.maximum(extent + 2 * args.box_padding, args.min_box)
    box = {
        "center": [round(float(c), 3) for c in center],
        "size": [round(float(s), 3) for s in size],
        "ligand_resname": resname,
    }
    (args.out_dir / f"{args.tag}_box.json").write_text(json.dumps(box, indent=2))
    logger.info("Box center=%s size=%s", box["center"], box["size"])

    # Reference ligand -> SDF (bond perception) + PDBQT (protonated, charged).
    ref_sdf = args.out_dir / f"{args.tag}_ref_ligand.sdf"
    ref_pdbqt = args.out_dir / f"{args.tag}_ref_ligand.pdbqt"
    r = _run([args.obabel, str(ligand_pdb), "-O", str(ref_sdf)])
    if r.returncode != 0 or not ref_sdf.exists():
        logger.warning("obabel sdf stderr: %s", r.stderr.strip())
    r = _run(
        [args.obabel, str(ligand_pdb), "-O", str(ref_pdbqt),
         "-p", "7.4", "--partialcharge", "gasteiger"]
    )
    if r.returncode != 0 or not ref_pdbqt.exists():
        logger.warning("obabel pdbqt stderr: %s", r.stderr.strip())

    # Receptor -> PDBQT via ADFRsuite prepare_receptor, falling back to obabel.
    receptor_pdbqt = args.out_dir / f"{args.tag}_receptor.pdbqt"
    r = _run(
        [args.prepare_receptor, "-r", str(receptor_pdb), "-o", str(receptor_pdbqt),
         "-A", "checkhydrogens", "-U", "nphs_lps_waters_nonstdres"]
    )
    if r.returncode != 0 or not receptor_pdbqt.exists():
        logger.warning(
            "prepare_receptor failed (%s); falling back to obabel.", r.stderr.strip()
        )
        r = _run(
            [args.obabel, str(receptor_pdb), "-O", str(receptor_pdbqt),
             "-xr", "-p", "7.4", "--partialcharge", "gasteiger"]
        )
        if r.returncode != 0 or not receptor_pdbqt.exists():
            logger.error("obabel receptor fallback also failed: %s", r.stderr.strip())
            raise SystemExit(1)

    n_rec_atoms = sum(
        1 for ln in receptor_pdbqt.read_text().splitlines()
        if ln.startswith(("ATOM", "HETATM"))
    )
    logger.info(
        "Wrote receptor.pdbqt (%d atoms), ref_ligand.pdbqt, ref_ligand.sdf, box.json "
        "to %s",
        n_rec_atoms,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
