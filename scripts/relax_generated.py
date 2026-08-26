"""Repair the local geometry of an already-generated output tree.

Relaxation needs neither the LM nor a GPU -- it is a function of the molecule,
not of the model -- so it runs as a post-process over a finished generation
rather than inside it. One sampling run therefore yields both numbers: point the
evaluator at the original tree for the unrelaxed column and at this one for the
relaxed column, which is how they should be reported (the baselines are not
relaxed either).

    python scripts/relax_generated.py --in-dir outputs/gen \
        --out-dir outputs/gen_relaxed --max-displacement 0.102

``--max-displacement`` is the tokenizer's own coordinate MAE, not a tuned knob;
:mod:`prolit.chem.relax` explains why and shows the measured curve. Molecules
MMFF cannot type are copied through unchanged rather than dropped, so the two
trees always hold the same molecules in the same order and the comparison stays
paired.

Passing ``--receptor-dir`` additionally slides each molecule off the receptor
wall as a rigid body (:mod:`prolit.chem.rigid_fit`), looking for
``<receptor-dir>/<target_id>/<target_id>_receptor.pdb``. That is a separate
defect from local geometry -- the model's placement error is global, not
per-atom -- and it is applied second, because a rigid transform commutes with
nothing the relaxation does but the relaxation would otherwise be resolving
strain against the wrong position.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from prolit.chem.pdb_io import read_heavy_atoms
from prolit.chem.relax import decoder_bond_error, relax_local_geometry
from prolit.chem.rigid_fit import rigid_pocket_fit, vdw_radii
from prolit.chem.torsion_fit import torsion_pocket_fit


def _turn_into_place(mol: Chem.Mol, receptor: tuple[np.ndarray, np.ndarray]) -> None:
    """Rigid placement plus a torsional settle, in place.

    Falls back to the rigid step alone for molecules RDKit will not perceive
    rings on, since without ring membership there is no telling which single
    bonds are safe to turn.
    """
    try:
        Chem.GetSymmSSSR(mol)
        coords = torsion_pocket_fit(mol, receptor[1], receptor[0])
    except (ValueError, RuntimeError):
        _slide_off_wall(mol, receptor)
        return
    conformer = mol.GetConformer()
    for i, xyz in enumerate(coords):
        conformer.SetAtomPosition(i, xyz.tolist())


def _slide_off_wall(mol: Chem.Mol, receptor: tuple[np.ndarray, np.ndarray]) -> float:
    """Rigidly move ``mol`` out of the receptor in place; returns the shift."""
    conformer = mol.GetConformer()
    coords = conformer.GetPositions()
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    heavy = np.array([e != "H" for e in elements])
    if not heavy.any():
        return 0.0
    # The fit is driven by the heavy atoms alone -- the same set the clash
    # metric uses -- and the resulting transform is then applied to every atom,
    # so hydrogens ride along with the heavy atoms they hang off.
    fit = rigid_pocket_fit(
        coords[heavy],
        vdw_radii([e for e in elements if e != "H"]),
        receptor[1],
        receptor[0],
    )
    for i, xyz in enumerate(fit.apply(coords)):
        conformer.SetAtomPosition(i, xyz.tolist())
    return fit.shift


def _receptor_for(receptor_dir: Path, target_id: str) -> tuple[np.ndarray, np.ndarray]:
    pdb = receptor_dir / target_id / f"{target_id}_receptor.pdb"
    if not pdb.exists():
        msg = f"no receptor for {target_id} at {pdb}"
        raise SystemExit(msg)
    elements, coords = read_heavy_atoms(pdb)
    return vdw_radii(elements), coords


def relax_sdf(
    src: Path,
    dst: Path,
    max_displacement: float | None,
    receptor: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    torsions: bool = False,
) -> tuple[int, int]:
    """Relax every record of ``src`` into ``dst``. Returns (relaxed, total)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    relaxed = total = 0
    # sanitize=False, then sanitize a COPY. Letting the supplier sanitize would
    # make it return None for the molecules that need repairing most, and those
    # would silently vanish from the relaxed tree -- shrinking the denominator
    # and raising every rate computed from it. Every input record must come out
    # the other side, repaired or not, or the two trees stop being comparable.
    with Chem.SDWriter(str(dst)) as writer:
        for mol in Chem.SDMolSupplier(str(src), sanitize=False, removeHs=False):
            if mol is None:
                continue
            total += 1
            candidate = Chem.Mol(mol)
            try:
                Chem.SanitizeMol(candidate)
            except (ValueError, RuntimeError):
                candidate = None
            radius = max_displacement
            if radius is None and candidate is not None:
                radius = decoder_bond_error(candidate)
            out = (
                relax_local_geometry(candidate, radius)
                if candidate is not None
                else None
            )
            if out is None:
                out = mol
            else:
                relaxed += 1
                # SDMolSupplier drops _Name onto the new molecule only if it
                # round-trips; carry it explicitly so the two trees stay keyed
                # the same way.
                if mol.HasProp("_Name"):
                    out.SetProp("_Name", mol.GetProp("_Name"))
            # A record named "ref" is the crystal ligand the evaluator scores
            # as the reference column. Sliding it would rewrite the yardstick,
            # so it passes through the rigid step untouched -- 15% of crystal
            # ligands do clash by this threshold, and that is the reference's
            # honest value.
            name = out.GetProp("_Name") if out.HasProp("_Name") else ""
            if receptor is not None and name != "ref" and out.GetNumConformers():
                if torsions:
                    _turn_into_place(out, receptor)
                else:
                    _slide_off_wall(out, receptor)
            writer.write(out)
    return relaxed, total


def _one(
    item: tuple[Path, Path, float | None, Path | None, bool],
) -> tuple[Path, int, int]:
    """One target, as a picklable unit of work for the process pool."""
    src, dst, max_displacement, receptor_dir, torsions = item
    RDLogger.DisableLog("rdApp.*")
    receptor = (
        _receptor_for(receptor_dir, src.parent.name)
        if receptor_dir is not None
        else None
    )
    r, n = relax_sdf(src, dst, max_displacement, receptor, torsions=torsions)
    return src, r, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--max-displacement",
        type=float,
        default=None,
        help="flat-bottom restraint radius in Angstroms. Omit to take it per "
        "molecule from that molecule's own worst bond-length error, which is "
        "what the restraint is trying to be (see prolit.chem.relax)",
    )
    parser.add_argument(
        "--receptor-dir",
        type=Path,
        default=None,
        help="also slide each molecule off the receptor wall as a rigid body; "
        "receptors are read from <dir>/<target_id>/<target_id>_receptor.pdb",
    )
    parser.add_argument(
        "--torsions",
        action="store_true",
        help="also turn the rotatable bonds when settling into the pocket; "
        "bond lengths and angles stay exactly as they were either way",
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="targets to process in parallel"
    )
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")

    sdfs = sorted(args.in_dir.glob("*/*/generated.sdf"))
    if not sdfs:
        sdfs = sorted(args.in_dir.glob("*/generated.sdf"))
    if not sdfs:
        msg = f"no generated.sdf under {args.in_dir}"
        raise SystemExit(msg)

    # manifest.json is deliberately NOT copied across. It records absolute
    # paths into the tree it was written for, so a copy would send the
    # evaluator back to the unrelaxed molecules while reporting the relaxed
    # directory -- silently, with no missing file to notice. Without it
    # run_evaluation falls back to globbing */generated.sdf, which finds
    # exactly the molecules that are actually here.
    work = [
        (src, args.out_dir / src.relative_to(args.in_dir), args.max_displacement,
         args.receptor_dir, args.torsions)
        for src in sdfs
    ]
    total_r = total_n = 0
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(_one, work))
    else:
        results = [_one(item) for item in work]
    for src, r, n in results:
        total_r += r
        total_n += n
        print(f"[relax] {src.relative_to(args.in_dir)}: {r}/{n}", flush=True)

    print(f"\n[relax] {total_r}/{total_n} molecules relaxed over {len(sdfs)} targets")


if __name__ == "__main__":
    main()
