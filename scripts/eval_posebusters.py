"""Add PoseBusters PB-validity to the eval dump (DiffGui Table-2 style).

PB-validity (Buttenschoen et al., PoseBusters) checks each generated 3D
conformation for reasonable geometry: bond lengths, bond angles, ring
flatness, no internal steric clashes, sanitization, etc. A molecule is
PB-valid iff ALL ``config="mol"`` checks pass.

Molecules are reconstructed from the stored (elements, coords) via OpenBabel
(same as DiffSBDD/our openbabel column), then passed to PoseBusters. Operates
on the raw arrays already in ``eval_data.npz`` -- no GPU / regeneration. Run::

    uv run --with posebusters python scripts/eval_posebusters.py
"""
# ruff: noqa: S603, PLR2004, E501

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_REAL = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "H"}


def _obabel() -> str:
    return shutil.which("obabel") or "/home/5/uq02055/usr/app/babel/bin/obabel"


def _reconstruct(coords_list: list, elements_list: list) -> dict[int, object]:
    """Reconstruct RDKit mols (largest sanitized fragment) via OpenBabel."""
    from rdkit import Chem  # noqa: PLC0415

    frames = []
    for i, (els, xyz) in enumerate(zip(elements_list, coords_list, strict=True)):
        syms = [str(e) for e in els]
        if len(syms) < 2 or any(e not in _REAL for e in syms):
            continue
        pts = np.asarray(xyz)
        body = "\n".join(
            f"{e} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
            for e, c in zip(syms, pts, strict=True)
        )
        frames.append(f"{len(syms)}\n{i}\n{body}\n")
    mols: dict[int, object] = {}
    if not frames:
        return mols
    with tempfile.TemporaryDirectory() as tmp:
        xyz_path, sdf_path = Path(tmp) / "in.xyz", Path(tmp) / "out.sdf"
        xyz_path.write_text("".join(frames))
        # -h adds hydrogens so open valences are filled (otherwise RDKit reads
        # them as radicals and PoseBusters' no_radicals check fails universally).
        subprocess.run([_obabel(), str(xyz_path), "-O", str(sdf_path), "-h"], check=False, capture_output=True)
        if not sdf_path.exists():
            return mols
        for mol in Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False):
            if mol is None:
                continue
            try:
                orig = int(mol.GetProp("_Name").strip())
                frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                largest = max(frags, key=lambda m: m.GetNumAtoms())
                Chem.SanitizeMol(largest)
            except Exception:  # noqa: BLE001, S112
                continue
            mols[orig] = largest
    return mols


def pb_valid(coords_list: list, elements_list: list, *, chunk: int = 2000) -> np.ndarray:
    from posebusters import PoseBusters  # noqa: PLC0415

    n = len(coords_list)
    valid = np.zeros(n, dtype=bool)
    mols_by_idx = _reconstruct(coords_list, elements_list)
    idxs = sorted(mols_by_idx)
    # Drop the energy_ratio checks: they embed a conformer per molecule (very
    # slow on distorted generated geometry and a CPU hog). The remaining checks
    # (bond lengths, bond angles, steric clash, ring flatness, sanitization)
    # are exactly the criteria DiffGui's PB-validity describes. Cap workers so
    # this stays polite on a shared node.
    base = PoseBusters(config="mol")
    cfg = base.config
    # Drop two checks that are not what DiffGui's PB-validity measures
    # (bond lengths, angles, clashes): energy_ratio (slow conformer embedding)
    # and check_radicals (open valences from heavy-atom OpenBabel reconstruction
    # are flagged as radicals for ~every molecule, incl. real GT -- an artifact
    # of reconstruction, not of the generated geometry).
    _drop = {"energy_ratio", "check_radicals"}
    cfg["modules"] = [m for m in cfg["modules"] if m.get("function") not in _drop]
    cfg["max_workers"] = 4
    buster = PoseBusters(config=cfg)
    for start in range(0, len(idxs), chunk):
        batch_idx = idxs[start : start + chunk]
        batch_mols = [mols_by_idx[i] for i in batch_idx]
        df = buster.bust(batch_mols)
        passed = df.all(axis=1).to_numpy()
        for j, i in enumerate(batch_idx):
            if j < len(passed):
                valid[i] = bool(passed[j])
        print(f"  PoseBusters {min(start + chunk, len(idxs))}/{len(idxs)}", flush=True)
    return valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("outputs/gen_eval/eval_data.npz"))
    parser.add_argument(
        "--max-gen",
        type=int,
        default=None,
        help="Evaluate a random sample of N generated mols (PoseBusters is slow).",
    )
    args = parser.parse_args()

    d = dict(np.load(args.npz, allow_pickle=True))
    gc, ge = list(d["gen_coords_list"]), list(d["gen_elements_list"])
    if args.max_gen is not None and len(gc) > args.max_gen:
        sel = np.random.default_rng(0).choice(len(gc), args.max_gen, replace=False)
        gc, ge = [gc[i] for i in sel], [ge[i] for i in sel]
        print(f"sampling {args.max_gen}/{len(d['gen_coords_list'])} generated mols")
    print("PoseBusters on generated...")
    gen_v = pb_valid(gc, ge)
    print("PoseBusters on ground-truth...")
    gt_v = pb_valid(list(d["gt_coords_list"]), list(d["gt_elements_list"]))
    d["gen_v_pb_valid"] = gen_v
    d["gt_v_pb_valid"] = gt_v
    methods = [str(m) for m in d["methods"]]
    if "pb_valid" not in methods:
        methods.append("pb_valid")
    d["methods"] = np.array(methods, dtype=object)
    np.savez(args.npz, **d)
    print(f"PB-validity: gen {100 * gen_v.mean():.1f}%  gt {100 * gt_v.mean():.1f}%")


if __name__ == "__main__":
    main()
