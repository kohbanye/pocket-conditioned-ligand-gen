"""Add a DiffSBDD-style validity column to the eval dump (OpenBabel + RDKit).

Replicates DiffSBDD's ``make_mol_openbabel`` + ``process_molecule`` validity:
OpenBabel perceives bonds/orders from the 3D coordinates, RDKit then takes the
largest fragment and sanitizes it -- a far more permissive (and field-standard)
definition than RDKit ``DetermineBonds(charge=0)``.

Operates on the raw coords/elements already stored in ``eval_data.npz`` -- no
GPU / regeneration needed. Re-saves the npz with ``{gen,gt}_v_openbabel`` and
appends ``openbabel`` to the ``methods`` list.

    uv run python scripts/eval_openbabel_validity.py --npz outputs/gen_eval/eval_data.npz
"""
# ruff: noqa: S603, PLR2004, E501

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from prolit.external_tools import tool_default

_REAL = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "H"}


def _obabel() -> str:
    found = tool_default("obabel")
    if not Path(found).exists():
        msg = "obabel CLI not found"
        raise FileNotFoundError(msg)
    return found


def _xyz_frame(idx: int, elements: list[str], coords: np.ndarray) -> str:
    body = "\n".join(
        f"{e} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
        for e, c in zip(elements, coords, strict=True)
    )
    return f"{len(elements)}\n{idx}\n{body}\n"  # comment line = original index


def openbabel_valid(coords_list: list, elements_list: list) -> np.ndarray:
    """DiffSBDD-style validity per molecule via OpenBabel bond perception."""
    from rdkit import Chem  # noqa: PLC0415

    n = len(coords_list)
    valid = np.zeros(n, dtype=bool)

    frames = []
    for i, (els, xyz) in enumerate(zip(elements_list, coords_list, strict=True)):
        syms = [str(e) for e in els]
        if len(syms) < 2 or any(e not in _REAL for e in syms):
            continue
        frames.append(_xyz_frame(i, syms, np.asarray(xyz)))
    if not frames:
        return valid

    obabel = _obabel()
    with tempfile.TemporaryDirectory() as tmp:
        xyz_path, sdf_path = Path(tmp) / "in.xyz", Path(tmp) / "out.sdf"
        xyz_path.write_text("".join(frames))
        subprocess.run(
            [obabel, str(xyz_path), "-O", str(sdf_path)],
            check=False,
            capture_output=True,
        )
        if not sdf_path.exists():
            return valid
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
        for mol in supplier:
            if mol is None:
                continue
            try:
                name = mol.GetProp("_Name").strip()
                orig = int(name)
            except (KeyError, ValueError):
                continue
            try:
                frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                largest = max(frags, key=lambda m: m.GetNumAtoms())
                Chem.SanitizeMol(largest)
            except Exception:  # noqa: BLE001, S112
                continue
            valid[orig] = True
    return valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("outputs/gen_eval/eval_data.npz"))
    args = parser.parse_args()

    d = dict(np.load(args.npz, allow_pickle=True))
    gen_v = openbabel_valid(list(d["gen_coords_list"]), list(d["gen_elements_list"]))
    gt_v = openbabel_valid(list(d["gt_coords_list"]), list(d["gt_elements_list"]))
    d["gen_v_openbabel"] = gen_v
    d["gt_v_openbabel"] = gt_v
    methods = [str(m) for m in d["methods"]]
    if "openbabel" not in methods:
        methods.append("openbabel")
    d["methods"] = np.array(methods, dtype=object)

    np.savez(args.npz, **d)
    print(
        f"OpenBabel/DiffSBDD-style validity: "
        f"gen {100 * gen_v.mean():.1f}%  gt {100 * gt_v.mean():.1f}%"
    )


if __name__ == "__main__":
    main()
