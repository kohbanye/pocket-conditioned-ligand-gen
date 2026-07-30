"""Re-perceive a molecule from bare coordinates, via Open Babel.

The models emit atoms and positions, not a bond graph. Turning that back into a
molecule is a chemistry step, and doing it the same way for every model is what
makes a generation comparison fair: a model whose SDF happens to carry tidier
bond blocks must not score better for it. So every generated pose -- ours and
the baselines' -- goes through this one path.

It lived in the generation benchmark, which meant the rescoring benchmark had to
import across to reach it. It is chemistry, not benchmarking.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from prolit.external_tools import find_tool

#: Elements worth evaluating or docking. "X"/"*" is the codebook's OTHER
#: catch-all: it has no valence and cannot be written to XYZ.
REAL_ELEMENTS = frozenset(
    {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "H"}
)

_MIN_ATOMS = 2


def xyz_block(elements: list[str], coords: np.ndarray) -> str:
    """Format atoms + positions as an XYZ block."""
    body = "\n".join(
        f"{e} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
        for e, c in zip(elements, np.asarray(coords), strict=True)
    )
    return f"{len(elements)}\n\n{body}\n"


def obabel_mol(
    elements: list[str],
    coords: np.ndarray,
    *,
    obabel: str | None = None,
    add_h: bool = True,
) -> Any | None:  # noqa: ANN401
    """Sanitized RDKit mol perceived from 3D coordinates, or None.

    ``add_h`` fills open valences so RDKit does not read them as radicals. Only
    the largest fragment is kept: distorted generated geometry sometimes
    perceives into several pieces, and scoring the biggest one is the
    conventional choice.
    """
    from rdkit import Chem  # noqa: PLC0415

    symbols = [str(e) for e in elements]
    if len(symbols) < _MIN_ATOMS or any(e not in REAL_ELEMENTS for e in symbols):
        return None
    binary = obabel or find_tool("obabel") or "obabel"
    with tempfile.TemporaryDirectory() as tmp:
        xyz, sdf = Path(tmp) / "in.xyz", Path(tmp) / "out.sdf"
        xyz.write_text(xyz_block(symbols, coords))
        cmd = [binary, str(xyz), "-O", str(sdf)]
        if add_h:
            cmd.append("-h")
        subprocess.run(cmd, check=False, capture_output=True)  # noqa: S603
        if not sdf.exists():
            return None
        supplier = Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False)
        mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        return None
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        largest = max(frags, key=lambda m: m.GetNumAtoms())
        Chem.SanitizeMol(largest)
    except Exception:  # noqa: BLE001
        # Perception can produce something RDKit refuses to sanitize; that is a
        # verdict on the pose, not an error worth propagating.
        return None
    return largest
