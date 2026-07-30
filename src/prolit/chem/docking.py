"""Small helpers for driving AutoDock Vina and reading what it writes.

Docking is a subprocess, not a library call, so every evaluation path that
scores a pose ends up doing the same four things: write the ligand somewhere
Open Babel can read it, run a command with a timeout, pull the affinity out of
Vina's stdout, and compare the optimized pose back to the input.

These used to live in ``scripts/dock_vina.py`` and be imported as
``from scripts.dock_vina import ...`` by its neighbours, which only resolved
when the process happened to start in the repository root. They are library code
and belong here.
"""

from __future__ import annotations

import re
import subprocess
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

#: Vina prints the as-is score on this line under ``--score_only``. Case-insensitive
#: because the wording has varied across Vina builds.
SCORE_RE = re.compile(
    r"Estimated Free Energy of Binding\s*:\s*(-?\d+\.?\d*)", re.IGNORECASE
)

_DEFAULT_TIMEOUT_S = 300


def run(
    cmd: list[str],
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    """Run a docking subprocess, capturing output and never raising on failure.

    Docking fails routinely on individual ligands (unparseable atoms, a pose
    outside the box), and one failure must not end a sweep over thousands, so
    the caller inspects ``returncode`` rather than catching.
    """
    return subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, check=False, timeout=timeout
    )


def parse_score(stdout: str) -> float | None:
    """Vina's ``--score_only`` affinity in kcal/mol, or None if it did not print one."""
    match = SCORE_RE.search(stdout)
    return float(match.group(1)) if match else None


def write_xyz(path: Path, elements: list[str], coords: list[list[float]]) -> None:
    """Write an XYZ file, the format Open Babel converts to PDBQT most reliably."""
    lines = [str(len(elements)), "generated"]
    lines += [
        f"{el} {x:.4f} {y:.4f} {z:.4f}"
        for el, (x, y, z) in zip(elements, coords, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n")


def read_pdbqt_heavy(path: Path) -> list[tuple[float, float, float]]:
    """Heavy-atom coordinates from a PDBQT (AutoDock hydrogen types start with 'H')."""
    out: list[tuple[float, float, float]] = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom_type = line[77:79].strip()
        if atom_type.startswith("H"):
            continue
        out.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return out


def heavy_rmsd(
    a: list[tuple[float, float, float]],
    b: list[tuple[float, float, float]],
) -> float | None:
    """In-place RMSD between two heavy-atom sets, or None if they disagree in size.

    No superposition: both poses are already in the receptor frame, so the
    displacement is the quantity of interest. A size mismatch means Open Babel
    changed the atom count, which makes the comparison meaningless rather than
    merely large.
    """
    if not a or len(a) != len(b):
        return None
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt((diff**2).sum(axis=1).mean()))
