"""The projections of the refiner's displacement must not change the molecule.

``_rigid_part`` and ``_rigid_torsion_part`` exist so a refiner can move a pose
without buying the move with chemistry: over 94 targets the free-displacement
head takes bonds out of tolerance from 10.3% to 48.8% and changes the perceived
SMILES of 52.2% of molecules. Both projections are exact by construction, so a
regression here is silent -- the poses still look plausible -- and these tests
are what makes it loud.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from generate_ligands_3d import (  # noqa: E402
    _project_displacement,
    _rigid_part,
    _rigid_torsion_part,
)


def _chain(n: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """A kinked chain with real bond lengths, and its bond list."""
    rng = np.random.default_rng(0)
    pos = np.zeros((n, 3), dtype=np.float32)
    d = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for i in range(1, n):
        d = d + 0.6 * rng.standard_normal(3).astype(np.float32)
        d /= np.linalg.norm(d)
        pos[i] = pos[i - 1] + 1.53 * d
    bonds = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    return pos, bonds


def _lengths(pos: np.ndarray, bonds: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pos[bonds[:, 0]] - pos[bonds[:, 1]], axis=1)


def _angles(pos: np.ndarray, bonds: np.ndarray) -> np.ndarray:
    out = []
    for k in range(len(bonds) - 1):
        a, b = bonds[k]
        c = bonds[k + 1][1]
        v1, v2 = pos[a] - pos[b], pos[c] - pos[b]
        out.append(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.array(out)


def test_torsion_projection_preserves_bonds_and_angles() -> None:
    """A distorted target must not drag bond lengths or angles along."""
    pos, bonds = _chain()
    rng = np.random.default_rng(1)
    # The kind of output a free-displacement refiner gives: every atom moved
    # independently, so bond lengths in `after` are wrong by ~0.3 A.
    after = pos + rng.normal(0.0, 0.3, pos.shape).astype(np.float32)
    assert np.abs(_lengths(after, bonds) - _lengths(pos, bonds)).max() > 0.1

    out = _rigid_torsion_part(pos, after, bonds)
    assert np.abs(_lengths(out, bonds) - _lengths(pos, bonds)).max() < 1e-4
    assert np.abs(_angles(out, bonds) - _angles(pos, bonds)).max() < 1e-4


def test_torsion_projection_follows_further_than_rigid() -> None:
    """Rotating a real torsion must reproduce that torsion exactly."""
    pos, bonds = _chain()
    from prolit.chem.torsions import rotatable_bonds  # noqa: PLC0415

    pairs, masks = rotatable_bonds(bonds, len(pos))
    assert len(pairs) > 0, "the chain must have rotatable bonds to test with"

    import torch  # noqa: PLC0415

    from prolit.model.torsion_transform import apply_torsions  # noqa: PLC0415

    angles = np.full(len(pairs), 0.4, dtype=np.float32)
    after = (
        apply_torsions(
            torch.from_numpy(pos).float(),
            torch.from_numpy(pairs).long(),
            torch.from_numpy(masks).bool(),
            torch.from_numpy(angles).float(),
        )
        .numpy()
        .astype(np.float32)
    )
    # A pure torsional move: the rigid projection cannot represent it, the
    # torsion projection reproduces it.
    rigid = _rigid_part(pos, after)
    tors = _rigid_torsion_part(pos, after, bonds)
    rmsd = lambda a, b: float(np.sqrt(((a - b) ** 2).sum(1).mean()))  # noqa: E731
    assert rmsd(tors, after) < 1e-3
    assert rmsd(rigid, after) > 10 * rmsd(tors, after)


def test_projection_is_identity_on_a_rigid_move() -> None:
    """Both projections must pass a pure rigid motion through untouched."""
    pos, bonds = _chain()
    theta = 0.7
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    after = (pos - pos.mean(0)) @ rot.T + pos.mean(0) + np.array([1.0, -2.0, 0.5])
    for out in (
        _rigid_part(pos, after.astype(np.float32)),
        _rigid_torsion_part(pos, after.astype(np.float32), bonds),
    ):
        assert np.abs(out - after).max() < 1e-3


@pytest.mark.parametrize("mode", ["none", "rigid", "torsion"])
def test_dispatch_covers_every_cli_choice(mode: str) -> None:
    """Every ``--refine-project`` choice must reach a projection, not fall through."""
    pos, bonds = _chain()
    rng = np.random.default_rng(2)
    after = (pos + rng.normal(0.0, 0.3, pos.shape)).astype(np.float32)
    out = _project_displacement(mode, pos, after, lambda: bonds)
    assert out.shape == pos.shape
    if mode == "none":
        assert np.allclose(out, after)
    else:
        assert np.abs(_lengths(out, bonds) - _lengths(pos, bonds)).max() < 1e-4
