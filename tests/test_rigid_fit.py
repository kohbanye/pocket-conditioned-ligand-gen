"""The rigid steric relief must be rigid, bounded, and never make things worse.

Those three are what let the module claim PoseBusters' intramolecular checks
are unchanged *by construction*: if the transform were not exactly rigid the
claim would be a hope rather than a fact, so it is asserted here rather than
argued in prose.
"""

from __future__ import annotations

import numpy as np
import pytest

from prolit.chem.rigid_fit import (
    CLASH_FRACTION,
    RigidTransform,
    _Overlap,
    rigid_pocket_fit,
    vdw_radii,
)


def _wall(spacing: float = 1.8, n: int = 9) -> np.ndarray:
    """A flat slab of receptor atoms in the z = 0 plane."""
    g = (np.arange(n) - (n - 1) / 2) * spacing
    x, y = np.meshgrid(g, g)
    return np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)


def _ligand(z: float, n: int = 6) -> np.ndarray:
    """A short chain of atoms held at height ``z`` above the wall."""
    rng = np.random.default_rng(3)
    coords = np.cumsum(rng.normal(0.0, 1.0, size=(n, 3)), axis=0)
    coords -= coords.mean(axis=0)
    coords[:, 2] += z
    return coords


def _fit(
    ligand: np.ndarray, receptor: np.ndarray, **kw: object
) -> tuple[RigidTransform, np.ndarray, np.ndarray]:
    lig_r = vdw_radii(["C"] * len(ligand))
    rec_r = vdw_radii(["C"] * len(receptor))
    return rigid_pocket_fit(ligand, lig_r, receptor, rec_r, **kw), lig_r, rec_r


def test_transform_is_exactly_rigid() -> None:
    receptor = _wall()
    ligand = _ligand(1.4)
    fit, _, _ = _fit(ligand, receptor)
    moved = fit.apply(ligand)

    def pairwise(x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(x[:, None] - x[None], axis=-1)

    # Every internal distance preserved: no bond, angle or torsion can change.
    assert np.allclose(pairwise(ligand), pairwise(moved), atol=1e-9)
    assert np.isclose(abs(np.linalg.det(fit.rotation)), 1.0)


def test_clash_free_pose_is_left_alone_without_settling() -> None:
    receptor = _wall()
    ligand = _ligand(8.0)          # far above the wall, nothing overlaps
    fit, _, _ = _fit(ligand, receptor, settle=False)
    assert fit.shift == 0.0
    assert np.allclose(fit.apply(ligand), ligand)


def test_settling_never_buys_contact_with_a_clash() -> None:
    """Stage two may move a clash-free pose, but not into the wall."""
    receptor = _wall()
    for height in (1.2, 2.0, 8.0):
        ligand = _ligand(height)
        fit, lig_r, rec_r = _fit(ligand, receptor)
        overlap = _Overlap(receptor, rec_r, lig_r)
        before = overlap.clashing_atoms(ligand)
        after = overlap.clashing_atoms(fit.apply(ligand))
        assert after <= before, f"height {height}: {before} -> {after}"


def test_settling_deepens_the_van_der_waals_contact() -> None:
    """A pose held off the wall settles onto it, without overlapping it."""
    receptor = _wall()
    ligand = _ligand(4.5)          # clash-free but too far out to be in contact
    fit, lig_r, rec_r = _fit(ligand, receptor)
    well = _Overlap(receptor, rec_r, lig_r, "lj")
    assert well(fit.apply(ligand)) < well(ligand)
    overlap = _Overlap(receptor, rec_r, lig_r)
    assert overlap.clashing_atoms(fit.apply(ligand)) == 0


def test_clash_is_relieved_and_never_worsened() -> None:
    receptor = _wall()
    ligand = _ligand(1.2)          # driven into the wall
    fit, lig_r, rec_r = _fit(ligand, receptor)
    overlap = _Overlap(receptor, rec_r, lig_r)
    assert overlap(ligand) > 0.0
    assert overlap(fit.apply(ligand)) < overlap(ligand)
    assert overlap.clashing_atoms(fit.apply(ligand)) <= overlap.clashing_atoms(ligand)


@pytest.mark.parametrize("max_translation", [0.5, 2.5])
def test_bounds_are_hard(max_translation: float) -> None:
    receptor = _wall()
    ligand = _ligand(0.0)          # buried in the wall, wants to run away
    fit, _, _ = _fit(
        ligand, receptor, max_translation=max_translation, max_rotation_deg=30.0
    )
    assert fit.shift <= max_translation + 1e-9
    angle = np.arccos(np.clip((np.trace(fit.rotation) - 1.0) / 2.0, -1.0, 1.0))
    assert np.rad2deg(angle) <= 30.0 + 1e-6


def test_a_known_displacement_is_recovered() -> None:
    """A pose displaced into the wall by a known amount comes back out.

    This is the control the module's headline number rests on: the search finds
    the way out when one exists, so a case where it does not is evidence about
    the pose, not about the optimiser.
    """
    receptor = _wall()
    good = _ligand(3.0)
    displaced = good - np.array([0.0, 0.0, 1.8])
    fit, lig_r, rec_r = _fit(displaced, receptor)
    overlap = _Overlap(receptor, rec_r, lig_r)
    assert overlap.clashing_atoms(displaced) > 0
    assert overlap.clashing_atoms(fit.apply(displaced)) == 0


def test_clash_fraction_matches_the_benchmark_threshold() -> None:
    # The objective and the metric must threshold the same contact, or the fix
    # optimises something the score does not measure.
    assert CLASH_FRACTION == 0.75
