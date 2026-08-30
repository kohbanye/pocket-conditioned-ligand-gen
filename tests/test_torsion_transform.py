"""The torsion output space must be unable to distort the molecule.

Every free-displacement refiner in this repository buys receptor contact by
shrinking the molecule: bonds out of tolerance go 10.0% (no refiner) -> 48.1%
(``refit_press0.6``), which costs PoseBusters validity 0.73 -> 0.42. Weighting
a bond loss does not stop it, because bonded pairs carry 0.077 of the loss. So
the guarantee has to come from the output space, and these tests are what
holds it.
"""

from __future__ import annotations

import numpy as np
import torch

from prolit.chem.torsions import rotatable_bonds, torsion_delta
from prolit.model.torsion_transform import apply_torsions, apply_transform


def _chain(n: int) -> tuple[np.ndarray, np.ndarray]:
    """A simple open chain: bonds 0-1-2-...-(n-1)."""
    pos = np.stack([np.arange(n, dtype=np.float64) * 1.5,
                    np.sin(np.arange(n)) * 0.6,
                    np.zeros(n)], axis=1)
    bonds = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    return pos, bonds


def _pairwise(x: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, x)


def test_torsion_preserves_every_bond_length_and_angle() -> None:
    pos, bonds = _chain(8)
    pairs, masks = rotatable_bonds(bonds, len(pos))
    assert len(pairs) > 0
    x = torch.tensor(pos)
    ang = torch.tensor(np.random.default_rng(0).uniform(-np.pi, np.pi, len(pairs)))
    y = apply_torsions(x, torch.tensor(pairs), torch.tensor(masks), ang)

    # bond lengths
    for u, v in bonds:
        b0 = (x[u] - x[v]).norm().item()
        b1 = (y[u] - y[v]).norm().item()
        assert abs(b0 - b1) < 1e-8, f"bond {u}-{v} changed {b0} -> {b1}"
    # bond angles (every bonded triple)
    for k in range(1, len(pos) - 1):
        def ang_at(p: torch.Tensor, k: int = k) -> float:
            a = p[k - 1] - p[k]
            b = p[k + 1] - p[k]
            return float(torch.dot(a, b) / (a.norm() * b.norm()))
        assert abs(ang_at(x) - ang_at(y)) < 1e-8


def test_rotatable_bonds_skips_rings() -> None:
    """A bond in a ring cannot be rotated without breaking the ring."""
    ring = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]], dtype=np.int64)
    pairs, _ = rotatable_bonds(ring, 6)
    assert len(pairs) == 0, f"ring bonds must not be rotatable, got {pairs}"


def test_rotatable_bonds_skips_terminal_bonds() -> None:
    """Rotating a leaf sweeps it on a cone; measured to fix 1% of buried atoms."""
    star = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int64)
    pairs, _ = rotatable_bonds(star, 4)
    assert len(pairs) == 0


def test_masks_exclude_the_axis_atoms() -> None:
    pos, bonds = _chain(6)
    pairs, masks = rotatable_bonds(bonds, len(pos))
    for (i, j), m in zip(pairs, masks, strict=True):
        assert not m[i], "axis atom i must not move"
        assert not m[j], "axis atom j must not move"
        assert m.any()


def test_zero_parameters_is_the_identity() -> None:
    pos, bonds = _chain(7)
    pairs, masks = rotatable_bonds(bonds, len(pos))
    x = torch.tensor(pos)
    y = apply_transform(
        x, torch.zeros(3, dtype=x.dtype), torch.zeros(3, dtype=x.dtype),
        torch.tensor(pairs), torch.tensor(masks),
        torch.zeros(len(pairs), dtype=x.dtype),
    )
    assert torch.allclose(x, y, atol=1e-7)


def test_full_transform_is_an_isometry_on_the_bond_graph() -> None:
    """Rigid part moves everything; bonds still must not stretch."""
    pos, bonds = _chain(9)
    pairs, masks = rotatable_bonds(bonds, len(pos))
    x = torch.tensor(pos)
    rng = np.random.default_rng(3)
    y = apply_transform(
        x,
        torch.tensor(rng.normal(size=3)),
        torch.tensor(rng.normal(size=3) * 0.4),
        torch.tensor(pairs), torch.tensor(masks),
        torch.tensor(rng.uniform(-np.pi, np.pi, len(pairs))),
    )
    for u, v in bonds:
        assert abs((x[u] - x[v]).norm().item() - (y[u] - y[v]).norm().item()) < 1e-8


def test_gradients_flow_to_every_parameter() -> None:
    pos, bonds = _chain(8)
    pairs, masks = rotatable_bonds(bonds, len(pos))
    x = torch.tensor(pos)
    tr = torch.zeros(3, dtype=x.dtype, requires_grad=True)
    rv = torch.full((3,), 0.1, dtype=x.dtype, requires_grad=True)
    an = torch.full((len(pairs),), 0.2, dtype=x.dtype, requires_grad=True)
    y = apply_transform(x, tr, rv, torch.tensor(pairs), torch.tensor(masks), an)
    y.pow(2).sum().backward()
    assert tr.grad is not None
    assert rv.grad is not None
    assert an.grad is not None
    assert torch.isfinite(tr.grad).all()
    assert torch.isfinite(rv.grad).all()
    assert torch.isfinite(an.grad).all()
    assert an.grad.abs().sum() > 0, "torsion angles must receive gradient"


def test_torsions_compose_order_independently() -> None:
    """Applying the same angles in any order gives the same pose.

    This is what makes the head's job well posed: it predicts all K angles from
    the corrupted pose in one shot, and they are applied sequentially. If the
    result depended on the order, the network would have to model the order
    too. It does not, because each rotation uses the CURRENT axis -- rotating
    an outer bond first conjugates the inner axis by the same rotation, which
    cancels.
    """
    pos, bonds = _chain(9)
    pairs, masks = rotatable_bonds(bonds, len(pos))
    assert len(pairs) >= 3
    x = torch.tensor(pos)
    ang = torch.tensor(np.random.default_rng(11).uniform(-np.pi, np.pi, len(pairs)))

    y1 = apply_torsions(x, torch.tensor(pairs), torch.tensor(masks), ang)
    order = np.array([2, 0, 1, *range(3, len(pairs))])
    y2 = apply_torsions(
        x,
        torch.tensor(pairs[order]),
        torch.tensor(masks[order]),
        ang[order],
    )
    assert torch.allclose(y1, y2, atol=1e-8), (y1 - y2).abs().max()


def test_torsion_delta_recovers_a_known_rotation() -> None:
    """The dihedral difference must be exactly the rotation that was applied.

    This is the well-posed target: it says how to turn THIS pose into THAT one,
    unlike "undo the synthetic twist", which points at a pose the model never
    sees (the stored x0 is itself 0.76 A from the crystal).
    """
    pos, bonds = _chain(9)
    pairs, masks = rotatable_bonds(bonds, len(pos))
    applied = np.random.default_rng(5).uniform(-1.0, 1.0, len(pairs))
    moved = apply_torsions(
        torch.tensor(pos), torch.tensor(pairs), torch.tensor(masks),
        torch.tensor(applied),
    ).numpy()
    # Going from `moved` back to `pos` should be -applied.
    got = torsion_delta(moved, pos, bonds, pairs)
    err = np.arctan2(np.sin(got + applied), np.cos(got + applied))
    assert np.abs(err).max() < 1e-5, np.abs(err).max()
