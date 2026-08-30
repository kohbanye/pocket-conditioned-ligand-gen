"""The torsion head must be equivariant in the right way, per output.

A translation and a rotation axis live in the frame (1o); a dihedral angle does
not (0e). Getting that wrong is silent -- the model still trains -- so it is
pinned here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from e3nn import o3

from prolit.chem.torsions import rotatable_bonds
from prolit.config import PoseRefinerConfig
from prolit.model.pose_refiner import FEATURE_FIELDS
from prolit.model.torsion_refiner import TorsionRefinerNet
from prolit.model.torsion_transform import apply_transform

N_LIG = 5


@pytest.fixture(scope="module")
def net():  # noqa: ANN201
    """Built once: e3nn tensor-product construction dominates this file's runtime."""
    cfg = PoseRefinerConfig()
    cfg.hidden_dim = 4
    cfg.n_layers = 1
    cfg.l_max = 1
    return TorsionRefinerNet(cfg).double()


def _setup(seed: int = 0):  # noqa: ANN202
    rng = np.random.default_rng(seed)
    pos = torch.tensor(rng.normal(size=(N_LIG, 3)) * 2.0)
    feat = torch.zeros(N_LIG, len(FEATURE_FIELDS), dtype=torch.long)
    t = torch.zeros(N_LIG, dtype=torch.float64)
    # Chain edges plus next-nearest, rather than the dense graph: the head only
    # needs messages to reach the torsion endpoints.
    src, dst = [], []
    for i in range(N_LIG):
        for j in range(N_LIG):
            if i != j and abs(i - j) == 1:
                src.append(i)
                dst.append(j)
    edge_src = torch.tensor(src)
    edge_dst = torch.tensor(dst)
    movable = torch.ones(N_LIG, dtype=torch.bool)
    edge_bond = torch.zeros(len(src), dtype=torch.long)
    bonds = np.array([[i, i + 1] for i in range(N_LIG - 1)], dtype=np.int64)
    pairs, masks = rotatable_bonds(bonds, N_LIG)
    return pos, feat, t, edge_src, edge_dst, movable, edge_bond, pairs, masks


def test_torsion_angles_are_rotation_invariant(net) -> None:  # noqa: ANN001
    """A dihedral does not change when the whole complex is rotated."""
    pos, feat, t, es, ed, mv, eb, pairs, _ = _setup()
    p = torch.tensor(pairs)
    with torch.no_grad():
        _, _, a0 = net(pos, feat, t, es, ed, mv, eb, p)
        rot = torch.tensor(
            o3_random_rotation(1), dtype=pos.dtype
        )
        _, _, a1 = net(pos @ rot.T, feat, t, es, ed, mv, eb, p)
    assert torch.allclose(a0, a1, atol=1e-6), f"{a0} vs {a1}"


def o3_random_rotation(seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    return o3.rand_matrix().double().numpy()


def test_translation_and_axis_are_equivariant(net) -> None:  # noqa: ANN001
    """Both 1o outputs must rotate with the input."""
    pos, feat, t, es, ed, mv, eb, pairs, _ = _setup(1)
    p = torch.tensor(pairs)
    rot = torch.tensor(o3_random_rotation(2), dtype=pos.dtype)
    with torch.no_grad():
        tr0, rv0, _ = net(pos, feat, t, es, ed, mv, eb, p)
        tr1, rv1, _ = net(pos @ rot.T, feat, t, es, ed, mv, eb, p)
    assert torch.allclose(tr0 @ rot.T, tr1, atol=1e-6)
    assert torch.allclose(rv0 @ rot.T, rv1, atol=1e-6)


def test_output_shapes_and_no_pairs(net) -> None:  # noqa: ANN001
    pos, feat, t, es, ed, mv, eb, pairs, _ = _setup(3)
    tr, rv, a = net(pos, feat, t, es, ed, mv, eb, torch.tensor(pairs))
    assert tr.shape == (1, 3)
    assert rv.shape == (1, 3)
    assert a.shape == (len(pairs),)
    tr, rv, a = net(pos, feat, t, es, ed, mv, eb, torch.zeros(0, 2, dtype=torch.long))
    assert a.numel() == 0


def test_end_to_end_keeps_bond_lengths(net) -> None:  # noqa: ANN001
    """Whatever the network predicts, the pose it produces cannot stretch a bond."""
    pos, feat, t, es, ed, mv, eb, pairs, masks = _setup(4)
    p, m = torch.tensor(pairs), torch.tensor(masks)
    tr, rv, a = net(pos, feat, t, es, ed, mv, eb, p)
    out = apply_transform(pos, tr[0], rv[0], p, m, a)
    bonds = [(i, i + 1) for i in range(N_LIG - 1)]
    for u, v in bonds:
        b0 = (pos[u] - pos[v]).norm().item()
        b1 = (out[u] - out[v]).norm().item()
        assert abs(b0 - b1) < 1e-7, f"bond {u}-{v}: {b0} -> {b1}"


def test_zero_angle_is_where_the_gradient_is_largest() -> None:
    """The state the head ran to must be the one it can most easily leave.

    The first run's head escaped to |cs| = 490 and emitted 0.42 deg against a
    22.9 deg corruption, because d(atan2)/d(cs) ~ 1/|cs| dies there. Normalising
    cs did not help (the chain rule keeps the factor). With pi * tanh, emitting
    zero means raw = 0, which is the gradient maximum.
    """
    raw = torch.zeros(3, requires_grad=True)
    target = torch.tensor([0.9, -0.6, 0.3])
    (math.pi * torch.tanh(raw) - target).pow(2).sum().backward()
    at_zero = raw.grad.abs().sum().item()

    far = torch.full((3,), 3.0, requires_grad=True)
    (math.pi * torch.tanh(far) - target).pow(2).sum().backward()
    at_far = far.grad.abs().sum().item()

    assert at_zero > at_far, (
        f"zero must not be a low-gradient hiding place: {at_zero} vs {at_far}"
    )
    assert at_zero > 1.0
