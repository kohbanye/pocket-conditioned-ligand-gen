"""E(3)-equivariance + pocket-freeze tests for the pose refiner.

These are the cheapest, highest-information gates: they must pass before any
data pipeline or training. They verify the two structural guarantees the design
relies on:

1. ``refine(R x + t, R pkt + t) = R refine(x, pkt) + t`` -- the displacement is a
   proper ``1o`` vector, so a rotation/translation of the whole complex just
   rotates/translates the refined pose (robust to the heuristic pocket frame).
2. Pocket atoms never move (they are frozen context, no incoming edges).
"""

from __future__ import annotations

import numpy as np
import torch

from prolit.config import PoseRefinerConfig, PoseRefineTrainingConfig
from prolit.data.pose_refine_dataset import make_collate
from prolit.model.pose_refiner import (
    FEATURE_FIELDS,
    NUM_FEATURE_FIELDS,
    PoseRefinerModule,
)


def _random_rotation(gen: torch.Generator) -> torch.Tensor:
    """A random proper rotation (det = +1) via QR of a Gaussian matrix."""
    a = torch.randn(3, 3, dtype=torch.float64, generator=gen)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))  # fix QR sign ambiguity
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _toy_batch(gen: torch.Generator, n_lig: int = 9, n_pkt: int = 20) -> dict:
    """A single-graph batch: ligand nodes (movable) + pocket nodes (frozen)."""
    n = n_lig + n_pkt
    pos = torch.randn(n, 3, dtype=torch.float64, generator=gen) * 3.0
    movable = torch.zeros(n, dtype=torch.bool)
    movable[:n_lig] = True

    feat = torch.stack(
        [torch.randint(0, vocab, (n,), generator=gen) for _, vocab in FEATURE_FIELDS],
        dim=1,
    )
    feat[:n_lig, 0] = 1  # source = ligand
    feat[n_lig:, 0] = 0  # source = protein

    # Edges: every destination is a ligand (movable) node.
    src, dst = [], []
    for i in range(n_lig):  # ligand-ligand (both directions)
        for j in range(n_lig):
            if i != j:
                src.append(j)
                dst.append(i)
    for i in range(n_lig):  # ligand<-pocket
        for p in range(n_lig, n):
            src.append(p)
            dst.append(i)
    edge_src = torch.tensor(src, dtype=torch.long)
    edge_dst = torch.tensor(dst, dtype=torch.long)

    return {
        "pos0": pos,
        "feat": feat,
        "movable": movable,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_bond": (torch.rand(edge_src.shape[0]) < 0.3).long(),
        "batch": torch.zeros(n, dtype=torch.long),
        "num_graphs": 1,
    }


def _module() -> PoseRefinerModule:
    cfg = PoseRefineTrainingConfig(
        model=PoseRefinerConfig(hidden_dim=32, n_layers=3, l_max=2, num_radial=8)
    )
    return PoseRefinerModule(cfg).double().eval()


def test_net_displacement_is_equivariant() -> None:
    gen = torch.Generator().manual_seed(0)
    mod = _module()
    batch = _toy_batch(gen)
    pos, t_node = (
        batch["pos0"],
        torch.full((batch["pos0"].shape[0],), 0.3, dtype=torch.float64),
    )

    with torch.no_grad():
        d0 = mod.net(
            pos,
            batch["feat"],
            t_node,
            batch["edge_src"],
            batch["edge_dst"],
            batch["movable"],
            batch["edge_bond"],
        )
        rot = _random_rotation(gen)
        trans = torch.randn(3, dtype=torch.float64, generator=gen)
        d1 = mod.net(
            pos @ rot.T + trans,
            batch["feat"],
            t_node,
            batch["edge_src"],
            batch["edge_dst"],
            batch["movable"],
            batch["edge_bond"],
        )

    # displacement rotates with the frame and is translation-invariant
    assert torch.allclose(d1, d0 @ rot.T, atol=1e-6), (d1 - d0 @ rot.T).abs().max()


def test_refine_is_equivariant_and_freezes_pocket() -> None:
    gen = torch.Generator().manual_seed(1)
    mod = _module()
    batch = _toy_batch(gen)
    rot = _random_rotation(gen)
    trans = torch.randn(3, dtype=torch.float64, generator=gen)

    refined = mod.refine(batch, n_steps=5)
    batch_rt = dict(batch)
    batch_rt["pos0"] = batch["pos0"] @ rot.T + trans
    refined_rt = mod.refine(batch_rt, n_steps=5)

    # equivariance of the whole ODE
    assert torch.allclose(refined_rt, refined @ rot.T + trans, atol=1e-5), (
        (refined_rt - (refined @ rot.T + trans)).abs().max()
    )
    # pocket atoms are untouched
    pkt = ~batch["movable"]
    assert torch.allclose(refined[pkt], batch["pos0"][pkt], atol=1e-8)


def test_feature_field_count() -> None:
    assert NUM_FEATURE_FIELDS == 9


def _synthetic_getitem(rng: np.random.Generator, n_lig: int, n_pkt: int) -> dict:
    """Mimic PoseRefineDataset.__getitem__ output for collate/step smoke tests."""
    x1 = (rng.standard_normal((n_lig, 3)) * 2.0).astype(np.float32)
    x0 = (x1 + rng.standard_normal((n_lig, 3)) * 0.3).astype(np.float32)
    lig_feat = np.stack(
        [rng.integers(0, v, n_lig) for _, v in FEATURE_FIELDS], axis=1
    ).astype(np.int64)
    lig_feat[:, 0] = 1
    bonds = np.array([[i, i + 1] for i in range(n_lig - 1)], dtype=np.int64)
    bond_ref = np.linalg.norm(x1[bonds[:, 0]] - x1[bonds[:, 1]], axis=1).astype(
        np.float32
    )
    pkt_x = (rng.standard_normal((n_pkt, 3)) * 3.0).astype(np.float32)
    pkt_feat = np.stack(
        [rng.integers(0, v, n_pkt) for _, v in FEATURE_FIELDS], axis=1
    ).astype(np.int64)
    pkt_feat[:, 0] = 0
    return {
        "x1": x1,
        "x0": x0,
        "lig_feat": lig_feat,
        "bonds": bonds,
        "bond_ref": bond_ref,
        "pkt_x": pkt_x,
        "pkt_feat": pkt_feat,
        "scale": 0.5,
    }


def test_collate_step_and_backward() -> None:
    gen = np.random.default_rng(0)
    samples = [_synthetic_getitem(gen, 9, 25), _synthetic_getitem(gen, 12, 30)]
    collate = make_collate(cutoff=8.0, max_pkt=16, knn=0)
    batch = collate(samples)

    # node count = sum of ligand + pocket atoms; edges point into ligand nodes
    assert batch["pos0"].shape[0] == (9 + 25) + (12 + 30)
    assert batch["movable"].sum().item() == 9 + 12
    assert bool(batch["movable"][batch["edge_dst"]].all())  # every dst is movable

    mod = PoseRefinerModule(
        PoseRefineTrainingConfig(model=PoseRefinerConfig(hidden_dim=32, n_layers=3))
    ).train()
    out = mod._compute(batch)  # noqa: SLF001
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in mod.net.parameters())

    refined = mod.refine(batch, n_steps=4)
    assert refined.shape == batch["pos0"].shape
    pkt = ~batch["movable"]
    assert torch.allclose(refined[pkt], batch["pos0"][pkt], atol=1e-6)
