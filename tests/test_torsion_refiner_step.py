"""The torsion refiner must train end to end and never distort the ligand.

This is the guarantee the whole design rests on: measured over 60 targets, the
free-displacement refiners take bonds out of tolerance from 10.0% to 48.1%, and
that is what costs PoseBusters validity 0.728 -> 0.427. A torsion+rigid output
cannot do it, and this test runs the real collate to prove the wiring keeps
that property rather than only the transform in isolation.
"""

from __future__ import annotations

import numpy as np
import torch

from prolit.config import PoseRefineTrainingConfig
from prolit.data.pose_refine_dataset import make_collate
from prolit.model.pose_refiner import NUM_FEATURE_FIELDS
from prolit.model.torsion_refiner import TorsionRefinerModule
from prolit.model.torsion_transform import apply_torsions

N_LIG, N_PKT = 10, 12


def _sample(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    bonds = np.array([[i, i + 1] for i in range(N_LIG - 1)], dtype=np.int64)
    x1 = np.stack(
        [np.arange(N_LIG) * 1.4, np.sin(np.arange(N_LIG)) * 0.7, np.zeros(N_LIG)],
        axis=1,
    ).astype(np.float32)
    return {
        "x1": x1,
        "x0": (x1 + rng.normal(scale=0.3, size=x1.shape)).astype(np.float32),
        "lig_feat": np.zeros((N_LIG, NUM_FEATURE_FIELDS), dtype=np.int64),
        "bonds": bonds,
        "bond_ref": np.full(len(bonds), 1.4, dtype=np.float32),
        "pkt_x": rng.normal(scale=4.0, size=(N_PKT, 3)).astype(np.float32),
        "pkt_feat": np.zeros((N_PKT, NUM_FEATURE_FIELDS), dtype=np.int64),
        "scale": 1.0,
    }


def _batch() -> dict:
    collate = make_collate(cutoff=8.0, max_pkt=32, knn=8)
    return collate([_sample(0), _sample(1)])


def test_collate_emits_torsion_dofs() -> None:
    b = _batch()
    assert b["tors_pairs"].shape[1] == 2
    assert b["tors_pairs"].shape[0] > 0, "an open chain must have rotatable bonds"
    assert b["tors_masks"].shape[0] == b["tors_pairs"].shape[0]
    assert b["tors_ptr"].tolist()[0] == 0
    assert b["tors_ptr"].tolist()[-1] == b["tors_pairs"].shape[0]
    assert b["lig_size"].tolist() == [N_LIG, N_LIG]


def test_predict_preserves_bond_lengths_through_the_real_collate() -> None:
    cfg = PoseRefineTrainingConfig()
    cfg.model.hidden_dim = 8
    cfg.model.n_layers = 1
    cfg.model.l_max = 1
    mod = TorsionRefinerModule(cfg)
    b = _batch()
    with torch.no_grad():
        out = mod.predict(b)
    for g in range(2):
        s = int(b["lig_start"][g])
        n = int(b["lig_size"][g])
        before, after = b["pos0"][s : s + n], out[s : s + n]
        for i in range(n - 1):
            d0 = (before[i] - before[i + 1]).norm().item()
            d1 = (after[i] - after[i + 1]).norm().item()
            assert abs(d0 - d1) < 1e-4, f"graph {g} bond {i}: {d0} -> {d1}"


def test_pocket_atoms_never_move() -> None:
    cfg = PoseRefineTrainingConfig()
    cfg.model.hidden_dim = 8
    cfg.model.n_layers = 1
    cfg.model.l_max = 1
    mod = TorsionRefinerModule(cfg)
    b = _batch()
    with torch.no_grad():
        out = mod.predict(b)
    frozen = ~b["movable"]
    assert torch.allclose(out[frozen], b["pos0"][frozen], atol=1e-6)


def test_training_step_produces_finite_gradients() -> None:
    cfg = PoseRefineTrainingConfig()
    cfg.model.hidden_dim = 8
    cfg.model.n_layers = 1
    cfg.model.l_max = 1
    mod = TorsionRefinerModule(cfg)
    loss = mod.training_step(_batch(), 0)
    loss.backward()
    grads = [p.grad for p in mod.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    tor = mod.net.torsion_head[0].weight.grad
    assert tor is not None
    assert tor.abs().sum() > 0, "the torsion head must be trained"


def test_torsion_corruption_moves_atoms_without_stretching_bonds() -> None:
    """The corruption must be invertible by the output space.

    Otherwise the torsion head sees no torsional error and learns to emit zero.
    """
    from prolit.data.pose_refine_dataset import PoseRefineDataset  # noqa: PLC0415

    rng = np.random.default_rng(0)
    n = 9
    x = np.stack(
        [np.arange(n) * 1.4, np.sin(np.arange(n)) * 0.7, np.zeros(n)], axis=1
    ).astype(np.float32)
    bonds = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    ds = PoseRefineDataset.__new__(PoseRefineDataset)
    ds.torsion_sigma = 0.6
    ds._rng = rng  # noqa: SLF001
    out = ds._twist(x, bonds)  # noqa: SLF001
    assert not np.allclose(out, x), "corruption must actually move something"
    for i in range(n - 1):
        d0 = float(np.linalg.norm(x[i] - x[i + 1]))
        d1 = float(np.linalg.norm(out[i] - out[i + 1]))
        assert abs(d0 - d1) < 1e-4, f"bond {i} stretched: {d0} -> {d1}"


def test_configure_optimizers_builds(monkeypatch) -> None:  # noqa: ANN001
    """The first submitted run died here: the config field is `learning_rate`.

    The other tests never touch the optimiser, so nothing caught it until the
    job had already been queued and failed.
    """
    cfg = PoseRefineTrainingConfig()
    cfg.model.hidden_dim = 8
    cfg.model.n_layers = 1
    cfg.model.l_max = 1
    mod = TorsionRefinerModule(cfg)

    class _Trainer:
        estimated_stepping_batches = 100

    monkeypatch.setattr(
        TorsionRefinerModule, "trainer", property(lambda _self: _Trainer())
    )
    out = mod.configure_optimizers()
    assert "optimizer" in out
    assert "lr_scheduler" in out
    assert len(out["optimizer"].param_groups) >= 1


def test_refine_matches_the_pose_refiner_interface() -> None:
    """Generation drives whichever refiner it holds through ``refine(batch)``.

    ``refine_ligand_canonical`` calls ``module.refine(batch, n_steps=...)``. If
    the torsion module did not answer to that name the trained model could not
    be deployed at all -- the training would have been wasted.
    """
    cfg = PoseRefineTrainingConfig()
    cfg.model.hidden_dim = 8
    cfg.model.n_layers = 1
    cfg.model.l_max = 1
    mod = TorsionRefinerModule(cfg)
    b = _batch()
    out = mod.refine(b, n_steps=1)
    assert out.shape == b["pos0"].shape
    # n_steps is accepted and ignored; both calls must agree
    assert torch.allclose(out, mod.refine(b), atol=0)


def test_checkpoint_carries_the_torsion_head() -> None:
    """Generation picks the module class by looking for this key."""
    cfg = PoseRefineTrainingConfig()
    cfg.model.hidden_dim = 8
    cfg.model.n_layers = 1
    cfg.model.l_max = 1
    sd = TorsionRefinerModule(cfg).state_dict()
    assert any(k.startswith("net.torsion_head") for k in sd), sorted(sd)[:5]


def test_collate_carries_the_applied_twist() -> None:
    """The corruption angle is known, so it can supervise the head directly."""
    from prolit.data.pose_refine_dataset import PoseRefineDataset  # noqa: PLC0415

    n = 9
    x = np.stack(
        [np.arange(n) * 1.4, np.sin(np.arange(n)) * 0.7, np.zeros(n)], axis=1
    ).astype(np.float32)
    bonds = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    ds = PoseRefineDataset.__new__(PoseRefineDataset)
    ds.torsion_sigma = 0.5
    ds._rng = np.random.default_rng(0)  # noqa: SLF001
    twisted = ds._twist(x, bonds)  # noqa: SLF001
    applied = ds._last_twist  # noqa: SLF001
    from prolit.chem.torsions import rotatable_bonds  # noqa: PLC0415

    pairs, masks = rotatable_bonds(bonds, n)
    assert len(applied) == len(pairs)
    assert np.abs(applied).max() > 0
    # Applying the recorded angles must undo the corruption.
    back = apply_torsions(
        torch.tensor(twisted, dtype=torch.float64),
        torch.tensor(pairs), torch.tensor(masks),
        torch.tensor(applied, dtype=torch.float64),
    )
    assert np.abs(back.numpy() - x).max() < 1e-4, np.abs(back.numpy() - x).max()


def test_angle_loss_reaches_the_torsion_head() -> None:
    cfg = PoseRefineTrainingConfig()
    cfg.model.hidden_dim = 8
    cfg.model.n_layers = 1
    cfg.model.l_max = 1
    cfg.torsion_angle_weight = 1.0
    mod = TorsionRefinerModule(cfg)
    b = _batch()
    k = b["tors_pairs"].shape[0]
    b["tors_twist"] = torch.full((k,), 0.5)
    out = mod._compute(b)  # noqa: SLF001
    assert "angle_mae" in out, "direct supervision did not engage"
    out["loss"].backward()
    g = mod.net.torsion_head[0].weight.grad
    assert g is not None
    assert g.abs().sum() > 0
