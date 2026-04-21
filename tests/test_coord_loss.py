"""Tests for the batched torch NeRF primitives and the coord-reconstruction loss.

Validates:
- Numerical equivalence of torch primitives vs. numpy references.
- Round-trip reconstruction via the batched torch NeRF.
- Differentiability (finite gradients, no NaN).
- Robustness to degenerate geometry.
- Uncertainty-weighting gradient directions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.descriptors import (
    _refs_orig_to_dfs,
    _segments_to_starts,
    collate_ligand_with_refs,
    collate_protein_with_segments,
)
from src.model.vqvae_module import TaskWeighting
from src.tokenizers.geometry import (
    canonical_virtual_ref,
    canonical_virtual_ref_batched,
    place_atom,
    place_atom_batched,
    project_unit_circle,
    spherical_to_cartesian,
    spherical_to_cartesian_batched,
)
from src.tokenizers.ligand import LigandDescriptor
from src.tokenizers.protein import (
    BackboneZMatrixDescriptor,
)
from src.tokenizers.vqvae import (
    TransformerVQVAE,
    TransformerVQVAEConfig,
    _reconstruct_coords_ligand,
    _reconstruct_coords_protein,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ethanol() -> tuple[
    list[tuple[str, float, float, float]],
    list[tuple[int, int, int]],
]:
    atoms: list[tuple[str, float, float, float]] = [
        ("C", 0.0, 0.0, 0.0),
        ("C", 1.54, 0.0, 0.0),
        ("O", 2.31, 1.26, 0.0),
        ("H", -0.36, 1.03, 0.0),
        ("H", -0.36, -0.51, 0.89),
        ("H", -0.36, -0.51, -0.89),
        ("H", 1.90, -0.51, 0.89),
        ("H", 1.90, -0.51, -0.89),
        ("H", 3.27, 1.05, 0.0),
    ]
    bonds: list[tuple[int, int, int]] = [
        (0, 1, 1),
        (1, 2, 1),
        (0, 3, 1),
        (0, 4, 1),
        (0, 5, 1),
        (1, 6, 1),
        (1, 7, 1),
        (2, 8, 1),
    ]
    return atoms, bonds


def _make_cyclopropane() -> tuple[
    list[tuple[str, float, float, float]],
    list[tuple[int, int, int]],
]:
    r = 0.87
    atoms: list[tuple[str, float, float, float]] = [
        ("C", r, 0.0, 0.0),
        ("C", r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3), 0.0),
        ("C", r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3), 0.0),
        ("H", r + 0.5, 0.5, 0.5),
        ("H", r + 0.5, -0.5, -0.5),
        ("H", -1.0, 1.2, 0.5),
        ("H", -1.0, 1.2, -0.5),
        ("H", -1.0, -1.2, 0.5),
        ("H", -1.0, -1.2, -0.5),
    ]
    bonds: list[tuple[int, int, int]] = [
        (0, 1, 1),
        (1, 2, 1),
        (0, 2, 1),
        (0, 3, 1),
        (0, 4, 1),
        (1, 5, 1),
        (1, 6, 1),
        (2, 7, 1),
        (2, 8, 1),
    ]
    return atoms, bonds


def _refs_original_to_dfs(
    refs_orig: np.ndarray,
    order: list[int],
) -> np.ndarray:
    """Convert ref indices from original-atom space to DFS-position space."""
    order_inv = np.full(len(order), -1, dtype=np.int64)
    for dfs_pos, orig in enumerate(order):
        order_inv[orig] = dfs_pos
    out = np.full_like(refs_orig, -1)
    for pos in range(refs_orig.shape[0]):
        for j in range(3):
            orig_idx = int(refs_orig[pos, j])
            out[pos, j] = -1 if orig_idx == -1 else order_inv[orig_idx]
    return out


def _make_anchored_ligand_descriptors(
    atoms: list[tuple[str, float, float, float]],
    bonds: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Run LigandDescriptor.compute in anchored mode and return (desc, refs, meta).

    Refs are returned in **DFS-position** space (not original-atom indices),
    matching what ``_reconstruct_coords_ligand`` expects.
    """
    coords = np.array([(a[1], a[2], a[3]) for a in atoms], dtype=np.float64)
    centroid = coords.mean(axis=0)
    rotation = np.eye(3)
    pocket_frame = (centroid, rotation)
    desc, _elements, meta = LigandDescriptor().compute(
        atoms, bonds, pocket_frame=pocket_frame
    )
    refs_orig = np.array(meta["refs"], dtype=np.int64)
    refs_dfs = _refs_original_to_dfs(refs_orig, meta["order"])
    return desc, refs_dfs, meta


# ---------------------------------------------------------------------------
# Primitive equivalence
# ---------------------------------------------------------------------------


def test_spherical_to_cartesian_batched_matches_numpy() -> None:
    rng = np.random.default_rng(42)
    for _ in range(10):
        r = float(rng.uniform(0.1, 3.0))
        theta = float(rng.uniform(0.0, np.pi))
        phi = float(rng.uniform(-np.pi, np.pi))
        np_out = spherical_to_cartesian(
            r, theta, float(np.sin(phi)), float(np.cos(phi))
        )

        t_out = spherical_to_cartesian_batched(
            torch.tensor(r),
            torch.tensor(theta),
            torch.tensor(np.sin(phi)),
            torch.tensor(np.cos(phi)),
        )
        np.testing.assert_allclose(t_out.numpy(), np_out, atol=1e-6)


def test_canonical_virtual_ref_batched_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        angle_ref = rng.normal(size=3)
        parent = rng.normal(size=3)
        np_out = canonical_virtual_ref(angle_ref, parent)
        t_out = canonical_virtual_ref_batched(
            torch.from_numpy(angle_ref),
            torch.from_numpy(parent),
        )
        np.testing.assert_allclose(t_out.numpy(), np_out, atol=1e-6)


def test_place_atom_batched_matches_numpy() -> None:
    rng = np.random.default_rng(123)
    for _ in range(20):
        ref_a = rng.normal(size=3)
        ref_b = rng.normal(size=3)
        ref_c = rng.normal(size=3)
        d = float(rng.uniform(0.8, 2.0))
        angle = float(rng.uniform(0.5, np.pi - 0.5))
        tau = float(rng.uniform(-np.pi, np.pi))

        np_out = place_atom(ref_a, ref_b, ref_c, d, angle, tau)
        t_out = place_atom_batched(
            torch.from_numpy(ref_a),
            torch.from_numpy(ref_b),
            torch.from_numpy(ref_c),
            torch.tensor(d, dtype=torch.float64),
            torch.tensor(angle, dtype=torch.float64),
            torch.tensor(np.sin(tau), dtype=torch.float64),
            torch.tensor(np.cos(tau), dtype=torch.float64),
        )
        np.testing.assert_allclose(t_out.numpy(), np_out, atol=1e-6)


def test_place_atom_batched_degenerate_cross() -> None:
    """ref_a - ref_b parallel to ref_b - ref_c must produce finite output."""
    # Deliberately colinear refs.
    ref_c = torch.zeros(3, dtype=torch.float64)
    ref_b = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    ref_a = torch.tensor(
        [2.0, 0.0, 0.0], dtype=torch.float64
    )  # colinear with ref_b-ref_c

    out = place_atom_batched(
        ref_a,
        ref_b,
        ref_c,
        torch.tensor(1.5, dtype=torch.float64),
        torch.tensor(1.9, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    assert torch.isfinite(out).all(), "degenerate NeRF produced non-finite output"


def test_place_atom_batched_leading_batch_dims() -> None:
    """Batched inputs with an extra leading dim must match per-row numpy."""
    rng = np.random.default_rng(7)
    b = 4
    ref_a = rng.normal(size=(b, 3))
    ref_b = rng.normal(size=(b, 3))
    ref_c = rng.normal(size=(b, 3))
    d = rng.uniform(0.8, 2.0, size=b)
    angle = rng.uniform(0.5, np.pi - 0.5, size=b)
    tau = rng.uniform(-np.pi, np.pi, size=b)

    t_out = place_atom_batched(
        torch.from_numpy(ref_a),
        torch.from_numpy(ref_b),
        torch.from_numpy(ref_c),
        torch.from_numpy(d),
        torch.from_numpy(angle),
        torch.from_numpy(np.sin(tau)),
        torch.from_numpy(np.cos(tau)),
    ).numpy()

    for i in range(b):
        np_out = place_atom(ref_a[i], ref_b[i], ref_c[i], d[i], angle[i], tau[i])
        np.testing.assert_allclose(t_out[i], np_out, atol=1e-6)


def test_project_unit_circle() -> None:
    """Off-circle (s, c) must be normalized; same torsion direction preserved."""
    sin_raw = torch.tensor(2.0)
    cos_raw = torch.tensor(0.0)
    s, c = project_unit_circle(sin_raw, cos_raw)
    assert torch.isclose(s * s + c * c, torch.tensor(1.0), atol=1e-6)
    # Should agree with feeding (1, 0) directly.
    s2, c2 = project_unit_circle(torch.tensor(1.0), torch.tensor(0.0))
    assert torch.isclose(s, s2, atol=1e-6)
    assert torch.isclose(c, c2, atol=1e-6)


def test_project_unit_circle_zero_safe() -> None:
    """(0, 0) must not produce NaN even though it has no defined angle."""
    s, c = project_unit_circle(torch.tensor(0.0), torch.tensor(0.0))
    assert torch.isfinite(s)
    assert torch.isfinite(c)


def test_place_atom_batched_gradients_finite() -> None:
    """Backprop through place_atom must produce finite gradients."""
    rng = np.random.default_rng(11)
    ref_a = torch.from_numpy(rng.normal(size=(3, 3))).requires_grad_()
    ref_b = torch.from_numpy(rng.normal(size=(3, 3))).requires_grad_()
    ref_c = torch.from_numpy(rng.normal(size=(3, 3))).requires_grad_()
    d = torch.tensor([1.5, 1.4, 1.6], dtype=torch.float64, requires_grad=True)
    angle = torch.tensor([1.9, 1.8, 2.0], dtype=torch.float64, requires_grad=True)
    s = torch.tensor([0.3, 0.0, 0.7], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([0.95, 1.0, 0.71], dtype=torch.float64, requires_grad=True)

    out = place_atom_batched(ref_a, ref_b, ref_c, d, angle, s, c)
    loss = out.pow(2).sum()
    loss.backward()

    for tensor in (ref_a, ref_b, ref_c, d, angle, s, c):
        assert torch.isfinite(tensor.grad).all(), "non-finite gradient detected"


# ---------------------------------------------------------------------------
# Placeholder for higher-level tests added when _reconstruct_coords lands.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ligand coord reconstruction (batched torch vs numpy reference)
# ---------------------------------------------------------------------------


def _numpy_ligand_coords(
    desc: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """Return coords in canonical frame in DFS order (matching torch output)."""
    # LigandDescriptor.descriptor_to_coords returns coords in ORIGINAL atom order.
    # The torch reconstruction produces coords in DFS order (same as desc rows).
    # Reorder numpy output accordingly.
    coords_orig = LigandDescriptor.descriptor_to_coords(desc, meta, pocket_frame=None)
    order: list[int] = meta["order"]
    return coords_orig[order]


def test_reconstruct_coords_ligand_matches_numpy_ethanol() -> None:
    atoms, bonds = _make_ethanol()
    desc, refs, meta = _make_anchored_ligand_descriptors(atoms, bonds)
    np_coords = _numpy_ligand_coords(desc, meta)

    desc_t = torch.from_numpy(desc).unsqueeze(0)  # (1, L, 4)
    refs_t = torch.from_numpy(refs).unsqueeze(0)  # (1, L, 3)
    mask_t = torch.ones(1, desc.shape[0], dtype=torch.bool)

    t_coords = _reconstruct_coords_ligand(desc_t, refs_t, mask_t, bond_length_min=0.5)
    np.testing.assert_allclose(t_coords.squeeze(0).numpy(), np_coords, atol=1e-5)


def test_reconstruct_coords_ligand_matches_numpy_cyclopropane() -> None:
    atoms, bonds = _make_cyclopropane()
    desc, refs, meta = _make_anchored_ligand_descriptors(atoms, bonds)
    np_coords = _numpy_ligand_coords(desc, meta)

    desc_t = torch.from_numpy(desc).unsqueeze(0)
    refs_t = torch.from_numpy(refs).unsqueeze(0)
    mask_t = torch.ones(1, desc.shape[0], dtype=torch.bool)

    t_coords = _reconstruct_coords_ligand(desc_t, refs_t, mask_t, bond_length_min=0.5)
    np.testing.assert_allclose(t_coords.squeeze(0).numpy(), np_coords, atol=1e-5)


def test_reconstruct_coords_ligand_batched_mixed_sizes() -> None:
    """Mix ethanol (9 atoms) and cyclopropane (9 atoms) in one batch."""
    eth_atoms, eth_bonds = _make_ethanol()
    cyc_atoms, cyc_bonds = _make_cyclopropane()
    eth_desc, eth_refs, eth_meta = _make_anchored_ligand_descriptors(
        eth_atoms, eth_bonds
    )
    cyc_desc, cyc_refs, cyc_meta = _make_anchored_ligand_descriptors(
        cyc_atoms, cyc_bonds
    )

    assert eth_desc.shape[0] == cyc_desc.shape[0]
    length = eth_desc.shape[0]

    desc_t = torch.stack(
        [torch.from_numpy(eth_desc), torch.from_numpy(cyc_desc)],
        dim=0,
    )  # (2, L, 4)
    refs_t = torch.stack(
        [torch.from_numpy(eth_refs), torch.from_numpy(cyc_refs)],
        dim=0,
    )
    mask_t = torch.ones(2, length, dtype=torch.bool)

    t_coords = _reconstruct_coords_ligand(desc_t, refs_t, mask_t, bond_length_min=0.5)
    np.testing.assert_allclose(
        t_coords[0].numpy(),
        _numpy_ligand_coords(eth_desc, eth_meta),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        t_coords[1].numpy(),
        _numpy_ligand_coords(cyc_desc, cyc_meta),
        atol=1e-5,
    )


def test_reconstruct_coords_ligand_padding_zeroed() -> None:
    """Rows beyond mask should be exactly zero in the output."""
    atoms, bonds = _make_ethanol()
    desc, refs, _meta = _make_anchored_ligand_descriptors(atoms, bonds)
    length = desc.shape[0]

    # Pad to length + 3 with garbage values.
    padded_desc = np.concatenate(
        [desc, np.full((3, 4), 999.0, dtype=np.float32)],
        axis=0,
    )
    padded_refs = np.concatenate([refs, np.full((3, 3), -1, dtype=np.int64)], axis=0)
    mask = np.concatenate(
        [np.ones(length, dtype=bool), np.zeros(3, dtype=bool)],
        axis=0,
    )

    coords = _reconstruct_coords_ligand(
        torch.from_numpy(padded_desc).unsqueeze(0),
        torch.from_numpy(padded_refs).unsqueeze(0),
        torch.from_numpy(mask).unsqueeze(0),
        bond_length_min=0.5,
    )
    assert (coords[0, length:].abs() < 1e-9).all(), "padded rows must be zero"


def test_reconstruct_coords_ligand_gradients_finite() -> None:
    atoms, bonds = _make_ethanol()
    desc, refs, _meta = _make_anchored_ligand_descriptors(atoms, bonds)

    desc_t = torch.from_numpy(desc).unsqueeze(0).requires_grad_()
    refs_t = torch.from_numpy(refs).unsqueeze(0)
    mask_t = torch.ones(1, desc.shape[0], dtype=torch.bool)

    coords = _reconstruct_coords_ligand(desc_t, refs_t, mask_t, bond_length_min=0.5)
    loss = coords.pow(2).sum()
    loss.backward()
    assert torch.isfinite(desc_t.grad).all()


# ---------------------------------------------------------------------------
# Protein backbone coord reconstruction
# ---------------------------------------------------------------------------


def _make_asymmetric_backbone(num_residues: int = 12) -> np.ndarray:
    coords = np.zeros((num_residues, 3, 3), dtype=np.float64)
    for i in range(num_residues):
        ca = np.array(
            [
                i * 3.0,
                np.sin(i * 0.5) * 2.0,
                np.cos(i * 0.7) * 0.5,
            ],
        )
        n = ca + np.array([0.47, -0.26, -1.0])
        c = ca + np.array([-0.47, 0.26, 0.5])
        coords[i] = [n, ca, c]
    return coords


def _segment_starts_from_segments(
    segments: list[tuple[int, int]],
    length: int,
) -> np.ndarray:
    starts = np.zeros(length, dtype=bool)
    for seg_start, _seg_end in segments:
        starts[seg_start] = True
    return starts


def test_reconstruct_coords_protein_matches_numpy_single_segment() -> None:
    backbone = _make_asymmetric_backbone(12)
    residue_ids = [("A", i) for i in range(12)]
    desc, meta = BackboneZMatrixDescriptor().compute(backbone, residue_ids)

    # Numpy reference (canonical frame).
    centroid = meta["centroid"]
    rotation = meta["rotation"]
    wc = np.zeros_like(backbone)
    for i in range(len(backbone)):
        for j in range(3):
            wc[i, j] = (backbone[i, j] - centroid) @ rotation.T

    starts = _segment_starts_from_segments(meta["segments"], len(desc))

    desc_t = torch.from_numpy(desc).unsqueeze(0)
    starts_t = torch.from_numpy(starts).unsqueeze(0)
    mask_t = torch.ones(1, len(desc), dtype=torch.bool)

    t_coords = _reconstruct_coords_protein(
        desc_t, starts_t, mask_t, bond_length_min=0.5
    )
    np.testing.assert_allclose(t_coords.squeeze(0).numpy(), wc, atol=1e-4)


def test_reconstruct_coords_protein_matches_numpy_multi_segment() -> None:
    backbone = _make_asymmetric_backbone(15)
    # Force two segments by making residue indices non-contiguous at position 7.
    residue_ids = [("A", i) for i in range(7)] + [("A", i + 10) for i in range(7, 15)]
    desc, meta = BackboneZMatrixDescriptor().compute(backbone, residue_ids)

    centroid = meta["centroid"]
    rotation = meta["rotation"]
    wc = np.zeros_like(backbone)
    for i in range(len(backbone)):
        for j in range(3):
            wc[i, j] = (backbone[i, j] - centroid) @ rotation.T

    starts = _segment_starts_from_segments(meta["segments"], len(desc))
    assert starts.sum() >= 2, "test setup requires >=2 segments"

    desc_t = torch.from_numpy(desc).unsqueeze(0)
    starts_t = torch.from_numpy(starts).unsqueeze(0)
    mask_t = torch.ones(1, len(desc), dtype=torch.bool)

    t_coords = _reconstruct_coords_protein(
        desc_t, starts_t, mask_t, bond_length_min=0.5
    )
    np.testing.assert_allclose(t_coords.squeeze(0).numpy(), wc, atol=1e-4)


# ---------------------------------------------------------------------------
# End-to-end integration: TransformerVQVAE.forward with coord_loss_enabled
# ---------------------------------------------------------------------------


def _make_ligand_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    atoms, bonds = _make_ethanol()
    desc, refs, _meta = _make_anchored_ligand_descriptors(atoms, bonds)
    desc_t = torch.from_numpy(desc).unsqueeze(0).float()
    refs_t = torch.from_numpy(refs).unsqueeze(0)
    mask_t = torch.ones(1, desc.shape[0], dtype=torch.bool)
    return desc_t, refs_t, mask_t


def test_vqvae_forward_coord_loss_zero_when_disabled() -> None:
    cfg = TransformerVQVAEConfig(descriptor_dim=4, coord_loss_enabled=False)
    model = TransformerVQVAE(cfg).eval()
    x, refs, mask = _make_ligand_batch()

    out = model(x, mask=mask, aux=refs)
    assert out["coord_loss"].item() == 0.0


def test_vqvae_forward_coord_loss_identical_inputs_zero() -> None:
    """If x_hat == x (bypass the network), coord_loss must be ~0."""
    cfg = TransformerVQVAEConfig(descriptor_dim=4, coord_loss_enabled=True)
    model = TransformerVQVAE(cfg).eval()
    # Inject trivial normalization (identity): mean=0, std=1.
    model.set_normalization(torch.zeros(4), torch.ones(4))
    x, refs, mask = _make_ligand_batch()

    # Direct call to the helper with identical inputs.
    coord_loss, _diag = model._compute_coord_loss(x, x, mask, aux=refs)  # noqa: SLF001
    assert coord_loss.item() < 1e-8


def test_vqvae_forward_coord_loss_nonzero_with_perturbation() -> None:
    cfg = TransformerVQVAEConfig(descriptor_dim=4, coord_loss_enabled=True)
    model = TransformerVQVAE(cfg).eval()
    model.set_normalization(torch.zeros(4), torch.ones(4))
    x, refs, mask = _make_ligand_batch()
    x_hat = x + 0.05  # perturb every descriptor

    coord_loss, diag = model._compute_coord_loss(x, x_hat, mask, aux=refs)  # noqa: SLF001
    assert coord_loss.item() > 0.0
    for key in ("coord_recon_max", "unit_circle_norm_err", "bond_clamp_frac"):
        assert key in diag


def test_vqvae_forward_coord_loss_requires_aux() -> None:
    cfg = TransformerVQVAEConfig(descriptor_dim=4, coord_loss_enabled=True)
    model = TransformerVQVAE(cfg).eval()
    model.set_normalization(torch.zeros(4), torch.ones(4))
    x, _refs, mask = _make_ligand_batch()

    with pytest.raises(ValueError, match="aux was not provided"):
        model(x, mask=mask, aux=None)


# ---------------------------------------------------------------------------
# TaskWeighting (Kendall & Gal 2018 uncertainty weighting)
# ---------------------------------------------------------------------------


def test_task_weighting_gradient_directions() -> None:
    """High loss → ∂L/∂s < 0 (increase weight); low loss → ∂L/∂s > 0."""
    tw = TaskWeighting()
    recon = torch.tensor(5.0)  # large → should pull log_var_recon down
    coord = torch.tensor(0.1)  # small → should push log_var_coord up
    loss = tw(recon, coord)
    loss.backward()

    assert tw.log_var_recon.grad.item() < 0.0
    assert tw.log_var_coord.grad.item() > 0.0


# ---------------------------------------------------------------------------
# Collate: ligand-with-refs and protein-with-segments
# ---------------------------------------------------------------------------


def test_collate_ligand_with_refs_shapes() -> None:
    b1 = (torch.randn(3, 4), torch.full((3, 3), 0, dtype=torch.long))
    b2 = (torch.randn(5, 4), torch.full((5, 3), 1, dtype=torch.long))
    desc, refs, mask = collate_ligand_with_refs([b1, b2])
    assert desc.shape == (2, 5, 4)
    assert refs.shape == (2, 5, 3)
    assert mask.shape == (2, 5)
    # Padded rows should have refs = -1 and mask = False
    assert (refs[0, 3:] == -1).all()
    assert not mask[0, 3:].any()
    assert mask[0, :3].all()
    assert mask[1].all()


def test_collate_protein_with_segments_shapes() -> None:
    b1 = (torch.randn(4, 12), torch.tensor([True, False, False, False]))
    b2 = (torch.randn(6, 12), torch.tensor([True, False, True, False, False, False]))
    desc, seg, mask = collate_protein_with_segments([b1, b2])
    assert desc.shape == (2, 6, 12)
    assert seg.shape == (2, 6)
    assert seg.dtype == torch.bool
    assert mask.shape == (2, 6)
    # Padded rows should have seg = False
    assert not seg[0, 4:].any()
    assert not mask[0, 4:].any()


def test_refs_orig_to_dfs_roundtrip() -> None:
    # order = [2, 0, 1] means DFS position 0 is atom 2, etc.
    order = [2, 0, 1]
    refs_orig = [(-1, -1, -1), (2, -1, -1), (0, 2, -1)]
    dfs = _refs_orig_to_dfs(refs_orig, order)
    # atom 2 → dfs_pos 0, atom 0 → dfs_pos 1
    assert dfs[0].tolist() == [-1, -1, -1]
    assert dfs[1].tolist() == [0, -1, -1]  # parent=atom 2 → dfs_pos 0
    assert dfs[2].tolist() == [1, 0, -1]  # parent=atom 0→dfs 1, angle_ref=atom 2→dfs 0


def test_segments_to_starts() -> None:
    segments = [(0, 5), (5, 10)]
    starts = _segments_to_starts(segments, 10)
    assert starts[0]
    assert starts[5]
    assert not starts[1]
    assert not starts[6]


def test_task_weighting_stationary_point() -> None:
    """At exp(-s) = 1/loss the gradient wrt s should be ~zero."""
    tw = TaskWeighting()
    loss_val = 2.5
    with torch.no_grad():
        tw.log_var_recon.fill_(float(np.log(loss_val)))  # s = log(σ²) = log(L)
        tw.log_var_coord.fill_(float(np.log(loss_val)))
    recon = torch.tensor(loss_val)
    coord = torch.tensor(loss_val)
    loss = tw(recon, coord)
    loss.backward()
    assert abs(tw.log_var_recon.grad.item()) < 1e-5
    assert abs(tw.log_var_coord.grad.item()) < 1e-5
