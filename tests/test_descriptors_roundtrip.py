"""Round-trip tests for the spherical multi-feature descriptors.

Ensure that ``compute → descriptor_to_coords`` recovers the input
Cartesian coords (modulo the known ``r=0`` ambiguity at the origin) for
both ligand atoms and protein backbone residues.
"""

from __future__ import annotations

import numpy as np

from src.tokenizers.ligand import LigandDescriptor
from src.tokenizers.protein import (
    BackboneSphericalDescriptor,
    _compute_canonical_frame,
)


def _make_pocket_frame(ca_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid, rotation = _compute_canonical_frame(ca_coords)
    return centroid, rotation


class TestLigandRoundTrip:
    """Compute spherical descriptor and reconstruct global coords."""

    def _make_simple_molecule(
        self,
    ) -> tuple[list[tuple[str, float, float, float]], list[tuple[int, int, int]]]:
        # Tiny mol: 5 carbons in a chain at distinct distances from origin.
        atoms: list[tuple[str, float, float, float]] = [
            ("C", 1.5, 0.0, 0.0),
            ("C", 3.0, 0.5, 0.2),
            ("C", 4.5, -0.2, 0.7),
            ("N", 5.5, 0.3, -0.5),
            ("O", 6.7, -0.4, 0.1),
        ]
        bonds = [(i, i + 1, 1) for i in range(len(atoms) - 1)]
        return atoms, bonds

    def test_round_trip_recovers_global_coords(self) -> None:
        atoms, bonds = self._make_simple_molecule()
        # Build a deterministic pocket frame around a synthetic backbone.
        ca_synthetic = np.array(
            [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]],
            dtype=np.float64,
        )
        pocket_frame = _make_pocket_frame(ca_synthetic)

        desc_calc = LigandDescriptor()
        descriptors, _elements, metadata = desc_calc.compute(
            atoms,
            bonds,
            pocket_frame=pocket_frame,
        )
        assert descriptors.shape == (len(atoms), LigandDescriptor.DESCRIPTOR_DIM)

        recovered = LigandDescriptor.descriptor_to_coords(
            descriptors,
            metadata,
            pocket_frame=pocket_frame,
        )
        original = np.array([(a[1], a[2], a[3]) for a in atoms], dtype=np.float64)
        np.testing.assert_allclose(recovered, original, atol=1e-4)

    def test_descriptor_dim_matches_schema(self) -> None:
        atoms, bonds = self._make_simple_molecule()
        ca_synthetic = np.array(
            [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]],
            dtype=np.float64,
        )
        pocket_frame = _make_pocket_frame(ca_synthetic)
        desc_calc = LigandDescriptor()
        descriptors, _elements, _meta = desc_calc.compute(
            atoms,
            bonds,
            pocket_frame=pocket_frame,
        )
        assert descriptors.shape[1] == LigandDescriptor.DESCRIPTOR_DIM

    def test_drops_hydrogens_from_output(self) -> None:
        atoms = [
            ("C", 1.5, 0.0, 0.0),
            ("H", 1.6, 1.0, 0.0),  # should be dropped
            ("C", 3.0, 0.0, 0.0),
        ]
        bonds = [(0, 1, 1), (0, 2, 1)]
        ca_synthetic = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        pocket_frame = _make_pocket_frame(ca_synthetic)
        desc_calc = LigandDescriptor()
        descriptors, elements, _meta = desc_calc.compute(
            atoms,
            bonds,
            pocket_frame=pocket_frame,
        )
        assert descriptors.shape[0] == 2
        assert elements == ["C", "C"]


class TestProteinRoundTrip:
    """Backbone spherical descriptor must reconstruct (N, CA, C) exactly."""

    def _synthetic_backbone(
        self,
    ) -> tuple[np.ndarray, list[tuple[str, int]], list[str]]:
        # 5 residues along x-axis with realistic N-CA-C offsets.
        n_res = 5
        backbone = np.zeros((n_res, 3, 3), dtype=np.float32)
        for i in range(n_res):
            ca = np.array([3.8 * i, 0.0, 0.0])
            n = ca + np.array([-1.0, 0.5, 0.0])
            c = ca + np.array([1.0, -0.3, 0.4])
            backbone[i, 0] = n
            backbone[i, 1] = ca
            backbone[i, 2] = c
        residue_ids = [("A", i + 1) for i in range(n_res)]
        aa = ["A", "L", "G", "V", "K"]
        return backbone, residue_ids, aa

    def test_round_trip_recovers_backbone(self) -> None:
        backbone, residue_ids, aa = self._synthetic_backbone()
        desc_calc = BackboneSphericalDescriptor()
        descriptors, metadata = desc_calc.compute(
            backbone,
            residue_ids,
            residue_names_one_letter=aa,
        )
        assert descriptors.shape == (
            len(backbone),
            BackboneSphericalDescriptor.DESCRIPTOR_DIM,
        )

        recovered = BackboneSphericalDescriptor.descriptor_to_backbone_coords(
            descriptors,
            metadata,
        )
        np.testing.assert_allclose(
            recovered.astype(np.float64),
            backbone.astype(np.float64),
            atol=1e-4,
        )
