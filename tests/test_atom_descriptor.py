"""Tests for the unified all-atom descriptor (protein + ligand)."""

from __future__ import annotations

import numpy as np

from src.tokenizers.atom import (
    LigandAtomDescriptor,
    ProteinAtomDescriptor,
    atom_descriptor_to_coords,
    rotate_atom_descriptor,
)
from src.tokenizers.descriptor_schema import (
    ATOM_DESCRIPTOR_DIM,
    ATOM_LAYOUT,
    BB_SC_BACKBONE_IDX,
    BB_SC_NA_IDX,
    BB_SC_SIDECHAIN_IDX,
    PROTEIN_AA_TO_IDX,
    PROTEIN_AA_X_IDX,
    SOURCE_LIGAND_IDX,
    SOURCE_PROTEIN_IDX,
    fields_by_name,
)
from src.tokenizers.geometry import random_rotation_matrix
from src.tokenizers.protein import PocketAtomData

_F = fields_by_name(ATOM_LAYOUT)


def _toy_ligand() -> tuple[
    list[tuple[str, float, float, float]], list[tuple[int, int, int]]
]:
    atoms = [
        ("C", 0.0, 0.0, 0.0),
        ("C", 1.5, 0.0, 0.0),
        ("O", 2.2, 1.1, 0.0),
        ("N", -1.0, 1.0, 0.5),
        ("C", -2.3, 0.6, 0.2),
    ]
    bonds = [(0, 1, 1), (1, 2, 2), (0, 3, 1), (3, 4, 1)]
    return atoms, bonds


class TestLigandAtomDescriptor:
    def test_shape_and_source(self) -> None:
        atoms, bonds = _toy_ligand()
        frame = (np.zeros(3), np.eye(3))
        desc, elems, _meta = LigandAtomDescriptor().compute(atoms, bonds, frame)
        assert desc.shape == (5, ATOM_DESCRIPTOR_DIM)
        assert len(elems) == 5
        assert np.all(desc[:, _F["source"].start] == SOURCE_LIGAND_IDX)
        # Ligand atoms take the X / NA placeholder buckets.
        assert np.all(desc[:, _F["aa"].start] == PROTEIN_AA_X_IDX)
        assert np.all(desc[:, _F["bb_sc"].start] == BB_SC_NA_IDX)

    def test_coord_roundtrip(self) -> None:
        atoms, bonds = _toy_ligand()
        centroid = np.array([0.3, -0.2, 0.1])
        rot = random_rotation_matrix(np.random.default_rng(0))
        desc, _elems, meta = LigandAtomDescriptor().compute(
            atoms, bonds, (centroid, rot)
        )
        coords = atom_descriptor_to_coords(desc, meta)
        expected = np.array([(a[1], a[2], a[3]) for a in atoms])
        np.testing.assert_allclose(coords, expected, atol=1e-5)

    def test_rotation_equivalence(self) -> None:
        # Rotating the descriptor == recomputing under the rotated frame.
        atoms, bonds = _toy_ligand()
        centroid = np.array([0.1, 0.2, -0.3])
        r0 = random_rotation_matrix(np.random.default_rng(1))
        r = random_rotation_matrix(np.random.default_rng(2))
        desc0, _e, _m = LigandAtomDescriptor().compute(atoms, bonds, (centroid, r0))
        via_recompute, _e2, _m2 = LigandAtomDescriptor().compute(
            atoms, bonds, (centroid, r @ r0)
        )
        via_rotate = rotate_atom_descriptor(desc0, r)
        np.testing.assert_allclose(via_rotate, via_recompute, atol=1e-4)


class TestProteinAtomDescriptor:
    def _toy_pocket(self) -> PocketAtomData:
        # Two residues: ALA (N, CA, C, O, CB) and SER (N, CA, C, O, CB, OG).
        names = ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O", "CB", "OG"]
        elems = ["N", "C", "C", "O", "C", "N", "C", "C", "O", "C", "O"]
        aa = ["A"] * 5 + ["S"] * 6
        chain = ["A"] * 11
        resseq = [10] * 5 + [11] * 6
        coords = np.random.default_rng(3).normal(size=(11, 3)).astype(np.float32)
        ca = coords[[1, 6]]
        return PocketAtomData(
            ca_coords=ca,
            atom_coords=coords,
            atom_elements=elems,
            atom_names=names,
            atom_aa=aa,
            atom_chain=chain,
            atom_resseq=resseq,
            residue_ids=[("A", 10), ("A", 11)],
            pocket_seq="AS",
        )

    def test_shape_source_and_context(self) -> None:
        pocket = self._toy_pocket()
        frame = (pocket.atom_coords.mean(0).astype(np.float64), np.eye(3))
        desc, _meta = ProteinAtomDescriptor().compute(pocket, {}, frame)
        assert desc.shape == (11, ATOM_DESCRIPTOR_DIM)
        assert np.all(desc[:, _F["source"].start] == SOURCE_PROTEIN_IDX)
        # Backbone N/CA/C/O -> backbone; CB/OG -> sidechain.
        bb_sc = desc[:, _F["bb_sc"].start]
        backbone_rows = [0, 1, 2, 3, 5, 6, 7, 8]
        sidechain_rows = [4, 9, 10]
        assert np.all(bb_sc[backbone_rows] == BB_SC_BACKBONE_IDX)
        assert np.all(bb_sc[sidechain_rows] == BB_SC_SIDECHAIN_IDX)
        assert not np.any(bb_sc == BB_SC_NA_IDX)
        # Residue type per atom.
        aa_col = desc[:, _F["aa"].start]
        assert np.all(aa_col[:5] == PROTEIN_AA_TO_IDX["A"])
        assert np.all(aa_col[5:] == PROTEIN_AA_TO_IDX["S"])

    def test_receptor_feats_propagate(self) -> None:
        pocket = self._toy_pocket()
        frame = (pocket.atom_coords.mean(0).astype(np.float64), np.eye(3))
        # Give a non-default chem tuple for the first atom (A/10/N).
        feats = {("A", 10, "N"): (0, 1, 1, 2, 3)}
        desc, _meta = ProteinAtomDescriptor().compute(pocket, feats, frame)
        assert desc[0, _F["charge"].start] == 0
        assert desc[0, _F["hybrid"].start] == 1
        assert desc[0, _F["aromatic"].start] == 1
        assert desc[0, _F["ring"].start] == 2
        assert desc[0, _F["numH"].start] == 3
        # An atom without a lookup entry keeps defaults (aromatic 0, ring NONE=4).
        assert desc[4, _F["aromatic"].start] == 0
