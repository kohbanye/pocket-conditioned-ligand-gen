"""Tests for the descriptor schema (vocab sizes, layouts, masks)."""

from __future__ import annotations

from src.tokenizers.descriptor_schema import (
    K_NEIGHBORS,
    LIGAND_DESCRIPTOR_DIM,
    LIGAND_LAYOUT,
    LIGAND_RECON_HEADS,
    PROTEIN_DESCRIPTOR_DIM,
    PROTEIN_LAYOUT,
    PROTEIN_RECON_HEADS,
    continuous_mask,
    fields_by_name,
)


class TestLayoutSums:
    """Layout absolute offsets must sum to declared descriptor dims."""

    def test_ligand_layout_total(self) -> None:
        assert LIGAND_LAYOUT[-1].end == LIGAND_DESCRIPTOR_DIM

    def test_protein_layout_total(self) -> None:
        assert PROTEIN_LAYOUT[-1].end == PROTEIN_DESCRIPTOR_DIM

    def test_ligand_layout_contiguous(self) -> None:
        cursor = 0
        for spec in LIGAND_LAYOUT:
            assert spec.start == cursor
            cursor += spec.length
        assert cursor == LIGAND_DESCRIPTOR_DIM

    def test_protein_layout_contiguous(self) -> None:
        cursor = 0
        for spec in PROTEIN_LAYOUT:
            assert spec.start == cursor
            cursor += spec.length
        assert cursor == PROTEIN_DESCRIPTOR_DIM


class TestContinuousMask:
    """Categorical slots must be flagged False so normalization skips them."""

    def test_ligand_continuous_mask_length(self) -> None:
        mask = continuous_mask(LIGAND_LAYOUT)
        assert len(mask) == LIGAND_DESCRIPTOR_DIM

    def test_protein_continuous_mask_length(self) -> None:
        mask = continuous_mask(PROTEIN_LAYOUT)
        assert len(mask) == PROTEIN_DESCRIPTOR_DIM

    def test_ligand_categorical_marked_false(self) -> None:
        mask = continuous_mask(LIGAND_LAYOUT)
        f = fields_by_name(LIGAND_LAYOUT)
        # Singleton categoricals at element/charge/hybrid/aromatic/ring/numH.
        for name in ("element", "charge", "hybrid", "aromatic", "ring", "numH"):
            spec = f[name]
            for i in range(spec.start, spec.end):
                assert not mask[i], f"{name}[{i}] should be categorical"
        # KNN element slots are also categorical.
        for i in range(f["knn_elements"].start, f["knn_elements"].end):
            assert not mask[i]

    def test_ligand_continuous_marked_true(self) -> None:
        mask = continuous_mask(LIGAND_LAYOUT)
        f = fields_by_name(LIGAND_LAYOUT)
        for name in ("coord", "knn_offsets"):
            spec = f[name]
            for i in range(spec.start, spec.end):
                assert mask[i], f"{name}[{i}] should be continuous"


class TestKnnDimensions:
    def test_knn_offsets_length_consistent_with_k(self) -> None:
        f = fields_by_name(LIGAND_LAYOUT)
        assert f["knn_offsets"].length == K_NEIGHBORS * 4
        assert f["knn_elements"].length == K_NEIGHBORS

    def test_protein_knn_offsets_length_consistent_with_k(self) -> None:
        f = fields_by_name(PROTEIN_LAYOUT)
        # 3 atoms x 4 spherical per neighbour residue = 12 dims/neighbour.
        assert f["knn_offsets"].length == K_NEIGHBORS * 12
        assert f["knn_aa"].length == K_NEIGHBORS


class TestReconHeads:
    """Decoder heads must match the categorical/continuous declaration in layout."""

    def test_ligand_recon_heads_match_layout_kinds(self) -> None:
        f = fields_by_name(LIGAND_LAYOUT)
        for name, kind, dim in LIGAND_RECON_HEADS:
            assert name in f
            spec = f[name]
            assert spec.kind == kind
            if kind == "continuous":
                assert spec.length == dim
            else:
                assert spec.vocab_size == dim or (name == "aromatic" and dim == 2)

    def test_protein_recon_heads_match_layout_kinds(self) -> None:
        f = fields_by_name(PROTEIN_LAYOUT)
        for name, kind, dim in PROTEIN_RECON_HEADS:
            assert name in f
            spec = f[name]
            assert spec.kind == kind
            if kind == "continuous":
                assert spec.length == dim
