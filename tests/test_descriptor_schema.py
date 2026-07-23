"""Tests for the descriptor schema (vocab sizes, layouts, masks)."""

from __future__ import annotations

from src.tokenizers.descriptor_schema import (
    ATOM_DESCRIPTOR_DIM,
    ATOM_LAYOUT,
    ATOM_PROTEIN_ONLY_HEADS,
    ATOM_RECON_HEADS,
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


class TestAtomLayout:
    """Unified all-atom descriptor: layout / mask / recon-head consistency."""

    def test_layout_total_and_contiguous(self) -> None:
        cursor = 0
        for spec in ATOM_LAYOUT:
            assert spec.start == cursor
            cursor += spec.length
        assert cursor == ATOM_DESCRIPTOR_DIM
        assert ATOM_LAYOUT[-1].end == ATOM_DESCRIPTOR_DIM

    def test_dim_is_33(self) -> None:
        # coord4 + source + 6 chem + aa + bb_sc + knn16 + knn_elem4 = 33.
        assert ATOM_DESCRIPTOR_DIM == 33

    def test_continuous_mask(self) -> None:
        mask = continuous_mask(ATOM_LAYOUT)
        assert len(mask) == ATOM_DESCRIPTOR_DIM
        f = fields_by_name(ATOM_LAYOUT)
        for name in ("coord", "knn_offsets"):
            spec = f[name]
            for i in range(spec.start, spec.end):
                assert mask[i], f"{name}[{i}] should be continuous"
        cat_names = (
            "source",
            "element",
            "charge",
            "hybrid",
            "aromatic",
            "ring",
            "numH",
            "aa",
            "bb_sc",
            "knn_elements",
        )
        for name in cat_names:
            spec = f[name]
            for i in range(spec.start, spec.end):
                assert not mask[i], f"{name}[{i}] should be categorical"

    def test_knn_dims(self) -> None:
        f = fields_by_name(ATOM_LAYOUT)
        assert f["knn_offsets"].length == K_NEIGHBORS * 4
        assert f["knn_elements"].length == K_NEIGHBORS

    def test_recon_heads_match_layout_kinds(self) -> None:
        f = fields_by_name(ATOM_LAYOUT)
        for name, kind, dim in ATOM_RECON_HEADS:
            assert name in f
            spec = f[name]
            assert spec.kind == kind
            if kind == "continuous":
                assert spec.length == dim
            else:
                assert spec.vocab_size == dim

    def test_source_has_no_recon_head(self) -> None:
        # ``source`` is an input-only conditioning flag.
        head_names = {name for name, _, _ in ATOM_RECON_HEADS}
        assert "source" not in head_names
        assert "knn_elements" not in head_names

    def test_protein_only_heads_present(self) -> None:
        head_names = {name for name, _, _ in ATOM_RECON_HEADS}
        assert head_names >= ATOM_PROTEIN_ONLY_HEADS
        assert {"aa", "bb_sc"} == ATOM_PROTEIN_ONLY_HEADS
