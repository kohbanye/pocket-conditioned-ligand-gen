"""Tests for the descriptor schema (vocab sizes, layouts, masks)."""

from __future__ import annotations

from src.tokenizers.descriptor_schema import (
    ATOM_DESCRIPTOR_DIM,
    ATOM_LAYOUT,
    ATOM_PROTEIN_ONLY_HEADS,
    ATOM_RECON_HEADS,
    K_NEIGHBORS,
    continuous_mask,
    fields_by_name,
)


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
