"""Tests for the multi-head Transformer VQ-VAE."""

from __future__ import annotations

import torch

from src.config import AtomVQVAEConfig
from src.data.descriptors import MoleculeDataset, collate_molecules
from src.tokenizers.descriptor_schema import (
    ATOM_DESCRIPTOR_DIM,
    SOURCE_LIGAND_IDX,
    SOURCE_PROTEIN_IDX,
)
from src.tokenizers.vqvae import TransformerVQVAE


class TestCollation:
    def test_uniform_length(self) -> None:
        mols = [torch.randn(5, ATOM_DESCRIPTOR_DIM) for _ in range(2)]
        padded, mask = collate_molecules(mols)
        assert padded.shape == (2, 5, ATOM_DESCRIPTOR_DIM)
        assert mask.shape == (2, 5)
        assert mask.all()

    def test_variable_length_padding(self) -> None:
        mols = [
            torch.randn(3, ATOM_DESCRIPTOR_DIM),
            torch.randn(7, ATOM_DESCRIPTOR_DIM),
        ]
        padded, mask = collate_molecules(mols)
        assert padded.shape == (2, 7, ATOM_DESCRIPTOR_DIM)
        assert mask[0, :3].all()
        assert not mask[0, 3:].any()
        assert mask[1, :7].all()


class TestMoleculeDataset:
    def test_length_and_getitem(self) -> None:
        mols = [torch.randn(i + 1, ATOM_DESCRIPTOR_DIM) for i in range(3)]
        ds = MoleculeDataset(mols)
        assert len(ds) == 3
        torch.testing.assert_close(ds[1], mols[1])


def _atom_config(**kwargs: object) -> AtomVQVAEConfig:
    defaults: dict = {
        "hidden_dim": 64,
        "latent_dim": 8,
        "codebook_size": 64,
        "num_transformer_layers": 2,
        "num_attention_heads": 4,
        "transformer_feedforward_dim": 128,
        "max_seq_len": 64,
    }
    defaults.update(kwargs)
    return AtomVQVAEConfig(**defaults)


def _make_random_atom_batch(
    b: int, seq_len: int, source: int | None = None
) -> torch.Tensor:
    """Random unified-atom descriptor with valid categorical indices.

    Layout: coord 0-3, source 4, element 5, charge 6, hybrid 7, aromatic 8,
    ring 9, numH 10, aa 11, bb_sc 12, knn_offsets 13-28, knn_elements 29-32.
    """
    x = torch.randn(b, seq_len, ATOM_DESCRIPTOR_DIM)
    if source is None:
        x[..., 4] = torch.randint(0, 2, (b, seq_len)).float()
    else:
        x[..., 4] = float(source)
    x[..., 5] = torch.randint(0, 12, (b, seq_len)).float()
    x[..., 6] = torch.randint(0, 5, (b, seq_len)).float()
    x[..., 7] = torch.randint(0, 5, (b, seq_len)).float()
    x[..., 8] = torch.randint(0, 2, (b, seq_len)).float()
    x[..., 9] = torch.randint(0, 5, (b, seq_len)).float()
    x[..., 10] = torch.randint(0, 5, (b, seq_len)).float()
    x[..., 11] = torch.randint(0, 21, (b, seq_len)).float()
    x[..., 12] = torch.randint(0, 3, (b, seq_len)).float()
    x[..., 29:33] = torch.randint(0, 12, (b, seq_len, 4)).float()
    return x


class TestAtomVQVAEForward:
    def test_forward_heads_shapes(self) -> None:
        model = TransformerVQVAE(_atom_config())
        model.train()
        b, seq_len = 2, 6
        x = _make_random_atom_batch(b, seq_len)
        mask = torch.ones(b, seq_len, dtype=torch.bool)
        out = model(x, mask=mask)
        recon = out["recon_outputs"]
        assert recon["coord"].shape == (b, seq_len, 4)
        assert recon["element"].shape == (b, seq_len, 12)
        assert recon["aa"].shape == (b, seq_len, 21)
        assert recon["bb_sc"].shape == (b, seq_len, 3)
        # ``source`` is input-only: no recon head.
        assert "source" not in recon

    def test_per_source_masking_runs(self) -> None:
        # A batch with a protein-atom sequence and a ligand-atom sequence:
        # aa/bb_sc loss masked to protein, clash to ligand. All heads finite.
        model = TransformerVQVAE(_atom_config())
        model.train()
        prot = _make_random_atom_batch(1, 6, source=SOURCE_PROTEIN_IDX)
        lig = _make_random_atom_batch(1, 6, source=SOURCE_LIGAND_IDX)
        x = torch.cat([prot, lig], dim=0)
        mask = torch.ones(2, 6, dtype=torch.bool)
        out = model(x, mask=mask)
        hl = out["head_losses"]
        for name in ("coord", "element", "charge", "aa", "bb_sc", "clash"):
            assert name in hl, name
            assert torch.isfinite(hl[name]).all()

    def test_clash_zero_for_all_protein_batch(self) -> None:
        model = TransformerVQVAE(_atom_config())
        model.train()
        x = _make_random_atom_batch(2, 6, source=SOURCE_PROTEIN_IDX)
        mask = torch.ones(2, 6, dtype=torch.bool)
        out = model(x, mask=mask)
        # No ligand rows -> no valid clash pairs -> zero penalty.
        assert out["head_losses"]["clash"].item() == 0.0

    def test_encode_decode_to_outputs(self) -> None:
        config = _atom_config()
        model = TransformerVQVAE(config)
        model.eval()
        x = _make_random_atom_batch(1, 10).squeeze(0)
        with torch.no_grad():
            indices = model.encode(x)
            outs = model.decode_to_outputs(indices)
        assert indices.shape == (10,)
        assert (indices >= 0).all()
        assert (indices < config.codebook_size).all()
        assert outs["coord"].shape == (10, 4)
        assert outs["aa"].shape == (10, 21)
