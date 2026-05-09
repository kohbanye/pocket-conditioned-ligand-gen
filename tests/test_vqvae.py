"""Tests for the multi-head Transformer VQ-VAE."""

from __future__ import annotations

import torch

from src.config import LigandVQVAEConfig, ProteinVQVAEConfig
from src.data.descriptors import MoleculeDataset, collate_molecules
from src.tokenizers.descriptor_schema import (
    LIGAND_DESCRIPTOR_DIM,
    PROTEIN_DESCRIPTOR_DIM,
)
from src.tokenizers.vqvae import TransformerVQVAE


def _ligand_config(**kwargs: object) -> LigandVQVAEConfig:
    defaults: dict = {
        "hidden_dim": 64,
        "latent_dim": 8,
        "codebook_size": 32,
        "num_transformer_layers": 2,
        "num_attention_heads": 4,
        "transformer_feedforward_dim": 128,
        "max_seq_len": 64,
    }
    defaults.update(kwargs)
    return LigandVQVAEConfig(**defaults)


def _protein_config(**kwargs: object) -> ProteinVQVAEConfig:
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
    return ProteinVQVAEConfig(**defaults)


def _make_random_ligand_batch(b: int, seq_len: int) -> torch.Tensor:
    """Random descriptor where categorical slots hold valid integer indices."""
    x = torch.randn(b, seq_len, LIGAND_DESCRIPTOR_DIM)
    # element idx ∈ [0, 12), charge ∈ [0, 5), hybrid ∈ [0, 5),
    # aromatic ∈ {0, 1}, ring ∈ [0, 5), numH ∈ [0, 5)
    x[..., 4] = torch.randint(0, 12, (b, seq_len)).float()
    x[..., 5] = torch.randint(0, 5, (b, seq_len)).float()
    x[..., 6] = torch.randint(0, 5, (b, seq_len)).float()
    x[..., 7] = torch.randint(0, 2, (b, seq_len)).float()
    x[..., 8] = torch.randint(0, 5, (b, seq_len)).float()
    x[..., 9] = torch.randint(0, 5, (b, seq_len)).float()
    # KNN element idx
    x[..., 26:30] = torch.randint(0, 12, (b, seq_len, 4)).float()
    return x


def _make_random_protein_batch(b: int, seq_len: int) -> torch.Tensor:
    x = torch.randn(b, seq_len, PROTEIN_DESCRIPTOR_DIM)
    x[..., 12] = torch.randint(0, 21, (b, seq_len)).float()
    x[..., 61:65] = torch.randint(0, 21, (b, seq_len, 4)).float()
    return x


class TestCollation:
    def test_uniform_length(self) -> None:
        mols = [torch.randn(5, LIGAND_DESCRIPTOR_DIM) for _ in range(2)]
        padded, mask = collate_molecules(mols)
        assert padded.shape == (2, 5, LIGAND_DESCRIPTOR_DIM)
        assert mask.shape == (2, 5)
        assert mask.all()

    def test_variable_length_padding(self) -> None:
        mols = [
            torch.randn(3, LIGAND_DESCRIPTOR_DIM),
            torch.randn(7, LIGAND_DESCRIPTOR_DIM),
        ]
        padded, mask = collate_molecules(mols)
        assert padded.shape == (2, 7, LIGAND_DESCRIPTOR_DIM)
        assert mask[0, :3].all()
        assert not mask[0, 3:].any()
        assert mask[1, :7].all()


class TestMoleculeDataset:
    def test_length_and_getitem(self) -> None:
        mols = [torch.randn(i + 1, LIGAND_DESCRIPTOR_DIM) for i in range(3)]
        ds = MoleculeDataset(mols)
        assert len(ds) == 3
        torch.testing.assert_close(ds[1], mols[1])


class TestLigandVQVAEForward:
    def test_forward_shapes(self) -> None:
        config = _ligand_config()
        model = TransformerVQVAE(config)
        model.train()

        b, seq_len = 2, 6
        x = _make_random_ligand_batch(b, seq_len)
        mask = torch.ones(b, seq_len, dtype=torch.bool)

        out = model(x, mask=mask)
        assert out["indices"].shape == (b, seq_len)
        assert out["reconstruction_loss"].shape == ()
        assert out["commitment_loss"].shape == ()

        # Heads
        recon = out["recon_outputs"]
        assert recon["coord"].shape == (b, seq_len, 4)
        assert recon["element"].shape == (b, seq_len, 12)
        assert recon["charge"].shape == (b, seq_len, 5)
        assert recon["aromatic"].shape == (b, seq_len, 2)

    def test_forward_with_padding(self) -> None:
        config = _ligand_config()
        model = TransformerVQVAE(config)
        model.eval()

        b, max_len = 2, 8
        x = _make_random_ligand_batch(b, max_len)
        mask = torch.zeros(b, max_len, dtype=torch.bool)
        mask[0, :5] = True
        mask[1, :max_len] = True

        out = model(x, mask=mask)
        assert (out["indices"][0, 5:] == -1).all()
        assert (out["indices"][1] >= 0).all()

    def test_encode_decode_to_outputs(self) -> None:
        config = _ligand_config()
        model = TransformerVQVAE(config)
        model.eval()

        x = _make_random_ligand_batch(1, 10).squeeze(0)
        with torch.no_grad():
            indices = model.encode(x)
            outs = model.decode_to_outputs(indices)
        assert indices.shape == (10,)
        assert (indices >= 0).all()
        assert (indices < config.codebook_size).all()
        assert outs["coord"].shape == (10, 4)
        assert outs["element"].shape == (10, 12)

    def test_gradient_flow(self) -> None:
        config = _ligand_config()
        model = TransformerVQVAE(config)
        model.train()

        x = _make_random_ligand_batch(2, 5)
        mask = torch.ones(2, 5, dtype=torch.bool)
        out = model(x, mask=mask)
        loss = out["reconstruction_loss"] + out["commitment_loss"]
        loss.backward()
        assert model.input_proj[0].weight.grad is not None
        assert model.recon_head_modules["coord"].weight.grad is not None
        assert model.recon_head_modules["element"].weight.grad is not None


class TestProteinVQVAEForward:
    def test_forward_shapes(self) -> None:
        config = _protein_config()
        model = TransformerVQVAE(config)
        model.train()

        b, seq_len = 2, 8
        x = _make_random_protein_batch(b, seq_len)
        mask = torch.ones(b, seq_len, dtype=torch.bool)
        out = model(x, mask=mask)
        recon = out["recon_outputs"]
        assert recon["coord"].shape == (b, seq_len, 12)
        assert recon["aa"].shape == (b, seq_len, 21)

    def test_per_head_losses_present(self) -> None:
        config = _protein_config()
        model = TransformerVQVAE(config)
        model.train()
        x = _make_random_protein_batch(1, 4)
        mask = torch.ones(1, 4, dtype=torch.bool)
        out = model(x, mask=mask)
        head_losses = out["head_losses"]
        assert "coord" in head_losses
        assert "aa" in head_losses
        assert head_losses["coord"].shape == ()
        assert head_losses["aa"].shape == ()
