"""Tests for VQ-VAE components: collation, LigandVQVAE and ProteinVQVAE."""

import torch

from src.config import LigandVQVAEConfig, ProteinVQVAEConfig, VQVAETrainingConfig
from src.data.descriptors import MoleculeDataset, collate_molecules
from src.model.vqvae_module import VQVAEModule
from src.tokenizers.ligand import LigandVQVAE
from src.tokenizers.protein import ProteinStructureVQVAE


class TestCollateMolecules:
    """Tests for the collate_molecules function."""

    def test_uniform_length(self) -> None:
        mols = [torch.randn(5, 4), torch.randn(5, 4)]
        padded, mask = collate_molecules(mols)
        assert padded.shape == (2, 5, 4)
        assert mask.shape == (2, 5)
        assert mask.all()

    def test_variable_length_padding(self) -> None:
        mols = [torch.randn(3, 4), torch.randn(7, 4), torch.randn(5, 4)]
        padded, mask = collate_molecules(mols)
        assert padded.shape == (3, 7, 4)
        assert mask.shape == (3, 7)
        # Check mask correctness
        assert mask[0, :3].all()
        assert not mask[0, 3:].any()
        assert mask[1, :7].all()
        assert mask[2, :5].all()
        assert not mask[2, 5:].any()

    def test_padded_values_are_zero(self) -> None:
        mols = [torch.randn(2, 4), torch.randn(5, 4)]
        padded, _mask = collate_molecules(mols)
        assert (padded[0, 2:] == 0).all()

    def test_original_values_preserved(self) -> None:
        mol = torch.randn(4, 4)
        padded, _mask = collate_molecules([mol, torch.randn(6, 4)])
        torch.testing.assert_close(padded[0, :4], mol)

    def test_single_molecule(self) -> None:
        mol = torch.randn(10, 4)
        padded, mask = collate_molecules([mol])
        assert padded.shape == (1, 10, 4)
        assert mask.all()


class TestMoleculeDataset:
    """Tests for MoleculeDataset."""

    def test_length(self) -> None:
        mols = [torch.randn(i + 1, 4) for i in range(5)]
        ds = MoleculeDataset(mols)
        assert len(ds) == 5

    def test_getitem(self) -> None:
        mols = [torch.randn(3, 4), torch.randn(7, 4)]
        ds = MoleculeDataset(mols)
        torch.testing.assert_close(ds[0], mols[0])
        torch.testing.assert_close(ds[1], mols[1])


class TestTransformerLigandVQVAE:
    """Tests for the Transformer-based LigandVQVAE."""

    def _make_config(self, **kwargs: object) -> LigandVQVAEConfig:
        defaults = {
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

    def test_forward_shapes(self) -> None:
        config = self._make_config()
        model = LigandVQVAE(config)
        model.train()

        b, seq_len = 4, 10
        x = torch.randn(b, seq_len, 4)
        mask = torch.ones(b, seq_len, dtype=torch.bool)

        out = model(x, mask=mask)
        assert out["reconstructed"].shape == (b, seq_len, 4)
        assert out["indices"].shape == (b, seq_len)
        assert out["reconstruction_loss"].shape == ()
        assert out["commitment_loss"].shape == ()

    def test_forward_with_padding(self) -> None:
        config = self._make_config()
        model = LigandVQVAE(config)
        model.train()

        b, max_len = 3, 12
        x = torch.randn(b, max_len, 4)
        mask = torch.zeros(b, max_len, dtype=torch.bool)
        mask[0, :5] = True
        mask[1, :12] = True
        mask[2, :8] = True

        out = model(x, mask=mask)
        assert out["reconstructed"].shape == (b, max_len, 4)
        # Padded positions should have index -1
        assert (out["indices"][0, 5:] == -1).all()
        assert (out["indices"][2, 8:] == -1).all()

    def test_forward_no_mask(self) -> None:
        config = self._make_config()
        model = LigandVQVAE(config)
        model.eval()

        x = torch.randn(2, 6, 4)
        out = model(x)
        assert out["reconstructed"].shape == (2, 6, 4)
        assert (out["indices"] >= 0).all()

    def test_encode_single_molecule(self) -> None:
        config = self._make_config()
        model = LigandVQVAE(config)
        model.eval()

        x = torch.randn(15, 4)
        with torch.no_grad():
            indices = model.encode(x)
        assert indices.shape == (15,)
        assert indices.dtype == torch.long
        assert (indices >= 0).all()
        assert (indices < config.codebook_size).all()

    def test_decode_single_molecule(self) -> None:
        config = self._make_config()
        model = LigandVQVAE(config)
        model.eval()

        indices = torch.randint(0, config.codebook_size, (10,))
        with torch.no_grad():
            reconstructed = model.decode(indices)
        assert reconstructed.shape == (10, 4)

    def test_encode_decode_roundtrip(self) -> None:
        config = self._make_config()
        model = LigandVQVAE(config)
        model.eval()

        x = torch.randn(8, 4)
        with torch.no_grad():
            indices = model.encode(x)
            reconstructed = model.decode(indices)
        assert reconstructed.shape == x.shape

    def test_gradient_flow(self) -> None:
        config = self._make_config()
        model = LigandVQVAE(config)
        model.train()

        x = torch.randn(2, 5, 4)
        mask = torch.ones(2, 5, dtype=torch.bool)
        out = model(x, mask=mask)
        loss = out["reconstruction_loss"] + out["commitment_loss"]
        loss.backward()

        # Check that encoder and decoder gradients exist
        assert model.input_proj[0].weight.grad is not None
        assert model.output_proj[-1].weight.grad is not None


class TestProteinVQVAE:
    """Tests for the deepened ProteinStructureVQVAE."""

    def _make_config(self, **kwargs: object) -> ProteinVQVAEConfig:
        defaults = {
            "hidden_dim": 64,
            "latent_dim": 16,
            "codebook_size": 64,
        }
        defaults.update(kwargs)
        return ProteinVQVAEConfig(**defaults)

    def test_forward_shapes(self) -> None:
        config = self._make_config()
        model = ProteinStructureVQVAE(config)
        model.train()

        x = torch.randn(32, 9)
        out = model(x)
        assert out["reconstructed"].shape == (32, 9)
        assert out["indices"].shape == (32,)

    def test_encoder_depth(self) -> None:
        config = self._make_config()
        model = ProteinStructureVQVAE(config)
        # Encoder should have 4 Linear layers (7 modules with ReLUs)
        linear_layers = [m for m in model.encoder if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 4

    def test_decoder_depth(self) -> None:
        config = self._make_config()
        model = ProteinStructureVQVAE(config)
        # Decoder should have 3 Linear layers
        linear_layers = [m for m in model.decoder if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 3


class TestVQVAEModuleBatchFormat:
    """Tests for VQVAEModule with the new batch format."""

    def test_forward_both_models(self) -> None:
        config = VQVAETrainingConfig(
            protein=ProteinVQVAEConfig(
                hidden_dim=32,
                latent_dim=8,
                codebook_size=16,
            ),
            ligand=LigandVQVAEConfig(
                hidden_dim=32,
                latent_dim=8,
                codebook_size=16,
                num_transformer_layers=1,
                num_attention_heads=4,
                transformer_feedforward_dim=64,
                max_seq_len=32,
            ),
        )
        module = VQVAEModule(config)
        module.train()

        # Protein forward (flat batch)
        prot_x = torch.randn(16, 9)
        prot_out = module.protein_vqvae(prot_x)
        assert prot_out["reconstructed"].shape == (16, 9)

        # Ligand forward (sequence batch with mask)
        lig_x = torch.randn(4, 10, 4)
        lig_mask = torch.ones(4, 10, dtype=torch.bool)
        lig_mask[0, 7:] = False
        lig_out = module.ligand_vqvae(lig_x, mask=lig_mask)
        assert lig_out["reconstructed"].shape == (4, 10, 4)
        assert (lig_out["indices"][0, 7:] == -1).all()

        # Both losses are scalar and backpropagable
        total_loss = (
            prot_out["reconstruction_loss"]
            + prot_out["commitment_loss"]
            + lig_out["reconstruction_loss"]
            + lig_out["commitment_loss"]
        )
        total_loss.backward()
