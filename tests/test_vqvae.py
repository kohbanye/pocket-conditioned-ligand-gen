"""Tests for VQ-VAE components: collation, LigandVQVAE and ProteinVQVAE."""

import pytest
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
            # These tests exercise the descriptor path only.
            "coord_loss_enabled": False,
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
        defaults: dict[str, object] = {
            "descriptor_dim": 9,
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


class TestCircleLoss:
    """Unit-circle penalty wired into TransformerVQVAE.forward()."""

    def _make_model(self, descriptor_dim: int = 4) -> LigandVQVAE:
        config = LigandVQVAEConfig(
            descriptor_dim=descriptor_dim,
            hidden_dim=32,
            latent_dim=8,
            codebook_size=16,
            num_transformer_layers=1,
            num_attention_heads=4,
            transformer_feedforward_dim=64,
            max_seq_len=16,
            coord_loss_enabled=False,
        )
        return LigandVQVAE(config)

    def test_circle_loss_present_in_output(self) -> None:
        model = self._make_model()
        model.eval()
        x = torch.randn(2, 5, 4)
        mask = torch.ones(2, 5, dtype=torch.bool)
        with torch.no_grad():
            out = model(x, mask=mask)
        assert "circle_loss" in out
        assert out["circle_loss"].shape == ()
        assert out["circle_loss"].item() >= 0.0

    def test_circle_loss_zero_on_unit_circle(self) -> None:
        """When sin² + cos² == 1 identically, penalty is ~0 despite noisy x_hat."""
        model = self._make_model()
        model.eval()
        # Drive x_hat toward unit-circle sin/cos by feeding unit-circle x with
        # identity-ish normalization: mean=0, std=1 means denorm == input.
        mean = torch.zeros(4)
        std = torch.ones(4)
        model.set_normalization(mean, std)

        x_hat = torch.zeros(2, 5, 4)
        x_hat[..., 2] = 1.0  # sin
        x_hat[..., 3] = 0.0  # cos
        mask = torch.ones(2, 5, dtype=torch.bool)
        loss = model._compute_circle_loss(x_hat, mask)  # noqa: SLF001
        assert loss.item() == 0.0

    def test_circle_loss_positive_off_circle(self) -> None:
        model = self._make_model()
        model.eval()
        model.set_normalization(torch.zeros(4), torch.ones(4))
        x_hat = torch.zeros(2, 5, 4)
        x_hat[..., 2] = 2.0  # sin too big
        x_hat[..., 3] = 2.0  # cos too big
        mask = torch.ones(2, 5, dtype=torch.bool)
        loss = model._compute_circle_loss(x_hat, mask)  # noqa: SLF001
        # (2² + 2² - 1)² = 7² = 49
        assert loss.item() == 49.0


class TestCoordLossRamp:
    """Warmup ramp for coord loss inside VQVAEModule._combine_losses."""

    def _make_module(self, warmup: int) -> VQVAEModule:
        config = VQVAETrainingConfig(
            coord_loss_warmup_epochs=warmup,
            protein=ProteinVQVAEConfig(
                descriptor_dim=12,
                hidden_dim=16,
                latent_dim=8,
                codebook_size=8,
                num_transformer_layers=1,
                num_attention_heads=2,
                transformer_feedforward_dim=32,
                max_seq_len=8,
                coord_loss_enabled=True,
            ),
            ligand=LigandVQVAEConfig(
                hidden_dim=16,
                latent_dim=8,
                codebook_size=8,
                num_transformer_layers=1,
                num_attention_heads=2,
                transformer_feedforward_dim=32,
                max_seq_len=8,
                coord_loss_enabled=True,
            ),
        )
        return VQVAEModule(config)

    def test_no_warmup_yields_ramp_one(self) -> None:
        module = self._make_module(warmup=0)
        # current_epoch defaults to 0 without a Trainer; warmup=0 bypasses it.
        assert module._coord_loss_ramp() == 1.0  # noqa: SLF001

    def test_warmup_linear_ramp(self, monkeypatch: "pytest.MonkeyPatch") -> None:
        module = self._make_module(warmup=10)
        # Patch the read-only LightningModule.current_epoch property at the
        # class level so the ramp formula sees a real epoch number.
        monkeypatch.setattr(
            type(module),
            "current_epoch",
            property(lambda _self: 0),
        )
        # Epoch 0 → (0 + 1) / 10 = 0.1
        assert module._coord_loss_ramp() == pytest.approx(0.1)  # noqa: SLF001

    def test_warmup_plateau_at_one(self, monkeypatch: "pytest.MonkeyPatch") -> None:
        module = self._make_module(warmup=2)
        monkeypatch.setattr(
            type(module),
            "current_epoch",
            property(lambda _self: 5),
        )
        assert module._coord_loss_ramp() == 1.0  # noqa: SLF001


class TestVQVAEModuleBatchFormat:
    """Tests for VQVAEModule with the new batch format."""

    def test_forward_both_models(self) -> None:
        config = VQVAETrainingConfig(
            protein=ProteinVQVAEConfig(
                descriptor_dim=12,
                hidden_dim=32,
                latent_dim=8,
                codebook_size=16,
                num_transformer_layers=1,
                num_attention_heads=4,
                transformer_feedforward_dim=64,
                max_seq_len=32,
                coord_loss_enabled=False,
            ),
            ligand=LigandVQVAEConfig(
                hidden_dim=32,
                latent_dim=8,
                codebook_size=16,
                num_transformer_layers=1,
                num_attention_heads=4,
                transformer_feedforward_dim=64,
                max_seq_len=32,
                coord_loss_enabled=False,
            ),
        )
        module = VQVAEModule(config)
        module.train()

        # Protein forward (sequence batch with mask)
        prot_x = torch.randn(4, 8, 12)
        prot_mask = torch.ones(4, 8, dtype=torch.bool)
        prot_mask[0, 6:] = False
        prot_out = module.protein_vqvae(prot_x, mask=prot_mask)
        assert prot_out["reconstructed"].shape == (4, 8, 12)
        assert (prot_out["indices"][0, 6:] == -1).all()

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
