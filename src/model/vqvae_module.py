"""Lightning module for joint protein + ligand VQ-VAE training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch
from torch import Tensor

from src.tokenizers.ligand import LigandVQVAE
from src.tokenizers.protein import ProteinStructureVQVAE

if TYPE_CHECKING:
    from src.config import VQVAETrainingConfig


class VQVAEModule(L.LightningModule):
    """Joint training of protein and ligand structure VQ-VAEs.

    Trains both VQ-VAEs simultaneously with combined loss.
    Logs per-model reconstruction loss, commitment loss, and
    codebook utilization metrics.
    """

    def __init__(self, config: VQVAETrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.protein_vqvae = ProteinStructureVQVAE(config.protein)
        self.ligand_vqvae = LigandVQVAE(config.ligand)

        # Track codebook usage for utilization metrics
        self.automatic_optimization = False

    def training_step(
        self,
        batch: dict[str, list[Tensor] | tuple[Tensor, Tensor]],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        opt = self.optimizers()
        if not isinstance(opt, torch.optim.Optimizer):  # pragma: no cover
            msg = "Expected a single optimizer"
            raise TypeError(msg)

        total_loss = torch.tensor(0.0, device=self.device)

        # Protein VQ-VAE (flat residue batching)
        if "protein" in batch:
            (prot_x,) = batch["protein"]
            prot_out = self.protein_vqvae(prot_x)
            prot_loss = prot_out["reconstruction_loss"] + prot_out["commitment_loss"]
            total_loss = total_loss + prot_loss
            self.log("train/protein_recon", prot_out["reconstruction_loss"])
            self.log("train/protein_commit", prot_out["commitment_loss"])
            self._log_utilization("train/protein", prot_out["indices"])

        # Ligand VQ-VAE (molecule-level batching with mask)
        if "ligand" in batch:
            lig_x, lig_mask = batch["ligand"]
            lig_out = self.ligand_vqvae(lig_x, mask=lig_mask)
            lig_loss = lig_out["reconstruction_loss"] + lig_out["commitment_loss"]
            total_loss = total_loss + lig_loss
            self.log("train/ligand_recon", lig_out["reconstruction_loss"])
            self.log("train/ligand_commit", lig_out["commitment_loss"])
            real_indices = lig_out["indices"][lig_mask]
            self._log_utilization("train/ligand", real_indices)

        self.log("train/total_loss", total_loss, prog_bar=True)

        opt.zero_grad()
        self.manual_backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()

    def validation_step(
        self,
        batch: dict[str, list[Tensor] | tuple[Tensor, Tensor]],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        if "protein" in batch:
            (prot_x,) = batch["protein"]
            prot_out = self.protein_vqvae(prot_x)
            self.log(
                "val/protein_recon",
                prot_out["reconstruction_loss"],
                sync_dist=True,
            )
            self.log("val/protein_commit", prot_out["commitment_loss"], sync_dist=True)
            self._log_utilization("val/protein", prot_out["indices"])

        if "ligand" in batch:
            lig_x, lig_mask = batch["ligand"]
            lig_out = self.ligand_vqvae(lig_x, mask=lig_mask)
            self.log("val/ligand_recon", lig_out["reconstruction_loss"], sync_dist=True)
            self.log("val/ligand_commit", lig_out["commitment_loss"], sync_dist=True)
            real_indices = lig_out["indices"][lig_mask]
            self._log_utilization("val/ligand", real_indices)

    def test_step(
        self,
        batch: dict[str, list[Tensor] | tuple[Tensor, Tensor]],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        if "protein" in batch:
            (prot_x,) = batch["protein"]
            prot_out = self.protein_vqvae(prot_x)
            self.log(
                "test/protein_recon",
                prot_out["reconstruction_loss"],
                sync_dist=True,
            )
            self.log("test/protein_commit", prot_out["commitment_loss"], sync_dist=True)
            self._log_utilization("test/protein", prot_out["indices"])

        if "ligand" in batch:
            lig_x, lig_mask = batch["ligand"]
            lig_out = self.ligand_vqvae(lig_x, mask=lig_mask)
            self.log(
                "test/ligand_recon",
                lig_out["reconstruction_loss"],
                sync_dist=True,
            )
            self.log(
                "test/ligand_commit",
                lig_out["commitment_loss"],
                sync_dist=True,
            )
            real_indices = lig_out["indices"][lig_mask]
            self._log_utilization("test/ligand", real_indices)

    def _log_utilization(self, prefix: str, indices: Tensor) -> None:
        """Log codebook utilization and perplexity."""
        unique_codes = indices.unique().numel()
        total_codes = (
            self.config.protein.codebook_size
            if "protein" in prefix
            else self.config.ligand.codebook_size
        )
        utilization = unique_codes / total_codes
        self.log(f"{prefix}_codebook_util", utilization)

        # Perplexity: exp(entropy of code usage distribution)
        counts = torch.bincount(indices, minlength=total_codes).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum()
        self.log(f"{prefix}_perplexity", entropy.exp())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
