"""Two single-modality VQ-VAEs presented as one unified encoder (ablation).

The tokenizer ablation compares the JOINT all-atom VQ-VAE (one codebook trained
on pooled protein+ligand atoms) against SEPARATELY-trained protein-only and
ligand-only VQ-VAEs. Downstream models want a single ``encode_batch`` that maps a
complex into one contiguous code space, exactly like the joint tokenizer.

:class:`SeparateVQVAE` wraps the two single-modality models: for a batch of atoms
of ONE modality (protein-pocket block or ligand block, as the tokenize scripts
already feed them) it (1) normalizes with that modality's own stats, (2) encodes
with that modality's VQ, and (3) places ligand codes above the protein range so
the combined ids are disjoint. It exposes ``encode_batch`` (and a ``.vqvae``
self-alias) so the existing ``module.vqvae.encode_batch`` call sites work with a
drop-in swap and ``AtomLMVocab(codebook_size=protein_codes + ligand_codes)``.

Because ``encode_batch`` normalizes internally, callers must pass RAW descriptors
(feed identity mean/std externally).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.config import AtomVQVAETrainingConfig
from src.model.vqvae_module import AtomVQVAEModule
from src.tokenizers.descriptor_schema import ATOM_LAYOUT, fields_by_name

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

_SOURCE_COL = fields_by_name(ATOM_LAYOUT)["source"].start
_SOURCE_PROTEIN = 0
_SOURCE_LIGAND = 1


def _load_vq(ckpt: Path, codebook_size: int, device: torch.device) -> AtomVQVAEModule:
    cfg = AtomVQVAETrainingConfig()
    cfg.atom.codebook_size = codebook_size
    module = AtomVQVAEModule.load_from_checkpoint(ckpt, config=cfg, map_location=device)
    module.eval().to(device)
    return module


class SeparateVQVAE:
    """Protein-only + ligand-only VQ-VAEs unified into one combined code space."""

    def __init__(  # noqa: PLR0913
        self,
        protein_module: AtomVQVAEModule,
        protein_mean: Tensor,
        protein_std: Tensor,
        ligand_module: AtomVQVAEModule,
        ligand_mean: Tensor,
        ligand_std: Tensor,
        protein_codebook_size: int,
        device: torch.device,
    ) -> None:
        self.protein = protein_module.vqvae
        self.ligand = ligand_module.vqvae
        self.protein.set_normalization(protein_mean, protein_std)
        self.ligand.set_normalization(ligand_mean, ligand_std)
        self._pmean = protein_mean.to(device).float()
        self._pstd = protein_std.to(device).float()
        self._lmean = ligand_mean.to(device).float()
        self._lstd = ligand_std.to(device).float()
        self.protein_codebook_size = protein_codebook_size
        self.device = device

    @classmethod
    def from_checkpoints(  # noqa: PLR0913
        cls,
        protein_ckpt: Path,
        protein_norm: Path,
        ligand_ckpt: Path,
        ligand_norm: Path,
        device: torch.device,
        codebook_size: int = 8192,
    ) -> SeparateVQVAE:
        """Load both single-modality VQ-VAEs and their normalization stats."""
        pm = _load_vq(protein_ckpt, codebook_size, device)
        lm = _load_vq(ligand_ckpt, codebook_size, device)
        pn = torch.load(protein_norm, weights_only=False)
        ln = torch.load(ligand_norm, weights_only=False)
        return cls(
            pm,
            pn["atom_mean"],
            pn["atom_std"],
            lm,
            ln["atom_mean"],
            ln["atom_std"],
            codebook_size,
            device,
        )

    @property
    def vqvae(self) -> SeparateVQVAE:
        """Self-alias so ``module.vqvae.encode_batch`` call sites work unchanged."""
        return self

    def set_normalization(self, mean: Tensor, std: Tensor) -> None:  # noqa: ARG002
        """No-op drop-in: each sub-VQ already holds its own modality stats.

        The tokenize scripts call ``module.vqvae.set_normalization(...)`` with the
        (identity) external stats; the real per-modality normalization happens
        inside :meth:`encode_batch`.
        """
        return

    @property
    def codebook_size(self) -> int:
        """Combined code count (protein codes then ligand codes)."""
        return 2 * self.protein_codebook_size

    @property
    def ligand_norm_stats(self) -> dict[str, Tensor]:
        """Ligand-VQ normalization stats keyed like the all-atom stats file.

        Generation's ligand-coord denorm reads ``atom_mean`` / ``atom_std`` from a
        stats dict; expose the ligand modality's stats (the same ones already
        applied to ``self.ligand`` via ``set_normalization``) under those keys so
        the decode path can denormalize without re-loading the stats file.
        """
        return {"atom_mean": self._lmean, "atom_std": self._lstd}

    @torch.no_grad()
    def encode_batch(self, x: Tensor, mask: Tensor) -> Tensor:
        """Encode a single-modality RAW-descriptor batch to combined-space codes.

        ``x`` is ``(B, L, descriptor_dim)`` UN-normalized (the source column picks
        the modality). Ligand codes are shifted up by ``protein_codebook_size`` so
        protein occupies ``[0, Pc)`` and ligand ``[Pc, 2*Pc)``. Returns ``(B, L)``
        long indices with ``-1`` at padded positions.
        """
        source = x[..., _SOURCE_COL].long()
        real = source[mask]
        if real.numel() == 0:
            return torch.full(mask.shape, -1, dtype=torch.long, device=x.device)
        uniq = torch.unique(real).tolist()
        if len(uniq) != 1:
            msg = f"encode_batch expects a single-modality batch, got sources {uniq}"
            raise ValueError(msg)
        if uniq[0] == _SOURCE_PROTEIN:
            xn = (x - self._pmean) / self._pstd
            idx = self.protein.encode_batch(xn, mask)
            return idx.masked_fill(~mask, -1)
        xn = (x - self._lmean) / self._lstd
        idx = self.ligand.encode_batch(xn, mask)
        shifted = idx + self.protein_codebook_size
        return shifted.masked_fill(~mask, -1)

    @torch.no_grad()
    def decode_to_outputs(self, indices: Tensor) -> dict[str, Tensor]:
        """Decode COMBINED-space ligand codes to the ligand VQ's recon outputs.

        ``indices`` are ``(N,)`` combined-space ids from the LM's ligand block,
        i.e. in ``[protein_codebook_size, 2 * protein_codebook_size)``. They are
        mapped back to the ligand VQ's own 0-based range
        (``id - protein_codebook_size``) and decoded by the ligand-only VQ.

        Returns the same per-head output dict as
        :meth:`TransformerVQVAE.decode_to_outputs`, so a caller can hold either
        tokenizer behind one name (the caller converts categorical logits via
        argmax and spherical coords to Cartesian).
        """
        ligand_indices = indices - self.protein_codebook_size
        return self.ligand.decode_to_outputs(ligand_indices)
