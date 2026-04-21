"""Generic Transformer-based VQ-VAE for structure tokenization.

Used by both protein and ligand tokenizers.  Processes a variable-length
sequence of per-element descriptors through a Transformer encoder, quantizes
per position via an EMA codebook, and decodes with a Transformer decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.tokenizers.codebook import EMACodebook
from src.tokenizers.geometry import (
    canonical_virtual_ref_batched,
    place_atom_batched,
    project_unit_circle,
    sinusoidal_positional_encoding,
    spherical_to_cartesian_batched,
)


@dataclass
class TransformerVQVAEConfig:
    """Base config for Transformer VQ-VAE (used by both protein and ligand)."""

    descriptor_dim: int = 4
    hidden_dim: int = 256
    latent_dim: int = 8
    codebook_size: int = 1024
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    transformer_feedforward_dim: int = 512
    transformer_dropout: float = 0.1
    max_seq_len: int = 256
    # 3D coord-reconstruction loss settings.  When disabled (default) the
    # forward pass returns coord_loss = 0 and behavior matches legacy.
    coord_loss_enabled: bool = False
    coord_loss_kind: str = "ligand"  # "ligand" | "protein_backbone"
    coord_loss_bond_length_min: float = 0.5


class TransformerVQVAE(nn.Module):
    """Transformer-based VQ-VAE for structure tokenization.

    Processes the full sequence, giving each element access to the context
    via self-attention before quantization.  Quantization is per-position
    (each position maps to one codebook entry).
    """

    def __init__(self, config: TransformerVQVAEConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim
        d = config.descriptor_dim
        z = config.latent_dim

        # Encoder
        self.input_norm = nn.LayerNorm(d)
        self.input_proj = nn.Sequential(
            nn.Linear(d, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=config.num_attention_heads,
            dim_feedforward=config.transformer_feedforward_dim,
            dropout=config.transformer_dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_transformer_layers,
        )
        self.latent_proj = nn.Linear(h, z)
        self.latent_norm = nn.LayerNorm(z)

        # Decoder
        self.latent_unproj = nn.Linear(z, h)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=config.num_attention_heads,
            dim_feedforward=config.transformer_feedforward_dim,
            dropout=config.transformer_dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=config.num_transformer_layers + 2,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, d),
        )

        # Positional encoding (buffer, not a parameter)
        self.register_buffer(
            "pos_encoding",
            sinusoidal_positional_encoding(config.max_seq_len, h),
        )

        # Descriptor normalization stats (injected via set_normalization).
        # Stored as buffers so they move with .to(device) and survive checkpoints.
        self.register_buffer("_desc_mean", torch.zeros(d))
        self.register_buffer("_desc_std", torch.ones(d))

        # Codebook
        self.codebook = EMACodebook(
            num_codes=config.codebook_size,
            code_dim=config.latent_dim,
            ema_decay=config.ema_decay,
            commitment_cost=config.commitment_cost,
        )

    def set_normalization(self, mean: Tensor, std: Tensor) -> None:
        """Inject descriptor normalization stats used to denormalize in 3D loss.

        Required before the first forward when ``coord_loss_enabled`` is True.
        """
        target_dtype = self._desc_mean.dtype
        target_device = self._desc_mean.device
        self._desc_mean = mean.to(dtype=target_dtype, device=target_device)
        self._desc_std = std.to(dtype=target_dtype, device=target_device)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        aux: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass with sequence context.

        Args:
            x: Descriptors of shape ``(B, L, descriptor_dim)``.
            mask: Boolean mask ``(B, L)``, ``True`` for real elements.
            aux: Auxiliary per-element metadata used by the 3D coord loss.
                For ``coord_loss_kind='ligand'``, a ``(B, L, 3)`` int tensor of
                ``(parent, angle_ref, dihedral_ref)`` Z-matrix indices (``-1``
                marks special cases). For ``coord_loss_kind='protein_backbone'``,
                a ``(B, L)`` bool tensor marking segment-start residues.
                Ignored when ``coord_loss_enabled`` is False.

        Returns:
            Dict with keys: reconstructed, indices, commitment_loss,
            reconstruction_loss, coord_loss, diagnostics.
        """
        b, seq_len, _ = x.shape

        if mask is None:
            mask = torch.ones(b, seq_len, dtype=torch.bool, device=x.device)

        # Encode
        h = self.input_proj(self.input_norm(x)) + self.pos_encoding[:seq_len]
        h = self.transformer_encoder(h, src_key_padding_mask=~mask)
        z = self.latent_norm(self.latent_proj(h))  # (B, L, latent_dim)

        # Quantize per-position (only real elements)
        # Cast to float32 for codebook (EMA updates need full precision)
        z_real = z[mask].float()  # (N_real, latent_dim)
        quantized_real, indices_real, commitment_loss, codebook_diag = self.codebook(
            z_real,
        )

        # Encoder output diversity: std across tokens, averaged over latent dims.
        # Collapses toward 0 when encoder maps all inputs to the same point.
        z_diversity = z_real.detach().std(dim=0).mean()

        # Scatter quantized vectors back to (B, L, latent_dim)
        quantized = torch.zeros_like(z)
        quantized[mask] = quantized_real.to(z.dtype)

        indices = torch.full(
            (b, seq_len),
            -1,
            dtype=torch.long,
            device=x.device,
        )
        indices[mask] = indices_real

        # Decode
        dec_in = self.latent_unproj(quantized) + self.pos_encoding[:seq_len]
        dec_out = self.transformer_decoder(dec_in, src_key_padding_mask=~mask)
        x_hat = self.output_proj(dec_out)  # (B, L, descriptor_dim)

        # Masked reconstruction loss
        diff_sq = (x - x_hat)[mask].pow(2)  # (N_real, descriptor_dim)
        reconstruction_loss = diff_sq.mean()
        # Per-token mean squared error; max spots pathological single samples
        # even when the batch mean looks fine.
        recon_max = diff_sq.detach().mean(dim=-1).max()

        # 3D coord-reconstruction loss (optional). Compares NeRF-reconstructed
        # coords from x against those from x_hat, both denormalized into Å.
        coord_loss = x.new_zeros(())
        coord_diag: dict[str, Tensor] = {}
        if self.config.coord_loss_enabled:
            coord_loss, coord_diag = self._compute_coord_loss(x, x_hat, mask, aux)

        return {
            "reconstructed": x_hat,
            "indices": indices,
            "commitment_loss": commitment_loss,
            "reconstruction_loss": reconstruction_loss,
            "coord_loss": coord_loss,
            "diagnostics": {
                **codebook_diag,
                **coord_diag,
                "z_diversity": z_diversity,
                "recon_max": recon_max,
            },
        }

    def _compute_coord_loss(
        self,
        x: Tensor,
        x_hat: Tensor,
        mask: Tensor,
        aux: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute 3D coord MSE (Å²) by NeRF-reconstructing from x and x_hat."""
        if aux is None:
            msg = (
                "coord_loss_enabled=True but aux was not provided to forward(). "
                f"Expected refs (B, L, 3) for kind='{self.config.coord_loss_kind}'."
            )
            raise ValueError(msg)

        mean = self._desc_mean.to(x.dtype)
        std = self._desc_std.to(x.dtype)
        x_denorm = x * std + mean
        x_hat_denorm = x_hat * std + mean

        bond_min = self.config.coord_loss_bond_length_min
        kind = self.config.coord_loss_kind

        if kind == "ligand":
            coords_true = _reconstruct_coords_ligand(x_denorm, aux, mask, bond_min)
            coords_pred = _reconstruct_coords_ligand(x_hat_denorm, aux, mask, bond_min)
            # (B, L, 3) per-atom diff → sum xyz → mean over real atoms
            diff_sq = (coords_true - coords_pred).pow(2).sum(dim=-1)  # (B, L)
            real_diff = diff_sq[mask]
        elif kind == "protein_backbone":
            coords_true = _reconstruct_coords_protein(x_denorm, aux, mask, bond_min)
            coords_pred = _reconstruct_coords_protein(x_hat_denorm, aux, mask, bond_min)
            # (B, L, 3, 3) -- sum over (N, CA, C) and xyz -> (B, L); mean over real rows
            diff_sq = (coords_true - coords_pred).pow(2).sum(dim=(-1, -2))
            real_diff = diff_sq[mask]
        else:
            msg = f"Unknown coord_loss_kind: {kind}"
            raise ValueError(msg)

        coord_loss = real_diff.mean() if real_diff.numel() > 0 else x.new_zeros(())

        # Diagnostics: predicted-side stats only (true side is the dataset).
        with torch.no_grad():
            coord_max = real_diff.max() if real_diff.numel() > 0 else x.new_zeros(())
            raw_s = x_hat_denorm[..., 2::4]  # every 4th starting at 2: sin slots
            raw_c = x_hat_denorm[..., 3::4]
            circle_err = (raw_s.pow(2) + raw_c.pow(2)).sqrt().sub(1.0).abs()
            circle_err = circle_err[mask].mean() if mask.any() else x.new_zeros(())
            bond_raw = x_hat_denorm[..., 0::4]  # every 4th starting at 0: bond slots
            clamp_frac = (
                (bond_raw < bond_min)[mask].float().mean()
                if mask.any()
                else x.new_zeros(())
            )

        return coord_loss, {
            "coord_recon_max": coord_max,
            "unit_circle_norm_err": circle_err,
            "bond_clamp_frac": clamp_frac,
        }

    def encode(self, x: Tensor) -> Tensor:
        """Encode descriptors to codebook indices.

        Args:
            x: ``(N, descriptor_dim)`` for a single sequence.

        Returns:
            Codebook indices of shape ``(N,)``.
        """
        x_seq = x.unsqueeze(0)  # (1, N, D)
        h = self.input_proj(self.input_norm(x_seq)) + self.pos_encoding[: x.shape[0]]
        h = self.transformer_encoder(h)
        z = self.latent_norm(self.latent_proj(h)).squeeze(0)  # (N, latent_dim)
        _, indices, _, _ = self.codebook(z)
        return indices

    def decode(self, indices: Tensor) -> Tensor:
        """Decode codebook indices back to descriptors.

        Args:
            indices: ``(N,)`` codebook indices for a single sequence.

        Returns:
            Reconstructed descriptors of shape ``(N, descriptor_dim)``.
        """
        quantized = self.codebook.lookup(indices)  # (N, latent_dim)
        q_seq = quantized.unsqueeze(0)  # (1, N, latent_dim)
        dec_in = self.latent_unproj(q_seq) + self.pos_encoding[: indices.shape[0]]
        dec_out = self.transformer_decoder(dec_in)
        return self.output_proj(dec_out).squeeze(0)  # (N, descriptor_dim)


# ---------------------------------------------------------------------------
# Differentiable Z-matrix → Cartesian reconstruction (used by coord loss).
# Mirrors LigandDescriptor.descriptor_to_coords and
# BackboneZMatrixDescriptor.descriptor_to_backbone_coords numerically.
# ---------------------------------------------------------------------------


def _reconstruct_coords_ligand(
    x_denorm: Tensor,  # (B, L, 4)
    refs: Tensor,  # (B, L, 3) int64
    mask: Tensor,  # (B, L) bool
    bond_length_min: float,
) -> Tensor:
    """Batched differentiable Z-matrix → Cartesian for ligands.

    Returns canonical-frame coords ``(B, L, 3)``; padded rows are zeroed.
    """
    b, seq_len, _ = x_denorm.shape

    d = x_denorm[..., 0].clamp_min(bond_length_min)
    theta = x_denorm[..., 1]
    sin_tau, cos_tau = project_unit_circle(x_denorm[..., 2], x_denorm[..., 3])

    parents = refs[..., 0]
    angle_refs = refs[..., 1]
    dihedral_refs = refs[..., 2]

    is_root = parents == -1
    is_second = (parents >= 0) & (angle_refs == -1)
    is_virtual = (parents >= 0) & (angle_refs >= 0) & (dihedral_refs == -1)

    coords_parts: list[Tensor] = []  # each (B, 3)
    zeros_b3 = x_denorm.new_zeros(b, 3)

    for pos in range(seq_len):
        d_p = d[:, pos]
        theta_p = theta[:, pos]
        s_p = sin_tau[:, pos]
        c_p = cos_tau[:, pos]

        if pos == 0:
            parent_pos = zeros_b3
            angle_ref_pos = zeros_b3
            dihedral_ref_pos = zeros_b3
        else:
            running = torch.stack(coords_parts, dim=1)  # (B, pos, 3)
            limit = pos - 1
            parent_safe = parents[:, pos].clamp(0, limit)
            angle_safe = angle_refs[:, pos].clamp(0, limit)
            dihedral_safe = dihedral_refs[:, pos].clamp(0, limit)
            parent_pos = running.gather(
                1,
                parent_safe.view(b, 1, 1).expand(-1, 1, 3),
            ).squeeze(1)
            angle_ref_pos = running.gather(
                1,
                angle_safe.view(b, 1, 1).expand(-1, 1, 3),
            ).squeeze(1)
            dihedral_ref_pos = running.gather(
                1,
                dihedral_safe.view(b, 1, 1).expand(-1, 1, 3),
            ).squeeze(1)

        # Candidate placements (all four computed; torch.where blends).
        root_xyz = spherical_to_cartesian_batched(d_p, theta_p, s_p, c_p)
        direction = spherical_to_cartesian_batched(
            torch.ones_like(d_p),
            theta_p,
            s_p,
            c_p,
        )
        second_xyz = parent_pos + d_p.unsqueeze(-1) * direction
        virtual_ref_pos = canonical_virtual_ref_batched(angle_ref_pos, parent_pos)
        virtual_xyz = place_atom_batched(
            virtual_ref_pos,
            angle_ref_pos,
            parent_pos,
            d_p,
            theta_p,
            s_p,
            c_p,
        )
        std_xyz = place_atom_batched(
            dihedral_ref_pos,
            angle_ref_pos,
            parent_pos,
            d_p,
            theta_p,
            s_p,
            c_p,
        )

        is_root_p = is_root[:, pos].unsqueeze(-1)
        is_second_p = is_second[:, pos].unsqueeze(-1)
        is_virtual_p = is_virtual[:, pos].unsqueeze(-1)

        placed = torch.where(
            is_root_p,
            root_xyz,
            torch.where(
                is_second_p,
                second_xyz,
                torch.where(is_virtual_p, virtual_xyz, std_xyz),
            ),
        )
        coords_parts.append(placed)

    coords = torch.stack(coords_parts, dim=1)  # (B, L, 3)
    return coords * mask.unsqueeze(-1).to(coords.dtype)


def _reconstruct_coords_protein(
    x_denorm: Tensor,  # (B, L, 12)
    segment_start: Tensor,  # (B, L) bool
    mask: Tensor,  # (B, L) bool
    bond_length_min: float,
) -> Tensor:
    """Batched differentiable backbone Z-matrix → Cartesian.

    Returns canonical-frame coords ``(B, L, 3, 3)`` for (N, CA, C); padded
    rows zeroed. Mirrors :func:`_decode_segment_start` /
    :func:`_decode_continuation` from ``src/tokenizers/protein.py``.
    """
    b, seq_len, _ = x_denorm.shape

    def _group(i0: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        d_ = x_denorm[..., i0].clamp_min(bond_length_min)
        th_ = x_denorm[..., i0 + 1]
        s_, c_ = project_unit_circle(x_denorm[..., i0 + 2], x_denorm[..., i0 + 3])
        return d_, th_, s_, c_

    d_n, th_n, s_n, c_n = _group(0)
    d_ca, th_ca, s_ca, c_ca = _group(4)
    d_c, th_c, s_c, c_c = _group(8)

    residues: list[Tensor] = []  # each (B, 3, 3)

    for pos in range(seq_len):
        # --- segment-start branch (always computable) ----------------------
        n_start = spherical_to_cartesian_batched(
            d_n[:, pos],
            th_n[:, pos],
            s_n[:, pos],
            c_n[:, pos],
        )
        direction = spherical_to_cartesian_batched(
            torch.ones_like(d_ca[:, pos]),
            th_ca[:, pos],
            s_ca[:, pos],
            c_ca[:, pos],
        )
        ca_start = n_start + d_ca[:, pos].unsqueeze(-1) * direction
        virtual = canonical_virtual_ref_batched(n_start, ca_start)
        c_start = place_atom_batched(
            virtual,
            n_start,
            ca_start,
            d_c[:, pos],
            th_c[:, pos],
            s_c[:, pos],
            c_c[:, pos],
        )
        segstart_res = torch.stack([n_start, ca_start, c_start], dim=1)  # (B, 3, 3)

        # --- continuation branch (needs prev residue) ----------------------
        if pos == 0:
            placed = segstart_res  # force segment-start for the first position
        else:
            prev_res = residues[-1]  # (B, 3, 3)
            prev_n = prev_res[:, 0]
            prev_ca = prev_res[:, 1]
            prev_c = prev_res[:, 2]
            n_cont = place_atom_batched(
                prev_n,
                prev_ca,
                prev_c,
                d_n[:, pos],
                th_n[:, pos],
                s_n[:, pos],
                c_n[:, pos],
            )
            ca_cont = place_atom_batched(
                prev_ca,
                prev_c,
                n_cont,
                d_ca[:, pos],
                th_ca[:, pos],
                s_ca[:, pos],
                c_ca[:, pos],
            )
            c_cont = place_atom_batched(
                prev_c,
                n_cont,
                ca_cont,
                d_c[:, pos],
                th_c[:, pos],
                s_c[:, pos],
                c_c[:, pos],
            )
            cont_res = torch.stack([n_cont, ca_cont, c_cont], dim=1)
            placed = torch.where(
                segment_start[:, pos].view(b, 1, 1),
                segstart_res,
                cont_res,
            )

        residues.append(placed)

    coords = torch.stack(residues, dim=1)  # (B, L, 3, 3)
    return coords * mask.view(b, seq_len, 1, 1).to(coords.dtype)
