"""Transformer VQ-VAE with mixed continuous + categorical descriptors.

The encoder consumes a single (B, L, D) descriptor tensor in which:
- continuous slots hold normalized real values (spherical coords + KNN
  offsets);
- categorical slots hold integer indices stored as float (cast back to long
  before embedding lookup).

A single codebook quantises the encoder's latent. The decoder is multi-head
and reconstructs both the continuous coords (regression) and the categorical
features (classification logits). This lets one VQ token per atom carry
position + element + atom features; the AR transformer downstream consumes
codebook indices directly without needing element-prefixed tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.tokenizers.codebook import EMACodebook
from src.tokenizers.descriptor_schema import (
    LIGAND_LAYOUT,
    LIGAND_RECON_HEADS,
    PROTEIN_LAYOUT,
    PROTEIN_RECON_HEADS,
    FieldSpec,
    fields_by_name,
)
from src.tokenizers.geometry import (
    project_unit_circle,
    sinusoidal_positional_encoding,
    spherical_to_cartesian_batched,
)


@dataclass
class TransformerVQVAEConfig:
    """Base config for Transformer VQ-VAE (used by both protein and ligand)."""

    descriptor_dim: int = 30  # overridden by domain configs
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
    # "ligand" or "protein" — picks the descriptor layout / recon heads.
    domain: str = "ligand"
    # Per-categorical embedding dim. Categorical slots map to learned vectors
    # of this size before being concatenated with continuous slots.
    categorical_embed_dim: int = 8


class TransformerVQVAE(nn.Module):
    """Transformer VQ-VAE with mixed-feature input + multi-head reconstruction."""

    def __init__(self, config: TransformerVQVAEConfig) -> None:
        super().__init__()
        self.config = config
        if config.domain not in ("ligand", "protein"):
            msg = f"Unknown domain: {config.domain!r}"
            raise ValueError(msg)

        self.layout: list[FieldSpec] = (
            LIGAND_LAYOUT if config.domain == "ligand" else PROTEIN_LAYOUT
        )
        self.recon_heads: list[tuple[str, str, int]] = (
            LIGAND_RECON_HEADS if config.domain == "ligand" else PROTEIN_RECON_HEADS
        )

        # ---- Embeddings for categorical input slots --------------------
        d_cat = config.categorical_embed_dim
        self.cat_embeddings = nn.ModuleDict()
        cat_input_dim = 0
        cont_input_dim = 0
        for spec in self.layout:
            if spec.kind == "categorical":
                # Each scalar position uses the same embedding table; KNN
                # element / aa slots reuse the singleton's table so that "C"
                # has one vector everywhere it appears.
                key = self._embedding_key(spec.name)
                if key not in self.cat_embeddings:
                    self.cat_embeddings[key] = nn.Embedding(spec.vocab_size, d_cat)
                cat_input_dim += spec.length * d_cat
            else:
                cont_input_dim += spec.length

        encoder_input_dim = cont_input_dim + cat_input_dim
        h = config.hidden_dim

        # ---- Encoder ---------------------------------------------------
        self.input_norm = nn.LayerNorm(encoder_input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(encoder_input_dim, h),
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
        self.latent_proj = nn.Linear(h, config.latent_dim)
        self.latent_norm = nn.LayerNorm(config.latent_dim)

        # ---- Decoder ---------------------------------------------------
        self.latent_unproj = nn.Linear(config.latent_dim, h)
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
        self.decoder_trunk = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.recon_head_modules = nn.ModuleDict()
        for name, kind, dim in self.recon_heads:
            out_dim = dim  # continuous: dim; categorical: vocab size
            self.recon_head_modules[name] = nn.Linear(h, out_dim)
            _ = kind  # explicitly note that kind is consumed at loss time

        # ---- Positional encoding --------------------------------------
        self.register_buffer(
            "pos_encoding",
            sinusoidal_positional_encoding(config.max_seq_len, h),
        )

        # ---- Continuous-slot normalization stats (set externally) -----
        # Only continuous slots carry meaningful stats; categorical slots
        # are normalized with mean=0, std=1 so they pass through unchanged.
        self.register_buffer("_desc_mean", torch.zeros(config.descriptor_dim))
        self.register_buffer("_desc_std", torch.ones(config.descriptor_dim))

        # ---- Codebook --------------------------------------------------
        self.codebook = EMACodebook(
            num_codes=config.codebook_size,
            code_dim=config.latent_dim,
            ema_decay=config.ema_decay,
            commitment_cost=config.commitment_cost,
        )

    # ------------------------------------------------------------------
    # Helper: shared embedding tables for "element"/"aa" across slots
    # ------------------------------------------------------------------
    @staticmethod
    def _embedding_key(field_name: str) -> str:
        # KNN slots share their singleton's embedding so the same atom type
        # has one canonical vector regardless of where it appears.
        if field_name == "knn_elements":
            return "element"
        if field_name == "knn_aa":
            return "aa"
        return field_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_normalization(self, mean: Tensor, std: Tensor) -> None:
        """Inject continuous-slot normalization stats from the DataModule."""
        target_dtype = self._desc_mean.dtype
        target_device = self._desc_mean.device
        self._desc_mean = mean.to(dtype=target_dtype, device=target_device)
        self._desc_std = std.to(dtype=target_dtype, device=target_device)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: ``(B, L, descriptor_dim)`` raw (already-normalized) descriptors.
            mask: ``(B, L)`` bool, ``True`` for real elements.

        Returns:
            dict with keys: indices, commitment_loss, reconstruction_loss,
            recon_outputs (per-head dict), diagnostics.
        """
        b, seq_len, _ = x.shape
        if mask is None:
            mask = torch.ones(b, seq_len, dtype=torch.bool, device=x.device)

        # 1. Encoder input embedding (continuous concat with categorical embeds).
        h_in = self._embed_descriptor(x)
        h = self.input_proj(self.input_norm(h_in)) + self.pos_encoding[:seq_len]
        h = self.transformer_encoder(h, src_key_padding_mask=~mask)
        z = self.latent_norm(self.latent_proj(h))  # (B, L, latent_dim)

        # 2. Quantize per-position over real elements only.
        z_real = z[mask].float()
        quantized_real, indices_real, commitment_loss, codebook_diag = self.codebook(
            z_real,
        )
        z_diversity = z_real.detach().std(dim=0).mean()

        quantized = torch.zeros_like(z)
        quantized[mask] = quantized_real.to(z.dtype)

        indices = torch.full(
            (b, seq_len),
            -1,
            dtype=torch.long,
            device=x.device,
        )
        indices[mask] = indices_real

        # 3. Decoder.
        dec_in = self.latent_unproj(quantized) + self.pos_encoding[:seq_len]
        dec_out = self.transformer_decoder(dec_in, src_key_padding_mask=~mask)
        trunk = self.decoder_trunk(dec_out)

        recon_outputs: dict[str, Tensor] = {}
        for name, _kind, _dim in self.recon_heads:
            recon_outputs[name] = self.recon_head_modules[name](trunk)

        # 4. Loss aggregation: continuous heads use Cartesian-space MSE
        #    (denormalized + spherical->Cartesian), categorical heads use CE.
        recon_loss, head_losses, recon_diag = self._compute_recon_loss(
            x,
            recon_outputs,
            mask,
        )

        return {
            "indices": indices,
            "commitment_loss": commitment_loss,
            "reconstruction_loss": recon_loss,
            "head_losses": head_losses,
            "recon_outputs": recon_outputs,
            "diagnostics": {
                **codebook_diag,
                **recon_diag,
                "z_diversity": z_diversity,
            },
        }

    def encode(self, x: Tensor) -> Tensor:
        """Encode a single sequence ``(N, descriptor_dim)`` to codebook indices."""
        x_seq = x.unsqueeze(0)
        h_in = self._embed_descriptor(x_seq)
        h = self.input_proj(self.input_norm(h_in)) + self.pos_encoding[: x.shape[0]]
        h = self.transformer_encoder(h)
        z = self.latent_norm(self.latent_proj(h)).squeeze(0)
        _, indices, _, _ = self.codebook(z)
        return indices

    @torch.no_grad()
    def encode_batch(self, x: Tensor, mask: Tensor) -> Tensor:
        """Encode a padded batch ``(B, L, descriptor_dim)`` to codebook indices.

        Mirrors the encoder path of :meth:`forward` but skips the decoder and
        loss. Returns ``(B, L)`` long indices with ``-1`` at padded positions.

        The caller MUST put the module in ``eval()`` mode: the EMA codebook only
        updates its statistics while ``training`` is ``True``, so eval mode keeps
        the frozen tokenizer deterministic.

        Args:
            x: ``(B, L, descriptor_dim)`` already-normalized descriptors.
            mask: ``(B, L)`` bool, ``True`` for real elements.
        """
        b, seq_len, _ = x.shape
        h_in = self._embed_descriptor(x)
        h = self.input_proj(self.input_norm(h_in)) + self.pos_encoding[:seq_len]
        h = self.transformer_encoder(h, src_key_padding_mask=~mask)
        z = self.latent_norm(self.latent_proj(h))  # (B, L, latent_dim)
        z_flat = z.reshape(b * seq_len, -1).float()
        _, indices_flat, _, _ = self.codebook(z_flat)
        indices = indices_flat.view(b, seq_len)
        indices = indices.masked_fill(~mask, -1)
        return indices

    def decode_to_outputs(self, indices: Tensor) -> dict[str, Tensor]:
        """Decode ``(N,)`` codebook indices into raw recon-head outputs.

        Returns a dict with one entry per head. The caller is responsible
        for converting categorical logits to indices (argmax) and continuous
        spherical outputs to Cartesian as needed.
        """
        quantized = self.codebook.lookup(indices)  # (N, latent_dim)
        q_seq = quantized.unsqueeze(0)
        dec_in = self.latent_unproj(q_seq) + self.pos_encoding[: indices.shape[0]]
        dec_out = self.transformer_decoder(dec_in)
        trunk = self.decoder_trunk(dec_out).squeeze(0)
        return {
            name: self.recon_head_modules[name](trunk)
            for name, _kind, _dim in self.recon_heads
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_descriptor(self, x: Tensor) -> Tensor:
        """Concatenate continuous slots with embedded categorical slots."""
        pieces: list[Tensor] = []
        for spec in self.layout:
            slot = x[..., spec.start : spec.end]
            if spec.kind == "continuous":
                pieces.append(slot)
            else:
                key = self._embedding_key(spec.name)
                # ``spec.length`` is 1 for singleton categoricals and K for
                # KNN slots. Cast to long for the embedding lookup, then
                # flatten the trailing two dims back into the feature axis.
                emb = self.cat_embeddings[key](slot.long())
                pieces.append(emb.reshape(*slot.shape[:-1], -1))
        return torch.cat(pieces, dim=-1)

    def _split_coord_head(
        self,
        coord_pred: Tensor,  # (B, L, 4) for ligand or (B, L, 12) for protein
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Split a coord head into r/θ/sin φ/cos φ groups (with unit-circle proj).

        For protein the head is 3 atoms x 4 dims = 12; we reshape to
        ``(B, L, 3, 4)`` and decompose along the last axis. Ligand stays
        ``(B, L, 1, 4)``.
        """
        b, seq_len, total = coord_pred.shape
        atoms = total // 4
        reshaped = coord_pred.view(b, seq_len, atoms, 4)
        r = reshaped[..., 0]
        theta = reshaped[..., 1]
        sphi, cphi = project_unit_circle(reshaped[..., 2], reshaped[..., 3])
        return r, theta, sphi, cphi

    def _compute_recon_loss(
        self,
        x: Tensor,
        recon_outputs: dict[str, Tensor],
        mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        """Compute multi-head reconstruction loss.

        - Continuous (coord) head: denormalize the predicted spherical
          values, convert to Cartesian, MSE against the ground-truth
          Cartesian (also denormalized + converted). This avoids the
          sin/cos-vs-MSE mismatch that biases the loss near θ=0.
        - Categorical heads: CE over the per-slot integer targets.
        """
        f = fields_by_name(self.layout)
        head_losses: dict[str, Tensor] = {}
        diag: dict[str, Tensor] = {}

        coord_field = f["coord"]
        coord_norm_target = x[..., coord_field.start : coord_field.end]
        coord_norm_pred = recon_outputs["coord"]

        # Denormalize coord slot only (categorical slots are not part of coord).
        mean = self._desc_mean[coord_field.start : coord_field.end].to(x.dtype)
        std = self._desc_std[coord_field.start : coord_field.end].to(x.dtype)
        coord_target = coord_norm_target * std + mean
        coord_pred = coord_norm_pred * std + mean

        # Spherical → Cartesian for both pred and target, MSE in canonical frame.
        r_t, th_t, s_t, c_t = self._split_coord_head(coord_target)
        r_p, th_p, s_p, c_p = self._split_coord_head(coord_pred)
        xyz_t = spherical_to_cartesian_batched(r_t, th_t, s_t, c_t)
        xyz_p = spherical_to_cartesian_batched(r_p, th_p, s_p, c_p)
        coord_diff = (xyz_t - xyz_p).pow(2).sum(dim=-1)  # (B, L, atoms)
        coord_diff_per_token = coord_diff.mean(dim=-1)  # (B, L)
        if mask.any():
            coord_loss = coord_diff_per_token[mask].mean()
            coord_max = coord_diff_per_token[mask].detach().max()
        else:
            coord_loss = x.new_zeros(())
            coord_max = x.new_zeros(())
        head_losses["coord"] = coord_loss
        diag["coord_max"] = coord_max
        diag["unit_circle_norm_err"] = (
            (coord_pred[..., 2::4].pow(2) + coord_pred[..., 3::4].pow(2))
            .sqrt()
            .sub(1.0)
            .abs()[mask]
            .mean()
            if mask.any()
            else x.new_zeros(())
        )

        # Categorical heads: target indices live in the same slot as input.
        for name, kind, _dim in self.recon_heads:
            if kind == "continuous":
                continue
            spec = f[name]
            target_idx = x[..., spec.start].long()  # singleton categorical
            logits = recon_outputs[name]  # (B, L, V)
            if mask.any():
                head_loss = F.cross_entropy(
                    logits[mask],
                    target_idx[mask],
                    reduction="mean",
                )
                pred = logits[mask].argmax(dim=-1)
                acc = (pred == target_idx[mask]).float().mean()
            else:
                head_loss = x.new_zeros(())
                acc = x.new_zeros(())
            head_losses[name] = head_loss
            diag[f"{name}_acc"] = acc

        # Aggregate (sum without weighting; module-level recon_weights are
        # applied in the training step).
        recon_loss = sum(head_losses.values()) if head_losses else x.new_zeros(())
        return recon_loss, head_losses, diag
