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
from src.tokenizers.geometry import sinusoidal_positional_encoding


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

        # Codebook
        self.codebook = EMACodebook(
            num_codes=config.codebook_size,
            code_dim=config.latent_dim,
            ema_decay=config.ema_decay,
            commitment_cost=config.commitment_cost,
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass with sequence context.

        Args:
            x: Descriptors of shape ``(B, L, descriptor_dim)``.
            mask: Boolean mask ``(B, L)``, ``True`` for real elements.

        Returns:
            Dict with keys: reconstructed, indices, commitment_loss,
            reconstruction_loss.
        """
        b, seq_len, _ = x.shape

        if mask is None:
            mask = torch.ones(b, seq_len, dtype=torch.bool, device=x.device)

        # Encode
        h = self.input_proj(x) + self.pos_encoding[:seq_len]
        h = self.transformer_encoder(h, src_key_padding_mask=~mask)
        z = self.latent_proj(h)  # (B, L, latent_dim)

        # Quantize per-position (only real elements)
        # Cast to float32 for codebook (EMA updates need full precision)
        z_real = z[mask].float()  # (N_real, latent_dim)
        quantized_real, indices_real, commitment_loss = self.codebook(z_real)

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
        diff = (x - x_hat)[mask]  # (N_real, descriptor_dim)
        reconstruction_loss = diff.pow(2).mean()

        return {
            "reconstructed": x_hat,
            "indices": indices,
            "commitment_loss": commitment_loss,
            "reconstruction_loss": reconstruction_loss,
        }

    def encode(self, x: Tensor) -> Tensor:
        """Encode descriptors to codebook indices.

        Args:
            x: ``(N, descriptor_dim)`` for a single sequence.

        Returns:
            Codebook indices of shape ``(N,)``.
        """
        x_seq = x.unsqueeze(0)  # (1, N, D)
        h = self.input_proj(x_seq) + self.pos_encoding[: x.shape[0]]
        h = self.transformer_encoder(h)
        z = self.latent_proj(h).squeeze(0)  # (N, latent_dim)
        _, indices, _ = self.codebook(z)
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
