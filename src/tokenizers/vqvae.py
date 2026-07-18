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
    ATOM_LAYOUT,
    ATOM_PROTEIN_ONLY_HEADS,
    ATOM_RECON_HEADS,
    LIGAND_LAYOUT,
    LIGAND_RECON_HEADS,
    PROTEIN_LAYOUT,
    PROTEIN_RECON_HEADS,
    SOURCE_LIGAND_IDX,
    SOURCE_PROTEIN_IDX,
    SOURCE_VOCAB,
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
        if config.domain not in ("ligand", "protein", "atom"):
            msg = f"Unknown domain: {config.domain!r}"
            raise ValueError(msg)

        _layouts: dict[str, list[FieldSpec]] = {
            "ligand": LIGAND_LAYOUT,
            "protein": PROTEIN_LAYOUT,
            "atom": ATOM_LAYOUT,
        }
        _heads: dict[str, list[tuple[str, str, int]]] = {
            "ligand": LIGAND_RECON_HEADS,
            "protein": PROTEIN_RECON_HEADS,
            "atom": ATOM_RECON_HEADS,
        }
        self.layout: list[FieldSpec] = _layouts[config.domain]
        self.recon_heads: list[tuple[str, str, int]] = _heads[config.domain]

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
        # ``self.codebook`` is the sole codebook in the default (single-book)
        # setup; under ``split_codebook`` it is the PROTEIN codebook and
        # ``self.codebook_ligand`` is a separate ligand-only codebook, routed by
        # the ``source`` slot. A learned source embedding is added to the
        # quantized vector so the shared decoder can tell which book it came
        # from (the two books live in the same latent space).
        self.codebook = EMACodebook(
            num_codes=config.codebook_size,
            code_dim=config.latent_dim,
            ema_decay=config.ema_decay,
            commitment_cost=config.commitment_cost,
        )
        self.split_codebook: bool = getattr(config, "split_codebook", False)
        if self.split_codebook:
            self.codebook_ligand = EMACodebook(
                num_codes=config.ligand_codebook_size,
                code_dim=config.latent_dim,
                ema_decay=config.ema_decay,
                commitment_cost=config.commitment_cost,
            )
            self.source_embed = nn.Embedding(len(SOURCE_VOCAB), config.latent_dim)

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

    def _quantize_split(
        self,
        z: Tensor,
        source: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Quantize ``(M, latent)`` rows against per-source codebooks.

        Protein rows use ``self.codebook``, ligand rows ``self.codebook_ligand``.
        Returned ``indices`` live in each book's own 0-based range (the caller /
        LM vocab keeps protein and ligand on disjoint token ranges). The
        commitment loss is the fraction-weighted sum so it matches the mean over
        all rows of the single-book path.
        """
        quantized = torch.zeros_like(z)
        indices = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        commit = z.new_zeros(())
        diag: dict[str, Tensor] = {}
        total = max(z.shape[0], 1)
        for src_idx, book, tag in (
            (SOURCE_PROTEIN_IDX, self.codebook, "protein"),
            (SOURCE_LIGAND_IDX, self.codebook_ligand, "ligand"),
        ):
            m = source == src_idx
            if not bool(m.any()):
                continue
            q, idx, c, d = book(z[m])
            quantized[m] = q.to(quantized.dtype)
            indices[m] = idx
            commit = commit + c * (int(m.sum()) / total)
            for k, v in d.items():
                diag[f"{tag}_{k}"] = v
        return quantized, indices, commit, diag

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
        if self.split_codebook:
            f = fields_by_name(self.layout)
            source_real = x[..., f["source"].start].long()[mask]
            quant_out = self._quantize_split(z_real, source_real)
        else:
            quant_out = self.codebook(z_real)
        quantized_real, indices_real, commitment_loss, codebook_diag = quant_out
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

        # 3. Decoder. Under split_codebook, add a per-position source embedding
        #    so the shared decoder can tell protein- vs ligand-book vectors apart
        #    (padded positions are dropped by the key-padding mask below).
        if self.split_codebook:
            src_full = x[..., fields_by_name(self.layout)["source"].start].long()
            quantized = quantized + self.source_embed(src_full)
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
        """Encode a single sequence ``(N, descriptor_dim)`` to codebook indices.

        Under ``split_codebook`` the per-row ``source`` slot routes each atom to
        its own book; indices are returned in each book's own 0-based range.
        """
        x_seq = x.unsqueeze(0)
        h_in = self._embed_descriptor(x_seq)
        h = self.input_proj(self.input_norm(h_in)) + self.pos_encoding[: x.shape[0]]
        h = self.transformer_encoder(h)
        z = self.latent_norm(self.latent_proj(h)).squeeze(0)
        if self.split_codebook:
            source = x[..., fields_by_name(self.layout)["source"].start].long()
            _, indices, _, _ = self._quantize_split(z, source)
            return indices
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
        if self.split_codebook:
            source_flat = x[..., fields_by_name(self.layout)["source"].start].long()
            source_flat = source_flat.reshape(b * seq_len)
            _, indices_flat, _, _ = self._quantize_split(z_flat, source_flat)
        else:
            _, indices_flat, _, _ = self.codebook(z_flat)
        indices = indices_flat.view(b, seq_len)
        return indices.masked_fill(~mask, -1)

    def decode_to_outputs(
        self,
        indices: Tensor,
        source_idx: int | None = None,
    ) -> dict[str, Tensor]:
        """Decode ``(N,)`` codebook indices into raw recon-head outputs.

        Returns a dict with one entry per head. The caller is responsible
        for converting categorical logits to indices (argmax) and continuous
        spherical outputs to Cartesian as needed.

        Under ``split_codebook``, ``source_idx`` selects which book to look the
        indices up in (all ``N`` positions share one source: a pocket OR a
        ligand) and adds the matching source embedding, mirroring
        :meth:`forward`.
        """
        if self.split_codebook:
            if source_idx is None:
                msg = "decode_to_outputs requires source_idx when split_codebook"
                raise ValueError(msg)
            book = (
                self.codebook
                if source_idx == SOURCE_PROTEIN_IDX
                else self.codebook_ligand
            )
            quantized = book.lookup(indices)  # (N, latent_dim)
            src = torch.full_like(indices, source_idx)
            quantized = quantized + self.source_embed(src)
        else:
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

    def _compute_recon_loss(  # noqa: PLR0915
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

        # Per-source row masks for the unified ``atom`` domain. ``source`` is a
        # singleton categorical input slot (absent for the legacy ligand/protein
        # domains, in which every row is the same source).
        source = x[..., f["source"].start].long() if "source" in f else None

        # Clash penalty (ligand atoms only): hinge on reconstructed atom pairs
        # closer than ``d_floor``. The per-atom coord head has no pairwise term,
        # so reconstructions frequently overlap (~77% sub-1.2 Å clashes vs ~10%
        # for GT); this directly penalises that. ``xyz_p`` is the decoded
        # Cartesian (denormalised) computed above; 1 atom per coord head.
        if self.config.domain == "ligand":
            clash_mask = mask
        elif self.config.domain == "atom" and source is not None:
            clash_mask = mask & (source == SOURCE_LIGAND_IDX)
        else:
            clash_mask = None
        if clash_mask is not None:
            d_floor = 1.2
            xyz = xyz_p.squeeze(2)  # (B, L, 3)
            pair = torch.cdist(xyz, xyz)  # (B, L, L)
            seq = xyz.shape[1]
            eye = torch.eye(seq, dtype=torch.bool, device=xyz.device).unsqueeze(0)
            valid = (clash_mask.unsqueeze(1) & clash_mask.unsqueeze(2)) & ~eye
            head_losses["clash"] = (
                torch.relu(d_floor - pair).pow(2)[valid].mean()
                if bool(valid.any())
                else x.new_zeros(())
            )
            n_clash = ((pair < d_floor) & valid).float().sum()
            diag["clash_pair_frac"] = n_clash / valid.float().sum().clamp_min(1)

        # Categorical heads: target indices live in the same slot as input.
        # Protein-context heads (aa / bb_sc) are only meaningful for protein
        # atoms, so their loss is restricted to ``source == protein`` rows.
        for name, kind, _dim in self.recon_heads:
            if kind == "continuous":
                continue
            if (
                self.config.domain == "atom"
                and source is not None
                and name in ATOM_PROTEIN_ONLY_HEADS
            ):
                head_mask = mask & (source == SOURCE_PROTEIN_IDX)
            else:
                head_mask = mask
            spec = f[name]
            target_idx = x[..., spec.start].long()  # singleton categorical
            logits = recon_outputs[name]  # (B, L, V)
            if head_mask.any():
                head_loss = F.cross_entropy(
                    logits[head_mask],
                    target_idx[head_mask],
                    reduction="mean",
                )
                pred = logits[head_mask].argmax(dim=-1)
                acc = (pred == target_idx[head_mask]).float().mean()
            else:
                head_loss = x.new_zeros(())
                acc = x.new_zeros(())
            head_losses[name] = head_loss
            diag[f"{name}_acc"] = acc

        # Aggregate (sum without weighting; module-level recon_weights are
        # applied in the training step).
        recon_loss = sum(head_losses.values()) if head_losses else x.new_zeros(())
        return recon_loss, head_losses, diag
