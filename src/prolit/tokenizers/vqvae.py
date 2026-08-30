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
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from prolit.tokenizers.codebook import EMACodebook
from prolit.tokenizers.descriptor_schema import (
    ATOM_DESCRIPTOR_DIM,
    ATOM_LAYOUT,
    ATOM_PROTEIN_ONLY_HEADS,
    ATOM_RECON_HEADS,
    SOURCE_LIGAND_IDX,
    SOURCE_PROTEIN_IDX,
    FieldSpec,
    fields_by_name,
)
from prolit.tokenizers.geometry import (
    project_unit_circle,
    sinusoidal_positional_encoding,
    spherical_to_cartesian_batched,
)

if TYPE_CHECKING:
    # The live config. TransformerVQVAEConfig below is the shape older
    # checkpoints were trained with, and is kept so they still load.
    from prolit.config import AtomVQVAEConfig

#: A KNN slot with no neighbour stores a zero displacement, which survives
#: denormalization as an exact zero. Real neighbours are >= 1 A away, so any
#: radius near zero is padding.
_PAD_RADIUS = 1e-6

#: Slack added to the sum of covalent radii when deciding a reference pair is
#: bonded. Same rule and value as :func:`prolit.chem.pdb_io.infer_bonds`, which
#: is reliable on crystal coordinates -- and the reference IS crystal, so the
#: bond graph derived here needs no stored connectivity.
_BOND_TOLERANCE = 0.4


def _covalent_radii() -> Tensor:
    """Covalent radius per element index of ``LIGAND_ELEMENT_VOCAB``.

    Built from the one table that already exists rather than a second copy, so
    the bond graph this module derives matches the one the writers perceive.
    ``OTHER`` gets radius 0, which makes every pair involving it unbonded.
    """
    from prolit.chem.pdb_io import _COVALENT_RADII  # noqa: PLC0415
    from prolit.tokenizers.descriptor_schema import (  # noqa: PLC0415
        LIGAND_ELEMENT_VOCAB,
    )

    return torch.tensor(
        [_COVALENT_RADII.get(e, 0.0) for e in LIGAND_ELEMENT_VOCAB],
        dtype=torch.float32,
    )


#: Distinct ``source`` values a row can carry (protein, ligand).
_N_SOURCES = 2


def _class_balance(target: Tensor, n_classes: int) -> Tensor:
    """Per-atom weights that make every present group count the same.

    A cross-entropy over an imbalanced field is minimised well enough by
    answering with the majority class, and the loss that results looks small.
    The aromatic head is 69% "not aromatic", and after 250 epochs of exactly
    this loss it reached recall **0.014** on the reference ligands -- always
    saying no, precision 0.833, and an accuracy of 0.695 that read as healthy in
    the logs. Worse, the ``constrained`` balancer releases a head once its loss
    is under the control run's level, so the collapsed answer satisfied the
    constraint and the head was then trained at weight zero.

    Weighting each atom by the inverse frequency of its own group, normalised so
    the weights average one, removes that shortcut without introducing a
    constant: the balance comes from the batch, and a field that is already
    balanced gets weights of one. The caller passes (source, class) pairs as the
    group, so "mostly protein" cannot hide "wrong on ligands" either.
    """
    counts = torch.bincount(target, minlength=n_classes).clamp_min(1)
    inverse = counts.numel() / counts.float()
    per_atom = inverse[target]
    return per_atom / per_atom.mean().clamp_min(1e-8)


@dataclass
class TransformerVQVAEConfig:
    """Config for the all-atom Transformer VQ-VAE."""

    descriptor_dim: int = ATOM_DESCRIPTOR_DIM
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
    # Retained so older checkpoints round-trip; only "atom" is supported.
    domain: str = "atom"
    # Per-categorical embedding dim. Categorical slots map to learned vectors
    # of this size before being concatenated with continuous slots.
    categorical_embed_dim: int = 8


class TransformerVQVAE(nn.Module):
    """Transformer VQ-VAE with mixed-feature input + multi-head reconstruction."""

    # Registered as buffers below; ``register_buffer`` is invisible to a type
    # checker, so declare them the way torch's own modules do.
    pos_encoding: Tensor
    _desc_mean: Tensor
    _desc_std: Tensor
    _cov_radii: Tensor

    def __init__(self, config: TransformerVQVAEConfig | AtomVQVAEConfig) -> None:
        super().__init__()
        self.config: TransformerVQVAEConfig | AtomVQVAEConfig = config
        if config.domain != "atom":
            msg = f"Unknown domain: {config.domain!r} (only 'atom' is supported)"
            raise ValueError(msg)

        self.layout: list[FieldSpec] = ATOM_LAYOUT
        self.recon_heads: list[tuple[str, str, int]] = list(ATOM_RECON_HEADS)
        # Opt-in local-geometry head. The KNN displacements are already in the
        # descriptor as an encoder input; this reconstructs them, which is the
        # only signal in the model that says where an atom sits relative to its
        # neighbours rather than to the pocket centroid.
        self.predict_knn_offsets: bool = getattr(config, "predict_knn_offsets", False)
        if self.predict_knn_offsets:
            knn = fields_by_name(self.layout)["knn_offsets"]
            self.recon_heads.append(("knn_offsets", "continuous", knn.length))

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

        # Covalent radii for deriving the reference bond graph in the loss.
        self.register_buffer("_cov_radii", _covalent_radii(), persistent=False)

        # ---- Codebook --------------------------------------------------
        # One book over both modalities: protein and ligand atoms are quantized
        # against the same codes, which is what makes these interface tokens a
        # single shared vocabulary. The separate-tokenizer ablation instead
        # trains two of these end to end and stitches them together in
        # :class:`~prolit.tokenizers.separate_vqvae.SeparateVQVAE`.
        self.codebook = EMACodebook(
            num_codes=config.codebook_size,
            code_dim=config.latent_dim,
            ema_decay=config.ema_decay,
            commitment_cost=config.commitment_cost,
        )

    # ------------------------------------------------------------------
    # Helper: shared embedding tables for "element"/"aa" across slots
    # ------------------------------------------------------------------
    @property
    def _ligand_weight(self) -> float:
        return float(getattr(self.config, "ligand_source_weight", 1.0))

    @property
    def _ligand_ema_weight(self) -> float:
        """Ligand weight for the codebook EMA, defaulting to the loss weight.

        Separate knobs because they act on different things: the EMA weight
        moves centroids, i.e. how much of the book the ligand gets, while the
        loss weight moves the encoder and decoder. Defaulting to the loss
        weight keeps every run made before they were split behaving identically.
        """
        w = getattr(self.config, "ligand_ema_weight", None)
        return self._ligand_weight if w is None else float(w)

    def _source_weights(self, x: Tensor, weight: float | None = None) -> Tensor:
        """``(B, L)`` per-atom weight: 1 for protein, ``weight`` for ligand.

        All ones when the two modalities are left unbalanced.
        """
        w = self._ligand_weight if weight is None else weight
        f = fields_by_name(self.layout)
        if w == 1.0 or "source" not in f:
            return torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        source = x[..., f["source"].start].long()
        return torch.where(
            source == SOURCE_LIGAND_IDX,
            x.new_full((), w),
            x.new_ones(()),
        )

    def _per_source_coord(
        self, x: Tensor, mask: Tensor, coord_diff_per_token: Tensor
    ) -> dict[str, Tensor]:
        """Coordinate error split by source, UNWEIGHTED.

        ``coord`` itself is a source-weighted mean: it is the training
        objective, but it is not comparable between runs that weight the
        sources differently. At ``ligand_source_weight`` 8.3 it is a
        ligand-heavy average and at 1.0 a plain one, so reading the two off
        against each other compares the weighting rather than the model. These
        two do not move when the weights do.
        """
        src_field = fields_by_name(self.layout).get("source")
        if src_field is None or not mask.any():
            return {}
        is_ligand = x[..., src_field.start].long() == SOURCE_LIGAND_IDX
        out: dict[str, Tensor] = {}
        for name, sel in (
            ("coord_protein", mask & ~is_ligand),
            ("coord_ligand", mask & is_ligand),
        ):
            out[name] = (
                coord_diff_per_token[sel].detach().mean()
                if sel.any()
                else x.new_zeros(())
            )
        return out

    @staticmethod
    def _weighted_mean(values: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
        """Mean of ``values`` over ``mask``, each element counted ``weights`` times."""
        if not bool(mask.any()):
            return values.new_zeros(())
        w = weights[mask]
        return (values[mask] * w).sum() / w.sum().clamp_min(1e-8)

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
    ) -> dict[str, Any]:
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
        # Per-source EMA weights: without them the shared book's centroids
        # follow whichever modality supplies more atoms (8.3:1 in CrossDocked).
        ema_w = self._ligand_ema_weight
        ema_weights = self._source_weights(x, ema_w)[mask].float()
        quantized_real, indices_real, commitment_loss, codebook_diag = self.codebook(
            z_real, weights=ema_weights if ema_w != 1.0 else None
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
        return indices.masked_fill(~mask, -1)

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

    def _ligand_pair_losses(
        self,
        x: Tensor,
        xyz_p: Tensor,  # (B, L, 1, 3) decoded Cartesian
        xyz_t: Tensor,  # (B, L, 1, 3) reference Cartesian
        real: Tensor,  # (B, L) every real atom; ligand rows are read off ``x``
        f: dict[str, FieldSpec],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Everything that needs a ligand-ligand distance matrix.

        The per-atom coord head has no pairwise term at all, so the decoder is
        free to push atoms into each other and to stretch bonds; both are scored
        here off one shared ``cdist``.
        """
        d_floor = float(getattr(self.config, "clash_floor", 1.2))
        lig = real & (x[..., f["source"].start].long() == SOURCE_LIGAND_IDX)
        xyz, xyz_ref = xyz_p.squeeze(2), xyz_t.squeeze(2)
        pair = torch.cdist(xyz, xyz)
        pair_ref = torch.cdist(xyz_ref, xyz_ref)
        seq = xyz.shape[1]
        eye = torch.eye(seq, dtype=torch.bool, device=xyz.device).unsqueeze(0)
        valid = (lig.unsqueeze(1) & lig.unsqueeze(2)) & ~eye

        # Contact penalty, scored only where the REFERENCE keeps the pair apart.
        # That is what lets the floor exceed a bond length: a bond, or a 1-3
        # contact inside a ring, excludes itself by its own reference distance,
        # so no bond graph is needed and the term only ever says "do not invent
        # a contact the crystal does not have".
        unified = getattr(self.config, "pair_distance_loss", False)
        scored = valid & (pair_ref >= d_floor)
        losses: dict[str, Tensor] = {}
        diag: dict[str, Tensor] = {}
        want_clash = (
            not getattr(self.config, "drop_clash", False)
            and (not unified or getattr(self.config, "keep_clash", False))
        )
        if want_clash:
            losses["clash"] = (
                torch.relu(d_floor - pair).pow(2)[scored].mean()
                if bool(scored.any())
                else x.new_zeros(())
            )
            n_clash = ((pair < d_floor) & scored).float().sum()
            diag["clash_pair_frac"] = n_clash / scored.float().sum().clamp_min(1)

        if unified:
            # Every pair, every source, one term. See the note on
            # ``pair_distance_loss`` for why the error is relative.
            cut = float(getattr(self.config, "pair_distance_cutoff", 15.0))
            floor = float(getattr(self.config, "pair_distance_floor", 0.3))
            near = (real.unsqueeze(1) & real.unsqueeze(2)) & ~eye & (pair_ref < cut)
            if bool(near.any()):
                rel = (pair[near] - pair_ref[near]) / pair_ref[near].clamp_min(floor)
                losses["pair"] = rel.pow(2).mean()
                diag["pair_mae"] = (pair[near] - pair_ref[near]).abs().mean().detach()
            else:
                losses["pair"] = x.new_zeros(())
                diag["pair_mae"] = x.new_zeros(())
            return losses, diag

        # Bonded and geminal distances, penalised on the DECODED coordinates.
        #
        # This is the difference between constraining the output and merely
        # predicting local geometry alongside it. An earlier attempt gave the
        # decoder a head reconstructing the descriptor's KNN displacements; that
        # head learned its target well (2.86 -> 1.01) and changed bond accuracy
        # not at all, because it is a separate output of the same trunk and
        # nothing ties it to what ``coord`` emits. Scoring the pairwise distances
        # OF the decoded coordinates has no such escape.
        if getattr(self.config, "local_distance_loss", False):
            rows = (
                lig
                if getattr(self.config, "local_distance_ligand_only", False)
                else real
            )
            all_valid = (rows.unsqueeze(1) & rows.unsqueeze(2)) & ~eye
            losses["local"], loc_diag = self._local_distance_loss(
                x, pair, pair_ref, all_valid, f
            )
            diag.update(loc_diag)
        if getattr(self.config, "bond_distance_loss", False):
            # Bond lengths and angles are worth fixing wherever they occur; the
            # clash floor above is not, so the two use different masks.
            bond_rows = (
                real
                if getattr(self.config, "bond_distance_all_sources", False)
                else lig
            )
            bond_valid = (bond_rows.unsqueeze(1) & bond_rows.unsqueeze(2)) & ~eye
            d12, d13, bond_diag = self._bonded_distance_losses(
                x, pair, pair_ref, bond_valid, f
            )
            losses["bond12"], losses["bond13"] = d12, d13
            diag.update(bond_diag)

        if getattr(self.config, "distance_map_loss", False):
            # Every pair the reference holds within the cutoff, protein rows
            # only. The ligand already has 1-2/1-3 and a clash floor; adding a
            # third pairwise term there would just re-score the same distances.
            cut = float(getattr(self.config, "distance_map_cutoff", 15.0))
            # With ``local_distance_loss`` the short-range term already covers
            # every atom, so this one does too: the two split by DISTANCE, not
            # by which molecule an atom belongs to.
            local_on = getattr(self.config, "local_distance_loss", False)
            rows = real if local_on else (real & ~lig)
            dm = (rows.unsqueeze(1) & rows.unsqueeze(2)) & ~eye & (pair_ref < cut)
            losses["dmap"] = (
                (pair[dm] - pair_ref[dm]).pow(2).mean()
                if bool(dm.any())
                else x.new_zeros(())
            )
            diag["dmap_mae"] = (
                (pair[dm] - pair_ref[dm]).abs().mean().detach()
                if bool(dm.any())
                else x.new_zeros(())
            )
        return losses, diag

    def _local_distance_loss(
        self,
        x: Tensor,
        pair: Tensor,
        pair_ref: Tensor,
        valid: Tensor,
        f: dict[str, FieldSpec],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """1-2 and 1-3 together, relative error, every atom.

        Keeps the local / long-range split that a single 15 A term destroys --
        collapsing the two took bond MAE from 0.075 to 0.233 A and PB-valid from
        0.685 to 0.084, because that neighbourhood holds thousands of pairs
        against a few dozen bonds. What it does drop is the 5.0 / 2.0 ratio
        between 1-2 and 1-3: a relative error already charges the same absolute
        slip more on a 1.5 A bond than on a 2.5 A geminal pair.
        """
        bonded, geminal = self._bonded_masks(x, pair_ref, valid, f)
        loc = bonded | geminal
        if not bool(loc.any()):
            return x.new_zeros(()), {}
        rel = (pair[loc] - pair_ref[loc]) / pair_ref[loc].clamp_min(0.3)
        diag = {
            "local_mae": (pair[loc] - pair_ref[loc]).abs().mean().detach(),
            "local_per_atom": (
                loc.float().sum() / valid.any(-1).float().sum().clamp_min(1)
            ),
        }
        return rel.pow(2).mean(), diag

    def _bonded_masks(
        self,
        x: Tensor,
        pair_ref: Tensor,
        valid: Tensor,
        f: dict[str, FieldSpec],
    ) -> tuple[Tensor, Tensor]:
        """The 1-2 and 1-3 pair masks, shared by both local formulations."""
        elem = x[..., f["element"].start].long()
        radii = self._cov_radii.to(x.device, x.dtype)[elem]
        cutoff = radii.unsqueeze(1) + radii.unsqueeze(2) + _BOND_TOLERANCE
        bonded = valid & (pair_ref < cutoff) & (radii.unsqueeze(1) > 0)
        two = torch.bmm(bonded.to(x.dtype), bonded.to(x.dtype)) > 0
        return bonded, valid & two & ~bonded

    def _bonded_distance_losses(
        self,
        x: Tensor,
        pair: Tensor,  # (B, L, L) decoded distances
        pair_ref: Tensor,  # (B, L, L) reference distances
        valid: Tensor,  # (B, L, L) ligand-ligand, off-diagonal
        f: dict[str, FieldSpec],
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Squared error on 1-2 (bonded) and 1-3 (geminal) distances.

        1-2 fixes bond lengths directly. 1-3 is what fixes bond ANGLES: with the
        two bond lengths pinned, the angle at the shared atom is determined by
        the distance across it, so the angle needs no explicit trigonometry.
        """
        elem = x[..., f["element"].start].long()  # (B, L)
        radii = self._cov_radii.to(x.device, x.dtype)[elem]  # (B, L)
        cutoff = radii.unsqueeze(1) + radii.unsqueeze(2) + _BOND_TOLERANCE
        bonded = valid & (pair_ref < cutoff) & (radii.unsqueeze(1) > 0)

        # Two bonds away: reachable in exactly 2 steps and not already bonded.
        b = bonded.to(x.dtype)
        two = torch.bmm(b, b) > 0
        geminal = valid & two & ~bonded

        def _mse(mask: Tensor) -> Tensor:
            if not bool(mask.any()):
                return x.new_zeros(())
            return (pair[mask] - pair_ref[mask]).pow(2).mean()

        d12, d13 = _mse(bonded), _mse(geminal)
        diag = {
            "bond12_mae": (
                (pair[bonded] - pair_ref[bonded]).abs().mean().detach()
                if bool(bonded.any())
                else x.new_zeros(())
            ),
            "bonds_per_atom": (
                bonded.float().sum() / valid.any(-1).float().sum().clamp_min(1)
            ),
        }
        return d12, d13, diag

    def _compute_recon_loss(  # noqa: C901, PLR0912, PLR0915
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
        weights = self._source_weights(x)

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
            coord_loss = self._weighted_mean(coord_diff_per_token, mask, weights)
            coord_max = coord_diff_per_token[mask].detach().max()
        else:
            coord_loss = x.new_zeros(())
            coord_max = x.new_zeros(())
        head_losses["coord"] = coord_loss
        diag["coord_max"] = coord_max
        diag.update(self._per_source_coord(x, mask, coord_diff_per_token))
        diag["unit_circle_norm_err"] = (
            (coord_pred[..., 2::4].pow(2) + coord_pred[..., 3::4].pow(2))
            .sqrt()
            .sub(1.0)
            .abs()[mask]
            .mean()
            if mask.any()
            else x.new_zeros(())
        )

        # Local geometry: reconstruct each atom's displacement to its K nearest
        # neighbours. Same treatment as ``coord`` -- denormalize, spherical to
        # Cartesian, MSE -- but the target is a relative vector, so getting it
        # right means getting bond lengths and angles right. Padded neighbour
        # slots (atoms with fewer than K neighbours) carry a zero displacement
        # and are masked out by their radius.
        if self.predict_knn_offsets and "knn_offsets" in recon_outputs:
            knn = f["knn_offsets"]
            k_mean = self._desc_mean[knn.start : knn.end].to(x.dtype)
            k_std = self._desc_std[knn.start : knn.end].to(x.dtype)
            k_target = x[..., knn.start : knn.end] * k_std + k_mean
            k_pred = recon_outputs["knn_offsets"] * k_std + k_mean
            rt, tt, st_, ct = self._split_coord_head(k_target)
            rp, tp, sp, cp = self._split_coord_head(k_pred)
            knn_t = spherical_to_cartesian_batched(rt, tt, st_, ct)
            knn_p = spherical_to_cartesian_batched(rp, tp, sp, cp)
            slot_valid = (rt.abs() > _PAD_RADIUS) & mask.unsqueeze(-1)  # (B, L, K)
            if bool(slot_valid.any()):
                per_slot = (knn_t - knn_p).pow(2).sum(dim=-1)  # (B, L, K)
                w_slot = weights.unsqueeze(-1).expand_as(per_slot)
                head_losses["knn_offsets"] = self._weighted_mean(
                    per_slot, slot_valid, w_slot
                )
                diag["knn_offset_rmse"] = (
                    per_slot[slot_valid].detach().mean().sqrt()
                )
            else:
                head_losses["knn_offsets"] = x.new_zeros(())

        # Per-source row masks for the unified ``atom`` domain. ``source`` is a
        # singleton categorical input slot (absent for the legacy ligand/protein
        # domains, in which every row is the same source).
        source = x[..., f["source"].start].long() if "source" in f else None

        # Contact penalty (ligand atoms only). The per-atom coord head has no
        # pairwise term, so the decoder freely pushes atoms into each other.
        #
        # The pair is scored only where the REFERENCE keeps the two atoms at
        # least ``d_floor`` apart, which is what lets the floor exceed a bond
        # length: a bonded pair, or a 1-3 contact inside a ring, is excluded by
        # its own reference distance rather than by a bond graph the tokenizer
        # does not have. So this term says "do not invent a contact the crystal
        # does not have" and never "push these atoms apart".
        #
        # At the historical floor of 1.2 A the term is nearly inert -- 68.5% of
        # reconstructions still hold a non-bonded pair under 2.0 A.
        if source is not None:
            pair_losses, pair_diag = self._ligand_pair_losses(
                x, xyz_p, xyz_t, mask, f
            )
            head_losses.update(pair_losses)
            diag.update(pair_diag)

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
                per_atom = F.cross_entropy(
                    logits[head_mask],
                    target_idx[head_mask],
                    reduction="none",
                )
                w = weights[head_mask]
                if self.config.balanced_chem_loss:
                    # Balance over (source, class) jointly, not class alone. A
                    # pocket brings ~200 atoms and its ligand ~25, so a head can
                    # satisfy a pooled target on protein rows while the ligand
                    # rows -- the ones generation uses -- stay wrong. Measured:
                    # the aromatic head reached recall 0.014 on reference
                    # ligands while its multiplier sat at 0.013, i.e. the
                    # constraint never even registered as violated.
                    group = target_idx[head_mask]
                    if source is not None:
                        group = group * _N_SOURCES + source[head_mask]
                    w = w * _class_balance(
                        group, int(group.max().item()) + 1
                    )
                head_loss = (per_atom * w).sum() / w.sum().clamp_min(1e-8)
                pred = logits[head_mask].argmax(dim=-1)
                acc = (pred == target_idx[head_mask]).float().mean()
            else:
                head_loss = x.new_zeros(())
                acc = x.new_zeros(())
            head_losses[name] = head_loss
            diag[f"{name}_acc"] = acc

        # Aggregate (sum without weighting; module-level recon_weights are
        # applied in the training step).
        recon_loss = sum(head_losses.values(), start=x.new_zeros(()))
        return recon_loss, head_losses, diag
