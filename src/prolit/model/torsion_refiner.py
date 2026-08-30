"""A refiner that emits (translation, rotation, torsions) instead of a field.

The existing :class:`~prolit.model.pose_refiner.PoseRefinerNet` predicts one
free 3-vector per atom. That output space contains "shrink the molecule", and
measured over 60 targets every refiner trained in it takes that option: bonds
out of tolerance go 10.0% (no refiner) -> 48.1% (``refit_press0.6``) /
50.3% (``refit_deploy``) while the clash rate goes 36.0% -> 25.0% / 28.3%.
Those are the same act -- a smaller molecule overlaps less -- and it costs
PoseBusters validity 0.728 -> 0.427 and strain 197 -> 693.

Weighting a bond loss cannot close that off: bonded pairs carry 0.077 of the
reconstruction loss, so ``lambda_bond`` is an order of magnitude short, and
``lambda_angle`` defaults to 0. Restricting the output space closes it by
construction -- see :mod:`prolit.model.torsion_transform`.

The space is expressive enough, and that was measured before this was built:
optimising rigid motion plus every torsion against a steric objective takes
atoms deeper than 0.5 A inside the receptor from 29.4% to 10.0% (FLOWR: 7.3%),
where rigid alone stops at 16.5% clash and terminal-only torsions fix 1%.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import lightning as L
import torch
from e3nn import o3
from torch import Tensor, nn

from prolit.model.pose_refiner import PoseRefinerNet, build_refiner_optimizers
from prolit.model.torsion_transform import apply_transform

if TYPE_CHECKING:
    from prolit.config import PoseRefinerConfig, PoseRefineTrainingConfig


class TorsionRefinerNet(nn.Module):
    """Backbone of :class:`PoseRefinerNet` with a pose-parameter head.

    Equivariance is respected per output: the translation and the rotation axis
    are 1o (they rotate with the frame), while each torsion angle is a 0e
    scalar (a dihedral is invariant under a global rotation, so it must be).
    """

    def __init__(self, config: PoseRefinerConfig) -> None:
        super().__init__()
        self.backbone = PoseRefinerNet(config)
        hidden = self.backbone.irreps_hidden
        # Two 1o channels: one becomes the translation, one the rotation axis
        # (its magnitude is the angle, so no separate scalar is needed).
        self.rigid_head = o3.Linear(hidden, o3.Irreps("2x1o"))
        # Torsions read the INVARIANT part of the two axis atoms. Using 0e only
        # is what makes a predicted dihedral independent of the pose's global
        # orientation; feeding it vectors would silently break that.
        n_scalar = sum(m for m, ir in hidden if ir.l == 0 and ir.p == 1)
        self.n_scalar = n_scalar
        self.torsion_head = nn.Sequential(
            nn.Linear(2 * n_scalar, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),  # raw -> angle via pi * tanh
        )

    def _hidden(  # noqa: PLR0913
        self,
        pos: Tensor,
        feat: Tensor,
        t_node: Tensor,
        edge_src: Tensor,
        edge_dst: Tensor,
        edge_bond: Tensor,
    ) -> Tensor:
        b = self.backbone
        scal = sum(emb(feat[:, i]) for i, emb in enumerate(b.embeds))
        from prolit.model.pose_refiner import _timestep_embedding  # noqa: PLC0415

        scal = scal + b.time_mlp(_timestep_embedding(t_node, b.time_dim))
        h = b.embed(scal)
        rel = pos[edge_dst] - pos[edge_src]
        edge_sh = o3.spherical_harmonics(
            b.irreps_sh, rel, normalize=True, normalization="component"
        )
        from e3nn.math import soft_one_hot_linspace  # noqa: PLC0415

        edge_radial = soft_one_hot_linspace(
            rel.norm(dim=1), 0.0, b.config.pocket_cutoff, b.config.num_radial,
            basis="bessel", cutoff=True,
        ).mul(b.config.num_radial**0.5)
        edge_radial = edge_radial + b.bond_embed(edge_bond)
        deg = pos.new_zeros(pos.shape[0]).index_add_(
            0, edge_dst, torch.ones_like(rel.norm(dim=1))
        )
        inv_sqrt_deg = deg.clamp(min=1.0).rsqrt()
        for conv in b.convs:
            h = h + b.gate(
                conv(h, edge_src, edge_dst, edge_sh, edge_radial, inv_sqrt_deg)
            )
        return h

    def forward(  # noqa: PLR0913
        self,
        pos: Tensor,
        feat: Tensor,
        t_node: Tensor,
        edge_src: Tensor,
        edge_dst: Tensor,
        movable: Tensor,
        edge_bond: Tensor,
        pairs: Tensor,  # (K, 2) rotatable-bond endpoints, GLOBAL node indices
        batch: Tensor | None = None,  # (N,) sample index per node
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns ``(translation (B,3), rot_vec (B,3), angles (K,))``.

        With ``batch=None`` the whole graph is one sample and B is 1.
        """
        h = self._hidden(pos, feat, t_node, edge_src, edge_dst, edge_bond)
        rigid = self.rigid_head(h)  # (N, 6)
        lig = movable.bool()
        if batch is None:
            batch = pos.new_zeros(pos.shape[0], dtype=torch.long)
        n_batch = int(batch.max().item()) + 1 if batch.numel() else 1
        # Pool over each sample's LIGAND nodes only: pocket nodes are frozen
        # context and must not pull the predicted motion around.
        w = lig.to(rigid.dtype)[:, None]
        num = rigid.new_zeros(n_batch, rigid.shape[1]).index_add_(0, batch, rigid * w)
        den = rigid.new_zeros(n_batch, 1).index_add_(0, batch, w).clamp_min(1.0)
        pooled = num / den
        translation, rot_vec = pooled[:, :3], pooled[:, 3:]
        if pairs.numel() == 0:
            return translation, rot_vec, pos.new_zeros(0)
        scal = h[:, : self.n_scalar]
        pair_feat = torch.cat([scal[pairs[:, 0]], scal[pairs[:, 1]]], dim=-1)
        # A BOUNDED DIRECT angle, not atan2 of a (cos, sin) pair.
        #
        # atan2 looks like the safe choice -- it has no 2*pi seam -- but it hands
        # the head an escape hatch and the head takes it. A random angle at init
        # is expensive, the cheapest descent direction is "emit zero" via a large
        # positive first component, and d(atan2)/d(cs) ~ 1/|cs| then vanishes.
        # Measured on the first run at epoch 2: |cs| had median **490**, the
        # predicted torsions had median |angle| 0.42 deg against a 22.9 deg
        # corruption, and the gradient there was 0.007 against 3.6 at |cs| = 1.
        # Normalising cs does NOT help -- the chain rule keeps the same 1/|cs|
        # factor on the raw output (measured: identical 0.00735).
        #
        # With pi * tanh, "emit zero" sits at raw = 0, where the gradient is at
        # its MAXIMUM (measured 11.3), so the state the head ran to is the one
        # it can most easily leave. The 2*pi seam is not a problem here because
        # the corruption is N(0, 0.4 rad): targets sit nowhere near +-pi.
        angles = math.pi * torch.tanh(self.torsion_head(pair_feat).squeeze(-1))
        return translation, rot_vec, angles


class TorsionRefinerModule(L.LightningModule):
    """Direct-regression trainer for :class:`TorsionRefinerNet`.

    Not flow matching. The output is a small set of pose parameters, not a field
    over atoms, so there is no interpolant to denoise -- the network sees the
    corrupted pose and names the transform that fixes it. The loss is on the
    resulting COORDINATES rather than on the parameters, because the same pose
    has many parameterisations (a torsion of +pi and -pi coincide) and
    regressing the parameters would punish a correct pose for being written
    differently.
    """

    def __init__(self, config: PoseRefineTrainingConfig) -> None:
        super().__init__()
        self.config: PoseRefineTrainingConfig = config
        self.save_hyperparameters()
        self.net = TorsionRefinerNet(config.model)
        self._last_angles: Tensor | None = None

    def predict(self, batch: dict[str, Tensor]) -> Tensor:
        """Apply the predicted transform, returning the full node positions."""
        pos = batch["pos0"]
        t0 = pos.new_zeros(pos.shape[0])
        trans, rot, angles = self.net(
            pos,
            batch["feat"],
            t0,
            batch["edge_src"],
            batch["edge_dst"],
            batch["movable"],
            batch["edge_bond"],
            batch["tors_pairs"],
            batch["batch"],
        )
        self._last_angles = angles
        out = pos.clone()
        starts = batch["lig_start"].tolist()
        sizes = batch["lig_size"].tolist()
        pair_ptr = batch["tors_ptr"].tolist()
        for b, (s, n) in enumerate(zip(starts, sizes, strict=True)):
            lo, hi = pair_ptr[b], pair_ptr[b + 1]
            local_pairs = batch["tors_pairs"][lo:hi] - s
            masks = batch["tors_masks"][lo:hi, :n]
            out[s : s + n] = apply_transform(
                pos[s : s + n], trans[b], rot[b], local_pairs, masks, angles[lo:hi]
            )
        return out

    @torch.no_grad()
    def refine(self, batch: dict[str, Tensor], n_steps: int | None = None) -> Tensor:  # noqa: ARG002
        """Same name and signature as :meth:`PoseRefinerModule.refine`.

        There is no ODE to step here -- the network names the transform in one
        pass -- so ``n_steps`` is accepted and ignored. Keeping the signature
        lets :func:`~prolit.model.pose_refiner.refine_ligand_canonical` drive
        either refiner without knowing which one it holds.
        """
        return self.predict(batch)

    def _compute(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        pred = self.predict(batch)
        mov = batch["movable"]
        err = (pred[mov] - batch["pos1"][mov]).pow(2).sum(-1)
        out = {"loss": err.mean(), "rmsd": err.mean().sqrt()}
        # Optional DIRECT supervision of the torsion angles.
        #
        # The corruption angle is generated by this repository, so the target is
        # known exactly -- it does not have to be inferred through a coordinate
        # loss. That distinction is the experiment: under coordinate MSE the
        # torsion head decays to 1-6% of the corruption in every setting tried
        # (sigma 0.4 and 0.1, with and without competing rigid corruption), and
        # the RMSD gain comes from the rigid heads compensating instead. If the
        # head still refuses to move when the angle is handed to it directly,
        # the pose does not determine the angle and no loss will fix it; if it
        # learns, the coordinate loss was the problem.
        w = getattr(self.config, "torsion_angle_weight", 0.0)
        if w > 0 and batch.get("tors_twist") is not None:
            tgt = batch["tors_twist"]
            ang = self._last_angles
            if ang is not None and ang.numel() == tgt.numel() and tgt.numel():
                # Compare on the circle so +-pi does not read as a huge error.
                d = torch.atan2(torch.sin(ang - tgt), torch.cos(ang - tgt))
                out["angle_mae"] = d.abs().mean()
                out["loss"] = out["loss"] + w * d.pow(2).mean()
        return out

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        out = self._compute(batch)
        bs = int(batch["num_graphs"])
        for name, val in out.items():
            self.log(
                f"{stage}/{name}",
                val,
                prog_bar=name == "loss",
                sync_dist=stage != "train",
                batch_size=bs,
            )
        return out["loss"]

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: ARG002
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        self._step(batch, "val")
        mov = batch["movable"]
        with torch.no_grad():
            refined = self.predict(batch)
        r_ref = (refined[mov] - batch["pos1"][mov]).pow(2).sum(-1).mean().sqrt()
        r_cor = (batch["pos0"][mov] - batch["pos1"][mov]).pow(2).sum(-1).mean().sqrt()
        bs = int(batch["num_graphs"])
        self.log(
            "val/rmsd_refined", r_ref, prog_bar=True, sync_dist=True, batch_size=bs
        )
        self.log("val/rmsd_corrupt", r_cor, sync_dist=True, batch_size=bs)
        self.log(
            "val/rmsd_gain", r_cor - r_ref, prog_bar=True, sync_dist=True, batch_size=bs
        )

    def configure_optimizers(self):  # noqa: ANN201
        """Same optimiser and schedule as :class:`PoseRefinerModule`.

        Shared rather than reimplemented: the config field is ``learning_rate``,
        not ``lr``, and writing it out again here is how the first submitted run
        died at optimiser setup.
        """
        return build_refiner_optimizers(self)
