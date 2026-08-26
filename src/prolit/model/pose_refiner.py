"""E(3)-equivariant ligand pose refiner (e3nn) + LightningModule.

The refiner is a small equivariant graph denoiser that fixes the *local
geometry* of a generated ligand pose (bond lengths/angles, steric clashes)
while preserving its global placement in the pocket. It refines only the
ligand heavy-atom coordinates; the pocket atoms are frozen context.

Formulation (see :class:`prolit.config.PoseRefineTrainingConfig`): a flow-matching
bridge from the VQ-VAE reconstruction of a native ligand (``x0`` -- the exact
corruption the generation pipeline emits) to the crystal pose (``x1``). We
interpolate ``x_t = (1-t) x0 + t x1`` and regress the clean pose ``x1`` with a
1o (odd-vector) displacement head, plus physical clash/bond auxiliary losses.

Equivariance: the network reads only relative edge vectors (via spherical
harmonics) and invariant per-atom chemistry scalars, and emits a ``1o`` vector
per ligand atom, so ``refine(R x + t, R pocket + t) = R refine(x, pocket) + t``
for any rotation/translation ``R, t``. This makes it robust to the (heuristic,
occasionally axis-flipping) pocket canonical frame.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from prolit.tokenizers.descriptor_schema import (
    ATOM_LAYOUT,
    BB_SC_NA_IDX,
    BB_SC_VOCAB,
    LIGAND_CHARGE_VOCAB,
    LIGAND_ELEMENT_VOCAB,
    LIGAND_HYBRID_VOCAB,
    LIGAND_NUMH_VOCAB,
    LIGAND_RING_VOCAB,
    PROTEIN_AA_VOCAB,
    PROTEIN_AA_X_IDX,
    SOURCE_LIGAND_IDX,
    SOURCE_VOCAB,
    fields_by_name,
)

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

    from prolit.config import PoseRefinerConfig, PoseRefineTrainingConfig

# Per-node categorical features, in the fixed order the dataset emits them.
# Every field is available at BOTH train time (real descriptors) and inference
# time (VQ-VAE decode heads for the ligand; receptor parse for the pocket).
FEATURE_FIELDS: tuple[tuple[str, int], ...] = (
    ("source", len(SOURCE_VOCAB)),
    ("element", len(LIGAND_ELEMENT_VOCAB)),
    ("charge", len(LIGAND_CHARGE_VOCAB)),
    ("hybrid", len(LIGAND_HYBRID_VOCAB)),
    ("aromatic", 2),
    ("ring", len(LIGAND_RING_VOCAB)),
    ("numH", len(LIGAND_NUMH_VOCAB)),
    ("aa", len(PROTEIN_AA_VOCAB)),
    ("bb_sc", len(BB_SC_VOCAB)),
)
NUM_FEATURE_FIELDS = len(FEATURE_FIELDS)

# The six chemistry heads both VQ-VAE decoders emit (LIGAND_RECON_HEADS /
# ATOM_RECON_HEADS share them); used to build ligand node features at inference.
LIG_CHEM_HEADS: tuple[str, ...] = (
    "element",
    "charge",
    "hybrid",
    "aromatic",
    "ring",
    "numH",
)


def _feats_from_named(named: dict[str, np.ndarray], n: int) -> np.ndarray:
    feat = np.zeros((n, NUM_FEATURE_FIELDS), dtype=np.int16)
    for k, (name, _) in enumerate(FEATURE_FIELDS):
        feat[:, k] = named[name]
    return feat


def pocket_feats_from_descriptor(prot_desc: np.ndarray) -> np.ndarray:
    """Extract the (M, 9) node-feature block from an all-atom ProteinAtomDescriptor."""
    f = fields_by_name(ATOM_LAYOUT)
    named = {name: prot_desc[:, f[name].start] for name, _ in FEATURE_FIELDS}
    return _feats_from_named(named, prot_desc.shape[0])


def ligand_feats_from_heads(chem: dict[str, np.ndarray], n: int) -> np.ndarray:
    """Build the (n, 9) ligand node-feature block from decoded VQ chem heads."""
    named: dict[str, np.ndarray] = dict(chem)
    named["source"] = np.full(n, SOURCE_LIGAND_IDX, dtype=np.int64)
    named["aa"] = np.full(n, PROTEIN_AA_X_IDX, dtype=np.int64)
    named["bb_sc"] = np.full(n, BB_SC_NA_IDX, dtype=np.int64)
    return _feats_from_named(named, n)


def _timestep_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding of a scalar diffusion time ``t in [0, 1]``."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _make_gate(irreps_spec: o3.Irreps) -> Gate:
    """Build an e3nn ``Gate`` for a hidden irreps spec.

    Even scalars use SiLU, odd scalars (pseudoscalars) use tanh (must be odd to
    stay ``0o``); every ``l > 0`` irrep is gated by a sigmoid over a fresh ``0e``
    scalar. ``gate.irreps_out`` is the usable hidden representation.
    """
    scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_spec if ir.l == 0])
    gated = o3.Irreps([(mul, ir) for mul, ir in irreps_spec if ir.l > 0])
    gates = o3.Irreps([(mul, "0e") for mul, _ in gated])
    act_scalars = [F.silu if ir.p == 1 else torch.tanh for _, ir in scalars]
    act_gates = [torch.sigmoid] * len(gates)
    return Gate(scalars, act_scalars, gates, act_gates, gated)


class _Convolution(nn.Module):
    """One equivariant message-passing step (e3nn tensor-product convolution).

    ``msg_ij = TP(f_j, Y(r_ij); W(|r_ij|))`` scattered onto the destination node
    and normalised by ``sqrt(deg)``. Edge geometry (spherical harmonics + radial
    basis) is precomputed once per network call and shared across layers; only
    the radial-to-weight MLP is per-layer.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_sh: o3.Irreps,
        irreps_out: o3.Irreps,
        num_radial: int,
        radial_hidden: int,
    ) -> None:
        super().__init__()
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in,
            irreps_sh,
            irreps_out,
            shared_weights=False,
            internal_weights=False,
        )
        self.fc = FullyConnectedNet(
            [num_radial, radial_hidden, self.tp.weight_numel], F.silu
        )
        self.irreps_out: o3.Irreps = irreps_out

    def forward(  # noqa: PLR0913
        self,
        f_in: Tensor,
        edge_src: Tensor,
        edge_dst: Tensor,
        edge_sh: Tensor,
        edge_radial: Tensor,
        inv_sqrt_deg: Tensor,
    ) -> Tensor:
        weight = self.fc(edge_radial)
        msg = self.tp(f_in[edge_src], edge_sh, weight)
        out = f_in.new_zeros(f_in.shape[0], self.irreps_out.dim)
        out.index_add_(0, edge_dst, msg)
        return out * inv_sqrt_deg[:, None]


class PoseRefinerNet(nn.Module):
    """Equivariant denoiser: (positions, chem features, time) -> 1o displacement."""

    def __init__(self, config: PoseRefinerConfig) -> None:
        super().__init__()
        self.config: PoseRefinerConfig = config
        h = config.hidden_dim
        lmax = config.l_max

        # Per-field categorical embeddings summed into the 0e scalar channel.
        self.embeds = nn.ModuleList(
            [nn.Embedding(vocab, h) for _, vocab in FEATURE_FIELDS]
        )
        self.time_dim: int = h
        self.time_mlp = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))

        # Bond-graph edge feature (invariant): tells the network which close
        # ligand pairs are BONDED (keep ~1.3-1.5 A) vs non-bonded (open up),
        # injected into the radial features that weight every message. Without
        # it the net cannot tell a bond from a clash and distorts intramolecular
        # geometry (bond lengths/angles) -> low PoseBusters validity.
        # 0 = non-bond, 1 = bond (binary; order can be added later).
        # Zero-init so training STARTS identical to the (stable) no-bond model
        # -- edge_radial is pure Bessel at step 0 -- and the bond signal is
        # blended in gradually. A non-zero init perturbs the radial features
        # from step 0 and destabilises training (observed divergence/OOM).
        self.bond_embed = nn.Embedding(2, config.num_radial)
        nn.init.zeros_(self.bond_embed.weight)

        # Hidden irreps: even + odd scalars, plus both parities for each l>=1.
        scal_o = max(4, h // 8)
        spec = f"{h}x0e + {scal_o}x0o"
        for ell in range(1, lmax + 1):
            mul = max(4, h // (2 ** (ell + 1)))
            spec += f" + {mul}x{ell}o + {mul}x{ell}e"
        self.gate = _make_gate(o3.Irreps(spec))
        self.irreps_hidden = self.gate.irreps_out
        self.irreps_sh: o3.Irreps = o3.Irreps.spherical_harmonics(lmax)

        # Lift the input scalars (h x 0e) into the hidden irreps; higher-l
        # channels start at zero and are built up by the convolutions. Pocket
        # nodes (no incoming edges) keep this lifted representation unchanged
        # through the residual stack, so they act as fixed context.
        self.embed = o3.Linear(o3.Irreps(f"{h}x0e"), self.irreps_hidden)
        self.convs = nn.ModuleList(
            _Convolution(
                self.irreps_hidden,
                self.irreps_sh,
                self.gate.irreps_in,
                config.num_radial,
                config.radial_hidden,
            )
            for _ in range(config.n_layers)
        )
        self.final = _Convolution(
            self.irreps_hidden,
            self.irreps_sh,
            o3.Irreps("1x1o"),
            config.num_radial,
            config.radial_hidden,
        )

    def forward(  # noqa: PLR0913
        self,
        pos: Tensor,  # (N, 3) current positions (ligand + pocket)
        feat: Tensor,  # (N, NUM_FEATURE_FIELDS) int categorical
        t_node: Tensor,  # (N,) diffusion time per node
        edge_src: Tensor,  # (E,) source node index
        edge_dst: Tensor,  # (E,) dest node index (always a ligand/movable node)
        movable: Tensor,  # (N,) bool, True for ligand nodes
        edge_bond: Tensor,  # (E,) int: 1 if the edge is a ligand bond, else 0
    ) -> Tensor:
        n = pos.shape[0]
        # --- input scalars: chem embeddings + time -------------------------
        scal = sum(emb(feat[:, i]) for i, emb in enumerate(self.embeds))
        scal = scal + self.time_mlp(_timestep_embedding(t_node, self.time_dim))
        h = self.embed(scal)

        # --- edge geometry (shared across layers) --------------------------
        rel = pos[edge_dst] - pos[edge_src]
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh, rel, normalize=True, normalization="component"
        )
        edge_len = rel.norm(dim=1)
        edge_radial = soft_one_hot_linspace(
            edge_len,
            0.0,
            self.config.pocket_cutoff,
            self.config.num_radial,
            basis="bessel",
            cutoff=True,
        ).mul(self.config.num_radial**0.5)
        # inject the bond-graph signal into the (invariant) radial features
        edge_radial = edge_radial + self.bond_embed(edge_bond)
        deg = pos.new_zeros(n).index_add_(0, edge_dst, torch.ones_like(edge_len))
        inv_sqrt_deg = deg.clamp(min=1.0).rsqrt()

        # --- residual message passing (pocket nodes stay fixed) ------------
        for conv in self.convs:
            h = h + self.gate(
                conv(h, edge_src, edge_dst, edge_sh, edge_radial, inv_sqrt_deg)
            )
        delta = self.final(h, edge_src, edge_dst, edge_sh, edge_radial, inv_sqrt_deg)
        return delta * movable[:, None]  # zero any residual pocket motion


class PoseRefinerModule(L.LightningModule):
    """Flow-matching trainer for :class:`PoseRefinerNet` (x1-prediction)."""

    def __init__(self, config: PoseRefineTrainingConfig) -> None:
        super().__init__()
        self.config: PoseRefineTrainingConfig = config
        self.save_hyperparameters()
        self.net = PoseRefinerNet(config.model)

    # -- geometry helpers ---------------------------------------------------
    def _interpolate(self, x0: Tensor, x1: Tensor, t_node: Tensor) -> Tensor:
        """Linear bridge x_t = (1-t) x0 + t x1 (+ optional Brownian-bridge noise)."""
        x_t = (1.0 - t_node[:, None]) * x0 + t_node[:, None] * x1
        sigma = self.config.model.bridge_sigma
        if sigma > 0:
            std = sigma * (t_node * (1.0 - t_node)).clamp(min=0).sqrt()
            x_t = x_t + std[:, None] * torch.randn_like(x_t)
        return x_t

    def _angle_loss(self, x: Tensor, batch: dict[str, Tensor]) -> Tensor:
        """Bond-angle loss: match the cosine of each bonded i-j-k angle to native
        (batch["pos1"]) -- directly targets the PoseBusters bond-angle check."""
        at = batch["angle_triples"]  # (3, T) global indices
        if at.shape[1] == 0:
            return x.new_zeros(())

        def cos(p: Tensor) -> Tensor:
            v1, v2 = p[at[0]] - p[at[1]], p[at[2]] - p[at[1]]
            return (v1 * v2).sum(-1) / (v1.norm(dim=1) * v2.norm(dim=1)).clamp(min=1e-6)

        return (cos(x) - cos(batch["pos1"])).pow(2).mean()

    def _physical_losses(
        self, x: Tensor, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Clash (intra-ligand + ligand-pocket) and bond-length losses on ``x``.

        ``d_floor`` alone has to sit below a bond length, which makes the term
        nearly inert: measured on held-out reconstructions, 68.5% still hold a
        non-bonded pair under 2.0 A and the refiner only lifts the closest
        non-bonded contact to 2.10 A where a force field reaches 2.29 A.

        ``nonbond_floor`` lifts that ceiling by scoring only the pairs the
        CRYSTAL (``pos1``) already keeps at least that far apart. A bond, or a
        1-3 contact inside a ring, is excluded by its own reference distance,
        so no bond graph is needed and the floor may exceed a bond length. The
        term then reads "do not invent a contact the crystal does not have".
        """
        m = self.config.model
        d_floor = m.d_floor
        nonbond = getattr(m, "nonbond_floor", None)
        ref = batch["pos1"]

        def _hinge(idx: Tensor) -> Tensor:
            d = (x[idx[0]] - x[idx[1]]).norm(dim=1)
            if nonbond is None:
                return F.relu(d_floor - d).pow(2).mean()
            d_ref = (ref[idx[0]] - ref[idx[1]]).norm(dim=1)
            keep = d_ref >= nonbond
            if not bool(keep.any()):
                return x.new_zeros(())
            return F.relu(nonbond - d[keep]).pow(2).mean()

        # intra-ligand clash over ligand-ligand edges
        ll = batch["ll_edge"]  # (2, E_ll) undirected pairs (i < j)
        clash = _hinge(ll) if ll.shape[1] > 0 else x.new_zeros(())
        # ligand-pocket clash over close ligand-pocket pairs
        lp = batch["lp_edge"]  # (2, E_lp): row0 ligand, row1 pocket
        pkt = _hinge(lp) if lp.shape[1] > 0 else x.new_zeros(())
        # bonded-distance anchor (topology / anti-collapse)
        bd = batch["bond_edge"]  # (2, B)
        if bd.shape[1] > 0:
            d = (x[bd[0]] - x[bd[1]]).norm(dim=1)
            bond = (d - batch["bond_ref"]).pow(2).mean()
        else:
            bond = x.new_zeros(())
        return clash, pkt, bond

    def _ramp(self) -> float:
        steps = self.config.model.lambda_ramp_steps
        if steps <= 0:
            return 1.0
        try:
            gs = self.global_step
        except RuntimeError:  # not attached to a Trainer (e.g. a smoke test)
            gs = 0
        return min(1.0, gs / steps)

    def _predict_x1(self, x_t: Tensor, t_node: Tensor, batch: dict) -> Tensor:
        delta = self.net(
            x_t,
            batch["feat"],
            t_node,
            batch["edge_src"],
            batch["edge_dst"],
            batch["movable"],
            batch["edge_bond"],
        )
        return x_t + delta

    def _compute(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Flow-matching loss (+ physical aux) for one batch, no logging."""
        x0, x1 = batch["pos0"], batch["pos1"]
        # one diffusion time per graph, broadcast to nodes
        t = torch.rand(int(batch["num_graphs"]), device=x0.device)
        t_node = t[batch["batch"]]
        x_t = self._interpolate(x0, x1, t_node)
        x_hat1 = self._predict_x1(x_t, t_node, batch)

        mov = batch["movable"]
        recon = (x_hat1[mov] - x1[mov]).pow(2).sum(-1).mean()
        clash, pkt, bond = self._physical_losses(x_hat1, batch)
        angle = self._angle_loss(x_hat1, batch)
        m = self.config.model
        w = self._ramp()
        loss = (
            recon
            + w * m.lambda_clash * clash
            + w * m.lambda_pkt * pkt
            + m.lambda_bond * bond  # bond anchor is on from step 0 (anti-collapse)
            + m.lambda_angle * angle  # bond-angle -> PoseBusters angle validity
        )
        return {
            "loss": loss,
            "recon": recon,
            "clash": clash,
            "pkt": pkt,
            "bond": bond,
            "angle": angle,
        }

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        out = self._compute(batch)
        sync = stage != "train"
        bs = int(batch["num_graphs"])
        for name, val in out.items():
            self.log(
                f"{stage}/{name}",
                val,
                prog_bar=name == "loss",
                sync_dist=sync,
                batch_size=bs,
            )
        return out["loss"]

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: ARG002
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        self._step(batch, "val")
        # Full-inference sanity metric: does refine() actually move x0 toward x1?
        # Both are in the same canonical frame, so per-atom RMSD needs no Kabsch.
        mov = batch["movable"]
        refined = self.refine(batch)
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

    @torch.no_grad()
    def refine(self, batch: dict[str, Tensor], n_steps: int | None = None) -> Tensor:
        """Refine x0 -> clean pose (ligand only).

        The network is an x1-predictor (trained to regress the clean pose from
        any interpolant x_t), so ``n_steps <= 1`` does a single forward pass from
        x0 (t=0) that *directly* estimates x1. This is far more robust than the
        multi-step velocity ODE, whose Euler steps visit off-manifold points and
        compound the per-step error (early in training the ODE gives ~4 A while
        the single-shot estimate tracks the ~0.6 A prediction accuracy). The
        ODE path (``n_steps > 1``) is kept for ablation.
        """
        n_steps = self.config.model.n_flow_steps if n_steps is None else n_steps
        x = batch["pos0"].clone()
        mov = batch["movable"].unsqueeze(-1)
        if n_steps <= 1:
            t_node = x.new_zeros(x.shape[0])
            x_hat1 = self._predict_x1(x, t_node, batch)
            return torch.where(mov, x_hat1, x)
        movf = mov.to(x.dtype)
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=x.device)
        for i in range(n_steps):
            t = ts[i]
            t_node = x.new_full((x.shape[0],), float(t))
            x_hat1 = self._predict_x1(x, t_node, batch)
            v = (x_hat1 - x) / (1.0 - t).clamp(min=1e-3)
            x = x + (ts[i + 1] - ts[i]) * v * movf
        return x

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        decay, no_decay = [], []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            (decay if param.ndim >= 2 else no_decay).append(param)  # noqa: PLR2004
        opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
        )
        total = int(self.trainer.estimated_stepping_batches)
        warmup = max(1, min(self.config.warmup_steps, total - 1))
        sched = SequentialLR(
            opt,
            schedulers=[
                LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup),
                CosineAnnealingLR(
                    opt,
                    T_max=max(1, total - warmup),
                    eta_min=self.config.learning_rate * self.config.min_lr_ratio,
                ),
            ],
            milestones=[warmup],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


@torch.no_grad()
def refine_ligand_canonical(  # noqa: PLR0913
    module: PoseRefinerModule,
    lig_canonical: np.ndarray,  # (n_lig, 3) corrupted ligand coords, canonical frame
    lig_feat: np.ndarray,  # (n_lig, 9) int node features
    pkt_canonical: np.ndarray,  # (n_pkt, 3) pocket coords, same canonical frame
    pkt_feat: np.ndarray,  # (n_pkt, 9) int node features
    *,
    device: torch.device,
    bonds: np.ndarray | None = None,  # (B, 2) ligand bond pairs for the bond feature
    n_steps: int = 1,
) -> np.ndarray:
    """Refine one ligand pose in the pocket canonical frame; returns (n_lig, 3).

    Wraps the corrupted pose in a single-graph batch (reusing the training
    collate so edge construction matches exactly). Defaults to ``n_steps=1``
    (single-shot x1-prediction) -- the validated production inference. ``bonds``
    (perceived from the pose) drives the bond-graph edge feature and should be
    passed so the refiner preserves intramolecular geometry; without it every
    edge is treated as non-bonded.
    """
    # Imported here, not at module scope: pose_refine_dataset imports this
    # module for its feature layout, so a top-level import would be circular.
    from prolit.data.pose_refine_dataset import make_collate  # noqa: PLC0415

    m = module.config.model
    bp = (
        np.zeros((0, 2), dtype=np.int64)
        if bonds is None
        else np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    )
    sample = {
        "x0": lig_canonical.astype(np.float32),
        "x1": lig_canonical.astype(np.float32),  # unused by refine()
        "lig_feat": lig_feat.astype(np.int64),
        "bonds": bp,
        "bond_ref": np.zeros((bp.shape[0],), dtype=np.float32),
        "pkt_x": pkt_canonical.astype(np.float32),
        "pkt_feat": pkt_feat.astype(np.int64),
        "scale": 0.0,
    }
    batch = make_collate(m.pocket_cutoff, m.max_pocket_neighbors, m.ligand_knn)(
        [sample]
    )
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    refined = module.refine(batch, n_steps=n_steps)
    return refined[batch["movable"]].cpu().numpy()


__all__ = [
    "FEATURE_FIELDS",
    "LIG_CHEM_HEADS",
    "NUM_FEATURE_FIELDS",
    "PoseRefinerModule",
    "PoseRefinerNet",
    "ligand_feats_from_heads",
    "pocket_feats_from_descriptor",
    "refine_ligand_canonical",
]
