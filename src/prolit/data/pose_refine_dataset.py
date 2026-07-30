"""LightningDataModule for the e3nn pose refiner.

Reads the corruption->native tuples written by
:mod:`pipelines.corpora.tokenize_pose_refine`
as a small set of concatenated memmaps (inode-safe: the pocket is stored ONCE
per complex and referenced by pointer from each corruption record). One example
is one (corrupted ligand pose ``x0``, native ligand pose ``x1``, pocket) tuple;
``collate_pose_refine`` assembles a flat batched graph with a ``movable`` mask
and the edge lists the refiner + its physical losses need.

On-disk layout, per split (``train`` / ``val``):
- ``lig_x1``  f32 (sum N_lig, 3)      native canonical ligand coords (per complex)
- ``lig_x0``  f32 (sum N_lig_rec, 3)  corrupted canonical coords (per record)
- ``lig_feat`` i16 (sum N_lig, 9)     ligand categorical features (per complex)
- ``lig_bonds`` i32 (sum B, 2)        heavy-atom bond pairs, local ligand indices
- ``lig_bond_ref`` f32 (sum B,)       native bonded distance |x1_i - x1_j|
- ``pkt_x``   f32 (sum M, 3)          pocket canonical coords (per complex)
- ``pkt_feat`` i16 (sum M, 9)         pocket categorical features (per complex)
- ``complexes`` i64 (C, 3)            per complex: [n_lig, n_pkt, n_bonds]
- ``records``  i64 (R,)               per record: complex id
- ``record_scale`` f32 (R,)           per record: corruption scale in [0, 1]
- ``meta.json``
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from prolit.model.pose_refiner import NUM_FEATURE_FIELDS
from prolit.seeding import DEFAULT_SEED, rng_for, torch_generator, worker_init_fn

if TYPE_CHECKING:
    from collections.abc import Callable

    from prolit.config import PoseRefineTrainingConfig

FEAT = NUM_FEATURE_FIELDS


class PoseRefineDataset(Dataset):
    """One (x0, x1, pocket) refinement example per corruption record."""

    def __init__(  # noqa: PLR0913
        self,
        data_dir: Path,
        split: str,
        jitter_sigma: float = 0.0,
        rigid_trans: float = 0.0,
        rigid_rot_deg: float = 0.0,
        rigid_prob: float = 1.0,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.jitter_sigma = jitter_sigma
        self.rigid_trans = rigid_trans
        self.rigid_rot_deg = rigid_rot_deg
        self.rigid_prob = rigid_prob
        # Named stream so the jitter is reproducible and does not correlate
        # with anything else seeded from the same run seed.
        self.seed = seed
        self._rng = rng_for(seed, f"pose-refine-jitter:{split}")
        d = Path(data_dir)
        self.lig_x1 = np.memmap(
            d / f"{split}.lig_x1", dtype=np.float32, mode="r"
        ).reshape(-1, 3)
        self.lig_x0 = np.memmap(
            d / f"{split}.lig_x0", dtype=np.float32, mode="r"
        ).reshape(-1, 3)
        self.lig_feat = np.memmap(
            d / f"{split}.lig_feat", dtype=np.int16, mode="r"
        ).reshape(-1, FEAT)
        self.lig_bonds = np.memmap(
            d / f"{split}.lig_bonds", dtype=np.int32, mode="r"
        ).reshape(-1, 2)
        self.lig_bond_ref = np.memmap(
            d / f"{split}.lig_bond_ref", dtype=np.float32, mode="r"
        )
        self.pkt_x = np.memmap(
            d / f"{split}.pkt_x", dtype=np.float32, mode="r"
        ).reshape(-1, 3)
        self.pkt_feat = np.memmap(
            d / f"{split}.pkt_feat", dtype=np.int16, mode="r"
        ).reshape(-1, FEAT)

        cx = np.fromfile(d / f"{split}.complexes", dtype=np.int64).reshape(-1, 3)
        self.n_lig, self.n_pkt, self.n_bonds = cx[:, 0], cx[:, 1], cx[:, 2]
        # per-complex start offsets into the concatenated per-complex streams
        self.lig_off = np.concatenate([[0], np.cumsum(self.n_lig)]).astype(np.int64)
        self.pkt_off = np.concatenate([[0], np.cumsum(self.n_pkt)]).astype(np.int64)
        self.bond_off = np.concatenate([[0], np.cumsum(self.n_bonds)]).astype(np.int64)

        self.record_cid = np.fromfile(d / f"{split}.records", dtype=np.int64)
        self.record_scale = np.fromfile(d / f"{split}.record_scale", dtype=np.float32)
        # x0 offset per record = cumsum of the (per-record) ligand atom counts
        rec_nlig = self.n_lig[self.record_cid]
        self.x0_off = np.concatenate([[0], np.cumsum(rec_nlig)]).astype(np.int64)

    def __len__(self) -> int:
        return int(self.record_cid.shape[0])

    def __getitem__(self, idx: int) -> dict:
        c = int(self.record_cid[idx])
        nl, npk, nb = int(self.n_lig[c]), int(self.n_pkt[c]), int(self.n_bonds[c])
        lo, po, bo = int(self.lig_off[c]), int(self.pkt_off[c]), int(self.bond_off[c])
        x0o = int(self.x0_off[idx])
        x0 = np.asarray(self.lig_x0[x0o : x0o + nl], dtype=np.float32)
        if self.jitter_sigma > 0:  # intramolecular distortion for the net to repair
            x0 = x0 + self._rng.normal(0.0, self.jitter_sigma, x0.shape).astype(
                np.float32
            )
        do_rigid = (
            self.rigid_trans > 0 or self.rigid_rot_deg > 0
        ) and self._rng.random() < self.rigid_prob
        if do_rigid:
            # MISPLACEMENT corruption: LM poses are mostly mis-PLACED (rigid
            # offset/tilt in the pocket), not just locally distorted, so the
            # refiner must learn to slide the whole ligand back into the pocket.
            # Applied about the ligand centroid to keep the internal geometry
            # (which the VQ round-trip already perturbs) untouched.
            cen = x0.mean(axis=0, keepdims=True)
            if self.rigid_rot_deg > 0:
                ang = np.deg2rad(self._rng.normal(0.0, self.rigid_rot_deg))
                axis = self._rng.normal(size=3)
                axis /= np.linalg.norm(axis) + 1e-8
                k = np.array(
                    [
                        [0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0],
                    ]
                )
                rot = np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * (k @ k)
                x0 = ((x0 - cen) @ rot.T.astype(np.float32)) + cen
            if self.rigid_trans > 0:
                x0 = x0 + self._rng.normal(0.0, self.rigid_trans, 3).astype(np.float32)
        return {
            "x1": np.asarray(self.lig_x1[lo : lo + nl], dtype=np.float32),
            "x0": x0,
            "lig_feat": np.asarray(self.lig_feat[lo : lo + nl], dtype=np.int64),
            "bonds": np.asarray(self.lig_bonds[bo : bo + nb], dtype=np.int64),
            "bond_ref": np.asarray(self.lig_bond_ref[bo : bo + nb], dtype=np.float32),
            "pkt_x": np.asarray(self.pkt_x[po : po + npk], dtype=np.float32),
            "pkt_feat": np.asarray(self.pkt_feat[po : po + npk], dtype=np.int64),
            "scale": float(self.record_scale[idx]),
        }


def _sample_edges(
    x0_lig: np.ndarray, pkt: np.ndarray, cutoff: float, max_pkt: int, knn: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ligand-ligand and ligand-pocket edges from the corrupted pose.

    Returns (ll_pairs [2, E_ll]  i<j undirected, lp_pairs [2, E_lp] lig-then-pkt),
    both as LOCAL indices (ligand 0..nl-1, pocket 0..npk-1). Connectivity is fixed
    from ``x0`` (the pose being refined) and reused across ODE steps.
    """
    nl = x0_lig.shape[0]
    # ligand-ligand
    dll = np.linalg.norm(x0_lig[:, None] - x0_lig[None], axis=-1)
    iu, ju = np.triu_indices(nl, k=1)
    if knn > 0 and nl > knn + 1:
        # keep, per atom, its knn nearest; symmetrise
        keep = np.zeros((nl, nl), dtype=bool)
        order = np.argsort(dll, axis=1)[:, 1 : knn + 1]
        rows = np.repeat(np.arange(nl), order.shape[1])
        keep[rows, order.reshape(-1)] = True
        keep = keep | keep.T
        m = keep[iu, ju]
    else:
        m = dll[iu, ju] < cutoff
    ll = np.stack([iu[m], ju[m]])

    # ligand-pocket (cap per ligand atom at the nearest max_pkt within cutoff)
    lp_l, lp_p = [], []
    if pkt.shape[0] > 0:
        dlp = np.linalg.norm(x0_lig[:, None] - pkt[None], axis=-1)  # (nl, npk)
        for i in range(nl):
            cand = np.flatnonzero(dlp[i] < cutoff)
            if cand.size > max_pkt:
                cand = cand[np.argsort(dlp[i, cand])[:max_pkt]]
            lp_l.append(np.full(cand.shape, i, dtype=np.int64))
            lp_p.append(cand.astype(np.int64))
    lp = (
        np.stack([np.concatenate(lp_l), np.concatenate(lp_p)])
        if lp_l and sum(len(a) for a in lp_l)
        else np.zeros((2, 0), dtype=np.int64)
    )
    return ll, lp, iu, ju  # iu/ju unused downstream but cheap to return


def make_collate(cutoff: float, max_pkt: int, knn: int) -> Callable:
    """Build a collate_fn that assembles a flat batched graph.

    Node order per sample: ligand atoms (movable) then pocket atoms (frozen).
    Message-passing edges (``edge_src``/``edge_dst``) always point INTO a ligand
    node: ligand-ligand (both directions) + pocket->ligand. Physical-loss edge
    lists (``ll_edge``/``lp_edge``/``bond_edge``) are undirected pair lists.
    """

    def collate(batch: list[dict]) -> dict[str, Tensor]:
        pos0, pos1, feat, movable, bvec = [], [], [], [], []
        esrc, edst, ebond = [], [], []
        ll_e, lp_e, bond_e, bond_ref, angle_t = [], [], [], [], []
        off = 0
        for b, s in enumerate(batch):
            nl, npk = s["x0"].shape[0], s["pkt_x"].shape[0]
            pos0.append(np.concatenate([s["x0"], s["pkt_x"]]))
            pos1.append(np.concatenate([s["x1"], s["pkt_x"]]))
            feat.append(np.concatenate([s["lig_feat"], s["pkt_feat"]]))
            mv = np.zeros(nl + npk, dtype=bool)
            mv[:nl] = True
            movable.append(mv)
            bvec.append(np.full(nl + npk, b, dtype=np.int64))

            lig_g = off  # ligand block starts at off; pocket block at off+nl
            pkt_g = off + nl
            bset = {
                (int(min(a, c)), int(max(a, c))) for a, c in s["bonds"].reshape(-1, 2)
            }
            ll, lp, _, _ = _sample_edges(s["x0"], s["pkt_x"], cutoff, max_pkt, knn)
            # message passing: ll both directions + pocket->ligand
            if ll.shape[1]:
                i, j = ll[0] + lig_g, ll[1] + lig_g
                esrc.append(np.concatenate([i, j]))
                edst.append(np.concatenate([j, i]))
                ll_e.append(np.stack([i, j]))
                flags = np.array(
                    [
                        1 if (int(ll[0][k]), int(ll[1][k])) in bset else 0
                        for k in range(ll.shape[1])
                    ],
                    dtype=np.int64,
                )
                ebond.append(np.concatenate([flags, flags]))  # i->j and j->i
            if lp.shape[1]:
                lg, pg = lp[0] + lig_g, lp[1] + pkt_g
                esrc.append(pg)
                edst.append(lg)
                lp_e.append(np.stack([lg, pg]))
                ebond.append(np.zeros(lp.shape[1], dtype=np.int64))  # pocket = non-bond
            if s["bonds"].shape[0]:
                bp = s["bonds"].T + lig_g  # (2, B) local ligand -> global
                bond_e.append(bp)
                bond_ref.append(s["bond_ref"])
                # bond-angle triples (i-j-k where i-j and j-k are bonds), for the
                # angle loss that targets PoseBusters bond-angle checks.
                nbr: dict[int, list[int]] = {}
                for a, c in s["bonds"].reshape(-1, 2):
                    nbr.setdefault(int(a), []).append(int(c))
                    nbr.setdefault(int(c), []).append(int(a))
                tri = [
                    (ns[x], j, ns[y])
                    for j, ns in nbr.items()
                    for x in range(len(ns))
                    for y in range(x + 1, len(ns))
                ]
                if tri:
                    angle_t.append(np.asarray(tri, dtype=np.int64).T + lig_g)  # (3, T)
            off += nl + npk

        def _cat(xs, cols, dt):  # noqa: ANN001, ANN202
            return (
                np.concatenate(xs, axis=1 if cols == 2 else 0)  # noqa: PLR2004
                if xs
                else np.zeros((2, 0) if cols == 2 else (0,), dtype=dt)  # noqa: PLR2004
            )

        return {
            "pos0": torch.from_numpy(np.concatenate(pos0)).float(),
            "pos1": torch.from_numpy(np.concatenate(pos1)).float(),
            "feat": torch.from_numpy(np.concatenate(feat)).long(),
            "movable": torch.from_numpy(np.concatenate(movable)),
            "batch": torch.from_numpy(np.concatenate(bvec)),
            "edge_src": torch.from_numpy(np.concatenate(esrc)).long(),
            "edge_dst": torch.from_numpy(np.concatenate(edst)).long(),
            "edge_bond": torch.from_numpy(np.concatenate(ebond)).long()
            if ebond
            else torch.zeros(0, dtype=torch.long),
            "ll_edge": torch.from_numpy(_cat(ll_e, 2, np.int64)).long(),
            "lp_edge": torch.from_numpy(_cat(lp_e, 2, np.int64)).long(),
            "bond_edge": torch.from_numpy(_cat(bond_e, 2, np.int64)).long(),
            "bond_ref": torch.from_numpy(_cat(bond_ref, 1, np.float32)).float(),
            "angle_triples": torch.from_numpy(
                np.concatenate(angle_t, axis=1)
                if angle_t
                else np.zeros((3, 0), dtype=np.int64)
            ).long(),
            "num_graphs": len(batch),
        }

    return collate


class PoseRefineDataModule(L.LightningDataModule):
    """Serves corruption->native refinement examples for train/val."""

    def __init__(self, config: PoseRefineTrainingConfig) -> None:
        super().__init__()
        self.config = config
        self._seed = getattr(config, "seed", DEFAULT_SEED)
        self.data_dir = Path(config.data_dir)
        self._datasets: dict[str, PoseRefineDataset] = {}

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        for split in ("train", "val"):
            if (self.data_dir / f"{split}.records").exists():
                train = split == "train"
                jitter = self.config.online_jitter_sigma if train else 0.0
                self._datasets[split] = PoseRefineDataset(
                    self.data_dir,
                    split,
                    jitter,
                    rigid_trans=self.config.online_rigid_trans if train else 0.0,
                    rigid_rot_deg=self.config.online_rigid_rot_deg if train else 0.0,
                    rigid_prob=self.config.online_rigid_prob,
                    seed=self._seed,
                )

    def _loader(self, split: str, *, shuffle: bool) -> DataLoader:
        m = self.config.model
        nw = self.config.num_workers
        return DataLoader(
            self._datasets[split],
            batch_size=self.config.micro_batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            # Reproducible shuffle order, and NumPy/random streams per
            # worker (torch seeds only its own RNG in workers).
            generator=torch_generator(self._seed, "pose-refine-shuffle"),
            worker_init_fn=worker_init_fn,
            num_workers=nw,
            persistent_workers=nw > 0,
            pin_memory=True,
            collate_fn=make_collate(
                m.pocket_cutoff, m.max_pocket_neighbors, m.ligand_knn
            ),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)


__all__ = ["PoseRefineDataModule", "PoseRefineDataset", "make_collate"]
