"""LightningDataModule for the pose-scoring head (complex tokens + RMSD label).

Reads the decoy token set from :mod:`pipelines.corpora.tokenize_decoys`
(``{split}.bin`` uint16 tokens + ``{split}.len`` uint16 doc lengths +
``{split}.rmsd`` float32 per-doc RMSD). Serves one pose per example with a
ligand-token mask (the head mean-pools the encoder output over the
``<l>..</l>`` codebook positions) and the RMSD regression target.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from prolit.seeding import DEFAULT_SEED, rng_for, torch_generator, worker_init_fn
from prolit.tokenizers.lm_vocab import L_CLOSE_ID, L_OPEN_ID, NUM_SPECIAL, PAD_ID

if TYPE_CHECKING:
    from prolit.config import RescoreTrainingConfig


def ligand_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask: codebook tokens strictly inside the first ``<l>..</l>``."""
    mask = np.zeros(arr.shape[0], dtype=bool)
    lo = np.flatnonzero(arr == L_OPEN_ID)
    hi = np.flatnonzero(arr == L_CLOSE_ID)
    if lo.size == 0 or hi.size == 0:
        return mask
    start, end = int(lo[0]) + 1, int(hi[-1])
    mask[start:end] = arr[start:end] >= NUM_SPECIAL
    return mask


def _check_input_mask(side: str, pool: str, mask_token_id: int | None) -> None:
    """Reject an input-masking request that would silently do nothing."""
    if mask_token_id is None:
        msg = "input_mask_prob > 0 needs mask_token_id"
        raise ValueError(msg)
    if side not in ("ligand", "pocket", "both"):
        msg = f"unknown input_mask_side {side!r}"
        raise ValueError(msg)
    if pool not in ("keep", "drop"):
        msg = f"unknown input_mask_pool {pool!r}"
        raise ValueError(msg)


class RescoreDataset(Dataset[dict[str, Tensor]]):
    """One (pose tokens, ligand mask, RMSD) example per doc."""

    def __init__(  # noqa: PLR0913
        self,
        bin_path: Path,
        len_path: Path,
        rmsd_path: Path,
        block_size: int,
        group_path: Path | None = None,
        *,
        divide_by_size: bool = False,
        drop_native: bool = False,
        max_label: float = 0.0,
        max_docs: int = 0,
        rotations_per_complex: int = 1,
        input_mask_prob: float = 0.0,
        input_mask_side: str = "ligand",
        input_mask_pool: str = "keep",
        mask_token_id: int | None = None,
        seed: int = DEFAULT_SEED,
        teacher_path: Path | None = None,
    ) -> None:
        self.block_size = block_size
        # Train on ligand efficiency (pK / heavy-atom count) instead of raw pK.
        # The all-atom tokenizer emits one token per heavy atom, so the ligand
        # token count IS the size -- no extra sidecar needed. Efficiency strips
        # the molecular-size trend the head otherwise rides (pK-size corr 0.37),
        # forcing it onto contact quality; eval multiplies back by size.
        self.divide_by_size = divide_by_size
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.lengths = np.fromfile(len_path, dtype=np.uint16).astype(np.int64)
        self.rmsd = np.fromfile(rmsd_path, dtype=np.float32)
        self.teacher = self._load_teacher(teacher_path)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)]).astype(np.int64)
        if group_path is not None and group_path.exists():
            # Explicit group ids, one per doc (affinity corpus: the protein).
            # The RMSD==0 heuristic below cannot work there -- that stream holds
            # pK, which is never exactly 0, so every doc would silently collapse
            # into a single group.
            self.doc_group = np.fromfile(group_path, dtype=np.int32).astype(np.int64)
        else:
            # Complex boundaries for ranking loss: each complex is written as its
            # native pose (RMSD exactly 0.0) followed by decoys, so RMSD==0.0
            # marks a new complex. doc_group[i] = complex index of doc i.
            starts = np.flatnonzero(self.rmsd == 0.0)
            if starts.size == 0:
                starts = np.array([0], dtype=np.int64)
            self.doc_group = (
                np.searchsorted(starts, np.arange(len(self.lengths)), side="right") - 1
            ).clip(min=0)
        # Hard-example specialist: keep only poses at or below ``max_label`` A.
        # Docking power is decided among the few near-native candidates (the best
        # available pose already sits at median rank 3), so a head that never
        # sees the easy 6-8 A decoys spends all of its capacity on the
        # distinctions that actually pick the winner.
        self.index = np.arange(len(self.lengths), dtype=np.int64)
        # Corpus-size ablation / fast iteration: docs are written complex by
        # complex, so a prefix is a subset of complexes with all their poses
        # (never a partial pose set, which would break the grouped losses).
        if max_docs > 0:
            self.index = self.index[:max_docs]
        if max_label > 0:
            self.index = self.index[self.rmsd[self.index] <= max_label]
        if drop_native:
            # Every complex here carries its crystal pose at RMSD exactly 0 and
            # its decoys are that pose perturbed, so "distance from the crystal
            # conformer" and "RMSD" are the SAME quantity in training -- and the
            # head learns the former as a stand-in for the latter. Measured on
            # CASF with RMSD held fixed within each complex, its score
            # correlates +0.484 with distance-from-crystal-conformer where
            # RTMScore's correlates +0.023. That is the largest partial
            # correlation found for any quantity, and it points the wrong way:
            # CASF's answers are docked poses sitting 0.3-0.8 A off the crystal
            # conformer, so the stand-in inverts. Dropping the exact-zero pose
            # leaves a set whose best member is merely near-native, which is the
            # comparison CASF actually poses.
            self.index = self.index[self.rmsd[self.index] > 0.0]
        # Rotation augmentation: index one entry per COMPLEX, draw one of its R
        # orientations per access. An epoch is then complex-sized (not R times
        # larger, which is how e250_rot8 memorised its corpus) while the frame
        # varies across epochs. See RescoreTrainingConfig.rotations_per_complex.
        self.rotations = max(1, int(rotations_per_complex))
        self._rot_rng = rng_for(seed, "rotation-sampling")
        # Input masking (train split only; the datamodule passes 0 for val).
        # Its own stream, so turning it on does not shift the rotation draws --
        # an arm that changed both at once would not be a one-flag control.
        self.input_mask_prob = float(input_mask_prob)
        self.input_mask_side = input_mask_side
        self.input_mask_pool = input_mask_pool
        self.mask_token_id = mask_token_id
        self._mask_rng = rng_for(seed, "input-masking")
        if self.input_mask_prob > 0:
            _check_input_mask(input_mask_side, input_mask_pool, mask_token_id)
        if self.rotations > 1:
            self.index = self._complex_index(
                per_doc_selection=bool(
                    max_docs or max_label > 0 or drop_native or divide_by_size
                )
            )
        keep = set(self.index.tolist())
        # Group ids may be sparse (protein ids), so index by value, not by range.
        # Groups index into THIS dataset's positions, not raw doc ids.
        pos = {int(d): i for i, d in enumerate(self.index)}
        by_group: dict[int, list[int]] = {}
        for d in self.index.tolist():
            if d in keep:
                by_group.setdefault(int(self.doc_group[d]), []).append(pos[d])
        self.groups = list(by_group.values())

    def _load_teacher(self, teacher_path: Path | None) -> np.ndarray | None:
        """Teacher predictions for distillation, one per DOCUMENT in file order.

        Served with the same index as the label, so a rotation block hands over
        the teacher value of the orientation it actually drew. A length mismatch
        means the sidecar was built against a different corpus, which would
        silently pair every document with someone else's teacher value.
        """
        if teacher_path is None:
            return None
        teacher = np.load(teacher_path).astype(np.float32)
        if teacher.shape != self.rmsd.shape:
            msg = (f"teacher sidecar has {teacher.shape} values for "
                   f"{self.rmsd.shape} documents")
            raise ValueError(msg)
        return teacher

    def _complex_index(self, *, per_doc_selection: bool) -> np.ndarray:
        """First doc of each complex, for a corpus of R orientations per complex.

        The corpus is written complex by complex, R consecutive docs each, so
        the complexes start at ``0, R, 2R, ...``. Every doc of one complex must
        carry the same label and the same group: a draw changes the ORIENTATION
        and nothing else, and a corpus that breaks that would silently move the
        target along with the frame.
        """
        if per_doc_selection:
            msg = (
                "rotations_per_complex > 1 does not compose with max_docs / "
                "max_label / drop_native / divide_by_size: those select or "
                "rescale per DOC, and a doc is no longer a complex here"
            )
            raise ValueError(msg)
        n_docs = len(self.lengths)
        if n_docs % self.rotations:
            msg = (
                f"{n_docs} docs is not a multiple of {self.rotations} "
                "rotations; the corpus is not R orientations per complex"
            )
            raise ValueError(msg)
        labels = self.rmsd.reshape(-1, self.rotations)
        groups = self.doc_group.reshape(-1, self.rotations)
        if not np.all(labels == labels[:, :1]):
            msg = (
                "a rotation block holds more than one label; "
                "docs are not grouped by complex"
            )
            raise ValueError(msg)
        if not np.all(groups == groups[:, :1]):
            msg = "a rotation block spans more than one group"
            raise ValueError(msg)
        return np.arange(n_docs // self.rotations, dtype=np.int64) * self.rotations

    def _apply_input_mask(self, arr: np.ndarray, lig: np.ndarray) -> np.ndarray:
        """Replace a share of the codebook tokens with ``<mask>``, in place.

        Returns the (possibly narrowed) ligand mask. ``arr`` is a fresh copy of
        the memmap slice, so writing into it does not touch the corpus.

        A masked position stops being a codebook id, so
        :meth:`ComplexRescoreModule._component_masks` -- which derives the
        protein side from the ids -- drops masked POCKET rows from the protein
        mean on its own. Only the ligand side is a choice, because its mask
        comes from here: ``keep`` leaves the row in, so the readout averages the
        state the encoder inferred for it, and ``drop`` takes it out.
        """
        is_code = arr >= NUM_SPECIAL
        if self.input_mask_side == "ligand":
            pool = np.flatnonzero(lig)
        elif self.input_mask_side == "pocket":
            pool = np.flatnonzero(is_code & ~lig)
        else:
            pool = np.flatnonzero(is_code)
        if pool.size == 0:
            return lig
        k = max(1, round(self.input_mask_prob * pool.size))
        chosen = self._mask_rng.choice(pool, size=k, replace=False)
        arr[chosen] = self.mask_token_id
        if self.input_mask_pool == "drop":
            lig = lig.copy()
            lig[chosen] = False
        return lig

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        idx = int(self.index[index])
        if self.rotations > 1:
            # One orientation per draw; the label and group are identical across
            # the block (asserted in __init__), so only the frame changes.
            idx += int(self._rot_rng.integers(self.rotations))
        s = int(self.offsets[idx])
        e = min(int(self.offsets[idx + 1]), s + self.block_size)
        arr = np.asarray(self.tokens[s:e], dtype=np.int64)
        lig = ligand_mask(arr)
        if self.input_mask_prob > 0:
            lig = self._apply_input_mask(arr, lig)
        label = float(self.rmsd[idx])
        if self.divide_by_size:
            label /= max(int(lig.sum()), 1)
        item = {
            "input_ids": torch.from_numpy(arr),
            "ligand_mask": torch.from_numpy(lig),
            "rmsd": torch.tensor(label, dtype=torch.float32),
            "length": torch.tensor(arr.shape[0], dtype=torch.int64),
            "group": torch.tensor(int(self.doc_group[idx]), dtype=torch.int64),
        }
        if self.teacher is not None:
            item["teacher"] = torch.tensor(
                float(self.teacher[idx]), dtype=torch.float32
            )
        return item


class GroupBatchSampler(Sampler[list[int]]):
    """Yield batches spanning ``complexes_per_batch`` whole complexes, so a
    pairwise ranking loss has same-complex pose pairs within each batch."""

    def __init__(  # noqa: PLR0913
        self,
        groups: list[list[int]],
        complexes_per_batch: int,
        *,
        shuffle: bool,
        seed: int = 0,
        max_per_group: int = 0,
        partition: bool = False,
    ) -> None:
        # Singletons are kept: they carry no pair, but the regression loss still
        # learns from them, and dropping them would throw away real data (for the
        # affinity corpus roughly half the proteins have only one ligand).
        self.groups = list(groups)
        self.k = max(1, complexes_per_batch)
        # Protein groups are wildly uneven (up to ~700 ligands); without a cap a
        # single group would blow up the batch. 0 = no cap (pose corpus, ~20/complex).
        self.max_per_group = max(0, max_per_group)
        # Partition instead of subsample: a group of 40 with a cap of 4 becomes
        # 10 units that land in 10 different batches, so the epoch still covers
        # every document once. Subsampling would show 4 of those 40 per epoch,
        # which on the union corpus with real protein groups meant an epoch saw
        # 46% of the documents.
        self.partition = partition
        if partition and self.max_per_group:
            units: list[list[int]] = []
            cap = self.max_per_group
            for g in self.groups:
                units.extend(g[i : i + cap] for i in range(0, len(g), cap))
            self.groups = units
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return (len(self.groups) + self.k - 1) // self.k

    def __iter__(self):  # noqa: ANN204
        rng = np.random.default_rng(self.seed + self.epoch)
        order = list(range(len(self.groups)))
        if self.shuffle:
            rng.shuffle(order)
            self.epoch += 1  # reshuffle next epoch (Lightning won't call set_epoch)
        for i in range(0, len(order), self.k):
            batch: list[int] = []
            for gi in order[i : i + self.k]:
                g = self.groups[gi]
                oversized = self.max_per_group and len(g) > self.max_per_group
                if not self.partition and oversized:
                    sel = rng.choice(len(g), self.max_per_group, replace=False)
                    g = [g[j] for j in sorted(sel.tolist())]
                batch.extend(g)
            if batch:
                yield batch


def collate_rescore(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    lengths = [int(b["length"]) for b in batch]
    max_len = max(lengths)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    ligand_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    rmsd = torch.empty(bsz, dtype=torch.float32)
    for i, b in enumerate(batch):
        n = lengths[i]
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = 1
        ligand_mask[i, :n] = b["ligand_mask"]
        rmsd[i] = b["rmsd"]
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ligand_mask": ligand_mask,
        "rmsd": rmsd,
    }
    if "teacher" in batch[0]:
        out["teacher"] = torch.stack([b["teacher"] for b in batch])
    if "group" in batch[0]:
        # Remap absolute complex ids to 0..K-1 within this batch for the loss.
        raw = torch.tensor([int(b["group"]) for b in batch])
        _, remap = torch.unique(raw, return_inverse=True)
        out["group_ids"] = remap
    return out


class RescoreDataModule(L.LightningDataModule):
    """Serves per-pose decoy examples for train/val splits."""

    def __init__(self, config: RescoreTrainingConfig) -> None:
        super().__init__()
        self.config = config
        self._seed = getattr(config, "seed", DEFAULT_SEED)
        self.token_dir = Path(config.token_dir)
        self._datasets: dict[str, RescoreDataset] = {}
        self._samplers: dict[str, GroupBatchSampler] = {}

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        for split in ("train", "val"):
            bin_path = self.token_dir / f"{split}.bin"
            if bin_path.exists():
                self._datasets[split] = RescoreDataset(
                    bin_path,
                    self.token_dir / f"{split}.len",
                    self.token_dir / f"{split}.rmsd",
                    self.config.block_size,
                    group_path=self.token_dir / f"{split}.grp",
                    divide_by_size=self.config.label_divide_by_size,
                    max_label=self.config.max_label,
                    max_docs=self.config.max_docs if split == "train" else 0,
                    drop_native=self.config.drop_native_pose,
                    # Val stays at one orientation per complex whatever the
                    # training corpus holds: the checkpoint-selection signal must
                    # not move when the augmentation does.
                    # Train only: val is the checkpoint-selection signal and
                    # must not be shifted toward a teacher.
                    teacher_path=(
                        Path(self.config.distill_path)
                        if split == "train" and getattr(self.config, "distill_path", "")
                        else None
                    ),
                    rotations_per_complex=(
                        getattr(self.config, "rotations_per_complex", 1)
                        if split == "train"
                        else 1
                    ),
                    # Train only, for the same reason the teacher is: val is
                    # the checkpoint-selection signal, and inference never
                    # masks, so a masked val would select on a protocol no
                    # reported number is produced under.
                    input_mask_prob=(
                        getattr(self.config, "input_mask_prob", 0.0)
                        if split == "train"
                        else 0.0
                    ),
                    input_mask_side=getattr(
                        self.config, "input_mask_side", "ligand"
                    ),
                    input_mask_pool=getattr(
                        self.config, "input_mask_pool", "keep"
                    ),
                    mask_token_id=self.config.model.mask_token_id,
                    seed=self._seed,
                )

    def _loader(self, split: str, *, shuffle: bool) -> DataLoader:
        nw = self.config.num_workers
        # Both the pairwise ranking loss and the listwise loss need whole
        # complexes in a batch, not a random sample of poses.
        # Both the listwise loss and the group-mean loss need whole groups in a
        # batch: with random batching a protein almost never appears twice, so
        # the group-mean term would collapse to the per-complex term.
        # ``group_batch_partition`` also turns grouped batching ON by itself.
        # Without that, a control with weight 0 silently fell back to random
        # batching while its treatment used whole-group batches -- the arms then
        # differ in batching AND in the loss, which is exactly the confound the
        # control exists to remove. Passing the flag is how a run asks for
        # grouped batching regardless of which grouped term (if any) is active.
        grouped = (
            self.config.listwise_loss_weight > 0
            or getattr(self.config, "group_mean_weight", 0.0) > 0
            or getattr(self.config, "group_batch_partition", False)
        )
        if grouped:
            sampler = GroupBatchSampler(
                self._datasets[split].groups,
                self.config.complexes_per_batch,
                shuffle=shuffle,
                max_per_group=self.config.max_per_group,
                partition=getattr(self.config, "group_batch_partition", False),
            )
            self._samplers[split] = sampler
            return DataLoader(
                self._datasets[split],
                batch_sampler=sampler,
                num_workers=nw,
                persistent_workers=nw > 0,
                pin_memory=True,
                collate_fn=collate_rescore,  # ty: ignore[invalid-argument-type]
            )
        return DataLoader(
            self._datasets[split],
            batch_size=self.config.micro_batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            # Reproducible shuffle order, and NumPy/random streams per
            # worker (torch seeds only its own RNG in workers).
            generator=torch_generator(self._seed, "rescore-shuffle"),
            worker_init_fn=worker_init_fn,
            num_workers=nw,
            persistent_workers=nw > 0,
            pin_memory=True,
            collate_fn=collate_rescore,  # ty: ignore[invalid-argument-type]
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)
