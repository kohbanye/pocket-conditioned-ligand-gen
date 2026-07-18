"""LightningDataModule for the pose-scoring head (complex tokens + RMSD label).

Reads the decoy token set from :mod:`scripts.tokenize_decoys`
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

from src.tokenizers.lm_vocab import L_CLOSE_ID, L_OPEN_ID, NUM_SPECIAL, PAD_ID

if TYPE_CHECKING:
    from src.config import RescoreTrainingConfig


def _ligand_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask: codebook tokens strictly inside the first ``<l>..</l>``."""
    mask = np.zeros(arr.shape[0], dtype=bool)
    lo = np.flatnonzero(arr == L_OPEN_ID)
    hi = np.flatnonzero(arr == L_CLOSE_ID)
    if lo.size == 0 or hi.size == 0:
        return mask
    start, end = int(lo[0]) + 1, int(hi[-1])
    mask[start:end] = arr[start:end] >= NUM_SPECIAL
    return mask


class RescoreDataset(Dataset):
    """One (pose tokens, ligand mask, RMSD) example per doc."""

    def __init__(
        self,
        bin_path: Path,
        len_path: Path,
        rmsd_path: Path,
        block_size: int,
        group_path: Path | None = None,
    ) -> None:
        self.block_size = block_size
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.lengths = np.fromfile(len_path, dtype=np.uint16).astype(np.int64)
        self.rmsd = np.fromfile(rmsd_path, dtype=np.float32)
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
        # Group ids may be sparse (protein ids), so index by value, not by range.
        by_group: dict[int, list[int]] = {}
        for i, g in enumerate(self.doc_group.tolist()):
            by_group.setdefault(g, []).append(i)
        self.groups = list(by_group.values())

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        s = int(self.offsets[idx])
        e = min(int(self.offsets[idx + 1]), s + self.block_size)
        arr = np.asarray(self.tokens[s:e], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(arr),
            "ligand_mask": torch.from_numpy(_ligand_mask(arr)),
            "rmsd": torch.tensor(float(self.rmsd[idx]), dtype=torch.float32),
            "length": torch.tensor(arr.shape[0], dtype=torch.int64),
            "group": torch.tensor(int(self.doc_group[idx]), dtype=torch.int64),
        }


class GroupBatchSampler(Sampler[list[int]]):
    """Yield batches spanning ``complexes_per_batch`` whole complexes, so a
    pairwise ranking loss has same-complex pose pairs within each batch."""

    def __init__(
        self,
        groups: list[list[int]],
        complexes_per_batch: int,
        *,
        shuffle: bool,
        seed: int = 0,
        max_per_group: int = 0,
    ) -> None:
        # Singletons are kept: they carry no pair, but the regression loss still
        # learns from them, and dropping them would throw away real data (for the
        # affinity corpus roughly half the proteins have only one ligand).
        self.groups = list(groups)
        self.k = max(1, complexes_per_batch)
        # Protein groups are wildly uneven (up to ~700 ligands); without a cap a
        # single group would blow up the batch. 0 = no cap (pose corpus, ~20/complex).
        self.max_per_group = max(0, max_per_group)
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
                if self.max_per_group and len(g) > self.max_per_group:
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
                )

    def _loader(self, split: str, *, shuffle: bool) -> DataLoader:
        nw = self.config.num_workers
        common = {
            "num_workers": nw,
            "persistent_workers": nw > 0,
            "pin_memory": True,
            "collate_fn": collate_rescore,
        }
        if self.config.ranking_loss_weight > 0:
            sampler = GroupBatchSampler(
                self._datasets[split].groups,
                self.config.complexes_per_batch,
                shuffle=shuffle,
                max_per_group=self.config.max_per_group,
            )
            self._samplers[split] = sampler
            return DataLoader(
                self._datasets[split], batch_sampler=sampler, **common
            )
        return DataLoader(
            self._datasets[split],
            batch_size=self.config.micro_batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            **common,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)
