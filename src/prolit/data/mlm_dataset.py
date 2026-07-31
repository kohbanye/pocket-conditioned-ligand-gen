"""LightningDataModule for the bidirectional complex-token MLM.

Reads the SAME token cache as the decoder LM (``{split}.bin`` uint16 stream +
``{split}.len`` uint16 doc lengths, produced by the ``tokenize_*`` scripts) but
serves **one complex per example** (not packed blocks) with BERT-style dynamic
masking and a padding attention mask for full bidirectional attention.

Masking rules (:class:`~prolit.config.MLMTrainingConfig`):

- Only *codebook* tokens (id ``>= NUM_SPECIAL``) are maskable — the structure
  markers ``<bos> <p> </p> <l> </l> <eos>`` are never corrupted, so the model
  always knows the protein/ligand block boundaries.
- ``ligand_only_masking`` restricts masking to the ``<l>..</l>`` span, yielding
  a condition-only MLM that models ``P(ligand | pocket)`` bidirectionally.
- Of the selected positions: ``mask_replace_prob`` -> ``<mask>``,
  ``mask_random_prob`` -> a random codebook token, remainder -> unchanged.
- ``labels`` hold the original token at masked positions and ``IGNORE_INDEX``
  everywhere else; at least one position is masked per example when maskable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from prolit.seeding import DEFAULT_SEED, derive_seed, torch_generator, worker_init_fn
from prolit.tokenizers.lm_vocab import L_CLOSE_ID, L_OPEN_ID, NUM_SPECIAL, PAD_ID

if TYPE_CHECKING:
    from prolit.config import MLMTrainingConfig

IGNORE_INDEX = -100


class MLMTokenDataset(Dataset[dict[str, Tensor]]):
    """Yields one complex per item with BERT-style masked inputs + labels."""

    def __init__(  # noqa: PLR0913
        self,
        bin_path: Path,
        len_path: Path,
        *,
        block_size: int,
        base_vocab_size: int,
        mask_token_id: int,
        mask_prob: float = 0.15,
        mask_replace_prob: float = 0.8,
        mask_random_prob: float = 0.1,
        ligand_only_masking: bool = False,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.block_size = block_size
        self.seed = seed
        # Masking is *dynamic*: the same document is masked differently each
        # epoch, which is the point of BERT-style pretraining. Folding the epoch
        # into the per-item seed keeps that while making the whole schedule
        # reproducible -- an unseeded generator gave dynamism but no
        # reproducibility, and a seed without the epoch would give the reverse.
        self.epoch = 0
        self.base_vocab_size = base_vocab_size
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.mask_replace_prob = mask_replace_prob
        self.mask_random_prob = mask_random_prob
        self.ligand_only_masking = ligand_only_masking

        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.doc_lengths = np.fromfile(len_path, dtype=np.uint16).astype(np.int64)
        self.doc_offsets = np.concatenate([[0], np.cumsum(self.doc_lengths)]).astype(
            np.int64
        )

    def __len__(self) -> int:
        return len(self.doc_lengths)

    def _maskable_positions(self, arr: np.ndarray) -> np.ndarray:
        """Indices eligible for masking (codebook tokens, optionally ligand-only)."""
        is_code = arr >= NUM_SPECIAL
        if not self.ligand_only_masking:
            return np.flatnonzero(is_code)
        # Positions strictly between the first <l> and the following </l>.
        lo = np.flatnonzero(arr == L_OPEN_ID)
        hi = np.flatnonzero(arr == L_CLOSE_ID)
        if lo.size == 0 or hi.size == 0:
            return np.empty(0, dtype=np.int64)
        start, end = int(lo[0]) + 1, int(hi[-1])
        in_span = np.zeros(arr.shape[0], dtype=bool)
        in_span[start:end] = True
        return np.flatnonzero(is_code & in_span)

    def set_epoch(self, epoch: int) -> None:
        """Advance the masking schedule (see the note in ``__init__``)."""
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        rng = np.random.default_rng(
            derive_seed(self.seed, f"mlm-mask:{self.epoch}:{index}")
        )
        s = int(self.doc_offsets[index])
        e = int(self.doc_offsets[index + 1])
        e = min(e, s + self.block_size)
        arr = np.asarray(self.tokens[s:e], dtype=np.int64)
        n = arr.shape[0]

        input_ids = arr.copy()
        labels = np.full(n, IGNORE_INDEX, dtype=np.int64)

        maskable = self._maskable_positions(arr)
        if maskable.size > 0:
            k = max(1, round(self.mask_prob * maskable.size))
            chosen = rng.choice(maskable, size=k, replace=False)
            labels[chosen] = arr[chosen]

            draw = rng.random(k)
            replace = draw < self.mask_replace_prob
            random_tok = (draw >= self.mask_replace_prob) & (
                draw < self.mask_replace_prob + self.mask_random_prob
            )
            # remainder (draw >= replace+random) is left unchanged.
            input_ids[chosen[replace]] = self.mask_token_id
            n_rand = int(random_tok.sum())
            if n_rand > 0:
                input_ids[chosen[random_tok]] = rng.integers(
                    NUM_SPECIAL, self.base_vocab_size, size=n_rand
                )

        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
            "length": torch.tensor(n, dtype=torch.int64),
        }


def collate_mlm(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Dynamic right-padding to the batch's longest example."""
    lengths = [int(b["length"]) for b in batch]
    max_len = max(lengths)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        n = lengths[i]
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        attention_mask[i, :n] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


class MLMTokenDataModule(L.LightningDataModule):
    """Serves per-complex masked token examples for train/val/test splits."""

    def __init__(self, config: MLMTrainingConfig) -> None:
        super().__init__()
        self.config = config
        self._seed = getattr(config, "seed", DEFAULT_SEED)
        self.token_dir = Path(config.token_dir)
        self._datasets: dict[str, MLMTokenDataset] = {}

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        model = self.config.model
        for split in ("train", "val", "test"):
            bin_path = self.token_dir / f"{split}.bin"
            len_path = self.token_dir / f"{split}.len"
            if bin_path.exists() and len_path.exists():
                self._datasets[split] = MLMTokenDataset(
                    bin_path,
                    len_path,
                    block_size=self.config.block_size,
                    base_vocab_size=model.base_vocab_size,
                    mask_token_id=model.mask_token_id,
                    mask_prob=self.config.mask_prob,
                    mask_replace_prob=self.config.mask_replace_prob,
                    mask_random_prob=self.config.mask_random_prob,
                    ligand_only_masking=self.config.ligand_only_masking,
                    seed=self._seed,
                )

    def _loader(self, split: str, *, shuffle: bool) -> DataLoader:
        nw = self.config.num_workers
        return DataLoader(
            self._datasets[split],
            batch_size=self.config.micro_batch_size,
            shuffle=shuffle,
            num_workers=nw,
            persistent_workers=nw > 0,
            pin_memory=True,
            drop_last=shuffle,
            # Reproducible shuffle order, and NumPy/random streams per
            # worker (torch seeds only its own RNG in workers).
            generator=torch_generator(self._seed, "mlm-shuffle"),
            worker_init_fn=worker_init_fn,
            # torch types collate_fn as Callable[[list[_T]], Any] with _T
            # bound by nothing, so no function satisfies it.
            collate_fn=collate_mlm,  # ty: ignore[invalid-argument-type]
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)
