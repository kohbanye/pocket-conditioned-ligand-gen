"""LightningDataModule for the autoregressive LM over packed VQ-VAE tokens.

Reads the token cache produced by ``pipelines/corpora/tokenize_crossdocked.py``
(``{split}.bin`` uint16 stream + ``{split}.len`` uint16 doc lengths) and packs
whole documents into fixed-length ``block_size`` blocks. Because complexes are
independent, packing carries per-document structure so the model can mask
cross-document attention and reset positions:

- ``segment_ids``: which document each token belongs to within its block
  (``-1`` for right padding).
- ``position_ids``: reset to 0 at every document boundary.
- ``labels``: ``input_ids`` with padding and each document's first token
  (``<bos>``) set to ``-100`` so no loss crosses a document boundary.

The 4D block-diagonal causal attention mask is built from ``segment_ids`` in
:class:`~prolit.model.lm_module.LigandLMModule` (on-device, per step).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from prolit.tokenizers.lm_vocab import L_OPEN_ID, PAD_ID

if TYPE_CHECKING:
    from prolit.config import LMTrainingConfig

PAD_SEGMENT = -1
IGNORE_INDEX = -100


def _pack_blocks(doc_lengths: np.ndarray, block_size: int) -> np.ndarray:
    """Greedily pack consecutive documents into blocks of at most ``block_size``.

    Returns an int64 array of document breakpoints of length ``num_blocks + 1``
    such that block ``b`` spans documents ``[breaks[b], breaks[b + 1])``. A
    single document longer than ``block_size`` gets its own block (and is later
    truncated, which should not happen for this corpus where max_len << 512).
    """
    breaks = [0]
    cur = 0
    for i, raw_length in enumerate(doc_lengths):
        length = int(raw_length)
        if cur > 0 and cur + length > block_size:
            breaks.append(i)
            cur = 0
        cur += length
    breaks.append(len(doc_lengths))
    return np.asarray(breaks, dtype=np.int64)


class PackedTokenDataset(Dataset):
    """Yields packed, padded ``block_size`` blocks with per-document structure."""

    def __init__(
        self,
        bin_path: Path,
        len_path: Path,
        block_size: int,
        *,
        mask_prompt: bool = False,
    ) -> None:
        self.block_size = block_size
        # condition-only training: mask the ``<bos><p> pocket </p>`` prompt of
        # each doc from the loss (loss only on the generated ``<l>`` block).
        self.mask_prompt = mask_prompt
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.doc_lengths = np.fromfile(len_path, dtype=np.uint16).astype(np.int64)
        self.doc_offsets = np.concatenate(
            [[0], np.cumsum(self.doc_lengths)]
        ).astype(np.int64)
        self.block_breaks = _pack_blocks(self.doc_lengths, block_size)

    def __len__(self) -> int:
        return len(self.block_breaks) - 1

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        s = int(self.block_breaks[idx])
        e = int(self.block_breaks[idx + 1])
        tok_start = int(self.doc_offsets[s])
        tok_end = int(self.doc_offsets[e])
        # Truncate to block_size (only triggers for a pathological single doc).
        tok_end = min(tok_end, tok_start + self.block_size)
        arr = np.asarray(self.tokens[tok_start:tok_end], dtype=np.int64)
        lengths = self.doc_lengths[s:e].copy()
        # Repair lengths if the final doc was truncated.
        total = int(lengths.sum())
        if total > len(arr):
            overflow = total - len(arr)
            lengths[-1] -= overflow

        n = len(arr)
        segment = np.repeat(np.arange(len(lengths)), lengths).astype(np.int64)
        position = np.concatenate(
            [np.arange(length) for length in lengths]
        ).astype(np.int64)
        labels = arr.copy()
        # First token of each document (cumulative starts) -> ignore in loss.
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        labels[starts] = IGNORE_INDEX
        if self.mask_prompt:
            # condition-only: mask each doc's prompt (everything before ``<l>``)
            # so loss falls only on the generated ligand block. A doc with no
            # ``<l>`` (e.g. protein-only) is fully masked.
            ends = np.cumsum(lengths).astype(np.int64)
            for doc_start, doc_end in zip(starts, ends, strict=False):
                rel = np.flatnonzero(arr[doc_start:doc_end] == L_OPEN_ID)
                cut = doc_start + int(rel[0]) if rel.size else doc_end
                labels[doc_start:cut] = IGNORE_INDEX

        bs = self.block_size
        input_ids = np.full(bs, PAD_ID, dtype=np.int64)
        out_labels = np.full(bs, IGNORE_INDEX, dtype=np.int64)
        out_segment = np.full(bs, PAD_SEGMENT, dtype=np.int64)
        out_position = np.zeros(bs, dtype=np.int64)
        input_ids[:n] = arr
        out_labels[:n] = labels
        out_segment[:n] = segment
        out_position[:n] = position

        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(out_labels),
            "segment_ids": torch.from_numpy(out_segment),
            "position_ids": torch.from_numpy(out_position),
        }


def collate_blocks(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {key: torch.stack([b[key] for b in batch]) for key in batch[0]}


class LMTokenDataModule(L.LightningDataModule):
    """Serves packed token blocks for train/val/test splits."""

    def __init__(self, config: LMTrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.token_dir = Path(config.token_dir)
        self.meta: dict | None = None
        self._datasets: dict[str, PackedTokenDataset] = {}

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        meta_path = self.token_dir / "meta.json"
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text())
        for split in ("train", "val", "test"):
            bin_path = self.token_dir / f"{split}.bin"
            len_path = self.token_dir / f"{split}.len"
            if bin_path.exists() and len_path.exists():
                self._datasets[split] = PackedTokenDataset(
                    bin_path,
                    len_path,
                    self.config.block_size,
                    mask_prompt=getattr(self.config, "mask_prompt", False),
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
            collate_fn=collate_blocks,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)
