"""Shared on-disk format for packed LM token streams.

Every corpus builder under ``pipelines/corpora/`` emits the same layout that
:class:`~prolit.data.lm_dataset.PackedTokenDataset` reads back:

    {split}.bin   uint16   all sequences concatenated end to end
    {split}.len   uint16   per-sequence token counts (cumsum -> doc offsets)

``SplitWriter`` streams sequences straight to disk so neither tokenizer has to
hold a whole split in memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class SplitWriter:
    """Streams packed uint16 tokens + per-doc lengths to ``{split}.bin/.len``."""

    def __init__(self, out_dir: Path, split: str) -> None:
        self.bin_path = out_dir / f"{split}.bin"
        self.len_path = out_dir / f"{split}.len"
        self._bin = self.bin_path.open("wb")
        self._len = self.len_path.open("wb")
        self.num_docs = 0
        self.num_tokens = 0
        self.max_len = 0

    def write(self, sequences: list[list[int]]) -> None:
        if not sequences:
            return
        flat = np.fromiter(
            (t for seq in sequences for t in seq),
            dtype=np.uint16,
        )
        lengths = np.fromiter((len(seq) for seq in sequences), dtype=np.uint16)
        self._bin.write(flat.tobytes())
        self._len.write(lengths.tobytes())
        self.num_docs += len(sequences)
        self.num_tokens += int(flat.size)
        self.max_len = max(self.max_len, int(lengths.max()))

    def close(self) -> None:
        self._bin.close()
        self._len.close()
