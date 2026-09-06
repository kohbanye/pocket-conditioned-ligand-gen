"""Shared on-disk format for packed LM token streams.

Every corpus builder under ``pipelines/corpora/`` emits the same layout that
:class:`~prolit.data.clm_dataset.PackedTokenDataset` reads back:

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

    #: Flush both files once this many bytes of lengths have accumulated.
    #: The two streams fill at wildly different rates -- a document contributes
    #: a few hundred bytes of tokens and exactly two bytes of length -- and on
    #: Lustre ``open()`` sizes its buffer from ``st_blksize``, which is
    #: megabytes. A corpus build killed at its walltime therefore had every
    #: token on disk and an EMPTY ``.len``, which is the one file without which
    #: the tokens cannot be cut back into documents. Measured on a GEOM build:
    #: 33.5 MB of ``train.bin`` against 0 bytes of ``train.len`` after five
    #: hours. 1 MB is ~500k documents, so this costs one syscall per half
    #: million and buys a corpus that survives being killed.
    _FLUSH_EVERY_BYTES = 1 << 20

    def __init__(self, out_dir: Path, split: str) -> None:
        self.bin_path = out_dir / f"{split}.bin"
        self.len_path = out_dir / f"{split}.len"
        self._bin = self.bin_path.open("wb")
        self._len = self.len_path.open("wb")
        self.num_docs = 0
        self.num_tokens = 0
        self.max_len = 0
        self._unflushed = 0

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
        self._unflushed += lengths.nbytes
        if self._unflushed >= self._FLUSH_EVERY_BYTES:
            self.flush()

    def flush(self) -> None:
        """Put both streams on disk, in the order that keeps them consistent.

        Lengths first: a ``.len`` that runs past the tokens it indexes is a
        loud failure at read time, while a ``.bin`` longer than its index just
        looks like a shorter corpus.
        """
        self._len.flush()
        self._bin.flush()
        self._unflushed = 0

    def close(self) -> None:
        self._bin.close()
        self._len.close()
