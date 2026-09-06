"""A corpus build killed at its walltime must leave a readable corpus.

``SplitWriter`` feeds two files at very different rates: a document adds a few
hundred bytes of tokens and exactly two bytes of length. On Lustre ``open()``
sizes its buffer from ``st_blksize``, which is megabytes, so the ``.len`` stream
can sit entirely in memory while the ``.bin`` has flushed dozens of times. A
GEOM build measured 33.5 MB of ``train.bin`` against **0 bytes** of
``train.len`` after five hours -- every token on disk, and the one file needed
to cut them back into documents still unwritten.

Corpus builds are the long jobs in this repository and walltime is how they
usually end, so the index has to reach disk while the job is alive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from prolit.data.token_io import SplitWriter

if TYPE_CHECKING:
    from pathlib import Path


def _docs(n: int, length: int = 40) -> list[list[int]]:
    return [[7 + (i % 100)] * length for i in range(n)]


def test_index_reaches_disk_before_close(tmp_path: Path) -> None:
    """Written but not closed, both files must already be readable."""
    w = SplitWriter(tmp_path, "train")
    # Enough documents to cross the flush threshold several times over.
    for doc in _docs(4 * SplitWriter._FLUSH_EVERY_BYTES // 2):  # noqa: SLF001
        w.write([doc])

    lengths = (tmp_path / "train.len").read_bytes()
    tokens = (tmp_path / "train.bin").read_bytes()
    assert lengths, "the length index is still buffered; a kill would lose it"
    assert tokens
    # And what did reach disk must be consistent: every indexed document's
    # tokens are present, so a truncated corpus reads as shorter, never as
    # corrupt.
    n_docs = len(lengths) // 2
    indexed = int(np.frombuffer(lengths, dtype=np.uint16)[:n_docs].sum())
    assert indexed * 2 <= len(tokens)

    w.close()


def test_close_still_writes_everything(tmp_path: Path) -> None:
    """The periodic flush must not lose the tail."""
    w = SplitWriter(tmp_path, "val")
    docs = _docs(1000, length=13)
    w.write(docs)
    w.close()

    lengths = np.frombuffer((tmp_path / "val.len").read_bytes(), dtype=np.uint16)
    tokens = np.frombuffer((tmp_path / "val.bin").read_bytes(), dtype=np.uint16)
    assert len(lengths) == 1000
    assert lengths.sum() == len(tokens) == 13000
    assert w.num_docs == 1000
    assert w.num_tokens == 13000
    assert w.max_len == 13
