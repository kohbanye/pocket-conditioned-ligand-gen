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


#: Written into a per-token label stream wherever no label applies: the
#: specials, the pocket block, and any atom the builder had no answer for. The
#: dataset turns it into the loss's ignore index.
CONTACT_IGNORE = 255


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

    #: Per-TOKEN auxiliary label streams, as ``{split}.{name}`` of uint8 with
    #: exactly as many entries as ``{split}.bin``. Distinct from the
    #: per-DOCUMENT sidecars a scoring cache carries (``.rmsd`` and friends),
    #: which have one entry per document instead.
    TOKEN_LABELS: tuple[str, ...] = ("res", "itype")

    def __init__(
        self, out_dir: Path, split: str, *, token_labels: bool = False
    ) -> None:
        self.bin_path = out_dir / f"{split}.bin"
        self.len_path = out_dir / f"{split}.len"
        self._bin = self.bin_path.open("wb")
        self._len = self.len_path.open("wb")
        #: Opened only when asked for: a corpus without them must not be left
        #: with empty label files, which would read as "every token ignored"
        #: rather than "this corpus has no labels".
        self._labels = (
            {
                name: (out_dir / f"{split}.{name}").open("wb")
                for name in self.TOKEN_LABELS
            }
            if token_labels
            else {}
        )
        self.num_docs = 0
        self.num_tokens = 0
        self.max_len = 0
        self._unflushed = 0

    def _write_labels(
        self,
        token_labels: dict[str, list[np.ndarray]] | None,
        n_sequences: int,
        n_tokens: int,
    ) -> None:
        """Append one uint8 entry per token, per configured label stream.

        The two length checks are the point. A label stream that drifts out of
        step with ``.bin`` cannot be recovered afterwards and trains the
        auxiliary head on other atoms' answers, with nothing to show that it
        happened -- so a mismatch is an exception here rather than a warning
        later.
        """
        if token_labels is None:
            msg = "writer was opened with token_labels=True but none were given"
            raise ValueError(msg)
        for name, stream in self._labels.items():
            per_doc = token_labels[name]
            if len(per_doc) != n_sequences:
                msg = (
                    f"{name}: {len(per_doc)} label arrays for "
                    f"{n_sequences} sequences"
                )
                raise ValueError(msg)
            joined = np.concatenate([np.asarray(a, dtype=np.uint8) for a in per_doc])
            if joined.size != n_tokens:
                msg = (
                    f"{name}: {joined.size} labels for {n_tokens} tokens; "
                    "the streams must stay positionally aligned"
                )
                raise ValueError(msg)
            stream.write(joined.tobytes())

    def write(
        self,
        sequences: list[list[int]],
        token_labels: dict[str, list[np.ndarray]] | None = None,
    ) -> None:
        """Append documents, and their per-token labels when configured.

        ``token_labels`` maps each name in :data:`TOKEN_LABELS` to one array per
        sequence, each the same length as its sequence. The lengths are checked
        rather than trusted: a label stream that drifts out of step with the
        tokens is unrecoverable afterwards and silently trains the auxiliary
        head on the wrong positions.
        """
        if not sequences:
            return
        flat = np.fromiter(
            (t for seq in sequences for t in seq),
            dtype=np.uint16,
        )
        lengths = np.fromiter((len(seq) for seq in sequences), dtype=np.uint16)
        self._bin.write(flat.tobytes())
        self._len.write(lengths.tobytes())
        if self._labels:
            self._write_labels(token_labels, len(sequences), flat.size)
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
        for stream in self._labels.values():
            stream.flush()
        self._unflushed = 0

    def close(self) -> None:
        self._bin.close()
        self._len.close()
        for stream in self._labels.values():
            stream.close()
