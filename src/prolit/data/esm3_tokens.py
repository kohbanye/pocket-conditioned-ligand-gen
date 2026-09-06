"""Reading the cached ESM3 structure tokens of a receptor.

ESM3's weights pull in a fork of ``transformers`` that must not be installed
beside the one ProLIT's language models need, so the encoder cannot run inside
the corpus builders. It runs once per receptor in the reconstruction
benchmark's interpreter (``pipelines/corpora/esm3_structure_tokens.py``) and
writes a cache keyed by structure id; every corpus builder reads it from here.

The cache stores **every residue of the structure**, not a pocket. ESM3 is at
its best given a whole chain -- encoding a pocket alone renumbers discontiguous
residues 1..L and presents residues angstroms apart as chain neighbours, which
costs 8 A of backbone RMSD -- and different corpora cut different pockets out of
the same receptor. Caching per structure and selecting per pocket keeps the
expensive half shared and the pocket definition where it belongs.

Layout on disk::

    <dir>/index.json          {"shards": N, "ids": {struct_id: [shard, slot]}}
    <dir>/shard_0000.npz      struct_ids, starts, chain, resid, token

Sharded rather than one file per structure: the group filesystem has a limited
inode budget and a receptor set runs to hundreds of thousands of entries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Esm3TokenCache", "write_shard"]


def write_shard(
    path: Path,
    entries: list[tuple[str, list[tuple[str, int]], np.ndarray]],
) -> None:
    """Write one shard: ``(struct_id, [(chain, resid), ...], token_ids)`` each."""
    starts = np.zeros(len(entries) + 1, dtype=np.int64)
    chains: list[str] = []
    resids: list[int] = []
    tokens: list[int] = []
    for i, (_sid, keys, toks) in enumerate(entries):
        starts[i + 1] = starts[i] + len(keys)
        chains.extend(c for c, _ in keys)
        resids.extend(int(r) for _, r in keys)
        tokens.extend(int(t) for t in toks)
    np.savez_compressed(
        path,
        struct_ids=np.array([e[0] for e in entries], dtype=object),
        starts=starts,
        chain=np.array(chains, dtype="<U4"),
        resid=np.asarray(resids, dtype=np.int32),
        token=np.asarray(tokens, dtype=np.int16),
    )


@dataclass
class Esm3TokenCache:
    """Per-structure ESM3 structure tokens, looked up by residue.

    Shards are memory-mapped on first touch and kept, so a builder that walks a
    corpus grouped by receptor pays each shard's decompression once.
    """

    root: Path
    _index: dict[str, tuple[int, int]] = field(default_factory=dict, init=False)
    _shards: dict[int, dict] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        # Every ``index*.json``, not just ``index.json``: several GPU jobs fill
        # one cache in parallel and each writes its own index over its own
        # shard numbers. Merging here rather than in a separate step means a
        # cache is usable the moment its parts land, and a part that failed is
        # visibly missing rather than quietly merged away.
        files = sorted(self.root.glob("index*.json"))
        if not files:
            msg = f"no index*.json under {self.root}"
            raise FileNotFoundError(msg)
        for f in files:
            for k, v in json.loads(f.read_text())["ids"].items():
                self._index[k] = tuple(v)

    def __contains__(self, struct_id: str) -> bool:
        return struct_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def _shard(self, i: int) -> dict:
        cached = self._shards.get(i)
        if cached is None:
            with np.load(self.root / f"shard_{i:04d}.npz", allow_pickle=True) as z:
                cached = {k: z[k] for k in ("starts", "chain", "resid", "token")}
            self._shards[i] = cached
        return cached

    def residue_tokens(self, struct_id: str) -> dict[tuple[str, int], int] | None:
        """``(chain, author resid) -> ESM3 structure code`` for one structure."""
        loc = self._index.get(struct_id)
        if loc is None:
            return None
        shard_idx, slot = loc
        z = self._shard(shard_idx)
        lo, hi = int(z["starts"][slot]), int(z["starts"][slot + 1])
        return {
            (str(c), int(r)): int(t)
            for c, r, t in zip(
                z["chain"][lo:hi], z["resid"][lo:hi], z["token"][lo:hi], strict=True
            )
        }

    def pocket_codes(
        self,
        struct_id: str,
        residue_ids: list[tuple[str, int]],
    ) -> list[int] | None:
        """Codes for ``residue_ids``, in the order given, or ``None`` if any is
        missing.

        Missing is not a silent skip. A pocket residue with no cached token means
        the receptor the cache was built from is not the receptor this pocket was
        cut from, and a stream built from the residues that happened to match
        would be a different pocket wearing the right name.
        """
        table = self.residue_tokens(struct_id)
        if table is None:
            return None
        out = []
        for key in residue_ids:
            code = table.get(key)
            if code is None:
                return None
            out.append(code)
        return out
