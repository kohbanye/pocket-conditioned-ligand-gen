"""The ESM3 token cache is a join key, so a silent miss is a wrong pocket.

The cache is written by one interpreter (the reconstruction benchmark's, which
has ESM3's forked transformers) and read by another (the corpus builders'). The
two never run in the same process, so nothing but this test checks that what one
writes is what the other reads -- and a pocket assembled from the residues that
happened to match would be a different pocket under the right name.
"""

from __future__ import annotations

import json

import numpy as np

from prolit.data.esm3_tokens import Esm3TokenCache, write_shard


def _cache(tmp_path, entries_per_shard):  # noqa: ANN001, ANN202
    index = {}
    for shard_idx, entries in enumerate(entries_per_shard):
        write_shard(tmp_path / f"shard_{shard_idx:04d}.npz", entries)
        for slot, (sid, _keys, _tok) in enumerate(entries):
            index[sid] = [shard_idx, slot]
    (tmp_path / "index.json").write_text(
        json.dumps({"shards": len(entries_per_shard), "ids": index})
    )
    return Esm3TokenCache(tmp_path)


def test_round_trip_across_shards(tmp_path) -> None:  # noqa: ANN001
    a = ("1abc", [("A", 10), ("A", 11), ("B", 3)], np.array([5, 4095, 0]))
    b = ("2xyz", [("A", 1)], np.array([777]))
    c = ("3pqr", [("C", -7), ("C", 0)], np.array([1, 2]))
    cache = _cache(tmp_path, [[a, b], [c]])

    assert len(cache) == 3
    assert "1abc" in cache
    assert cache.residue_tokens("1abc") == {("A", 10): 5, ("A", 11): 4095, ("B", 3): 0}
    assert cache.residue_tokens("2xyz") == {("A", 1): 777}
    # Negative author numbering happens in real PDB files and must survive int16
    # keys and the shard boundary alike.
    assert cache.residue_tokens("3pqr") == {("C", -7): 1, ("C", 0): 2}


def test_pocket_codes_follow_the_order_asked_for(tmp_path) -> None:  # noqa: ANN001
    entry = ("1abc", [("A", 1), ("A", 2), ("A", 3)], np.array([11, 22, 33]))
    cache = _cache(tmp_path, [[entry]])
    assert cache.pocket_codes("1abc", [("A", 3), ("A", 1)]) == [33, 11]


def test_a_missing_residue_fails_rather_than_shortens(tmp_path) -> None:  # noqa: ANN001
    """A pocket the cache cannot fully cover is not a shorter pocket."""
    entry = ("1abc", [("A", 1)], np.array([11]))
    cache = _cache(tmp_path, [[entry]])
    assert cache.pocket_codes("1abc", [("A", 1), ("A", 2)]) is None
    assert cache.pocket_codes("nosuch", [("A", 1)]) is None
    assert cache.residue_tokens("nosuch") is None


def test_codes_stay_inside_esm3s_codebook(tmp_path) -> None:  # noqa: ANN001
    """int16 storage must not wrap a legitimate 4095 into something else."""
    entry = ("1abc", [("A", i) for i in range(4)], np.array([0, 1, 4094, 4095]))
    cache = _cache(tmp_path, [[entry]])
    codes = cache.pocket_codes("1abc", [("A", i) for i in range(4)])
    assert codes == [0, 1, 4094, 4095]
    assert all(0 <= c < 4096 for c in codes)


def test_shard_cache_is_bounded(tmp_path) -> None:  # noqa: ANN001
    """An unbounded cache asked for 141 GB on a 23 GB node and hung the run.

    The npz shards compress about 18-fold, so "keep every shard you touch" is
    3.53 GB per process for the PLINDER cache. Forty workers each doing that
    were OOM-killed, and ``multiprocessing.Pool`` blocks forever on a dead
    worker's result -- the symptom was a frozen CPU counter, which reads as a
    deadlock rather than as memory exhaustion.
    """
    shards = [
        [(f"s{i}", [("A", 1), ("A", 2)], np.array([10 + i, 20 + i]))] for i in range(6)
    ]
    cache = _cache(tmp_path, shards)
    cache.max_shards = 2

    for i in range(6):
        assert cache.residue_tokens(f"s{i}") is not None
        # The bound is on an internal structure: the public API cannot tell a
        # bounded cache from an unbounded one.
        assert len(cache._shards) <= 2, "cache grew past its bound"  # noqa: SLF001

    # Bounded, but still correct: an evicted shard is re-read, not lost.
    assert cache.residue_tokens("s0") == {("A", 1): 10, ("A", 2): 20}


def test_default_bound_keeps_a_small_cache_whole(tmp_path) -> None:  # noqa: ANN001
    """The CrossDocked cache is 16 shards, so the default must not thrash it."""
    shards = [[(f"s{i}", [("A", 1)], np.array([i]))] for i in range(16)]
    cache = _cache(tmp_path, shards)
    for i in range(16):
        cache.residue_tokens(f"s{i}")
    assert len(cache._shards) == 16  # noqa: SLF001
