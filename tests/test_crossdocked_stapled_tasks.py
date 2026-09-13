"""The fan-out must cover every pose exactly once.

The stapled CrossDocked corpus is built by a few hundred processes across
several nodes and then concatenated. Two failure modes are invisible in the
output and fatal to it: a (shard, slice) pair claimed by two partitions puts
the same poses in the corpus twice, and one claimed by none drops them. Neither
shows up as an error -- the corpus just has the wrong contents -- so the
partitioning is checked here rather than trusted.

The slicing itself is ``pair_idx % n_slices``, which is what makes a slice
cheap to skip while walking a tar; the property that matters is that the
slices of a shard are disjoint and together are the whole shard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CORPORA = Path(__file__).resolve().parents[1] / "pipelines" / "corpora"
if str(_CORPORA) not in sys.path:
    sys.path.insert(0, str(_CORPORA))


@pytest.fixture(scope="module")
def tasks_fn():  # noqa: ANN201
    from tokenize_crossdocked_stapled import _tasks  # noqa: PLC0415

    return _tasks


@pytest.mark.parametrize("n_parts", [1, 2, 3, 8, 10])
def test_partitions_cover_every_task_exactly_once(tasks_fn, n_parts: int) -> None:  # noqa: ANN001
    shards, slices = list(range(35)), 16
    seen: list[tuple[int, int]] = []
    for k in range(n_parts):
        seen.extend(tasks_fn(shards, slices, n_parts, k))
    assert len(seen) == len(set(seen)), "a task was claimed twice"
    assert set(seen) == {(s, j) for s in shards for j in range(slices)}


def test_partitions_are_within_one_task_of_each_other(tasks_fn) -> None:  # noqa: ANN001
    """An interleaved split, so no partition gets a whole tar more than another."""
    sizes = [len(tasks_fn(list(range(35)), 16, 8, k)) for k in range(8)]
    assert max(sizes) - min(sizes) <= 1


def test_every_partition_touches_many_shards(tasks_fn) -> None:  # noqa: ANN001
    """Blocked assignment would put one tar's poses in one output only.

    That is not wrong on its own, but it makes a partition's runtime depend on
    which tars it drew, and the tars are not the same size.
    """
    for k in range(8):
        shards_hit = {s for s, _ in tasks_fn(list(range(35)), 16, 8, k)}
        assert len(shards_hit) >= 30
