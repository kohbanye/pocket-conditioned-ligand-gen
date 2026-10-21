"""Partitioning a large group keeps the epoch whole; subsampling does not.

With real protein groups the union corpus has groups of several hundred
ligands. Under the legacy cap the sampler drew ``max_per_group`` of them per
epoch and the rest of that group went unseen -- an epoch covered 46% of the
documents. Partitioning splits the group across batches instead, so every
document appears exactly once and a batch still holds several ligands of one
protein (which is what a grouped objective needs).
"""

from __future__ import annotations

from collections import Counter

from prolit.data.rescore_dataset import GroupBatchSampler

# 10 + 2 + 1 + 7 = 20 documents in four groups.
GROUPS = [list(range(10)), [10, 11], [12], list(range(13, 20))]
CAP = 4


def _batches(**kw: object) -> list[list[int]]:
    sampler = GroupBatchSampler(GROUPS, 2, shuffle=False, seed=0, **kw)  # ty: ignore[invalid-argument-type]
    return list(sampler)


def test_partition_covers_every_document_exactly_once() -> None:
    seen = Counter(d for b in _batches(max_per_group=CAP, partition=True) for d in b)
    assert sorted(seen) == list(range(20))
    assert set(seen.values()) == {1}


def test_subsampling_drops_documents_of_a_large_group() -> None:
    seen = Counter(d for b in _batches(max_per_group=CAP, partition=False) for d in b)
    # The 10-document group contributes only CAP of its members in an epoch.
    assert sum(1 for d in seen if d < 10) == CAP
    assert len(seen) < 20


def test_partition_respects_the_batch_bound() -> None:
    for batch in _batches(max_per_group=CAP, partition=True):
        assert len(batch) <= CAP * 2  # complexes_per_batch=2 units of at most CAP


def test_partition_keeps_group_members_together_in_a_unit() -> None:
    batches = _batches(max_per_group=CAP, partition=True)
    # The 7-document group splits into 4 + 3, and each chunk stays contiguous.
    chunks = [[d for d in b if 13 <= d < 20] for b in batches]
    non_empty = [sorted(c) for c in chunks if c]
    assert [len(c) for c in non_empty] == [4, 3]
    assert non_empty[0] == [13, 14, 15, 16]


def test_no_cap_means_whole_groups_as_before() -> None:
    seen = Counter(d for b in _batches(max_per_group=0, partition=True) for d in b)
    assert sorted(seen) == list(range(20))
    assert set(seen.values()) == {1}
