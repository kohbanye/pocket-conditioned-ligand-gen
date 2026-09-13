"""Merging scoring caches must carry the labels, and must not merge groups.

``mix.py`` was written for the LM pretraining caches, which have no sidecars.
Pointed at a scoring cache it used to copy ``.bin`` and ``.len`` and drop
``.rmsd`` and ``.grp`` -- the documents survive and the supervision does not, so
training starts and learns nothing in particular.

Group ids are local to a cache. Concatenated unchanged, the first protein of
each input collides on id 0, and a grouped loss ranks ligands of two unrelated
targets against each other.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MIX = Path(__file__).resolve().parents[1] / "pipelines" / "corpora" / "mix.py"


def _cache(
    path: Path,
    docs: list[list[int]],
    labels: list[float],
    groups: list[int],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    stream = np.concatenate([np.asarray(d, np.uint16) for d in docs])
    stream.tofile(path / "train.bin")
    np.asarray([len(d) for d in docs], np.uint16).tofile(path / "train.len")
    np.asarray(labels, np.float32).tofile(path / "train.rmsd")
    np.asarray(groups, np.int32).tofile(path / "train.grp")
    torch.save({"vocab_size": 8199, "atom_codebook_size": 8192}, path / "meta.pt")


def _run(inputs: list[Path], out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(_MIX), "--inputs", *map(str, inputs),
         "--out-dir", str(out), "--splits", "train"],
        capture_output=True, text=True, check=False,
    )


def test_labels_and_groups_survive_the_merge(tmp_path: Path) -> None:
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    _cache(a, [[1, 2, 3], [4, 5]], [6.1, 7.2], [0, 1])
    _cache(b, [[7, 8]], [5.3], [0])
    assert _run([a, b], out).returncode == 0, "merge failed"

    labels = np.fromfile(out / "train.rmsd", dtype=np.float32)
    assert labels.tolist() == pytest.approx([6.1, 7.2, 5.3])
    lengths = np.fromfile(out / "train.len", dtype=np.uint16)
    assert lengths.tolist() == [3, 2, 2]


def test_group_ids_are_offset_so_the_inputs_stay_separate(tmp_path: Path) -> None:
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    _cache(a, [[1], [2]], [1.0, 2.0], [0, 1])
    _cache(b, [[3], [4]], [3.0, 4.0], [0, 0])
    assert _run([a, b], out).returncode == 0
    groups = np.fromfile(out / "train.grp", dtype=np.int32).tolist()
    assert groups == [0, 1, 2, 2], "b's group 0 must not collide with a's"
    assert len(set(groups[:2]) & set(groups[2:])) == 0


def test_an_input_without_groups_gets_singletons(tmp_path: Path) -> None:
    """``tokenize_affinity_pdbbind`` writes no ``.grp``; the merge must still align.

    One singleton group per document is the honest reading of "this document
    shares a target with nothing I know of": a grouped loss draws no pair from
    it. Skipping the input would leave .grp shorter than .len and shift every
    later group onto the wrong document.
    """
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    _cache(a, [[1], [2]], [1.0, 2.0], [0, 0])
    _cache(b, [[3], [4], [5]], [3.0, 4.0, 5.0], [0, 0, 0])
    (b / "train.grp").unlink()
    assert _run([a, b], out).returncode == 0
    groups = np.fromfile(out / "train.grp", dtype=np.int32).tolist()
    assert len(groups) == 5
    assert groups[:2] == [0, 0], "a keeps its single shared group"
    assert len(set(groups[2:])) == 3, "b's docs are singletons"
    assert not set(groups[:2]) & set(groups[2:])


def test_a_missing_label_sidecar_is_refused_not_misaligned(tmp_path: Path) -> None:
    """Silently merging would shift every later label onto the wrong document."""
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    _cache(a, [[1], [2]], [1.0, 2.0], [0, 1])
    _cache(b, [[3]], [3.0], [0])
    (b / "train.rmsd").unlink()
    result = _run([a, b], out)
    assert result.returncode != 0
    assert ".rmsd" in result.stderr, result.stderr
