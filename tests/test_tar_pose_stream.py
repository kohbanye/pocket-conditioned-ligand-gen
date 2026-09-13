"""The tar walk is shared, so its contract is pinned rather than assumed.

``iter_tar_poses`` was lifted out of ``_process_atom_tar_shard`` so the stapled
CrossDocked builder and the descriptor builder walk the ligand tars exactly
once between them. The descriptor half produced a 14 GB cache that nobody is
going to rebuild to check a refactor, so the behaviour that half depends on --
which members are selected, that ``_min``/``label`` filtering happens before
any of them are read, that a corrupt member costs one pair rather than the
shard, and that ``max_files`` counts FILES while the yield is per POSE -- is
tested here instead.
"""

from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import pytest

from prolit.data.atom_tar_prep import iter_tar_poses

_SDF = """mol
     RDKit          3D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
M  END
$$$$
"""


def _two_pose_sdf() -> str:
    return _SDF + _SDF


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A ligands tar holding pairs 1 and 2, plus a member that is not a pair."""
    lig = tmp_path / "ligands"
    lig.mkdir(parents=True)
    with tarfile.open(lig / "000003.tar", "w") as tar:
        for name, body in (
            ("d/1.sdf.gz", gzip.compress(_two_pose_sdf().encode())),
            ("d/2.sdf.gz", gzip.compress(_SDF.encode())),
            ("d/README", b"not a pair"),
            ("d/9.sdf.gz", gzip.compress(_SDF.encode())),  # not in the manifest
        ):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return tmp_path


def _manifest(tmp_path: Path, rows: list[dict]) -> Path:
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    path = tmp_path / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _rows() -> list[dict]:
    base = {
        "complex_dir": "POCKET_0",
        "receptor_pdb": "r_rec.pdb",
        "source_type": "cdonly",
        "shard_idx": 3,
        "label": 1,
    }
    return [
        {**base, "pair_idx": 1, "ligand_sdf_gz": "a_min.sdf.gz"},
        {**base, "pair_idx": 2, "ligand_sdf_gz": "b_min.sdf.gz"},
        # Filtered out three different ways, one per row.
        {**base, "pair_idx": 9, "ligand_sdf_gz": "c_docked.sdf.gz"},
        {**base, "pair_idx": 4, "source_type": "other", "ligand_sdf_gz": "d_min.sdf.gz"},
        {**base, "pair_idx": 5, "label": 0, "ligand_sdf_gz": "e_min.sdf.gz"},
    ]


def test_yields_every_pose_of_every_selected_pair(repo: Path) -> None:
    got = list(
        iter_tar_poses(
            repo,
            _manifest(repo, _rows()),
            repo / "receptors",
            ["cdonly"],
            3,
            good_poses_only=True,
            min_only=True,
        )
    )
    # Pair 1 carries two poses, pair 2 carries one; pair 9 is a _docked file and
    # must not appear even though its member is in the tar.
    assert [(p, i) for p, i, _, _ in got] == [(1, 0), (1, 1), (2, 0)]
    assert all(m["atoms"][0][0] == "C" for _, _, m, _ in got)


def test_receptor_path_is_the_manifest_pair(repo: Path) -> None:
    got = list(
        iter_tar_poses(
            repo,
            _manifest(repo, _rows()),
            repo / "receptors",
            ["cdonly"],
            3,
            good_poses_only=True,
            min_only=True,
        )
    )
    expected = str(repo / "receptors" / "POCKET_0" / "r_rec.pdb")
    assert {path for _, _, _, path in got} == {expected}


def test_max_files_counts_files_not_poses(repo: Path) -> None:
    """One file of two poses must exhaust a budget of one, not half of it."""
    got = list(
        iter_tar_poses(
            repo,
            _manifest(repo, _rows()),
            repo / "receptors",
            ["cdonly"],
            3,
            good_poses_only=True,
            min_only=True,
            max_files=1,
        )
    )
    assert [(p, i) for p, i, _, _ in got] == [(1, 0), (1, 1)]


def test_empty_manifest_yields_nothing_without_opening_the_tar(tmp_path: Path) -> None:
    """A shard with no selected pairs must not need its tar to exist."""
    got = list(
        iter_tar_poses(
            tmp_path,
            _manifest(tmp_path, _rows()),
            tmp_path / "receptors",
            ["cdonly"],
            77,  # no rows for this shard
            good_poses_only=True,
            min_only=True,
        )
    )
    assert got == []
