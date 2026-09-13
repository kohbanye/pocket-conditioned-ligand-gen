"""A fan-out that silently does nothing looks exactly like a fan-out that works.

``esm3_structure_tokens.py --part K/N`` splits a directory manifest by file and
a single-file manifest by record. The single-file case is the one worth a test:
splitting it by file gives part 0 the whole manifest and parts 1..N-1 nothing,
so N-1 jobs exit 0 having encoded zero structures while the cache still ends up
complete -- because part 0 did all of it, N times slower than the fan-out was
sized for. Nothing in the output says so.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

_CORPORA = Path(__file__).resolve().parents[1] / "pipelines" / "corpora"
if str(_CORPORA) not in sys.path:
    sys.path.insert(0, str(_CORPORA))


@pytest.fixture(scope="module")
def load():  # noqa: ANN201
    from esm3_structure_tokens import _load_manifest  # noqa: PLC0415

    return _load_manifest


def _write_file(tmp_path: Path, n: int) -> Path:
    p = tmp_path / "m.jsonl"
    p.write_text("".join(json.dumps({"id": f"s{i}"}) + "\n" for i in range(n)))
    return p


def _write_dir(tmp_path: Path, files: int, per_file: int) -> Path:
    d = tmp_path / "manifests"
    d.mkdir()
    for f in range(files):
        with gzip.open(d / f"part{f}.jsonl.gz", "wt") as fh:
            for i in range(per_file):
                fh.write(json.dumps({"id": f"f{f}s{i}"}) + "\n")
    return d


@pytest.mark.parametrize("n_parts", [1, 2, 3, 4, 8])
def test_single_file_parts_cover_every_record_once(load, tmp_path: Path, n_parts: int) -> None:  # noqa: ANN001
    p = _write_file(tmp_path, 101)
    seen = [r["id"] for k in range(n_parts) for r in load(p, k, n_parts)]
    assert len(seen) == len(set(seen)) == 101
    assert all(len(load(p, k, n_parts)) > 0 for k in range(n_parts))


def test_single_file_without_parts_is_the_whole_file(load, tmp_path: Path) -> None:  # noqa: ANN001
    assert len(load(_write_file(tmp_path, 7))) == 7


def test_directory_still_splits_by_file(load, tmp_path: Path) -> None:  # noqa: ANN001
    """The BioLiP cache was written this way; its behaviour must not move."""
    d = _write_dir(tmp_path, files=4, per_file=5)
    seen = [r["id"] for k in range(2) for r in load(d, k, 2)]
    assert len(seen) == len(set(seen)) == 20
    assert len(load(d, 0, 2)) == 10  # two whole files, not ten interleaved records


def _pdb() -> str:
    return (
        "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1      12.313   4.741  -4.886  1.00  0.00           C\n"
    )


@pytest.fixture(scope="module")
def read_text():  # noqa: ANN201
    from esm3_structure_tokens import _read_text  # noqa: PLC0415

    return _read_text


def test_reads_from_a_zip_member(read_text, tmp_path: Path) -> None:  # noqa: ANN001
    """PLINDER ships 1060 zips; the reader started out tar-only."""
    import zipfile  # noqa: PLC0415

    z = tmp_path / "07.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("7abc__1__A/receptor.pdb", _pdb())
    cache: dict = {}
    got = read_text({"id": "7abc", "zip": str(z), "member": "7abc__1__A/receptor.pdb"}, cache)
    assert got == _pdb()
    assert len(cache) == 1, "the archive handle should be reused across records"


def test_reads_from_a_tar_member(read_text, tmp_path: Path) -> None:  # noqa: ANN001
    import io  # noqa: PLC0415
    import tarfile  # noqa: PLC0415

    t = tmp_path / "shard.tar"
    body = _pdb().encode()
    with tarfile.open(t, "w") as tf:
        info = tarfile.TarInfo("5xyz/receptor.pdb")
        info.size = len(body)
        tf.addfile(info, io.BytesIO(body))
    got = read_text({"id": "5xyz", "tar": str(t), "member": "5xyz/receptor.pdb"}, {})
    assert got == _pdb()


def test_a_missing_member_is_none_not_an_exception(read_text, tmp_path: Path) -> None:  # noqa: ANN001
    """One absent receptor must cost one structure, not the shard."""
    import zipfile  # noqa: PLC0415

    z = tmp_path / "08.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("a/receptor.pdb", _pdb())
    assert read_text({"id": "x", "zip": str(z), "member": "nope/receptor.pdb"}, {}) is None
