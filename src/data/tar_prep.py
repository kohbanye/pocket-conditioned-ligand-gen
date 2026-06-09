"""Low-inode descriptor prep: read ligands straight from packed tar shards.

The HuggingFace dataset stores all ligand poses in a handful of tar shards
(``ligands/{shard:06d}.tar``), each holding members named
``{pair_idx:07d}.sdf.gz``. Extracting them produces ~25M files and exhausts the
filesystem inode quota. This module instead streams each tar, decompresses
members in memory, and computes descriptors directly -- so the only files on
disk are the ~35 tars, ~2900 extracted receptor PDBs, and the descriptor shards.

Parallelism is one worker per tar shard. To bound memory, each worker writes its
own shard files incrementally; the parent then renames them to the sequential
``shard_{idx:04d}.pt`` layout the rest of the pipeline expects.
"""

from __future__ import annotations

import gzip
import logging
import tarfile
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.data.descriptors import (
    _DEFAULT_SHARD_SIZE,
    _process_pose,
    _save_shard_metadata,
)
from src.tokenizers.ligand import LigandDescriptor, parse_sdf_text
from src.tokenizers.protein import (
    BackboneSphericalDescriptor,
    precompute_pocket_candidates,
)

if TYPE_CHECKING:
    from src.config import PocketExtractionConfig

logger = logging.getLogger(__name__)


def _pair_idx_from_member(name: str) -> int | None:
    base = name.rsplit("/", 1)[-1]
    suffix = ".sdf.gz"
    if not base.endswith(suffix):
        return None
    try:
        return int(base[: -len(suffix)])
    except ValueError:
        return None


def _load_shard_pair_map(
    manifest_path: Path,
    shard_idx: int,
    source_types: list[str],
    receptors_dir: Path,
) -> dict[int, str]:
    """Map ``pair_idx -> receptor abs path`` for one tar shard, filtered by type."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    df = pq.read_table(
        manifest_path,
        columns=["pair_idx", "complex_dir", "receptor_pdb", "source_type", "shard_idx"],
    ).to_pandas()
    df = df[(df["shard_idx"] == shard_idx) & (df["source_type"].isin(source_types))]
    pair_map: dict[int, str] = {}
    # Intern receptor paths (only ~2900 unique) to keep the map small.
    interned: dict[tuple[str, str], str] = {}
    for row in df.itertuples(index=False):
        key = (str(row.complex_dir), str(row.receptor_pdb))
        path = interned.get(key)
        if path is None:
            path = str(receptors_dir / key[0] / key[1])
            interned[key] = path
        pair_map[int(row.pair_idx)] = path
    return pair_map


def _process_tar_shard(  # noqa: C901, PLR0915
    args: tuple[int, Path, Path, Path, list[str], dict, Path, int | None],
) -> tuple[list[str], list[int], set[str], int]:
    """Stream one ligand tar, compute descriptors, write shard files.

    Returns ``(shard_file_paths, shard_counts, unique_elements, attempted)``.
    """
    from src.config import PocketExtractionConfig  # noqa: PLC0415

    (
        shard_idx,
        repo_dir,
        manifest_path,
        receptors_dir,
        source_types,
        pocket_cfg_dict,
        out_dir,
        max_files,
    ) = args
    pocket_cfg = PocketExtractionConfig(**pocket_cfg_dict)
    protein_desc = BackboneSphericalDescriptor()
    ligand_desc = LigandDescriptor()

    pair_map = _load_shard_pair_map(
        manifest_path, shard_idx, source_types, receptors_dir
    )
    if not pair_map:
        return [], [], set(), 0

    @lru_cache(maxsize=256)
    def _get_precomputed(rec_path: str) -> object | None:
        try:
            return precompute_pocket_candidates(Path(rec_path))
        except Exception:
            logger.exception("Error parsing PDB %s", rec_path)
            return None

    tar_path = Path(repo_dir) / "ligands" / f"{shard_idx:06d}.tar"
    shard_files: list[str] = []
    shard_counts: list[int] = []
    elements: set[str] = set()
    buffer: list[dict] = []
    attempted = 0
    part = 0
    files_seen = 0

    def _flush() -> None:
        nonlocal buffer, part
        if not buffer:
            return
        path = Path(out_dir) / f"tmp_{shard_idx:03d}_{part:04d}.pt"
        torch.save(buffer, path)
        shard_files.append(str(path))
        shard_counts.append(len(buffer))
        for cplx in buffer:
            elements.update(cplx["elements"])
        part += 1
        buffer = []

    with tarfile.open(tar_path, "r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            pair_idx = _pair_idx_from_member(member.name)
            if pair_idx is None or pair_idx not in pair_map:
                continue
            if max_files is not None and files_seen >= max_files:
                break
            files_seen += 1
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            try:
                text = gzip.decompress(fileobj.read()).decode("utf-8", "replace")
                molecules = parse_sdf_text(text)
                precomputed = (
                    _get_precomputed(pair_map[pair_idx]) if molecules else None
                )
            except Exception:
                logger.exception("Read error: pair %d shard %d", pair_idx, shard_idx)
                continue
            if not molecules or precomputed is None:
                continue
            # Each SDF holds multiple docked poses; emit one descriptor per pose.
            for pose_idx, mol in enumerate(molecules):
                attempted += 1
                try:
                    result = _process_pose(
                        mol, precomputed, pocket_cfg, protein_desc, ligand_desc
                    )
                except Exception:
                    logger.exception(
                        "Error pair %d pose %d (shard %d)",
                        pair_idx,
                        pose_idx,
                        shard_idx,
                    )
                    continue
                if result is not None:
                    result["pair_idx"] = pair_idx
                    result["pose_idx"] = pose_idx
                    buffer.append(result)
                    if len(buffer) >= _DEFAULT_SHARD_SIZE:
                        _flush()
    _flush()
    logger.info(
        "Tar shard %d: %d ok / %d attempted, %d shard parts",
        shard_idx,
        sum(shard_counts),
        attempted,
        len(shard_files),
    )
    return shard_files, shard_counts, elements, attempted


def prepare_descriptors_from_tars(  # noqa: PLR0913
    repo_dir: Path,
    receptors_dir: Path,
    cache_dir: Path,
    source_types: list[str],
    pocket_config: PocketExtractionConfig,
    num_workers: int,
    max_files_per_tar: int | None = None,
) -> tuple[int, list[int]]:
    """Build the descriptor shard cache by streaming ligand tars (no extraction).

    ``max_files_per_tar`` caps the SDF files processed per tar (debug/smoke runs).
    """
    import multiprocessing  # noqa: PLC0415

    import pyarrow.parquet as pq  # noqa: PLC0415

    manifest_path = Path(repo_dir) / "manifest.parquet"
    shard_dir = Path(cache_dir) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    df = pq.read_table(manifest_path, columns=["shard_idx", "source_type"]).to_pandas()
    df = df[df["source_type"].isin(source_types)]
    shard_indices = sorted(int(s) for s in df["shard_idx"].unique())
    logger.info(
        "Streaming %d ligand tars for source_types=%s",
        len(shard_indices),
        source_types,
    )

    pocket_cfg_dict = asdict(pocket_config)
    tasks = [
        (
            si,
            Path(repo_dir),
            manifest_path,
            Path(receptors_dir),
            list(source_types),
            pocket_cfg_dict,
            shard_dir,
            max_files_per_tar,
        )
        for si in shard_indices
    ]

    tmp_files: list[str] = []
    tmp_counts: list[int] = []
    elements: set[str] = set()
    attempted_total = 0

    workers = max(1, min(num_workers, len(tasks)))
    with multiprocessing.Pool(workers) as pool:
        for files, counts, elems, attempted in pool.imap_unordered(
            _process_tar_shard, tasks
        ):
            tmp_files.extend(files)
            tmp_counts.extend(counts)
            elements |= elems
            attempted_total += attempted
            logger.info(
                "Progress: %d ok so far (%d attempted)",
                sum(tmp_counts),
                attempted_total,
            )

    # Renumber the per-worker shard parts into the sequential layout.
    final_counts: list[int] = []
    for idx, (src, count) in enumerate(zip(tmp_files, tmp_counts, strict=True)):
        Path(src).rename(shard_dir / f"shard_{idx:04d}.pt")
        final_counts.append(count)

    total_count = sum(final_counts)
    if total_count == 0:
        msg = "No descriptors computed from tars -- check paths / source_types"
        raise RuntimeError(msg)

    _save_shard_metadata(Path(cache_dir), total_count, final_counts, elements)
    logger.info(
        "Done: %d complexes in %d shards (%d attempted)",
        total_count,
        len(final_counts),
        attempted_total,
    )
    return total_count, final_counts
