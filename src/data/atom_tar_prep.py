"""Low-inode all-atom descriptor prep: stream ligand tars, all-heavy-atom pockets.

All-atom counterpart of :mod:`src.data.tar_prep`. For each ligand pose it builds
the 33-D unified atom descriptor for BOTH the ligand atoms and every heavy atom
of the pocket residues (Full ligand-parity chemistry via one RDKit receptor
parse per receptor), and writes the ``data/descriptor_cache_allatom`` shard
cache. Optionally keeps only ``label == 1`` (native-like) poses.

Like ``tar_prep``, ligands are read straight from ``ligands/{shard:06d}.tar`` so
the only files on disk are the ~35 tars, the receptor PDBs, and the shards.
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

from src.data.atom_descriptors import (
    _atom_process_pose,
    _save_atom_shard_metadata,
)
from src.data.descriptors import _DEFAULT_SHARD_SIZE
from src.data.tar_prep import _pair_idx_from_member
from src.tokenizers.atom import (
    LigandAtomDescriptor,
    ProteinAtomDescriptor,
    precompute_receptor_atom_features,
)
from src.tokenizers.ligand import parse_sdf_text
from src.tokenizers.protein import precompute_pocket_atom_candidates

if TYPE_CHECKING:
    from src.config import PocketExtractionConfig

logger = logging.getLogger(__name__)


def _load_shard_pair_map(  # noqa: PLR0913
    manifest_path: Path,
    shard_idx: int,
    source_types: list[str],
    receptors_dir: Path,
    *,
    good_poses_only: bool,
    min_only: bool,
) -> dict[int, str]:
    """Map ``pair_idx -> receptor abs path`` for one tar shard.

    Filtered by source type and (optionally) ``label == 1`` and ``_min`` files.

    ``label`` is per docking-run FILE, not per pose: a ``label==1`` ``_docked``
    file still holds ~20 poses (1 near-native + 19 worse). Keeping all of them
    re-admits decoys. ``min_only`` (default) restricts to the ``*_min.sdf.gz``
    minimized near-native poses (1 molecule each) -- the clean good-pose set.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    columns = ["pair_idx", "complex_dir", "receptor_pdb", "source_type", "shard_idx"]
    available = set(pq.read_schema(manifest_path).names)
    if good_poses_only and "label" in available:
        columns.append("label")
    if min_only and "ligand_sdf_gz" in available:
        columns.append("ligand_sdf_gz")
    # Predicate pushdown on shard_idx: each worker loads only its tar's rows
    # (~70k) instead of the full multi-million-row manifest, keeping per-worker
    # memory at a few MB so many workers fit on one node.
    df = pq.read_table(
        manifest_path,
        columns=columns,
        filters=[("shard_idx", "=", shard_idx)],
    ).to_pandas()
    df = df[df["source_type"].isin(source_types)]
    if good_poses_only:
        if "label" not in df.columns:
            msg = "good_poses_only requested but manifest has no 'label' column"
            raise KeyError(msg)
        df = df[df["label"] == 1]
    if min_only:
        if "ligand_sdf_gz" not in df.columns:
            msg = "min_only requested but manifest has no 'ligand_sdf_gz' column"
            raise KeyError(msg)
        df = df[df["ligand_sdf_gz"].str.endswith("_min.sdf.gz")]

    pair_map: dict[int, str] = {}
    interned: dict[tuple[str, str], str] = {}
    for row in df.itertuples(index=False):
        key = (str(row.complex_dir), str(row.receptor_pdb))
        path = interned.get(key)
        if path is None:
            path = str(receptors_dir / key[0] / key[1])
            interned[key] = path
        pair_map[int(row.pair_idx)] = path
    return pair_map


def _process_atom_tar_shard(  # noqa: C901, PLR0915
    args: tuple[int, Path, Path, Path, list[str], dict, Path, int | None, bool, bool],
) -> tuple[list[str], list[int], set[str], int]:
    """Stream one ligand tar, compute all-atom descriptors, write shard parts."""
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
        good_poses_only,
        min_only,
    ) = args
    pocket_cfg = PocketExtractionConfig(**pocket_cfg_dict)
    protein_desc = ProteinAtomDescriptor()
    ligand_desc = LigandAtomDescriptor()

    pair_map = _load_shard_pair_map(
        manifest_path,
        shard_idx,
        source_types,
        receptors_dir,
        good_poses_only=good_poses_only,
        min_only=min_only,
    )
    if not pair_map:
        return [], [], set(), 0

    @lru_cache(maxsize=256)
    def _get_receptor(rec_path: str) -> tuple[object, dict] | None:
        try:
            precomputed = precompute_pocket_atom_candidates(Path(rec_path))
            feats = precompute_receptor_atom_features(Path(rec_path))
        except Exception:
            logger.exception("Error parsing receptor %s", rec_path)
            return None
        return precomputed, feats

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
                receptor = _get_receptor(pair_map[pair_idx]) if molecules else None
            except Exception:
                logger.exception("Read error: pair %d shard %d", pair_idx, shard_idx)
                continue
            if not molecules or receptor is None:
                continue
            precomputed, feats = receptor
            for pose_idx, mol in enumerate(molecules):
                attempted += 1
                try:
                    result = _atom_process_pose(
                        mol, precomputed, feats, pocket_cfg, protein_desc, ligand_desc
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
        "Tar shard %d: %d ok / %d attempted, %d parts",
        shard_idx,
        sum(shard_counts),
        attempted,
        len(shard_files),
    )
    return shard_files, shard_counts, elements, attempted


def prepare_atom_descriptors_from_tars(  # noqa: PLR0913
    repo_dir: Path,
    receptors_dir: Path,
    cache_dir: Path,
    source_types: list[str],
    pocket_config: PocketExtractionConfig,
    num_workers: int,
    *,
    good_poses_only: bool = True,
    min_only: bool = True,
    max_files_per_tar: int | None = None,
) -> tuple[int, list[int]]:
    """Build the all-atom descriptor shard cache by streaming ligand tars."""
    import multiprocessing  # noqa: PLC0415

    import pyarrow.parquet as pq  # noqa: PLC0415

    manifest_path = Path(repo_dir) / "manifest.parquet"
    shard_dir = Path(cache_dir) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    df = pq.read_table(manifest_path, columns=["shard_idx", "source_type"]).to_pandas()
    df = df[df["source_type"].isin(source_types)]
    shard_indices = sorted(int(s) for s in df["shard_idx"].unique())
    logger.info(
        "Streaming %d ligand tars (source_types=%s, good_poses_only=%s, min_only=%s)",
        len(shard_indices),
        source_types,
        good_poses_only,
        min_only,
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
            good_poses_only,
            min_only,
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
            _process_atom_tar_shard, tasks
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

    final_counts: list[int] = []
    for idx, (src, count) in enumerate(zip(tmp_files, tmp_counts, strict=True)):
        Path(src).rename(shard_dir / f"shard_{idx:04d}.pt")
        final_counts.append(count)

    total_count = sum(final_counts)
    if total_count == 0:
        msg = "No atom descriptors computed from tars -- check paths / source_types"
        raise RuntimeError(msg)

    _save_atom_shard_metadata(Path(cache_dir), total_count, final_counts, elements)
    logger.info(
        "Done: %d complexes in %d shards (%d attempted)",
        total_count,
        len(final_counts),
        attempted_total,
    )
    return total_count, final_counts
