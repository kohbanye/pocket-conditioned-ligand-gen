"""Descriptor-cache storage: shards, normalization stats, and CrossDocked I/O.

The modality-agnostic half of the descriptor pipeline. It owns the on-disk
cache format (sharded ``.pt`` files plus a manifest and a stats file), the
streaming Welford pass that produces ``normalization_stats.pt``, and the
parsing of CrossDocked's raw ``.types`` / ``.gninatypes`` files into
(receptor, ligand) pairs.

What a descriptor *contains* lives in :mod:`prolit.data.atom_descriptors`, which
builds ProLIT's 33-D per-atom rows on top of this.

Normalization is computed only over **continuous** slots; categorical columns
are forced to mean=0, std=1 so values pass through unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


import numpy as np  # noqa: TC002
import torch
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from prolit.config import (
        HubDatasetConfig,
    )

logger = logging.getLogger(__name__)

# Shard-based caching constants.
_DEFAULT_SHARD_SIZE = 50_000
_SHARD_DIR_NAME = "shards"
_SHARD_METADATA_FILE = "shard_metadata.pt"
_NORMALIZATION_STATS_FILE = "normalization_stats.pt"
# Bump when shard layout changes; _setup_from_shards refuses older caches.
_SHARD_SCHEMA_VERSION = 4


# ---------------------------------------------------------------------------
# Path helpers (legacy types-file pipeline)
# ---------------------------------------------------------------------------


def _gninatypes_to_pdb(gninatypes_path: str) -> str:
    import re  # noqa: PLC0415

    return re.sub(r"_\d+\.gninatypes$", ".pdb", gninatypes_path)


def _gninatypes_to_sdf(gninatypes_path: str) -> str:
    import re  # noqa: PLC0415

    return re.sub(r"_\d+\.gninatypes$", ".sdf.gz", gninatypes_path)


def _extract_pose_index(gninatypes_path: str) -> int:
    import re  # noqa: PLC0415

    m = re.search(r"_(\d+)\.gninatypes$", gninatypes_path)
    return int(m.group(1)) if m else 0


def _parse_types_file(types_path: Path) -> list[tuple[str, str, int]]:
    entries: list[tuple[str, str, int]] = []
    for line in types_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:  # noqa: PLR2004
            continue
        rec_pdb = _gninatypes_to_pdb(parts[3])
        lig_sdf = _gninatypes_to_sdf(parts[4])
        pose_idx = _extract_pose_index(parts[4])
        entries.append((rec_pdb, lig_sdf, pose_idx))
    return entries


def _parse_all_types_files(
    types_dir: Path,
    max_pairs: int | None = None,
) -> list[tuple[str, str, int]]:
    train_files = sorted(types_dir.glob("cdonly_*train0.types"))
    test_files = sorted(types_dir.glob("cdonly_*test0.types"))
    all_files = train_files + test_files
    if not all_files:
        msg = f"No cdonly_*train0/test0.types files found in {types_dir}"
        raise FileNotFoundError(msg)

    entries: list[tuple[str, str, int]] = []
    for types_file in all_files:
        for entry in _parse_types_file(types_file):
            entries.append(entry)
            if max_pairs is not None and len(entries) >= max_pairs:
                break
        if max_pairs is not None and len(entries) >= max_pairs:
            break
    logger.info("Loaded %d entries from %s", len(entries), types_dir)
    return entries


def _load_pairs_from_manifest(
    hub_config: HubDatasetConfig,
    max_pairs: int | None = None,
) -> tuple[list[tuple[str, str, int]], Path]:
    """Load pairs from a HuggingFace Hub manifest.

    Returns ``(pairs, base_dir)`` where *pairs* are
    ``(receptor_rel_path, ligand_rel_path, pair_idx)``.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    cache_dir = Path(hub_config.cache_dir)
    manifest_path = cache_dir / "repo" / "manifest.parquet"
    if not manifest_path.exists():
        msg = f"Manifest not found: {manifest_path}"
        raise FileNotFoundError(msg)

    table = pq.read_table(manifest_path)
    df = table.to_pandas()
    if hub_config.source_types:
        df = df[df["source_type"].isin(hub_config.source_types)]
    if getattr(hub_config, "good_poses_only", False):
        if "label" not in df.columns:
            msg = "good_poses_only requested but manifest has no 'label' column"
            raise KeyError(msg)
        before = len(df)
        df = df[df["label"] == 1]
        logger.info(
            "good_poses_only: kept %d / %d label==1 poses", len(df), before
        )
    if max_pairs is not None:
        df = df.head(max_pairs)

    pairs: list[tuple[str, str, int]] = []
    for _, row in df.iterrows():
        rec_rel = f"{row['complex_dir']}/{row['receptor_pdb']}"
        lig_rel = f"{row['pair_idx']:07d}.sdf.gz"
        pairs.append((rec_rel, lig_rel, int(row["pair_idx"])))

    logger.info("Loaded %d pairs from manifest", len(pairs))
    return pairs, cache_dir


# ---------------------------------------------------------------------------
# Grouped processing
# ---------------------------------------------------------------------------


def _group_entries_by_receptor(
    abs_entries: list[tuple[str, str, int, int]],
) -> list[tuple[str, list[tuple[str, list[tuple[int, int]]]]]]:
    from collections import defaultdict  # noqa: PLC0415

    grouped: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for rec, lig, pose_idx, pair_idx in abs_entries:
        grouped[rec][lig].append((pose_idx, pair_idx))

    return [
        (rec, [(sdf, poses) for sdf, poses in sdf_dict.items()])
        for rec, sdf_dict in grouped.items()
    ]


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------


def _write_shard(
    shard_dir: Path,
    shard_idx: int,
    data: list[dict[str, np.ndarray | list[str] | int]],
) -> None:
    torch.save(data, shard_dir / f"shard_{shard_idx:04d}.pt")


def _collect_grouped_results_sharded(
    batch_results: Iterable[tuple[list[dict[str, np.ndarray | list[str] | int]], int]],
    total_entries: int,
    shard_dir: Path,
    shard_size: int = _DEFAULT_SHARD_SIZE,
) -> tuple[int, list[int], set[str]]:
    buffer: list[dict[str, np.ndarray | list[str] | int]] = []
    shard_idx = 0
    shard_counts: list[int] = []
    total_count = 0
    unique_elements: set[str] = set()

    num_done = 0
    num_skipped = 0

    for group_results, group_total in batch_results:
        buffer.extend(group_results)
        group_skipped = group_total - len(group_results)
        num_done += group_total
        num_skipped += group_skipped

        if num_done % 10000 < group_total or num_done == total_entries:
            logger.info(
                "Progress: %d / %d done (%d ok, %d skipped)",
                num_done,
                total_entries,
                total_count + len(buffer),
                num_skipped,
            )

        while len(buffer) >= shard_size:
            shard_data = buffer[:shard_size]
            buffer = buffer[shard_size:]
            for cplx in shard_data:
                unique_elements.update(cplx["elements"])  # type: ignore[arg-type]
            _write_shard(shard_dir, shard_idx, shard_data)
            shard_counts.append(len(shard_data))
            total_count += len(shard_data)
            shard_idx += 1
            logger.info(
                "Wrote shard %04d (%d complexes)",
                shard_idx - 1,
                shard_counts[-1],
            )

    if buffer:
        for cplx in buffer:
            unique_elements.update(cplx["elements"])  # type: ignore[arg-type]
        _write_shard(shard_dir, shard_idx, buffer)
        shard_counts.append(len(buffer))
        total_count += len(buffer)
        logger.info("Wrote shard %04d (%d complexes)", shard_idx, shard_counts[-1])

    logger.info("Done: %d processed, %d skipped", total_count, num_skipped)
    return total_count, shard_counts, unique_elements


def _save_shard_metadata(
    cache_dir: Path,
    total_count: int,
    shard_counts: list[int],
    unique_elements: set[str],
) -> None:
    metadata = {
        "total_count": total_count,
        "shard_counts": shard_counts,
        "schema_version": _SHARD_SCHEMA_VERSION,
    }
    torch.save(metadata, cache_dir / _SHARD_METADATA_FILE)
    torch.save(sorted(unique_elements), cache_dir / "ligand_elements.pt")
    logger.info(
        "Cached %d complexes in %d shards (schema v%d)",
        total_count,
        len(shard_counts),
        _SHARD_SCHEMA_VERSION,
    )


def _iter_shards(
    shard_dir: Path,
    shard_counts: list[int],
) -> Iterable[tuple[int, list[dict[str, np.ndarray | list[str]]]]]:
    global_offset = 0
    for shard_idx, count in enumerate(shard_counts):
        shard_path = shard_dir / f"shard_{shard_idx:04d}.pt"
        shard_data: list[dict[str, np.ndarray | list[str]]] = torch.load(
            shard_path,
            weights_only=False,
        )
        yield global_offset, shard_data
        del shard_data
        global_offset += count


# ---------------------------------------------------------------------------
# Welford normalization (continuous slots only)
# ---------------------------------------------------------------------------


def _welford_update_batch(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    batch: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    batch_count = len(batch)
    if batch_count == 0:
        return count, mean, m2
    batch_mean = batch.mean(axis=0)
    batch_var = batch.var(axis=0, ddof=0)
    new_count = count + batch_count
    delta = batch_mean - mean
    new_mean = mean + delta * (batch_count / new_count)
    new_m2 = m2 + batch_var * batch_count + delta**2 * (count * batch_count / new_count)
    return new_count, new_mean, new_m2


def _force_passthrough_for_categorical(
    mean: np.ndarray,
    std: np.ndarray,
    cont_mask: list[bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Set categorical slots to mean=0, std=1 so they pass through unchanged."""
    out_mean = mean.copy()
    out_std = std.copy()
    for i, is_continuous in enumerate(cont_mask):
        if not is_continuous:
            out_mean[i] = 0.0
            out_std[i] = 1.0
    return out_mean, out_std


# ---------------------------------------------------------------------------
# Datasets and collators
# ---------------------------------------------------------------------------


class MoleculeDataset(Dataset[Tensor]):
    """Dataset of variable-length descriptor sequences (one tensor per item)."""

    def __init__(self, molecules: list[Tensor]) -> None:
        self.molecules = molecules

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Tensor:
        return self.molecules[idx]


def collate_molecules(batch: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Pad variable-length descriptor tensors and create attention mask."""
    max_len = max(mol.shape[0] for mol in batch)
    descriptor_dim = batch[0].shape[1]

    padded = torch.zeros(len(batch), max_len, descriptor_dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, mol in enumerate(batch):
        n = mol.shape[0]
        padded[i, :n] = mol
        mask[i, :n] = True

    return padded, mask


__all__ = [
    "MoleculeDataset",
    "collate_molecules",
]
