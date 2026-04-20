"""DataModule for computing and caching VQ-VAE training descriptors.

Processes CrossDocked2020 protein-ligand complexes into per-residue and
per-atom descriptors for joint VQ-VAE training.  Data is split at the
**complex** level into train / val / test to prevent data leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset

from src.tokenizers.ligand import LigandDescriptor, parse_sdf
from src.tokenizers.protein import (
    BackboneZMatrixDescriptor,
    _compute_canonical_frame,
    extract_pocket_from_candidates,
    precompute_pocket_candidates,
)

if TYPE_CHECKING:
    from src.config import (
        CrossDockedConfig,
        HubDatasetConfig,
        PocketExtractionConfig,
        VQVAETrainingConfig,
    )

logger = logging.getLogger(__name__)

# Shard-based caching constants.
_DEFAULT_SHARD_SIZE = 50_000
_SHARD_DIR_NAME = "shards"
_SHARD_METADATA_FILE = "shard_metadata.pt"
_NORMALIZATION_STATS_FILE = "normalization_stats.pt"

# Per-worker state for multiprocessing (set by _worker_init).
_worker_protein_desc: BackboneZMatrixDescriptor | None = None
_worker_ligand_desc: LigandDescriptor | None = None
_worker_pocket_config: PocketExtractionConfig | None = None


def _gninatypes_to_pdb(gninatypes_path: str) -> str:
    """Convert a receptor .gninatypes path to the corresponding .pdb path.

    Example: ``subdir/5f74_A_rec_0.gninatypes`` -> ``subdir/5f74_A_rec.pdb``
    """
    import re  # noqa: PLC0415

    return re.sub(r"_\d+\.gninatypes$", ".pdb", gninatypes_path)


def _gninatypes_to_sdf(gninatypes_path: str) -> str:
    """Convert a ligand .gninatypes path to the corresponding .sdf.gz path.

    Example: ``subdir/5f74_A_rec_5f74_amp_lig_tt_docked_0.gninatypes``
           -> ``subdir/5f74_A_rec_5f74_amp_lig_tt_docked.sdf.gz``
    """
    import re  # noqa: PLC0415

    return re.sub(r"_\d+\.gninatypes$", ".sdf.gz", gninatypes_path)


def _extract_pose_index(gninatypes_path: str) -> int:
    """Extract the docked-pose index from a .gninatypes filename.

    Example: ``subdir/5f74_A_rec_5f74_amp_lig_tt_docked_3.gninatypes`` -> 3
    """
    import re  # noqa: PLC0415

    m = re.search(r"_(\d+)\.gninatypes$", gninatypes_path)
    return int(m.group(1)) if m else 0


def _parse_types_file(types_path: Path) -> list[tuple[str, str, int]]:
    """Parse a .types file to extract (receptor_pdb, ligand_sdf, pose_idx).

    Each line has format:
        label score1 score2 receptor.gninatypes ligand.gninatypes #comment

    Each docked pose is kept as a separate entry.
    """
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
    """Parse train0 and test0 types files and return all entries.

    Each docked pose is a separate entry (no deduplication).
    When *max_pairs* is set, stops reading once enough entries have
    been collected.
    """
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
) -> tuple[list[tuple[str, str]], Path]:
    """Load pairs from a HuggingFace Hub manifest.

    Returns ``(pairs, base_dir)`` where *pairs* are
    ``(receptor_rel_path, ligand_rel_path)`` and *base_dir* is the
    root directory that both paths are relative to.

    The receptor path is relative to ``hub_cache/receptors/`` and the
    ligand path is relative to ``hub_cache/ligands/``.  To unify them
    under a single base directory we return the cache root and use
    prefixed paths.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    cache_dir = Path(hub_config.cache_dir)
    manifest_path = cache_dir / "repo" / "manifest.parquet"
    if not manifest_path.exists():
        msg = f"Manifest not found: {manifest_path}"
        raise FileNotFoundError(msg)

    table = pq.read_table(manifest_path)
    df = table.to_pandas()

    # Filter by source_type
    if hub_config.source_types:
        df = df[df["source_type"].isin(hub_config.source_types)]

    if max_pairs is not None:
        df = df.head(max_pairs)

    pairs: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        rec_rel = f"{row['complex_dir']}/{row['receptor_pdb']}"
        lig_rel = f"{row['pair_idx']:07d}.sdf.gz"
        pairs.append((rec_rel, lig_rel))

    logger.info("Loaded %d pairs from manifest", len(pairs))
    return pairs, cache_dir


def _worker_init(pocket_config_dict: dict) -> None:
    """Initialize per-worker descriptor calculators (called once per process)."""
    global _worker_protein_desc, _worker_ligand_desc, _worker_pocket_config  # noqa: PLW0603

    from src.config import PocketExtractionConfig  # noqa: PLC0415

    _worker_protein_desc = BackboneZMatrixDescriptor()
    _worker_ligand_desc = LigandDescriptor()
    _worker_pocket_config = PocketExtractionConfig(**pocket_config_dict)


# ---------------------------------------------------------------------------
# Grouped processing: parse each PDB/SDF once across all poses
# ---------------------------------------------------------------------------


def _group_entries_by_receptor(
    abs_entries: list[tuple[str, str, int]],
) -> list[tuple[str, list[tuple[str, list[int]]]]]:
    """Group entries by receptor PDB, then by SDF file.

    Returns a list of ``(rec_path, sdf_groups)`` where *sdf_groups* is
    ``[(sdf_path, [pose_indices]), ...]``.
    """
    from collections import defaultdict  # noqa: PLC0415

    # rec -> sdf -> [pose_idx]
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for rec, lig, pose_idx in abs_entries:
        grouped[rec][lig].append(pose_idx)

    return [
        (rec, [(sdf, poses) for sdf, poses in sdf_dict.items()])
        for rec, sdf_dict in grouped.items()
    ]


def _process_pose(
    mol: dict,
    precomputed: object,
    pocket_config: object,
    protein_desc: BackboneZMatrixDescriptor,
    ligand_desc: LigandDescriptor,
) -> dict[str, np.ndarray | list[str]] | None:
    """Process a single pose given a pre-parsed molecule and receptor."""
    heavy = [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"]
    if not heavy:
        return None
    lig_coords = np.array(heavy, dtype=np.float32)

    pocket = extract_pocket_from_candidates(
        precomputed,
        lig_coords,
        pocket_config,  # type: ignore[arg-type]
    )
    if pocket is None:
        return None
    backbone_coords, _pocket_seq, residue_ids = pocket

    ca_coords = backbone_coords[:, 1].astype(np.float64)
    centroid, rotation = _compute_canonical_frame(ca_coords)
    pocket_frame = (centroid, rotation)

    prot_desc_arr, _prot_meta = protein_desc.compute(
        backbone_coords,
        residue_ids,
        pocket_frame=pocket_frame,
    )
    lig_desc_arr, elements, _lig_meta = ligand_desc.compute(
        mol["atoms"],
        mol["bonds"],
        pocket_frame=pocket_frame,
    )
    if len(lig_desc_arr) == 0:
        return None

    return {"protein": prot_desc_arr, "ligand": lig_desc_arr, "elements": elements}


def _worker_process_receptor_group(
    args: tuple[str, list[tuple[str, list[int]]]],
) -> tuple[list[dict[str, np.ndarray | list[str]]], int]:
    """Process all entries for one receptor PDB.

    Parses the PDB once and each SDF once across all poses.
    Returns ``(results, total_poses)`` where *total_poses* is the number
    of poses attempted (for progress tracking).
    """
    rec_path, sdf_groups = args
    results: list[dict[str, np.ndarray | list[str]]] = []
    total = sum(len(poses) for _, poses in sdf_groups)

    rec_full = Path(rec_path)
    if not rec_full.exists():
        return results, total

    try:
        precomputed = precompute_pocket_candidates(rec_full)
    except Exception:
        logger.exception("Error parsing PDB %s", rec_path)
        return results, total

    for sdf_path, pose_indices in sdf_groups:
        sdf_full = Path(sdf_path)
        if not sdf_full.exists():
            continue

        try:
            molecules = parse_sdf(sdf_full)
        except Exception:
            logger.exception("Error parsing SDF %s", sdf_path)
            continue

        for pose_idx in pose_indices:
            if pose_idx >= len(molecules):
                continue
            try:
                result = _process_pose(
                    molecules[pose_idx],
                    precomputed,
                    _worker_pocket_config,
                    _worker_protein_desc,  # type: ignore[arg-type]
                    _worker_ligand_desc,  # type: ignore[arg-type]
                )
            except Exception:
                logger.exception(
                    "Error: %s / %s pose %d",
                    rec_path,
                    sdf_path,
                    pose_idx,
                )
                continue
            if result is not None:
                results.append(result)

    return results, total


def _write_shard(
    shard_dir: Path,
    shard_idx: int,
    data: list[dict[str, np.ndarray | list[str]]],
) -> None:
    """Write one shard file to disk."""
    torch.save(data, shard_dir / f"shard_{shard_idx:04d}.pt")


def _collect_grouped_results_sharded(
    batch_results: Iterable[tuple[list[dict[str, np.ndarray | list[str]]], int]],
    total_entries: int,
    shard_dir: Path,
    shard_size: int = _DEFAULT_SHARD_SIZE,
) -> tuple[int, list[int], set[str]]:
    """Collect results from grouped processing, writing shards incrementally.

    Returns ``(total_count, shard_counts, unique_elements)`` where
    *shard_counts* is the number of complexes in each shard file.
    """
    buffer: list[dict[str, np.ndarray | list[str]]] = []
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

        # Flush full shards
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

    # Flush remaining buffer
    if buffer:
        for cplx in buffer:
            unique_elements.update(cplx["elements"])  # type: ignore[arg-type]
        _write_shard(shard_dir, shard_idx, buffer)
        shard_counts.append(len(buffer))
        total_count += len(buffer)
        logger.info("Wrote shard %04d (%d complexes)", shard_idx, shard_counts[-1])

    logger.info("Done: %d processed, %d skipped", total_count, num_skipped)
    return total_count, shard_counts, unique_elements


def _process_entries_sharded(
    abs_entries: list[tuple[str, str, int]],
    pocket_config: PocketExtractionConfig,
    shard_dir: Path,
    num_workers: int = 0,
    shard_size: int = _DEFAULT_SHARD_SIZE,
) -> tuple[int, list[int], set[str]]:
    """Process entries and write shards incrementally.

    Returns ``(total_count, shard_counts, unique_elements)``.
    """
    from dataclasses import asdict  # noqa: PLC0415

    total_entries = len(abs_entries)
    receptor_groups = _group_entries_by_receptor(abs_entries)
    logger.info(
        "Grouped %d entries into %d receptor groups",
        total_entries,
        len(receptor_groups),
    )
    pocket_config_dict = asdict(pocket_config)

    if num_workers > 0:
        import multiprocessing  # noqa: PLC0415

        logger.info("Processing with %d workers", num_workers)
        with multiprocessing.Pool(
            num_workers,
            initializer=_worker_init,
            initargs=(pocket_config_dict,),
        ) as pool:
            batch_results = pool.imap_unordered(
                _worker_process_receptor_group,
                receptor_groups,
                chunksize=4,
            )
            return _collect_grouped_results_sharded(
                batch_results,
                total_entries,
                shard_dir,
                shard_size,
            )

    _worker_init(pocket_config_dict)
    batch_iter = (_worker_process_receptor_group(group) for group in receptor_groups)
    return _collect_grouped_results_sharded(
        batch_iter,
        total_entries,
        shard_dir,
        shard_size,
    )


def _save_shard_metadata(
    cache_dir: Path,
    total_count: int,
    shard_counts: list[int],
    unique_elements: set[str],
) -> None:
    """Save shard metadata and element vocabulary."""
    metadata = {
        "total_count": total_count,
        "shard_counts": shard_counts,
    }
    torch.save(metadata, cache_dir / _SHARD_METADATA_FILE)
    torch.save(sorted(unique_elements), cache_dir / "ligand_elements.pt")
    logger.info(
        "Cached %d complexes in %d shards",
        total_count,
        len(shard_counts),
    )


def _iter_shards(
    shard_dir: Path,
    shard_counts: list[int],
) -> Iterable[tuple[int, list[dict[str, np.ndarray | list[str]]]]]:
    """Yield ``(global_offset, shard_data)`` for each shard file."""
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


def _welford_update_batch(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    batch: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Merge a batch of observations into Welford accumulators (Chan's algorithm)."""
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


class MoleculeDataset(Dataset[Tensor]):
    """Dataset of variable-length molecule descriptor sequences."""

    def __init__(self, molecules: list[Tensor]) -> None:
        self.molecules = molecules

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Tensor:
        return self.molecules[idx]  # (N_atoms, descriptor_dim)


def collate_molecules(batch: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Pad variable-length molecules and create attention mask.

    Args:
        batch: List of tensors, each ``(N_atoms_i, descriptor_dim)``.

    Returns:
        Tuple of ``(padded, mask)`` where *padded* has shape
        ``(B, max_len, descriptor_dim)`` and *mask* is a boolean tensor
        of shape ``(B, max_len)`` (``True`` for real atoms).
    """
    max_len = max(mol.shape[0] for mol in batch)
    descriptor_dim = batch[0].shape[1]

    padded = torch.zeros(len(batch), max_len, descriptor_dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, mol in enumerate(batch):
        n = mol.shape[0]
        padded[i, :n] = mol
        mask[i, :n] = True

    return padded, mask


class ShardedMoleculeDataset(IterableDataset[Tensor]):
    """Lazily streams normalized descriptors from on-disk shards.

    Each shard file is loaded once per epoch per worker, avoiding the need
    to hold all data in RAM simultaneously.
    """

    def __init__(  # noqa: PLR0913
        self,
        shard_dir: Path,
        shard_plan: list[tuple[int, list[int]]],
        key: str,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.shard_dir = shard_dir
        self.shard_plan = shard_plan
        self.key = key
        self.mean = mean
        self.std = std
        self.shuffle = shuffle
        self.length = sum(len(indices) for _, indices in shard_plan)

    def __len__(self) -> int:
        """Total number of items (used by Lightning for progress bars)."""
        return self.length

    def __iter__(self):  # noqa: ANN204
        import random as _random  # noqa: PLC0415

        worker_info = torch.utils.data.get_worker_info()
        plan = self.shard_plan

        if worker_info is not None:
            # Round-robin partition of shards across workers
            plan = [
                plan[i]
                for i in range(len(plan))
                if i % worker_info.num_workers == worker_info.id
            ]

        if self.shuffle:
            rng = _random.Random()  # noqa: S311
            plan = [(si, list(li)) for si, li in plan]
            rng.shuffle(plan)
            for _, li in plan:
                rng.shuffle(li)

        for shard_idx, local_indices in plan:
            shard_path = self.shard_dir / f"shard_{shard_idx:04d}.pt"
            shard_data: list[dict[str, np.ndarray]] = torch.load(
                shard_path,
                weights_only=False,
            )
            for local_idx in local_indices:
                desc = shard_data[local_idx][self.key]
                yield torch.from_numpy(
                    (desc - self.mean) / self.std,
                ).float()
            del shard_data


def _split_and_normalize(  # noqa: PLR0913
    complexes: list[dict[str, np.ndarray | list[str]]],
    indices: list[int],
    protein_mean: np.ndarray,
    protein_std: np.ndarray,
    ligand_mean: np.ndarray,
    ligand_std: np.ndarray,
) -> tuple[list[Tensor], list[Tensor]]:
    """Extract, normalize, and build per-complex descriptor lists for a split.

    Returns ``(protein_pocket_list, ligand_molecule_list)``.
    """
    protein_parts = [complexes[i]["protein"] for i in indices]
    ligand_parts = [complexes[i]["ligand"] for i in indices]

    protein_pockets = [
        torch.from_numpy(
            (prot - protein_mean) / protein_std  # type: ignore[operator]
        ).float()
        for prot in protein_parts
    ]

    ligand_molecules = [
        torch.from_numpy(
            (lig - ligand_mean) / ligand_std  # type: ignore[operator]
        ).float()
        for lig in ligand_parts
    ]

    return protein_pockets, ligand_molecules


class ComplexDescriptorDataModule(L.LightningDataModule):
    """DataModule that computes and caches descriptors for VQ-VAE training.

    Data is split at the **complex** level to prevent data leakage between
    train, validation, and test sets.
    """

    def __init__(
        self,
        training_config: VQVAETrainingConfig,
        data_config: CrossDockedConfig,
        hub_config: HubDatasetConfig | None = None,
    ) -> None:
        super().__init__()
        self.training_config = training_config
        self.data_config = data_config
        self.hub_config = hub_config
        self.data_dir = Path(data_config.data_dir)
        self.cache_dir = self.data_dir / "descriptor_cache"

        self.protein_train: list[Tensor] | None = None
        self.protein_val: list[Tensor] | None = None
        self.protein_test: list[Tensor] | None = None
        self.ligand_train: list[Tensor] | None = None
        self.ligand_val: list[Tensor] | None = None
        self.ligand_test: list[Tensor] | None = None
        self.norm_stats: dict[str, Tensor] | None = None

        # Sharded lazy-loading state (set by _setup_from_shards)
        self._train_plan: list[tuple[int, list[int]]] | None = None
        self._val_plan: list[tuple[int, list[int]]] | None = None
        self._test_plan: list[tuple[int, list[int]]] | None = None
        self._shard_dir: Path | None = None

    def prepare_data(self) -> None:
        """Compute descriptors from CrossDocked2020 and cache to disk."""
        if (self.cache_dir / _SHARD_METADATA_FILE).exists() or (
            self.cache_dir / "complexes.pt"
        ).exists():
            logger.info("Descriptor cache already exists at %s", self.cache_dir)
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        shard_dir = self.cache_dir / _SHARD_DIR_NAME
        shard_dir.mkdir(parents=True, exist_ok=True)

        if self.hub_config is not None:
            pairs, cache_dir = _load_pairs_from_manifest(
                self.hub_config,
                max_pairs=self.data_config.max_pairs,
            )
            receptor_dir = cache_dir / "receptors"
            ligand_dir = cache_dir / "ligands"
            abs_entries = [
                (str(receptor_dir / rec), str(ligand_dir / lig), 0)
                for rec, lig in pairs
            ]
        else:
            types_dir = self.data_dir / "types"
            entries = _parse_all_types_files(
                types_dir,
                max_pairs=self.data_config.max_pairs,
            )
            crossdocked_dir = self.data_dir / "CrossDocked2020"
            abs_entries = [
                (str(crossdocked_dir / rec), str(crossdocked_dir / lig), pose_idx)
                for rec, lig, pose_idx in entries
            ]

        logger.info("Processing %d complex poses", len(abs_entries))

        total_count, shard_counts, unique_elements = _process_entries_sharded(
            abs_entries,
            self.training_config.pocket,
            shard_dir=shard_dir,
            num_workers=self.training_config.num_workers,
        )

        if total_count == 0:
            msg = "No descriptors computed -- check data paths"
            raise RuntimeError(msg)

        _save_shard_metadata(self.cache_dir, total_count, shard_counts, unique_elements)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Load cached descriptors and split into train/val/test at complex level."""
        if (self.cache_dir / _SHARD_METADATA_FILE).exists():
            self._setup_from_shards()
        elif (self.cache_dir / "complexes.pt").exists():
            self._setup_from_monolithic()
        else:
            msg = f"No descriptor cache found at {self.cache_dir}"
            raise FileNotFoundError(msg)

    def _setup_from_monolithic(self) -> None:
        """Load from legacy monolithic ``complexes.pt`` cache."""
        complexes: list[dict[str, np.ndarray | list[str]]] = torch.load(
            self.cache_dir / "complexes.pt",
            weights_only=False,
        )

        n = len(complexes)
        rng = torch.Generator().manual_seed(self.data_config.random_state)
        perm = torch.randperm(n, generator=rng).tolist()

        n_test = int(n * self.data_config.test_size)
        n_val = int(n * self.data_config.val_size)
        test_indices = perm[:n_test]
        val_indices = perm[n_test : n_test + n_val]
        train_indices = perm[n_test + n_val :]

        logger.info(
            "Complex-level split: %d train, %d val, %d test (total %d)",
            len(train_indices),
            len(val_indices),
            len(test_indices),
            n,
        )

        train_protein = np.concatenate(
            [complexes[i]["protein"] for i in train_indices],
            axis=0,  # type: ignore[arg-type]
        )
        train_ligand = np.concatenate(
            [complexes[i]["ligand"] for i in train_indices],
            axis=0,  # type: ignore[arg-type]
        )
        protein_mean = train_protein.mean(axis=0)
        protein_std = train_protein.std(axis=0) + 1e-8
        ligand_mean = train_ligand.mean(axis=0)
        ligand_std = train_ligand.std(axis=0) + 1e-8

        self.norm_stats = {
            "protein_mean": torch.from_numpy(protein_mean),
            "protein_std": torch.from_numpy(protein_std),
            "ligand_mean": torch.from_numpy(ligand_mean),
            "ligand_std": torch.from_numpy(ligand_std),
        }
        torch.save(self.norm_stats, self.cache_dir / _NORMALIZATION_STATS_FILE)

        self.protein_train, self.ligand_train = _split_and_normalize(
            complexes, train_indices, protein_mean, protein_std, ligand_mean, ligand_std
        )
        self.protein_val, self.ligand_val = _split_and_normalize(
            complexes, val_indices, protein_mean, protein_std, ligand_mean, ligand_std
        )
        self.protein_test, self.ligand_test = _split_and_normalize(
            complexes, test_indices, protein_mean, protein_std, ligand_mean, ligand_std
        )

        self._log_split_sizes()

    def _setup_from_shards(self) -> None:
        """Stream-load shards with two passes: stats then normalization."""
        metadata: dict = torch.load(
            self.cache_dir / _SHARD_METADATA_FILE,
            weights_only=False,
        )
        n = metadata["total_count"]
        shard_counts: list[int] = metadata["shard_counts"]
        shard_dir = self.cache_dir / _SHARD_DIR_NAME

        # --- Deterministic complex-level split ---
        rng = torch.Generator().manual_seed(self.data_config.random_state)
        perm = torch.randperm(n, generator=rng).tolist()

        n_test = int(n * self.data_config.test_size)
        n_val = int(n * self.data_config.val_size)
        train_set = set(perm[n_test + n_val :])
        val_set = set(perm[n_test : n_test + n_val])
        test_set = set(perm[:n_test])

        logger.info(
            "Complex-level split: %d train, %d val, %d test (total %d)",
            len(train_set),
            len(val_set),
            len(test_set),
            n,
        )

        # --- Pass 1: Compute normalization stats from train split (Welford) ---
        stats_path = self.cache_dir / _NORMALIZATION_STATS_FILE
        if stats_path.exists():
            self.norm_stats = torch.load(stats_path, weights_only=False)
            logger.info("Loaded cached normalization stats from %s", stats_path)
        else:
            prot_count = 0
            prot_mean = np.zeros(12, dtype=np.float64)
            prot_m2 = np.zeros(12, dtype=np.float64)
            lig_count = 0
            lig_mean = np.zeros(4, dtype=np.float64)
            lig_m2 = np.zeros(4, dtype=np.float64)

            for global_offset, shard_data in _iter_shards(shard_dir, shard_counts):
                for local_idx, cplx in enumerate(shard_data):
                    if (global_offset + local_idx) not in train_set:
                        continue
                    prot_count, prot_mean, prot_m2 = _welford_update_batch(
                        prot_count,
                        prot_mean,
                        prot_m2,
                        cplx["protein"].astype(np.float64),  # type: ignore[union-attr]
                    )
                    lig_count, lig_mean, lig_m2 = _welford_update_batch(
                        lig_count,
                        lig_mean,
                        lig_m2,
                        cplx["ligand"].astype(np.float64),  # type: ignore[union-attr]
                    )

            protein_mean = prot_mean.astype(np.float32)
            protein_std = (np.sqrt(prot_m2 / prot_count) + 1e-8).astype(np.float32)
            ligand_mean = lig_mean.astype(np.float32)
            ligand_std = (np.sqrt(lig_m2 / lig_count) + 1e-8).astype(np.float32)

            self.norm_stats = {
                "protein_mean": torch.from_numpy(protein_mean),
                "protein_std": torch.from_numpy(protein_std),
                "ligand_mean": torch.from_numpy(ligand_mean),
                "ligand_std": torch.from_numpy(ligand_std),
            }
            torch.save(self.norm_stats, stats_path)
            logger.info("Pass 1 done: normalization stats computed from train split")

        # --- Build per-split shard plans (no full data load needed) ---
        from collections import defaultdict  # noqa: PLC0415

        train_by_shard: dict[int, list[int]] = defaultdict(list)
        val_by_shard: dict[int, list[int]] = defaultdict(list)
        test_by_shard: dict[int, list[int]] = defaultdict(list)

        global_offset = 0
        for shard_idx, count in enumerate(shard_counts):
            for local_idx in range(count):
                global_idx = global_offset + local_idx
                if global_idx in train_set:
                    train_by_shard[shard_idx].append(local_idx)
                elif global_idx in val_set:
                    val_by_shard[shard_idx].append(local_idx)
                else:
                    test_by_shard[shard_idx].append(local_idx)
            global_offset += count

        self._shard_dir = shard_dir
        self._train_plan = sorted(train_by_shard.items())
        self._val_plan = sorted(val_by_shard.items())
        self._test_plan = sorted(test_by_shard.items())

        logger.info("Shard plans built (no full data load needed)")
        self._log_split_sizes()

    def _log_split_sizes(self) -> None:
        if self._train_plan is not None:
            n_train = sum(len(li) for _, li in self._train_plan)
            n_val = sum(len(li) for _, li in self._val_plan or [])
            n_test = sum(len(li) for _, li in self._test_plan or [])
            logger.info(
                "Complexes: %d train, %d val, %d test",
                n_train,
                n_val,
                n_test,
            )
            return
        logger.info(
            "Protein pockets: %d train, %d val, %d test",
            len(self.protein_train),  # type: ignore[arg-type]
            len(self.protein_val),  # type: ignore[arg-type]
            len(self.protein_test),  # type: ignore[arg-type]
        )
        logger.info(
            "Ligand molecules: %d train, %d val, %d test",
            len(self.ligand_train),  # type: ignore[arg-type]
            len(self.ligand_val),  # type: ignore[arg-type]
            len(self.ligand_test),  # type: ignore[arg-type]
        )

    def _build_loader(
        self,
        molecules: list[Tensor],
        *,
        shuffle: bool,
    ) -> DataLoader:
        nw = self.training_config.num_workers
        return DataLoader(
            MoleculeDataset(molecules),
            batch_size=self.training_config.mol_batch_size,
            shuffle=shuffle,
            num_workers=nw,
            persistent_workers=nw > 0,
            pin_memory=True,
            collate_fn=collate_molecules,
        )

    def _build_sharded_loader(
        self,
        shard_plan: list[tuple[int, list[int]]],
        key: str,
        *,
        shuffle: bool,
    ) -> DataLoader:
        """Build a DataLoader backed by :class:`ShardedMoleculeDataset`."""
        dataset = ShardedMoleculeDataset(
            shard_dir=self._shard_dir,  # type: ignore[arg-type]
            shard_plan=shard_plan,
            key=key,
            mean=self.norm_stats[f"{key}_mean"].numpy(),  # type: ignore[union-attr]
            std=self.norm_stats[f"{key}_std"].numpy(),  # type: ignore[union-attr]
            shuffle=shuffle,
        )
        nw = self.training_config.num_workers
        return DataLoader(
            dataset,
            batch_size=self.training_config.mol_batch_size,
            num_workers=nw,
            persistent_workers=nw > 0,
            pin_memory=True,
            collate_fn=collate_molecules,
        )

    def _make_combined_loader(
        self,
        shard_plan: list[tuple[int, list[int]]] | None,
        protein_mols: list[Tensor] | None,
        ligand_mols: list[Tensor] | None,
        *,
        shuffle: bool,
    ) -> CombinedLoader:
        if shard_plan is not None:
            return CombinedLoader(
                {
                    "protein": self._build_sharded_loader(
                        shard_plan,
                        "protein",
                        shuffle=shuffle,
                    ),
                    "ligand": self._build_sharded_loader(
                        shard_plan,
                        "ligand",
                        shuffle=shuffle,
                    ),
                },
                mode="max_size_cycle",
            )
        if protein_mols is None or ligand_mols is None:
            msg = "setup() must be called before creating dataloaders"
            raise RuntimeError(msg)
        return CombinedLoader(
            {
                "protein": self._build_loader(protein_mols, shuffle=shuffle),
                "ligand": self._build_loader(ligand_mols, shuffle=shuffle),
            },
            mode="max_size_cycle",
        )

    def train_dataloader(self) -> CombinedLoader:
        """Return train dataloaders for protein and ligand."""
        return self._make_combined_loader(
            self._train_plan,
            self.protein_train,
            self.ligand_train,
            shuffle=True,
        )

    def val_dataloader(self) -> CombinedLoader:
        """Return validation dataloaders for protein and ligand."""
        return self._make_combined_loader(
            self._val_plan,
            self.protein_val,
            self.ligand_val,
            shuffle=False,
        )

    def test_dataloader(self) -> CombinedLoader:
        """Return test dataloaders for protein and ligand."""
        return self._make_combined_loader(
            self._test_plan,
            self.protein_test,
            self.ligand_test,
            shuffle=False,
        )
