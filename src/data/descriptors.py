"""DataModule for computing and caching VQ-VAE training descriptors.

Each shard entry stores ``{"protein", "ligand", "elements", "pair_idx"}``
where ``protein`` is ``(L, PROTEIN_DESCRIPTOR_DIM)`` and ``ligand`` is
``(N_atoms, LIGAND_DESCRIPTOR_DIM)``. Both descriptors are spherical-from-
pocket-centroid, with element / AA / atom-feature columns embedded directly
in the descriptor (see :mod:`src.tokenizers.descriptor_schema`).

Normalization is computed only over **continuous** slots; categorical
columns are forced to mean=0, std=1 so values pass through unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

from typing import ClassVar

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset

from src.tokenizers.descriptor_schema import (
    LIGAND_DESCRIPTOR_DIM,
    LIGAND_LAYOUT,
    PROTEIN_DESCRIPTOR_DIM,
    PROTEIN_LAYOUT,
    continuous_mask,
)
from src.tokenizers.ligand import LigandDescriptor, parse_sdf
from src.tokenizers.protein import (
    AA_3TO1,
    BackboneSphericalDescriptor,
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
# Bump when shard layout changes; _setup_from_shards refuses older caches.
_SHARD_SCHEMA_VERSION = 4

# Per-worker state for multiprocessing (set by _worker_init).
_worker_protein_desc: BackboneSphericalDescriptor | None = None
_worker_ligand_desc: LigandDescriptor | None = None
_worker_pocket_config: PocketExtractionConfig | None = None


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


def _worker_init(pocket_config_dict: dict) -> None:
    """Initialize per-worker descriptor calculators (called once per process)."""
    global _worker_protein_desc, _worker_ligand_desc, _worker_pocket_config  # noqa: PLW0603

    from src.config import PocketExtractionConfig  # noqa: PLC0415

    _worker_protein_desc = BackboneSphericalDescriptor()
    _worker_ligand_desc = LigandDescriptor()
    _worker_pocket_config = PocketExtractionConfig(**pocket_config_dict)


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


def _process_pose(
    mol: dict,
    precomputed: object,
    pocket_config: object,
    protein_desc: BackboneSphericalDescriptor,
    ligand_desc: LigandDescriptor,
) -> dict[str, np.ndarray | list[str]] | None:
    """Process one (receptor, ligand pose) pair into shard-ready descriptors."""
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
    backbone_coords, pocket_seq, residue_ids = pocket

    ca_coords = backbone_coords[:, 1].astype(np.float64)
    centroid, rotation = _compute_canonical_frame(ca_coords)
    pocket_frame = (centroid, rotation)

    prot_desc_arr, _prot_meta = protein_desc.compute(
        backbone_coords,
        residue_ids,
        pocket_frame=pocket_frame,
        residue_names_one_letter=list(pocket_seq),
    )
    lig_desc_arr, elements, _lig_meta = ligand_desc.compute(
        mol["atoms"],
        mol["bonds"],
        pocket_frame=pocket_frame,
    )
    if len(lig_desc_arr) == 0:
        return None

    return {
        "protein": prot_desc_arr,
        "ligand": lig_desc_arr,
        "elements": elements,
    }


def _worker_process_receptor_group(
    args: tuple[str, list[tuple[str, list[tuple[int, int]]]]],
) -> tuple[list[dict[str, np.ndarray | list[str] | int]], int]:
    rec_path, sdf_groups = args
    results: list[dict[str, np.ndarray | list[str] | int]] = []
    total = sum(len(poses) for _, poses in sdf_groups)

    rec_full = Path(rec_path)
    if not rec_full.exists():
        return results, total

    try:
        precomputed = precompute_pocket_candidates(rec_full)
    except Exception:
        logger.exception("Error parsing PDB %s", rec_path)
        return results, total

    for sdf_path, pose_specs in sdf_groups:
        sdf_full = Path(sdf_path)
        if not sdf_full.exists():
            continue
        try:
            molecules = parse_sdf(sdf_full)
        except Exception:
            logger.exception("Error parsing SDF %s", sdf_path)
            continue
        for pose_idx, pair_idx in pose_specs:
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
                result["pair_idx"] = pair_idx
                results.append(result)

    return results, total


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


def _process_entries_sharded(
    abs_entries: list[tuple[str, str, int, int]],
    pocket_config: PocketExtractionConfig,
    shard_dir: Path,
    num_workers: int = 0,
    shard_size: int = _DEFAULT_SHARD_SIZE,
) -> tuple[int, list[int], set[str]]:
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


class ShardedMoleculeDataset(IterableDataset):
    """Lazily streams normalized descriptors from shards.

    Yields one descriptor tensor per entry (collation pads + builds the mask).
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
        if key not in {"protein", "ligand"}:
            msg = f"Unsupported shard key: {key!r}"
            raise ValueError(msg)
        self.shard_dir = shard_dir
        self.shard_plan = shard_plan
        self.key = key
        self.mean = mean
        self.std = std
        self.shuffle = shuffle
        self.length = sum(len(indices) for _, indices in shard_plan)

    def __len__(self) -> int:
        return self.length

    def __iter__(self):  # noqa: ANN204
        import random as _random  # noqa: PLC0415

        worker_info = torch.utils.data.get_worker_info()
        plan = self.shard_plan

        if worker_info is not None:
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
                entry = shard_data[local_idx]
                desc = entry[self.key]
                yield torch.from_numpy((desc - self.mean) / self.std).float()
            del shard_data


class ComplexDescriptorDataModule(L.LightningDataModule):
    """DataModule that computes and caches descriptors for VQ-VAE training."""

    AUX_KEY: ClassVar[dict[str, str]] = {"ligand": "ligand", "protein": "protein"}

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

        self._train_plan: list[tuple[int, list[int]]] | None = None
        self._val_plan: list[tuple[int, list[int]]] | None = None
        self._test_plan: list[tuple[int, list[int]]] | None = None
        self._shard_dir: Path | None = None

    def prepare_data(self) -> None:
        if (self.cache_dir / _SHARD_METADATA_FILE).exists():
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
                (str(receptor_dir / rec), str(ligand_dir / lig), 0, pair_idx)
                for rec, lig, pair_idx in pairs
            ]
        else:
            types_dir = self.data_dir / "types"
            entries = _parse_all_types_files(
                types_dir,
                max_pairs=self.data_config.max_pairs,
            )
            crossdocked_dir = self.data_dir / "CrossDocked2020"
            abs_entries = [
                (str(crossdocked_dir / rec), str(crossdocked_dir / lig), pose_idx, -1)
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
        if not (self.cache_dir / _SHARD_METADATA_FILE).exists():
            msg = (
                f"No descriptor shard cache found at {self.cache_dir}. "
                "Run prepare_data() first."
            )
            raise FileNotFoundError(msg)
        self._setup_from_shards()

    # ------------------------------------------------------------------
    # Shard-based setup
    # ------------------------------------------------------------------

    def _setup_from_shards(self) -> None:  # noqa: C901, PLR0912, PLR0915
        metadata: dict = torch.load(
            self.cache_dir / _SHARD_METADATA_FILE,
            weights_only=False,
        )
        cached_version = int(metadata.get("schema_version", 1))
        if cached_version < _SHARD_SCHEMA_VERSION:
            msg = (
                f"Shard cache at {self.cache_dir} is schema v{cached_version} "
                f"(expected v{_SHARD_SCHEMA_VERSION}). Delete the "
                "descriptor_cache directory and re-run prepare_data() "
                "to regenerate."
            )
            raise RuntimeError(msg)
        n = metadata["total_count"]
        shard_counts: list[int] = metadata["shard_counts"]
        shard_dir = self.cache_dir / _SHARD_DIR_NAME

        if self.hub_config is not None:
            train_set, val_set, test_set = self._fold_split_from_manifest(
                shard_dir,
                shard_counts,
            )
        else:
            logger.warning(
                "types-file path: using legacy random %.0f/%.0f/%.0f split",
                (1 - self.data_config.test_size - self.data_config.val_size) * 100,
                self.data_config.val_size * 100,
                self.data_config.test_size * 100,
            )
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

        # --- Welford pass over train split, then enforce passthrough on
        #     categorical slots so values reach the encoder unchanged.
        stats_path = self.cache_dir / _NORMALIZATION_STATS_FILE
        if stats_path.exists():
            self.norm_stats = torch.load(stats_path, weights_only=False)
            logger.info("Loaded cached normalization stats from %s", stats_path)
        else:
            prot_count = 0
            prot_mean = np.zeros(PROTEIN_DESCRIPTOR_DIM, dtype=np.float64)
            prot_m2 = np.zeros(PROTEIN_DESCRIPTOR_DIM, dtype=np.float64)
            lig_count = 0
            lig_mean = np.zeros(LIGAND_DESCRIPTOR_DIM, dtype=np.float64)
            lig_m2 = np.zeros(LIGAND_DESCRIPTOR_DIM, dtype=np.float64)

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
            protein_std = (np.sqrt(prot_m2 / max(prot_count, 1)) + 1e-8).astype(
                np.float32
            )
            ligand_mean = lig_mean.astype(np.float32)
            ligand_std = (np.sqrt(lig_m2 / max(lig_count, 1)) + 1e-8).astype(np.float32)

            protein_mean, protein_std = _force_passthrough_for_categorical(
                protein_mean,
                protein_std,
                continuous_mask(PROTEIN_LAYOUT),
            )
            ligand_mean, ligand_std = _force_passthrough_for_categorical(
                ligand_mean,
                ligand_std,
                continuous_mask(LIGAND_LAYOUT),
            )

            self.norm_stats = {
                "protein_mean": torch.from_numpy(protein_mean),
                "protein_std": torch.from_numpy(protein_std),
                "ligand_mean": torch.from_numpy(ligand_mean),
                "ligand_std": torch.from_numpy(ligand_std),
            }
            torch.save(self.norm_stats, stats_path)
            logger.info("Welford normalization stats computed from train split")

        # --- Per-split shard plans -------------------------------------
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
                elif global_idx in test_set:
                    test_by_shard[shard_idx].append(local_idx)
            global_offset += count

        self._shard_dir = shard_dir
        self._train_plan = sorted(train_by_shard.items())
        self._val_plan = sorted(val_by_shard.items())
        self._test_plan = sorted(test_by_shard.items())

        logger.info("Shard plans built (no full data load needed)")
        self._log_split_sizes()

    def _fold_split_from_manifest(
        self,
        shard_dir: Path,
        shard_counts: list[int],
    ) -> tuple[set[int], set[int], set[int]]:
        import pyarrow.parquet as pq  # noqa: PLC0415

        assert self.hub_config is not None  # noqa: S101
        manifest_path = Path(self.hub_config.cache_dir) / "repo" / "manifest.parquet"
        fold = self.hub_config.fold
        source_types = list(self.hub_config.source_types)

        # Some source types (notably "other") have no ``{st}_fold{fold}`` column
        # in the manifest. Request only the columns that exist; entries whose
        # source type lacks a fold column are treated as trainval ("train") so
        # they are not silently dropped from the corpus.
        available = set(pq.read_schema(manifest_path).names)
        fold_cols = [
            f"{st}_fold{fold}"
            for st in source_types
            if f"{st}_fold{fold}" in available
        ]
        df = pq.read_table(
            manifest_path,
            columns=["pair_idx", "source_type", *fold_cols],
        ).to_pandas()
        df = df[df["source_type"].isin(source_types)]
        fold_map: dict[int, str] = {}
        for row in df.itertuples(index=False):
            col = f"{row.source_type}_fold{fold}"
            label = getattr(row, col, None) if col in available else "train"
            if label is not None:
                fold_map[int(row.pair_idx)] = label

        test_globals: list[int] = []
        trainval_globals: list[int] = []
        missing = 0
        global_offset = 0
        for shard_idx, count in enumerate(shard_counts):
            shard = torch.load(
                shard_dir / f"shard_{shard_idx:04d}.pt",
                weights_only=False,
            )
            for local_idx, cplx in enumerate(shard):
                gi = global_offset + local_idx
                pid = int(cplx["pair_idx"])
                label = fold_map.get(pid)
                if label == "test":
                    test_globals.append(gi)
                elif label == "train":
                    trainval_globals.append(gi)
                else:
                    missing += 1
            global_offset += count
        if missing:
            logger.warning(
                "%d entries missing fold-%d label in manifest; skipped",
                missing,
                fold,
            )

        rng = torch.Generator().manual_seed(self.data_config.random_state)
        perm = torch.randperm(len(trainval_globals), generator=rng).tolist()
        n_val = int(len(trainval_globals) * self.data_config.val_size)
        val_set = {trainval_globals[i] for i in perm[:n_val]}
        train_set = {trainval_globals[i] for i in perm[n_val:]}
        test_set = set(test_globals)

        logger.info(
            "Fold-%d split: %d train, %d val, %d test (total %d, skipped %d)",
            fold,
            len(train_set),
            len(val_set),
            len(test_set),
            len(train_set) + len(val_set) + len(test_set),
            missing,
        )
        return train_set, val_set, test_set

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

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def _build_sharded_loader(
        self,
        shard_plan: list[tuple[int, list[int]]],
        key: str,
        *,
        shuffle: bool,
    ) -> DataLoader:
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
        *,
        shuffle: bool,
    ) -> CombinedLoader:
        if shard_plan is None:
            msg = "setup() must be called before creating dataloaders"
            raise RuntimeError(msg)
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

    def train_dataloader(self) -> CombinedLoader:
        return self._make_combined_loader(self._train_plan, shuffle=True)

    def val_dataloader(self) -> CombinedLoader:
        return self._make_combined_loader(self._val_plan, shuffle=False)

    def test_dataloader(self) -> CombinedLoader:
        return self._make_combined_loader(self._test_plan, shuffle=False)


# ``AA_3TO1`` is re-exported for backward compatibility with any caller
# that imports it via ``src.data.descriptors``.
__all__ = [
    "AA_3TO1",
    "BackboneSphericalDescriptor",
    "ComplexDescriptorDataModule",
    "LigandDescriptor",
    "MoleculeDataset",
    "ShardedMoleculeDataset",
    "collate_molecules",
    "parse_sdf",
]
