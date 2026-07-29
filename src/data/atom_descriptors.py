"""DataModule for the unified all-atom VQ-VAE (one codebook over all atoms).

Differences from :class:`~src.data.descriptors.ComplexDescriptorDataModule`:

- the protein pocket is expanded to **every heavy atom** of the pocket residues
  (not backbone N/CA/C only); protein and ligand atoms share the 33-D
  :data:`ATOM_LAYOUT` descriptor,
- an optional ``label == 1`` (good-pose) filter is applied at manifest load,
- a **single** training stream: each complex contributes its protein-atom
  sequence AND its ligand-atom sequence as separate items, normalized by one
  pooled mean/std vector, so one VQ-VAE / one codebook tokenizes both.

The shard cache lives in ``data/descriptor_cache_allatom`` (the legacy
residue-level cache is left untouched). Sharding / Welford / manifest / fold
helpers are reused from :mod:`src.data.descriptors`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from src.data.descriptors import (
    _DEFAULT_SHARD_SIZE,
    _NORMALIZATION_STATS_FILE,
    _SHARD_DIR_NAME,
    _SHARD_METADATA_FILE,
    _collect_grouped_results_sharded,
    _force_passthrough_for_categorical,
    _group_entries_by_receptor,
    _iter_shards,
    _load_pairs_from_manifest,
    _welford_update_batch,
    collate_molecules,
)
from src.tokenizers.atom import (
    LigandAtomDescriptor,
    ProteinAtomDescriptor,
    precompute_receptor_atom_features,
)
from src.tokenizers.descriptor_schema import (
    ATOM_DESCRIPTOR_DIM,
    ATOM_LAYOUT,
    continuous_mask,
)
from src.tokenizers.ligand import parse_sdf
from src.tokenizers.protein import (
    _compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    precompute_pocket_atom_candidates,
)

if TYPE_CHECKING:
    from src.config import (
        AtomVQVAETrainingConfig,
        CrossDockedConfig,
        HubDatasetConfig,
        PocketExtractionConfig,
    )

logger = logging.getLogger(__name__)

# Bumped independently of the legacy cache. The atom cache stores ``protein`` /
# ``ligand`` arrays both at ATOM_DESCRIPTOR_DIM plus ``elements`` / ``pair_idx``.
_ATOM_SHARD_SCHEMA_VERSION = 1

# Per-worker state (set by ``_atom_worker_init``).
_w_prot_desc: ProteinAtomDescriptor | None = None
_w_lig_desc: LigandAtomDescriptor | None = None
_w_pocket_config: PocketExtractionConfig | None = None


# ---------------------------------------------------------------------------
# Extraction workers
# ---------------------------------------------------------------------------


def _atom_worker_init(pocket_config_dict: dict) -> None:
    global _w_prot_desc, _w_lig_desc, _w_pocket_config  # noqa: PLW0603

    from src.config import PocketExtractionConfig  # noqa: PLC0415

    _w_prot_desc = ProteinAtomDescriptor()
    _w_lig_desc = LigandAtomDescriptor()
    _w_pocket_config = PocketExtractionConfig(**pocket_config_dict)


def _atom_process_pose(  # noqa: PLR0913
    mol: dict,
    precomputed_atoms: object,
    receptor_feats: dict,
    pocket_config: object,
    prot_desc: ProteinAtomDescriptor,
    lig_desc: LigandAtomDescriptor,
    ligand_frame: str = "pocket",
) -> dict[str, np.ndarray | list[str]] | None:
    """Process one (receptor, ligand pose) pair into all-atom descriptors.

    ``ligand_frame`` selects the reference frame for the LIGAND descriptors:

    - ``"pocket"`` (default) — the shared pocket-anchored canonical frame, so
      the ligand's placement relative to the receptor is encoded per atom.
    - ``"local"`` — a canonical frame built from the ligand's own heavy atoms,
      mirroring how single-modality ligand tokenizers (Mol-StrucTok, Geo2Seq)
      encode a molecule. The resulting tokens are SE(3)-invariant and carry NO
      information about where the ligand sits in the pocket; that placement has
      to be transmitted separately. This is the ablation arm that makes the
      interface metrics meaningful.
    """
    heavy = [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"]
    if not heavy:
        return None
    lig_coords = np.array(heavy, dtype=np.float32)

    pocket = extract_pocket_atoms_from_candidates(
        precomputed_atoms,  # type: ignore[arg-type]
        lig_coords,
        pocket_config,  # type: ignore[arg-type]
    )
    if pocket is None or pocket.atom_coords.shape[0] == 0:
        return None

    centroid, rotation = _compute_canonical_frame(pocket.ca_coords.astype(np.float64))
    frame = (centroid, rotation)

    if ligand_frame == "local":
        lig_frame = _compute_canonical_frame(lig_coords.astype(np.float64))
    elif ligand_frame == "pocket":
        lig_frame = frame
    else:
        msg = f"unknown ligand_frame {ligand_frame!r} (expected 'pocket' or 'local')"
        raise ValueError(msg)

    prot_arr, _prot_meta = prot_desc.compute(pocket, receptor_feats, frame)
    lig_arr, elements, _lig_meta = lig_desc.compute(
        mol["atoms"], mol["bonds"], lig_frame
    )
    if len(lig_arr) == 0:
        return None

    return {"protein": prot_arr, "ligand": lig_arr, "elements": elements}


def _atom_worker_process_receptor_group(
    args: tuple[str, list[tuple[str, list[tuple[int, int]]]]],
) -> tuple[list[dict[str, np.ndarray | list[str] | int]], int]:
    rec_path, sdf_groups = args
    results: list[dict[str, np.ndarray | list[str] | int]] = []
    total = sum(len(poses) for _, poses in sdf_groups)

    rec_full = Path(rec_path)
    if not rec_full.exists():
        return results, total

    try:
        precomputed = precompute_pocket_atom_candidates(rec_full)
        receptor_feats = precompute_receptor_atom_features(rec_full)
    except Exception:
        logger.exception("Error parsing receptor %s", rec_path)
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
                result = _atom_process_pose(
                    molecules[pose_idx],
                    precomputed,
                    receptor_feats,
                    _w_pocket_config,
                    _w_prot_desc,  # type: ignore[arg-type]
                    _w_lig_desc,  # type: ignore[arg-type]
                )
            except Exception:
                logger.exception("Error: %s / %s pose %d", rec_path, sdf_path, pose_idx)
                continue
            if result is not None:
                result["pair_idx"] = pair_idx
                results.append(result)
    return results, total


def _atom_process_entries_sharded(
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
            initializer=_atom_worker_init,
            initargs=(pocket_config_dict,),
        ) as pool:
            batch_results = pool.imap_unordered(
                _atom_worker_process_receptor_group,
                receptor_groups,
                chunksize=4,
            )
            return _collect_grouped_results_sharded(
                batch_results, total_entries, shard_dir, shard_size
            )

    _atom_worker_init(pocket_config_dict)
    batch_iter = (
        _atom_worker_process_receptor_group(group) for group in receptor_groups
    )
    return _collect_grouped_results_sharded(
        batch_iter, total_entries, shard_dir, shard_size
    )


def _save_atom_shard_metadata(
    cache_dir: Path,
    total_count: int,
    shard_counts: list[int],
    unique_elements: set[str],
) -> None:
    metadata = {
        "total_count": total_count,
        "shard_counts": shard_counts,
        "schema_version": _ATOM_SHARD_SCHEMA_VERSION,
        "descriptor_kind": "atom",
        "descriptor_dim": ATOM_DESCRIPTOR_DIM,
    }
    torch.save(metadata, cache_dir / _SHARD_METADATA_FILE)
    torch.save(sorted(unique_elements), cache_dir / "ligand_elements.pt")
    logger.info(
        "Cached %d complexes in %d shards (atom schema v%d)",
        total_count,
        len(shard_counts),
        _ATOM_SHARD_SCHEMA_VERSION,
    )


def _atom_fold_split_from_manifest(
    hub_config: HubDatasetConfig,
    random_state: int,
    val_size: float,
    shard_dir: Path,
    shard_counts: list[int],
) -> tuple[set[int], set[int], set[int]]:
    """Official CrossDocked fold split keyed by ``pair_idx`` (see legacy module)."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    manifest_path = Path(hub_config.cache_dir) / "repo" / "manifest.parquet"
    fold = hub_config.fold
    source_types = list(hub_config.source_types)

    available = set(pq.read_schema(manifest_path).names)
    fold_cols = [
        f"{st}_fold{fold}" for st in source_types if f"{st}_fold{fold}" in available
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
        shard = torch.load(shard_dir / f"shard_{shard_idx:04d}.pt", weights_only=False)
        for local_idx, cplx in enumerate(shard):
            gi = global_offset + local_idx
            label = fold_map.get(int(cplx["pair_idx"]))
            if label == "test":
                test_globals.append(gi)
            elif label == "train":
                trainval_globals.append(gi)
            else:
                missing += 1
        global_offset += count
    if missing:
        logger.warning("%d entries missing fold-%d label; skipped", missing, fold)

    rng = torch.Generator().manual_seed(random_state)
    perm = torch.randperm(len(trainval_globals), generator=rng).tolist()
    n_val = int(len(trainval_globals) * val_size)
    val_set = {trainval_globals[i] for i in perm[:n_val]}
    train_set = {trainval_globals[i] for i in perm[n_val:]}
    test_set = set(test_globals)
    logger.info(
        "Fold-%d split: %d train, %d val, %d test",
        fold,
        len(train_set),
        len(val_set),
        len(test_set),
    )
    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# Single-stream dataset
# ---------------------------------------------------------------------------


class AtomShardedDataset(IterableDataset):
    """Streams normalized atom sequences; each entry yields protein + ligand."""

    def __init__(  # noqa: PLR0913
        self,
        shard_dir: Path,
        shard_plan: list[tuple[int, list[int]]],
        mean: np.ndarray,
        std: np.ndarray,
        *,
        shuffle: bool = False,
        keys: tuple[str, ...] = ("protein", "ligand"),
    ) -> None:
        super().__init__()
        self.shard_dir = shard_dir
        self.shard_plan = shard_plan
        self.mean = mean
        self.std = std
        self.shuffle = shuffle
        # Which atom streams to emit per entry. ("protein","ligand") = joint;
        # a single-element tuple trains a single-modality (protein/ligand-only)
        # VQ-VAE on the SAME complexes (the ablation baseline tokenizers).
        self.keys = keys
        self.length = len(keys) * sum(len(indices) for _, indices in shard_plan)

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
            shard_data: list[dict[str, np.ndarray]] = torch.load(
                self.shard_dir / f"shard_{shard_idx:04d}.pt",
                weights_only=False,
            )
            for local_idx in local_indices:
                entry = shard_data[local_idx]
                for key in self.keys:
                    desc = entry[key]
                    if desc.shape[0] == 0:
                        continue
                    yield torch.from_numpy((desc - self.mean) / self.std).float()
            del shard_data


class AtomComplexDescriptorDataModule(L.LightningDataModule):
    """Computes/caches all-atom descriptors and serves one unified VQ-VAE stream."""

    def __init__(
        self,
        training_config: AtomVQVAETrainingConfig,
        data_config: CrossDockedConfig,
        hub_config: HubDatasetConfig | None = None,
        modality: str = "both",
    ) -> None:
        super().__init__()
        self.training_config = training_config
        self.data_config = data_config
        self.hub_config = hub_config
        self.data_dir = Path(data_config.data_dir)
        self.cache_dir = self.data_dir / "descriptor_cache_allatom"

        # Ablation: restrict the VQ-VAE training stream to one atom modality.
        # "both" = joint tokenizer; "protein"/"ligand" = single-modality baseline
        # trained on the SAME complexes' atoms. Single-modality runs get their own
        # normalization stats file so they never clobber the joint stats.
        self.modality = modality
        self._keys: tuple[str, ...] = (
            ("protein",)
            if modality == "protein"
            else ("ligand",)
            if modality == "ligand"
            else ("protein", "ligand")
        )

        self.norm_stats: dict[str, Tensor] | None = None
        self._train_plan: list[tuple[int, list[int]]] | None = None
        self._val_plan: list[tuple[int, list[int]]] | None = None
        self._test_plan: list[tuple[int, list[int]]] | None = None
        self._shard_dir: Path | None = None

    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        if (self.cache_dir / _SHARD_METADATA_FILE).exists():
            logger.info("Atom descriptor cache already exists at %s", self.cache_dir)
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        shard_dir = self.cache_dir / _SHARD_DIR_NAME
        shard_dir.mkdir(parents=True, exist_ok=True)

        if self.hub_config is None:
            msg = "AtomComplexDescriptorDataModule requires a hub_config (manifest)"
            raise ValueError(msg)

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
        logger.info("Processing %d complex poses (all-atom)", len(abs_entries))

        total_count, shard_counts, unique_elements = _atom_process_entries_sharded(
            abs_entries,
            self.training_config.pocket,
            shard_dir=shard_dir,
            num_workers=self.training_config.num_workers,
        )
        if total_count == 0:
            msg = "No atom descriptors computed -- check data paths"
            raise RuntimeError(msg)
        _save_atom_shard_metadata(
            self.cache_dir, total_count, shard_counts, unique_elements
        )

    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        if not (self.cache_dir / _SHARD_METADATA_FILE).exists():
            msg = f"No atom shard cache at {self.cache_dir}; run prepare_data() first."
            raise FileNotFoundError(msg)

        metadata: dict = torch.load(
            self.cache_dir / _SHARD_METADATA_FILE, weights_only=False
        )
        if int(metadata.get("schema_version", 0)) < _ATOM_SHARD_SCHEMA_VERSION:
            msg = f"Atom cache at {self.cache_dir} is stale; delete + regenerate."
            raise RuntimeError(msg)
        shard_counts: list[int] = metadata["shard_counts"]
        shard_dir = self.cache_dir / _SHARD_DIR_NAME

        if self.hub_config is None:
            msg = "AtomComplexDescriptorDataModule requires a hub_config"
            raise ValueError(msg)
        train_set, val_set, test_set = _atom_fold_split_from_manifest(
            self.hub_config,
            self.data_config.random_state,
            self.data_config.val_size,
            shard_dir,
            shard_counts,
        )

        self._compute_or_load_norm_stats(shard_dir, shard_counts, train_set)
        self._build_plans(shard_counts, train_set, val_set, test_set)
        self._shard_dir = shard_dir
        logger.info(
            "Atom stream: %d train, %d val, %d test complexes",
            len(train_set),
            len(val_set),
            len(test_set),
        )

    def _compute_or_load_norm_stats(
        self,
        shard_dir: Path,
        shard_counts: list[int],
        train_set: set[int],
    ) -> None:
        stats_name = (
            _NORMALIZATION_STATS_FILE
            if self.modality == "both"
            else _NORMALIZATION_STATS_FILE.replace(".pt", f"_{self.modality}.pt")
        )
        stats_path = self.cache_dir / stats_name
        if stats_path.exists():
            self.norm_stats = torch.load(stats_path, weights_only=False)
            logger.info("Loaded cached atom normalization stats from %s", stats_path)
            return

        count = 0
        mean = np.zeros(ATOM_DESCRIPTOR_DIM, dtype=np.float64)
        m2 = np.zeros(ATOM_DESCRIPTOR_DIM, dtype=np.float64)
        for global_offset, shard_data in _iter_shards(shard_dir, shard_counts):
            for local_idx, cplx in enumerate(shard_data):
                if (global_offset + local_idx) not in train_set:
                    continue
                for key in self._keys:
                    arr = cplx[key]
                    if arr.shape[0] == 0:  # type: ignore[union-attr]
                        continue
                    count, mean, m2 = _welford_update_batch(
                        count,
                        mean,
                        m2,
                        arr.astype(np.float64),  # type: ignore[union-attr]
                    )

        atom_mean = mean.astype(np.float32)
        atom_std = (np.sqrt(m2 / max(count, 1)) + 1e-8).astype(np.float32)
        atom_mean, atom_std = _force_passthrough_for_categorical(
            atom_mean, atom_std, continuous_mask(ATOM_LAYOUT)
        )
        self.norm_stats = {
            "atom_mean": torch.from_numpy(atom_mean),
            "atom_std": torch.from_numpy(atom_std),
        }
        torch.save(self.norm_stats, stats_path)
        logger.info("Atom Welford normalization stats computed from train split")

    def _build_plans(
        self,
        shard_counts: list[int],
        train_set: set[int],
        val_set: set[int],
        test_set: set[int],
    ) -> None:
        from collections import defaultdict  # noqa: PLC0415

        train_by_shard: dict[int, list[int]] = defaultdict(list)
        val_by_shard: dict[int, list[int]] = defaultdict(list)
        test_by_shard: dict[int, list[int]] = defaultdict(list)

        global_offset = 0
        for shard_idx, count in enumerate(shard_counts):
            for local_idx in range(count):
                gi = global_offset + local_idx
                if gi in train_set:
                    train_by_shard[shard_idx].append(local_idx)
                elif gi in val_set:
                    val_by_shard[shard_idx].append(local_idx)
                elif gi in test_set:
                    test_by_shard[shard_idx].append(local_idx)
            global_offset += count

        self._train_plan = sorted(train_by_shard.items())
        self._val_plan = sorted(val_by_shard.items())
        self._test_plan = sorted(test_by_shard.items())

    # ------------------------------------------------------------------
    def _loader(
        self,
        plan: list[tuple[int, list[int]]] | None,
        *,
        shuffle: bool,
    ) -> DataLoader:
        if plan is None:
            msg = "setup() must be called before creating dataloaders"
            raise RuntimeError(msg)
        assert self.norm_stats is not None  # noqa: S101
        dataset = AtomShardedDataset(
            shard_dir=self._shard_dir,  # type: ignore[arg-type]
            shard_plan=plan,
            mean=self.norm_stats["atom_mean"].numpy(),
            std=self.norm_stats["atom_std"].numpy(),
            shuffle=shuffle,
            keys=self._keys,
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

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train_plan, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val_plan, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test_plan, shuffle=False)


__all__ = [
    "AtomComplexDescriptorDataModule",
    "AtomShardedDataset",
]
