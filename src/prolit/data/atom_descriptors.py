"""DataModule for the ProLIT all-atom VQ-VAE (one codebook over all atoms).

Builds the descriptor rows themselves:

- the protein pocket is expanded to **every heavy atom** of the pocket residues;
  protein and ligand atoms share the 33-D :data:`ATOM_LAYOUT` descriptor,
- an optional ``label == 1`` (good-pose) filter is applied at manifest load,
- a **single** training stream: each complex contributes its protein-atom
  sequence AND its ligand-atom sequence as separate items, normalized by one
  pooled mean/std vector, so one VQ-VAE / one codebook tokenizes both.

The shard cache lives in ``data/descriptor_cache_allatom``. Sharding / Welford /
manifest / fold helpers come from :mod:`prolit.data.descriptors`.
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

from prolit.data.descriptors import (
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
from prolit.data.holdout import evaluation_pdbs, pdb_id_from_receptor
from prolit.tokenizers.atom import (
    LigandAtomDescriptor,
    ProteinAtomDescriptor,
    precompute_receptor_atom_features,
)
from prolit.tokenizers.descriptor_schema import (
    ATOM_DESCRIPTOR_DIM,
    ATOM_LAYOUT,
    continuous_mask,
    fields_by_name,
)
from prolit.tokenizers.ligand import parse_sdf
from prolit.tokenizers.protein import (
    compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    precompute_pocket_atom_candidates,
)

if TYPE_CHECKING:
    from prolit.config import (
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

    from prolit.config import PocketExtractionConfig  # noqa: PLC0415

    _w_prot_desc = ProteinAtomDescriptor()
    _w_pocket_config = PocketExtractionConfig(**pocket_config_dict)
    _w_lig_desc = LigandAtomDescriptor(_w_pocket_config.atom_order)


def _atom_process_pose(  # noqa: PLR0913
    mol: dict,
    precomputed_atoms: object,
    receptor_feats: dict,
    pocket_config: object,
    prot_desc: ProteinAtomDescriptor,
    lig_desc: LigandAtomDescriptor,
    ligand_frame: str = "pocket",
) -> dict[str, np.ndarray | list[str] | int] | None:
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

    centroid, rotation = compute_canonical_frame(pocket.ca_coords.astype(np.float64))
    frame = (centroid, rotation)

    if ligand_frame == "local":
        lig_frame = compute_canonical_frame(lig_coords.astype(np.float64))
    elif ligand_frame == "pocket":
        lig_frame = frame
    else:
        msg = f"unknown ligand_frame {ligand_frame!r} (expected 'pocket' or 'local')"
        raise ValueError(msg)

    prot_arr, _prot_meta = prot_desc.compute(pocket, receptor_feats, frame)
    pkt_canonical = pkt_elements = None
    # Read off pocket_config, not a separate argument: the tar-streaming
    # workers receive only asdict(pocket_config), so a standalone keyword
    # silently stayed False and the cache came out identical to the old one.
    if getattr(pocket_config, "pocket_context", False):
        # Pocket atoms expressed in the LIGAND's frame, so the two agree even in
        # the `local` ablation where the ligand has a frame of its own.
        lig_centroid, lig_rotation = lig_frame
        pkt_canonical = (
            pocket.atom_coords.astype(np.float64) - lig_centroid
        ) @ lig_rotation.T
        element_field = fields_by_name(ATOM_LAYOUT)["element"]
        pkt_elements = prot_arr[:, element_field.start].astype(np.int64)
    lig_arr, elements, _lig_meta = lig_desc.compute(
        mol["atoms"], mol["bonds"], lig_frame, pkt_canonical, pkt_elements
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


def _fold_labels(
    df: object,
    hub_config: HubDatasetConfig,
    fold: int,
    available: set[str],
) -> dict[int, str]:
    """Map ``pair_idx`` to ``train`` / ``test`` / ``excluded``.

    ``excluded`` is a third state the CrossDocked folds do not have: complexes
    an evaluation set downstream of the tokenizer will be scored on, which the
    fold knows nothing about. 169 of the 285 CASF-2016 core-set entries sit on
    the fold-0 *train* side, so without this the tokenizer sees them.
    """
    excluded_pdbs: set[str] = set()
    if getattr(hub_config, "exclude_eval_pdbs", False):
        excluded_pdbs = evaluation_pdbs(
            cd_manifest=None,  # the fold split already covers it
            casf_list=Path(hub_config.casf_pdb_list),
            include_sbdd=True,
        )

    fold_map: dict[int, str] = {}
    n_excluded = 0
    for row in df.itertuples(index=False):  # type: ignore[attr-defined]
        col = f"{row.source_type}_fold{fold}"
        label = getattr(row, col, None) if col in available else "train"
        if excluded_pdbs:
            pid = pdb_id_from_receptor(str(row.receptor_pdb))
            if pid is not None and pid in excluded_pdbs:
                label = "excluded"
                n_excluded += 1
        if label is not None:
            fold_map[int(row.pair_idx)] = label
    if excluded_pdbs:
        logger.info(
            "Held out %d manifest rows on %d evaluation PDB ids (CASF + sbdd)",
            n_excluded,
            len(excluded_pdbs),
        )
    return fold_map


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
        columns=["pair_idx", "source_type", "receptor_pdb", *fold_cols],
    ).to_pandas()
    df = df[df["source_type"].isin(source_types)]

    fold_map = _fold_labels(df, hub_config, fold, available)

    test_globals: list[int] = []
    trainval_globals: list[int] = []
    missing = 0
    held_out = 0
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
            elif label == "excluded":
                held_out += 1
            else:
                missing += 1
        global_offset += count
    if held_out:
        logger.info("%d cached complexes held out as evaluation PDBs", held_out)
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


def _dist_rank_world() -> tuple[int, int]:
    """This process's rank and the world size; ``(0, 1)`` outside DDP."""
    dist = torch.distributed
    if not dist.is_available():  # ty: ignore[possibly-missing-attribute]
        return 0, 1
    if not dist.is_initialized():  # ty: ignore[possibly-missing-attribute]
        return 0, 1
    return dist.get_rank(), dist.get_world_size()  # ty: ignore[possibly-missing-attribute]


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
        shard_by_rank: bool = True,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.shard_dir = shard_dir
        self.shard_plan = shard_plan
        self.mean = mean
        self.std = std
        self.shuffle = shuffle
        # Captured in the parent process: a dataloader worker is forked and does
        # not inherit the process group, so ``is_initialized()`` is False by the
        # time __iter__ runs.
        rank, world = _dist_rank_world()
        # Splitting the stream across ranks is a TRAINING optimisation: it is
        # what stops N GPUs doing the same work N times. Validation must not do
        # it. Every run that split the val stream logged no ``val/atom_coord``
        # at all -- b_ddp4fix, b_shard2, b_shard4, b_range4 -- so
        # ``ModelCheckpoint(monitor="val/atom_coord")`` never fired and the run
        # finished with last.ckpt and no best checkpoint. The runs that left the
        # val stream whole (b_ddp2, b_ddp4, b_ddp4b) validated every epoch. The
        # cost of validating the whole set on every rank is 4x redundant work on
        # a tenth of the data; the cost of the split was silently losing model
        # selection, which is not a trade.
        self.shard_by_rank = shard_by_rank
        self.rank, self.world_size = (rank, world) if shard_by_rank else (0, 1)
        # Which atom streams to emit per entry. ("protein","ligand") = joint;
        # a single-element tuple trains a single-modality (protein/ligand-only)
        # VQ-VAE on the SAME complexes (the ablation baseline tokenizers).
        self.keys = keys
        # Per-RANK length, and it has to be EXACT, not an estimate.
        #
        # Lightning ends a training epoch -- and only then runs validation -- on
        # ``batch_idx + 1 == num_training_batches``, and it takes that count
        # from ``len(dataloader)``, i.e. from here. Each worker forms and drops
        # its own ragged tail, so the batches actually yielded are
        # ``sum_w floor(items_w / B)``, which is at or below
        # ``floor(sum_w items_w / B)``. Whenever it is strictly below, the
        # equality never holds, the epoch ends by StopIteration instead, and
        # **validation never runs at all**: no ``val/atom_coord`` is logged, so
        # ``ModelCheckpoint`` never fires and the run finishes with last.ckpt
        # and no best checkpoint. One GPU happened to make the two agree
        # (1565 = 1565) which is why this survived; four made it 391 against a
        # real 382 and silently cost model selection on every DDP run.
        #
        # Hence batch_size and num_workers: the true count depends on both, so
        # a length computed without them cannot be right.
        per_worker = self._worker_item_counts(batch_size, num_workers)
        self.length = batch_size * sum(items // batch_size for items in per_worker)

    def _worker_item_counts(self, batch_size: int, num_workers: int) -> list[int]:
        """Items each dataloader worker will emit, in worker order."""
        plan = [
            (shard_idx, indices[self.rank :: self.world_size])
            for shard_idx, indices in self.shard_plan
        ]
        if batch_size < 1:
            msg = "batch_size must be >= 1"
            raise ValueError(msg)
        workers = max(1, num_workers)
        return [
            len(self.keys)
            * sum(
                len(indices)
                for i, (_, indices) in enumerate(plan)
                if i % workers == w
            )
            for w in range(workers)
        ]

    def __len__(self) -> int:
        return self.length

    def __iter__(self):  # noqa: ANN204
        import random as _random  # noqa: PLC0415

        # Split across DDP ranks by interleaving the entries INSIDE each shard,
        # then split this rank's shards across its dataloader workers. An
        # IterableDataset gets no DistributedSampler -- there is no sampler for
        # Lightning to replace -- so without a rank split every rank reads every
        # shard and N GPUs do the same work N times (measured: 4 GPUs took 24.9
        # min against one GPU's 23.1 for the same 1565 steps).
        #
        # Every rank keeps all 35 shards, and that redundancy is not an
        # oversight -- it is what makes the batch counts match. With
        # ``num_workers`` > 0 each worker forms and drops its OWN ragged tail,
        # so a rank emits sum_w floor(entries_w / B), not floor(entries / B).
        # Ranks holding different numbers of shards therefore emit different
        # numbers of batches, and DDP's epoch-end collective waits forever.
        #
        # Measured, four ranks, 16 workers:
        #   entries interleaved in-shard -> 678 / 678 / 678 / 678 batches, 35
        #       shards each. Works: 9.1 min for 4 epochs, 2.54x one GPU.
        #   contiguous entry ranges      -> 680 / 680 / 682 / 683 batches, 12 /
        #       10 / 9 / 7 shards. NCCL ALLREDUCE timed out after 30 min. It
        #       would have cut per-rank I/O from 14.7 GB to ~4 GB -- shard
        #       loading runs at 175 MB/s and is the whole 68 s fixed term in
        #       T(N) = 279/N + 68 s/epoch -- but making the counts agree needs
        #       a global truncation across every (rank, worker) pair.
        #   whole shards round-robin     -> 77k-96k entries per rank. Deadlock.
        plan = self.shard_plan
        if self.world_size > 1:
            plan = [
                (shard_idx, indices[self.rank :: self.world_size])
                for shard_idx, indices in plan
            ]

        worker_info = torch.utils.data.get_worker_info()
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
        shard_by_rank: bool,
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
            shard_by_rank=shard_by_rank,
            batch_size=self.training_config.mol_batch_size,
            num_workers=self.training_config.num_workers,
        )
        nw = self.training_config.num_workers
        return DataLoader(
            dataset,
            batch_size=self.training_config.mol_batch_size,
            num_workers=nw,
            persistent_workers=nw > 0,
            pin_memory=True,
            # Ranks differ by up to 25 entries out of 351k after the in-shard
            # split; dropping the ragged tail makes every rank emit exactly the
            # same number of batches, which is what DDP's epoch-end collective
            # requires. Unsharded loaders see identical data on every rank, so
            # their counts already agree and the tail is only lost work.
            drop_last=shard_by_rank,
            # torch types collate_fn as Callable[[list[_T]], Any] with _T
            # bound by nothing, so no function satisfies it.
            collate_fn=collate_molecules,  # ty: ignore[invalid-argument-type]
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train_plan, shuffle=True, shard_by_rank=True)

    def val_dataloader(self) -> DataLoader:
        # Whole set on every rank -- see AtomShardedDataset.__init__ for why
        # splitting it cost model selection outright.
        return self._loader(self._val_plan, shuffle=False, shard_by_rank=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test_plan, shuffle=False, shard_by_rank=False)


__all__ = [
    "AtomComplexDescriptorDataModule",
    "AtomShardedDataset",
]
