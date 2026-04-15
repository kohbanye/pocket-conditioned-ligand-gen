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
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.tokenizers.ligand import LigandDescriptor, parse_sdf
from src.tokenizers.protein import PocketDescriptor, extract_pocket

if TYPE_CHECKING:
    from src.config import (
        CrossDockedConfig,
        HubDatasetConfig,
        PocketExtractionConfig,
        VQVAETrainingConfig,
    )

logger = logging.getLogger(__name__)

# Per-worker state for multiprocessing (set by _worker_init).
_worker_protein_desc: PocketDescriptor | None = None
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


def _parse_types_file(types_path: Path) -> list[tuple[str, str]]:
    """Parse a .types file to extract (receptor_pdb, ligand_sdf) pairs.

    Each line has format:
        label score1 score2 receptor.gninatypes ligand.gninatypes #comment

    Converts .gninatypes paths to .pdb / .sdf.gz paths and deduplicates.
    """
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for line in types_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:  # noqa: PLR2004
            continue
        rec_pdb = _gninatypes_to_pdb(parts[3])
        lig_sdf = _gninatypes_to_sdf(parts[4])
        key = (rec_pdb, lig_sdf)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def _parse_all_types_files(
    types_dir: Path,
    max_pairs: int | None = None,
) -> list[tuple[str, str]]:
    """Parse train0 and test0 types files and return deduplicated union.

    When *max_pairs* is set, stops reading once enough unique pairs have
    been collected (avoids parsing multi-GB files unnecessarily).
    """
    train_files = sorted(types_dir.glob("cdonly_*train0.types"))
    test_files = sorted(types_dir.glob("cdonly_*test0.types"))
    all_files = train_files + test_files
    if not all_files:
        msg = f"No cdonly_*train0/test0.types files found in {types_dir}"
        raise FileNotFoundError(msg)

    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for types_file in all_files:
        for pair in _parse_types_file(types_file):
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                if max_pairs is not None and len(pairs) >= max_pairs:
                    break
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    logger.info("Loaded %d unique pairs from %s", len(pairs), types_dir)
    return pairs


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


def _get_ligand_coords_from_sdf(sdf_path: Path) -> np.ndarray | None:
    """Extract heavy-atom coordinates from an SDF file for pocket extraction."""
    molecules = parse_sdf(sdf_path)
    if not molecules:
        return None
    mol = molecules[0]
    heavy_atoms = [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"]
    if not heavy_atoms:
        return None
    return np.array(heavy_atoms, dtype=np.float32)


def _process_complex(
    rec_path: Path,
    lig_path: Path,
    protein_desc: PocketDescriptor,
    ligand_desc: LigandDescriptor,
    pocket_config: PocketExtractionConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Process one complex and return (protein_desc, ligand_desc, elements)."""
    lig_coords = _get_ligand_coords_from_sdf(lig_path)
    if lig_coords is None:
        return None

    pocket = extract_pocket(rec_path, lig_coords, pocket_config)
    if pocket is None:
        return None
    backbone_coords, _pocket_seq = pocket

    prot_desc, prot_metadata = protein_desc.compute(backbone_coords)
    pocket_frame = (prot_metadata["centroid"], prot_metadata["rotation"])

    molecules = parse_sdf(lig_path)
    if not molecules:
        return None
    mol = molecules[0]
    lig_desc_arr, elements, _lig_metadata = ligand_desc.compute(
        mol["atoms"],
        mol["bonds"],
        pocket_frame=pocket_frame,
    )
    if len(lig_desc_arr) == 0:
        return None

    return prot_desc, lig_desc_arr, elements


def _worker_init(pocket_config_dict: dict) -> None:
    """Initialize per-worker descriptor calculators (called once per process)."""
    global _worker_protein_desc, _worker_ligand_desc, _worker_pocket_config  # noqa: PLW0603

    from src.config import PocketExtractionConfig  # noqa: PLC0415

    _worker_protein_desc = PocketDescriptor()
    _worker_ligand_desc = LigandDescriptor()
    _worker_pocket_config = PocketExtractionConfig(**pocket_config_dict)


def _worker_process_one(
    args: tuple[str, str, str],
) -> dict[str, np.ndarray | list[str]] | None:
    """Process a single complex in a worker process.

    *args* is ``(rec_path, lig_path, base_dir)``.  When *base_dir* is
    empty, the paths are treated as absolute.
    """
    rec_str, lig_str, base_str = args
    if base_str:
        rec_full = Path(base_str) / rec_str
        lig_full = Path(base_str) / lig_str
    else:
        rec_full = Path(rec_str)
        lig_full = Path(lig_str)

    if not rec_full.exists() or not lig_full.exists():
        return None

    try:
        result = _process_complex(
            rec_full,
            lig_full,
            _worker_protein_desc,  # type: ignore[arg-type]
            _worker_ligand_desc,  # type: ignore[arg-type]
            _worker_pocket_config,  # type: ignore[arg-type]
        )
    except Exception:
        logger.exception("Error processing %s / %s", rec_str, lig_str)
        return None

    if result is None:
        return None

    prot, lig, elems = result
    return {"protein": prot, "ligand": lig, "elements": elems}


def _process_pairs(
    pairs: list[tuple[str, str]],
    crossdocked_dir: Path,
    pocket_config: PocketExtractionConfig,
    num_workers: int = 0,
) -> list[dict[str, np.ndarray | list[str]]]:
    """Process a list of (receptor, ligand) pairs into complex dicts.

    When *num_workers* > 0, processing runs in parallel using a
    ``multiprocessing.Pool``.

    Returns a list of dicts, each containing raw (un-normalized) descriptors:
      - ``"protein"``: ``ndarray`` of shape ``(n_residues, 9)``
      - ``"ligand"``: ``ndarray`` of shape ``(n_atoms, 4)``
      - ``"elements"``: ``list[str]`` of per-atom element symbols
    """
    from dataclasses import asdict  # noqa: PLC0415

    crossdocked_str = str(crossdocked_dir)
    work_items = [(rec, lig, crossdocked_str) for rec, lig in pairs]

    pocket_config_dict = asdict(pocket_config)

    if num_workers > 0:
        import multiprocessing  # noqa: PLC0415

        logger.info("Processing with %d workers", num_workers)
        with multiprocessing.Pool(
            num_workers,
            initializer=_worker_init,
            initargs=(pocket_config_dict,),
        ) as pool:
            results = pool.imap_unordered(
                _worker_process_one,
                work_items,
                chunksize=64,
            )
            return _collect_results(results, len(work_items))

    # Single-process fallback
    _worker_init(pocket_config_dict)
    results_iter = (_worker_process_one(item) for item in work_items)
    return _collect_results(results_iter, len(work_items))


def _process_pairs_from_abs(
    abs_pairs: list[tuple[str, str]],
    pocket_config: PocketExtractionConfig,
    num_workers: int = 0,
) -> list[dict[str, np.ndarray | list[str]]]:
    """Like ``_process_pairs`` but accepts absolute paths directly."""
    from dataclasses import asdict  # noqa: PLC0415

    work_items = [(rec, lig, "") for rec, lig in abs_pairs]
    pocket_config_dict = asdict(pocket_config)

    if num_workers > 0:
        import multiprocessing  # noqa: PLC0415

        logger.info("Processing with %d workers", num_workers)
        with multiprocessing.Pool(
            num_workers,
            initializer=_worker_init,
            initargs=(pocket_config_dict,),
        ) as pool:
            results = pool.imap_unordered(
                _worker_process_one,
                work_items,
                chunksize=64,
            )
            return _collect_results(results, len(work_items))

    _worker_init(pocket_config_dict)
    results_iter = (_worker_process_one(item) for item in work_items)
    return _collect_results(results_iter, len(work_items))


def _collect_results(
    results: Iterable[dict[str, np.ndarray | list[str]] | None],
    total: int,
) -> list[dict[str, np.ndarray | list[str]]]:
    """Collect results from an iterator, logging progress."""
    complexes: list[dict[str, np.ndarray | list[str]]] = []
    num_skipped = 0

    for num_done, result in enumerate(results, 1):
        if result is None:
            num_skipped += 1
        else:
            complexes.append(result)

        if num_done % 1000 == 0:
            logger.info(
                "Progress: %d / %d done (%d ok, %d skipped)",
                num_done,
                total,
                len(complexes),
                num_skipped,
            )

    logger.info("Done: %d processed, %d skipped", len(complexes), num_skipped)
    return complexes


def _save_complexes(
    cache_dir: Path,
    complexes: list[dict[str, np.ndarray | list[str]]],
) -> None:
    """Save per-complex descriptors and element vocabulary to cache."""
    torch.save(complexes, cache_dir / "complexes.pt")

    unique_elements = sorted(
        {e for c in complexes for e in c["elements"]}  # type: ignore[union-attr]
    )
    torch.save(unique_elements, cache_dir / "ligand_elements.pt")

    logger.info("Cached %d complexes", len(complexes))


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


def _split_and_normalize(  # noqa: PLR0913
    complexes: list[dict[str, np.ndarray | list[str]]],
    indices: list[int],
    protein_mean: np.ndarray,
    protein_std: np.ndarray,
    ligand_mean: np.ndarray,
    ligand_std: np.ndarray,
) -> tuple[Tensor, list[Tensor]]:
    """Extract, normalize, and concatenate descriptors for a split.

    Returns ``(protein_tensor, ligand_molecule_list)``.
    """
    protein_parts = [complexes[i]["protein"] for i in indices]
    ligand_parts = [complexes[i]["ligand"] for i in indices]

    protein_cat = np.concatenate(protein_parts, axis=0)  # type: ignore[arg-type]
    protein_norm = (protein_cat - protein_mean) / protein_std
    protein_tensor = torch.from_numpy(protein_norm).float()

    ligand_molecules = [
        torch.from_numpy(
            (lig - ligand_mean) / ligand_std  # type: ignore[operator]
        ).float()
        for lig in ligand_parts
    ]

    return protein_tensor, ligand_molecules


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

        self.protein_train: torch.Tensor | None = None
        self.protein_val: torch.Tensor | None = None
        self.protein_test: torch.Tensor | None = None
        self.ligand_train: list[Tensor] | None = None
        self.ligand_val: list[Tensor] | None = None
        self.ligand_test: list[Tensor] | None = None
        self.norm_stats: dict[str, Tensor] | None = None

    def prepare_data(self) -> None:
        """Compute descriptors from CrossDocked2020 and cache to disk."""
        if (self.cache_dir / "complexes.pt").exists():
            logger.info("Descriptor cache already exists at %s", self.cache_dir)
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.hub_config is not None:
            pairs, cache_dir = _load_pairs_from_manifest(
                self.hub_config,
                max_pairs=self.data_config.max_pairs,
            )
            receptor_dir = cache_dir / "receptors"
            ligand_dir = cache_dir / "ligands"
            # Build absolute paths for _process_pairs: receptor from
            # receptors/, ligand from ligands/
            abs_pairs = [
                (str(receptor_dir / rec), str(ligand_dir / lig)) for rec, lig in pairs
            ]
        else:
            types_dir = self.data_dir / "types"
            pairs = _parse_all_types_files(
                types_dir,
                max_pairs=self.data_config.max_pairs,
            )
            crossdocked_dir = self.data_dir / "CrossDocked2020"
            abs_pairs = [
                (str(crossdocked_dir / rec), str(crossdocked_dir / lig))
                for rec, lig in pairs
            ]

        logger.info("Processing %d complex pairs", len(abs_pairs))

        # _process_pairs expects (rec, lig) strings and a base dir.
        # Since abs_pairs already has absolute paths, pass Path("/") as
        # the base dir so joining is a no-op.
        complexes = _process_pairs_from_abs(
            abs_pairs,
            self.training_config.pocket,
            num_workers=self.training_config.num_workers,
        )

        if not complexes:
            msg = "No descriptors computed -- check data paths"
            raise RuntimeError(msg)

        _save_complexes(self.cache_dir, complexes)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Load cached descriptors and split into train/val/test at complex level."""
        complexes: list[dict[str, np.ndarray | list[str]]] = torch.load(
            self.cache_dir / "complexes.pt",
            weights_only=False,
        )

        # --- Complex-level split ---
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

        # --- Compute normalization from train split only ---
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

        # --- Normalize and build per-split tensors ---
        self.protein_train, self.ligand_train = _split_and_normalize(
            complexes, train_indices, protein_mean, protein_std, ligand_mean, ligand_std
        )
        self.protein_val, self.ligand_val = _split_and_normalize(
            complexes, val_indices, protein_mean, protein_std, ligand_mean, ligand_std
        )
        self.protein_test, self.ligand_test = _split_and_normalize(
            complexes, test_indices, protein_mean, protein_std, ligand_mean, ligand_std
        )

        logger.info(
            "Protein residues: %d train, %d val, %d test",
            len(self.protein_train),
            len(self.protein_val),
            len(self.protein_test),
        )
        logger.info(
            "Ligand molecules: %d train, %d val, %d test",
            len(self.ligand_train),
            len(self.ligand_val),
            len(self.ligand_test),
        )

    def _build_protein_loader(
        self,
        data: torch.Tensor,
        *,
        shuffle: bool,
    ) -> DataLoader:
        bs = self.training_config.batch_size
        nw = self.training_config.num_workers
        return DataLoader(
            TensorDataset(data),
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            persistent_workers=nw > 0,
        )

    def _build_ligand_loader(
        self,
        molecules: list[Tensor],
        *,
        shuffle: bool,
    ) -> DataLoader:
        bs = self.training_config.ligand_mol_batch_size
        nw = self.training_config.num_workers
        return DataLoader(
            MoleculeDataset(molecules),
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            persistent_workers=nw > 0,
            collate_fn=collate_molecules,
        )

    def train_dataloader(self) -> CombinedLoader:
        """Return train dataloaders for protein and ligand."""
        if self.protein_train is None or self.ligand_train is None:
            msg = "setup() must be called before train_dataloader()"
            raise RuntimeError(msg)
        return CombinedLoader(
            {
                "protein": self._build_protein_loader(
                    self.protein_train,
                    shuffle=True,
                ),
                "ligand": self._build_ligand_loader(
                    self.ligand_train,
                    shuffle=True,
                ),
            },
            mode="max_size_cycle",
        )

    def val_dataloader(self) -> CombinedLoader:
        """Return validation dataloaders for protein and ligand."""
        if self.protein_val is None or self.ligand_val is None:
            msg = "setup() must be called before val_dataloader()"
            raise RuntimeError(msg)
        return CombinedLoader(
            {
                "protein": self._build_protein_loader(
                    self.protein_val,
                    shuffle=False,
                ),
                "ligand": self._build_ligand_loader(
                    self.ligand_val,
                    shuffle=False,
                ),
            },
            mode="max_size_cycle",
        )

    def test_dataloader(self) -> CombinedLoader:
        """Return test dataloaders for protein and ligand."""
        if self.protein_test is None or self.ligand_test is None:
            msg = "setup() must be called before test_dataloader()"
            raise RuntimeError(msg)
        return CombinedLoader(
            {
                "protein": self._build_protein_loader(
                    self.protein_test,
                    shuffle=False,
                ),
                "ligand": self._build_ligand_loader(
                    self.ligand_test,
                    shuffle=False,
                ),
            },
            mode="max_size_cycle",
        )
