"""DataModule for computing and caching VQ-VAE training descriptors.

Processes CrossDocked2020 protein-ligand complexes into per-residue and
per-atom SE(3)-invariant descriptors for joint VQ-VAE training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, TensorDataset

from src.tokenizers.ligand import SE3InvariantDescriptor, parse_sdf
from src.tokenizers.protein import ProteinBackboneDescriptor, extract_pocket

if TYPE_CHECKING:
    from src.config import (
        CrossDockedConfig,
        PocketExtractionConfig,
        VQVAETrainingConfig,
    )

logger = logging.getLogger(__name__)


def _gninatypes_to_pdb(gninatypes_path: str) -> str:
    """Convert a receptor .gninatypes path to the corresponding .pdb path.

    Example: ``subdir/5f74_A_rec_0.gninatypes`` → ``subdir/5f74_A_rec.pdb``
    """
    import re  # noqa: PLC0415

    return re.sub(r"_\d+\.gninatypes$", ".pdb", gninatypes_path)


def _gninatypes_to_sdf(gninatypes_path: str) -> str:
    """Convert a ligand .gninatypes path to the corresponding .sdf.gz path.

    Example: ``subdir/5f74_A_rec_5f74_amp_lig_tt_docked_0.gninatypes``
           → ``subdir/5f74_A_rec_5f74_amp_lig_tt_docked.sdf.gz``
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
    protein_desc: ProteinBackboneDescriptor,
    ligand_desc: SE3InvariantDescriptor,
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

    prot_desc = protein_desc.compute(backbone_coords)

    molecules = parse_sdf(lig_path)
    if not molecules:
        return None
    mol = molecules[0]
    lig_desc_arr, elements = ligand_desc.compute(mol["atoms"], mol["bonds"])
    if len(lig_desc_arr) == 0:
        return None

    return prot_desc, lig_desc_arr, elements


def _save_descriptors(
    cache_dir: Path,
    all_protein_desc: list[np.ndarray],
    all_ligand_desc: list[np.ndarray],
    all_ligand_elements: list[list[str]],
) -> None:
    """Normalize and save descriptors to cache directory."""
    protein_all = np.concatenate(all_protein_desc, axis=0)
    ligand_all = np.concatenate(all_ligand_desc, axis=0)

    protein_mean = protein_all.mean(axis=0)
    protein_std = protein_all.std(axis=0) + 1e-8
    ligand_mean = ligand_all.mean(axis=0)
    ligand_std = ligand_all.std(axis=0) + 1e-8

    protein_all = (protein_all - protein_mean) / protein_std
    ligand_all = (ligand_all - ligand_mean) / ligand_std

    torch.save(
        torch.from_numpy(protein_all),
        cache_dir / "protein_descriptors.pt",
    )
    torch.save(
        torch.from_numpy(ligand_all),
        cache_dir / "ligand_descriptors.pt",
    )
    torch.save(
        {
            "protein_mean": torch.from_numpy(protein_mean),
            "protein_std": torch.from_numpy(protein_std),
            "ligand_mean": torch.from_numpy(ligand_mean),
            "ligand_std": torch.from_numpy(ligand_std),
        },
        cache_dir / "normalization_stats.pt",
    )

    unique_elements = sorted({e for elems in all_ligand_elements for e in elems})
    torch.save(unique_elements, cache_dir / "ligand_elements.pt")

    logger.info(
        "Cached %d protein residue descriptors, %d ligand atom descriptors",
        len(protein_all),
        len(ligand_all),
    )


class ComplexDescriptorDataModule(L.LightningDataModule):
    """DataModule that computes and caches descriptors for VQ-VAE training."""

    def __init__(
        self,
        training_config: VQVAETrainingConfig,
        data_config: CrossDockedConfig,
    ) -> None:
        super().__init__()
        self.training_config = training_config
        self.data_config = data_config
        self.data_dir = Path(data_config.data_dir)
        self.cache_dir = self.data_dir / "descriptor_cache"

        self.protein_desc = ProteinBackboneDescriptor(
            num_neighbors=training_config.protein.num_neighbors,
        )
        self.ligand_desc = SE3InvariantDescriptor(
            max_neighbors=training_config.ligand.max_neighbors,
        )

        self.protein_train: torch.Tensor | None = None
        self.protein_val: torch.Tensor | None = None
        self.ligand_train: torch.Tensor | None = None
        self.ligand_val: torch.Tensor | None = None

    def prepare_data(self) -> None:
        """Compute descriptors from CrossDocked2020 and cache to disk."""
        if (self.cache_dir / "protein_descriptors.pt").exists():
            logger.info("Descriptor cache already exists at %s", self.cache_dir)
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        types_dir = self.data_dir / "types"
        types_files = sorted(types_dir.glob("*train0.types"))
        if not types_files:
            msg = f"No .types files found in {types_dir}"
            raise FileNotFoundError(msg)

        pairs = _parse_types_file(types_files[0])
        if self.data_config.max_pairs is not None:
            pairs = pairs[: self.data_config.max_pairs]
        logger.info("Processing %d complex pairs", len(pairs))

        crossdocked_dir = self.data_dir / "CrossDocked2020"
        all_protein: list[np.ndarray] = []
        all_ligand: list[np.ndarray] = []
        all_elements: list[list[str]] = []
        num_processed = 0
        num_skipped = 0

        for rec_path, lig_path in pairs:
            rec_full = crossdocked_dir / rec_path
            lig_full = crossdocked_dir / lig_path

            if not rec_full.exists() or not lig_full.exists():
                num_skipped += 1
                continue

            try:
                result = _process_complex(
                    rec_full, lig_full,
                    self.protein_desc, self.ligand_desc,
                    self.training_config.pocket,
                )
            except Exception:
                logger.exception("Error processing %s / %s", rec_path, lig_path)
                num_skipped += 1
                continue

            if result is None:
                num_skipped += 1
                continue

            prot, lig, elems = result
            all_protein.append(prot)
            all_ligand.append(lig)
            all_elements.append(elems)
            num_processed += 1

            if num_processed % 1000 == 0:
                logger.info("Processed %d (%d skipped)", num_processed, num_skipped)

        logger.info("Done: %d processed, %d skipped", num_processed, num_skipped)

        if not all_protein or not all_ligand:
            msg = "No descriptors computed — check data paths"
            raise RuntimeError(msg)

        _save_descriptors(self.cache_dir, all_protein, all_ligand, all_elements)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Load cached descriptors and split into train/val."""
        protein_all = torch.load(
            self.cache_dir / "protein_descriptors.pt", weights_only=True,
        )
        ligand_all = torch.load(
            self.cache_dir / "ligand_descriptors.pt", weights_only=True,
        )

        val_frac = self.data_config.val_size
        rng = torch.Generator().manual_seed(self.data_config.random_state)

        n_prot = len(protein_all)
        prot_perm = torch.randperm(n_prot, generator=rng)
        n_prot_val = int(n_prot * val_frac)
        self.protein_val = protein_all[prot_perm[:n_prot_val]]
        self.protein_train = protein_all[prot_perm[n_prot_val:]]

        n_lig = len(ligand_all)
        lig_perm = torch.randperm(n_lig, generator=rng)
        n_lig_val = int(n_lig * val_frac)
        self.ligand_val = ligand_all[lig_perm[:n_lig_val]]
        self.ligand_train = ligand_all[lig_perm[n_lig_val:]]

    def _build_loader(
        self, data: torch.Tensor, *, shuffle: bool,
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

    def train_dataloader(self) -> CombinedLoader:
        """Return train dataloaders for protein and ligand."""
        if self.protein_train is None or self.ligand_train is None:
            msg = "setup() must be called before train_dataloader()"
            raise RuntimeError(msg)
        return CombinedLoader(
            {
                "protein": self._build_loader(self.protein_train, shuffle=True),
                "ligand": self._build_loader(self.ligand_train, shuffle=True),
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
                "protein": self._build_loader(self.protein_val, shuffle=False),
                "ligand": self._build_loader(self.ligand_val, shuffle=False),
            },
            mode="max_size_cycle",
        )
