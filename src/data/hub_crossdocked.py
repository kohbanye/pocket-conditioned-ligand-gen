"""DataModule for loading CrossDocked2020 from HuggingFace Hub.

Downloads receptor archives and ligand tar shards, extracts them to a local
cache, and provides pair access via a Parquet manifest.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Never

import lightning as L
import pyarrow.parquet as pq

if TYPE_CHECKING:
    import pandas as pd

    from src.config import HubDatasetConfig

logger = logging.getLogger(__name__)

# Sentinel file written after all shards have been extracted.
_EXTRACTION_DONE = ".extraction_done"


class HubCrossDockedDataModule(L.LightningDataModule):
    """DataModule that loads CrossDocked2020 data from HuggingFace Hub.

    Workflow:
    1. ``prepare_data()`` downloads from HF Hub and extracts archives.
    2. ``get_pairs()`` reads the manifest and returns (receptor, ligand)
       path pairs that the existing descriptor pipeline can consume.
    """

    def __init__(self, config: HubDatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.receptors_dir = self.cache_dir / "receptors"
        self.ligands_dir = self.cache_dir / "ligands"
        self._manifest: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Download & extraction
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Download dataset from HuggingFace Hub and extract archives."""
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        # Download the full repository (or update if revision changed)
        local_repo = Path(
            snapshot_download(
                repo_id=self.config.repo_id,
                repo_type="dataset",
                revision=self.config.revision,
                local_dir=self.cache_dir / "repo",
            )
        )

        self._extract_receptors(local_repo)
        self._extract_ligands(local_repo)

    def _extract_receptors(self, repo_dir: Path) -> None:
        """Extract receptor PDB archives to cache."""
        done_marker = self.receptors_dir / _EXTRACTION_DONE
        if done_marker.exists():
            return

        self.receptors_dir.mkdir(parents=True, exist_ok=True)

        receptor_archives = sorted((repo_dir / "receptors").glob("shard-*.tar.gz"))
        logger.info("Extracting %d receptor archive(s)", len(receptor_archives))

        for archive in receptor_archives:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(self.receptors_dir, filter="data")

        done_marker.touch()
        logger.info("Receptors extracted to %s", self.receptors_dir)

    def _extract_ligands(self, repo_dir: Path) -> None:
        """Extract ligand tar shards to cache."""
        done_marker = self.ligands_dir / _EXTRACTION_DONE
        if done_marker.exists():
            return

        self.ligands_dir.mkdir(parents=True, exist_ok=True)

        ligand_shards = sorted((repo_dir / "ligands").glob("*.tar"))
        logger.info("Extracting %d ligand shard(s)", len(ligand_shards))

        for shard in ligand_shards:
            with tarfile.open(shard, "r") as tar:
                tar.extractall(self.ligands_dir, filter="data")

        done_marker.touch()
        logger.info("Ligands extracted to %s", self.ligands_dir)

    # ------------------------------------------------------------------
    # Manifest access
    # ------------------------------------------------------------------

    def _load_manifest(self, repo_dir: Path | None = None) -> pd.DataFrame:
        """Load and cache the manifest Parquet file."""
        if self._manifest is not None:
            return self._manifest

        # Try repo_dir first, then cache_dir/repo
        candidates = []
        if repo_dir is not None:
            candidates.append(repo_dir / "manifest.parquet")
        candidates.append(self.cache_dir / "repo" / "manifest.parquet")

        for path in candidates:
            if path.exists():
                self._manifest = pq.read_table(path).to_pandas()
                logger.info("Loaded manifest: %d pairs", len(self._manifest))
                return self._manifest

        msg = "manifest.parquet not found in any expected location"
        raise FileNotFoundError(msg)

    def get_pairs(
        self,
        fold: int | None = None,
        split: str | None = None,
        source_types: list[str] | None = None,
        max_pairs: int | None = None,
    ) -> list[tuple[str, str]]:
        """Return (receptor_pdb_relpath, ligand_sdf_relpath) pairs.

        Paths are relative to ``receptors_dir`` and ``ligands_dir`` respectively.
        These can be converted to absolute paths via ``resolve_receptor_path``
        and ``resolve_ligand_path``.

        Args:
            fold: Fold number (0, 1, 2).  When set together with *split*,
                filters rows whose ``{source_type}_fold{fold}`` column
                matches *split*.
            split: "train" or "test".
            source_types: Filter by source type (e.g. ``["cdonly"]``).
                Defaults to the config value.
            max_pairs: Maximum number of pairs to return.
        """
        df = self._load_manifest()

        # Filter by source_type
        types_filter = source_types or self.config.source_types
        if types_filter:
            df = df[df["source_type"].isin(types_filter)]

        # Filter by fold/split
        if fold is not None and split is not None:
            # Check fold columns for the requested source types
            mask = df.index < 0  # start with all-False
            for st in types_filter or df["source_type"].unique():
                col = f"{st}_fold{fold}"
                if col in df.columns:
                    mask = mask | (df[col] == split)
            df = df[mask]

        if max_pairs is not None:
            df = df.head(max_pairs)

        pairs: list[tuple[str, str]] = []
        for _, row in df.iterrows():
            rec_rel = f"{row['complex_dir']}/{row['receptor_pdb']}"
            lig_rel = f"{row['pair_idx']:07d}.sdf.gz"
            pairs.append((rec_rel, lig_rel))

        return pairs

    def resolve_receptor_path(self, rec_rel: str) -> Path:
        """Convert a relative receptor path to an absolute path."""
        return self.receptors_dir / rec_rel

    def resolve_ligand_path(self, lig_rel: str) -> Path:
        """Convert a relative ligand path to an absolute path."""
        return self.ligands_dir / lig_rel

    # ------------------------------------------------------------------
    # LightningDataModule interface (delegated to descriptor pipeline)
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Not used directly — the descriptor DataModule handles splitting."""

    def train_dataloader(self) -> Never:  # type: ignore[override]
        raise NotImplementedError

    def val_dataloader(self) -> Never:  # type: ignore[override]
        raise NotImplementedError

    def test_dataloader(self) -> Never:  # type: ignore[override]
        raise NotImplementedError
