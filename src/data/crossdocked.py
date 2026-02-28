import logging
import subprocess
import tarfile
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from src.config import CrossDockedConfig

logger = logging.getLogger(__name__)


class CrossDockedDataModule(L.LightningDataModule):
    """DataModule for CrossDocked2020 dataset.

    Downloads and extracts the CrossDocked2020 dataset from
    http://bits.csb.pitt.edu/files/crossdock2020/

    The dataset contains protein-ligand cross-docking complexes.
    """

    def __init__(self, config: CrossDockedConfig) -> None:
        super().__init__()
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.crossdocked_dir = self.data_dir / "CrossDocked2020"

    def prepare_data(self) -> None:
        """Download and extract CrossDocked2020 dataset."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Download and extract types tarball
        types_tarball = self.data_dir / self.config.types_tarball
        self._download_and_extract_types(types_tarball)

        # Download and extract main data tarball
        data_tarball = self.data_dir / self.config.data_tarball
        self._download_and_extract_data(data_tarball)

    def _download_file(self, url: str, dest: Path) -> None:
        """Download a file using wget."""
        if dest.exists():
            logger.info("File already exists: %s", dest)
            return

        logger.info("Downloading %s to %s", url, dest)
        subprocess.run(  # noqa: S603
            ["wget", "-c", "-O", str(dest), url],  # noqa: S607
            check=True,
        )

    def _download_and_extract_types(self, tarball_path: Path) -> None:
        """Download and extract types tarball to data_dir."""
        # Check if already extracted (types files are in data_dir directly)
        types_files = list(self.data_dir.glob("*.types"))
        if types_files:
            logger.info(
                "Types files already extracted: %d files found",
                len(types_files),
            )
            return

        url = f"{self.config.base_url}/{self.config.types_tarball}"
        self._download_file(url, tarball_path)

        logger.info("Extracting %s to %s", tarball_path, self.data_dir)
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=self.data_dir, filter="data")

    def _download_and_extract_data(self, tarball_path: Path) -> None:
        """Download and extract main data tarball to CrossDocked2020/."""
        # Check if already extracted
        if self.crossdocked_dir.exists() and any(self.crossdocked_dir.iterdir()):
            logger.info("Data already extracted to %s", self.crossdocked_dir)
            return

        url = f"{self.config.base_url}/{self.config.data_tarball}"
        self._download_file(url, tarball_path)

        self.crossdocked_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Extracting %s to %s (this may take a while...)",
            tarball_path,
            self.crossdocked_dir,
        )
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=self.crossdocked_dir, filter="data")

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Set up train/val/test datasets."""

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        raise NotImplementedError
