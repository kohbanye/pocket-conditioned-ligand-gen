import logging
import shutil
import subprocess
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

    def _extract_tarball(self, tarball_path: Path, dest: Path) -> None:
        """Extract a .tgz tarball using pigz for parallel decompression."""
        if shutil.which("pigz"):
            logger.info("Extracting %s with pigz (parallel)", tarball_path)
            pigz = subprocess.Popen(  # noqa: S603
                ["pigz", "-dc", str(tarball_path)],  # noqa: S607
                stdout=subprocess.PIPE,
            )
            subprocess.run(  # noqa: S603
                ["tar", "xf", "-", "-C", str(dest)],  # noqa: S607
                stdin=pigz.stdout,
                check=True,
            )
            pigz.wait()
            if pigz.returncode != 0:
                msg = f"pigz failed with return code {pigz.returncode}"
                raise RuntimeError(msg)
        else:
            logger.info("Extracting %s with tar (single-threaded)", tarball_path)
            subprocess.run(  # noqa: S603
                ["tar", "xzf", str(tarball_path), "-C", str(dest)],  # noqa: S607
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

        self._extract_tarball(tarball_path, self.data_dir)

    def _download_and_extract_data(self, tarball_path: Path) -> None:
        """Download and extract main data tarball to CrossDocked2020/."""
        # Use a marker file to distinguish complete vs partial extraction
        done_marker = self.crossdocked_dir / ".extraction_done"
        if done_marker.exists():
            logger.info("Data already extracted to %s", self.crossdocked_dir)
            return

        # Clean up partial extraction from a previous interrupted run
        if self.crossdocked_dir.exists():
            logger.info("Removing partial extraction: %s", self.crossdocked_dir)
            shutil.rmtree(self.crossdocked_dir)

        url = f"{self.config.base_url}/{self.config.data_tarball}"
        self._download_file(url, tarball_path)

        self.crossdocked_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Extracting %s to %s (this may take a while...)",
            tarball_path,
            self.crossdocked_dir,
        )
        self._extract_tarball(tarball_path, self.crossdocked_dir)
        done_marker.touch()

        self._cleanup_extracted(tarball_path)

    def _cleanup_extracted(self, tarball_path: Path | None = None) -> None:
        """Remove unused .gninatypes files and the tarball after extraction."""
        cleanup_marker = self.crossdocked_dir / ".cleanup_done"
        if cleanup_marker.exists():
            return

        num_deleted = 0
        bytes_freed = 0
        for gninatypes_file in self.crossdocked_dir.rglob("*.gninatypes"):
            bytes_freed += gninatypes_file.stat().st_size
            gninatypes_file.unlink()
            num_deleted += 1

        logger.info(
            "Deleted %d .gninatypes files (%.1f GB freed)",
            num_deleted,
            bytes_freed / 1e9,
        )

        if tarball_path is not None:
            tarball_path.unlink(missing_ok=True)
            logger.info("Deleted tarball %s", tarball_path)

        cleanup_marker.touch()

    def setup(self, stage: str | None = None) -> None:
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
