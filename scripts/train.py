"""Training script for pocket-conditioned ligand generation."""

import logging

from src.config import CrossDockedConfig
from src.data.crossdocked import CrossDockedDataModule

logging.basicConfig(level=logging.INFO)


def main() -> None:
    config = CrossDockedConfig()
    dm = CrossDockedDataModule(config)
    dm.prepare_data()


if __name__ == "__main__":
    main()
