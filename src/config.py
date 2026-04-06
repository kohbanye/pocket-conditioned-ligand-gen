from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CrossDockedConfig:
    data_dir: Path = Path("data")
    base_url: str = "http://bits.csb.pitt.edu/files/crossdock2020"
    data_tarball: str = "CrossDocked2020_v1.3.tgz"
    types_tarball: str = "CrossDocked2020_v1.3_types.tgz"
    batch_size: int = 32
    num_workers: int = 4
    test_size: float = 0.1
    val_size: float = 0.1
    random_state: int = 42
    max_pairs: int | None = None


@dataclass
class PocketExtractionConfig:
    """Config for extracting pocket residues around a ligand."""

    distance_cutoff: float = 8.0
    max_residues: int = 128


@dataclass
class ProteinVQVAEConfig:
    """Config for protein backbone structure VQ-VAE."""

    num_neighbors: int = 16
    descriptor_dim: int = 20  # num_neighbors + 4
    hidden_dim: int = 128
    latent_dim: int = 8
    codebook_size: int = 512
    commitment_cost: float = 0.25
    ema_decay: float = 0.99


@dataclass
class LigandVQVAEConfig:
    """Config for ligand structure VQ-VAE (Mol-StrucTok style)."""

    descriptor_dim: int = 14  # 4 generation + 10 understanding
    hidden_dim: int = 128
    latent_dim: int = 5
    codebook_size: int = 256
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    max_neighbors: int = 4


@dataclass
class VQVAETrainingConfig:
    """Config for joint VQ-VAE training."""

    learning_rate: float = 3e-4
    batch_size: int = 65536
    max_epochs: int = 100
    num_workers: int = 16
    protein: ProteinVQVAEConfig = field(default_factory=ProteinVQVAEConfig)
    ligand: LigandVQVAEConfig = field(default_factory=LigandVQVAEConfig)
    pocket: PocketExtractionConfig = field(default_factory=PocketExtractionConfig)
