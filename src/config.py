from dataclasses import dataclass, field
from pathlib import Path

from src.tokenizers.descriptor_schema import (
    LIGAND_DESCRIPTOR_DIM,
    PROTEIN_DESCRIPTOR_DIM,
)


@dataclass
class HubDatasetConfig:
    """Config for loading CrossDocked2020 from HuggingFace Hub."""

    repo_id: str = "sakano/crossdocked2020"
    cache_dir: Path = Path("data/hub_cache")
    # Selects the manifest column ``{source_type}_fold{fold}`` used to assign
    # the official CrossDocked2020 train/test split when ``_setup_from_shards``
    # builds the descriptor split.
    fold: int = 0
    source_types: list[str] = field(default_factory=lambda: ["cdonly"])
    revision: str | None = None


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


# ---------------------------------------------------------------------------
# Multi-head VQ-VAE recon weights
# ---------------------------------------------------------------------------
#
# Continuous coord MSE is in Å² and dominates early training; categorical CEs
# are unitless and ~log(vocab) at init. The defaults below were chosen so all
# heads contribute on a similar order of magnitude after a few epochs (CE for
# 12-class element ≈ log(12) ≈ 2.5, MSE ≈ a few Å² → roughly comparable).
# Tune via the training-script CLI flag.


def _default_ligand_recon_weights() -> dict[str, float]:
    return {
        "coord": 1.0,
        "element": 0.5,
        "charge": 0.1,
        "hybrid": 0.1,
        "aromatic": 0.1,
        "ring": 0.1,
        "numH": 0.1,
    }


def _default_protein_recon_weights() -> dict[str, float]:
    return {
        "coord": 1.0,
        "aa": 0.5,
    }


@dataclass
class ProteinVQVAEConfig:
    """Config for protein backbone structure VQ-VAE."""

    descriptor_dim: int = PROTEIN_DESCRIPTOR_DIM
    hidden_dim: int = 256
    latent_dim: int = 16
    codebook_size: int = 4096
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    transformer_feedforward_dim: int = 512
    transformer_dropout: float = 0.1
    max_seq_len: int = 256
    domain: str = "protein"
    categorical_embed_dim: int = 8
    recon_weights: dict[str, float] = field(
        default_factory=_default_protein_recon_weights,
    )


@dataclass
class LigandVQVAEConfig:
    """Config for ligand structure VQ-VAE (spherical + features)."""

    descriptor_dim: int = LIGAND_DESCRIPTOR_DIM
    hidden_dim: int = 256
    latent_dim: int = 8
    codebook_size: int = 2048
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    transformer_feedforward_dim: int = 512
    transformer_dropout: float = 0.1
    max_seq_len: int = 256
    domain: str = "ligand"
    categorical_embed_dim: int = 8
    recon_weights: dict[str, float] = field(
        default_factory=_default_ligand_recon_weights,
    )


@dataclass
class VQVAETrainingConfig:
    """Config for joint VQ-VAE training."""

    learning_rate: float = 3e-4
    mol_batch_size: int = 4096
    max_epochs: int = 100
    num_workers: int = 16
    precision: str = "bf16-mixed"
    protein: ProteinVQVAEConfig = field(default_factory=ProteinVQVAEConfig)
    ligand: LigandVQVAEConfig = field(default_factory=LigandVQVAEConfig)
    pocket: PocketExtractionConfig = field(default_factory=PocketExtractionConfig)
