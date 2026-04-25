from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HubDatasetConfig:
    """Config for loading CrossDocked2020 from HuggingFace Hub."""

    repo_id: str = "sakano/crossdocked2020"
    cache_dir: Path = Path("data/hub_cache")
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


@dataclass
class ProteinVQVAEConfig:
    """Config for protein backbone structure VQ-VAE."""

    descriptor_dim: int = 12  # 3 backbone atoms (N, CA, C) x 4D Z-matrix each
    hidden_dim: int = 256
    latent_dim: int = 16
    codebook_size: int = 2048
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    # Transformer context parameters
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    transformer_feedforward_dim: int = 512
    transformer_dropout: float = 0.1
    max_seq_len: int = 256
    # 3D coord-reconstruction loss
    coord_loss_enabled: bool = True
    coord_loss_kind: str = "protein_backbone"
    coord_loss_bond_length_min: float = 0.5
    # Unit-circle penalty on (sin tau, cos tau) slots of the denormalized
    # descriptor: lambda * sum((s^2 + c^2 - 1)^2). Keeps decoder outputs on
    # the unit circle -- otherwise the NeRF step sees drifting phases and
    # angle error accumulates through the backbone. Set 0 to disable.
    circle_loss_weight: float = 0.0


@dataclass
class LigandVQVAEConfig:
    """Config for ligand structure VQ-VAE (Z-matrix descriptors)."""

    descriptor_dim: int = 4  # bond_length, bond_angle, sin_dihedral, cos_dihedral
    hidden_dim: int = 256
    latent_dim: int = 8
    codebook_size: int = 1024
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    # Transformer context parameters
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    transformer_feedforward_dim: int = 512
    transformer_dropout: float = 0.1
    max_seq_len: int = 256
    # 3D coord-reconstruction loss
    coord_loss_enabled: bool = True
    coord_loss_kind: str = "ligand"
    coord_loss_bond_length_min: float = 0.5
    circle_loss_weight: float = 0.0


@dataclass
class VQVAETrainingConfig:
    """Config for joint VQ-VAE training."""

    learning_rate: float = 3e-4
    mol_batch_size: int = 4096
    max_epochs: int = 100
    num_workers: int = 16
    precision: str = "bf16-mixed"
    # Linear ramp of ``coord_loss`` from 0 → 1 over the first N epochs.  With
    # ``TaskWeighting`` the combined objective is unstable when ``coord`` is
    # orders of magnitude larger than ``recon`` at init, so we hold it at 0
    # until ``recon`` has had a chance to decrease.  During the ramp phase
    # (and when 0) we bypass ``TaskWeighting`` so ``log_var_coord`` does not
    # drift to -∞.  0 disables the warmup (ramp = 1 from epoch 0).
    coord_loss_warmup_epochs: int = 0
    # When True, force (sin, cos) slots to mean=0, std=1 in the descriptor
    # normalization stats so the unit-circle constraint ``s² + c² = 1`` is
    # preserved in the network's input/output space.  Requires the cached
    # ``normalization_stats.pt`` to be deleted so it is recomputed.
    skip_sincos_normalization: bool = False
    protein: ProteinVQVAEConfig = field(default_factory=ProteinVQVAEConfig)
    ligand: LigandVQVAEConfig = field(default_factory=LigandVQVAEConfig)
    pocket: PocketExtractionConfig = field(default_factory=PocketExtractionConfig)
