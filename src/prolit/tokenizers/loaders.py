"""Checkpoint loaders for every trained ProLIT component.

Each of these was previously open-coded in the eval scripts and again in the
benchmarks, which is how the two drifted: the same checkpoint would be loaded
with a differently-configured module depending on which caller you went
through. They live here so there is exactly one way to turn a path into a
model, and so a caller does not need to know which Lightning module wraps what.

Every loader returns an eval-mode module on the requested device. The VQ-VAE
loaders return the inner :class:`~prolit.tokenizers.vqvae.TransformerVQVAE`
rather than its Lightning wrapper, because inference only ever wants
``encode`` / ``encode_batch`` / ``decode_to_outputs``.

``codebook_size`` is the size of the code range the component was *trained*
against. For the separate-tokenizer arm that is the COMBINED size of the two
sub-codebooks (8192 for 4096+4096), because both arms present one contiguous
code space to everything downstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pathlib import Path


def load_norm_stats(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load the descriptor normalization statistics that accompany a VQ-VAE.

    A VQ-VAE decodes into the descriptor space these statistics define, so they
    must always travel with the checkpoint: the wrong file yields plausible but
    silently mis-scaled coordinates rather than an error.
    """
    stats = torch.load(path, weights_only=False)
    return {k: v.to(device) for k, v in stats.items()}


def load_tokenizer(
    ckpt: str | Path,
    codebook_size: int,
    device: torch.device,
    norm_stats: dict[str, torch.Tensor] | None = None,
) -> Any:  # noqa: ANN401
    """Load the joint all-atom VQ-VAE (the ProLIT tokenizer).

    Pass ``norm_stats`` to bake the normalization into the module, so callers
    can hand it raw descriptors.
    """
    from prolit.config import AtomVQVAETrainingConfig  # noqa: PLC0415
    from prolit.model.vqvae_module import AtomVQVAEModule  # noqa: PLC0415

    config = AtomVQVAETrainingConfig()
    config.atom.codebook_size = codebook_size
    module = (
        AtomVQVAEModule.load_from_checkpoint(
            str(ckpt), config=config, map_location=device
        )
        .eval()
        .to(device)
    )
    if norm_stats is not None:
        module.vqvae.set_normalization(norm_stats["atom_mean"], norm_stats["atom_std"])
    return module.vqvae


def load_separate_tokenizer(  # noqa: PLR0913
    protein_ckpt: str | Path,
    protein_norm: str | Path,
    ligand_ckpt: str | Path,
    ligand_norm: str | Path,
    device: torch.device,
    codebook_size: int,
) -> Any:  # noqa: ANN401
    """Load the separate-tokenizer ablation arm as one combined code space.

    ``codebook_size`` is the PER-MODALITY size here (4096 for the 4096+4096
    arm); the combined space is twice that, with protein codes in
    ``[0, codebook_size)`` and ligand codes above them. Each sub-VQ normalizes
    with its own modality statistics internally, so the caller feeds it RAW
    descriptors -- unlike :func:`load_tokenizer`.
    """
    from prolit.tokenizers.separate_vqvae import SeparateVQVAE  # noqa: PLC0415

    return SeparateVQVAE.from_checkpoints(
        protein_ckpt,
        protein_norm,
        ligand_ckpt,
        ligand_norm,
        device,
        codebook_size=codebook_size,
    )


def load_causal_lm(
    ckpt: str | Path,
    codebook_size: int,
    device: torch.device,
) -> Any:  # noqa: ANN401
    """Load ProLIT-CLM, the generative decoder over interface tokens."""
    from prolit.config import LMTrainingConfig  # noqa: PLC0415
    from prolit.model.lm_module import LigandLMModule  # noqa: PLC0415

    config = LMTrainingConfig()
    config.model.atom_codebook_size = codebook_size
    return (
        LigandLMModule.load_from_checkpoint(
            str(ckpt), config=config, map_location=device
        )
        .eval()
        .to(device)
        .model
    )


def load_masked_lm(
    ckpt: str | Path,
    codebook_size: int,
    device: torch.device,
) -> tuple[Any, int]:
    """Load ProLIT-MLM. Returns ``(model, mask_token_id)``.

    The ``<mask>`` id sits one past the real vocabulary so codebook offsets are
    unchanged and the token caches stay valid; callers masking tokens need it.
    """
    from prolit.config import MLMTrainingConfig  # noqa: PLC0415
    from prolit.model.mlm_module import ComplexMLMModule  # noqa: PLC0415

    config = MLMTrainingConfig()
    config.model.atom_codebook_size = codebook_size
    module = (
        ComplexMLMModule.load_from_checkpoint(
            str(ckpt), config=config, map_location=device
        )
        .eval()
        .to(device)
    )
    return module.model, config.model.mask_token_id


def load_scoring_head(
    ckpt: str | Path,
    codebook_size: int,
    device: torch.device,
    pooling: str = "mean",
) -> Any:  # noqa: ANN401
    """Load a pose-rescoring or affinity head (encoder + pooled MLP).

    ``pooling`` must match how the head was trained -- "mean", "meanmax" or
    "attn". It changes the head's input width, so a mismatch surfaces as a shape
    error at load time rather than as quietly wrong scores.
    """
    from prolit.config import RescoreTrainingConfig  # noqa: PLC0415
    from prolit.model.rescore_module import ComplexRescoreModule  # noqa: PLC0415

    config = RescoreTrainingConfig()
    config.model.atom_codebook_size = codebook_size
    config.pooling = pooling
    return (
        ComplexRescoreModule.load_from_checkpoint(
            str(ckpt), config=config, map_location=device
        )
        .eval()
        .to(device)
    )


def load_pose_refiner(ckpt: str | Path, device: torch.device) -> Any:  # noqa: ANN401
    """Load the E(3)-equivariant flow-matching pose refiner."""
    from prolit.model.pose_refiner import PoseRefinerModule  # noqa: PLC0415

    return (
        PoseRefinerModule.load_from_checkpoint(str(ckpt), map_location=device)
        .eval()
        .to(device)
    )
