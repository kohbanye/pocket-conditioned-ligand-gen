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

from pathlib import Path
from typing import Any

import torch


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

    Delegates to :func:`load_atom_vqvae` rather than building a config, because
    this function used to build one and could therefore not open any checkpoint
    trained with the constrained balancer -- which is all of them. That it is
    the *documented public entry point* while the internal loader had the fix
    is the wrong way round; see that function for what goes wrong.
    """
    module = load_atom_vqvae(ckpt, device, codebook_size=codebook_size)
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
        Path(protein_ckpt),
        Path(protein_norm),
        Path(ligand_ckpt),
        Path(ligand_norm),
        device,
        codebook_size=codebook_size,
    )


def load_causal_lm(
    ckpt: str | Path,
    codebook_size: int,
    device: torch.device,
) -> Any:  # noqa: ANN401
    """Load ProLIT-CLM, the generative decoder over interface tokens."""
    from prolit.config import CLMTrainingConfig  # noqa: PLC0415
    from prolit.model.clm_module import ProLITCLMModule  # noqa: PLC0415

    config = CLMTrainingConfig()
    config.model.atom_codebook_size = codebook_size
    return (
        ProLITCLMModule.load_from_checkpoint(
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
    from prolit.model.mlm_module import ProLITMLMModule  # noqa: PLC0415

    config = MLMTrainingConfig()
    config.model.atom_codebook_size = codebook_size
    module = (
        ProLITMLMModule.load_from_checkpoint(
            str(ckpt), config=config, map_location=device
        )
        .eval()
        .to(device)
    )
    return module.model, config.model.mask_token_id


def load_atom_vqvae(
    ckpt: str | Path,
    device: torch.device,
    *,
    codebook_size: int | None = None,
) -> Any:  # noqa: ANN401
    """Load an all-atom VQ-VAE from a checkpoint, using the checkpoint's config.

    Passing a freshly built config instead is the mistake this exists to stop.
    Lightning pickles the config INSTANCE into ``hyper_parameters``, and the
    module registers buffers off it -- ``loss_balancing="constrained"`` adds one
    ``_lam_`` per chemistry head -- so a default config builds a module whose
    state_dict is missing exactly those keys and the load dies on "Unexpected
    key(s)". Every corpus builder had its own copy of that bug, which meant no
    checkpoint trained with a balancer could be tokenized with at all.

    ``codebook_size`` is checked, not applied: it is what the caller believes it
    asked for, and a mismatch means the wrong run was named. Silently honouring
    it would build a vocabulary the weights do not match.
    """
    from prolit.model.vqvae_module import AtomVQVAEModule  # noqa: PLC0415

    module = AtomVQVAEModule.load_from_checkpoint(str(ckpt), map_location=device)
    found = module.config.atom.codebook_size
    if codebook_size is not None and found != codebook_size:
        msg = f"{ckpt} was trained with codebook_size {found}, not {codebook_size}"
        raise ValueError(msg)
    return module.eval().to(device)


def load_pose_refiner(ckpt: str | Path, device: torch.device) -> Any:  # noqa: ANN401
    """Load the E(3)-equivariant flow-matching pose refiner."""
    from prolit.model.pose_refiner import PoseRefinerModule  # noqa: PLC0415

    return (
        PoseRefinerModule.load_from_checkpoint(str(ckpt), map_location=device)
        .eval()
        .to(device)
    )
