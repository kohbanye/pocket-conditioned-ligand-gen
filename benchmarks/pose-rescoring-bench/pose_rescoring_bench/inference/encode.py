"""Shared tokenizer/encoder loading + complex encoding (ported from the source repo).

Model loading and the small conversions this benchmark needs on top of it. The
fixed-pocket encoder itself is :class:`prolit.api.PoseEncoder` -- it used to be
copied here, which meant two implementations of the recipe that produces the
docking-power table.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from prolit.chem.mol2 import mol_to_dict, parse_mol2_multi  # noqa: F401
from prolit.config import (
    AtomVQVAETrainingConfig,
    MLMTrainingConfig,
    PocketExtractionConfig,
    ProLITMLMConfig,
    RescoreTrainingConfig,
)
from prolit.data.rescore_dataset import ligand_mask
from prolit.model.mlm_module import ProLITMLMModule
from prolit.model.rescore_module import ComplexRescoreModule
from prolit.tokenizers.lm_vocab import AtomLMVocab
from prolit.tokenizers.loaders import load_atom_vqvae
from prolit.tokenizers.pose_encoder import PoseEncoder

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from prolit.model.vqvae_module import AtomVQVAEModule

    from pose_rescoring_bench.config import PathsConfig
    from pose_rescoring_bench.variants import AffinityCkpts, RescoringCkpts


def load_vqvae(
    ckpt: Path,
    norm_stats: Path,
    codebook_size: int,
    device: torch.device,
) -> tuple[AtomVQVAEModule, np.ndarray, np.ndarray]:
    """Load the all-atom VQ-VAE tokenizer + its normalization stats (eval mode)."""
    cfg = AtomVQVAETrainingConfig()
    cfg.atom.codebook_size = codebook_size
    module = load_atom_vqvae(ckpt, device)
    module.eval().to(device)
    norm = torch.load(norm_stats, weights_only=False)
    module.vqvae.set_normalization(norm["atom_mean"], norm["atom_std"])
    return module, norm["atom_mean"].numpy(), norm["atom_std"].numpy()


def load_separate_vqvae(  # noqa: PLR0913
    protein_ckpt: Path,
    protein_norm: Path,
    ligand_ckpt: Path,
    ligand_norm: Path,
    codebook_size: int,
    device: torch.device,
) -> tuple[object, np.ndarray, np.ndarray]:
    """Load protein-only + ligand-only VQ-VAEs as one combined-code-space encoder.

    ``codebook_size`` is the PER-MODALITY sub-codebook size (e.g. 8192); the
    combined vocab (2x that) is passed separately to :func:`make_encoder`. Returns
    ``(sep, identity_mean, identity_std)`` with identity RAW-descriptor stats
    (``np.zeros(33)`` / ``np.ones(33)``): :class:`SeparateVQVAE` normalizes each
    modality internally, so :class:`PoseEncoder` must feed it RAW descriptors.
    """
    from prolit.tokenizers.separate_vqvae import SeparateVQVAE  # noqa: PLC0415

    sep = SeparateVQVAE.from_checkpoints(
        protein_ckpt,
        protein_norm,
        ligand_ckpt,
        ligand_norm,
        device,
        codebook_size=codebook_size,
    )
    return sep, np.zeros(33, dtype=np.float32), np.ones(33, dtype=np.float32)


def load_tokenizer(
    ckpts: RescoringCkpts | AffinityCkpts,
    paths: PathsConfig,
    device: torch.device,
) -> tuple[object, np.ndarray, np.ndarray]:
    """Load a variant's tokenizer (joint single VQ or separate protein+ligand VQs).

    Dispatches on ``ckpts.is_separate``: the separate arm loads two single-modality
    VQ-VAEs into one combined code space (feeding identity RAW-descriptor stats);
    the joint arm loads the single combined VQ with the shared normalization stats.
    Returns ``(module, mean, std)`` ready for :func:`make_encoder`.
    """
    if ckpts.is_separate:
        pv, lv = ckpts.protein_vqvae, ckpts.ligand_vqvae
        pn, ln = ckpts.protein_norm, ckpts.ligand_norm
        if pv is None or lv is None or pn is None or ln is None:
            msg = "separate variant is missing protein/ligand vqvae or norm stats"
            raise ValueError(msg)
        # ``codebook_size`` is the COMBINED vocab (2x per-modality); each
        # single-modality sub-VQ uses half of it (16384 -> 8192, 8192 -> 4096).
        return load_separate_vqvae(
            paths.ckpt(pv),
            paths.ckpt(pn),
            paths.ckpt(lv),
            paths.ckpt(ln),
            ckpts.codebook_size // 2,
            device,
        )
    vqvae_ckpt = ckpts.vqvae
    if vqvae_ckpt is None:
        msg = "variant is missing its vqvae checkpoint"
        raise ValueError(msg)
    return load_vqvae(
        paths.ckpt(vqvae_ckpt),
        paths.norm_stats,
        ckpts.codebook_size,
        device,
    )


def load_mlm(ckpt: Path, codebook_size: int, device: torch.device) -> tuple[Any, int]:
    """Load the complex-token MLM backbone; return (model, mask_token_id)."""
    cfg = MLMTrainingConfig(model=ProLITMLMConfig(atom_codebook_size=codebook_size))
    mlm = ProLITMLMModule.load_from_checkpoint(
        ckpt,
        config=cfg,
        map_location=device,
    ).model
    mlm.eval().to(device)
    return mlm, cfg.model.mask_token_id


def load_rescorer(
    ckpt: Path,
    codebook_size: int,
    device: torch.device,
) -> ComplexRescoreModule:
    """Load a rescoring/affinity head.

    The checkpoint's own config is the base rather than a fresh one built from
    the caller's arguments: the stored instance carries the training settings
    the module reads at construction time, and rebuilding it from scratch has
    already produced a head whose submodules did not match its state_dict.
    """
    stored = torch.load(ckpt, map_location="cpu", weights_only=False).get(
        "hyper_parameters", {}
    ).get("config")
    cfg = stored if isinstance(stored, RescoreTrainingConfig) else None
    if cfg is None:
        cfg = RescoreTrainingConfig(
            model=ProLITMLMConfig(atom_codebook_size=codebook_size)
        )
    else:
        cfg.model.atom_codebook_size = codebook_size
    rescorer = ComplexRescoreModule.load_from_checkpoint(
        ckpt,
        config=cfg,
        map_location=device,
    )
    rescorer.eval().to(device)
    return rescorer


_RESCORE_VL_RE = re.compile(r"rescore-e\d+-vl([0-9.]+)\.ckpt$")


def _rescore_val_loss(path: Path) -> float:
    """Parse the val-loss encoded in a ``rescore-eNN-vlX.XXXX.ckpt`` filename."""
    m = _RESCORE_VL_RE.search(path.name)
    return float(m.group(1)) if m is not None else float("inf")


def resolve_rescore_ckpt(source_repo: Path, spec: str) -> Path:
    """Resolve a head checkpoint: an exact ``*.ckpt`` path or a run-name to glob.

    If ``spec`` ends in ``.ckpt`` it is treated as a path relative to the source
    repo. Otherwise it is a rescore run-name whose ``checkpoints`` directory is
    globbed for ``rescore-*.ckpt``, returning the one with the LOWEST val-loss.
    """
    if spec.endswith(".ckpt"):
        return source_repo / spec
    ckpt_dir = source_repo / "pocket-ligand-rescore" / spec / "checkpoints"
    candidates = sorted(ckpt_dir.glob("rescore-*.ckpt"))
    if not candidates:
        msg = f"no rescore checkpoints found for head {spec!r} under {ckpt_dir}"
        raise FileNotFoundError(msg)
    return min(candidates, key=_rescore_val_loss)


def sequence_ligand_mask(seq: Sequence[int]) -> np.ndarray:
    """0/1 mask marking ligand-token positions in an assembled sequence."""
    return ligand_mask(np.asarray(seq))


def make_encoder(  # noqa: PLR0913
    module: AtomVQVAEModule,
    mean: np.ndarray,
    std: np.ndarray,
    codebook_size: int,
    device: torch.device,
    max_residues: int,
) -> PoseEncoder:
    """Construct a :class:`~prolit.api.PoseEncoder` with the standard vocab + config."""
    return PoseEncoder(
        module.vqvae,
        mean,
        std,
        AtomLMVocab(codebook_size=codebook_size),
        device,
        PocketExtractionConfig(max_residues=max_residues),
    )
