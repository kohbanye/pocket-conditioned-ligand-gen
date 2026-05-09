"""Tokenize all CrossDocked2020 complexes into text format.

Applies trained VQ-VAEs to encode 3D structures into discrete tokens,
then assembles the ``<p>...<s>...<l>...`` text format.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from src.config import CrossDockedConfig, PocketExtractionConfig, VQVAETrainingConfig
from src.data.descriptors import _parse_types_file
from src.model.vqvae_module import VQVAEModule
from src.tokenizers.ligand import LigandDescriptor, parse_sdf
from src.tokenizers.protein import (
    BackboneSphericalDescriptor,
    _compute_canonical_frame,
    extract_full_sequence,
    extract_pocket,
)
from src.tokenizers.sequence import TokenSequenceAssembler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TokenizerContext:
    """Holds all objects needed for tokenizing complexes."""

    module: VQVAEModule
    protein_desc: BackboneSphericalDescriptor
    ligand_desc: LigandDescriptor
    assembler: TokenSequenceAssembler
    pocket_config: PocketExtractionConfig
    protein_mean: Tensor
    protein_std: Tensor
    ligand_mean: Tensor
    ligand_std: Tensor
    device: torch.device


def _load_context(config: VQVAETrainingConfig, data_dir: Path) -> TokenizerContext:
    """Load VQ-VAE checkpoint and normalization stats."""
    checkpoint_dir = Path("checkpoints")
    ckpt_files = sorted(checkpoint_dir.glob("vqvae-*.ckpt"))
    if not ckpt_files:
        msg = f"No VQ-VAE checkpoint found in {checkpoint_dir}"
        raise FileNotFoundError(msg)

    logger.info("Loading VQ-VAE from %s", ckpt_files[-1])
    module = VQVAEModule.load_from_checkpoint(str(ckpt_files[-1]), config=config)
    module.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)

    stats = torch.load(
        data_dir / "descriptor_cache" / "normalization_stats.pt",
        weights_only=True,
    )

    return TokenizerContext(
        module=module,
        protein_desc=BackboneSphericalDescriptor(),
        ligand_desc=LigandDescriptor(),
        assembler=TokenSequenceAssembler(),
        pocket_config=PocketExtractionConfig(),
        protein_mean=stats["protein_mean"].to(device),
        protein_std=stats["protein_std"].to(device),
        ligand_mean=stats["ligand_mean"].to(device),
        ligand_std=stats["ligand_std"].to(device),
        device=device,
    )


@torch.no_grad()
def _tokenize_complex(
    rec_path: Path,
    lig_path: Path,
    ctx: TokenizerContext,
) -> str | None:
    """Tokenize a single protein-ligand complex.

    With the spherical multi-feature VQ-VAE each atom / residue is one
    codebook integer; element and AA identity are recovered by the decoder.
    The AR sequence still keeps a separate ``<s>...</s>`` block for the AA
    sequence so language-model retrieval can attend to it directly.
    """
    import numpy as np  # noqa: PLC0415

    molecules = parse_sdf(lig_path)
    if not molecules:
        return None
    mol = molecules[0]
    heavy = [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"]
    if not heavy:
        return None
    lig_coords = np.array(heavy, dtype=np.float32)

    pocket = extract_pocket(rec_path, lig_coords, ctx.pocket_config)
    if pocket is None:
        return None
    backbone_coords, pocket_seq, residue_ids = pocket

    full_seq = extract_full_sequence(rec_path)

    ca_coords = backbone_coords[:, 1].astype(np.float64)
    centroid, rotation = _compute_canonical_frame(ca_coords)
    pocket_frame = (centroid, rotation)

    # Protein structure tokens — single integer per residue.
    prot_desc, _prot_meta = ctx.protein_desc.compute(
        backbone_coords,
        residue_ids,
        pocket_frame=pocket_frame,
        residue_names_one_letter=list(pocket_seq),
    )
    prot_t = torch.from_numpy(prot_desc).to(ctx.device)
    prot_t = (prot_t - ctx.protein_mean) / ctx.protein_std
    prot_indices = ctx.module.protein_vqvae.encode(prot_t).cpu().tolist()
    pocket_tokens = [str(code) for code in prot_indices]

    # Ligand tokens — single integer per heavy atom.
    lig_desc, _elements_sym, _lig_meta = ctx.ligand_desc.compute(
        mol["atoms"],
        mol["bonds"],
        pocket_frame=pocket_frame,
    )
    if len(lig_desc) == 0:
        return None

    lig_t = torch.from_numpy(lig_desc).to(ctx.device)
    lig_t = (lig_t - ctx.ligand_mean) / ctx.ligand_std
    lig_indices = ctx.module.ligand_vqvae.encode(lig_t).cpu().tolist()
    ligand_tokens = [str(code) for code in lig_indices]

    return ctx.assembler.assemble(pocket_tokens, full_seq, ligand_tokens)


def main() -> None:
    config = VQVAETrainingConfig()
    data_config = CrossDockedConfig()
    data_dir = Path(data_config.data_dir)

    ctx = _load_context(config, data_dir)

    types_dir = data_dir / "types"
    types_files = sorted(types_dir.glob("*train0.types"))
    if not types_files:
        msg = f"No .types files found in {types_dir}"
        raise FileNotFoundError(msg)

    entries = _parse_types_file(types_files[0])
    crossdocked_dir = data_dir / "CrossDocked2020"
    output_path = data_dir / "tokenized" / "sequences.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Tokenizing %d poses...", len(entries))
    num_ok = 0
    num_skip = 0

    with output_path.open("w") as f:
        for rec_path, lig_path, _pose_idx in entries:
            rec_full = crossdocked_dir / rec_path
            lig_full = crossdocked_dir / lig_path

            if not rec_full.exists() or not lig_full.exists():
                num_skip += 1
                continue

            try:
                result = _tokenize_complex(rec_full, lig_full, ctx)
            except Exception:
                logger.exception("Error: %s / %s", rec_path, lig_path)
                num_skip += 1
                continue

            if result is not None:
                f.write(result + "\n")
                num_ok += 1
            else:
                num_skip += 1

            if (num_ok + num_skip) % 1000 == 0:
                logger.info("Progress: %d ok, %d skipped", num_ok, num_skip)

    logger.info("Done: %d written to %s (%d skipped)", num_ok, output_path, num_skip)


if __name__ == "__main__":
    main()
