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
from src.data.descriptors import _get_ligand_coords_from_sdf, _parse_types_file
from src.model.vqvae_module import VQVAEModule
from src.tokenizers.ligand import SE3InvariantDescriptor, parse_sdf
from src.tokenizers.protein import (
    ProteinBackboneDescriptor,
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
    protein_desc: ProteinBackboneDescriptor
    ligand_desc: SE3InvariantDescriptor
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
        protein_desc=ProteinBackboneDescriptor(config.protein.num_neighbors),
        ligand_desc=SE3InvariantDescriptor(config.ligand.max_neighbors),
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
    """Tokenize a single protein-ligand complex."""
    lig_coords = _get_ligand_coords_from_sdf(lig_path)
    if lig_coords is None:
        return None

    pocket = extract_pocket(rec_path, lig_coords, ctx.pocket_config)
    if pocket is None:
        return None
    backbone_coords, pocket_seq = pocket

    full_seq = extract_full_sequence(rec_path)

    # Protein structure tokens
    prot_desc = ctx.protein_desc.compute(backbone_coords)
    prot_t = torch.from_numpy(prot_desc).to(ctx.device)
    prot_t = (prot_t - ctx.protein_mean) / ctx.protein_std
    prot_indices = ctx.module.protein_vqvae.encode(prot_t).cpu().tolist()
    pocket_tokens = [
        f"{aa}_{code}" for aa, code in zip(pocket_seq, prot_indices, strict=True)
    ]

    # Ligand tokens
    molecules = parse_sdf(lig_path)
    if not molecules:
        return None
    mol = molecules[0]
    lig_desc, elements = ctx.ligand_desc.compute(mol["atoms"], mol["bonds"])
    if len(lig_desc) == 0:
        return None

    lig_t = torch.from_numpy(lig_desc).to(ctx.device)
    lig_t = (lig_t - ctx.ligand_mean) / ctx.ligand_std
    lig_indices = ctx.module.ligand_vqvae.encode(lig_t).cpu().tolist()
    ligand_tokens = [
        f"{elem}_{code}" for elem, code in zip(elements, lig_indices, strict=True)
    ]

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

    pairs = _parse_types_file(types_files[0])
    crossdocked_dir = data_dir / "CrossDocked2020"
    output_path = data_dir / "tokenized" / "sequences.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Tokenizing %d complexes...", len(pairs))
    num_ok = 0
    num_skip = 0

    with output_path.open("w") as f:
        for rec_path, lig_path in pairs:
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
