"""Stage 2: convert cached descriptors into LM token streams via the VQ-VAE.

Loads the frozen "2x" VQ-VAE checkpoint, reuses
:class:`~src.data.descriptors.ComplexDescriptorDataModule` to obtain the
train/val/test split + normalization stats, then for every complex encodes the
protein-pocket and ligand descriptors to codebook indices and assembles a flat
LM sequence ``<bos><p>..</p><l>..</l><eos>`` (see
:mod:`src.tokenizers.lm_vocab`).

Each split is written as a packed token stream:

    {split}.bin   uint16   all sequences concatenated end to end
    {split}.len   uint16   per-sequence token counts (cumsum -> doc offsets)
    meta.json              vocab sizes, counts, total tokens

Run (existing 2.5M v4 cache, single GPU)::

    uv run python scripts/tokenize_dataset.py \
        --from-hub --source-types cdonly it0 it2_redocked \
        --cache-dir data/descriptor_cache_v4 \
        --ckpt "<path-to-2x-vqvae-checkpoint>.ckpt" \
        --out-dir data/lm_tokens
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.config import CrossDockedConfig, HubDatasetConfig, VQVAETrainingConfig
from src.data.descriptors import ComplexDescriptorDataModule, collate_molecules
from src.data.token_io import SplitWriter
from src.model.vqvae_module import VQVAEModule
from src.tokenizers.lm_vocab import LMVocab

if TYPE_CHECKING:
    import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize(desc: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((desc - mean) / std).float()


def _encode_buffer(
    module: VQVAEModule,
    vocab: LMVocab,
    prot_descs: list[torch.Tensor],
    lig_descs: list[torch.Tensor],
    device: torch.device,
) -> list[list[int]]:
    """Encode a buffer of complexes into LM token sequences."""
    prot_x, prot_mask = collate_molecules(prot_descs)
    lig_x, lig_mask = collate_molecules(lig_descs)
    prot_idx = module.protein_vqvae.encode_batch(
        prot_x.to(device), prot_mask.to(device)
    ).cpu()
    lig_idx = module.ligand_vqvae.encode_batch(
        lig_x.to(device), lig_mask.to(device)
    ).cpu()

    sequences: list[list[int]] = []
    for i in range(len(prot_descs)):
        p_codes = prot_idx[i][prot_mask[i]].tolist()
        l_codes = lig_idx[i][lig_mask[i]].tolist()
        sequences.append(vocab.build_sequence(p_codes, l_codes))
    return sequences


def _tokenize_split(  # noqa: PLR0913
    module: VQVAEModule,
    vocab: LMVocab,
    shard_dir: Path,
    plan: list[tuple[int, list[int]]],
    norm: dict[str, np.ndarray],
    writer: SplitWriter,
    batch_size: int,
    device: torch.device,
) -> None:
    from tqdm import tqdm  # noqa: PLC0415

    prot_buf: list[torch.Tensor] = []
    lig_buf: list[torch.Tensor] = []

    def flush() -> None:
        if not prot_buf:
            return
        seqs = _encode_buffer(module, vocab, prot_buf, lig_buf, device)
        writer.write(seqs)
        prot_buf.clear()
        lig_buf.clear()

    for shard_idx, local_indices in tqdm(plan, desc=writer.bin_path.stem):
        shard = torch.load(shard_dir / f"shard_{shard_idx:04d}.pt", weights_only=False)
        for local_idx in local_indices:
            cplx = shard[local_idx]
            prot_buf.append(
                _normalize(cplx["protein"], norm["protein_mean"], norm["protein_std"])
            )
            lig_buf.append(
                _normalize(cplx["ligand"], norm["ligand_mean"], norm["ligand_std"])
            )
            if len(prot_buf) >= batch_size:
                flush()
        del shard
    flush()


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="VQ-VAE checkpoint.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/lm_tokens"))
    parser.add_argument("--from-hub", action="store_true")
    parser.add_argument("--hub-repo-id", type=str, default=None)
    parser.add_argument("--source-types", type=str, nargs="+", default=None)
    parser.add_argument("--ligand-codebook-size", type=int, default=4096)
    parser.add_argument("--protein-codebook-size", type=int, default=8192)
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help=(
            "Override the cache's normalization stats with this .pt file. "
            "REQUIRED when tokenizing a cache other than the one the VQ-VAE was "
            "trained on (e.g. the full 25M cache): pass the VQ-VAE's training "
            "stats so the encoder sees inputs normalized exactly as in training."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-pairs", type=int, default=None, help="Debug subset.")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
    )
    args = parser.parse_args()

    config = VQVAETrainingConfig()
    config.ligand.codebook_size = args.ligand_codebook_size
    config.protein.codebook_size = args.protein_codebook_size

    data_config = CrossDockedConfig()
    if args.max_pairs is not None:
        data_config.max_pairs = args.max_pairs

    hub_config = None
    if args.from_hub:
        hub_config = HubDatasetConfig()
        if args.hub_repo_id is not None:
            hub_config.repo_id = args.hub_repo_id
        if args.source_types is not None:
            hub_config.source_types = args.source_types

    torch.set_float32_matmul_precision("high")

    # Build the split + normalization stats from the descriptor cache.
    dm = ComplexDescriptorDataModule(config, data_config, hub_config=hub_config)
    dm.cache_dir = args.cache_dir
    dm.setup()
    if args.norm_stats is not None:
        dm.norm_stats = torch.load(args.norm_stats, weights_only=False)
        logger.info("Overriding normalization stats from %s", args.norm_stats)
    assert dm.norm_stats is not None  # noqa: S101
    norm = {k: v.numpy() for k, v in dm.norm_stats.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = VQVAEModule.load_from_checkpoint(
        args.ckpt,
        config=config,
        map_location=device,
    )
    module.eval()
    module.to(device)
    # Re-inject normalization stats (used by the coord head; harmless for encode).
    module.protein_vqvae.set_normalization(
        dm.norm_stats["protein_mean"], dm.norm_stats["protein_std"]
    )
    module.ligand_vqvae.set_normalization(
        dm.norm_stats["ligand_mean"], dm.norm_stats["ligand_std"]
    )

    vocab = LMVocab(
        protein_codebook_size=args.protein_codebook_size,
        ligand_codebook_size=args.ligand_codebook_size,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plans = {
        "train": dm._train_plan,  # noqa: SLF001
        "val": dm._val_plan,  # noqa: SLF001
        "test": dm._test_plan,  # noqa: SLF001
    }
    shard_dir = dm._shard_dir  # noqa: SLF001
    assert shard_dir is not None  # noqa: S101

    meta: dict = {
        "vocab_size": vocab.vocab_size,
        "protein_codebook_size": args.protein_codebook_size,
        "ligand_codebook_size": args.ligand_codebook_size,
        "protein_offset": vocab.protein_offset,
        "ligand_offset": vocab.ligand_offset,
        "splits": {},
    }
    for split in args.splits:
        plan = plans[split]
        if not plan:
            logger.warning("Split %s has no entries; skipping", split)
            continue
        writer = SplitWriter(args.out_dir, split)
        _tokenize_split(
            module,
            vocab,
            shard_dir,
            plan,
            norm,
            writer,
            args.batch_size,
            device,
        )
        writer.close()
        meta["splits"][split] = {
            "num_docs": writer.num_docs,
            "num_tokens": writer.num_tokens,
            "max_len": writer.max_len,
        }
        logger.info(
            "%s: %d docs, %d tokens, max_len=%d",
            split,
            writer.num_docs,
            writer.num_tokens,
            writer.max_len,
        )

    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote token cache + meta.json to %s", args.out_dir)


if __name__ == "__main__":
    main()
