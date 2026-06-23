"""Tokenize CrossDocked complexes into unified all-atom LM token streams.

All-atom counterpart of ``scripts/tokenize_dataset.py``: loads the frozen
all-atom VQ-VAE, encodes BOTH the protein-pocket atoms and the ligand atoms
with the SAME codebook, and assembles ``<bos><p> prot-atoms </p><l> lig-atoms
</l><eos>`` over one shared code range (:class:`AtomLMVocab`).

Rotation augmentation: the cached descriptors live in the pocket canonical
frame; for ``--num-rotations > 1`` we re-express each complex under extra random
orientations (the SAME rotation applied to protein + ligand) to recover token
count from the smaller good-pose corpus. Rotation 0 is the true canonical frame.

Run (single GPU)::

    uv run python scripts/tokenize_dataset_atom.py \
        --ckpt "<atom-vqvae>.ckpt" --cache-dir data/descriptor_cache_allatom \
        --source-types cdonly --codebook-size 8192 --num-rotations 4 \
        --out-dir data/lm_tokens_allatom
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.config import AtomVQVAETrainingConfig, CrossDockedConfig, HubDatasetConfig
from src.data.atom_descriptors import AtomComplexDescriptorDataModule
from src.data.descriptors import collate_molecules
from src.data.token_io import SplitWriter
from src.model.vqvae_module import AtomVQVAEModule
from src.tokenizers.atom import rotate_atom_descriptor
from src.tokenizers.geometry import random_rotation_matrix
from src.tokenizers.lm_vocab import AtomLMVocab

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_IDENTITY = np.eye(3, dtype=np.float64)


def _encode_buffer(
    module: AtomVQVAEModule,
    vocab: AtomLMVocab,
    prot_descs: list[torch.Tensor],
    lig_descs: list[torch.Tensor],
    device: torch.device,
) -> list[list[int]]:
    prot_x, prot_mask = collate_molecules(prot_descs)
    lig_x, lig_mask = collate_molecules(lig_descs)
    prot_idx = module.vqvae.encode_batch(
        prot_x.to(device), prot_mask.to(device)
    ).cpu()
    lig_idx = module.vqvae.encode_batch(lig_x.to(device), lig_mask.to(device)).cpu()
    sequences: list[list[int]] = []
    for i in range(len(prot_descs)):
        p_codes = prot_idx[i][prot_mask[i]].tolist()
        l_codes = lig_idx[i][lig_mask[i]].tolist()
        sequences.append(vocab.build_sequence(p_codes, l_codes))
    return sequences


def _tokenize_split(  # noqa: PLR0913
    module: AtomVQVAEModule,
    vocab: AtomLMVocab,
    shard_dir: Path,
    plan: list[tuple[int, list[int]]],
    mean: np.ndarray,
    std: np.ndarray,
    writer: SplitWriter,
    batch_size: int,
    num_rotations: int,
    rng: np.random.Generator,
    device: torch.device,
) -> None:
    from tqdm import tqdm  # noqa: PLC0415

    prot_buf: list[torch.Tensor] = []
    lig_buf: list[torch.Tensor] = []

    def flush() -> None:
        if not prot_buf:
            return
        writer.write(_encode_buffer(module, vocab, prot_buf, lig_buf, device))
        prot_buf.clear()
        lig_buf.clear()

    def _norm(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((arr - mean) / std).float()

    for shard_idx, local_indices in tqdm(plan, desc=writer.bin_path.stem):
        shard = torch.load(shard_dir / f"shard_{shard_idx:04d}.pt", weights_only=False)
        for local_idx in local_indices:
            cplx = shard[local_idx]
            prot_raw = cplx["protein"]
            lig_raw = cplx["ligand"]
            for r in range(num_rotations):
                if r == 0:
                    prot_a, lig_a = prot_raw, lig_raw
                else:
                    rot = random_rotation_matrix(rng)
                    prot_a = rotate_atom_descriptor(prot_raw, rot)
                    lig_a = rotate_atom_descriptor(lig_raw, rot)
                prot_buf.append(_norm(prot_a))
                lig_buf.append(_norm(lig_a))
                if len(prot_buf) >= batch_size:
                    flush()
        del shard
    flush()


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="Atom VQ-VAE ckpt.")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/descriptor_cache_allatom")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/lm_tokens_allatom"))
    parser.add_argument("--source-types", type=str, nargs="+", default=["cdonly"])
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Override the cache's atom normalization stats (.pt).",
    )
    parser.add_argument("--num-rotations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=None, help="Debug subset.")
    parser.add_argument("--include-decoys", action="store_true")
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["train", "val", "test"]
    )
    args = parser.parse_args()

    config = AtomVQVAETrainingConfig()
    config.atom.codebook_size = args.codebook_size

    data_config = CrossDockedConfig()
    if args.max_pairs is not None:
        data_config.max_pairs = args.max_pairs

    hub_config = HubDatasetConfig()
    hub_config.source_types = args.source_types
    hub_config.good_poses_only = not args.include_decoys

    torch.set_float32_matmul_precision("high")

    dm = AtomComplexDescriptorDataModule(config, data_config, hub_config=hub_config)
    dm.cache_dir = args.cache_dir
    dm.setup()
    if args.norm_stats is not None:
        dm.norm_stats = torch.load(args.norm_stats, weights_only=False)
        logger.info("Overriding atom normalization stats from %s", args.norm_stats)
    assert dm.norm_stats is not None  # noqa: S101
    mean = dm.norm_stats["atom_mean"].numpy()
    std = dm.norm_stats["atom_std"].numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = AtomVQVAEModule.load_from_checkpoint(
        args.ckpt, config=config, map_location=device
    )
    module.eval()
    module.to(device)
    module.vqvae.set_normalization(
        dm.norm_stats["atom_mean"], dm.norm_stats["atom_std"]
    )

    vocab = AtomLMVocab(codebook_size=args.codebook_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plans = {
        "train": dm._train_plan,  # noqa: SLF001
        "val": dm._val_plan,  # noqa: SLF001
        "test": dm._test_plan,  # noqa: SLF001
    }
    shard_dir = dm._shard_dir  # noqa: SLF001
    assert shard_dir is not None  # noqa: S101
    rng = np.random.default_rng(args.seed)

    meta: dict = {
        "vocab_size": vocab.vocab_size,
        "atom_codebook_size": args.codebook_size,
        "atom_offset": vocab.offset,
        "num_rotations": args.num_rotations,
        "all_atom": True,
        "splits": {},
    }
    for split in args.splits:
        plan = plans[split]
        if not plan:
            logger.warning("Split %s has no entries; skipping", split)
            continue
        # Validation / test are not augmented (evaluate the true canonical frame).
        n_rot = args.num_rotations if split == "train" else 1
        writer = SplitWriter(args.out_dir, split)
        _tokenize_split(
            module,
            vocab,
            shard_dir,
            plan,
            mean,
            std,
            writer,
            args.batch_size,
            n_rot,
            rng,
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
    logger.info("Wrote all-atom token cache + meta.json to %s", args.out_dir)


if __name__ == "__main__":
    main()
