"""Tokenize GEOM conformers into ligand-only LM sequences for pretraining.

Stage-0 of the two-stage curriculum: before fine-tuning the pocket-conditioned
LM on CrossDocked complexes, pretrain it on a large, diverse set of *valid*
3D ligand conformers (GEOM) so it learns realistic ligand geometry and stops
emitting distorted shapes.

Pipeline (mirrors ``scripts/tokenize_dataset.py`` but ligand-only):

1. Load the frozen "2x" ligand VQ-VAE + its **training** normalization stats
   (``--norm-stats``, REQUIRED — same caveat as the CrossDocked tokenizer:
   the encoder must see inputs normalized exactly as in VQ-VAE training).
2. Stream GEOM conformers (:mod:`src.data.geom`), molecule-level split.
3. **Rotation augmentation**: a ligand has no pocket to anchor its orientation,
   so the canonical frame is arbitrary. For each conformer we emit ``--num-
   rotations`` independent uniform-random orientations. The descriptor's radial
   / categorical content is rotation-invariant, so we compute it **once** per
   conformer and cheaply re-express it under each random rotation
   (:func:`rotate_ligand_descriptor`).
4. VQ-VAE-encode and assemble ``<bos><p></p><l> ligand tokens </l><eos>``: the
   empty ``<p></p>`` keeps the exact format the fine-tuning corpus uses, so
   fine-tuning is the same code path with the pocket block filled in.

Output is the standard packed token cache (``{split}.bin`` / ``{split}.len`` /
``meta.json``) consumed by :class:`~src.data.lm_dataset.LMTokenDataModule`.

Run (single GPU)::

    uv run python scripts/tokenize_geom.py \
        --geom-root data/geom/rdkit_folder \
        --ckpt "pocket-ligand-vqvae/3dvcbp0h/.../ligand_coord=0.1501.ckpt" \
        --norm-stats data/descriptor_cache_v4/normalization_stats.pt \
        --subsets drugs --max-confs-per-mol 5 --num-rotations 8 \
        --out-dir data/lm_tokens_geom
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.config import VQVAETrainingConfig
from src.data.descriptors import collate_molecules
from src.data.geom import (
    iter_geom_tar_conformers,
    iter_mol_conformers,
    load_geom_refs,
)
from src.data.token_io import SplitWriter

if TYPE_CHECKING:
    from collections.abc import Iterator
from src.model.vqvae_module import VQVAEModule
from src.tokenizers.geometry import random_rotation_matrix
from src.tokenizers.ligand import LigandDescriptor, rotate_ligand_descriptor
from src.tokenizers.lm_vocab import LMVocab

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_IDENTITY = np.eye(3, dtype=np.float64)


def _count_heavy(atoms: list[tuple[str, float, float, float]]) -> int:
    return sum(1 for a in atoms if a[0] != "H")


def _heavy_centroid(atoms: list[tuple[str, float, float, float]]) -> np.ndarray:
    heavy = np.array(
        [(a[1], a[2], a[3]) for a in atoms if a[0] != "H"],
        dtype=np.float64,
    )
    return heavy.mean(axis=0)


class _Tokenizer:
    """Buffers per-split descriptors and flushes them through the VQ-VAE."""

    def __init__(  # noqa: PLR0913
        self,
        module: VQVAEModule,
        vocab: LMVocab,
        lig_mean: np.ndarray,
        lig_std: np.ndarray,
        writers: dict[str, SplitWriter],
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.module = module
        self.vocab = vocab
        self.lig_mean = lig_mean
        self.lig_std = lig_std
        self.writers = writers
        self.batch_size = batch_size
        self.device = device
        self._buf: dict[str, list[torch.Tensor]] = {s: [] for s in writers}

    def add(self, split: str, descriptor: np.ndarray) -> None:
        norm = (descriptor - self.lig_mean) / self.lig_std
        self._buf[split].append(torch.from_numpy(norm).float())
        if len(self._buf[split]) >= self.batch_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        buf = self._buf[split]
        if not buf:
            return
        lig_x, lig_mask = collate_molecules(buf)
        idx = self.module.ligand_vqvae.encode_batch(
            lig_x.to(self.device), lig_mask.to(self.device)
        ).cpu()
        seqs = [
            self.vocab.build_sequence([], idx[i][lig_mask[i]].tolist())
            for i in range(len(buf))
        ]
        self.writers[split].write(seqs)
        buf.clear()

    def flush_all(self) -> None:
        for split in self._buf:
            self.flush(split)


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--geom-tar",
        type=Path,
        default=None,
        help="Stream conformers directly from rdkit_folder.tar.gz (inode-safe: "
        "no extraction). Preferred on inode-constrained filesystems.",
    )
    src.add_argument(
        "--geom-root",
        type=Path,
        default=None,
        help="Read conformers from an already-extracted rdkit_folder directory.",
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="VQ-VAE checkpoint.")
    parser.add_argument(
        "--norm-stats",
        type=Path,
        required=True,
        help="VQ-VAE training normalization stats (.pt). REQUIRED so the frozen "
        "encoder sees inputs normalized exactly as in training.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/lm_tokens_geom"))
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["drugs"],
        choices=["drugs", "qm9"],
    )
    parser.add_argument("--max-confs-per-mol", type=int, default=5)
    parser.add_argument(
        "--num-rotations",
        type=int,
        default=8,
        help="Independent random orientations emitted per conformer.",
    )
    parser.add_argument("--heavy-atom-max", type=int, default=120)
    parser.add_argument("--ligand-codebook-size", type=int, default=4096)
    parser.add_argument("--protein-codebook-size", type=int, default=8192)
    parser.add_argument("--val-frac", type=float, default=0.005)
    parser.add_argument("--test-frac", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-mols", type=int, default=None, help="Debug subset.")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- VQ-VAE + normalization stats --------------------------------------
    config = VQVAETrainingConfig()
    config.ligand.codebook_size = args.ligand_codebook_size
    config.protein.codebook_size = args.protein_codebook_size
    module = VQVAEModule.load_from_checkpoint(
        args.ckpt, config=config, map_location=device
    )
    module.eval()
    module.to(device)
    norm_stats = torch.load(args.norm_stats, weights_only=False)
    module.ligand_vqvae.set_normalization(
        norm_stats["ligand_mean"], norm_stats["ligand_std"]
    )
    lig_mean = norm_stats["ligand_mean"].numpy()
    lig_std = norm_stats["ligand_std"].numpy()

    vocab = LMVocab(
        protein_codebook_size=args.protein_codebook_size,
        ligand_codebook_size=args.ligand_codebook_size,
    )

    # --- Unified (split, conformer) source ---------------------------------
    wanted = set(args.splits)

    from tqdm import tqdm  # noqa: PLC0415

    def _mol_source() -> Iterator[tuple[str, dict]]:
        """Yield ``(split, conformer_dict)`` from either a tar or a directory."""
        if args.geom_tar is not None:
            yield from iter_geom_tar_conformers(
                args.geom_tar,
                args.subsets,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                seed=args.seed,
                max_confs_per_mol=args.max_confs_per_mol,
                max_mols=args.max_mols,
            )
        else:
            refs = load_geom_refs(
                args.geom_root,
                args.subsets,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                seed=args.seed,
                max_mols=args.max_mols,
            )
            logger.info("GEOM refs: %d molecules", len(refs))
            for ref in refs:
                for mol in iter_mol_conformers(
                    args.geom_root, ref, args.max_confs_per_mol
                ):
                    yield ref.split, mol

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {s: SplitWriter(args.out_dir, s) for s in args.splits}
    tok = _Tokenizer(
        module, vocab, lig_mean, lig_std, writers, args.batch_size, device
    )
    descriptor = LigandDescriptor()
    rng = np.random.default_rng(args.seed)

    n_confs = 0
    n_skipped = 0
    for split, mol in tqdm(_mol_source(), desc="geom-confs", unit="conf"):
        if split not in wanted:
            continue
        atoms = mol["atoms"]
        n_heavy = _count_heavy(atoms)
        if n_heavy < 1 or n_heavy > args.heavy_atom_max:
            n_skipped += 1
            continue
        centroid = _heavy_centroid(atoms)
        base_desc, _elements, _meta = descriptor.compute(
            atoms, mol["bonds"], pocket_frame=(centroid, _IDENTITY)
        )
        if base_desc.shape[0] == 0:
            n_skipped += 1
            continue
        n_confs += 1
        for _ in range(args.num_rotations):
            rot = random_rotation_matrix(rng)
            tok.add(split, rotate_ligand_descriptor(base_desc, rot))
    tok.flush_all()

    meta: dict = {
        "vocab_size": vocab.vocab_size,
        "protein_codebook_size": args.protein_codebook_size,
        "ligand_codebook_size": args.ligand_codebook_size,
        "protein_offset": vocab.protein_offset,
        "ligand_offset": vocab.ligand_offset,
        "pretrain": {
            "source": "geom",
            "subsets": args.subsets,
            "max_confs_per_mol": args.max_confs_per_mol,
            "num_rotations": args.num_rotations,
            "conformers_used": n_confs,
            "conformers_skipped": n_skipped,
            "seed": args.seed,
            "ligand_only": True,
        },
        "splits": {},
    }
    for split, writer in writers.items():
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
    logger.info(
        "Wrote GEOM token cache to %s (%d conformers, %d skipped)",
        args.out_dir,
        n_confs,
        n_skipped,
    )


if __name__ == "__main__":
    main()
