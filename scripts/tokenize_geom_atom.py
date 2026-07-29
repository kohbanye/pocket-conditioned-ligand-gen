"""Tokenize GEOM conformers into ligand-only all-atom LM sequences (pretrain).

All-atom counterpart of ``scripts/tokenize_geom.py``: stage-0 of the curriculum
for the unified pipeline. Encodes GEOM ligand conformers with the frozen
all-atom VQ-VAE (ligand atoms only, source=ligand) and emits
``<bos><p></p><l> ligand atom tokens </l><eos>`` over the single atom code range
(:class:`AtomLMVocab`) -- the empty ``<p></p>`` matches the fine-tuning format.

Run (single GPU)::

    uv run python scripts/tokenize_geom_atom.py \
        --geom-tar data/geom/rdkit_folder.tar.gz \
        --ckpt "<atom-vqvae>.ckpt" \
        --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
        --subsets drugs --max-confs-per-mol 5 --num-rotations 8 \
        --out-dir data/lm_tokens_geom_allatom
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.config import AtomVQVAETrainingConfig
from src.data.descriptors import collate_molecules
from src.data.geom import (
    iter_geom_tar_conformers,
    iter_mol_conformers,
    load_geom_refs,
)
from src.data.token_io import SplitWriter
from src.model.vqvae_module import AtomVQVAEModule
from src.tokenizers.atom import LigandAtomDescriptor, rotate_atom_descriptor
from src.tokenizers.geometry import random_rotation_matrix
from src.tokenizers.lm_vocab import AtomLMVocab

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    """Buffers per-split ligand-atom descriptors and flushes through the VQ-VAE."""

    def __init__(  # noqa: PLR0913
        self,
        module: AtomVQVAEModule,
        vocab: AtomLMVocab,
        mean: np.ndarray,
        std: np.ndarray,
        writers: dict[str, SplitWriter],
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.module = module
        self.vocab = vocab
        self.mean = mean
        self.std = std
        self.writers = writers
        self.batch_size = batch_size
        self.device = device
        self._buf: dict[str, list[torch.Tensor]] = {s: [] for s in writers}

    def add(self, split: str, descriptor: np.ndarray) -> None:
        norm = (descriptor - self.mean) / self.std
        self._buf[split].append(torch.from_numpy(norm).float())
        if len(self._buf[split]) >= self.batch_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        buf = self._buf[split]
        if not buf:
            return
        lig_x, lig_mask = collate_molecules(buf)
        idx = self.module.vqvae.encode_batch(
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


def main() -> None:  # noqa: C901, PLR0915, PLR0912
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--geom-tar", type=Path, default=None)
    src.add_argument("--geom-root", type=Path, default=None)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Atom VQ-VAE ckpt (joint). Omit when using --separate-*-ckpt.",
    )
    parser.add_argument("--separate-protein-ckpt", type=Path, default=None)
    parser.add_argument("--separate-protein-norm", type=Path, default=None)
    parser.add_argument("--separate-ligand-ckpt", type=Path, default=None)
    parser.add_argument("--separate-ligand-norm", type=Path, default=None)
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Atom VQ-VAE training normalization stats (.pt).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/lm_tokens_geom_allatom")
    )
    parser.add_argument(
        "--subsets", type=str, nargs="+", default=["drugs"], choices=["drugs", "qm9"]
    )
    parser.add_argument("--max-confs-per-mol", type=int, default=5)
    parser.add_argument("--num-rotations", type=int, default=8)
    parser.add_argument("--heavy-atom-max", type=int, default=120)
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--val-frac", type=float, default=0.005)
    parser.add_argument("--test-frac", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-mols", type=int, default=None, help="Debug subset.")
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=1,
        help="Split the (deterministically-ordered) conformer stream into this "
        "many disjoint partitions and keep only conformers whose stream index "
        "conf_idx %% num_partitions == partition_index. Run one job per index and "
        "concatenate the partial caches (build_mixed_pretrain_cache.py). The "
        "molecule->split assignment is derived from the SMILES + seed identically "
        "in every partition, so partial corpora concatenate without leakage.",
    )
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["train", "val", "test"]
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AtomVQVAETrainingConfig()
    config.atom.codebook_size = args.codebook_size
    if args.separate_protein_ckpt is not None:
        # ABLATION separate-tokenizers mode: protein-only VQ + ligand-only VQ
        # unified into one code space. Feed RAW descriptors (identity external
        # norm) -- SeparateVQVAE normalizes per modality internally. GEOM is
        # ligand-only, so tokens land in the ligand half of the 2*codebook_size
        # combined single-range AtomLMVocab.
        from src.tokenizers.descriptor_schema import (  # noqa: PLC0415
            ATOM_DESCRIPTOR_DIM,
        )
        from src.tokenizers.separate_vqvae import SeparateVQVAE  # noqa: PLC0415

        module = SeparateVQVAE.from_checkpoints(
            args.separate_protein_ckpt,
            args.separate_protein_norm,
            args.separate_ligand_ckpt,
            args.separate_ligand_norm,
            device,
            codebook_size=args.codebook_size,
        )
        mean = np.zeros(ATOM_DESCRIPTOR_DIM, dtype=np.float32)
        std = np.ones(ATOM_DESCRIPTOR_DIM, dtype=np.float32)
        vocab: AtomLMVocab = AtomLMVocab(
            codebook_size=2 * args.codebook_size
        )
    else:
        module = AtomVQVAEModule.load_from_checkpoint(
            args.ckpt, config=config, map_location=device
        )
        module.eval()
        module.to(device)
        norm_stats = torch.load(args.norm_stats, weights_only=False)
        module.vqvae.set_normalization(norm_stats["atom_mean"], norm_stats["atom_std"])
        mean = norm_stats["atom_mean"].numpy()
        std = norm_stats["atom_std"].numpy()

        vocab = AtomLMVocab(codebook_size=args.codebook_size)
    wanted = set(args.splits)

    from tqdm import tqdm  # noqa: PLC0415

    def _mol_source() -> Iterator[tuple[str, dict]]:
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
            for ref in refs:
                for mol in iter_mol_conformers(
                    args.geom_root, ref, args.max_confs_per_mol
                ):
                    yield ref.split, mol

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {s: SplitWriter(args.out_dir, s) for s in args.splits}
    tok = _Tokenizer(module, vocab, mean, std, writers, args.batch_size, device)
    descriptor = LigandAtomDescriptor()
    rng = np.random.default_rng(args.seed)

    n_confs = 0
    n_skipped = 0
    for conf_idx, (split, mol) in enumerate(
        tqdm(_mol_source(), desc="geom-confs", unit="conf")
    ):
        # Deterministic, non-overlapping partition of the raw conformer stream
        # (mirrors the shard_idx %% num_partitions filter of tokenize_dataset_atom).
        if (
            args.num_partitions > 1
            and conf_idx % args.num_partitions != args.partition_index
        ):
            continue
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
            tok.add(split, rotate_atom_descriptor(base_desc, rot))
    tok.flush_all()

    meta: dict = {
        "vocab_size": vocab.vocab_size,
        "all_atom": True,
        "pretrain": {
            "source": "geom",
            "subsets": args.subsets,
            "max_confs_per_mol": args.max_confs_per_mol,
            "num_rotations": args.num_rotations,
            "conformers_used": n_confs,
            "conformers_skipped": n_skipped,
            "seed": args.seed,
            "ligand_only": True,
            "num_partitions": args.num_partitions,
            "partition_index": args.partition_index,
        },
        "splits": {},
    }
    meta["atom_codebook_size"] = (
        2 * args.codebook_size
        if args.separate_protein_ckpt is not None
        else args.codebook_size
    )
    meta["atom_offset"] = vocab.offset
    if args.separate_protein_ckpt is not None:
        meta["separate_tokenizers"] = True
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
        "Wrote GEOM all-atom token cache to %s (%d conformers, %d skipped)",
        args.out_dir,
        n_confs,
        n_skipped,
    )


if __name__ == "__main__":
    main()
