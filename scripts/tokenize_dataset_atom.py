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


def _build_pocket_plans(  # noqa: PLR0913
    shard_dir: Path,
    shard_counts: list[int],
    manifest_path: Path,
    source_types: list[str],
    val_frac: float,
    max_per_pocket: int,
    seed: int,
    casf_pdbs: set[str] | None = None,
    num_partitions: int = 1,
    partition_index: int = 0,
) -> tuple[list, list]:
    """Pocket-level train/val plans over fold0-train pockets, capped per pocket.

    Holds out ``val_frac`` of the fold0-TRAIN pockets as a held-out-pocket val
    (disjoint from the fold0-test eval pockets), and caps complexes per pocket so
    no pocket dominates the corpus.

    For parallel tokenization, ``num_partitions``/``partition_index`` restrict the
    work to shards where ``shard_idx % num_partitions == partition_index``. The
    train/val pocket assignment is derived from the manifest + ``seed`` (identical
    in every partition), so a pocket spanning shards keeps ONE consistent label
    across partitions and the partial corpora concatenate without leakage.
    """
    from collections import defaultdict  # noqa: PLC0415

    import pyarrow.parquet as pq  # noqa: PLC0415

    df = pq.read_table(
        manifest_path,
        columns=[
            "pair_idx",
            "complex_dir",
            "source_type",
            "cdonly_fold0",
            "receptor_pdb",
        ],
    ).to_pandas()
    df = df[df["source_type"].isin(source_types)]
    if casf_pdbs:
        pdb = df["receptor_pdb"].str.extract(r"^([0-9a-zA-Z]{4})_")[0].str.lower()
        n_before = len(df)
        df = df[~pdb.isin(casf_pdbs)]
        logger.info("CASF-excluded %d CrossDocked pairs", n_before - len(df))
    pair_to_pocket = dict(zip(df["pair_idx"], df["complex_dir"], strict=False))
    train_pockets = sorted(
        df[df["cdonly_fold0"] == "train"]["complex_dir"].dropna().unique()
    )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_pockets))
    n_val = int(len(train_pockets) * val_frac)
    val_pockets = {train_pockets[i] for i in perm[:n_val]}
    pocket_split = {p: ("val" if p in val_pockets else "train") for p in train_pockets}

    by_pocket: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for shard_idx, _count in enumerate(shard_counts):
        if num_partitions > 1 and shard_idx % num_partitions != partition_index:
            continue
        shard = torch.load(shard_dir / f"shard_{shard_idx:04d}.pt", weights_only=False)
        for local_idx, cplx in enumerate(shard):
            pocket = pair_to_pocket.get(int(cplx["pair_idx"]))
            if pocket in pocket_split:
                by_pocket[pocket].append((shard_idx, local_idx))
        del shard

    train_by_shard: dict[int, list[int]] = defaultdict(list)
    val_by_shard: dict[int, list[int]] = defaultdict(list)
    for pocket, entries in by_pocket.items():
        kept = entries
        if len(entries) > max_per_pocket:
            keep = rng.choice(len(entries), max_per_pocket, replace=False)
            kept = [entries[i] for i in keep]
        target = val_by_shard if pocket_split[pocket] == "val" else train_by_shard
        for shard_idx, local_idx in kept:
            target[shard_idx].append(local_idx)
    train_plan = sorted((si, sorted(lis)) for si, lis in train_by_shard.items())
    val_plan = sorted((si, sorted(lis)) for si, lis in val_by_shard.items())
    logger.info(
        "Pocket split: %d train / %d val pockets (cap %d/pocket)",
        len(train_pockets) - n_val,
        n_val,
        max_per_pocket,
    )
    return train_plan, val_plan


def main() -> None:  # noqa: PLR0915, C901, PLR0912
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--pocket-split",
        action="store_true",
        help="Split by POCKET over fold0-train pockets (hold out a fraction as a "
        "held-out-pocket val; disjoint from the fold0-test eval pockets) and cap "
        "complexes per pocket. Emits train+val only. Use for the conditional "
        "fine-tune so CrossDocked's ~1.7k pockets don't dominate / overfit.",
    )
    parser.add_argument("--pocket-val-frac", type=float, default=0.12)
    parser.add_argument("--max-per-pocket", type=int, default=32)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=1,
        help="Parallel tokenization: process only shards where "
        "shard_idx %% num_partitions == partition_index. Run one job per index "
        "(distinct --out-dir) then concatenate. Pocket-split only.",
    )
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=None, help="Debug subset.")
    parser.add_argument(
        "--casf-pdbs",
        type=Path,
        default=None,
        help="Newline-separated CASF-2016 core PDB ids to hold out (leak-free).",
    )
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.separate_protein_ckpt is not None:
        # ABLATION separate-tokenizers mode: protein-only VQ + ligand-only VQ
        # unified into one code space. Feed RAW descriptors (identity external
        # norm) -- SeparateVQVAE normalizes per modality internally. Combined
        # single-range AtomLMVocab over 2*codebook_size codes.
        from src.tokenizers.separate_vqvae import SeparateVQVAE  # noqa: PLC0415

        module = SeparateVQVAE.from_checkpoints(
            args.separate_protein_ckpt,
            args.separate_protein_norm,
            args.separate_ligand_ckpt,
            args.separate_ligand_norm,
            device,
            codebook_size=args.codebook_size,
        )
        dim = dm.norm_stats["atom_mean"].numel()
        mean = np.zeros(dim, dtype=np.float32)
        std = np.ones(dim, dtype=np.float32)
        vocab = AtomLMVocab(codebook_size=2 * args.codebook_size)
    else:
        mean = dm.norm_stats["atom_mean"].numpy()
        std = dm.norm_stats["atom_std"].numpy()
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
    shard_dir = dm._shard_dir  # noqa: SLF001
    assert shard_dir is not None  # noqa: S101

    if args.pocket_split:
        shard_counts = torch.load(
            dm.cache_dir / "shard_metadata.pt", weights_only=False
        )["shard_counts"]
        manifest_path = Path(hub_config.cache_dir) / "repo" / "manifest.parquet"
        casf_pdbs = None
        if args.casf_pdbs is not None and args.casf_pdbs.exists():
            casf_pdbs = {
                p.strip().lower()
                for p in args.casf_pdbs.read_text().split()
                if p.strip()
            }
        train_plan, val_plan = _build_pocket_plans(
            shard_dir,
            shard_counts,
            manifest_path,
            args.source_types,
            args.pocket_val_frac,
            args.max_per_pocket,
            args.split_seed,
            casf_pdbs=casf_pdbs,
            num_partitions=args.num_partitions,
            partition_index=args.partition_index,
        )
        plans = {"train": train_plan, "val": val_plan}
        args.splits = ["train", "val"]
    else:
        plans = {
            "train": dm._train_plan,  # noqa: SLF001
            "val": dm._val_plan,  # noqa: SLF001
            "test": dm._test_plan,  # noqa: SLF001
        }
    rng = np.random.default_rng(args.seed)

    meta: dict = {
        "vocab_size": vocab.vocab_size,
        "num_rotations": args.num_rotations,
        "all_atom": True,
        "splits": {},
    }
    # In separate-tokenizers mode the combined code space is 2x (protein codes
    # then ligand codes), so downstream must see the doubled size.
    meta["atom_codebook_size"] = (
        2 * args.codebook_size
        if args.separate_protein_ckpt is not None
        else args.codebook_size
    )
    meta["atom_offset"] = vocab.offset
    if args.separate_protein_ckpt is not None:
        meta["separate_tokenizers"] = True
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
