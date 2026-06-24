"""Tokenize PLINDER holo pockets into protein-only LM pretraining sequences.

Stage-0 (protein side) of the curriculum: build a large, diverse pocket-language
corpus from PLINDER (>300k holo binding sites) so the LM learns ``p(pocket)``
beyond CrossDocked's ~1.7k pockets, before the pocket-conditioned fine-tune.

Pipeline (inode-safe; never extracts the zips):
1. Read PLINDER ``splits/split.parquet`` -> train/val systems. Drop systems
   whose PDB id is a CrossDocked fold-0 TEST receptor (leakage).
2. Stream each ``systems/*.zip`` member in memory. For each kept system, parse
   ``receptor.pdb`` + ``ligand_files/*.sdf``, extract the all-atom pocket (8 A /
   <=50 residues around the bound ligand) with the same extraction the
   CrossDocked path uses (CPU, multiprocess one worker per zip).
3. Encode each pocket with the frozen all-atom VQ-VAE and emit
   ``<bos><p> pocket atoms </p><l></l><eos>`` (empty ligand block, matching the
   complex format) with rotation augmentation.

Run (single GPU)::

    uv run python scripts/tokenize_plinder_protein.py \
        --ckpt "<atom-vqvae>.ckpt" \
        --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
        --systems-dir data/plinder/systems --split data/plinder/split.parquet \
        --num-rotations 4 --out-dir data/lm_tokens_protein_plinder
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.config import AtomVQVAETrainingConfig, PocketExtractionConfig
from src.data.descriptors import collate_molecules
from src.data.token_io import SplitWriter
from src.model.vqvae_module import AtomVQVAEModule
from src.tokenizers.atom import (
    ProteinAtomDescriptor,
    precompute_receptor_atom_features_from_text,
    rotate_atom_descriptor,
)
from src.tokenizers.geometry import random_rotation_matrix
from src.tokenizers.ligand import parse_sdf_text
from src.tokenizers.lm_vocab import AtomLMVocab
from src.tokenizers.protein import (
    _compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    precompute_pocket_atom_candidates_from_text,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Per-worker state.
_w_prot_desc: ProteinAtomDescriptor | None = None
_w_pocket_config: PocketExtractionConfig | None = None
_w_allowed: dict[str, str] = {}


def _system_pdb(system_id: str) -> str:
    """PLINDER system_id -> 4-char PDB id (e.g. '1abc__1__1.A__1.B' -> '1abc')."""
    return system_id.split("__", 1)[0].lower()


def _load_allowed_systems(
    split_path: Path,
    cd_manifest: Path,
) -> dict[str, str]:
    """Map kept ``system_id -> split`` ('train'/'val'), leak-filtered vs CD test."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    sp = pq.read_table(split_path).to_pandas()
    split_col = (
        "split"
        if "split" in sp.columns
        else next(c for c in sp.columns if "split" in c.lower())
    )
    sid_col = "system_id" if "system_id" in sp.columns else sp.columns[0]

    cd = pq.read_table(
        cd_manifest, columns=["receptor_pdb", "source_type", "cdonly_fold0"]
    ).to_pandas()
    cd = cd[(cd["source_type"] == "cdonly") & (cd["cdonly_fold0"] == "test")]
    cd_test_pdbs = set(
        cd["receptor_pdb"].str.extract(r"^([0-9a-zA-Z]{4})_")[0].str.lower().dropna()
    )

    allowed: dict[str, str] = {}
    n_leak = 0
    for sid, label in zip(sp[sid_col], sp[split_col], strict=True):
        if label not in ("train", "val"):
            continue
        if _system_pdb(str(sid)) in cd_test_pdbs:
            n_leak += 1
            continue
        allowed[str(sid)] = label
    logger.info(
        "PLINDER systems kept: %d (train/val), leak-dropped vs CD test: %d",
        len(allowed),
        n_leak,
    )
    return allowed


def _worker_init(pocket_config_dict: dict, allowed: dict[str, str]) -> None:
    global _w_prot_desc, _w_pocket_config, _w_allowed  # noqa: PLW0603
    _w_prot_desc = ProteinAtomDescriptor()
    _w_pocket_config = PocketExtractionConfig(**pocket_config_dict)
    _w_allowed = allowed


def _ligand_heavy_coords(zf: zipfile.ZipFile, lig_members: list[str]) -> np.ndarray:
    coords: list[tuple[float, float, float]] = []
    for m in lig_members:
        text = zf.read(m).decode("utf-8", "replace")
        for mol in parse_sdf_text(text):
            coords.extend((a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H")
    return (
        np.array(coords, dtype=np.float32) if coords else np.empty((0, 3), np.float32)
    )


def _process_zip(zip_path: str) -> list[tuple[str, np.ndarray]]:  # noqa: C901
    """Extract base (un-normalized) pocket descriptors for kept systems in a zip."""
    out: list[tuple[str, np.ndarray]] = []
    try:
        zf = zipfile.ZipFile(zip_path)
    except Exception:
        logger.exception("Bad zip %s", zip_path)
        return out

    # Group members by top-level system dir.
    systems: dict[str, list[str]] = {}
    for name in zf.namelist():
        top = name.split("/", 1)[0]
        if top:
            systems.setdefault(top, []).append(name)

    for sid, members in systems.items():
        label = _w_allowed.get(sid)
        if label is None:
            continue
        rec = next((m for m in members if m.endswith("/receptor.pdb")), None)
        ligs = [m for m in members if "/ligand_files/" in m and m.endswith(".sdf")]
        if rec is None or not ligs:
            continue
        try:
            rec_text = zf.read(rec).decode("utf-8", "replace")
            lig_coords = _ligand_heavy_coords(zf, ligs)
            if lig_coords.shape[0] == 0:
                continue
            precomp = precompute_pocket_atom_candidates_from_text(rec_text)
            pocket = extract_pocket_atoms_from_candidates(
                precomp, lig_coords, _w_pocket_config
            )
            if pocket is None or pocket.atom_coords.shape[0] == 0:
                continue
            feats = precompute_receptor_atom_features_from_text(rec_text)
            centroid, rotation = _compute_canonical_frame(
                pocket.ca_coords.astype(np.float64)
            )
            desc, _meta = _w_prot_desc.compute(pocket, feats, (centroid, rotation))
        except Exception:
            logger.exception("Error on system %s in %s", sid, zip_path)
            continue
        if desc.shape[0] > 0:
            out.append((label, desc))
    zf.close()
    return out


class _Encoder:
    """Buffers (split, descriptor) rows and flushes them through the atom VQ-VAE."""

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
        if split not in self._buf:
            return
        norm = (descriptor - self.mean) / self.std
        self._buf[split].append(torch.from_numpy(norm).float())
        if len(self._buf[split]) >= self.batch_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        buf = self._buf[split]
        if not buf:
            return
        x, mask = collate_molecules(buf)
        idx = self.module.vqvae.encode_batch(
            x.to(self.device), mask.to(self.device)
        ).cpu()
        seqs = [
            self.vocab.build_sequence(idx[i][mask[i]].tolist(), [])
            for i in range(len(buf))
        ]
        self.writers[split].write(seqs)
        buf.clear()

    def flush_all(self) -> None:
        for split in self._buf:
            self.flush(split)


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="Atom VQ-VAE ckpt.")
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument(
        "--systems-dir", type=Path, default=Path("data/plinder/systems")
    )
    parser.add_argument(
        "--split", type=Path, default=Path("data/plinder/split.parquet")
    )
    parser.add_argument(
        "--cd-manifest", type=Path, default=Path("data/hub_cache/repo/manifest.parquet")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/lm_tokens_protein_plinder")
    )
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--max-residues", type=int, default=50)
    parser.add_argument("--num-rotations", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-zips", type=int, default=None, help="Debug subset.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AtomVQVAETrainingConfig()
    config.atom.codebook_size = args.codebook_size
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

    allowed = _load_allowed_systems(args.split, args.cd_manifest)
    zips = sorted(str(p) for p in args.systems_dir.glob("*.zip"))
    if args.max_zips is not None:
        zips = zips[: args.max_zips]
    logger.info("Streaming %d PLINDER system zips", len(zips))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {s: SplitWriter(args.out_dir, s) for s in ("train", "val")}
    enc = _Encoder(module, vocab, mean, std, writers, args.batch_size, device)
    rng = np.random.default_rng(args.seed)

    from dataclasses import asdict  # noqa: PLC0415

    pocket_cfg = PocketExtractionConfig(max_residues=args.max_residues)
    n_pockets = 0

    def _consume(results: Iterable[tuple[str, np.ndarray]]) -> None:
        nonlocal n_pockets
        for label, desc in results:
            n_pockets += 1
            n_rot = args.num_rotations if label == "train" else 1
            for r in range(n_rot):
                da = (
                    desc
                    if r == 0
                    else rotate_atom_descriptor(desc, random_rotation_matrix(rng))
                )
                enc.add(label, da)

    from tqdm import tqdm  # noqa: PLC0415

    if args.num_workers > 0:
        import multiprocessing  # noqa: PLC0415

        with multiprocessing.Pool(
            args.num_workers,
            initializer=_worker_init,
            initargs=(asdict(pocket_cfg), allowed),
        ) as pool:
            for results in tqdm(
                pool.imap_unordered(_process_zip, zips), total=len(zips), desc="zips"
            ):
                _consume(results)
    else:
        _worker_init(asdict(pocket_cfg), allowed)
        for zp in tqdm(zips, desc="zips"):
            _consume(_process_zip(zp))

    enc.flush_all()
    meta: dict = {
        "vocab_size": vocab.vocab_size,
        "atom_codebook_size": args.codebook_size,
        "atom_offset": vocab.offset,
        "all_atom": True,
        "pretrain": {
            "source": "plinder",
            "num_rotations": args.num_rotations,
            "pockets_used": n_pockets,
            "protein_only": True,
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
    logger.info("Wrote PLINDER protein-only token cache to %s", args.out_dir)


if __name__ == "__main__":
    main()
