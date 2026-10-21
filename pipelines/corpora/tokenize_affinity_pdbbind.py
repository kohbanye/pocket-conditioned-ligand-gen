"""Binding-affinity corpus from PDBbind v2020 (refined + general), labelled pK.

This is the corpus GenScore/RTMScore train on: ~19k curated protein-ligand
complexes with an experimental Kd/Ki/IC50 and a crystal pose. It replaces the
BioLIP-scraped affinity corpus, whose labels come from four sources of mixed
quality -- the diagnosis pinned the affinity head's weakness on leaning on
molecular size rather than pocket-specific contacts, and cleaner/larger training
data is the most direct lever on scoring power.

Labels are read from the pre-parsed split CSVs under the PDBbind tree
(``PDB_ID, Label_pKd_pKi, Measure_Type, ...``); ``Label_pKd_pKi`` is already
-log10(molar). The CASF-2016 core is held out (it is the benchmark) -- it is
already absent from those CSVs, and excluded again here as a safety net.

Each complex is tokenized ONCE (crystal pose): the pocket is carved from
``{pdb}_protein.pdb`` around the ligand's heavy atoms and the ligand from
``{pdb}_ligand.sdf`` -- exactly the setup CASF eval uses, so train and test see
the same pocket construction. Output reuses the RescoreDataset layout
(``{split}.bin/.len/.rmsd`` with pK in the label stream) so ``train_rescore.py``
consumes it unchanged.

Run (single GPU)::

    uv run python pipelines/corpora/tokenize_affinity_pdbbind.py \
        --ckpt <atom-vqvae>.ckpt \
        --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
        --out-dir data/lm_tokens_affinity_pdbbind
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import torch

# Sibling modules in this directory, imported by bare name: Python puts a
# script's own directory on sys.path[0], so this resolves from any cwd.
from tokenize_decoys import _RmsdWriter

from prolit.config import AtomVQVAETrainingConfig, PocketExtractionConfig
from prolit.seeding import add_seed_argument, rng_for, seed_from_args
from prolit.tokenizers.geometry import random_rotation_matrix
from prolit.tokenizers.ligand import parse_sdf
from prolit.tokenizers.lm_vocab import AtomLMVocab
from prolit.tokenizers.loaders import load_atom_vqvae
from prolit.tokenizers.pose_encoder import PoseEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_labels(
    pdbbind_dir: Path, splits: tuple[str, ...], kinds: set[str] | None
) -> dict[str, float]:
    """pdb -> pK, read from the pre-parsed split CSVs. ``kinds`` (e.g. {"KD",
    "KI"}) restricts the measurement type; CASF is Kd/Ki only, but IC50 nearly
    doubles the data and PDBbind curates it, so it is kept by default."""
    out: dict[str, float] = {}
    for split in splits:
        csv_path = pdbbind_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                pdb = row["PDB_ID"].strip().lower()
                measure = row["Measure_Type"].strip().upper().rstrip("<>=~")
                if kinds is not None and measure not in kinds:
                    continue
                try:
                    pk = float(row["Label_pKd_pKi"])
                except (ValueError, KeyError):
                    continue
                out[pdb] = pk
    return out


def _find_complex(roots: list[Path], pdb: str) -> Path | None:
    for r in roots:
        d = r / pdb
        if (d / f"{pdb}_protein.pdb").exists() and (d / f"{pdb}_ligand.sdf").exists():
            return d
    return None


def main() -> None:  # noqa: C901, PLR0915
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--norm-stats", type=Path, required=True)
    p.add_argument(
        "--pdbbind-dir",
        type=Path,
        default=Path(os.environ.get("PDBBIND_DIR", "data/pdbbind")),
        help="PDBbind v2020 root (refined + general). Or set PDBBIND_DIR.",
    )
    p.add_argument("--casf-pdbs", type=Path, default=Path("data/casf2016_pdbs.txt"))
    p.add_argument(
        "--out-dir", type=Path, default=Path("data/lm_tokens_affinity_pdbbind")
    )
    p.add_argument("--codebook-size", type=int, default=8192)
    p.add_argument("--max-residues", type=int, default=50)
    p.add_argument("--pocket-distance", choices=("ca", "heavy"), default="ca")
    p.add_argument("--min-heavy", type=int, default=6)
    p.add_argument("--max-heavy", type=int, default=60)
    p.add_argument(
        "--affinity-types",
        default="KD,KI,IC50",
        help="Measurement types to keep. 'KD,KI' matches CASF exactly but halves "
        "the data; IC50 is PDBbind-curated so it is kept by default.",
    )
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument(
        "--num-rotations",
        type=int,
        default=1,
        help="Documents per TRAIN complex, each under its own random rotation of "
        "the whole complex before quantization (the BioLiP builder's flag, same "
        "semantics). The pocket-canonical frame makes a pose's tokens "
        "frame-dependent through the codebook, so one orientation per complex "
        "teaches the head to read quantization noise as signal. Val is always 1: "
        "it measures the model, not the augmentation. k=0 is the unrotated "
        "complex, so 1 reproduces the original corpus byte for byte.",
    )
    add_seed_argument(p, default=0)
    args = p.parse_args()
    seed_from_args(args)
    # Same stream name as the BioLiP builder, so a rotation is drawn the same
    # way in both halves of a union corpus.
    rot_rng = rng_for(args.seed, "affinity_rotations")

    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = AtomVQVAETrainingConfig()
    cfg.atom.codebook_size = args.codebook_size
    module = load_atom_vqvae(args.ckpt, device)
    module.eval().to(device)
    norm = torch.load(args.norm_stats, weights_only=False)
    module.vqvae.set_normalization(norm["atom_mean"], norm["atom_std"])
    enc = PoseEncoder(
        module.vqvae,
        norm["atom_mean"].numpy(),
        norm["atom_std"].numpy(),
        AtomLMVocab(codebook_size=args.codebook_size),
        device,
        PocketExtractionConfig(
            max_residues=args.max_residues, pocket_distance=args.pocket_distance
        ),
    )

    kinds = {k.strip().upper() for k in args.affinity_types.split(",") if k.strip()}
    labels = _read_labels(args.pdbbind_dir, ("train", "val", "test"), kinds or None)
    excluded = set()
    if args.casf_pdbs.exists():
        excluded = {x.lower() for x in args.casf_pdbs.read_text().split() if x.strip()}
    labels = {k: v for k, v in labels.items() if k not in excluded}
    roots = [args.pdbbind_dir / "refined-set", args.pdbbind_dir / "v2020-other-PL"]
    logger.info(
        "PDBbind labels (%s, CASF excluded): %d complexes",
        ",".join(sorted(kinds)),
        len(labels),
    )

    pdbs = sorted(labels)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pdbs)
    val_pdbs = set(pdbs[: int(len(pdbs) * args.val_frac)])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "train": _RmsdWriter(args.out_dir, "train"),
        "val": _RmsdWriter(args.out_dir, "val"),
    }

    from tqdm import tqdm  # noqa: PLC0415

    n_ok = n_nodir = n_badlig = n_bounds = n_setup = 0
    pks: list[float] = []
    for pdb in tqdm(pdbs, desc="complexes"):
        cdir = _find_complex(roots, pdb)
        if cdir is None:
            n_nodir += 1
            continue
        try:
            mols = parse_sdf(cdir / f"{pdb}_ligand.sdf")
            if not mols:
                n_badlig += 1
                continue
            mol = mols[0]
            heavy = np.array(
                [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"], np.float32
            )
            if not (args.min_heavy <= heavy.shape[0] <= args.max_heavy):
                n_bounds += 1
                continue
            protein_text = (cdir / f"{pdb}_protein.pdb").read_text()
            setup = enc.setup_pocket(protein_text, heavy)
            if setup is None:
                n_setup += 1
                continue
            p_codes, frame = setup
            split = "val" if pdb in val_pdbs else "train"
            # Val stays at one orientation: it measures the model, not the
            # augmentation. The rotations of one complex are written
            # consecutively, which is what RescoreDataset's
            # rotations_per_complex indexing assumes.
            n_rot = args.num_rotations if split == "train" else 1
            descs = enc.ligand_descs([mol], frame)
            wrote = 0
            for k in range(max(n_rot, 1)):
                # k == 0 is the unrotated complex, so --num-rotations 1
                # reproduces the original corpus byte for byte.
                rot = None if k == 0 else random_rotation_matrix(rot_rng)
                codes = p_codes if rot is None else enc.pocket_codes_rotated(rot)
                seq = enc.seqs_from_descs(codes, descs, rotation=rot)[0]
                if seq is None:
                    continue
                writers[split].write(seq, float(labels[pdb]))
                wrote += 1
            if wrote == 0:
                n_badlig += 1
                continue
            pks.append(labels[pdb])
            n_ok += 1
        except Exception:
            logger.exception("failed %s", pdb)
            continue

    meta = {
        "pocket_distance": args.pocket_distance,
        "max_residues": args.max_residues,
        "vocab_size": AtomLMVocab(codebook_size=args.codebook_size).vocab_size,
        "atom_codebook_size": args.codebook_size,
        "source": "pdbbind2020_affinity_pk",
        "label": "pK (-log10 molar)",
        "affinity_types": sorted(kinds),
        "num_rotations": args.num_rotations,
        "n_complexes": n_ok,
        "train_docs": writers["train"].num_docs,
        "val_docs": writers["val"].num_docs,
        "max_len": max(writers["train"].max_len, writers["val"].max_len),
    }
    for w in writers.values():
        w.close()
    torch.save(meta, args.out_dir / "meta.pt")
    a = np.array(pks) if pks else np.zeros(1)
    logger.info(
        "DONE: %d ok (no-dir %d, bad-lig %d, bounds %d, setup %d) -> "
        "train %d / val %d | pK med %.2f std %.2f | max_len %d",
        n_ok,
        n_nodir,
        n_badlig,
        n_bounds,
        n_setup,
        writers["train"].num_docs,
        writers["val"].num_docs,
        float(np.median(a)),
        float(a.std()),
        meta["max_len"],
    )


if __name__ == "__main__":
    main()
