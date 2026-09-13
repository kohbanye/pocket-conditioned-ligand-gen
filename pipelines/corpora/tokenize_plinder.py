"""Tokenize PLINDER holo pockets into all-atom LM sequences.

Two modes over the same PLINDER stream (inode-safe; never extracts the zips):

- **protein-only** (default): ``<bos><p> pocket </p><l></l><eos>`` -- the
  p(pocket) side of the mixed pretraining corpus (>300k holo pockets, far more
  than CrossDocked's ~1.7k).
- **--complex**: ``<bos><p> pocket </p><l> ligand </l><eos>`` -- pocket-LIGAND
  pairs for the conditional fine-tune. PLINDER's ~300k distinct pockets (vs
  CrossDocked's ~2.9k) directly attack the pocket-generalisation over-fitting;
  a drug-like MW / heavy-atom filter keeps the ligand distribution sane.

For each kept system: parse ``receptor.pdb`` + ``ligand_files/*.sdf``, take the
largest ligand molecule, extract the all-atom pocket (8 A / <=50 res) around it,
and atom-VQ tokenize (CPU extraction multiprocessed one worker per zip; GPU
encode in the parent). Rotation augmentation. Train/val routed by PLINDER split.
Systems whose PDB is a CrossDocked fold-0 TEST receptor are dropped (leakage).

Run (single GPU)::

    uv run python pipelines/corpora/tokenize_plinder.py --complex \
        --ckpt "<atom-vqvae>.ckpt" \
        --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
        --num-rotations 2 --out-dir data/lm_tokens_complex_plinder
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prolit.config import AtomVQVAETrainingConfig, PocketExtractionConfig
from prolit.data.token_io import SplitWriter
from prolit.data.token_stream import ComplexTokenEncoder
from prolit.seeding import add_seed_argument, seed_from_args
from prolit.tokenizers.atom import (
    LigandAtomDescriptor,
    ProteinAtomDescriptor,
    precompute_receptor_atom_features_from_text,
    rotate_atom_descriptor,
)
from prolit.tokenizers.geometry import random_rotation_matrix
from prolit.tokenizers.ligand import parse_sdf_text
from prolit.tokenizers.lm_vocab import AtomLMVocab
from prolit.tokenizers.loaders import load_atom_vqvae
from prolit.tokenizers.protein import (
    compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    precompute_pocket_atom_candidates_from_text,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Per-worker state.
_w_prot_desc: ProteinAtomDescriptor | None = None
_w_lig_desc: LigandAtomDescriptor | None = None
_w_pocket_config: PocketExtractionConfig | None = None
_w_allowed: dict[str, str] = {}
_w_complex: bool = False
_w_min_heavy: int = 6
_w_max_heavy: int = 60
#: The whole stapled encoding is CPU work -- ESM3's tokens are already
#: cached and ConfSeq is not needed at all in protein-only mode -- so the
#: worker produces the finished token stream rather than a descriptor for
#: a GPU pass that would have nothing to do.
_w_stapled: object | None = None


def _system_pdb(system_id: str) -> str:
    """PLINDER system_id -> 4-char PDB id (e.g. '1abc__1__1.A__1.B' -> '1abc')."""
    return system_id.split("__", 1)[0].lower()


def _load_allowed_systems(  # noqa: PLR0913
    split_path: Path,
    cd_manifest: Path,
    *,
    complex_mode: bool,
    mw_min: float,
    mw_max: float,
    casf_pdbs: set[str] | None = None,
) -> dict[str, str]:
    """Map kept ``system_id -> split`` ('train'/'val'), leak-filtered + drug-like.

    ``casf_pdbs`` (lowercase 4-char PDB ids) are dropped so the CASF-2016 core is
    held out of pretraining -- required for an honest rescoring benchmark, since
    continuing to train on them causes native-pose memorization.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    casf = casf_pdbs or set()
    sp = pq.read_table(split_path).to_pandas()
    split_col = (
        "split"
        if "split" in sp.columns
        else next(c for c in sp.columns if "split" in c.lower())
    )
    sid_col = "system_id" if "system_id" in sp.columns else sp.columns[0]
    mw_col = "system_proper_ligand_max_molecular_weight"

    cd = pq.read_table(
        cd_manifest, columns=["receptor_pdb", "source_type", "cdonly_fold0"]
    ).to_pandas()
    cd = cd[(cd["source_type"] == "cdonly") & (cd["cdonly_fold0"] == "test")]
    cd_test_pdbs = set(
        cd["receptor_pdb"].str.extract(r"^([0-9a-zA-Z]{4})_")[0].str.lower().dropna()
    )

    allowed: dict[str, str] = {}
    n_leak = n_mw = n_casf = 0
    for row in sp.itertuples(index=False):
        label = getattr(row, split_col)
        if label not in ("train", "val"):
            continue
        sid = str(getattr(row, sid_col))
        if _system_pdb(sid) in casf:
            n_casf += 1
            continue
        if _system_pdb(sid) in cd_test_pdbs:
            n_leak += 1
            continue
        if complex_mode and mw_col in sp.columns:
            mw = getattr(row, mw_col, None)
            if mw is None or not (mw_min <= float(mw) <= mw_max):
                n_mw += 1
                continue
        allowed[sid] = label
    logger.info(
        "PLINDER kept: %d | CASF-dropped: %d | CD-leak-dropped: %d | "
        "MW-filtered: %d (complex=%s)",
        len(allowed),
        n_casf,
        n_leak,
        n_mw,
        complex_mode,
    )
    return allowed


def _worker_init(  # noqa: PLR0913
    pocket_config_dict: dict,
    allowed: dict[str, str],
    *,
    complex_mode: bool,
    min_heavy: int,
    max_heavy: int,
    stapled: dict | None = None,
) -> None:
    global _w_prot_desc, _w_lig_desc, _w_pocket_config  # noqa: PLW0603
    global _w_allowed, _w_complex, _w_min_heavy, _w_max_heavy  # noqa: PLW0603
    global _w_stapled  # noqa: PLW0603
    _w_prot_desc = ProteinAtomDescriptor()
    _w_lig_desc = LigandAtomDescriptor()
    _w_pocket_config = PocketExtractionConfig(**pocket_config_dict)
    _w_allowed = allowed
    _w_complex = complex_mode
    _w_min_heavy = min_heavy
    _w_max_heavy = max_heavy
    _w_stapled = None
    if stapled:
        from prolit.data.esm3_tokens import Esm3TokenCache  # noqa: PLC0415
        from prolit.tokenizers.stapled import (  # noqa: PLC0415
            ConfSeqVocab,
            StapledVocab,
        )
        from prolit.tokenizers.stapled_encoder import (  # noqa: PLC0415
            StapledEncoder,
        )

        _w_stapled = StapledEncoder(
            cache=Esm3TokenCache(Path(stapled["cache"])),
            confseq_repo=Path(stapled["confseq_repo"]),
            vocab=StapledVocab(confseq=ConfSeqVocab.load(Path(stapled["vocab"]))),
            pocket_cfg=_w_pocket_config,
        )


def _largest_ligand(zf: zipfile.ZipFile, lig_members: list[str]) -> dict | None:
    """Parse all ligand SDFs; return the molecule with the most heavy atoms."""
    best = None
    best_n = 0
    for m in lig_members:
        text = zf.read(m).decode("utf-8", "replace")
        for mol in parse_sdf_text(text):
            n = sum(1 for a in mol["atoms"] if a[0] != "H")
            if n > best_n:
                best_n, best = n, mol
    return best


def _process_zip(zip_path: str) -> list[tuple]:  # noqa: C901, PLR0912, PLR0915
    """Extract (label, prot_desc-or-stapled-stream, lig_desc|None) per system."""
    out: list[tuple] = []
    try:
        zf = zipfile.ZipFile(zip_path)
    except Exception:
        logger.exception("Bad zip %s", zip_path)
        return out

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
            mol = _largest_ligand(zf, ligs)
            if mol is None:
                continue
            heavy = np.array(
                [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"],
                dtype=np.float32,
            )
            if _w_complex and not (_w_min_heavy <= heavy.shape[0] <= _w_max_heavy):
                continue
            if heavy.shape[0] == 0:
                continue
            rec_text = zf.read(rec).decode("utf-8", "replace")
            precomp = precompute_pocket_atom_candidates_from_text(rec_text)
            if _w_stapled is not None:
                # Protein-only, and protein-only is all this arm can be here:
                # ProLIT's PLINDER corpus writes no ligand either. The ligand is
                # still what CHOOSES the pocket, in both arms, which is why the
                # heavy atoms above are computed before this branch.
                spocket = _w_stapled.setup_pocket_precomputed(sid, precomp, heavy)
                if spocket is None:
                    continue
                seq = _w_stapled.vocab.build_sequence(spocket.codes, [], None)
                out.append((label, seq, None))
                continue
            pocket = extract_pocket_atoms_from_candidates(
                precomp, heavy, _w_pocket_config
            )
            if pocket is None or pocket.atom_coords.shape[0] == 0:
                continue
            feats = precompute_receptor_atom_features_from_text(rec_text)
            frame = compute_canonical_frame(pocket.ca_coords.astype(np.float64))
            prot_desc, _pm = _w_prot_desc.compute(pocket, feats, frame)
            if prot_desc.shape[0] == 0:
                continue
            lig_desc = None
            if _w_complex:
                lig_desc, _e, _lm = _w_lig_desc.compute(
                    mol["atoms"], mol["bonds"], frame
                )
                if lig_desc.shape[0] == 0:
                    continue
        except Exception:
            logger.exception("Error on system %s in %s", sid, zip_path)
            continue
        out.append((label, prot_desc, lig_desc))
    zf.close()
    return out


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
    parser.add_argument("--norm-stats", type=Path, default=None)
    parser.add_argument(
        "--stapled-esm3-cache",
        type=Path,
        default=None,
        help="Cache of ESM3 structure tokens keyed by PLINDER system id "
        "(pipelines/corpora/esm3_structure_tokens.py). Switches the builder to "
        "the ESM3 x ConfSeq baseline: no VQ-VAE, no GPU, no rotations. In "
        "protein-only mode ConfSeq is not used at all -- there is no ligand to "
        "write -- so only the vocabulary's layout is taken from it.",
    )
    parser.add_argument("--stapled-vocab", type=Path, default=None)
    parser.add_argument(
        "--zip-timeout",
        type=int,
        default=1800,
        help="seconds one zip may take before it is abandoned. The biggest "
        "PLINDER zip (1418 systems) measures 193 s, so this is a wide margin "
        "whose job is to bound a worker that has died, not a slow one.",
    )
    parser.add_argument(
        "--confseq-repo", type=Path, default=Path("third_party/ConfSeq")
    )
    parser.add_argument(
        "--systems-dir", type=Path, default=Path("data/plinder/systems")
    )
    parser.add_argument(
        "--split", type=Path, default=Path("data/plinder/split.parquet")
    )
    parser.add_argument(
        "--cd-manifest",
        type=Path,
        default=Path("data/hub_cache/repo/manifest.parquet"),
    )
    parser.add_argument(
        "--casf-pdbs",
        type=Path,
        default=None,
        help="Newline-separated PDB ids to hold out of pretraining. The name "
        "says CASF, but the published corpus passed data/eval_holdout_pdbs.txt "
        "(9329 ids), NOT data/casf2016_pdbs.txt (285) -- the CASF core is a "
        "subset of the evaluation holdout. Passing the narrow list, or nothing "
        "(the default), trains on thousands of evaluation structures and no "
        "message says so.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/lm_tokens_protein_plinder")
    )
    parser.add_argument(
        "--complex",
        action="store_true",
        help="Emit <p>pocket</p><l>ligand</l> (default: protein-only <l></l>).",
    )
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--max-residues", type=int, default=50)
    parser.add_argument("--num-rotations", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mw-min", type=float, default=150.0)
    parser.add_argument("--mw-max", type=float, default=600.0)
    parser.add_argument("--min-heavy", type=int, default=6)
    parser.add_argument("--max-heavy", type=int, default=60)
    add_seed_argument(parser, default=0)
    parser.add_argument("--max-zips", type=int, default=None, help="Debug subset.")
    args = parser.parse_args()
    seed_from_args(args)

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stapled_mode = args.stapled_esm3_cache is not None
    stapled_cfg = (
        {
            "cache": str(args.stapled_esm3_cache),
            "confseq_repo": str(args.confseq_repo),
            "vocab": str(args.stapled_vocab),
        }
        if stapled_mode
        else None
    )
    config = AtomVQVAETrainingConfig()
    config.atom.codebook_size = args.codebook_size
    if stapled_mode:
        # No VQ-VAE, so no normalization statistics and nothing to put on a GPU.
        from prolit.tokenizers.stapled import (  # noqa: PLC0415
            ConfSeqVocab,
            StapledVocab,
        )

        module = mean = std = None
        vocab: Any = StapledVocab(confseq=ConfSeqVocab.load(args.stapled_vocab))
    elif args.separate_protein_ckpt is not None:
        # ABLATION separate-tokenizers mode: protein-only VQ + ligand-only VQ
        # unified into one code space. Feed RAW descriptors (identity external
        # norm) -- SeparateVQVAE normalizes per modality internally. Combined
        # single-range AtomLMVocab over 2*codebook_size codes.
        from prolit.tokenizers.descriptor_schema import (  # noqa: PLC0415
            ATOM_DESCRIPTOR_DIM,
        )
        from prolit.tokenizers.separate_vqvae import SeparateVQVAE  # noqa: PLC0415

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
        vocab = AtomLMVocab(codebook_size=2 * args.codebook_size)
    else:
        module = load_atom_vqvae(args.ckpt, device)
        module.eval()
        module.to(device)
        norm_stats = torch.load(args.norm_stats, weights_only=False)
        module.vqvae.set_normalization(norm_stats["atom_mean"], norm_stats["atom_std"])
        mean = norm_stats["atom_mean"].numpy()
        std = norm_stats["atom_std"].numpy()
        vocab = AtomLMVocab(codebook_size=args.codebook_size)

    casf_pdbs = None
    if args.casf_pdbs is not None and args.casf_pdbs.exists():
        casf_pdbs = {
            p.strip().lower() for p in args.casf_pdbs.read_text().split() if p.strip()
        }
    allowed = _load_allowed_systems(
        args.split,
        args.cd_manifest,
        complex_mode=args.complex,
        mw_min=args.mw_min,
        mw_max=args.mw_max,
        casf_pdbs=casf_pdbs,
    )
    zips = sorted(str(p) for p in args.systems_dir.glob("*.zip"))
    if args.max_zips is not None:
        zips = zips[: args.max_zips]
    logger.info("Streaming %d PLINDER zips (complex=%s)", len(zips), args.complex)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {s: SplitWriter(args.out_dir, s) for s in ("train", "val")}
    enc = (
        None
        if stapled_mode
        else ComplexTokenEncoder(
            module.vqvae, vocab, mean, std, writers, args.batch_size, device
        )
    )
    rng = np.random.default_rng(args.seed)

    from dataclasses import asdict  # noqa: PLC0415

    pocket_cfg = PocketExtractionConfig(max_residues=args.max_residues)
    n_used = 0

    def _consume(results: list[tuple]) -> None:
        nonlocal n_used
        for label, prot, lig in results:
            n_used += 1
            if stapled_cfg is not None:
                # The worker already produced the finished stream, and no
                # rotation augmentation: ESM3's codes are invariant, so a
                # rotated copy is a duplicate document rather than a view.
                writers[label].write([prot])
                continue
            n_rot = args.num_rotations if label == "train" else 1
            for r in range(n_rot):
                if r == 0:
                    enc.add(label, prot, lig)
                else:
                    rot = random_rotation_matrix(rng)
                    enc.add(
                        label,
                        rotate_atom_descriptor(prot, rot),
                        rotate_atom_descriptor(lig, rot) if lig is not None else None,
                    )

    from tqdm import tqdm  # noqa: PLC0415

    init_args = (asdict(pocket_cfg), allowed)
    init_kwargs = {
        "complex_mode": args.complex,
        "min_heavy": args.min_heavy,
        "max_heavy": args.max_heavy,
        "stapled": stapled_cfg,
    }
    if args.num_workers > 0:
        import functools  # noqa: PLC0415
        import multiprocessing  # noqa: PLC0415

        # apply_async + get(timeout), not imap_unordered. A Pool does not
        # notice a worker that dies -- and these workers CAN die: the ESM3
        # shard cache decompresses ~18x, so an unbounded one in 40 processes
        # asked for 141 GB on a 23 GB node, the OOM killer took them, and
        # imap_unordered then blocked on results that would never arrive. The
        # run sat for 78 minutes with a frozen CPU counter and no output, and
        # died at its walltime having written no meta.json. The cache is
        # bounded now; this is so the next unforeseen death costs one zip.
        n_timeout = 0
        with multiprocessing.Pool(
            args.num_workers,
            initializer=functools.partial(_worker_init, **init_kwargs),
            initargs=init_args,
        ) as pool:
            pending = [(z, pool.apply_async(_process_zip, (z,))) for z in zips]
            for zp, ar in tqdm(pending, total=len(zips), desc="zips"):
                try:
                    _consume(ar.get(timeout=args.zip_timeout))
                except multiprocessing.TimeoutError:
                    n_timeout += 1
                    logger.warning("zip %s exceeded %ds; skipped", zp, args.zip_timeout)
        if n_timeout:
            logger.warning(
                "skipped %d of %d zips on the time budget", n_timeout, len(zips)
            )
    else:
        _worker_init(*init_args, **init_kwargs)
        for zp in tqdm(zips, desc="zips"):
            _consume(_process_zip(zp))

    if enc is not None:
        enc.flush_all()
    meta: dict = {
        "vocab_size": vocab.vocab_size,
        "all_atom": True,
        "pretrain": {
            "source": "plinder",
            "mode": "complex" if args.complex else "protein_only",
            # What was applied, not what was asked for: the stapled path
            # writes one document per system and ignores --num-rotations.
            "num_rotations": 1 if stapled_mode else args.num_rotations,
            "systems_used": n_used,
        },
        "splits": {},
    }
    if stapled_mode:
        from prolit.tokenizers.lm_vocab import NUM_SPECIAL  # noqa: PLC0415

        meta["stapled"] = {
            "protein_tokenizer": "esm3_structure_v0",
            "ligand_tokenizer": "confseq",
            "mode": "protein_only",
            "esm3_cache": str(args.stapled_esm3_cache),
            "confseq_vocab_path": str(args.stapled_vocab),
        }
        meta["atom_codebook_size"] = vocab.vocab_size - NUM_SPECIAL
        meta["atom_offset"] = NUM_SPECIAL
    else:
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
    logger.info("Wrote PLINDER token cache to %s", args.out_dir)


if __name__ == "__main__":
    main()
