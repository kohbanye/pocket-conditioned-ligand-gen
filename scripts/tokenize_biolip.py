"""Tokenize BioLIP2 holo complexes into all-atom LM sequences.

BioLIP2 (~989k biologically-relevant interaction sites over ~200k PDBs -- far
more distinct pockets than PLINDER's ~300k or CrossDocked's ~2.9k) is streamed
from the ``.tar.bz2`` receptor/ligand buckets downloaded by
``scripts/download_biolip.py`` (inode-safe; never extracted). This mirrors
``scripts/tokenize_plinder_protein.py`` -- CPU pocket/descriptor extraction is
multiprocessed one worker per bucket, GPU atom-VQ encoding runs in the parent --
but reads BioLIP's PDB ligands (bond orders recovered from the CCD template
SMILES) instead of PLINDER SDFs.

Membership comes from ``BioLiP.txt`` (col1 PDB, col2 receptor chain, col5 ligand
CCD, col6 ligand chain, col7 serial). Files inside a bucket keyed by the middle
two PDB chars: receptor ``{pdb}{chain}.pdb``, ligand
``{pdb}_{ccd}_{ligchain}_{serial}.pdb``. Sites whose PDB is a CrossDocked
fold-0 TEST receptor or a CASF-2016 core-set PDB are dropped (leakage). A
per-PDB train/val split keeps all sites of a PDB on one side.

Run (single GPU)::

    uv run python scripts/tokenize_biolip.py --complex \
        --ckpt "<atom-vqvae>.ckpt" \
        --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
        --casf-pdbs data/casf2016_pdbs.txt \
        --num-rotations 2 --out-dir data/lm_tokens_complex_biolip
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import tarfile
from pathlib import Path

import numpy as np
import torch

from prolit.config import AtomVQVAETrainingConfig, PocketExtractionConfig
from prolit.data.descriptors import collate_molecules
from prolit.data.token_io import SplitWriter
from prolit.model.vqvae_module import AtomVQVAEModule
from prolit.tokenizers.atom import (
    LigandAtomDescriptor,
    ProteinAtomDescriptor,
    precompute_receptor_atom_features_from_text,
    rotate_atom_descriptor,
)
from prolit.tokenizers.geometry import random_rotation_matrix
from prolit.tokenizers.ligand import parse_ligand_pdb_text
from prolit.tokenizers.lm_vocab import AtomLMVocab
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
_w_labels: dict[str, str] = {}
_w_ccd_smiles: dict[str, str] = {}
_w_biolip_dir: Path = Path("data/biolip")
_w_complex: bool = True
_w_min_heavy: int = 6
_w_max_heavy: int = 50


def _bucket_code(pdb: str) -> str:
    """Middle two chars of the 4-char PDB id (e.g. '101m' -> '01')."""
    return pdb[1:3]


def _parse_biolip_txt(path: Path) -> list[tuple[str, str, str, str, str]]:
    """(pdb, rchain, ccd, ligchain, serial) per biologically-relevant site."""
    sites: list[tuple[str, str, str, str, str]] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 7:  # noqa: PLR2004
                continue
            # col1 PDB, col2 receptor chain, col5 ligand CCD, col6 ligand chain,
            # col7 serial.
            sites.append((c[0].lower(), c[1], c[4], c[5], c[6]))
    return sites


def _load_ccd_smiles(path: Path) -> dict[str, str]:
    """CCD 3-letter id -> template SMILES (semicolon-joined variants)."""
    out: dict[str, str] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) >= 5 and c[0]:  # noqa: PLR2004
                out[c[0]] = c[4]
    return out


def _cd_test_pdbs(cd_manifest: Path) -> set[str]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    cd = pq.read_table(
        cd_manifest, columns=["receptor_pdb", "source_type", "cdonly_fold0"]
    ).to_pandas()
    cd = cd[(cd["source_type"] == "cdonly") & (cd["cdonly_fold0"] == "test")]
    return set(
        cd["receptor_pdb"].str.extract(r"^([0-9a-zA-Z]{4})_")[0].str.lower().dropna()
    )


def _assign_splits(
    sites: list[tuple],
    excluded: set[str],
    *,
    val_frac: float,
    seed: int,
    dedup: bool,
) -> tuple[dict[str, list[tuple]], dict[str, str]]:
    """Group kept sites by bucket + assign each PDB wholly to train or val.

    With ``dedup`` (default), keep one site per ``(pdb, ccd)`` -- the same ligand
    in multiple chains of a homo-oligomer is a near-identical binding mode
    (crystallographic copy), a 3.88x redundancy that only slows the single-GPU
    encoder without adding pocket diversity.
    """
    pdbs = sorted({s[0] for s in sites if s[0] not in excluded})
    rng = np.random.default_rng(seed)
    rng.shuffle(pdbs)
    n_val = int(len(pdbs) * val_frac)
    val_pdbs = set(pdbs[:n_val])
    labels = {p: ("val" if p in val_pdbs else "train") for p in pdbs}

    by_bucket: dict[str, list[tuple]] = {}
    seen: set[tuple[str, str]] = set()
    kept = 0
    for s in sites:
        if s[0] in excluded or s[0] not in labels:
            continue
        if dedup:
            key = (s[0], s[2])  # (pdb, ccd)
            if key in seen:
                continue
            seen.add(key)
        by_bucket.setdefault(_bucket_code(s[0]), []).append(s)
        kept += 1
    logger.info(
        "BioLIP kept: %d over %d PDBs (val: %d); excluded: %d; dedup=%s",
        kept,
        len(pdbs),
        n_val,
        len(excluded),
        dedup,
    )
    return by_bucket, labels


def _worker_init(  # noqa: PLR0913
    pocket_config_dict: dict,
    labels: dict[str, str],
    ccd_smiles: dict[str, str],
    biolip_dir: str,
    *,
    complex_mode: bool,
    min_heavy: int,
    max_heavy: int,
) -> None:
    global _w_prot_desc, _w_lig_desc, _w_pocket_config  # noqa: PLW0603
    global _w_labels, _w_ccd_smiles, _w_biolip_dir  # noqa: PLW0603
    global _w_complex, _w_min_heavy, _w_max_heavy  # noqa: PLW0603
    _w_prot_desc = ProteinAtomDescriptor()
    _w_lig_desc = LigandAtomDescriptor()
    _w_pocket_config = PocketExtractionConfig(**pocket_config_dict)
    _w_labels = labels
    _w_ccd_smiles = ccd_smiles
    _w_biolip_dir = Path(biolip_dir)
    _w_complex = complex_mode
    _w_min_heavy = min_heavy
    _w_max_heavy = max_heavy


def _read_needed(kind: str, code: str, needed: set[str]) -> dict[str, bytes]:
    """Stream one ``.tar.bz2`` bucket ONCE, extracting only ``needed`` basenames.

    Streaming mode (``r|bz2``) is a single sequential decompression: extract
    each wanted member as it is encountered and stop once all are found. The
    seekable ``r:bz2`` + ``getmembers``/``extractfile`` path re-decompresses the
    whole stream per file (O(n^2)) and reads every member -- catastrophic on the
    dense buckets (thousands of PDBs share a 2-char code).
    """
    path = _w_biolip_dir / kind / f"{kind}_{code}.tar.bz2"
    members: dict[str, bytes] = {}
    if not path.exists() or not needed:
        return members
    try:
        with tarfile.open(path, mode="r|bz2") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                bn = Path(m.name).name
                if bn in needed:
                    f = tf.extractfile(m)
                    if f is not None:
                        members[bn] = f.read()
                    if len(members) == len(needed):
                        break
    except Exception:
        logger.exception("Bad bucket %s/%s", kind, code)
    return members


def _process_bucket(task: tuple[str, list[tuple]]) -> list[tuple]:  # noqa: C901, PLR0912
    """Extract (label, prot_desc, lig_desc) for kept sites in one bucket."""
    code, site_list = task
    out: list[tuple] = []
    if not site_list:
        return out
    needed_rec = {f"{p}{rc}.pdb" for p, rc, _c, _l, _s in site_list}
    needed_lig = {f"{p}_{cc}_{lc}_{s}.pdb" for p, _rc, cc, lc, s in site_list}
    receptors = _read_needed("receptor", code, needed_rec)
    ligands = _read_needed("ligand", code, needed_lig)
    if not receptors or not ligands:
        return out

    # Parse each unique receptor PDB (BioPython/RDKit, ~100ms) ONCE per bucket;
    # dense buckets have many sites sharing a receptor chain, so re-parsing per
    # site was the dominant cost.
    rec_cache: dict[str, tuple | None] = {}
    for pdb, rchain, ccd, ligchain, serial in site_list:
        label = _w_labels.get(pdb)
        if label is None:
            continue
        rec_name = f"{pdb}{rchain}.pdb"
        rec_bytes = receptors.get(rec_name)
        lig_bytes = ligands.get(f"{pdb}_{ccd}_{ligchain}_{serial}.pdb")
        if rec_bytes is None or lig_bytes is None:
            continue
        try:
            mol = parse_ligand_pdb_text(
                lig_bytes.decode("utf-8", "replace"), _w_ccd_smiles.get(ccd)
            )
            if mol is None:
                continue
            heavy = np.array(
                [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"],
                dtype=np.float32,
            )
            if not (_w_min_heavy <= heavy.shape[0] <= _w_max_heavy):
                continue
            if rec_name not in rec_cache:
                rec_text = rec_bytes.decode("utf-8", "replace")
                rec_cache[rec_name] = (
                    precompute_pocket_atom_candidates_from_text(rec_text),
                    precompute_receptor_atom_features_from_text(rec_text),
                )
            if rec_cache[rec_name] is None:
                continue
            precomp, feats = rec_cache[rec_name]
            pocket = extract_pocket_atoms_from_candidates(
                precomp, heavy, _w_pocket_config
            )
            if pocket is None or pocket.atom_coords.shape[0] == 0:
                continue
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
            logger.exception("Error on %s_%s_%s_%s", pdb, ccd, ligchain, serial)
            continue
        out.append((label, prot_desc, lig_desc))
    return out


class _Encoder:
    """Buffers (prot, lig|None) rows and flushes them through the atom VQ-VAE."""

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
        self._prot: dict[str, list[torch.Tensor]] = {s: [] for s in writers}
        self._lig: dict[str, list[torch.Tensor | None]] = {s: [] for s in writers}

    def _norm(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((arr - self.mean) / self.std).float()

    def add(self, split: str, prot: np.ndarray, lig: np.ndarray | None) -> None:
        if split not in self._prot:
            return
        self._prot[split].append(self._norm(prot))
        self._lig[split].append(self._norm(lig) if lig is not None else None)
        if len(self._prot[split]) >= self.batch_size:
            self.flush(split)

    def _encode(self, descs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x, mask = collate_molecules(descs)
        idx = self.module.vqvae.encode_batch(
            x.to(self.device), mask.to(self.device)
        ).cpu()
        return idx, mask

    def flush(self, split: str) -> None:
        prot = self._prot[split]
        if not prot:
            return
        pidx, pmask = self._encode(prot)
        ligs = self._lig[split]
        if all(x is None for x in ligs):
            seqs = [
                self.vocab.build_sequence(pidx[i][pmask[i]].tolist(), [])
                for i in range(len(prot))
            ]
        else:
            lidx, lmask = self._encode([x for x in ligs if x is not None])
            seqs = [
                self.vocab.build_sequence(
                    pidx[i][pmask[i]].tolist(), lidx[i][lmask[i]].tolist()
                )
                for i in range(len(prot))
            ]
        self.writers[split].write(seqs)
        prot.clear()
        ligs.clear()

    def flush_all(self) -> None:
        for split in list(self._prot):
            self.flush(split)


def main() -> None:  # noqa: PLR0915, PLR0912, C901
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
    parser.add_argument("--biolip-dir", type=Path, default=Path("data/biolip"))
    parser.add_argument(
        "--cd-manifest", type=Path, default=Path("data/hub_cache/repo/manifest.parquet")
    )
    parser.add_argument(
        "--casf-pdbs",
        type=Path,
        default=None,
        help="Text file of CASF-2016 core-set PDB ids (one per line) to exclude.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/lm_tokens_complex_biolip")
    )
    parser.add_argument(
        "--complex",
        action="store_true",
        help="Emit <p>pocket</p><l>ligand</l> (default: protein-only <l></l>).",
    )
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--max-residues", type=int, default=50)
    parser.add_argument("--num-rotations", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-heavy", type=int, default=6)
    parser.add_argument("--max-heavy", type=int, default=50)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bucket-timeout",
        type=int,
        default=180,
        help="Skip a bucket whose worker exceeds this many seconds (a hung "
        "RDKit template match). The densest legit bucket takes ~34s.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Keep every site (default: one per (pdb, ccd), a 3.88x reduction).",
    )
    parser.add_argument("--max-buckets", type=int, default=None, help="Debug subset.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AtomVQVAETrainingConfig()
    config.atom.codebook_size = args.codebook_size
    if args.separate_protein_ckpt is not None:
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

    sites = _parse_biolip_txt(args.biolip_dir / "BioLiP.txt.gz")
    ccd_smiles = _load_ccd_smiles(args.biolip_dir / "ligand.tsv.gz")
    excluded = _cd_test_pdbs(args.cd_manifest)
    if args.casf_pdbs is not None and args.casf_pdbs.exists():
        casf = {p.lower() for p in args.casf_pdbs.read_text().split() if p.strip()}
        excluded |= casf
        logger.info("CASF-2016 exclusion: %d PDBs", len(casf))
    else:
        logger.warning("No --casf-pdbs given: CASF-2016 NOT excluded (leak risk).")

    by_bucket, labels = _assign_splits(
        sites,
        excluded,
        val_frac=args.val_frac,
        seed=args.seed,
        dedup=not args.no_dedup,
    )
    codes = sorted(by_bucket)
    if args.max_buckets is not None:
        codes = codes[: args.max_buckets]
    logger.info("Processing %d BioLIP buckets (complex=%s)", len(codes), args.complex)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {s: SplitWriter(args.out_dir, s) for s in ("train", "val")}
    enc = _Encoder(module, vocab, mean, std, writers, args.batch_size, device)
    rng = np.random.default_rng(args.seed)
    n_used = 0

    def _consume(results: list[tuple]) -> None:
        nonlocal n_used
        for label, prot, lig in results:
            n_used += 1
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

    from dataclasses import asdict  # noqa: PLC0415

    from tqdm import tqdm  # noqa: PLC0415

    pocket_cfg = PocketExtractionConfig(max_residues=args.max_residues)
    # Each task carries only its bucket's site list (small), NOT the whole
    # 958k-site dict -- avoids the per-worker dict copy (the 1.195T vmem OOM).
    tasks = [(c, by_bucket[c]) for c in codes]
    init_args = (asdict(pocket_cfg), labels, ccd_smiles, str(args.biolip_dir))
    init_kwargs = {
        "complex_mode": args.complex,
        "min_heavy": args.min_heavy,
        "max_heavy": args.max_heavy,
    }
    if args.num_workers > 0:
        import functools  # noqa: PLC0415
        import multiprocessing  # noqa: PLC0415

        # apply_async + per-bucket get(timeout) so a single pathological ligand
        # (RDKit AssignBondOrdersFromTemplate can hang on highly symmetric mols)
        # can't stall the whole job -- the hung bucket is skipped, the run still
        # finishes cleanly (writer.close() + meta). imap_unordered offered no
        # such escape hatch and a lone hang blocked it until the h_rt wall.
        n_timeout = 0
        with multiprocessing.Pool(
            args.num_workers,
            initializer=functools.partial(_worker_init, **init_kwargs),
            initargs=init_args,
        ) as pool:
            asyncs = [
                (t[0], pool.apply_async(_process_bucket, (t,))) for t in tasks
            ]
            for code, ar in tqdm(asyncs, total=len(asyncs), desc="buckets"):
                try:
                    _consume(ar.get(timeout=args.bucket_timeout))
                except multiprocessing.TimeoutError:
                    n_timeout += 1
                    logger.warning("bucket %s timed out; skipping", code)
        if n_timeout:
            logger.warning("skipped %d timed-out buckets", n_timeout)
    else:
        _worker_init(*init_args, **init_kwargs)
        for task in tqdm(tasks, desc="buckets"):
            _consume(_process_bucket(task))

    enc.flush_all()
    meta: dict = {
        "vocab_size": vocab.vocab_size,
        "all_atom": True,
        "pretrain": {
            "source": "biolip2",
            "mode": "complex" if args.complex else "protein_only",
            "num_rotations": args.num_rotations,
            "systems_used": n_used,
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
    logger.info("Wrote BioLIP token cache to %s", args.out_dir)


if __name__ == "__main__":
    main()
