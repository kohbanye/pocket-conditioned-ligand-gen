"""Binding-affinity corpus for the STAPLED baseline (ESM3 x ConfSeq + pose).

The affinity counterpart of ``tokenize_affinity_biolip.py`` and
``tokenize_affinity_pdbbind.py``: the same complexes, the same labels, the same
exclusions and the same train/val split, written with the stapled tokenizer
(:mod:`prolit.tokenizers.stapled`) instead of ProLIT's VQ-VAE. The split is not
re-derived but replayed -- the site parse, the seeded shuffle and the
protein-group val assignment are the ProLIT builders' own code and RNG calls --
so the two arms are selected on the same validation proteins.

Two stages, because ESM3 runs in a different interpreter:

``prep``
    Streams the BioLiP buckets ONCE and keeps the receptor and ligand records of
    every affinity site in one ``sites.jsonl.gz`` (one file, not thousands: the
    group disk has an inode budget), and writes the ESM3 manifests for every
    receptor the existing caches do not cover.

``build``
    Encodes one source (``biolip`` or ``pdbbind``) into a scoring cache with
    the RescoreDataset layout, reading ESM3 codes from one or more caches.

A complex the stapled tokenizer cannot write (no ESM3 code for a pocket
residue, ConfSeq failure, out-of-vocabulary token) is dropped and counted by
reason in ``meta.pt`` -- never replaced by anything that would credit the
tokens with information they do not carry.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import logging
import tarfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenize_affinity_biolip import _parse_affinity_sites
from tokenize_affinity_pdbbind import _find_complex, _read_labels
from tokenize_biolip import _bucket_code, _load_ccd_smiles
from tokenize_decoys import _cd_test_pdbs, _RmsdWriter

from prolit.config import PocketExtractionConfig
from prolit.data.esm3_tokens import Esm3TokenCache
from prolit.seeding import add_seed_argument, rng_for, seed_from_args
from prolit.tokenizers.geometry import random_rotation_matrix
from prolit.tokenizers.stapled import ConfSeqVocab, StapledVocab
from prolit.tokenizers.stapled_encoder import StapledEncoder, StapledPocket

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Site lists, replayed from the ProLIT builders
# ---------------------------------------------------------------------------


def _read_ids(path: Path) -> set[str]:
    return {x.lower() for x in path.read_text().split() if x.strip()}


def biolip_sites(args: argparse.Namespace) -> list[tuple]:
    """The ProLIT BioLiP builder's site list, before its shuffle."""
    kinds = {k.strip().upper() for k in args.affinity_types.split(",") if k.strip()}
    sites = _parse_affinity_sites(
        args.biolip_dir / "BioLiP.txt.gz",
        pk_min=args.pk_min,
        pk_max=args.pk_max,
        kinds=kinds or None,
    )
    excluded = _cd_test_pdbs(args.cd_manifest) | _read_ids(args.biolip_holdout)
    return [s for s in sites if s[0] not in excluded]


def biolip_split(
    sites: list[tuple], seed: int, val_frac: float
) -> tuple[list[tuple], dict[tuple[str, str], int], set[int]]:
    """Shuffle, protein groups and val groups, exactly as the ProLIT builder.

    Every RNG call here is the one ``tokenize_affinity_biolip.main`` makes, in
    the same order, on the same generator, so the val proteins are the same.
    """
    sites = list(sites)
    rng = np.random.default_rng(seed)
    rng.shuffle(sites)
    gid_of: dict[str, int] = {}
    site_gid: dict[tuple[str, str], int] = {}
    for s in sites:
        key = s[6] or f"__nouniprot_{s[0]}_{s[2]}"
        site_gid[(s[0], s[2])] = gid_of.setdefault(key, len(gid_of))
    per_group = Counter(site_gid.values())
    order = rng.permutation(len(gid_of)).tolist()
    target = len(sites) * val_frac
    val_gids: set[int] = set()
    acc = 0
    for g in order:
        if acc >= target:
            break
        val_gids.add(g)
        acc += per_group[g]
    return sites, site_gid, val_gids


def pdbbind_split(
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, float], set[str]]:
    """PDBbind ids, labels and val ids, exactly as the ProLIT builder."""
    kinds = {k.strip().upper() for k in args.affinity_types.split(",") if k.strip()}
    labels = _read_labels(args.pdbbind_dir, ("train", "val", "test"), kinds or None)
    excluded = _read_ids(args.pdbbind_holdout)
    labels = {k: v for k, v in labels.items() if k not in excluded}
    pdbs = sorted(labels)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pdbs)
    val = set(pdbs[: int(len(pdbs) * args.val_frac)])
    return pdbs, labels, val


def _site_key(s: tuple) -> str:
    pdb, rchain, ccd, ligchain, serial = s[:5]
    return f"{pdb}|{rchain}|{ccd}|{ligchain}|{serial}"


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------


def _stream(path: Path, needed: set[str]) -> dict[str, str]:
    """One sequential pass over a ``.tar.bz2`` bucket, keeping ``needed``."""
    out: dict[str, str] = {}
    if not path.exists() or not needed:
        return out
    try:
        with tarfile.open(path, mode="r|bz2") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                bn = Path(m.name).name
                if bn in needed:
                    f = tf.extractfile(m)
                    if f is not None:
                        out[bn] = f.read().decode("utf-8", "replace")
                    if len(out) == len(needed):
                        break
    except Exception:
        logger.exception("bad bucket %s", path)
    return out


def _prep_bucket(task: tuple[str, str, list[tuple]]) -> list[dict]:
    biolip_dir, code, site_list = task
    root = Path(biolip_dir)
    recs = _stream(
        root / "receptor" / f"receptor_{code}.tar.bz2",
        {f"{s[0]}{s[1]}.pdb" for s in site_list},
    )
    ligs = _stream(
        root / "ligand" / f"ligand_{code}.tar.bz2",
        {f"{s[0]}_{s[2]}_{s[3]}_{s[4]}.pdb" for s in site_list},
    )
    out = []
    for s in site_list:
        rec = recs.get(f"{s[0]}{s[1]}.pdb")
        lig = ligs.get(f"{s[0]}_{s[2]}_{s[3]}_{s[4]}.pdb")
        if rec is not None and lig is not None:
            out.append({"key": _site_key(s), "rec": rec, "lig": lig})
    return out


def _cached_ids(dirs: list[Path]) -> set[str]:
    ids: set[str] = set()
    for d in dirs:
        for f in sorted(Path(d).glob("index*.json")):
            ids |= set(json.loads(f.read_text())["ids"])
    return ids


def prep(args: argparse.Namespace) -> None:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    sites = biolip_sites(args)
    by_bucket: dict[str, list[tuple]] = {}
    for s in sites:
        by_bucket.setdefault(_bucket_code(s[0]), []).append(s)
    logger.info("biolip: %d sites in %d buckets", len(sites), len(by_bucket))

    have = _cached_ids(args.esm3_cache)
    tasks = [(str(args.biolip_dir), c, by_bucket[c]) for c in sorted(by_bucket)]
    n_store = 0
    manifest: dict[str, str] = {}
    store = args.work_dir / "biolip_sites.jsonl.gz"
    with gzip.open(store, "wt") as fh, Pool(args.num_workers) as pool:
        for rows in pool.imap_unordered(_prep_bucket, tasks):
            for row in rows:
                fh.write(json.dumps(row) + "\n")
                n_store += 1
                pdb, rchain = row["key"].split("|")[:2]
                sid = f"{pdb}{rchain}"
                if sid not in have and sid not in manifest:
                    manifest[sid] = row["rec"]
    logger.info("site store: %d of %d sites -> %s", n_store, len(sites), store)
    with (args.work_dir / "esm3_manifest_biolip.jsonl").open("w") as fh:
        for sid in sorted(manifest):
            fh.write(json.dumps({"id": sid, "pdb": manifest[sid]}) + "\n")
    logger.info("biolip receptors missing from the ESM3 cache: %d", len(manifest))

    pdbs, _labels, _val = pdbbind_split(args)
    roots = [args.pdbbind_dir / "refined-set", args.pdbbind_dir / "v2020-other-PL"]
    n_pdbbind = 0
    with (args.work_dir / "esm3_manifest_pdbbind.jsonl").open("w") as fh:
        for pdb in sorted(pdbs):
            d = _find_complex(roots, pdb)
            if d is None:
                continue
            fh.write(
                json.dumps({"id": pdb, "path": str(d / f"{pdb}_protein.pdb")}) + "\n"
            )
            n_pdbbind += 1
    logger.info("pdbbind manifest: %d of %d labelled complexes", n_pdbbind, len(pdbs))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


class _TableCache:
    """The one-structure slice of an ESM3 cache a worker needs for one task."""

    def __init__(self, struct_id: str, table: dict[tuple[str, int], int]) -> None:
        self.struct_id = struct_id
        self.table = table

    def pocket_codes(
        self, struct_id: str, residue_ids: list[tuple[str, int]]
    ) -> list[int] | None:
        if struct_id != self.struct_id:
            return None
        out = []
        for key in residue_ids:
            code = self.table.get(key)
            if code is None:
                return None
            out.append(code)
        return out


class _MultiCache:
    """Several ESM3 caches read as one, first hit wins."""

    def __init__(self, dirs: list[Path]) -> None:
        self.caches = [Esm3TokenCache(Path(d), max_shards=1024) for d in dirs]

    def residue_tokens(self, struct_id: str) -> dict[tuple[str, int], int] | None:
        for c in self.caches:
            if struct_id in c:
                return c.residue_tokens(struct_id)
        return None


_W: dict[str, Any] = {}


def _worker_init(
    confseq_repo: str, vocab_path: str, max_residues: int, pocket_distance: str
) -> None:
    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    _W["enc"] = StapledEncoder(
        cache=None,
        confseq_repo=Path(confseq_repo),
        vocab=StapledVocab(confseq=ConfSeqVocab.load(Path(vocab_path))),
        pocket_cfg=PocketExtractionConfig(
            max_residues=max_residues, pocket_distance=pocket_distance
        ),
    )


def _rotated_pocket(pocket: StapledPocket, rot: np.ndarray) -> StapledPocket:
    return StapledPocket(
        codes=pocket.codes,
        backbone=pocket.backbone @ rot.T,
        residue_ids=pocket.residue_ids,
    )


def _encode_task(  # noqa: C901, PLR0911 -- one guard per way a complex can fail
    task: dict,
) -> tuple[int, list[list[int]], str]:
    """One complex -> its documents (one per orientation), or a failure reason.

    Each early return names a distinct reason the complex is unusable, and the
    caller counts them separately: collapsing them would turn "the alphabet is
    too small" and "ConfSeq cannot encode this" into one number.
    """
    from prolit.tokenizers.ligand import (  # noqa: PLC0415
        parse_ligand_pdb_text,
        parse_sdf,
    )

    enc: StapledEncoder = _W["enc"]
    idx = task["idx"]
    try:
        if task["kind"] == "biolip":
            mol = parse_ligand_pdb_text(task["lig"], task["smiles"])
            if mol is not None:
                mol.pop("chemistry", None)
            protein_text = task["rec"]
        else:
            mols = parse_sdf(Path(task["sdf"]))
            mol = mols[0] if mols else None
            protein_text = Path(task["protein"]).read_text()
        if mol is None:
            return idx, [], "parse_failed"
        heavy_idx = np.array(
            [k for k, a in enumerate(mol["atoms"]) if a[0] != "H"], dtype=int
        )
        heavy = np.array(
            [mol["atoms"][k][1:4] for k in heavy_idx], dtype=np.float32
        ).reshape(-1, 3)
        if not (task["min_heavy"] <= heavy.shape[0] <= task["max_heavy"]):
            return idx, [], "heavy_atom_bounds"
        if task["table"] is None:
            return idx, [], "esm3_missing_structure"
        enc.cache = _TableCache(task["sid"], task["table"])  # ty: ignore[invalid-assignment]
        pocket = enc.setup_pocket(task["sid"], protein_text, heavy)
        if pocket is None:
            return idx, [], "pocket_or_esm3_residue"
        docs: list[list[int]] = []
        why = "ok"
        for rot in task["rotations"]:
            if rot is None:
                p, atoms = pocket, mol["atoms"]
            else:
                r = np.asarray(rot)
                p = _rotated_pocket(pocket, r)
                atoms = [
                    (a[0], *(np.asarray(a[1:4], dtype=np.float64) @ r.T), *a[4:])
                    for a in mol["atoms"]
                ]
            seq, why = enc.ligand_seq_with_reason(p, atoms, mol["bonds"], heavy_idx)
            if seq is None:
                return idx, [], why
            docs.append(seq)
    except Exception as exc:  # noqa: BLE001
        return idx, [], f"exception:{type(exc).__name__}"
    return idx, docs, why


def _rotations(seed: int, name: str, n: int) -> list:
    """k = 0 is the deposited frame, so one orientation is the plain corpus."""
    rng = rng_for(seed, f"stapled_affinity_rotations:{name}")
    return [None] + [random_rotation_matrix(rng).tolist() for _ in range(n - 1)]


def build(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915
    cache = _MultiCache(args.esm3_cache)
    vocab = StapledVocab(confseq=ConfSeqVocab.load(args.stapled_vocab))
    common = {"min_heavy": args.min_heavy, "max_heavy": args.max_heavy}
    tasks: list[dict] = []
    meta_rows: list[tuple[str, float, int]] = []  # split, label, group

    if args.source == "biolip":
        sites, site_gid, val_gids = biolip_split(
            biolip_sites(args), args.seed, args.val_frac
        )
        store: dict[str, dict] = {}
        with gzip.open(args.work_dir / "biolip_sites.jsonl.gz", "rt") as fh:
            for line in fh:
                row = json.loads(line)
                store[row["key"]] = row
        ccd_smiles = _load_ccd_smiles(args.biolip_dir / "ligand.tsv.gz")
        by_bucket: dict[str, list[tuple]] = {}
        for s in sites[: args.max_complexes]:
            by_bucket.setdefault(_bucket_code(s[0]), []).append(s)
        for code in sorted(by_bucket):
            for s in by_bucket[code]:
                pdb, rchain, ccd = s[0], s[1], s[2]
                gid = site_gid[(pdb, ccd)]
                split = "val" if gid in val_gids else "train"
                row = store.get(_site_key(s))
                sid = f"{pdb}{rchain}"
                n_rot = args.num_rotations if split == "train" else 1
                tasks.append(
                    {
                        "idx": len(tasks),
                        "kind": "biolip",
                        "rec": None if row is None else row["rec"],
                        "lig": None if row is None else row["lig"],
                        "smiles": ccd_smiles.get(ccd),
                        "sid": sid,
                        "table": cache.residue_tokens(sid),
                        "rotations": _rotations(args.seed, f"{pdb}_{ccd}", n_rot),
                        **common,
                    }
                )
                meta_rows.append((split, float(s[5]), gid))
                if row is None:
                    tasks[-1]["kind"] = "missing"
    else:
        pdbs, labels, val = pdbbind_split(args)
        roots = [args.pdbbind_dir / "refined-set", args.pdbbind_dir / "v2020-other-PL"]
        for pdb in pdbs[: args.max_complexes]:
            d = _find_complex(roots, pdb)
            split = "val" if pdb in val else "train"
            n_rot = args.num_rotations if split == "train" else 1
            tasks.append(
                {
                    "idx": len(tasks),
                    "kind": "pdbbind" if d is not None else "missing",
                    "protein": None if d is None else str(d / f"{pdb}_protein.pdb"),
                    "sdf": None if d is None else str(d / f"{pdb}_ligand.sdf"),
                    "sid": pdb,
                    "table": cache.residue_tokens(pdb),
                    "rotations": _rotations(args.seed, pdb, n_rot),
                    **common,
                }
            )
            # PDBbind carries no protein groups: each complex is its own, as
            # mix.py would assign when the half is concatenated.
            meta_rows.append((split, float(labels[pdb]), -1))

    logger.info("%s: %d complexes to encode", args.source, len(tasks))
    # The tasks already hold every receptor and ligand record the workers need,
    # so the site store is dead weight that 38 forked workers would each copy
    # the moment the GC walks it. Drop it, then freeze what is left out of the
    # generational GC so a worker's collections do not un-share the parent's
    # pages.
    if args.source == "biolip":
        store.clear()
    gc.collect()
    gc.freeze()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {s: _RmsdWriter(args.out_dir, s) for s in ("train", "val")}
    groups: dict[str, list[int]] = {"train": [], "val": []}
    #: Documents written per accepted complex, per split: a complex's rotations
    #: are one group, so the group stream is built from these counts.
    doc_counts: dict[str, list[int]] = {"train": [], "val": []}
    reasons: Counter[str] = Counter()
    labels_ok: list[float] = []
    runnable = [t for t in tasks if t["kind"] != "missing"]
    reasons["missing_structure"] = len(tasks) - len(runnable)
    results: dict[int, tuple[list[list[int]], str]] = {}
    with Pool(
        args.num_workers,
        initializer=_worker_init,
        initargs=(
            str(args.confseq_repo),
            str(args.stapled_vocab),
            args.max_residues,
            args.pocket_distance,
        ),
    ) as pool:
        for n, (idx, docs, why) in enumerate(
            pool.imap_unordered(_encode_task, runnable, chunksize=4), 1
        ):
            results[idx] = (docs, why)
            if n % 1000 == 0:
                logger.info("encoded %d / %d", n, len(runnable))
    # Written in task order, not completion order, so a rebuild is byte-identical
    # and the rotations of one complex stay consecutive (what
    # ``--rotations-per-complex`` indexing assumes).
    for t in tasks:
        split, label, gid = meta_rows[t["idx"]]
        if t["idx"] not in results:
            continue
        docs, why = results[t["idx"]]
        if not docs:
            reasons[why] += 1
            continue
        reasons["ok"] += 1
        labels_ok.append(label)
        doc_counts[split].append(len(docs))
        for seq in docs:
            writers[split].write(seq, label)
            groups[split].append(gid if gid >= 0 else -1)
    for w in writers.values():
        w.close()
    for split, gl in groups.items():
        arr = np.asarray(gl, dtype=np.int64)
        if args.source == "pdbbind":
            # PDBbind carries no protein grouping, so every complex is its own
            # group -- but a group is a COMPLEX, not a document: the rotations
            # of one complex share it. Numbering per document instead makes a
            # rotation block span R groups, which is what
            # ``RescoreDataset._complex_index`` refuses (and it is right to:
            # a grouped loss would then rank one complex's own orientations
            # against each other).
            counts = np.asarray(doc_counts[split], dtype=np.int64)
            arr = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
        arr.astype(np.int32).tofile(args.out_dir / f"{split}.grp")

    meta = {
        "vocab_size": vocab.vocab_size,
        # The LM's embedding table is atom_codebook_size + specials, as for
        # every stapled corpus (--atom-codebook-size 12733).
        "atom_codebook_size": vocab.vocab_size - 7,
        "stapled": True,
        "source": f"stapled_{args.source}_affinity_pk",
        "label": "pK (-log10 molar)",
        "affinity_types": args.affinity_types,
        "pocket_distance": args.pocket_distance,
        "max_residues": args.max_residues,
        "num_rotations": args.num_rotations,
        "esm3_caches": [str(p) for p in args.esm3_cache],
        "n_complexes": reasons["ok"],
        "n_attempted": len(tasks),
        "drop_reasons": dict(reasons),
        "train_docs": writers["train"].num_docs,
        "val_docs": writers["val"].num_docs,
        "max_len": max(writers["train"].max_len, writers["val"].max_len),
        "seed": args.seed,
    }
    torch.save(meta, args.out_dir / "meta.pt")
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    a = np.asarray(labels_ok) if labels_ok else np.zeros(1)
    logger.info(
        "DONE %s: %s | train %d / val %d docs | pK med %.2f | max_len %d",
        args.source,
        dict(reasons),
        writers["train"].num_docs,
        writers["val"].num_docs,
        float(np.median(a)),
        meta["max_len"],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prep", "build"))
    p.add_argument("--source", choices=("biolip", "pdbbind"), default="biolip")
    p.add_argument("--work-dir", type=Path, default=Path("data/stapled_affinity"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--biolip-dir", type=Path, default=Path("data/biolip"))
    p.add_argument(
        "--cd-manifest", type=Path, default=Path("data/hub_cache/repo/manifest.parquet")
    )
    p.add_argument(
        "--biolip-holdout",
        type=Path,
        default=Path("data/eval_holdout_pdbs.txt"),
        help="--casf-pdbs of the ProLIT BioLiP half (lm_tokens_e250aff_r1)",
    )
    p.add_argument(
        "--pdbbind-dir",
        type=Path,
        default=Path("/gs/bs/tga-ohuelab/ysato/data/P-L"),
    )
    p.add_argument(
        "--pdbbind-holdout",
        type=Path,
        default=Path("data/affinity_noconf_holdout_pdbs.txt"),
        help="--casf-pdbs of the ProLIT PDBbind half "
        "(lm_tokens_e250aff_pdbbind_noconf)",
    )
    p.add_argument("--esm3-cache", type=Path, nargs="+", default=[])
    p.add_argument(
        "--stapled-vocab", type=Path, default=Path("data/stapled/confseq_vocab.json")
    )
    p.add_argument("--confseq-repo", type=Path, default=Path("third_party/ConfSeq"))
    p.add_argument("--affinity-types", default="KD,KI")
    p.add_argument("--max-residues", type=int, default=50)
    p.add_argument("--pocket-distance", choices=("ca", "heavy"), default="ca")
    p.add_argument("--min-heavy", type=int, default=6)
    p.add_argument("--max-heavy", type=int, default=80)
    p.add_argument("--pk-min", type=float, default=2.0)
    p.add_argument("--pk-max", type=float, default=13.0)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--num-rotations", type=int, default=1)
    p.add_argument("--max-complexes", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=8)
    add_seed_argument(p, default=7)
    args = p.parse_args()
    seed_from_args(args)
    if args.stage == "prep":
        prep(args)
        return
    if args.out_dir is None:
        p.error("build needs --out-dir")
    if not args.esm3_cache:
        p.error("build needs --esm3-cache")
    build(args)


if __name__ == "__main__":
    main()
