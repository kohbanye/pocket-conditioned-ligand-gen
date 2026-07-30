"""Real docking-decoy corpus for the pose-rescoring head.

The synthetic rigid/torsion decoys (``tokenize_decoys.py``) are geometrically
implausible (random perturbations clash), so a head trained on them learns
"clash == bad" -- which does NOT transfer to CASF, whose decoys are real,
non-clashing docking poses in alternative binding modes. This script instead
**redocks** each BioLIP native ligand into its own pocket with AutoDock Vina and
labels every output pose by heavy-atom RMSD to the crystal ligand, reproducing
the CASF docking-decoy distribution directly.

Three phases on ONE node (96-core + H100):
- Phase 0 (main): stream each ``.tar.bz2`` bucket ONCE, dump receptor/ligand PDBs
  to a scratch dir (per-site tar reads are O(n^2)).
- Phase A (CPU pool): per complex -- receptor/ligand -> pdbqt, Vina search
  (--num_modes), heavy-atom RMSD via a coordinate-matched permutation (obabel
  reorders atoms but does not move them), then pocket + per-pose ligand atom
  descriptors. All CPU; returns numpy descriptors + RMSDs.
- Phase B (GPU main): VQ-encode protein + pose ligand descriptors, assemble
  ``<p>..</p><l>..</l>`` sequences, stream ``.bin/.len/.rmsd`` (RescoreDataset).

Run (interactive H100 node)::

    .venv/bin/python pipelines/corpora/build_docking_decoys.py \
        --ckpt <atom-vqvae>.ckpt \
        --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
        --out-dir data/lm_tokens_dock_decoys --n-complexes 6000 --workers 90
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch

# Sibling modules in this directory, imported by bare name: Python puts a
# script's own directory on sys.path[0], so this resolves from any cwd.
from tokenize_biolip import (
    _bucket_code,
    _load_ccd_smiles,
    _parse_biolip_txt,
)
from tokenize_decoys import _cd_test_pdbs, _RmsdWriter

from prolit.config import AtomVQVAETrainingConfig, PocketExtractionConfig
from prolit.data.descriptors import collate_molecules
from prolit.external_tools import tool_default
from prolit.model.vqvae_module import AtomVQVAEModule
from prolit.seeding import add_seed_argument, seed_from_args
from prolit.tokenizers.atom import (
    LigandAtomDescriptor,
    ProteinAtomDescriptor,
    precompute_receptor_atom_features_from_text,
)
from prolit.tokenizers.ligand import parse_ligand_pdb_text
from prolit.tokenizers.lm_vocab import AtomLMVocab
from prolit.tokenizers.protein import (
    compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    precompute_pocket_atom_candidates_from_text,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    global _CFG  # noqa: PLW0603
    _CFG = cfg
    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, check=False, timeout=timeout
    )


def _pdbqt_models(text: str) -> list[np.ndarray]:
    """Heavy-atom coords (in file order) for each MODEL in a (multi-)pdbqt."""
    poses: list[np.ndarray] = []
    cur: list[tuple[float, float, float]] = []
    in_model = False
    for ln in text.splitlines():
        if ln.startswith("MODEL"):
            cur, in_model = [], True
        elif ln.startswith("ENDMDL"):
            poses.append(np.array(cur, np.float64) if cur else np.empty((0, 3)))
            in_model = False
        elif ln.startswith(("ATOM", "HETATM")):
            atype = ln[77:].strip() or ln[12:16].strip()
            if atype[:1].upper() == "H":  # AutoDock H / HD -> skip
                continue
            cur.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    if in_model and cur:  # single-model file without MODEL/ENDMDL wrapper
        poses.append(np.array(cur, np.float64))
    if not poses and cur:
        poses.append(np.array(cur, np.float64))
    return poses


def _stage_bucket(task: tuple) -> int:
    """Extract needed receptor/ligand PDBs from one bucket's tarballs (parallel).

    Single-threaded bz2 decompression of ~700 dense buckets in the main process
    is the startup bottleneck (~2 files/s); fan it out over the worker pool.
    """
    code, needed_rec, needed_lig = task
    cfg = _CFG
    bio = Path(cfg["biolip_dir"])
    scratch = Path(cfg["scratch"])
    n = 0
    for kind, sub, needed in (
        ("receptor", "rec", set(needed_rec)),
        ("ligand", "lig", set(needed_lig)),
    ):
        path = bio / kind / f"{kind}_{code}.tar.bz2"
        if not path.exists() or not needed:
            continue
        try:
            with tarfile.open(path, mode="r|bz2") as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    bn = Path(m.name).name
                    if bn in needed:
                        f = tf.extractfile(m)
                        if f is not None:
                            (scratch / sub / bn).write_bytes(f.read())
                            n += 1
                        needed.discard(bn)
                    if not needed:
                        break
        except Exception:  # noqa: BLE001, S110
            pass
    return n


def _dock_and_describe(task: tuple) -> dict | None:  # noqa: C901, PLR0911, PLR0912, PLR0915
    """Redock one native complex; return pocket + per-pose ligand descriptors."""
    cfg = _CFG
    rec_path, lig_path, _ccd, pdb, smiles = task
    try:
        mol = parse_ligand_pdb_text(
            Path(lig_path).read_text("utf-8", "replace"), smiles
        )
        if mol is None:
            return None
        atoms = mol["atoms"]
        heavy_local = [i for i, a in enumerate(atoms) if a[0] != "H"]
        heavy = np.array(
            [(atoms[i][1], atoms[i][2], atoms[i][3]) for i in heavy_local], np.float64
        )
        nh = heavy.shape[0]
        if not (cfg["min_heavy"] <= nh <= cfg["max_heavy"]):
            return None
        rec_text = Path(rec_path).read_text("utf-8", "replace")

        with tempfile.TemporaryDirectory(dir=cfg["tmp_dir"]) as td:
            tdp = Path(td)
            recq, ligq = tdp / "r.pdbqt", tdp / "l.pdbqt"
            _run([cfg["obabel"], rec_path, "-O", str(recq), "-xr"])
            _run([cfg["obabel"], lig_path, "-O", str(ligq), "-p", "7.4"])
            if not recq.exists() or not ligq.exists():
                return None
            in_heavy = _pdbqt_models(ligq.read_text())
            if not in_heavy or in_heavy[0].shape[0] != nh:
                return None
            p0 = in_heavy[0]
            d = np.linalg.norm(p0[:, None, :] - heavy[None, :, :], axis=2)
            perm = d.argmin(axis=1)  # pdbqt atom j -> native heavy slot
            if len(set(perm.tolist())) != nh or d[np.arange(nh), perm].max() > 0.6:  # noqa: PLR2004
                return None
            cx, cy, cz = heavy.mean(0)
            ext = heavy.max(0) - heavy.min(0) + 8.0
            box = [
                "--center_x",
                f"{cx:.2f}",
                "--center_y",
                f"{cy:.2f}",
                "--center_z",
                f"{cz:.2f}",
                "--size_x",
                f"{max(20.0, ext[0]):.2f}",
                "--size_y",
                f"{max(20.0, ext[1]):.2f}",
                "--size_z",
                f"{max(20.0, ext[2]):.2f}",
            ]
            outq = tdp / "poses.pdbqt"
            _run(
                [
                    cfg["vina"],
                    "--receptor",
                    str(recq),
                    "--ligand",
                    str(ligq),
                    "--out",
                    str(outq),
                    "--num_modes",
                    str(cfg["num_modes"]),
                    "--exhaustiveness",
                    str(cfg["exhaustiveness"]),
                    "--cpu",
                    "1",
                    "--seed",
                    "1",
                    *box,
                ],
                timeout=cfg["dock_timeout"],
            )
            if not outq.exists():
                return None
            pose_coords = [
                p for p in _pdbqt_models(outq.read_text()) if p.shape[0] == nh
            ]
        if not pose_coords:
            return None

        # native (rmsd 0) + docked poses, all mapped back to native atom order
        heavy_full = np.array([(a[1], a[2], a[3]) for a in atoms], np.float64)
        pose_mols = [mol]
        rmsds = [0.0]
        for pc in pose_coords:
            back = np.empty((nh, 3))
            back[perm] = pc
            rmsds.append(float(np.sqrt(((back - heavy) ** 2).sum(1).mean())))
            new_full = heavy_full.copy()
            new_full[heavy_local] = back
            pose_mols.append(
                {
                    "atoms": [
                        (
                            a[0],
                            float(new_full[i][0]),
                            float(new_full[i][1]),
                            float(new_full[i][2]),
                        )
                        for i, a in enumerate(atoms)
                    ],
                    "bonds": mol["bonds"],
                }
            )

        precomp = precompute_pocket_atom_candidates_from_text(rec_text)
        pocket = extract_pocket_atoms_from_candidates(
            precomp, heavy.astype(np.float32), cfg["pocket_cfg"]
        )
        if pocket is None or pocket.atom_coords.shape[0] == 0:
            return None
        feats = precompute_receptor_atom_features_from_text(rec_text)
        frame = compute_canonical_frame(pocket.ca_coords.astype(np.float64))
        prot_desc, _ = ProteinAtomDescriptor().compute(pocket, feats, frame)
        if prot_desc.shape[0] == 0:
            return None
        lig_desc_fn = LigandAtomDescriptor()
        lig_descs, out_rmsds = [], []
        for m, r in zip(pose_mols, rmsds, strict=True):
            ld, _e, _m = lig_desc_fn.compute(m["atoms"], m["bonds"], frame)
            if ld.shape[0] == 0:
                continue
            lig_descs.append(ld.astype(np.float32))
            out_rmsds.append(r)
        if len(lig_descs) < 2:  # need native + >=1 decoy  # noqa: PLR2004
            return None
    except Exception:  # noqa: BLE001
        return None
    return {
        "pdb": pdb,
        "prot_desc": prot_desc.astype(np.float32),
        "lig_descs": lig_descs,
        "rmsds": out_rmsds,
    }


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--biolip-dir", type=Path, default=Path("data/biolip"))
    parser.add_argument(
        "--cd-manifest",
        type=Path,
        default=Path("data/hub_cache/repo/manifest.parquet"),
    )
    parser.add_argument(
        "--casf-pdbs", type=Path, default=Path("data/casf2016_pdbs.txt")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/lm_tokens_dock_decoys")
    )
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--max-residues", type=int, default=50)
    parser.add_argument("--n-complexes", type=int, default=6000)
    parser.add_argument("--num-modes", type=int, default=20)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--dock-timeout", type=int, default=300)
    parser.add_argument("--min-heavy", type=int, default=8)
    parser.add_argument("--max-heavy", type=int, default=45)
    parser.add_argument("--val-frac", type=float, default=0.03)
    add_seed_argument(parser, default=0)
    parser.add_argument("--workers", type=int, default=90)
    parser.add_argument("--vina", type=str, default=tool_default("vina"))
    parser.add_argument(
        "--obabel", type=str, default=tool_default("obabel")
    )
    parser.add_argument("--tmp-dir", type=str, default=str(Path.home() / "tmpdir"))
    args = parser.parse_args()
    seed_from_args(args)

    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- complex selection (CASF/CrossDocked-test excluded, deduped) ----
    sites = _parse_biolip_txt(args.biolip_dir / "BioLiP.txt.gz")
    ccd_smiles = _load_ccd_smiles(args.biolip_dir / "ligand.tsv.gz")
    excluded = _cd_test_pdbs(args.cd_manifest)
    if args.casf_pdbs.exists():
        excluded |= {p.lower() for p in args.casf_pdbs.read_text().split() if p.strip()}
    rng = np.random.default_rng(args.seed)
    seen: set = set()
    uniq: list[tuple] = []
    for s in sites:
        if s[0] in excluded or (s[0], s[2]) in seen:
            continue
        seen.add((s[0], s[2]))
        uniq.append(s)
    rng.shuffle(uniq)
    uniq = uniq[: args.n_complexes]
    val_pdbs = {s[0] for s in uniq[: int(len(uniq) * args.val_frac)]}
    logger.info("selected %d native complexes to redock", len(uniq))

    from tqdm import tqdm  # noqa: PLC0415

    scratch = Path(tempfile.mkdtemp(dir=args.tmp_dir, prefix="dockdecoy_"))
    (scratch / "rec").mkdir()
    (scratch / "lig").mkdir()
    cfg = {
        "vina": args.vina,
        "obabel": args.obabel,
        "tmp_dir": args.tmp_dir,
        "num_modes": args.num_modes,
        "exhaustiveness": args.exhaustiveness,
        "dock_timeout": args.dock_timeout,
        "min_heavy": args.min_heavy,
        "max_heavy": args.max_heavy,
        "biolip_dir": str(args.biolip_dir),
        "scratch": str(scratch),
        "pocket_cfg": PocketExtractionConfig(max_residues=args.max_residues),
    }

    # Fork the CPU pool BEFORE any CUDA init so workers inherit the already
    # imported torch/rdkit (COW, no per-worker re-import / RAM blow-up) and never
    # carry a CUDA context. The pool serves both phases.
    ctx = mp.get_context("fork")
    pool = ctx.Pool(args.workers, initializer=_init_worker, initargs=(cfg,))

    # ---- Phase 0: stage receptor/ligand PDBs, one task per bucket (parallel) ----
    by_bucket: dict[str, list[tuple]] = {}
    for s in uniq:
        by_bucket.setdefault(_bucket_code(s[0]), []).append(s)
    bucket_tasks = [
        (
            code,
            [f"{p}{rc}.pdb" for p, rc, _c, _l, _s in site_list],
            [f"{p}_{cc}_{lc}_{s}.pdb" for p, _rc, cc, lc, s in site_list],
        )
        for code, site_list in by_bucket.items()
    ]
    staged = sum(
        tqdm(
            pool.imap_unordered(_stage_bucket, bucket_tasks, chunksize=1),
            total=len(bucket_tasks),
            desc="stage",
        )
    )
    tasks: list[tuple] = []
    for site_list in by_bucket.values():
        for pdb, rchain, ccd, ligchain, serial in site_list:
            rp = scratch / "rec" / f"{pdb}{rchain}.pdb"
            lp = scratch / "lig" / f"{pdb}_{ccd}_{ligchain}_{serial}.pdb"
            if rp.exists() and lp.exists():
                tasks.append((str(rp), str(lp), ccd, pdb, ccd_smiles.get(ccd)))
    logger.info(
        "Phase 0 done: staged %d files over %d buckets -> %d dockable tasks",
        staged,
        len(bucket_tasks),
        len(tasks),
    )

    # ---- Phase A: parallel Vina docking + descriptors ----
    result_iter = pool.imap_unordered(_dock_and_describe, tasks, chunksize=1)

    # ---- Phase B setup: VQ-VAE on GPU (after fork) ----
    vq_cfg = AtomVQVAETrainingConfig()
    vq_cfg.atom.codebook_size = args.codebook_size
    module = AtomVQVAEModule.load_from_checkpoint(
        args.ckpt, config=vq_cfg, map_location=device
    )
    module.eval().to(device)
    norm = torch.load(args.norm_stats, weights_only=False)
    module.vqvae.set_normalization(norm["atom_mean"], norm["atom_std"])
    mean, std = norm["atom_mean"].numpy(), norm["atom_std"].numpy()
    vocab = AtomLMVocab(codebook_size=args.codebook_size)

    @torch.no_grad()
    def encode(descs: list[np.ndarray]) -> list[list[int]]:
        tensors = [torch.from_numpy((d - mean) / std).float() for d in descs]
        x, m = collate_molecules(tensors)
        idx = module.vqvae.encode_batch(x.to(device), m.to(device)).cpu()
        return [idx[i][m[i]].tolist() for i in range(len(descs))]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "train": _RmsdWriter(args.out_dir, "train"),
        "val": _RmsdWriter(args.out_dir, "val"),
    }
    n_ok = n_pose = 0
    for res in tqdm(result_iter, total=len(tasks), desc="dock"):
        if res is None:
            continue
        codes = encode([res["prot_desc"], *res["lig_descs"]])
        p_codes, lig_code_list = codes[0], codes[1:]
        split = "val" if res["pdb"] in val_pdbs else "train"
        for lc, r in zip(lig_code_list, res["rmsds"], strict=True):
            writers[split].write(vocab.build_sequence(p_codes, lc), r)
            n_pose += 1
        n_ok += 1
    pool.close()
    pool.join()

    meta = {
        "vocab_size": vocab.vocab_size,
        "atom_codebook_size": args.codebook_size,
        "source": "biolip2_vina_docking_decoys",
        "n_complexes": n_ok,
        "n_poses": n_pose,
        "train_docs": writers["train"].num_docs,
        "val_docs": writers["val"].num_docs,
        "max_len": max(writers["train"].max_len, writers["val"].max_len),
    }
    for w in writers.values():
        w.close()
    torch.save(meta, args.out_dir / "meta.pt")
    logger.info(
        "DONE: %d complexes / %d poses -> train %d / val %d docs (max_len %d)",
        n_ok,
        n_pose,
        writers["train"].num_docs,
        writers["val"].num_docs,
        meta["max_len"],
    )
    import shutil  # noqa: PLC0415

    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
