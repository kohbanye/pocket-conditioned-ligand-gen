"""Build a decoy-pose training set (complex tokens + RMSD) for the scoring head.

Zero-shot PLL already ranks native above decoy (35% -> 55% docking power as the
MLM trains); a discriminative head trained on RMSD-labelled decoys should push
further (the Interformer / DeepBSP recipe). We generate decoys WITHOUT docking
software or re-downloading CrossDocked types: take CASF-excluded BioLIP native
complexes and rigidly perturb the ligand (rotation about its centroid +
translation) by a graded magnitude, so the RMSD to the crystal pose is known
exactly. The protein pocket is fixed, so only the ligand codes change -- exactly
the P(ligand pose | pocket) signal the head must learn to score.

Output: ``{split}.bin`` (uint16 tokens) + ``{split}.len`` + ``{split}.rmsd``
(float32, one per doc). Native pose is RMSD 0.

Run (single GPU)::

    uv run python pipelines/corpora/tokenize_decoys.py \
        --ckpt "<atom-vqvae>.ckpt" \
        --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
        --n-complexes 12000 --n-decoys 16 --out-dir data/lm_tokens_decoys
"""

from __future__ import annotations

import argparse
import json
import logging
import zlib
from pathlib import Path

import numpy as np
import torch

from pipelines.corpora.tokenize_biolip import (
    _bucket_code,
    _cd_test_pdbs,
    _load_ccd_smiles,
    _parse_biolip_txt,
    _read_needed,
)
from prolit.config import AtomVQVAETrainingConfig, PocketExtractionConfig
from prolit.model.vqvae_module import AtomVQVAEModule
from prolit.tokenizers.ligand import parse_ligand_pdb_text
from prolit.tokenizers.lm_vocab import AtomLMVocab
from scripts.eval_casf_rescore import _PoseEncoder, _random_rotation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for a unit ``axis`` and ``angle`` (radians)."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def _perturb(
    coords: np.ndarray, rng: np.random.Generator, scale: float
) -> tuple[np.ndarray, float]:
    """Rigidly rotate+translate heavy-atom coords; return (new_coords, RMSD)."""
    centroid = coords.mean(0)
    angle = rng.uniform(0, scale * np.pi / 2)  # up to 90 deg at scale=1
    trans = rng.normal(size=3)
    trans = trans / (np.linalg.norm(trans) + 1e-9) * rng.uniform(0, scale * 6.0)
    rot = _rotation(rng.normal(size=3), angle)
    new = (coords - centroid) @ rot.T + centroid + trans
    rmsd = float(np.sqrt(((new - coords) ** 2).sum(axis=1).mean()))
    return new, rmsd


def _conf_perturb(  # noqa: C901
    atoms: list, bonds: list, rng: np.random.Generator, scale: float
) -> np.ndarray | None:
    """Rotate a random subset of rotatable-bond torsions + a small rigid shift.

    Produces a VALID-geometry conformational decoy (like a real docking pose that
    is near-native in place but wrong in torsions), which rigid perturbation
    alone cannot -- closing the train/test decoy-distribution gap. Returns the
    perturbed (all-atom) coords, or ``None`` if the RDKit build fails.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import rdMolTransforms as rmt  # noqa: PLC0415, N813
    from rdkit.Geometry import Point3D  # noqa: PLC0415

    bt = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    rw = Chem.RWMol()
    for a in atoms:
        try:
            rw.AddAtom(Chem.Atom(a[0]))
        except Exception:  # noqa: BLE001
            rw.AddAtom(Chem.Atom("C"))
    n = len(atoms)
    for i, j, t in bonds:
        if 0 <= i < n and 0 <= j < n and i != j:
            try:
                rw.AddBond(i, j, bt.get(t, Chem.BondType.SINGLE))
            except Exception:  # noqa: BLE001, S112
                continue
    mol = rw.GetMol()
    conf = Chem.Conformer(n)
    for i, a in enumerate(atoms):
        conf.SetAtomPosition(i, Point3D(float(a[1]), float(a[2]), float(a[3])))
    mol.AddConformer(conf)
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    except Exception:  # noqa: BLE001
        return None
    patt = Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")
    rot = mol.GetSubstructMatches(patt) if patt is not None else ()
    c = mol.GetConformer()
    for b1, b2 in rot:
        if rng.random() > 0.5:  # noqa: PLR2004
            continue
        a1 = [
            x.GetIdx()
            for x in mol.GetAtomWithIdx(b1).GetNeighbors()
            if x.GetIdx() != b2
        ]
        a4 = [
            x.GetIdx()
            for x in mol.GetAtomWithIdx(b2).GetNeighbors()
            if x.GetIdx() != b1
        ]
        if not a1 or not a4:
            continue
        try:
            cur = rmt.GetDihedralDeg(c, a1[0], b1, b2, a4[0])
            rmt.SetDihedralDeg(
                c, a1[0], b1, b2, a4[0], cur + rng.uniform(-scale * 180, scale * 180)
            )
        except Exception:  # noqa: BLE001, S112
            continue
    new = np.array([list(c.GetAtomPosition(i)) for i in range(n)], dtype=np.float64)
    return _perturb(new, rng, scale * 0.4)[0]  # small rigid on top


def _kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform (rot, trans) that best superposes ``p`` onto ``q``."""
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
    d = float(np.sign(np.linalg.det(vt.T @ u.T)))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, qc - rot @ pc


def _conformer_decoys(  # noqa: PLR0913
    atoms: list, bonds: list, hidx: np.ndarray, base: np.ndarray, n: int, seed: int
) -> list[np.ndarray]:
    """Freshly embedded conformers, rigidly superposed onto the native pose.

    The decoys a docking program produces near the native site have *correct
    placement but an independently generated internal conformer*; perturbing the
    crystal conformer (rigidly or by torsion) never yields that. A head trained
    only on perturbed-crystal decoys can therefore learn "the exact crystal
    conformer is the native one" and bury a genuinely good redocked pose -- the
    observed failure mode (good poses ranked >10 on 14 CASF targets). These
    decoys put realistic 0.5-2.5 A near-natives in the training distribution.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    bt = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    rw = Chem.RWMol()
    for a in atoms:
        try:
            rw.AddAtom(Chem.Atom(a[0]))
        except Exception:  # noqa: BLE001
            rw.AddAtom(Chem.Atom("C"))
    na = len(atoms)
    for i, j, t in bonds:
        if 0 <= i < na and 0 <= j < na and i != j:
            try:
                rw.AddBond(i, j, bt.get(t, Chem.BondType.SINGLE))
            except Exception:  # noqa: BLE001, S112
                continue
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001
        return []  # caller falls back to the perturbation decoys
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.3
    params.maxIterations = 200
    try:
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=3 * n, params=params)
    except Exception:  # noqa: BLE001
        return []
    poses = []
    for cid in cids:
        conf = mol.GetConformer(cid)
        new = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(na)], dtype=np.float64
        )
        rot, trans = _kabsch(new[hidx], base[hidx])
        new = new @ rot.T + trans
        poses.append(
            (float(np.sqrt(((new[hidx] - base[hidx]) ** 2).sum(1).mean())), new)
        )
    if not poses:
        return []
    # A docking run *searches* conformers, so its pose set is graded: the best
    # sampled conformer sits near the native and the rest fan out. A random
    # embedding is not -- it lands 2-4 A away. Sample the pool evenly along its
    # own RMSD order so the closest conformer is always kept and the class spans
    # near-native to clearly-wrong, like a real pose set.
    poses.sort(key=lambda t: t[0])
    idx = np.unique(np.linspace(0, len(poses) - 1, num=n).round().astype(int))
    return [poses[i][1] for i in idx]


class _RmsdWriter:
    """Streams tokens (.bin/.len), one float32 RMSD per doc (.rmsd), and the
    per-ligand-atom displacement (.disp float32, ``.dlen`` uint16 counts).

    The per-atom stream is dense supervision: one RMSD scalar tells the head only
    that a pose is wrong, while a displacement per ligand token tells it *which
    atoms* are misplaced -- ~30 labels per pose instead of 1, and exactly the
    quantity the pose score aggregates.
    """

    def __init__(self, out_dir: Path, split: str) -> None:
        self._bin = (out_dir / f"{split}.bin").open("wb")
        self._len = (out_dir / f"{split}.len").open("wb")
        self._rmsd = (out_dir / f"{split}.rmsd").open("wb")
        self._disp = (out_dir / f"{split}.disp").open("wb")
        self._dlen = (out_dir / f"{split}.dlen").open("wb")
        self.num_docs = 0
        self.num_tokens = 0
        self.max_len = 0

    def write(
        self, seq: list[int], rmsd: float, disp: np.ndarray | None = None
    ) -> None:
        arr = np.asarray(seq, dtype=np.uint16)
        self._bin.write(arr.tobytes())
        self._len.write(np.asarray([len(seq)], dtype=np.uint16).tobytes())
        self._rmsd.write(np.asarray([rmsd], dtype=np.float32).tobytes())
        d = np.asarray([] if disp is None else disp, dtype=np.float32)
        self._disp.write(d.tobytes())
        self._dlen.write(np.asarray([d.shape[0]], dtype=np.uint16).tobytes())
        self.num_docs += 1
        self.num_tokens += len(seq)
        self.max_len = max(self.max_len, len(seq))
        # Flush the small .len/.rmsd streams periodically so a long run's partial
        # output stays inspectable and survives interruption (they otherwise sit
        # in an 8 KiB buffer for thousands of poses).
        if self.num_docs % 256 == 0:
            self._len.flush()
            self._rmsd.flush()
            self._bin.flush()
            self._disp.flush()
            self._dlen.flush()

    def close(self) -> None:
        self._bin.close()
        self._len.close()
        self._rmsd.close()
        self._disp.close()
        self._dlen.close()


def main() -> None:  # noqa: C901, PLR0915, PLR0912
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=None)
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
        "--casf-pdbs", type=Path, default=Path("data/casf2016_pdbs.txt")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/lm_tokens_decoys"))
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--max-residues", type=int, default=50)
    parser.add_argument("--n-complexes", type=int, default=12000)
    parser.add_argument("--n-decoys", type=int, default=16)
    parser.add_argument("--min-heavy", type=int, default=6)
    parser.add_argument("--max-heavy", type=int, default=50)
    parser.add_argument("--val-frac", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-conformer-decoys",
        type=int,
        default=0,
        help="Of --n-decoys, how many come from freshly embedded conformers "
        "superposed on the native pose (docking-like near-natives). The rest "
        "alternate rigid / torsion perturbation as before.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the receptor buckets across this many array tasks.",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument(
        "--num-rot",
        type=int,
        default=1,
        help="Emit each complex under this many random frame rotations "
        "(1 = canonical frame only). Data augmentation against VQ code noise.",
    )
    parser.add_argument(
        "--skip-canonical",
        action="store_true",
        help="Make every emitted rotation random (no canonical-frame copy), so a "
        "second pass over the same complexes yields rotated duplicates to "
        "concatenate onto an existing canonical corpus.",
    )
    args = parser.parse_args()

    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = AtomVQVAETrainingConfig()
    cfg.atom.codebook_size = args.codebook_size
    if args.separate_protein_ckpt is not None:
        # ABLATION separate-tokenizers mode: protein-only VQ + ligand-only VQ
        # unified into one code space. Feed RAW descriptors (identity external
        # norm) via _PoseEncoder -- SeparateVQVAE normalizes per modality
        # internally. Combined single-range AtomLMVocab over 2*codebook_size
        # codes. _PoseEncoder encodes pocket + ligand in separate (single-
        # modality) encode_batch calls, which SeparateVQVAE requires.
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
        module = AtomVQVAEModule.load_from_checkpoint(
            args.ckpt, config=cfg, map_location=device
        )
        module.eval().to(device)
        norm = torch.load(args.norm_stats, weights_only=False)
        module.vqvae.set_normalization(norm["atom_mean"], norm["atom_std"])
        mean = norm["atom_mean"].numpy()
        std = norm["atom_std"].numpy()
        vocab = AtomLMVocab(codebook_size=args.codebook_size)
    enc = _PoseEncoder(
        module,
        mean,
        std,
        vocab,
        device,
        PocketExtractionConfig(max_residues=args.max_residues),
    )

    # CASF-excluded, deduped (pdb, ccd) native sites, shuffled to a subset.
    sites = _parse_biolip_txt(args.biolip_dir / "BioLiP.txt.gz")
    ccd_smiles = _load_ccd_smiles(args.biolip_dir / "ligand.tsv.gz")
    excluded = _cd_test_pdbs(args.cd_manifest)
    if args.casf_pdbs.exists():
        excluded |= {p.lower() for p in args.casf_pdbs.read_text().split() if p.strip()}
    rng = np.random.default_rng(args.seed)
    seen: set = set()
    uniq = []
    for s in sites:
        if s[0] in excluded:
            continue
        key = (s[0], s[2])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    rng.shuffle(uniq)
    uniq = uniq[: args.n_complexes]
    logger.info(
        "decoy source: %d native complexes (x%d decoys)", len(uniq), args.n_decoys
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "train": _RmsdWriter(args.out_dir, "train"),
        "val": _RmsdWriter(args.out_dir, "val"),
    }
    val_pdbs = {s[0] for s in uniq[: int(len(uniq) * args.val_frac)]}

    # group by bucket to stream each tar once
    by_bucket: dict[str, list[tuple]] = {}
    for s in uniq:
        by_bucket.setdefault(_bucket_code(s[0]), []).append(s)

    from tqdm import tqdm  # noqa: PLC0415

    # Shard by bucket so an array job can build a large corpus in parallel; the
    # complex list and val split are computed identically in every task (same
    # seed), so the shards concatenate into one consistent corpus.
    codes = sorted(by_bucket)[args.shard_id :: args.num_shards]
    logger.info("shard %d/%d: %d buckets", args.shard_id, args.num_shards, len(codes))

    n_ok = 0
    for code in tqdm(codes, desc="buckets"):
        site_list = by_bucket[code]
        needed_rec = {f"{p}{rc}.pdb" for p, rc, _c, _l, _s in site_list}
        needed_lig = {f"{p}_{cc}_{lc}_{s}.pdb" for p, _rc, cc, lc, s in site_list}
        # _read_needed uses a module-global biolip dir; set it once.
        import pipelines.corpora.tokenize_biolip as tb  # noqa: PLC0415

        tb._w_biolip_dir = args.biolip_dir  # noqa: SLF001
        receptors = _read_needed("receptor", code, needed_rec)
        ligands = _read_needed("ligand", code, needed_lig)
        for pdb, rchain, ccd, ligchain, serial in site_list:
            rec = receptors.get(f"{pdb}{rchain}.pdb")
            lig = ligands.get(f"{pdb}_{ccd}_{ligchain}_{serial}.pdb")
            if rec is None or lig is None:
                continue
            try:
                mol = parse_ligand_pdb_text(
                    lig.decode("utf-8", "replace"), ccd_smiles.get(ccd)
                )
                if mol is None:
                    continue
                heavy_idx = [i for i, a in enumerate(mol["atoms"]) if a[0] != "H"]
                heavy = np.array(
                    [
                        (mol["atoms"][i][1], mol["atoms"][i][2], mol["atoms"][i][3])
                        for i in heavy_idx
                    ],
                    np.float32,
                )
                if not (args.min_heavy <= heavy.shape[0] <= args.max_heavy):
                    continue
                setup = enc.setup_pocket(rec.decode("utf-8", "replace"), heavy)
                if setup is None:
                    continue
                p_codes, frame = setup
                split = "val" if pdb in val_pdbs else "train"
                # Build native + decoys (rigid + conformational), then encode all
                # poses in ONE batched VQ call (per-pose batch-1 was the bottleneck).
                base = np.array([(a[1], a[2], a[3]) for a in mol["atoms"]], np.float64)
                hidx = np.array(heavy_idx)
                mols, rmsds = [mol], [0.0]
                disps = [np.zeros(len(hidx), dtype=np.float32)]
                # Freshly embedded conformers superposed on the native: the
                # docking-like near-native class that perturbation cannot make.
                confs = (
                    _conformer_decoys(
                        mol["atoms"],
                        mol["bonds"],
                        hidx,
                        base,
                        args.n_conformer_decoys,
                        # crc32, not hash(): PYTHONHASHSEED randomizes str hashes
                        # per process, so shards would not be reproducible.
                        seed=zlib.crc32(f"{pdb}_{ccd}".encode()) % (2**31),
                    )
                    if args.n_conformer_decoys > 0
                    else []
                )
                n_pert = args.n_decoys - len(confs)
                for k in range(args.n_decoys):
                    if k >= n_pert:
                        # Kept as embedded (already RMSD-graded): this class is
                        # purely "right place, independently generated conformer".
                        new = confs[k - n_pert]
                    else:
                        scale = (k + 1) / max(1, n_pert)
                        if k % 2 == 0:
                            new = _perturb(base, rng, scale)[0]
                        else:
                            new = _conf_perturb(mol["atoms"], mol["bonds"], rng, scale)
                            if new is None:
                                new = _perturb(base, rng, scale)[0]
                    d = np.linalg.norm(new[hidx] - base[hidx], axis=1)
                    disps.append(d.astype(np.float32))
                    rmsds.append(float(np.sqrt((d**2).mean())))
                    mols.append({
                        "atoms": [
                            (a[0], float(new[i][0]), float(new[i][1]), float(new[i][2]))
                            for i, a in enumerate(mol["atoms"])
                        ],
                        "bonds": mol["bonds"],
                    })
                # Descriptors once; each extra rotation only re-quantizes them.
                # A rotated frame is a different tokenization of the SAME complex
                # (the VQ-VAE was pretrained with this augmentation), so it
                # teaches the head to score the pose, not the code pattern.
                descs = enc.ligand_descs(mols, frame)
                for r in range(args.num_rot):
                    canon = r == 0 and not args.skip_canonical
                    rot = None if canon else _random_rotation(rng)
                    pc = p_codes if rot is None else enc.pocket_codes_rotated(rot)
                    seqs = enc.seqs_from_descs(pc, descs, rotation=rot)
                    if seqs[0] is None:
                        break  # native must encode
                    for seq, rmsd, dsp, dsc in zip(
                        seqs, rmsds, disps, descs, strict=True
                    ):
                        if seq is None:
                            continue
                        # Per-atom labels are only usable when they line up with
                        # the emitted ligand tokens (one row per heavy atom).
                        ok = dsc.shape[0] == dsp.shape[0]
                        writers[split].write(seq, rmsd, dsp if ok else None)
                else:
                    n_ok += 1
            except Exception:
                logger.exception("failed %s_%s", pdb, ccd)
                continue

    meta = {
        "vocab_size": vocab.vocab_size,
        # Separate-tokenizers mode doubles the code space (protein then ligand).
        "atom_codebook_size": (
            2 * args.codebook_size
            if args.separate_protein_ckpt is not None
            else args.codebook_size
        ),
        "source": "biolip2_rigid_decoys",
        "n_decoys": args.n_decoys,
        "complexes_used": n_ok,
        "splits": {},
    }
    if args.separate_protein_ckpt is not None:
        meta["separate_tokenizers"] = True
    for split, w in writers.items():
        w.close()
        meta["splits"][split] = {
            "num_docs": w.num_docs,
            "num_tokens": w.num_tokens,
            "max_len": w.max_len,
        }
        logger.info("%s: %d docs (%d tokens)", split, w.num_docs, w.num_tokens)
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("wrote decoy token set to %s", args.out_dir)


if __name__ == "__main__":
    main()
