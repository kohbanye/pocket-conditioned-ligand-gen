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

    uv run python scripts/tokenize_decoys.py \
        --ckpt "<atom-vqvae>.ckpt" \
        --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
        --n-complexes 12000 --n-decoys 16 --out-dir data/lm_tokens_decoys
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from scripts.eval_casf_rescore import _PoseEncoder
from scripts.tokenize_biolip import (
    _bucket_code,
    _cd_test_pdbs,
    _load_ccd_smiles,
    _parse_biolip_txt,
    _read_needed,
)
from src.config import AtomVQVAETrainingConfig, PocketExtractionConfig
from src.model.vqvae_module import AtomVQVAEModule
from src.tokenizers.ligand import parse_ligand_pdb_text
from src.tokenizers.lm_vocab import AtomLMVocab

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


class _RmsdWriter:
    """Streams tokens (.bin/.len) + one float32 RMSD per doc (.rmsd)."""

    def __init__(self, out_dir: Path, split: str) -> None:
        self._bin = (out_dir / f"{split}.bin").open("wb")
        self._len = (out_dir / f"{split}.len").open("wb")
        self._rmsd = (out_dir / f"{split}.rmsd").open("wb")
        self.num_docs = 0
        self.num_tokens = 0
        self.max_len = 0

    def write(self, seq: list[int], rmsd: float) -> None:
        arr = np.asarray(seq, dtype=np.uint16)
        self._bin.write(arr.tobytes())
        self._len.write(np.asarray([len(seq)], dtype=np.uint16).tobytes())
        self._rmsd.write(np.asarray([rmsd], dtype=np.float32).tobytes())
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

    def close(self) -> None:
        self._bin.close()
        self._len.close()
        self._rmsd.close()


def main() -> None:  # noqa: C901, PLR0915, PLR0912
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
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
    args = parser.parse_args()

    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = AtomVQVAETrainingConfig()
    cfg.atom.codebook_size = args.codebook_size
    module = AtomVQVAEModule.load_from_checkpoint(
        args.ckpt, config=cfg, map_location=device
    )
    module.eval().to(device)
    norm = torch.load(args.norm_stats, weights_only=False)
    module.vqvae.set_normalization(norm["atom_mean"], norm["atom_std"])
    enc = _PoseEncoder(
        module,
        norm["atom_mean"].numpy(),
        norm["atom_std"].numpy(),
        AtomLMVocab(codebook_size=args.codebook_size),
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

    n_ok = 0
    for code in tqdm(sorted(by_bucket), desc="buckets"):
        site_list = by_bucket[code]
        needed_rec = {f"{p}{rc}.pdb" for p, rc, _c, _l, _s in site_list}
        needed_lig = {f"{p}_{cc}_{lc}_{s}.pdb" for p, _rc, cc, lc, s in site_list}
        # _read_needed uses a module-global biolip dir; set it once.
        import scripts.tokenize_biolip as tb  # noqa: PLC0415

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
                for k in range(args.n_decoys):
                    scale = (k + 1) / args.n_decoys
                    if k % 2 == 0:
                        new = _perturb(base, rng, scale)[0]
                    else:
                        new = _conf_perturb(mol["atoms"], mol["bonds"], rng, scale)
                        if new is None:
                            new = _perturb(base, rng, scale)[0]
                    rmsds.append(
                        float(np.sqrt(((new[hidx] - base[hidx]) ** 2).sum(1).mean()))
                    )
                    mols.append({
                        "atoms": [
                            (a[0], float(new[i][0]), float(new[i][1]), float(new[i][2]))
                            for i, a in enumerate(mol["atoms"])
                        ],
                        "bonds": mol["bonds"],
                    })
                seqs = enc.ligand_seqs_batch(p_codes, mols, frame)
                if seqs[0] is None:
                    continue  # native must encode
                for seq, rmsd in zip(seqs, rmsds, strict=True):
                    if seq is not None:
                        writers[split].write(seq, rmsd)
                n_ok += 1
            except Exception:
                logger.exception("failed %s_%s", pdb, ccd)
                continue

    meta = {
        "vocab_size": AtomLMVocab(codebook_size=args.codebook_size).vocab_size,
        "atom_codebook_size": args.codebook_size,
        "source": "biolip2_rigid_decoys",
        "n_decoys": args.n_decoys,
        "complexes_used": n_ok,
        "splits": {},
    }
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
