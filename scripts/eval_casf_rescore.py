"""Fail-fast CASF-2016 docking-power test for the complex-token MLM rescorer.

Central hypothesis: the masked pseudo-log-likelihood of a ligand pose under the
complex-token MLM ranks a native-like pose above decoys. For each CASF-2016 core
target we extract the pocket ONCE around the crystal ligand (the protein does
not move, so protein codes are fixed and only the ligand codes vary with pose),
encode the native + ~100 docking decoys with the all-atom VQ-VAE, and score each
by ligand PLL (:func:`src.model.mlm_score.ligand_pll`).

Metrics (subset unless --max-targets is unset):
- docking power: fraction of targets whose top-PLL pose is within 2 A RMSD.
- ranking quality: mean Spearman(-PLL, RMSD) across targets (native-like should
  score high, so PLL should ANTI-correlate with RMSD).

Run (single GPU)::

    uv run python scripts/eval_casf_rescore.py \
        --mlm-ckpt pocket-ligand-mlm/liqftueb/checkpoints/mlm-e00-vl1.1349.ckpt \
        --vqvae-ckpt "<atom-vqvae>.ckpt" \
        --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
        --max-targets 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from src.config import (
    AtomVQVAETrainingConfig,
    ComplexMLMConfig,
    MLMTrainingConfig,
    PocketExtractionConfig,
)
from src.data.descriptors import collate_molecules
from src.model.mlm_module import ComplexMLMModule
from src.model.mlm_score import ligand_pll
from src.model.vqvae_module import AtomVQVAEModule
from src.tokenizers.atom import (
    LigandAtomDescriptor,
    ProteinAtomDescriptor,
    precompute_receptor_atom_features_from_text,
)
from src.tokenizers.lm_vocab import AtomLMVocab
from src.tokenizers.protein import (
    _compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    precompute_pocket_atom_candidates_from_text,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _mol_to_dict(mol) -> dict | None:  # noqa: ANN001
    from rdkit import Chem  # noqa: PLC0415

    try:
        conf = mol.GetConformer()
    except ValueError:
        return None
    bt = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    atoms = []
    for i, a in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        atoms.append((a.GetSymbol(), p.x, p.y, p.z))
    bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bt.get(b.GetBondType(), 1))
        for b in mol.GetBonds()
    ]
    return {"atoms": atoms, "bonds": bonds}


def _parse_mol2_multi(text: str) -> list[tuple[str, dict]]:
    """(pose_name, mol_dict) for each molecule in a multi-``@<TRIPOS>MOLECULE`` file."""
    from rdkit import Chem  # noqa: PLC0415

    out: list[tuple[str, dict]] = []
    for chunk in text.split("@<TRIPOS>MOLECULE")[1:]:
        block = "@<TRIPOS>MOLECULE" + chunk
        name = next((ln.strip() for ln in chunk.splitlines() if ln.strip()), "")
        mol = Chem.MolFromMol2Block(block, sanitize=False, removeHs=False)
        if mol is None:
            continue
        d = _mol_to_dict(mol)
        if d is not None:
            out.append((name, d))
    return out


class _PoseEncoder:
    """Fixed-pocket encoder: protein codes once, ligand codes per pose."""

    def __init__(self, module, mean, std, vocab, device, pocket_cfg) -> None:  # noqa: ANN001, PLR0913
        self.module = module
        self.mean = mean
        self.std = std
        self.vocab = vocab
        self.device = device
        self.pocket_cfg = pocket_cfg
        self.prot_desc = ProteinAtomDescriptor()
        self.lig_desc = LigandAtomDescriptor()

    def _encode(self, desc: np.ndarray) -> list[int]:
        x, mask = collate_molecules(
            [torch.from_numpy((desc - self.mean) / self.std).float()]
        )
        idx = self.module.vqvae.encode_batch(
            x.to(self.device), mask.to(self.device)
        ).cpu()
        return idx[0][mask[0]].tolist()

    def setup_pocket(self, protein_text: str, native_heavy: np.ndarray) -> tuple | None:
        """Extract the pocket around the crystal ligand; return (p_codes, frame)."""
        precomp = precompute_pocket_atom_candidates_from_text(protein_text)
        pocket = extract_pocket_atoms_from_candidates(
            precomp, native_heavy, self.pocket_cfg
        )
        if pocket is None or pocket.atom_coords.shape[0] == 0:
            return None
        feats = precompute_receptor_atom_features_from_text(protein_text)
        frame = _compute_canonical_frame(pocket.ca_coords.astype(np.float64))
        prot_desc, _ = self.prot_desc.compute(pocket, feats, frame)
        if prot_desc.shape[0] == 0:
            return None
        return self._encode(prot_desc), frame

    def ligand_seq(self, p_codes: list[int], mol: dict, frame) -> list[int] | None:  # noqa: ANN001
        lig_desc, _e, _m = self.lig_desc.compute(mol["atoms"], mol["bonds"], frame)
        if lig_desc.shape[0] == 0:
            return None
        return self.vocab.build_sequence(p_codes, self._encode(lig_desc))

    def ligand_seqs_batch(
        self, p_codes: list[int], mols: list[dict], frame: tuple
    ) -> list[list[int] | None]:
        """Encode MANY ligand poses in ONE VQ call (per-pose batch-1 is the
        decoy-generation bottleneck). Returns one sequence (or None) per mol."""
        descs = [self.lig_desc.compute(m["atoms"], m["bonds"], frame)[0] for m in mols]
        valid = [(i, d) for i, d in enumerate(descs) if d.shape[0] > 0]
        out: list[list[int] | None] = [None] * len(mols)
        if not valid:
            return out
        tensors = [
            torch.from_numpy((d - self.mean) / self.std).float() for _, d in valid
        ]
        x, mask = collate_molecules(tensors)
        idx = self.module.vqvae.encode_batch(
            x.to(self.device), mask.to(self.device)
        ).cpu()
        for k, (i, _) in enumerate(valid):
            out[i] = self.vocab.build_sequence(p_codes, idx[k][mask[k]].tolist())
        return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:  # noqa: PLR2004
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _zscore(x: np.ndarray) -> np.ndarray:
    """Standardize within a target's pose set so head + PLL are combinable."""
    s = float(x.std())
    return (x - x.mean()) / s if s > 1e-8 else x - x.mean()  # noqa: PLR2004


def main() -> None:  # noqa: C901, PLR0915, PLR0912
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlm-ckpt", type=Path, required=True)
    parser.add_argument("--vqvae-ckpt", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--casf-dir", type=Path, default=Path("data/casf2016"))
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--max-residues", type=int, default=50)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--max-decoys", type=int, default=None, help="Debug subset.")
    parser.add_argument("--native-thresh", type=float, default=2.0)
    parser.add_argument(
        "--score-mode",
        choices=["pll", "head", "ensemble"],
        default="pll",
        help="pll = zero-shot masked pseudo-likelihood; head = trained RMSD head.",
    )
    parser.add_argument(
        "--rescore-ckpt",
        type=Path,
        default=None,
        help="ComplexRescoreModule ckpt (head mode).",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "meanmax", "attn", "xattn", "pairsum"],
        default="mean",
        help="Must match the head ckpt's pooling (meanmax head has a 2H input).",
    )
    parser.add_argument(
        "--exclude-native",
        action="store_true",
        help="Rank docking decoys only (drop the crystal SDF native) -- the "
        "honest docking-power test, immune to a native-vs-decoy format shortcut.",
    )
    parser.add_argument(
        "--out-csv", type=Path, default=None, help="Per-target results CSV."
    )
    parser.add_argument(
        "--pll-weight",
        type=float,
        default=1.0,
        help="ensemble mode: weight on z(PLL) added to z(head).",
    )
    parser.add_argument(
        "--dump-scores",
        type=Path,
        default=None,
        help="Write per-pose (rmsd, head, pll) scores here for offline sweeps.",
    )
    args = parser.parse_args()

    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vq_cfg = AtomVQVAETrainingConfig()
    vq_cfg.atom.codebook_size = args.codebook_size
    module = AtomVQVAEModule.load_from_checkpoint(
        args.vqvae_ckpt, config=vq_cfg, map_location=device
    )
    module.eval().to(device)
    norm = torch.load(args.norm_stats, weights_only=False)
    module.vqvae.set_normalization(norm["atom_mean"], norm["atom_std"])
    mean, std = norm["atom_mean"].numpy(), norm["atom_std"].numpy()

    mlm_cfg = MLMTrainingConfig(
        model=ComplexMLMConfig(atom_codebook_size=args.codebook_size)
    )
    mlm = ComplexMLMModule.load_from_checkpoint(
        args.mlm_ckpt, config=mlm_cfg, map_location=device
    ).model
    mlm.eval().to(device)
    mask_id = mlm_cfg.model.mask_token_id
    vocab = AtomLMVocab(codebook_size=args.codebook_size)
    enc = _PoseEncoder(
        module,
        mean,
        std,
        vocab,
        device,
        PocketExtractionConfig(max_residues=args.max_residues),
    )

    need_head = args.score_mode in ("head", "ensemble")
    need_pll = args.score_mode in ("pll", "ensemble")
    rescorer = None
    if need_head:
        from src.config import RescoreTrainingConfig  # noqa: PLC0415
        from src.data.rescore_dataset import _ligand_mask  # noqa: PLC0415
        from src.model.rescore_module import ComplexRescoreModule  # noqa: PLC0415

        rescorer = ComplexRescoreModule.load_from_checkpoint(
            args.rescore_ckpt,
            config=RescoreTrainingConfig(
                model=ComplexMLMConfig(atom_codebook_size=args.codebook_size),
                pooling=args.pooling,
            ),
            map_location=device,
        )
        rescorer.eval().to(device)

    @torch.no_grad()
    def head_score(seq: list[int]) -> float:
        ids = torch.tensor([seq], device=device)
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "ligand_mask": torch.tensor(
                _ligand_mask(np.asarray(seq)), device=device
            ).unsqueeze(0),
        }
        return -float(rescorer(batch).item())  # lower predicted RMSD = higher score

    def pll_score(seq: list[int]) -> float:
        return ligand_pll(mlm, seq, mask_id, device)

    from src.tokenizers.ligand import parse_sdf  # noqa: PLC0415

    targets = sorted(
        p.name for p in (args.casf_dir / "coreset").iterdir() if p.is_dir()
    )
    if args.max_targets is not None:
        targets = targets[: args.max_targets]

    successes = 0
    scored = 0
    spearmans: list[float] = []
    rows: list[tuple] = []
    pose_rows: list[tuple] = []
    for tid in targets:
        prot = args.casf_dir / "coreset" / tid / f"{tid}_protein.pdb"
        native_sdf = args.casf_dir / "coreset" / tid / f"{tid}_ligand.sdf"
        decoys = args.casf_dir / "decoys_docking" / f"{tid}_decoys.mol2"
        rmsd_dat = args.casf_dir / "decoys_docking" / f"{tid}_rmsd.dat"
        if not (prot.exists() and native_sdf.exists() and decoys.exists()):
            continue
        try:
            native = parse_sdf(native_sdf)[0]
            native_heavy = np.array(
                [(a[1], a[2], a[3]) for a in native["atoms"] if a[0] != "H"], np.float32
            )
            setup = enc.setup_pocket(prot.read_text(), native_heavy)
            if setup is None:
                continue
            p_codes, frame = setup
            rmsd = {}
            for ln in rmsd_dat.read_text().splitlines():
                if ln.startswith("#") or not ln.strip():
                    continue
                name, val = ln.split()[:2]
                rmsd[name] = float(val)
            poses = _parse_mol2_multi(decoys.read_text())
            if args.max_decoys is not None:
                poses = poses[: args.max_decoys]

            names, rmsds = [], []
            raw_heads, raw_plls = [], []
            # Crystal native pose (RMSD 0), parsed from SDF. Optionally excluded:
            # it is easy to identify (and parsed by a different reader than the
            # mol2 decoys), so ranking decoys-only is the honest docking-power
            # test -- some docking decoys are themselves < 2 A.
            if not args.exclude_native:
                seq = enc.ligand_seq(p_codes, native, frame)
                if seq is not None:
                    names.append(f"{tid}_native")
                    rmsds.append(0.0)
                    if need_head:
                        raw_heads.append(head_score(seq))
                    if need_pll:
                        raw_plls.append(pll_score(seq))
            for name, mol in poses:
                if name not in rmsd:
                    continue
                seq = enc.ligand_seq(p_codes, mol, frame)
                if seq is None:
                    continue
                names.append(name)
                rmsds.append(rmsd[name])
                if need_head:
                    raw_heads.append(head_score(seq))
                if need_pll:
                    raw_plls.append(pll_score(seq))
        except Exception:
            logger.exception("target %s failed", tid)
            continue

        n_scored = len(raw_heads) if need_head else len(raw_plls)
        if n_scored < 3:  # noqa: PLR2004
            continue
        rmsds_a = np.array(rmsds)
        if args.dump_scores is not None:
            for i, nm in enumerate(names):
                pose_rows.append(
                    (
                        tid,
                        nm,
                        float(rmsds_a[i]),
                        float(raw_heads[i]) if need_head else float("nan"),
                        float(raw_plls[i]) if need_pll else float("nan"),
                    )
                )
        if args.score_mode == "ensemble":
            plls_a = _zscore(np.array(raw_heads)) + args.pll_weight * _zscore(
                np.array(raw_plls)
            )
        elif need_head:
            plls_a = np.array(raw_heads)
        else:
            plls_a = np.array(raw_plls)
        top = int(np.argmax(plls_a))  # highest PLL = predicted best pose
        success = rmsds_a[top] <= args.native_thresh
        successes += int(success)
        scored += 1
        sp = _spearman(-plls_a, rmsds_a)  # anti-correlate: high PLL <-> low RMSD
        spearmans.append(sp)
        rows.append((tid, float(rmsds_a[top]), int(success), sp, len(plls_a)))
        logger.info(
            "%s: top-pose RMSD %.2f (%s) | spearman(-PLL,RMSD) %.2f | %d poses",
            tid,
            rmsds_a[top],
            "HIT" if success else "miss",
            sp,
            len(plls_a),
        )

    if scored:
        logger.info(
            "=== CASF docking power (top1<=%.1fA): %d/%d = %.1f%% | mean rho %.2f ===",
            args.native_thresh,
            successes,
            scored,
            100 * successes / scored,
            float(np.nanmean(spearmans)),
        )
    else:
        logger.info("no targets scored")

    if args.out_csv is not None and rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w") as f:
            f.write("pdbid,top_pose_rmsd,success,spearman,n_poses\n")
            for tid, r, ok, sp, n in rows:
                f.write(f"{tid},{r:.3f},{ok},{sp:.3f},{n}\n")
        logger.info("wrote %d rows to %s", len(rows), args.out_csv)

    if args.dump_scores is not None and pose_rows:
        args.dump_scores.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_scores.open("w") as f:
            f.write("pdbid,pose,rmsd,head,pll\n")
            for tid, nm, r, h, pl in pose_rows:
                f.write(f"{tid},{nm},{r:.3f},{h:.6f},{pl:.6f}\n")
        logger.info("wrote %d per-pose scores to %s", len(pose_rows), args.dump_scores)


if __name__ == "__main__":
    main()
