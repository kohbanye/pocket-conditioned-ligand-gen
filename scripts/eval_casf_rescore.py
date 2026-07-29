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
    rotate_atom_descriptor,
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


_MOL2_BOND = {"1": 1, "2": 2, "3": 3, "ar": 4, "am": 1}


def _mol2_records(block: str) -> dict | None:  # noqa: C901
    """Read ATOM/BOND records straight out of a mol2 block.

    RDKit's mol2 reader rejects some CASF decoy blocks outright -- e.g. the
    peptide ligand of 3uri, whose near-native poses carry extra ``NORMAL`` /
    ``ALT_TYPE`` sections. Those poses were then silently missing from the
    scored set (for 3uri, that was every pose under 2 A, making the target
    unwinnable). The head only needs elements, coordinates and bond orders, all
    of which are plain columns here.
    """
    if "@<TRIPOS>ATOM" not in block:
        return None
    atoms: list[tuple[str, float, float, float]] = []
    for ln in block.split("@<TRIPOS>ATOM")[1].split("@<TRIPOS>")[0].splitlines():
        c = ln.split()
        if len(c) < 6:  # noqa: PLR2004
            continue
        try:
            x, y, z = float(c[2]), float(c[3]), float(c[4])
        except ValueError:
            continue
        # SYBYL type "C.3" / "N.ar" -> element; "Du"/"LP" dummies are skipped.
        el = c[5].split(".")[0]
        if el in ("Du", "LP"):
            continue
        atoms.append((el.capitalize(), x, y, z))
    if not atoms:
        return None
    bonds: list[tuple[int, int, int]] = []
    if "@<TRIPOS>BOND" in block:
        for ln in block.split("@<TRIPOS>BOND")[1].split("@<TRIPOS>")[0].splitlines():
            c = ln.split()
            if len(c) < 4:  # noqa: PLR2004
                continue
            try:
                i, j = int(c[1]) - 1, int(c[2]) - 1
            except ValueError:
                continue
            if 0 <= i < len(atoms) and 0 <= j < len(atoms):
                bonds.append((i, j, _MOL2_BOND.get(c[3], 1)))
    return {"atoms": atoms, "bonds": bonds}


def _parse_mol2_multi(text: str) -> list[tuple[str, dict]]:
    """(pose_name, mol_dict) for each molecule in a multi-``@<TRIPOS>MOLECULE`` file."""
    from rdkit import Chem  # noqa: PLC0415

    out: list[tuple[str, dict]] = []
    for chunk in text.split("@<TRIPOS>MOLECULE")[1:]:
        block = "@<TRIPOS>MOLECULE" + chunk
        name = next((ln.strip() for ln in chunk.splitlines() if ln.strip()), "")
        mol = Chem.MolFromMol2Block(block, sanitize=False, removeHs=False)
        d = _mol_to_dict(mol) if mol is not None else None
        if d is None:
            d = _mol2_records(block)  # RDKit refused this block
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
        # Kept for rotated re-encodings of this same pocket (TTA).
        self.prot_desc_raw = prot_desc
        return self._encode(prot_desc), frame

    def pocket_codes_rotated(self, rotation: np.ndarray) -> list[int]:
        """Re-encode the cached pocket descriptor under an extra frame rotation."""
        return self._encode(rotate_atom_descriptor(self.prot_desc_raw, rotation))

    def ligand_descs(self, mols: list[dict], frame: tuple) -> list[np.ndarray]:
        """Descriptors (pre-quantization) for many poses, computed once so that
        rotated variants come from :func:`rotate_atom_descriptor` instead of a
        full recompute."""
        return [self.lig_desc.compute(m["atoms"], m["bonds"], frame)[0] for m in mols]

    def seqs_from_descs(
        self,
        p_codes: list[int],
        descs: list[np.ndarray],
        rotation: np.ndarray | None = None,
        batch_size: int = 64,
    ) -> list[list[int] | None]:
        """Quantize pose descriptors (optionally rotated) into token sequences."""
        out: list[list[int] | None] = [None] * len(descs)
        valid = [(i, d) for i, d in enumerate(descs) if d.shape[0] > 0]
        for s in range(0, len(valid), batch_size):
            chunk = valid[s : s + batch_size]
            arrs = [
                d if rotation is None else rotate_atom_descriptor(d, rotation)
                for _, d in chunk
            ]
            tensors = [
                torch.from_numpy((a - self.mean) / self.std).float() for a in arrs
            ]
            x, mask = collate_molecules(tensors)
            idx = self.module.vqvae.encode_batch(
                x.to(self.device), mask.to(self.device)
            ).cpu()
            for k, (i, _) in enumerate(chunk):
                out[i] = self.vocab.build_sequence(p_codes, idx[k][mask[k]].tolist())
        return out

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


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random 3D rotation (QR of a Gaussian matrix, det fixed to +1)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _zscore(x: np.ndarray) -> np.ndarray:
    """Standardize within a target's pose set so head + PLL are combinable."""
    s = float(x.std())
    return (x - x.mean()) / s if s > 1e-8 else x - x.mean()  # noqa: PLR2004


def main() -> None:  # noqa: C901, PLR0915, PLR0912
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlm-ckpt", type=Path, required=True)
    parser.add_argument("--vqvae-ckpt", type=Path, default=None)
    # Separate-tokenizer ablation: a protein-only and a ligand-only VQ-VAE whose
    # codes are unified into one space (protein ids first, then ligand ids
    # offset by --codebook-size). Mirrors tokenize_decoys.py's separate mode, so
    # a head trained on separate-tokenized decoys can be evaluated here.
    parser.add_argument("--separate-protein-ckpt", type=Path, default=None)
    parser.add_argument("--separate-protein-norm", type=Path, default=None)
    parser.add_argument("--separate-ligand-ckpt", type=Path, default=None)
    parser.add_argument("--separate-ligand-norm", type=Path, default=None)
    parser.add_argument("--norm-stats", type=Path, default=None)
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
        "--tta-rotations",
        type=int,
        default=1,
        help="Average the head over this many random frame rotations (1 = off). "
        "Same physical pose, different VQ codes -> cancels quantization noise.",
    )
    parser.add_argument(
        "--interaction-layers",
        type=int,
        default=0,
        help="Must match the head ckpt: trainable transformer layers before pooling.",
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
    if args.separate_protein_ckpt is not None:
        from src.tokenizers.descriptor_schema import (  # noqa: PLC0415
            ATOM_DESCRIPTOR_DIM,
        )
        from src.tokenizers.separate_vqvae import SeparateVQVAE  # noqa: PLC0415

        module = SeparateVQVAE.from_checkpoints(
            args.separate_protein_ckpt, args.separate_protein_norm,
            args.separate_ligand_ckpt, args.separate_ligand_norm,
            device, codebook_size=args.codebook_size,
        )
        # SeparateVQVAE normalizes per modality internally, so feed RAW
        # descriptors through _PoseEncoder (identity external normalization).
        mean = np.zeros(ATOM_DESCRIPTOR_DIM, dtype=np.float32)
        std = np.ones(ATOM_DESCRIPTOR_DIM, dtype=np.float32)
        vocab_codebook = 2 * args.codebook_size
    else:
        module = AtomVQVAEModule.load_from_checkpoint(
            args.vqvae_ckpt, config=vq_cfg, map_location=device
        )
        module.eval().to(device)
        norm = torch.load(args.norm_stats, weights_only=False)
        module.vqvae.set_normalization(norm["atom_mean"], norm["atom_std"])
        mean, std = norm["atom_mean"].numpy(), norm["atom_std"].numpy()
        vocab_codebook = args.codebook_size

    # In separate-tokenizer mode the MLM's code space is the COMBINED one
    # (protein ids then ligand ids), i.e. twice the per-modality codebook.
    mlm_cfg = MLMTrainingConfig(
        model=ComplexMLMConfig(atom_codebook_size=vocab_codebook)
    )
    mlm = ComplexMLMModule.load_from_checkpoint(
        args.mlm_ckpt, config=mlm_cfg, map_location=device
    ).model
    mlm.eval().to(device)
    mask_id = mlm_cfg.model.mask_token_id
    vocab = AtomLMVocab(codebook_size=vocab_codebook)
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

        # Prefer the config stored in the checkpoint: it records every option
        # that changes the module's parameters (pooling, interaction layers, the
        # per-atom auxiliary head, ...). Rebuilding it from CLI flags silently
        # drifts -- an atom-aux checkpoint failed to load because the eval had no
        # such flag, so its atom_head weights had nowhere to go.
        saved = torch.load(
            args.rescore_ckpt, map_location="cpu", weights_only=False
        ).get("hyper_parameters", {})
        cfg = saved.get("config")
        if cfg is None:
            cfg = RescoreTrainingConfig(
                model=ComplexMLMConfig(atom_codebook_size=args.codebook_size),
                pooling=args.pooling,
                head_interaction_layers=args.interaction_layers,
            )
        else:
            logger.info(
                "using checkpoint config: pooling=%s interaction=%d atom_aux=%.2f",
                cfg.pooling,
                cfg.head_interaction_layers,
                cfg.atom_aux_weight,
            )
        rescorer = ComplexRescoreModule.load_from_checkpoint(
            args.rescore_ckpt, config=cfg, map_location=device
        )
        rescorer.eval().to(device)

    @torch.no_grad()
    def head_scores(seqs: list[list[int]], batch_size: int = 32) -> list[float]:
        """Score many poses per forward pass (one-pose-per-call made a full CASF
        eval GPU-idle-bound, which matters once TTA multiplies the pose count)."""
        out: list[float] = []
        for s in range(0, len(seqs), batch_size):
            chunk = seqs[s : s + batch_size]
            n = max(len(x) for x in chunk)
            ids = torch.zeros((len(chunk), n), dtype=torch.long)
            attn = torch.zeros((len(chunk), n), dtype=torch.long)
            lig = torch.zeros((len(chunk), n), dtype=torch.bool)
            for j, seq in enumerate(chunk):
                arr = np.asarray(seq)
                ids[j, : len(seq)] = torch.from_numpy(arr.astype(np.int64))
                attn[j, : len(seq)] = 1
                lig[j, : len(seq)] = torch.from_numpy(_ligand_mask(arr))
            batch = {
                "input_ids": ids.to(device),
                "attention_mask": attn.to(device),
                "ligand_mask": lig.to(device),
            }
            # lower predicted RMSD = higher score
            out.extend((-rescorer(batch)).float().cpu().tolist())
        return out

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

            # Crystal native pose (RMSD 0), parsed from SDF. Optionally excluded:
            # it is easy to identify (and parsed by a different reader than the
            # mol2 decoys), so ranking decoys-only is the honest docking-power
            # test -- some docking decoys are themselves < 2 A.
            cand = [] if args.exclude_native else [(f"{tid}_native", native, 0.0)]
            cand += [(nm, m, rmsd[nm]) for nm, m in poses if nm in rmsd]
            descs = enc.ligand_descs([m for _, m, _ in cand], frame)
            seqs0 = enc.seqs_from_descs(p_codes, descs)
            keep = [i for i, s in enumerate(seqs0) if s is not None]
            names = [cand[i][0] for i in keep]
            rmsds = [cand[i][2] for i in keep]
            raw_plls: list[float] = []
            raw_heads = np.zeros(len(keep))
            if need_head:
                # Test-time rotation augmentation: the tokens are expressed in the
                # pocket's PCA frame, so an extra global rotation re-quantizes the
                # SAME physical complex into different codes (the VQ-VAE was
                # pretrained with this augmentation, so rotated frames are in
                # distribution). Averaging the predictions over rotations cancels
                # quantization noise, which is exactly what separates a 0.5 A pose
                # from a 1.5 A one.
                for r in range(args.tta_rotations):
                    if r == 0:
                        seqs = [seqs0[i] for i in keep]
                        pc = p_codes
                    else:
                        rot = _random_rotation(np.random.default_rng(1000 + r))
                        pc = enc.pocket_codes_rotated(rot)
                        srot = enc.seqs_from_descs(pc, descs, rotation=rot)
                        seqs = [srot[i] if srot[i] is not None else seqs0[i]
                                for i in keep]
                    raw_heads += np.asarray(head_scores(seqs))
                raw_heads /= max(1, args.tta_rotations)
            if need_pll:
                raw_plls = [pll_score(seqs0[i]) for i in keep]
            raw_heads = list(raw_heads)
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
