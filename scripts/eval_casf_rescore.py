"""Fail-fast CASF-2016 docking-power test for the complex-token MLM rescorer.

Central hypothesis: the masked pseudo-log-likelihood of a ligand pose under the
complex-token MLM ranks a native-like pose above decoys. For each CASF-2016 core
target we extract the pocket ONCE around the crystal ligand (the protein does
not move, so protein codes are fixed and only the ligand codes vary with pose),
encode the native + ~100 docking decoys with the all-atom VQ-VAE, and score each
by ligand PLL (:func:`prolit.model.mlm_score.ligand_pll`).

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

from prolit.chem.mol2 import parse_mol2_multi
from prolit.config import (
    AtomVQVAETrainingConfig,
    ComplexMLMConfig,
    MLMTrainingConfig,
    PocketExtractionConfig,
)
from prolit.model.mlm_module import ComplexMLMModule
from prolit.model.mlm_score import ligand_pll
from prolit.model.vqvae_module import AtomVQVAEModule
from prolit.seeding import add_seed_argument, seed_from_args
from prolit.tokenizers.geometry import random_rotation_matrix
from prolit.tokenizers.lm_vocab import AtomLMVocab
from prolit.tokenizers.pose_encoder import PoseEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    add_seed_argument(parser)
    args = parser.parse_args()
    seed_from_args(args)

    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vq_cfg = AtomVQVAETrainingConfig()
    vq_cfg.atom.codebook_size = args.codebook_size
    if args.separate_protein_ckpt is not None:
        from prolit.tokenizers.descriptor_schema import (  # noqa: PLC0415
            ATOM_DESCRIPTOR_DIM,
        )
        from prolit.tokenizers.separate_vqvae import SeparateVQVAE  # noqa: PLC0415

        module = SeparateVQVAE.from_checkpoints(
            args.separate_protein_ckpt, args.separate_protein_norm,
            args.separate_ligand_ckpt, args.separate_ligand_norm,
            device, codebook_size=args.codebook_size,
        )
        # SeparateVQVAE normalizes per modality internally, so feed RAW
        # descriptors through PoseEncoder (identity external normalization).
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
    enc = PoseEncoder(
        module.vqvae,
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
        from prolit.config import RescoreTrainingConfig  # noqa: PLC0415
        from prolit.data.rescore_dataset import ligand_mask  # noqa: PLC0415
        from prolit.model.rescore_module import ComplexRescoreModule  # noqa: PLC0415

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
                lig[j, : len(seq)] = torch.from_numpy(ligand_mask(arr))
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

    from prolit.tokenizers.ligand import parse_sdf  # noqa: PLC0415

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
            poses = parse_mol2_multi(decoys.read_text())
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
                        rot = random_rotation_matrix(np.random.default_rng(1000 + r))
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
