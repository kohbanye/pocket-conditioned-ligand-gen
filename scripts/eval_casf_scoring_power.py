"""CASF-2016 scoring + ranking power for the complex-token rescorer.

Docking power asks "which POSE is native-like"; scoring/ranking power ask "how
STRONG is the binding" -- a different question. Our head regresses pose RMSD, so
it is a pose selector by design and is expected to carry little affinity signal;
the MLM's masked pseudo-log-likelihood of the crystal complex is a plausible
affinity proxy (a more "likely" complex may be a better binder). This script
measures both honestly on the 285 crystal poses.

- scoring power : Pearson R between the score and experimental logKa.
- ranking power : mean Spearman within each CASF cluster (5 ligands / target).

Run::

    uv run python scripts/eval_casf_scoring_power.py --mlm-ckpt ... --vqvae-ckpt ... \
        --norm-stats ... --rescore-ckpt ... --out-csv out.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from prolit.config import (
    AtomVQVAETrainingConfig,
    MLMTrainingConfig,
    PocketExtractionConfig,
    ProLITMLMConfig,
)
from prolit.model.mlm_module import ProLITMLMModule
from prolit.model.mlm_score import ligand_pll
from prolit.tokenizers.lm_vocab import AtomLMVocab
from prolit.tokenizers.loaders import load_atom_vqvae
from prolit.tokenizers.pose_encoder import PoseEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_affinity(path: Path) -> dict[str, tuple[float, str]]:
    """pdbid -> (logKa, cluster). CoreSet.dat: #code resl year logKa Ka target."""
    out: dict[str, tuple[float, str]] = {}
    for ln in path.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        c = ln.split()
        if len(c) >= 6:  # noqa: PLR2004
            out[c[0].lower()] = (float(c[3]), c[5])
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:  # noqa: PLR2004
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(set(x.tolist())) < 2:  # noqa: PLR2004
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return _pearson(rx.astype(float), ry.astype(float))


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    p = argparse.ArgumentParser()
    p.add_argument("--mlm-ckpt", type=Path, required=True)
    p.add_argument("--vqvae-ckpt", type=Path, required=True)
    p.add_argument("--norm-stats", type=Path, required=True)
    p.add_argument("--rescore-ckpt", type=Path, default=None)
    p.add_argument(
        "--affinity-head",
        action="store_true",
        help="The head regresses pK (higher = stronger), so use its raw output. "
        "Without this the head is treated as an RMSD head (lower = better) and "
        "its sign is flipped.",
    )
    p.add_argument(
        "--efficiency-head",
        action="store_true",
        help="Head was trained on ligand efficiency (pK / heavy-atom count); "
        "multiply its output by the ligand token count to recover pK. Implies "
        "--affinity-head.",
    )
    p.add_argument("--casf-dir", type=Path, default=Path("data/casf2016"))
    p.add_argument("--codebook-size", type=int, default=8192)
    p.add_argument("--max-residues", type=int, default=50)
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()

    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sibling module; see the note in generate_ligands_for_target.py.

    from prolit.tokenizers.ligand import parse_sdf  # noqa: PLC0415

    vq_cfg = AtomVQVAETrainingConfig()
    vq_cfg.atom.codebook_size = args.codebook_size
    module = load_atom_vqvae(args.vqvae_ckpt, device)
    module.eval().to(device)
    norm = torch.load(args.norm_stats, weights_only=False)
    module.vqvae.set_normalization(norm["atom_mean"], norm["atom_std"])

    mlm_cfg = MLMTrainingConfig(
        model=ProLITMLMConfig(atom_codebook_size=args.codebook_size)
    )
    mlm = ProLITMLMModule.load_from_checkpoint(
        args.mlm_ckpt, config=mlm_cfg, map_location=device
    ).model
    mlm.eval().to(device)
    mask_id = mlm_cfg.model.mask_token_id

    enc = PoseEncoder(
        module.vqvae,
        norm["atom_mean"].numpy(),
        norm["atom_std"].numpy(),
        AtomLMVocab(codebook_size=args.codebook_size),
        device,
        PocketExtractionConfig(max_residues=args.max_residues),
    )

    rescorer = None
    if args.rescore_ckpt is not None:
        from prolit.config import RescoreTrainingConfig  # noqa: PLC0415
        from prolit.data.rescore_dataset import ligand_mask  # noqa: PLC0415
        from prolit.model.rescore_module import ComplexRescoreModule  # noqa: PLC0415

        rescorer = ComplexRescoreModule.load_from_checkpoint(
            args.rescore_ckpt,
            config=RescoreTrainingConfig(
                model=ProLITMLMConfig(atom_codebook_size=args.codebook_size)
            ),
            map_location=device,
        )
        rescorer.eval().to(device)

    aff = _load_affinity(args.casf_dir / "power_scoring" / "CoreSet.dat")
    rows = []
    for tid in sorted(
        p_.name for p_ in (args.casf_dir / "coreset").iterdir() if p_.is_dir()
    ):
        if tid not in aff:
            continue
        prot = args.casf_dir / "coreset" / tid / f"{tid}_protein.pdb"
        sdf = args.casf_dir / "coreset" / tid / f"{tid}_ligand.sdf"
        if not (prot.exists() and sdf.exists()):
            continue
        try:
            native = parse_sdf(sdf)[0]
            heavy = np.array(
                [(a[1], a[2], a[3]) for a in native["atoms"] if a[0] != "H"], np.float32
            )
            setup = enc.setup_pocket(prot.read_text(), heavy)
            if setup is None:
                continue
            p_codes, frame = setup
            seq = enc.ligand_seq(p_codes, native, frame)
            if seq is None:
                continue
            pll = ligand_pll(mlm, seq, mask_id, device)
            head = float("nan")
            if rescorer is not None:
                ids = torch.tensor([seq], device=device)
                lig_mask = ligand_mask(np.asarray(seq))
                with torch.no_grad():
                    raw = float(
                        rescorer(
                            {
                                "input_ids": ids,
                                "attention_mask": torch.ones_like(ids),
                                "ligand_mask": torch.tensor(
                                    lig_mask, device=device
                                ).unsqueeze(0),
                            }
                        ).item()
                    )
                if args.efficiency_head:
                    # head predicts pK per heavy atom -> multiply back by size.
                    raw *= max(int(lig_mask.sum()), 1)
                # affinity head predicts pK (higher = stronger); the pose head
                # predicts RMSD (lower = better), so only the latter is flipped.
                head = raw if (args.affinity_head or args.efficiency_head) else -raw
            logka, cluster = aff[tid]
            rows.append((tid, logka, cluster, pll, head))
            logger.info("%s: logKa %.2f | PLL %.3f | head %.3f", tid, logka, pll, head)
        except Exception:
            logger.exception("target %s failed", tid)
            continue

    if not rows:
        logger.info("no targets scored")
        return
    logka = np.array([r[1] for r in rows])
    plls = np.array([r[3] for r in rows])
    heads = np.array([r[4] for r in rows])

    logger.info("=== CASF-2016 scoring power (Pearson R vs logKa), n=%d ===", len(rows))
    logger.info("  MLM PLL : R = %.3f", _pearson(plls, logka))
    if rescorer is not None:
        logger.info("  head    : R = %.3f", _pearson(heads, logka))

    # ranking power: mean Spearman within each cluster
    by_cluster: dict[str, list] = {}
    for r in rows:
        by_cluster.setdefault(r[2], []).append(r)
    for name, idx in (("MLM PLL", 3), ("head", 4)):
        if name == "head" and rescorer is None:
            continue
        sps = []
        for entries in by_cluster.values():
            if len(entries) < 3:  # noqa: PLR2004
                continue
            s = np.array([e[idx] for e in entries])
            a = np.array([e[1] for e in entries])
            sps.append(_spearman(s, a))
        logger.info(
            "=== ranking power (mean in-cluster Spearman) %s: %.3f (%d clusters) ===",
            name,
            float(np.nanmean(sps)),
            len(sps),
        )

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w") as f:
            f.write("pdbid,logka,cluster,pll,head\n")
            for tid, lk, cl, pl, hd in rows:
                f.write(f"{tid},{lk},{cl},{pl:.6f},{hd:.6f}\n")
        logger.info("wrote %d rows to %s", len(rows), args.out_csv)


if __name__ == "__main__":
    main()
