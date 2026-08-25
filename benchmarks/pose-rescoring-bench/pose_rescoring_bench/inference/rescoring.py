"""CASF-2016 pose-rescoring inference -> per-pose dump (pdbid,pose,rmsd,head,pll).

Ported from ``scripts/eval_casf_docking_power.py`` (inference half only; metrics live
in :mod:`pose_rescoring_bench.metrics.rescoring`). For each target we extract the pocket
once
around the crystal ligand, then score the docking decoys (native excluded by
default — the honest docking-power protocol).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from prolit.model.mlm_score import ligand_pll
from prolit.seeding import rng_for
from prolit.tokenizers.geometry import random_rotation_matrix
from prolit.tokenizers.ligand import parse_sdf

from pose_rescoring_bench.inference.encode import (
    load_mlm,
    load_rescorer,
    load_tokenizer,
    make_encoder,
    parse_mol2_multi,
    resolve_rescore_ckpt,
    sequence_ligand_mask,
)

if TYPE_CHECKING:
    from pathlib import Path

    from prolit.api import PoseEncoder

    from pose_rescoring_bench.config import PathsConfig, RescoringConfig
    from pose_rescoring_bench.variants import RescoringCkpts

logger = logging.getLogger(__name__)
_MIN_SCORED = 3


def _read_rmsd(path: Path) -> dict[str, float]:
    rmsd: dict[str, float] = {}
    for ln in path.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        name, val = ln.split()[:2]
        rmsd[name] = float(val)
    return rmsd


def score_casf(
    ckpts: RescoringCkpts,
    paths: PathsConfig,
    cfg: RescoringConfig,
    head_index: int = 0,
) -> pd.DataFrame:
    """Score CASF poses with one variant head (vqvae, mlm, head) -> per-pose DataFrame.

    ``head_index`` selects which pose head to use; run once per head to build the
    multi-head ensemble downstream. ``pll`` is head-independent and is filled
    only when ``cfg.score_mode`` reads it -- it is the more expensive of the two
    scorers, so a head-mode run does not pay for a column nothing consumes.
    The codebook size is taken from ``ckpts.codebook_size`` (8192 for the joint
    tokenizer, 16384 for the separate protein+ligand tokenizer).
    """
    torch.set_float32_matmul_precision("high")
    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]  # rdkit C-ext
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    codebook_size = ckpts.codebook_size
    module, mean, std = load_tokenizer(ckpts, paths, device)
    mlm_ckpt = ckpts.mlm
    if mlm_ckpt is None:
        msg = "rescoring variant is missing its mlm checkpoint"
        raise ValueError(msg)
    mlm, mask_id = load_mlm(paths.ckpt(mlm_ckpt), codebook_size, device)
    enc = make_encoder(
        module,  # ty: ignore[invalid-argument-type]  # SeparateVQVAE duck-types the module
        mean,
        std,
        codebook_size,
        device,
        cfg.max_residues,
    )
    head_spec = ckpts.heads[head_index] if ckpts.heads else None
    rescorer = (
        load_rescorer(
            resolve_rescore_ckpt(paths.source_repo, head_spec.ckpt),
            codebook_size,
            device,
        )
        if head_spec is not None
        else None
    )

    casf = paths.casf_dir
    targets = sorted(p.name for p in (casf / "coreset").iterdir() if p.is_dir())
    if cfg.max_targets is not None:
        targets = targets[: cfg.max_targets]

    rows: list[dict] = []
    for tid in targets:
        rows.extend(_score_target(tid, casf, enc, mlm, mask_id, rescorer, cfg, device))
    cols = ["pdbid", "pose", "rmsd", "head", "pll"]
    return pd.DataFrame(rows, columns=cols)  # ty: ignore[invalid-argument-type]


def _head_scores(
    seqs: list[list[int] | None],
    rescorer: Any,  # noqa: ANN401  (source-repo head)
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Predicted RMSD for each sequence (NaN where the pose did not encode)."""
    out = np.full(len(seqs), np.nan)
    idx = [i for i, q in enumerate(seqs) if q is not None]
    for start in range(0, len(idx), batch_size):
        chunk = idx[start : start + batch_size]
        width = max(len(seqs[i]) for i in chunk)  # ty: ignore[possibly-unbound]
        ids = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
        att = torch.zeros_like(ids)
        lig = torch.zeros((len(chunk), width), dtype=torch.bool, device=device)
        for row, i in enumerate(chunk):
            q = seqs[i]
            ids[row, : len(q)] = torch.tensor(q, device=device)
            att[row, : len(q)] = 1
            lig[row, : len(q)] = torch.tensor(
                sequence_ligand_mask(q), device=device
            )
        with torch.no_grad():
            pred = rescorer(
                {"input_ids": ids, "attention_mask": att, "ligand_mask": lig}
            )
        out[chunk] = pred.float().cpu().numpy()
    return out


def _frame_averaged(  # noqa: PLR0913
    mols: list[dict],
    p_codes: list[int],
    frame: Any,  # noqa: ANN401  (source-repo pocket frame)
    enc: PoseEncoder,
    rescorer: Any,  # noqa: ANN401  (source-repo head)
    n_frames: int,
    tid: str,
    device: torch.device,
) -> np.ndarray:
    """Mean predicted RMSD over ``n_frames`` frame rotations of the complex.

    The descriptors are computed once and rotated, so extra frames cost one
    quantization and one head pass each, not a re-featurization.
    """
    descs = enc.ligand_descs(mols, frame)
    rng = rng_for(0, f"frame_average:{tid}")
    acc = np.zeros(len(mols))
    seen = np.zeros(len(mols))
    for k in range(max(n_frames, 1)):
        rot = None if k == 0 else random_rotation_matrix(rng)
        codes = p_codes if rot is None else enc.pocket_codes_rotated(rot)
        pred = _head_scores(
            enc.seqs_from_descs(codes, descs, rotation=rot), rescorer, device
        )
        ok = ~np.isnan(pred)
        acc[ok] += pred[ok]
        seen[ok] += 1
    return np.where(seen > 0, acc / np.maximum(seen, 1), np.nan)


def _score_target(  # noqa: PLR0913
    tid: str,
    casf: Path,
    enc: PoseEncoder,
    mlm: Any,  # noqa: ANN401  (source-repo model)
    mask_id: int,
    rescorer: Any,  # noqa: ANN401  (source-repo head)
    cfg: RescoringConfig,
    device: torch.device,
) -> list[dict]:
    prot = casf / "coreset" / tid / f"{tid}_protein.pdb"
    native_sdf = casf / "coreset" / tid / f"{tid}_ligand.sdf"
    decoys = casf / "decoys_docking" / f"{tid}_decoys.mol2"
    rmsd_dat = casf / "decoys_docking" / f"{tid}_rmsd.dat"
    if not (prot.exists() and native_sdf.exists() and decoys.exists()):
        return []
    try:
        native = parse_sdf(native_sdf)[0]
        native_heavy = np.array(
            [(a[1], a[2], a[3]) for a in native["atoms"] if a[0] != "H"],
            np.float32,
        )
        setup = enc.setup_pocket(prot.read_text(), native_heavy)
        if setup is None:
            return []
        p_codes, frame = setup
        rmsd = _read_rmsd(rmsd_dat)
        poses = parse_mol2_multi(decoys.read_text())
        scored: list[tuple[str, dict, float]] = []
        if not cfg.exclude_native:
            scored.append((f"{tid}_native", native, 0.0))
        scored.extend(
            (name, mol, rmsd[name]) for name, mol in poses if name in rmsd
        )
        # One frame reproduces the original per-pose path; more than one
        # averages the head over frames the score should already be blind to.
        avg = (
            _frame_averaged(
                [m for _, m, _ in scored],
                p_codes,
                frame,
                enc,
                rescorer,
                cfg.n_frames,
                tid,
                device,
            )
            if rescorer is not None and cfg.n_frames > 1
            else None
        )
        rows: list[dict] = []
        for i, (name, mol, ref) in enumerate(scored):
            rows.extend(
                _score_pose(
                    name,
                    mol,
                    ref,
                    tid,
                    p_codes,
                    frame,
                    enc,
                    mlm,
                    mask_id,
                    rescorer,
                    device,
                    head_override=None if avg is None else float(avg[i]),
                    want_pll=cfg.score_mode in ("pll", "ensemble"),
                )
            )
    except Exception:
        logger.exception("target %s failed", tid)
        return []
    return rows if len(rows) >= _MIN_SCORED else []


def _score_pose(  # noqa: PLR0913
    name: str,
    mol: dict,
    rmsd: float,
    tid: str,
    p_codes: list[int],
    frame: Any,  # noqa: ANN401  (source-repo pocket frame)
    enc: PoseEncoder,
    mlm: Any,  # noqa: ANN401  (source-repo model)
    mask_id: int,
    rescorer: Any,  # noqa: ANN401  (source-repo head)
    device: torch.device,
    head_override: float | None = None,
    *,
    want_pll: bool = False,
) -> list[dict]:
    seq = enc.ligand_seq(p_codes, mol, frame)
    if seq is None:
        return []
    head = float("nan")
    if head_override is not None:
        head = -head_override  # lower predicted RMSD = higher score
    elif rescorer is not None:
        ids = torch.tensor([seq], device=device)
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "ligand_mask": torch.tensor(
                sequence_ligand_mask(seq), device=device
            ).unsqueeze(0),
        }
        with torch.no_grad():
            head = -float(rescorer(batch).item())  # lower predicted RMSD = higher score
    # One encoder pass per ligand token, so this costs more than the whole
    # frame-averaged head: 126 s against 40 s over 600 poses on one GPU. It is
    # the zero-shot scorer, not an input to the head's score, so it is only
    # computed when ``score_mode`` actually reads it.
    pll = float(ligand_pll(mlm, seq, mask_id, device)) if want_pll else float("nan")
    return [{"pdbid": tid, "pose": name, "rmsd": rmsd, "head": head, "pll": pll}]
