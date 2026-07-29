"""CASF-2016 pose-rescoring inference -> per-pose dump (pdbid,pose,rmsd,head,pll).

Ported from ``scripts/eval_casf_rescore.py`` (inference half only; metrics live
in :mod:`ctbench.metrics.rescoring`). For each target we extract the pocket once
around the crystal ligand, then score the docking decoys (native excluded by
default — the honest docking-power protocol).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from ctbench.inference import ensure_source_repo_importable
from ctbench.inference.encode import (
    ComplexEncoder,
    ligand_mask,
    load_mlm,
    load_rescorer,
    load_tokenizer,
    make_encoder,
    parse_mol2_multi,
    resolve_rescore_ckpt,
)

ensure_source_repo_importable()

from src.model.mlm_score import ligand_pll  # noqa: E402
from src.tokenizers.ligand import parse_sdf  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    from ctbench.config import PathsConfig, RescoringConfig
    from ctbench.variants import RescoringCkpts

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
    multi-head ensemble downstream. ``pll`` is head-independent and always filled.
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
            head_spec.pooling,
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


def _score_target(  # noqa: PLR0913
    tid: str,
    casf: Path,
    enc: ComplexEncoder,
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
        rows: list[dict] = []
        if not cfg.exclude_native:
            rows.extend(
                _score_pose(
                    f"{tid}_native",
                    native,
                    0.0,
                    tid,
                    p_codes,
                    frame,
                    enc,
                    mlm,
                    mask_id,
                    rescorer,
                    device,
                )
            )
        for name, mol in poses:
            if name in rmsd:
                rows.extend(
                    _score_pose(
                        name,
                        mol,
                        rmsd[name],
                        tid,
                        p_codes,
                        frame,
                        enc,
                        mlm,
                        mask_id,
                        rescorer,
                        device,
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
    enc: ComplexEncoder,
    mlm: Any,  # noqa: ANN401  (source-repo model)
    mask_id: int,
    rescorer: Any,  # noqa: ANN401  (source-repo head)
    device: torch.device,
) -> list[dict]:
    seq = enc.ligand_seq(p_codes, mol, frame)
    if seq is None:
        return []
    head = float("nan")
    if rescorer is not None:
        ids = torch.tensor([seq], device=device)
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "ligand_mask": torch.tensor(ligand_mask(seq), device=device).unsqueeze(0),
        }
        with torch.no_grad():
            head = -float(rescorer(batch).item())  # lower predicted RMSD = higher score
    pll = float(ligand_pll(mlm, seq, mask_id, device))
    return [{"pdbid": tid, "pose": name, "rmsd": rmsd, "head": head, "pll": pll}]
