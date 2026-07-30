"""CASF-2016 affinity inference -> per-complex dump (pdbid,logka,cluster,pll,head).

Ported from ``scripts/eval_casf_scoring.py`` (inference half only). Each crystal
complex is encoded once and scored by the affinity head (raw output = pK) and by
the MLM pseudo-log-likelihood; metrics live in
:mod:`pose_rescoring_bench.metrics.affinity`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from prolit.model.mlm_score import ligand_pll
from prolit.tokenizers.ligand import parse_sdf

from pose_rescoring_bench.inference.encode import (
    load_mlm,
    load_rescorer,
    load_tokenizer,
    make_encoder,
    resolve_rescore_ckpt,
    sequence_ligand_mask,
)

if TYPE_CHECKING:
    from pathlib import Path

    from prolit.api import PoseEncoder

    from pose_rescoring_bench.config import AffinityConfig, PathsConfig
    from pose_rescoring_bench.variants import AffinityCkpts

logger = logging.getLogger(__name__)
_MIN_CORESET_COLS = 6


def load_coreset_labels(path: Path) -> dict[str, tuple[float, str]]:
    """pdbid -> (logKa, cluster) from ``power_scoring/CoreSet.dat``."""
    out: dict[str, tuple[float, str]] = {}
    for ln in path.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        c = ln.split()
        if len(c) >= _MIN_CORESET_COLS:
            out[c[0].lower()] = (float(c[3]), c[5])
    return out


def score_casf(
    ckpts: AffinityCkpts,
    paths: PathsConfig,
    cfg: AffinityConfig,
    head_index: int = 0,
) -> pd.DataFrame:
    """Score CASF crystal complexes with one head of a variant -> per-complex DataFrame.

    A variant's affinity ensemble is produced by running this once per head
    (``head_index``) and z-summing the resulting dumps downstream. The codebook
    size is taken from ``ckpts.codebook_size`` (8192 joint, 16384 separate).
    """
    torch.set_float32_matmul_precision("high")
    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]  # rdkit C-ext
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    codebook_size = ckpts.codebook_size
    module, mean, std = load_tokenizer(ckpts, paths, device)
    mlm_ckpt = ckpts.mlm
    if mlm_ckpt is None:
        msg = "affinity variant is missing its mlm checkpoint"
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
    head_spec = ckpts.heads[head_index]
    rescorer = load_rescorer(
        resolve_rescore_ckpt(paths.source_repo, head_spec.ckpt),
        head_spec.pooling,
        codebook_size,
        device,
    )

    casf = paths.casf_dir
    labels = load_coreset_labels(casf / "power_scoring" / "CoreSet.dat")
    rows: list[dict] = []
    for tid in sorted(p.name for p in (casf / "coreset").iterdir() if p.is_dir()):
        if tid not in labels:
            continue
        row = _score_complex(tid, casf, labels, enc, mlm, mask_id, rescorer, device)
        if row is not None:
            rows.append(row)
    cols = ["pdbid", "logka", "cluster", "pll", "head"]
    return pd.DataFrame(rows, columns=cols)  # ty: ignore[invalid-argument-type]


def _score_complex(  # noqa: PLR0913
    tid: str,
    casf: Path,
    labels: dict[str, tuple[float, str]],
    enc: PoseEncoder,
    mlm: Any,  # noqa: ANN401  (source-repo model)
    mask_id: int,
    rescorer: Any,  # noqa: ANN401  (source-repo head)
    device: torch.device,
) -> dict | None:
    prot = casf / "coreset" / tid / f"{tid}_protein.pdb"
    sdf = casf / "coreset" / tid / f"{tid}_ligand.sdf"
    if not (prot.exists() and sdf.exists()):
        return None
    try:
        native = parse_sdf(sdf)[0]
        heavy = np.array(
            [(a[1], a[2], a[3]) for a in native["atoms"] if a[0] != "H"],
            np.float32,
        )
        setup = enc.setup_pocket(prot.read_text(), heavy)
        if setup is None:
            return None
        p_codes, frame = setup
        seq = enc.ligand_seq(p_codes, native, frame)
        if seq is None:
            return None
        pll = float(ligand_pll(mlm, seq, mask_id, device))
        ids = torch.tensor([seq], device=device)
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "ligand_mask": torch.tensor(
                sequence_ligand_mask(seq), device=device
            ).unsqueeze(0),
        }
        with torch.no_grad():
            head = float(rescorer(batch).item())  # affinity head: raw output = pK
    except Exception:
        logger.exception("target %s failed", tid)
        return None
    logka, cluster = labels[tid]
    return {"pdbid": tid, "logka": logka, "cluster": cluster, "pll": pll, "head": head}
