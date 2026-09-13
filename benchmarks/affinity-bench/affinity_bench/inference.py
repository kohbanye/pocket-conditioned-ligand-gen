"""Score the CASF-2016 core set with one affinity arm -> a per-complex dump.

Each of the 285 core complexes is encoded once from its crystal pose and scored
by the head (whose raw output is pK) and by the MLM's pseudo-log-likelihood,
which is recorded as a zero-shot reference rather than used by the tables.

Unlike pose rescoring there are no decoys here: the pose is the deposited one,
and what is being asked is how tightly the ligand binds, not whether it is
placed correctly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from prolit.api import (
    AtomLMVocab,
    PoseEncoder,
    ligand_mask,
    load_masked_lm,
    load_norm_stats,
    load_scoring_head,
    load_separate_tokenizer,
    load_tokenizer,
    parse_sdf,
    random_rotation_matrix,
)
from prolit.config import PocketExtractionConfig
from prolit.model.mlm_score import ligand_pll
from prolit.seeding import rng_for
from prolit_bench.runs import resolve_scoring_head

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from affinity_bench.config import AffinityConfig, PathsConfig
    from affinity_bench.variants import Variant

logger = logging.getLogger(__name__)
_MIN_CORESET_COLS = 6
#: Descriptor dimension of the all-atom schema, for the separate arm's identity
#: normalization (it normalizes per modality internally, so the encoder feeds it
#: raw descriptors).
_DESC_DIM = 33


def load_coreset_labels(path: Path) -> dict[str, tuple[float, str]]:
    """``pdbid -> (measured pK, cluster id)`` from ``power_scoring/CoreSet.dat``."""
    out: dict[str, tuple[float, str]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split()
        if len(cols) >= _MIN_CORESET_COLS:
            out[cols[0].lower()] = (float(cols[3]), cols[5])
    return out


@dataclass(frozen=True)
class Complex:
    """One scorable complex: where its files are, and what it is worth."""

    pdbid: str
    protein: Path
    ligand: Path
    logka: float
    #: CASF groups five ligands per target and ranking power is computed inside
    #: those groups. A set without such groups uses the pdbid, making every
    #: complex its own singleton -- scoring power still works, ranking power is
    #: simply not defined there, and reports it as nan rather than inventing it.
    cluster: str


def casf_complexes(casf_dir: Path) -> list[Complex]:
    """The CASF-2016 core set, from ``coreset/`` plus ``power_scoring``."""
    labels = load_coreset_labels(casf_dir / "power_scoring" / "CoreSet.dat")
    out = []
    for d in sorted(p for p in (casf_dir / "coreset").iterdir() if p.is_dir()):
        tid = d.name
        if tid not in labels:
            continue
        logka, cluster = labels[tid]
        out.append(
            Complex(tid, d / f"{tid}_protein.pdb", d / f"{tid}_ligand.sdf",
                    logka, cluster)
        )
    return out


def pdbbind_complexes(
    root: Path,
    ids_file: Path,
    labels_csv: tuple[Path, ...] = (),
) -> list[Complex]:
    """PDBbind-layout complexes named by ``ids_file``, labelled from its CSVs.

    Used for the held-out subset of PDBbind: of its 2,355 Kd/Ki complexes with
    structures, only the ~281 that are also in ``eval_holdout_pdbs.txt`` were
    excluded from our training corpora. The rest are training data and scoring
    them would measure memorisation, so the caller passes the id list rather
    than the bench walking the directory.
    """
    import csv  # noqa: PLC0415

    wanted = {x.strip().lower() for x in ids_file.read_text().split() if x.strip()}
    labels: dict[str, float] = {}
    for path in labels_csv:
        if not path.exists():
            continue
        with path.open() as fh:
            for row in csv.DictReader(fh):
                pid = (row.get("PDB_ID") or "").strip().lower()
                if pid in wanted and row.get("Label_pKd_pKi"):
                    labels[pid] = float(row["Label_pKd_pKi"])
    out = []
    for pid in sorted(wanted & set(labels)):
        for sub in ("refined-set", "v2020-other-PL"):
            d = root / sub / pid
            if d.is_dir():
                out.append(
                    Complex(pid, d / f"{pid}_protein.pdb", d / f"{pid}_ligand.sdf",
                            labels[pid], pid)
                )
                break
    return out


def _build_encoder(
    variant: Variant,
    paths: PathsConfig,
    cfg: AffinityConfig,
    device: torch.device,
) -> PoseEncoder:
    """Load the arm's tokenizer and wrap it in the standard pocket encoder."""
    if variant.is_separate:
        missing = [
            n
            for n, v in (
                ("protein_vqvae", variant.protein_vqvae),
                ("ligand_vqvae", variant.ligand_vqvae),
                ("protein_norm", variant.protein_norm),
                ("ligand_norm", variant.ligand_norm),
            )
            if v is None
        ]
        if missing:
            msg = f"separate arm {variant.name!r} is missing {missing}"
            raise ValueError(msg)
        vq = load_separate_tokenizer(
            paths.ckpt(variant.protein_vqvae),  # ty: ignore[invalid-argument-type]
            paths.ckpt(variant.protein_norm),  # ty: ignore[invalid-argument-type]
            paths.ckpt(variant.ligand_vqvae),  # ty: ignore[invalid-argument-type]
            paths.ckpt(variant.ligand_norm),  # ty: ignore[invalid-argument-type]
            device,
            # The per-modality sub-codebook is half the combined code space.
            variant.codebook_size // 2,
        )
        mean = np.zeros(_DESC_DIM, dtype=np.float32)
        std = np.ones(_DESC_DIM, dtype=np.float32)
    else:
        if variant.vqvae is None:
            msg = f"arm {variant.name!r} has no tokenizer checkpoint"
            raise ValueError(msg)
        norm = load_norm_stats(paths.norm_stats, device)
        vq = load_tokenizer(
            paths.ckpt(variant.vqvae), variant.codebook_size, device, norm
        )
        mean = norm["atom_mean"].cpu().numpy()
        std = norm["atom_std"].cpu().numpy()
    return PoseEncoder(
        vq,
        mean,
        std,
        AtomLMVocab(codebook_size=variant.codebook_size),
        device,
        PocketExtractionConfig(max_residues=cfg.max_residues),
    )


def score_complexes(  # noqa: PLR0913
    variant: Variant,
    paths: PathsConfig,
    cfg: AffinityConfig,
    head_index: int = 0,
    seed: int = 0,
    complexes: list[Complex] | None = None,
) -> pd.DataFrame:
    """Score a set of complexes with one head of an arm; CASF-2016 by default.

    Named for what it does rather than for the one set it began with: it now
    also scores the held-out PDBbind slice, and a function called ``score_casf``
    that scores PDBbind is how a name stops describing its behaviour.

    ``head_index`` selects which head, so an arm reported as an ensemble is
    produced by running this once per head and z-summing the dumps downstream.
    ``complexes`` overrides the evaluation set, so the same arm can be measured
    on a held-out slice of PDBbind without a second scoring path to keep in
    step with this one.
    """
    import pandas as pd  # noqa: PLC0415

    torch.set_float32_matmul_precision("high")
    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = _build_encoder(variant, paths, cfg, device)
    mlm, mask_id = load_masked_lm(
        paths.ckpt(variant.mlm), variant.codebook_size, device
    )
    head_spec = variant.require_heads()[head_index]
    head = load_scoring_head(
        resolve_scoring_head(paths.source_repo, head_spec.ckpt),
        variant.codebook_size,
        device,
    )

    items = complexes if complexes is not None else casf_complexes(paths.casf_dir)
    if cfg.max_targets is not None:
        items = items[: cfg.max_targets]

    rows: list[dict] = []
    for item in items:
        # One stream per complex, derived from the run seed, so a complex's
        # frames do not depend on how many complexes came before it.
        rng = rng_for(seed, f"casf_frames:{item.pdbid}")
        row = _score_complex(item, enc, mlm, mask_id, head, device, cfg, rng)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows, columns=["pdbid", "logka", "cluster", "pll", "head"])  # ty: ignore[invalid-argument-type]  # pandas stub: list[str] columns


def _score_complex(  # noqa: PLR0913
    item: Complex,
    enc: PoseEncoder,
    mlm: Any,  # noqa: ANN401
    mask_id: int,
    head: Any,  # noqa: ANN401
    device: torch.device,
    cfg: AffinityConfig,
    rng: np.random.Generator,
) -> dict | None:
    tid, protein, sdf = item.pdbid, item.protein, item.ligand
    if not (protein.exists() and sdf.exists()):
        return None
    try:
        native = parse_sdf(sdf)[0]
        heavy = np.array(
            [(a[1], a[2], a[3]) for a in native["atoms"] if a[0] != "H"],
            np.float32,
        )
        setup = enc.setup_pocket(protein.read_text(), heavy)
        if setup is None:
            return None
        p_codes, frame = setup
        # Computed once; each extra frame rotates the stored descriptors and
        # re-quantizes, rather than recomputing RDKit features per orientation.
        descs = enc.ligand_descs([native], frame)
        preds: list[float] = []
        for k in range(max(cfg.n_frames, 1)):
            rot = None if k == 0 else random_rotation_matrix(rng)
            codes = p_codes if rot is None else enc.pocket_codes_rotated(rot)
            seq = enc.seqs_from_descs(codes, descs, rotation=rot)[0]
            if seq is None:
                continue
            preds.append(_predict(seq, head, device))
            if k == 0:
                pll = float(ligand_pll(mlm, seq, mask_id, device))
        if not preds:
            return None
    except Exception:
        logger.exception("target %s failed", tid)
        return None
    return {
        "pdbid": tid,
        "logka": item.logka,
        "cluster": item.cluster,
        "pll": pll,
        # The affinity head's raw output IS the pK, so it is averaged directly.
        # (The pose head's output is an RMSD and is negated to become a score;
        # that sign flip does not belong here.)
        "head": float(np.mean(preds)),
    }


def _predict(seq: list[int], head: Any, device: torch.device) -> float:  # noqa: ANN401
    ids = torch.tensor([seq], device=device)
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "ligand_mask": torch.tensor(
            ligand_mask(np.asarray(seq)), device=device
        ).unsqueeze(0),
    }
    with torch.no_grad():
        return float(head(batch).item())
