"""Assemble comparison and ablation tables from the ``results/`` dump tree.

Directory layout (``results/<task>/<method_or_variant>/...``)::

    rescoring/<variant>/*.csv       per-pose head dumps (pdbid,pose,rmsd,head,pll)
    rescoring/{rtmscore,genscore}/pose_scores.csv   (pdbid,pose,native_score)
    rescoring/vina/pose_scores.csv                  (pdbid,pose,rmsd,head)
    generation/<variant>/per_model.csv, per_target.csv, per_molecule.parquet
    generation/baselines/per_model.csv, per_target.csv

Two report kinds per task:
- **comparison** — ours (``joint``) vs existing methods (reproduces the paper);
- **ablation** — ``joint_nocasf`` (fair joint-side control) vs ``separate``
  (separately-trained protein+ligand tokenizers), learned heads only (the
  tokenizer-dependent part), significance vs ``joint_nocasf``.

Variants whose dumps are absent are silently skipped, so the ablation report
grows automatically as the source repo trains the separate-arm downstream models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from pose_rescoring_bench import aggregate
from pose_rescoring_bench.metrics import rescoring as R

if TYPE_CHECKING:
    from pathlib import Path
from pose_rescoring_bench.variants import ABLATION_ORDER

Report = tuple[pd.DataFrame, pd.DataFrame]  # (metrics, significance)


# --------------------------------------------------------------- rescoring ---
def _rescoring_variant_heads(variant_dir: Path) -> list[pd.DataFrame]:
    return [
        R.orient(pd.read_csv(m), raw_col="head")
        for m in sorted(variant_dir.glob("*.csv"))
    ]


def _rescoring_baseline(
    path: Path,
    rmsd_key: pd.DataFrame,
    raw_col: str,
) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "rmsd" not in d.columns:
        d = d.merge(rmsd_key, on=["pdbid", "pose"], how="inner")
    return R.orient(d, raw_col=raw_col)


def rescoring_comparison(results_dir: Path) -> Report:
    """Reproduce the paper pose table: baselines + Vina + ours (heads, Vina-fused)."""
    root = results_dir / "rescoring"
    heads = _rescoring_variant_heads(root / "joint")
    rmsd_key = heads[0][["pdbid", "pose", "rmsd"]]
    scored: dict[str, pd.DataFrame] = {}
    vina_path = root / "vina" / "pose_scores.csv"
    vina = (
        _rescoring_baseline(vina_path, rmsd_key, "head") if vina_path.exists() else None
    )
    for name, sub in (("RTMScore", "rtmscore"), ("GenScore", "genscore")):
        p = root / sub / "pose_scores.csv"
        if p.exists():
            scored[name] = _rescoring_baseline(p, rmsd_key, "native_score")
    if vina is not None:
        scored["Vina"] = vina
    scored["OURS (3-head)"] = R.zsum(heads)
    if vina is not None:
        scored["OURS (3-head + Vina)"] = R.zsum([*heads, vina])
    metrics = aggregate.rescoring_metrics(scored)
    ref = "GenScore" if "GenScore" in scored else "OURS (3-head)"
    sig = aggregate.rescoring_pairwise(scored, reference=ref)
    return metrics, sig


def rescoring_ablation(results_dir: Path) -> Report:
    """Ablation on learned heads only (Vina excluded — it is tokenizer-independent)."""
    root = results_dir / "rescoring"
    variants: dict[str, pd.DataFrame] = {}
    for name in ABLATION_ORDER:
        d = root / name
        if d.exists() and any(d.glob("*.csv")):
            variants[name] = R.zsum(_rescoring_variant_heads(d))
    metrics = aggregate.rescoring_metrics(variants)
    sig = (
        aggregate.rescoring_pairwise(variants, reference="joint_nocasf")
        if "joint_nocasf" in variants and len(variants) > 1
        else _empty_sig()
    )
    return metrics, sig


# -------------------------------------------------------------- generation ---
def _gen_per_model_row(variant_dir: Path, model: str) -> pd.Series | None:
    p = variant_dir / "per_model.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p).set_index("model")
    return df.loc[model] if model in df.index else df.iloc[0]


def generation_comparison(results_dir: Path, our_model: str = "own_t085_on") -> Report:
    root = results_dir / "generation"
    per_model: dict[str, pd.Series] = {}
    base_p = root / "baselines" / "per_model.csv"
    if base_p.exists():
        base = pd.read_csv(base_p).set_index("model")
        for m in ("diffgui", "targetdiff", "diffsbdd"):
            if m in base.index:
                per_model[m] = base.loc[m]
    ours = _gen_per_model_row(root / "joint", our_model)
    if ours is not None:
        per_model["OURS (joint)"] = ours
    metrics = aggregate.generation_table(per_model)

    ours_t = _gen_per_target(root / "joint", our_model)
    base_t = root / "baselines" / "per_target.csv"
    sig = _empty_sig()
    if ours_t is not None and base_t.exists():
        bt = pd.read_csv(base_t)
        per_target = {"OURS (joint)": ours_t}
        for m in ("diffgui", "targetdiff", "diffsbdd"):
            sub = bt[bt.model == m]
            if not sub.empty:
                per_target[m] = sub
        ref = "diffgui" if "diffgui" in per_target else "OURS (joint)"
        sig = aggregate.generation_pairwise(per_target, reference=ref)
    return metrics, sig


def _gen_per_target(variant_dir: Path, model: str) -> pd.DataFrame | None:
    p = variant_dir / "per_target.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[df.model == model]
    return sub if not sub.empty else df


def generation_ablation(results_dir: Path, our_model: str = "own_t085_on") -> Report:
    root = results_dir / "generation"
    per_model: dict[str, pd.Series] = {}
    per_target: dict[str, pd.DataFrame] = {}
    for name in ABLATION_ORDER:
        row = _gen_per_model_row(root / name, our_model)
        tgt = _gen_per_target(root / name, our_model)
        if row is not None:
            per_model[name] = row
        if tgt is not None:
            per_target[name] = tgt
    metrics = aggregate.generation_table(per_model) if per_model else pd.DataFrame()
    sig = (
        aggregate.generation_pairwise(per_target, reference="joint_nocasf")
        if "joint_nocasf" in per_target and len(per_target) > 1
        else _empty_sig()
    )
    return metrics, sig


def _empty_sig() -> pd.DataFrame:
    return pd.DataFrame()
