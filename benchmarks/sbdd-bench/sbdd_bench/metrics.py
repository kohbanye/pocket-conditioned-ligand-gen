"""Evaluate one model's generated ligands for one target, end to end.

Pulls together every category onto a single (model × target) sample set and
produces two things:

* a **per-molecule** table (one row per generated ligand) with every metric, and
* a **per-target summary** with aggregates and the composite *hit-rate* — the
  metric the benchmark treats as headline:

    a generated molecule is a **hit** iff it is simultaneously
      valid · PoseBusters-valid · binds better than the reference (Vina Dock)
      · synthesizable (SA ≤ 5) · drug-like (QED ≥ 0.4).
    The reported hit-rate is the *scaffold-unique* fraction, so a mode-collapsed
    model cannot inflate it by emitting the same hit scaffold many times.

This "valid & physically-plausible & bindable & synthesizable" rate is far more
faithful to real use than mean Vina alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sbdd_bench import chem, diversity, docking, pose
from sbdd_bench.molio import GenMol, load_generated, read_ref_mol
from sbdd_bench.types import Target

# Composite-hit thresholds (the benchmark's success definition; overridable).
HIT = {"sa_max": 5.0, "qed_min": 0.4}


@dataclass
class EvalConfig:
    dock: bool = True
    pose_quality: bool = True
    interactions: bool = False
    dock_modes: tuple[str, ...] = ("score", "min", "dock")
    dock_workers: int | None = None
    dock_exhaustiveness: int = 8
    dock_limit: int | None = None  # dock only first N valid mols (Vina is slow)
    train_smiles: set[str] | None = None
    hit: dict = field(default_factory=lambda: dict(HIT))


def _tanimoto(mol, ref_fp):
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    if mol is None or ref_fp is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return float(DataStructs.TanimotoSimilarity(fp, ref_fp))
    except Exception:  # noqa: BLE001
        return None


def _ref_genmol(target: Target) -> GenMol | None:
    """The reference ligand as a GenMol, so it can be docked as a positive
    control / threshold with the identical pipeline."""
    ref = read_ref_mol(target.ref_ligand_sdf)
    if ref is None or ref.GetNumConformers() == 0:
        return None
    conf = ref.GetConformer()
    els = [a.GetSymbol() for a in ref.GetAtoms()]
    xyz = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(ref.GetNumAtoms())], dtype=np.float64,
    )
    return GenMol(idx=-1, elements=els, coords=xyz, mol=ref,
                  smiles=None, tag="ref")


def evaluate_target(
    model: str,
    target: Target,
    sdf_path,
    cfg: EvalConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or EvalConfig()
    from rdkit.Chem import AllChem

    gen = load_generated(sdf_path)
    # Some models (Ours) write the reference ligand into generated.sdf as a
    # positive control; drop it so every model is scored on generated molecules
    # only. The reference is docked separately as the threshold.
    gen = [g for g in gen if not str(g.tag).startswith("ref")]
    ref_gm = _ref_genmol(target)
    ref_fp = (
        AllChem.GetMorganFingerprintAsBitVect(ref_gm.mol, 2, nBits=2048)
        if ref_gm and ref_gm.mol else None
    )

    # ---- per-molecule chemistry + reference similarity ----
    rows: list[dict] = []
    for g in gen:
        row = {"model": model, "target_id": target.target_id, "idx": g.idx, "tag": g.tag}
        row.update(chem.molecule_metrics(g.mol))
        row["smiles"] = g.smiles
        row["tanimoto_ref"] = _tanimoto(g.mol, ref_fp)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, _summary(model, target, df, cfg, ref_scores={})

    # ---- docking (Vina Score / Min / Dock), including the reference ----
    ref_scores: dict = {}
    if cfg.dock and target.receptor_pdbqt:
        to_dock = gen
        if cfg.dock_limit is not None:
            valid_idx = [g.idx for g in gen if g.mol is not None][: cfg.dock_limit]
            keep = set(valid_idx)
            to_dock = [g for g in gen if g.idx in keep]
        dock_rows = docking.dock_generated(
            to_dock, target.receptor_pdbqt, target.box,
            modes=cfg.dock_modes, workers=cfg.dock_workers,
            exhaustiveness=cfg.dock_exhaustiveness,
        )
        dock_df = pd.DataFrame(dock_rows)
        if not dock_df.empty:
            cols = [c for c in ("vina_score", "vina_min", "vina_dock", "min_rmsd") if c in dock_df]
            df = df.merge(dock_df[["idx", *cols]], on="idx", how="left")
        if ref_gm is not None:
            rr = docking.dock_generated(
                [ref_gm], target.receptor_pdbqt, target.box,
                modes=cfg.dock_modes, workers=1,
                exhaustiveness=cfg.dock_exhaustiveness,
            )
            if rr:
                ref_scores = {f"ref_{k}": rr[0].get(k) for k in ("vina_score", "vina_min", "vina_dock")}

    # ---- pose quality (PoseBusters, clashes, strain) ----
    if cfg.pose_quality:
        pb = pose.pb_validity(gen)
        df["pb_valid"] = df["idx"].map(pb)
        prot_el, prot_xyz = pose.read_protein_heavy(target.receptor_pdb)
        clash, strain = {}, {}
        for g in gen:
            if g.elements:
                clash[g.idx] = pose.clash_count(g.elements, g.coords, prot_el, prot_xyz)
            if g.mol is not None:
                strain[g.idx] = pose.strain_energy(g.mol)
        df["clash_count"] = df["idx"].map(clash)
        df["strain_energy"] = df["idx"].map(strain)

    # ---- interactions (optional) ----
    if cfg.interactions:
        from sbdd_bench import interactions

        inter = interactions.interaction_recovery(target.receptor_pdb, ref_gm.mol if ref_gm else None, gen)
        if inter:
            df["key_residue_recovery"] = df["idx"].map(lambda i: (inter.get(i) or {}).get("key_residue_recovery"))
            df["ifp_tanimoto"] = df["idx"].map(lambda i: (inter.get(i) or {}).get("ifp_tanimoto"))

    # ---- composite hit flag ----
    df["hit"] = _hit_flags(df, ref_scores, cfg.hit)
    return df, _summary(model, target, df, cfg, ref_scores)


def _hit_flags(df: pd.DataFrame, ref_scores: dict, hit: dict) -> pd.Series:
    ref_dock = ref_scores.get("ref_vina_dock")
    valid = df.get("valid", pd.Series(False, index=df.index)).fillna(False)
    pb = df["pb_valid"].fillna(False) if "pb_valid" in df else pd.Series(True, index=df.index)
    sa_ok = df["sa"].le(hit["sa_max"]) if "sa" in df else pd.Series(True, index=df.index)
    qed_ok = df["qed"].ge(hit["qed_min"]) if "qed" in df else pd.Series(True, index=df.index)
    if "vina_dock" in df and ref_dock is not None:
        bind_ok = df["vina_dock"].le(ref_dock)
    elif "vina_dock" in df:
        bind_ok = df["vina_dock"].le(0)  # at least a favourable (negative) score
    else:
        bind_ok = pd.Series(False, index=df.index)
    return (valid & pb & sa_ok.fillna(False) & qed_ok.fillna(False) & bind_ok.fillna(False))


def _frac(series) -> float | None:
    s = series.dropna()
    return float(s.mean()) if len(s) else None


def _summary(model, target, df: pd.DataFrame, cfg: EvalConfig, ref_scores: dict) -> dict:
    s = {"model": model, "target_id": target.target_id, "n_generated": len(df)}
    if df.empty:
        return s
    valid = df[df.get("valid", False) == True]  # noqa: E712
    smis = valid["smiles"].dropna().tolist() if "smiles" in valid else []
    s["validity"] = _frac(df["valid"]) if "valid" in df else None
    s["connected"] = _frac(df["connected"]) if "connected" in df else None
    for col in ("qed", "sa", "logp"):
        if col in df:
            s[f"{col}_mean"] = float(df[col].dropna().mean()) if df[col].notna().any() else None
    for col in ("lipinski", "veber", "pains_free"):
        if col in df:
            s[f"{col}_frac"] = _frac(df[col])
    for col in ("vina_score", "vina_min", "vina_dock"):
        if col in df and df[col].notna().any():
            s[f"{col}_median"] = float(df[col].median())
            s[f"{col}_mean"] = float(df[col].mean())
    if "pb_valid" in df:
        s["pb_valid_rate"] = _frac(df["pb_valid"])
    if "clash_count" in df and df["clash_count"].notna().any():
        s["clash_mean"] = float(df["clash_count"].mean())
        s["clash_free_rate"] = float((df["clash_count"] == 0).mean())
    if "strain_energy" in df and df["strain_energy"].notna().any():
        s["strain_median"] = float(df["strain_energy"].median())
    if "tanimoto_ref" in df and df["tanimoto_ref"].notna().any():
        s["tanimoto_ref_mean"] = float(df["tanimoto_ref"].mean())
    if "key_residue_recovery" in df and df["key_residue_recovery"].notna().any():
        s["key_residue_recovery_mean"] = float(df["key_residue_recovery"].mean())
    # diversity (over valid molecules)
    s.update({f"div_{k}": v for k, v in diversity.diversity_metrics(smis, cfg.train_smiles).items()})
    s.update(ref_scores)
    # composite hit-rate
    if "hit" in df:
        s["hit_rate"] = float(df["hit"].mean())
        hit_scaffs = {
            diversity.bemis_murcko_scaffold(x)
            for x in valid.loc[df["hit"].fillna(False), "smiles"].dropna()
        }
        hit_scaffs.discard(None)
        s["hit_scaffold_unique_rate"] = len(hit_scaffs) / len(df) if len(df) else None
    return s
