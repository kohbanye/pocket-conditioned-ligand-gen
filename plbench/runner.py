"""Drive every model over a sample set and collect a long-format results table.

One row per (sample, model, modality) with reconstruction metrics. The own
model, when present, is passed in pre-materialized (it defines the sample set).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from tqdm import tqdm

from plbench import adapters, metrics
from plbench.adapters.base import ReconstructionModel
from plbench.types import ReconResult, Sample


def _subset_to_pocket(mod, pocket: set[tuple[str, int]]):
    """Restrict a protein modality's rows to the pocket residues. Returns
    (ref, rec, n_tokens_pocket) or (None, None, None) if not applicable."""
    if mod.res_keys is None:
        return None, None, None
    keep = [i for i, k in enumerate(mod.res_keys) if k in pocket]
    if not keep:
        return None, None, None
    return mod.ref[keep], mod.rec[keep], len(keep)


def _metric_row(result, mod, ref, rec, eval_scope, is_protein) -> dict:
    m = metrics.all_metrics(ref, rec, protein=is_protein)
    return {
        "sample_id": result.sample_id,
        "model": result.model,
        "modality": mod.modality,
        "ok": True,
        "error": None,
        "atom_kind": mod.atom_kind,
        "eval_scope": eval_scope,
        "n_residues": mod.n_residues,
        "n_tokens": mod.n_tokens,
        "runtime_s": result.runtime_s,
        **m,
    }


def _rows_from_result(
    result: ReconResult,
    pocket_keys: dict[str, set] | None = None,
) -> list[dict]:
    if not result.ok:
        return [
            {
                "sample_id": result.sample_id,
                "model": result.model,
                "modality": None,
                "ok": False,
                "error": result.error,
                "runtime_s": result.runtime_s,
            }
        ]
    pocket = (pocket_keys or {}).get(result.sample_id)
    rows = []
    for mod in result.modalities:
        is_protein = mod.modality == "protein_backbone"
        has_pocket = is_protein and pocket is not None and mod.res_keys is not None
        # Base row: the model's native reconstruction. For ESM3/FoldToken this is
        # the whole protein ("full"); the own model only ever does the pocket.
        rows.append(
            _metric_row(
                result, mod, mod.ref, mod.rec,
                "full" if has_pocket else "native", is_protein,
            )
        )
        # Extra row: the same reconstruction scored only on the pocket residues,
        # so ESM3/FoldToken (full) and the own pocket model share a row.
        if has_pocket:
            sref, srec, _ = _subset_to_pocket(mod, pocket)
            if sref is not None:
                rows.append(
                    _metric_row(result, mod, sref, srec, "pocket", is_protein)
                )
    return rows


def run(
    model_names: Iterable[str],
    samples: list[Sample],
    *,
    prebuilt: dict[str, ReconstructionModel] | None = None,
    adapter_kwargs: dict | None = None,
    pocket_keys: dict[str, set] | None = None,
) -> pd.DataFrame:
    """Run each model over ``samples`` and collect metric rows.

    If ``pocket_keys`` maps sample_id -> set of (chain, resid), protein metrics
    for models that report per-residue identity are restricted to those residues
    (a full-protein reconstruction scored on the pocket residues only).
    """
    prebuilt = prebuilt or {}
    adapter_kwargs = adapter_kwargs or {}
    rows: list[dict] = []

    for name in model_names:
        model = prebuilt.get(name) or adapters.build(name, **adapter_kwargs.get(name, {}))
        try:
            model.setup()
        except Exception as exc:  # noqa: BLE001
            for s in samples:
                rows.append(
                    {"sample_id": s.sample_id, "model": name, "modality": None,
                     "ok": False, "error": f"setup failed: {exc!r}"}
                )
            continue

        applicable = [s for s in samples if model.can_protein or s.ligand_sdf]
        results = model.reconstruct_batch(tqdm(applicable, desc=name, unit="cx"))
        for result in results:
            rows.extend(_rows_from_result(result, pocket_keys))
        model.teardown()

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, modality, eval_scope) stats over successful reconstructions.

    Reports the **median** as the headline (reconstruction RMSD is heavy-tailed —
    a few hard structures inflate the mean), plus the mean for reference.
    """
    ok = df[df["ok"]].copy()
    if ok.empty:
        return ok
    keys = ["model", "modality"]
    if "eval_scope" in ok.columns:
        ok["eval_scope"] = ok["eval_scope"].fillna("native")
        keys.append("eval_scope")
    g = ok.groupby(keys)
    out = {"n": g.size()}
    if "kabsch_rmsd" in ok.columns:
        out["kabsch_rmsd_median"] = g["kabsch_rmsd"].median()
        out["kabsch_rmsd_mean"] = g["kabsch_rmsd"].mean()
    for col in ("tm_score", "lddt"):
        if col in ok.columns:
            out[f"{col}_median"] = g[col].median()
    for col in ("n_tokens", "n_atoms"):
        if col in ok.columns:
            out[col] = g[col].mean()
    return pd.DataFrame(out).round(4).reset_index()
