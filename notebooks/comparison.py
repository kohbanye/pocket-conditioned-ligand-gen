"""Marimo notebook: compare reconstruction quality across models.

Run with:  uv run marimo edit notebooks/comparison.py
Reads the parquet written by scripts/run_reconstruction.py (results/).
"""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from plbench import metrics, paths, runner  # noqa: F401

    return REPO_ROOT, mo, np, pd, paths, runner


@app.cell
def _(mo, paths):
    # Pick a results file.
    result_files = sorted(paths.RESULTS_DIR.glob("*.parquet"))
    file_dropdown = mo.ui.dropdown(
        options={p.name: str(p) for p in result_files},
        value=result_files[-1].name if result_files else None,
        label="results file",
    )
    file_dropdown
    return (file_dropdown,)


@app.cell
def _(file_dropdown, mo, pd):
    mo.stop(not file_dropdown.value, mo.md("**No results yet.** Run `scripts/run_reconstruction.py` first."))
    df = pd.read_parquet(file_dropdown.value)
    mo.md(f"Loaded **{len(df)}** rows, models: {sorted(df['model'].unique())}")
    return (df,)


@app.cell
def _(df, mo, runner):
    # Summary table: mean metrics per (model, modality).
    summary = runner.summarize(df)
    mo.ui.table(summary, label="Mean metrics per model × modality")
    return (summary,)


@app.cell
def _(df, mo):
    import matplotlib.pyplot as plt
    import seaborn as sns

    _prot = df[(df["ok"]) & (df["modality"] == "protein_backbone")].copy()
    mo.stop(_prot.empty, mo.md("_No protein-backbone reconstructions to plot._"))
    # Label with eval scope so it's clear ESM3/FoldToken are full-protein
    # reconstructions scored on the pocket residues, vs the own pocket model.
    if "eval_scope" in _prot.columns:
        _prot["label"] = _prot["model"] + " (" + _prot["eval_scope"].fillna("native") + ")"
    else:
        _prot["label"] = _prot["model"]

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))
    for _ax, _metric in zip(_axes, ["kabsch_rmsd", "tm_score", "lddt"], strict=False):
        if _metric not in _prot.columns:
            continue
        sns.boxplot(data=_prot, x="label", y=_metric, ax=_ax)
        sns.stripplot(data=_prot, x="label", y=_metric, ax=_ax, color="0.3", size=3, alpha=0.4)
        _ax.set_title(f"protein backbone — {_metric}")
        _ax.tick_params(axis="x", rotation=20)
    _fig.tight_layout()
    _fig
    return (plt, sns)


@app.cell
def _(df, mo, sns):
    # Ligand reconstruction (own model only).
    _lig = df[(df["ok"]) & (df["modality"] == "ligand")]
    mo.stop(_lig.empty, mo.md("_No ligand reconstructions (own model only)._"))
    _lax = sns.histplot(data=_lig, x="kabsch_rmsd", hue="model", bins=30)
    _lax.set_title("ligand heavy-atom reconstruction RMSD (Å)")
    _lax.figure
    return


@app.cell
def _(df, mo):
    # Per-sample head-to-head: paired pocket RMSD across models.
    _pp = df[(df["ok"]) & (df["modality"] == "protein_backbone")]
    _wide = _pp.pivot_table(index="sample_id", columns="model", values="kabsch_rmsd")
    mo.ui.table(_wide.round(3), label="Per-sample protein backbone kabsch RMSD (Å)")
    return


if __name__ == "__main__":
    app.run()
