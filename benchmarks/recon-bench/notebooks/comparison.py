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
    from recon_bench import metrics, paths, runner  # noqa: F401

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
    # Display names used in every legend / axis / table.
    DISPLAY = {"own_allatom.joint": "ProLIT", "own_allatom.separate": "ProLIT (separate)",
               "esm3": "ESM3", "foldtoken": "FoldToken4",
               "token_mol": "Token-Mol", "bio2token": "Bio2Token", "confseq": "ConfSeq"}
    ORDER = ["ProLIT", "ProLIT (separate)", "ESM3", "FoldToken4", "Token-Mol",
             "Bio2Token", "ConfSeq"]
    df = pd.read_parquet(file_dropdown.value)
    df["model_disp"] = df["model"].map(DISPLAY).fillna(df["model"])
    mo.md(f"Loaded **{len(df)}** rows, models: {sorted(df['model'].unique())}")
    return DISPLAY, ORDER, df


@app.cell
def _(DISPLAY, df, mo, runner):
    # Summary table: mean metrics per (model, modality, eval_scope).
    # ESM3/FoldToken appear twice — full (whole protein) and pocket (scored on
    # the pocket residues only) — alongside the own pocket model.
    summary = runner.summarize(df)
    summary.insert(0, "Model", summary["model"].map(DISPLAY).fillna(summary["model"]))
    mo.ui.table(summary, label="Mean metrics per model × modality × eval scope")
    return (summary,)


@app.cell
def _(df, mo):
    import matplotlib.pyplot as plt
    import seaborn as sns

    _prot = df[(df["ok"]) & (df["modality"] == "protein_backbone")].copy()
    mo.stop(_prot.empty, mo.md("_No protein-backbone reconstructions to plot._"))
    # Label = model + eval scope. ESM3/FoldToken4 appear as both "(full)" — the
    # whole-protein RMSD — and "(pocket)" — scored on the pocket residues only —
    # next to "Ours (native)" (own model is pocket-only).
    _scope = _prot["eval_scope"].fillna("native") if "eval_scope" in _prot.columns else "native"
    _prot["label"] = _prot["model_disp"] + " (" + _scope + ")"
    _order = [
        lbl for lbl in
        ["Ours (native)", "ESM3 (full)", "ESM3 (pocket)",
         "FoldToken4 (full)", "FoldToken4 (pocket)"]
        if lbl in set(_prot["label"])
    ]

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for _ax, _metric in zip(_axes, ["kabsch_rmsd", "tm_score", "lddt"], strict=False):
        if _metric not in _prot.columns:
            continue
        sns.boxplot(data=_prot, x="label", y=_metric, order=_order, ax=_ax, showfliers=False)
        sns.stripplot(data=_prot, x="label", y=_metric, order=_order, ax=_ax,
                      color="0.25", size=2.5, alpha=0.35)
        _ax.set_title(f"protein backbone — {_metric}")
        _ax.set_xlabel("")
        _ax.tick_params(axis="x", rotation=25)
    _fig.tight_layout()
    _fig
    return (plt, sns)


@app.cell
def _(df, mo, plt, sns):
    # Ligand & complex reconstruction. "complex" = Ours' pocket CA + ligand
    # aligned jointly (penalises ligand-pose drift relative to the pocket).
    _lc = df[(df["ok"]) & (df["modality"].isin(["ligand", "complex"]))].copy()
    mo.stop(_lc.empty, mo.md("_No ligand/complex reconstructions._"))
    _lc["label"] = _lc["model_disp"] + " (" + _lc["modality"] + ")"
    _lorder = [
        x for x in ["Ours (ligand)", "Ours (complex)", "Token-Mol (ligand)"]
        if x in set(_lc["label"])
    ]
    _fig2, _ax2 = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=_lc, x="label", y="kabsch_rmsd", order=_lorder, ax=_ax2[0], showfliers=False)
    _ax2[0].set_title("ligand / complex heavy-atom RMSD (Å)")
    _ax2[0].set_xlabel("")
    _ax2[0].tick_params(axis="x", rotation=15)
    sns.histplot(data=_lc, x="kabsch_rmsd", hue="label", hue_order=_lorder, bins=30, ax=_ax2[1])
    _ax2[1].set_title("RMSD distribution")
    _fig2.tight_layout()
    _fig2
    return


@app.cell
def _(df, mo):
    # Per-sample head-to-head on the SAME pocket residues: Ours (native) vs
    # ESM3/FoldToken4 (pocket-restricted).
    _scope = df["eval_scope"].fillna("native") if "eval_scope" in df.columns else "native"
    _pp = df[(df["ok"]) & (df["modality"] == "protein_backbone")
             & _scope.isin(["pocket", "native"])]
    _wide = _pp.pivot_table(index="sample_id", columns="model_disp", values="kabsch_rmsd")
    mo.ui.table(_wide.round(3), label="Per-sample pocket-residue kabsch RMSD (Å)")
    return


if __name__ == "__main__":
    app.run()
