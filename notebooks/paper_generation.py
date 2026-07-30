# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "pyarrow",
# ]
# ///
"""Paper figure/table: all-atom generation vs SBDD baselines (Vina & quality).

Best all-atom generation config (see docs/results/best_allatom_configs.md):
  placement LM p6lpk7br + refiner refine_atom_bond_v1 + sampling temperature 0.85.
Targets 2ity/1iep/3pbl, 150 samples each. Baselines from sbdd-bench official run.
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():

    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    return Path, mo, np, os, pd, plt, stats


@app.cell
def _(Path, os):
    # Project-internal analysis notebook: absolute repo root is reliable here.
    # The repository root, found from this file so the notebook runs from any
    # checkout. PROLIT_ROOT overrides it.
    REPO = Path(
        os.environ.get("PROLIT_ROOT")
        or Path(__file__).resolve().parent.parent
    )
    GEN = REPO / "outputs" / "generation"
    return (GEN,)


@app.cell
def _(mo):
    mo.md("""
    # Generation — all-atom best vs. SBDD baselines

    Metric of record is the **raw Vina score** of the generated pose
    (`--dock-modes score`), plus Vina local-minimized (`min`) and standard
    molecular-quality metrics. Lower Vina = stronger predicted binding.
    """)
    return


@app.cell
def _(GEN, pd):
    ours_model = pd.read_csv(GEN / "tsweep_per_model.csv")
    base_model = pd.read_csv(GEN / "baseline_per_model.csv")
    ours_tgt = pd.read_csv(GEN / "tsweep_per_target.csv")
    base_tgt = pd.read_csv(GEN / "baseline_per_target.csv")
    return base_model, base_tgt, ours_model, ours_tgt


@app.cell
def _(mo):
    mo.md("""
    ## Sampling-temperature sweep (our model)
    """)
    return


@app.cell
def _(mo, ours_model):
    temp_cols = [
        "model", "vina_score_mean", "vina_min_mean", "pb_valid_rate",
        "clash_free_rate", "div_uniqueness", "div_scaffold_diversity",
    ]
    temp_view = (
        ours_model[temp_cols]
        .assign(temperature=lambda d: d["model"].map(
            {"own_t07_on": 0.70, "own_t085_on": 0.85, "own_t10_on": 1.00}))
        .sort_values("temperature")
        .round(3)
    )
    mo.md(
        "T=0.85 is the knee: best Vina without diversity collapse "
        "(uniqueness stays 0.998, same as T=1.0)."
    )
    return (temp_view,)


@app.cell
def _(temp_view):
    temp_view
    return


@app.cell
def _(mo):
    mo.md("""
    ## Head-to-head comparison (our best = T=0.85)
    """)
    return


@app.cell
def _(base_model, ours_model, pd):
    cols = [
        "vina_score_mean", "vina_min_mean", "pb_valid_rate", "clash_free_rate",
        "qed_mean", "sa_mean", "div_scaffold_diversity",
    ]
    nice = {
        "vina_score_mean": "Vina score ↓", "vina_min_mean": "Vina min ↓",
        "pb_valid_rate": "PB-valid ↑", "clash_free_rate": "Clash-free ↑",
        "qed_mean": "QED ↑", "sa_mean": "SA ↓", "div_scaffold_diversity": "Scaffold div ↑",
    }
    rows = []
    for _, r in base_model[base_model.model != "own"].iterrows():
        rows.append({"method": r["model"], **{nice[c]: r[c] for c in cols}})
    ob = ours_model[ours_model.model == "own_t085_on"].iloc[0]
    rows.append({"method": "OURS (all-atom, T=0.85)", **{nice[c]: ob[c] for c in cols}})
    compare = pd.DataFrame(rows).set_index("method").round(3)
    # order rows: baselines then ours
    compare = compare.reindex(["diffgui", "targetdiff", "diffsbdd", "OURS (all-atom, T=0.85)"])
    return (compare,)


@app.cell
def _(compare):
    compare
    return


@app.cell
def _(compare, plt):
    fig_bar, ax_bar = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, col, ttl in [
        (ax_bar[0], "Vina score ↓", "Raw Vina score (lower = better)"),
        (ax_bar[1], "Vina min ↓", "Vina local-min (lower = better)"),
    ]:
        s = compare[col]
        colors = ["#8a8a8a" if m != "OURS (all-atom, T=0.85)" else "#c0392b" for m in s.index]
        ax.barh([m.replace(" (all-atom, T=0.85)", "\n(ours)") for m in s.index], s.values, color=colors)
        ax.set_title(ttl, fontsize=10)
        ax.axvline(0, color="k", lw=0.6)
        for i, v in enumerate(s.values):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    fig_bar.tight_layout()
    fig_bar
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-target breakdown
    The mean is dominated by target identity: 1iep is easy (deep negative),
    2ity was the historical bottleneck (mis-placement) and is where the
    placement re-finetune + low temperature helped most.
    """)
    return


@app.cell
def _(ours_tgt):
    pt = ours_tgt[ours_tgt.model == "own_t085_on"][
        ["target_id", "vina_score_mean", "vina_min_mean", "clash_free_rate", "pb_valid_rate"]
    ].sort_values("target_id").round(3).set_index("target_id")
    pt
    return


@app.cell
def _(base_tgt, ours_tgt, stats):
    # Paired significance across the 3 targets vs DiffGui (the strongest baseline).
    ours_pt = (ours_tgt[ours_tgt.model == "own_t085_on"]
               .set_index("target_id")["vina_score_mean"])
    dg_pt = (base_tgt[base_tgt.model == "diffgui"]
             .set_index("target_id")["vina_score_mean"])
    common = sorted(set(ours_pt.index) & set(dg_pt.index))
    o = ours_pt.loc[common].values
    g = dg_pt.loc[common].values
    tstat, pval = stats.ttest_rel(o, g)
    stat_md = (
        f"**Ours vs DiffGui (paired over {len(common)} targets):** "
        f"mean diff = {(o - g).mean():+.2f} kcal/mol, t = {tstat:.2f}, "
        f"**p = {pval:.3f}** → not significant (n=3 has almost no power). "
        f"We beat DiffSBDD/TargetDiff on the point estimate; the DiffGui gap is "
        f"driven by 2 of 3 targets and reverses on 1iep."
    )
    return (stat_md,)


@app.cell
def _(mo, stat_md):
    mo.md(stat_md)
    return


@app.cell
def _(GEN, np, pd, plt):
    # Per-molecule Vina score distribution: ours (T=0.85) vs baselines pooled.
    ours_mol = pd.read_parquet(GEN / "tsweep_per_molecule.parquet")
    ours_v = ours_mol[ours_mol.model == "own_t085_on"]["vina_score"].dropna()
    ours_v = ours_v[np.isfinite(ours_v)]
    fig_h, ax_h = plt.subplots(figsize=(6, 3.2))
    ax_h.hist(ours_v, bins=40, color="#c0392b", alpha=0.8)
    ax_h.axvline(ours_v.median(), color="k", ls="--", lw=1,
                 label=f"median {ours_v.median():.2f}")
    ax_h.set_xlabel("Vina score (per generated molecule)")
    ax_h.set_ylabel("count")
    ax_h.set_title("Ours (T=0.85) — per-molecule Vina distribution")
    ax_h.legend(fontsize=8)
    fig_h.tight_layout()
    fig_h
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    **Takeaway.** With the all-atom tokenizer, raw Vina went from +1.21
    (pre-loop) to **−5.33** (T=0.85), beating DiffSBDD (−4.40) and TargetDiff
    (−4.76) and closing most of the gap to DiffGui (−6.54). The two levers that
    mattered: placement re-finetune (decisive) and sampling temperature 0.85.
    """)
    return


if __name__ == "__main__":
    app.run()
