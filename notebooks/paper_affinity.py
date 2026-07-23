# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "numpy",
#     "scipy",
#     "matplotlib",
# ]
# ///
"""Paper figure/table: all-atom affinity head (CASF-2016 scoring & ranking power).

Our model (see docs/best_allatom_configs.md):
  leak-free MLM backbone wxlhgqx3 + pK-regression heads. Best number = a fixed
  5-head ensemble (no test-set selection). Scoring power = Pearson R of predicted
  vs experimental pKa over 285 complexes; ranking power = mean within-cluster
  Spearman rho over the 57 CASF target clusters.
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr

    return Path, mo, np, pd, pearsonr, plt, spearmanr


@app.cell
def _(Path):
    REPO = Path("/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen")
    CASF = REPO / "outputs" / "casf"
    BASE = REPO.parent / "baselines" / "casf_work"
    return BASE, CASF


@app.cell
def _(mo):
    mo.md("""
    # Affinity — CASF-2016 scoring & ranking power

    The same encoder+pooling+MLP as the pose head, but the regression target is
    **pK** instead of RMSD. Reported number is a **fixed 5-head ensemble**
    (no test-set selection). Uses the leak-free MLM backbone (`wxlhgqx3`,
    CASF fully excluded from pretraining).
    """)
    return


@app.cell
def _(np, pearsonr, spearmanr):
    def scoring_R(df):
        d = df.dropna(subset=["logka", "head"])
        return pearsonr(d["logka"], d["head"])[0]

    def ranking_rho(df):
        d = df.dropna(subset=["logka", "head"])
        rs = []
        for _, g in d.groupby("cluster"):
            if len(g) >= 3:
                r = spearmanr(g["logka"], g["head"]).correlation
                if np.isfinite(r):
                    rs.append(r)
        return float(np.mean(rs))

    return ranking_rho, scoring_R


@app.cell
def _(CASF, pd):
    # The fixed leak-free 5 (LF5): 2x2 (mean/attn) x (IC50/Kd-Ki) + meanmax x Kd-Ki.
    LF5 = {
        "mean × IC50": "affinity_power_lf.csv",
        "attn × IC50": "affinity_all_attn.csv",
        "mean × Kd/Ki": "affinity_kdki_mean.csv",
        "attn × Kd/Ki": "affinity_kdki_attn.csv",
        "meanmax × Kd/Ki": "affinity_kdki_meanmax.csv",
    }
    lf5_frames = {k: pd.read_csv(CASF / v) for k, v in LF5.items()}
    return (lf5_frames,)


@app.cell
def _(lf5_frames, pd, ranking_rho, scoring_R):
    member_table = pd.DataFrame(
        [{"head": k, "scoring R": round(scoring_R(d), 3), "ranking ρ": round(ranking_rho(d), 3)}
         for k, d in lf5_frames.items()]
    ).set_index("head")
    return (member_table,)


@app.cell
def _(mo):
    mo.md("""
    ## The five ensemble members
    """)
    return


@app.cell
def _(member_table):
    member_table
    return


@app.cell
def _(lf5_frames):
    # Fixed z-sum ensemble across the 5 members (aligned on pdbid).
    ens = None
    for i, (k, d) in enumerate(lf5_frames.items()):
        dd = d.dropna(subset=["logka", "head"]).sort_values("pdbid").reset_index(drop=True)
        if ens is None:
            ens = dd[["pdbid", "logka", "cluster"]].copy()
            ens["head"] = 0.0
        h = dd["head"].values
        ens["head"] = ens["head"].values + (h - h.mean()) / h.std()
    return (ens,)


@app.cell
def _(mo):
    mo.md("""
    ## Head-to-head comparison
    """)
    return


@app.cell
def _(CASF, ens, pd, ranking_rho, scoring_R):
    # Baseline aggregates from the finished cross-method run.
    method_cmp = pd.read_csv(CASF / "method_comparison.csv").set_index("method")
    rows = []
    for m in ["GenScore", "Boltz-2", "Vina"]:
        if m in method_cmp.index:
            rows.append({"method": m,
                         "scoring R ↑": round(method_cmp.loc[m, "scoring_R"], 3),
                         "ranking ρ ↑": round(method_cmp.loc[m, "ranking_rho"], 3)})
    rows.append({"method": "OURS (LF5 ensemble)",
                 "scoring R ↑": round(scoring_R(ens), 3),
                 "ranking ρ ↑": round(ranking_rho(ens), 3)})
    affinity_table = (pd.DataFrame(rows).set_index("method")
                      .reindex(["GenScore", "Boltz-2", "OURS (LF5 ensemble)", "Vina"]))
    return (affinity_table,)


@app.cell
def _(affinity_table):
    affinity_table
    return


@app.cell
def _(mo):
    mo.md("""
    **Reading it.** The LF5 ensemble is **statistically tied with GenScore on
    both** metrics (scoring Steiger p=0.21, ranking Wilcoxon p=0.14) and **beats
    Boltz-2 on scoring** on the point estimate. RTMScore is absent — it has no
    scoring function (pose selector only).
    """)
    return


@app.cell
def _(BASE, ens, pd, pearsonr, plt):
    # Scatter: our ensemble (z-sum, scale-free) vs experimental pKa, next to GenScore.
    gen = pd.read_csv(BASE / "scoring_power_genscore.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    r_ours = pearsonr(ens["logka"], ens["head"])[0]
    ax[0].scatter(ens["head"], ens["logka"], s=16, alpha=0.6, color="#c0392b")
    ax[0].set_title(f"OURS (LF5 ensemble)   R = {r_ours:.3f}", fontsize=10)
    ax[0].set_xlabel("predicted (ensemble z-sum)")
    ax[0].set_ylabel("experimental pKa")

    r_gen = pearsonr(gen["logka"], gen["score"])[0]
    ax[1].scatter(gen["score"], gen["logka"], s=16, alpha=0.6, color="#2c3e50")
    ax[1].set_title(f"GenScore   R = {r_gen:.3f}", fontsize=10)
    ax[1].set_xlabel("GenScore prediction")

    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-cluster ranking (where the gap lives)
    Ranking power is the mean of the within-cluster Spearman ρ. Our weakness is
    concentrated in a minority of clusters where sub-0.5 pK differences between
    congeners are not resolved (a tokenizer-resolution limit, not a pooling bug —
    see memory `project_mlm_rescorer` for the full 9-lever diagnosis).
    """)
    return


@app.cell
def _(ens, np, plt, spearmanr):
    per_cluster = []
    for cid, g in ens.groupby("cluster"):
        if len(g) >= 3:
            per_cluster.append(spearmanr(g["logka"], g["head"]).correlation)
    per_cluster = np.array([r for r in per_cluster if np.isfinite(r)])
    fig_c, ax_c = plt.subplots(figsize=(6, 3.2))
    ax_c.hist(per_cluster, bins=15, color="#c0392b", alpha=0.8)
    ax_c.axvline(per_cluster.mean(), color="k", ls="--", lw=1,
                 label=f"mean ρ = {per_cluster.mean():.3f}")
    ax_c.set_xlabel("within-cluster Spearman ρ")
    ax_c.set_ylabel("clusters")
    ax_c.set_title("Per-cluster ranking power (LF5 ensemble)")
    ax_c.legend(fontsize=8)
    fig_c.tight_layout()
    fig_c
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    **Takeaway.** A lightweight pK head on the all-atom complex tokens reaches
    **R=0.790 / ρ=0.674**, statistically indistinguishable from GenScore and
    beating Boltz-2 on scoring — with no 3D coordinates at inference, only
    discrete tokens. The pose head and this affinity head are separate
    specialists sharing the same encoder.
    """)
    return


if __name__ == "__main__":
    app.run()
