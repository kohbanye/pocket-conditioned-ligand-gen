# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas>=2.0",
#     "numpy>=1.26",
#     "matplotlib>=3.8",
# ]
# ///
"""EGFR 2ITY — docking of pocket-conditioned generated ligands.

Summarises the experiment run by ``scripts/run_egfr_2ity_docking.sh``:
~10k ligands generated conditioned on the EGFR kinase pocket (PDB 2ITY,
co-crystallised with gefitinib / Iressa), each docked with AutoDock Vina in
two modes — Vina Score (the generated pose scored as-is, ``score_only``) and
Vina Min (after local minimisation, ``local_only``) — with the crystal
gefitinib pose as a positive-control reference.

Open with::

    uv run marimo edit notebooks/egfr_2ity_docking.py

Point it at a different results dir via the ``EGFR_OUT`` env var.
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

    # Enlarged fonts for readability; x/y axis labels are deliberately large.
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 19,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    })
    return Path, mo, np, os, pd, plt


@app.cell
def _(Path, os):
    # Results directory (override with EGFR_OUT=... for the smoke-test outputs).
    data_dir = Path(os.environ.get("EGFR_OUT", "outputs/egfr_2ity"))
    return (data_dir,)


@app.cell
def _(mo):
    mo.md(r"""
    # EGFR (2ITY) — docking pocket-conditioned generated ligands

    **Target.** EGFR kinase domain, PDB **2ITY**, co-crystallised with the
    approved inhibitor **gefitinib (Iressa, ligand `IRE`)**. The pocket
    (residues around the crystal ligand) conditions the autoregressive
    ligand LM; ~10k ligands are sampled and decoded to 3D.

    **Docking (AutoDock Vina 1.2.7).** Each generated pose is evaluated two
    ways, following the standard SBDD protocol (cf. TargetDiff Vina
    Score / Vina Min):

    - **Vina Score** — `--score_only` on the generated coordinates (no movement).
    - **Vina Min** — `--local_only`, a local energy minimisation starting
      from the generated pose, then scored.

    The crystal **gefitinib** pose is docked identically as a positive
    control. Lower (more negative) kcal/mol is better.
    """)
    return


@app.cell
def _(data_dir, pd):
    meta = pd.read_csv(data_dir / "generated_meta.csv")
    dock = pd.read_csv(data_dir / "docking_results.csv")
    # Single per-ligand table; suffix overlapping columns from the dock side.
    df = meta.merge(
        dock.drop(columns=["tag", "n_atoms"]), on="idx", how="left"
    )
    ref = df[df["idx"] < 0].iloc[0] if (df["idx"] < 0).any() else None
    gen = df[df["idx"] >= 0].copy()
    return gen, meta, ref


@app.cell
def _(gen, meta, mo, ref):
    n_total = len(meta[meta["idx"] >= 0])
    n_term = int((meta[meta["idx"] >= 0]["terminated"]).sum())
    n_dockable = int((meta[meta["idx"] >= 0]["dockable"]).sum())
    n_docked = int((gen["dock_ok"] == True).sum())  # noqa: E712
    ref_line = (
        f"gefitinib reference — Vina Score **{ref['score_as_is']:.2f}**, "
        f"Vina Min **{ref['score_opt']:.2f}** kcal/mol"
        if ref is not None
        else "no reference row found"
    )
    mo.md(
        f"""
        ## Generation → docking funnel

        | stage | count | of generated |
        |---|---:|---:|
        | generated (sampled) | {n_total} | 100.0% |
        | cleanly terminated (`</l>`) | {n_term} | {100 * n_term / n_total:.1f}% |
        | dockable (real elements, size ok) | {n_dockable} | {100 * n_dockable / n_total:.1f}% |
        | docked OK (Vina scored) | {n_docked} | {100 * n_docked / n_total:.1f}% |

        Positive control: {ref_line}.
        """
    )
    return (n_docked,)


@app.cell
def _(gen, mo, np):
    ok = gen[gen["dock_ok"] == True].copy()  # noqa: E712
    asis = ok["score_as_is"].to_numpy(dtype=float)
    opt = ok["score_opt"].to_numpy(dtype=float)
    asis = asis[np.isfinite(asis)]
    opt = opt[np.isfinite(opt)]
    mo.md(f"Docked OK: **{len(ok)}** ligands with finite scores.")
    return asis, ok, opt


@app.cell
def _(asis, mo, np, opt, ref):
    def _stats(x):
        return (
            f"{np.median(x):.2f} | {np.mean(x):.2f} | {np.min(x):.2f} | "
            f"{100 * np.mean(x < 0):.0f}%"
        )

    better = ""
    if ref is not None:
        f_asis = 100 * np.mean(asis < float(ref["score_as_is"]))
        f_opt = 100 * np.mean(opt < float(ref["score_opt"]))
        better = (
            f"\n\n**Beating the gefitinib reference:** "
            f"{f_asis:.1f}% of Vina Score values and {f_opt:.1f}% of Vina Min "
            f"values are below (better than) the crystal inhibitor."
        )
    mo.md(
        f"""
        ## Score distributions — Vina Score vs Vina Min

        | mode | median | mean | best | fraction < 0 |
        |---|---:|---:|---:|---:|
        | Vina Score (as-is pose) | {_stats(asis)} |
        | Vina Min (optimized pose) | {_stats(opt)} |

        Raw generated poses carry clashes, so many **Vina Score** values are weak
        or positive; a single local minimisation removes most clashes and pulls
        the **Vina Min** distribution sharply negative.{better}
        """
    )
    return


@app.cell
def _(asis, np, opt, plt, ref):
    fig_hist, ax_hist = plt.subplots(figsize=(9, 5))
    lo = float(np.floor(min(asis.min(), opt.min())))
    hi = float(np.ceil(min(20.0, max(asis.max(), opt.max()))))
    bins = np.linspace(lo, hi, 60)
    ax_hist.hist(np.clip(asis, lo, hi), bins=bins, alpha=0.55,
                 label="Vina Score", color="#d98c5f")
    ax_hist.hist(np.clip(opt, lo, hi), bins=bins, alpha=0.55,
                 label="Vina Min", color="#4c78a8")
    if ref is not None:
        ax_hist.axvline(float(ref["score_as_is"]), color="#7f3b08", ls=":", lw=2.2,
                        label=f"Gefitinib Vina Score ({ref['score_as_is']:.2f})")
        ax_hist.axvline(float(ref["score_opt"]), color="#08306b", ls="--", lw=2.2,
                        label=f"Gefitinib Vina Min ({ref['score_opt']:.2f})")
    ax_hist.set_xlabel("Docking score (kcal/mol)")
    ax_hist.set_ylabel("number of ligands")
    ax_hist.set_title("Generated-ligand docking scores: Vina Score vs Vina Min")
    ax_hist.legend()
    fig_hist.tight_layout()
    fig_hist
    return


@app.cell
def _(np, ok, plt, ref):
    # Per-ligand improvement from minimisation, and Vina Score vs Vina Min.
    fig_sc, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5))

    paired = ok.dropna(subset=["score_as_is", "score_opt"])
    pa = paired["score_as_is"].to_numpy(dtype=float)
    po = paired["score_opt"].to_numpy(dtype=float)
    ax_a.scatter(np.clip(pa, None, 20), po, s=6, alpha=0.3, color="#4c78a8")
    lim = [min(po.min(), -10), 20]
    ax_a.plot(lim, lim, color="#888", ls=":", lw=1)
    if ref is not None:
        ax_a.scatter([float(ref["score_as_is"])], [float(ref["score_opt"])],
                     color="crimson", s=90, marker="*", label="Gefitinib", zorder=5)
        ax_a.legend()
    ax_a.set_xlabel("Vina Score (kcal/mol)")
    ax_a.set_ylabel("Vina Min (kcal/mol)")
    ax_a.set_title("Vina Score vs Vina Min (below diagonal = improved)")

    improvement = pa - po  # positive = minimisation lowered the score
    ax_b.hist(np.clip(improvement, -2, 25), bins=50, color="#72b07a")
    ax_b.set_xlabel("improvement (Vina Score − Vina Min, kcal/mol)")
    ax_b.set_ylabel("number of ligands")
    ax_b.set_title(f"minimisation gain (median {np.median(improvement):.2f})")
    fig_sc.tight_layout()
    fig_sc
    return


@app.cell
def _(ok, plt):
    # How far did minimisation move the pose?
    rmsd = ok["opt_rmsd"].dropna().to_numpy(dtype=float)
    fig_rmsd, ax_rmsd = plt.subplots(figsize=(8, 4.5))
    ax_rmsd.hist(rmsd, bins=50, color="#9a77b8")
    ax_rmsd.set_xlabel("heavy-atom RMSD: generated → minimised pose (Å)")
    ax_rmsd.set_ylabel("number of ligands")
    ax_rmsd.set_title("Pose movement during Vina Min (local optimization)")
    fig_rmsd.tight_layout()
    fig_rmsd
    return


@app.cell
def _(mo, ok):
    # Top hits by optimized score.
    cols = ["idx", "formula", "n_atoms", "n_atoms_docked",
            "score_as_is", "score_opt", "opt_rmsd"]
    top = (
        ok.dropna(subset=["score_opt"])
        .sort_values("score_opt")
        .head(20)[cols]
        .round(2)
        .reset_index(drop=True)
    )
    mo.md("## Top 20 generated ligands (best Vina Min)")
    return (top,)


@app.cell
def _(mo, top):
    mo.ui.table(top, selection=None)
    return


@app.cell
def _(np, ok, plt):
    # Score vs molecule size — bigger ligands tend to score lower; sanity check.
    fig_sz, ax_sz = plt.subplots(figsize=(8, 4.5))
    nat = ok["n_atoms"].to_numpy(dtype=float)
    sc = ok["score_opt"].to_numpy(dtype=float)
    m = np.isfinite(sc)
    ax_sz.scatter(nat[m], sc[m], s=6, alpha=0.25, color="#4c78a8")
    ax_sz.set_xlabel("heavy-atom count")
    ax_sz.set_ylabel("Vina Min (kcal/mol)")
    ax_sz.set_title("Vina Min vs ligand size")
    fig_sz.tight_layout()
    fig_sz
    return


@app.cell
def _(mo, n_docked):
    mo.md(
        f"""
        ## Takeaways

        - **{n_docked}** generated ligands were successfully docked against EGFR.
        - The **Vina Score** vs **Vina Min** gap quantifies how much of the score
          is limited by generated-pose geometry (clashes) versus the underlying
          chemistry: a large gain from a *local* minimisation (no global
          re-docking) means the model places roughly the right atoms but with
          imperfect local geometry.
        - Compare both distributions against the **gefitinib** crystal-pose
          reference to judge whether the generator proposes binders competitive
          with a known inhibitor.

        *Caveats.* Vina scores generated heavy-atom structures whose bonds and
        protonation are re-perceived by Open Babel; the largest fragment is
        docked when a molecule is disconnected. These are docking estimates, not
        validated affinities — treat them as a relative screen.
        """
    )
    return


if __name__ == "__main__":
    app.run()
