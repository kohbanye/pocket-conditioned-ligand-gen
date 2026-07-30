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
"""Paper figure/table: all-atom pose rescoring (CASF-2016 docking power).

Our model (see docs/results/best_allatom_configs.md):
  MLM backbone j90rlrgm + RMSD-regression heads (v2 mean / v6 meanmax / v7 attn),
  optionally z-sum fused with the classical Vina score.
Metric: docking power = fraction of targets whose top-scored decoy is within a
cutoff of native. DP@2Å = standard CASF docking power; DP@1Å = near-native.
All methods scored on the identical CASF decoy pose set (decoys only).
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import glob
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    return Path, glob, mo, np, os, pd, plt, spearmanr


@app.cell
def _(Path, os):
    # The repository root, found from this file so the notebook runs from any
    # checkout. PROLIT_ROOT overrides it.
    REPO = Path(
        os.environ.get("PROLIT_ROOT")
        or Path(__file__).resolve().parent.parent
    )
    CASF = REPO / "outputs" / "casf"
    BASE = REPO.parent / "baselines" / "casf_work"  # RTMScore / GenScore per-pose
    return BASE, CASF


@app.cell
def _(mo):
    mo.md("""
    # Pose rescoring — CASF-2016 docking power

    Each method ranks the docked decoy poses of a target; success = its
    **top-scored** pose is within the RMSD cutoff of the crystal native.
    Our learned head regresses RMSD (higher score = more native-like); the
    winning system fuses several heads with the classical Vina score.
    """)
    return


@app.cell
def _(CASF, pd):
    # rmsd lookup shared by every method (same decoy set). Our head CSVs carry it.
    v2 = pd.read_csv(CASF / "pose_scores.csv")      # v2 = mean pool
    rmsd_key = v2[["pdbid", "pose", "rmsd"]].copy()
    return (rmsd_key,)


@app.cell
def _(BASE, CASF, glob, pd, rmsd_key, spearmanr):
    def orient(d):
        # Orient a raw score column into a "native-likeness" (higher = more
        # native-like) regardless of each method's storage convention, by making
        # it anti-correlate with RMSD over the full pooled pose set. Robust because
        # every method here has a clear (|rho|>0.2) signal.
        raw = d["native_score"].values
        if spearmanr(raw, d["rmsd"]).correlation > 0:  # higher raw ↔ higher rmsd (worse)
            d = d.assign(native_score=-raw)
        return d

    def our_head(csv):
        d = pd.read_csv(CASF / csv)[["pdbid", "pose", "rmsd", "head"]]
        return orient(d.rename(columns={"head": "native_score"}))

    def vina_method():
        d = pd.read_csv(CASF / "pose_scores_vina.csv")[["pdbid", "pose", "rmsd", "head"]]
        return orient(d.rename(columns={"head": "native_score"}))

    def baseline(subdir):
        rows = []
        for f in glob.glob(str(BASE / subdir / "*_score.dat")):
            t = pd.read_csv(f, sep="\t")
            t.columns = ["pose", "native_score"][: len(t.columns)]
            t["pdbid"] = t["pose"].str.split("_").str[0]
            rows.append(t)
        d = pd.concat(rows, ignore_index=True).merge(rmsd_key, on=["pdbid", "pose"], how="inner")
        return orient(d)

    methods = {
        "OURS v2 (mean)": our_head("pose_scores.csv"),
        "OURS v6 (meanmax)": our_head("pose_scores_v6.csv"),
        "OURS v7 (attn)": our_head("pose_scores_v7.csv"),
        "Vina": vina_method(),
        "RTMScore": baseline("scores_rtmscore"),
        "GenScore": baseline("scores_genscore"),
    }
    return (methods,)


@app.cell
def _(np, spearmanr):
    def docking_power(df, cut):
        ok = tot = 0
        for _, g in df.groupby("pdbid"):
            tot += 1
            if g.loc[g["native_score"].idxmax(), "rmsd"] < cut:
                ok += 1
        return 100.0 * ok / tot

    def ranking_rho(df):
        rs = []
        for _, g in df.groupby("pdbid"):
            if len(g) >= 3:
                r = spearmanr(g["native_score"], g["rmsd"]).correlation
                if np.isfinite(r):
                    rs.append(-r)  # native-like high score vs low rmsd -> report +
        return float(np.mean(rs))

    def zsum(dfs):
        # z-score native_score within each target, then sum across methods.
        merged = None
        for i, d in enumerate(dfs):
            dd = d.copy()
            dd["z"] = dd.groupby("pdbid")["native_score"].transform(
                lambda s: (s - s.mean()) / (s.std() + 1e-9))
            dd = dd[["pdbid", "pose", "rmsd", "z"]].rename(columns={"z": f"z{i}"})
            merged = dd if merged is None else merged.merge(dd.drop(columns="rmsd"),
                                                            on=["pdbid", "pose"], how="inner")
        zcols = [c for c in merged.columns if c.startswith("z")]
        merged["native_score"] = merged[zcols].sum(axis=1)
        return merged

    return docking_power, ranking_rho, zsum


@app.cell
def _(docking_power, methods, pd, ranking_rho, zsum):
    ens_pose = zsum([methods["OURS v2 (mean)"], methods["OURS v6 (meanmax)"],
                     methods["OURS v7 (attn)"]])
    ens_pose_vina = zsum([methods["OURS v2 (mean)"], methods["OURS v6 (meanmax)"],
                          methods["OURS v7 (attn)"], methods["Vina"]])
    ens_v2_vina = zsum([methods["OURS v2 (mean)"], methods["Vina"]])

    table_rows = []
    for name, d in [
        ("RTMScore", methods["RTMScore"]),
        ("GenScore", methods["GenScore"]),
        ("Vina", methods["Vina"]),
        ("OURS v2 (single head)", methods["OURS v2 (mean)"]),
        ("OURS 3-head ensemble", ens_pose),
        ("OURS v2 + Vina", ens_v2_vina),
        ("OURS 3-head + Vina", ens_pose_vina),
    ]:
        table_rows.append({
            "method": name,
            "n_targets": d["pdbid"].nunique(),
            "DP@2Å (docking power) ↑": round(docking_power(d, 2.0), 1),
            "DP@1Å (near-native) ↑": round(docking_power(d, 1.0), 1),
            "Spearman ρ ↑": round(ranking_rho(d), 3),
        })
    pose_table = pd.DataFrame(table_rows).set_index("method")
    return ens_pose, pose_table


@app.cell
def _(pose_table):
    pose_table
    return


@app.cell
def _(mo):
    mo.md("""
    **Reading it.** On the standard 2Å docking power we sit just under GenScore
    and below RTMScore. On the stricter **near-native (1Å)** metric the
    Vina-fused ensemble is competitive with / edges the baselines — this is the
    metric where the learned+physics consensus helps most. The single head has
    the best ranking ρ; adding Vina trades ρ for near-native top-1.
    """)
    return


@app.cell
def _(plt, pose_table):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, col in zip(ax, ["DP@2Å (docking power) ↑", "DP@1Å (near-native) ↑"]):
        s = pose_table[col]
        colors = ["#c0392b" if m.startswith("OURS") else "#8a8a8a" for m in s.index]
        a.barh(list(s.index), s.values, color=colors)
        a.set_title(col, fontsize=10)
        a.invert_yaxis()
        for i, v in enumerate(s.values):
            a.text(v, i, f" {v:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig
    return


@app.cell
def _(ens_pose, mo, plt):
    # score-vs-RMSD scatter for one illustrative CASF target (a success case:
    # >=20 decoys and the top-scored pose is near-native). Chosen dynamically so
    # we never hard-code a pdbid that is absent from the CASF-285 rescoring set.
    def _pick(df):
        for pid, gg in df.groupby("pdbid"):
            if len(gg) >= 20 and gg.loc[gg["native_score"].idxmax(), "rmsd"] < 1.0:
                return pid
        return df["pdbid"].iloc[0]

    tid = _pick(ens_pose)
    g = ens_pose[ens_pose.pdbid == tid]
    fig_s, ax_s = plt.subplots(figsize=(5, 3.4))
    ax_s.scatter(g["rmsd"], g["native_score"], s=14, alpha=0.7, color="#2c3e50")
    top = g.loc[g["native_score"].idxmax()]
    ax_s.scatter([top["rmsd"]], [top["native_score"]], color="#c0392b", s=60,
                 zorder=5, label=f"top-scored (rmsd {top['rmsd']:.2f}Å)")
    ax_s.axvline(2.0, color="gray", ls="--", lw=1)
    ax_s.set_xlabel("pose RMSD to native (Å)")
    ax_s.set_ylabel("native-likeness (ensemble z-sum)")
    ax_s.set_title(f"{tid}: decoy poses, our 3-head ensemble")
    ax_s.legend(fontsize=8)
    fig_s.tight_layout()
    _ = mo.md(f"Illustrative target **{tid}** — the top-scored decoy is near-native.")
    fig_s
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    **Note on ensemble composition.** The pose-rescoring system fuses **pose
    heads (v2/v6/v7) + Vina** — *not* the affinity head. The affinity head
    (pK regression) is a separate specialist and carries essentially no
    pose-RMSD signal by design. Both tasks' best results are ensembles, but of
    different members (pose: pose-heads + Vina; affinity: 5 affinity heads).
    """)
    return


if __name__ == "__main__":
    app.run()
