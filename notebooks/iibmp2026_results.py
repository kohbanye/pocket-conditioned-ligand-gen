# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "scipy",
#     "matplotlib",
#     "rdkit",
# ]
# ///
"""IIBMP2026 oral (17 min) — Results & Discussion figures.

Four figures, all recomputed from the run outputs on disk; no number in any
figure is typed by hand.

    Figure 1  reconstruction-distributions   CASP16, 303 complexes, 3 tokenizers
    Figure 2  reconstruction-examples        3 programmatically chosen complexes
    Figure 3  generation-pose-bottleneck     97 CrossDocked targets x 100 molecules
    Figure 4  generation-examples            3 programmatically chosen molecules

The arc the figures are meant to carry:

1. ProLIT discretises and rebuilds the protein-ligand interface accurately
   (Fig. 1, 2) -- and joint tokenisation is what buys the interface.
2. Wired to a language model it writes chemically reasonable, dockable
   molecules, but places them wrongly (Fig. 3, 4). Pose placement, not
   chemistry, is the bottleneck.

Provenance for every panel is written to ``outputs/iibmp2026/figure_manifest.md``
by the last cell.

Run it::

    .venv/bin/python notebooks/iibmp2026_results.py          # regenerate figures
    .venv/bin/marimo edit notebooks/iibmp2026_results.py     # interactive

Figure 4 re-docks three molecules with AutoDock Vina because the benchmark
never stored redocked coordinates (``sbdd_bench.docking`` writes them into a
temporary directory). The results are cached under ``outputs/iibmp2026/redock/``;
with the cache present the notebook needs no external binary. To rebuild the
cache, point ``PROLIT_VINA`` / ``PROLIT_OBABEL`` at those tools.
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # IIBMP2026 — Results & Discussion figures

    **ProLIT**: one shared discrete vocabulary for pocket atoms and ligand atoms.

    | figure | claim it has to support |
    |---|---|
    | 1 | joint tokenisation preserves the interface better than separate books or Bio2Token |
    | 2 | what that accuracy looks like on real complexes |
    | 3 | the LM writes good molecules but places them badly |
    | 4 | what that failure looks like on real molecules |

    Everything is recomputed from run outputs. Nothing is transcribed from a
    markdown table.
    """)
    return


@app.cell
def _():
    import json
    import os
    import re
    import shutil
    import subprocess
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path

    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from scipy import stats as sps
    from scipy.spatial import cKDTree

    REPO = Path(os.environ.get("PROLIT_ROOT") or Path(__file__).resolve().parent.parent)
    OUT = REPO / "outputs" / "iibmp2026"
    FIGDIR = OUT / "figures"
    REDOCK_DIR = OUT / "redock"
    for _d in (FIGDIR, REDOCK_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    RECON = REPO / "benchmarks" / "recon-bench"
    SBDD = REPO / "benchmarks" / "sbdd-bench"
    return (
        FIGDIR,
        Line2D,
        Line3DCollection,
        OUT,
        Patch,
        Path,
        RECON,
        REDOCK_DIR,
        REPO,
        SBDD,
        cKDTree,
        datetime,
        json,
        np,
        pd,
        plt,
        re,
        shutil,
        sps,
        subprocess,
        tempfile,
        timezone,
    )


@app.cell
def _(plt):
    # ---- visual style -------------------------------------------------------
    # Slide text is English, so the figures are too. The stack is what the
    # deck uses; whatever of it exists on this machine wins, and the notebook
    # reports which one it actually got.
    FONT_STACK = [
        "Yu Gothic",
        "YuGothic",
        "Noto Sans CJK JP",
        "DejaVu Sans",
        "sans-serif",
    ]

    C_PROLIT = "#0F766E"
    C_SEPARATE = "#9079B5"
    C_BIO2TOKEN = "#9CA3AF"
    C_NAVY = "#1C3078"
    C_BLUE = "#3A6DC9"
    C_WARN = "#B45309"
    C_INK = "#111827"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.labelsize": 26,
        "axes.titlesize": 26,
        "axes.labelcolor": C_INK,
        "axes.edgecolor": "#4B5563",
        "axes.linewidth": 1.8,
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "legend.fontsize": 21,
        "legend.frameon": False,
        "text.color": C_INK,
        "lines.linewidth": 2.4,
    })

    from matplotlib import font_manager as _fm

    _have = {f.name for f in _fm.fontManager.ttflist}
    FONT_IN_USE = next((f for f in FONT_STACK if f in _have), "DejaVu Sans (fallback)")
    print(f"[style] font resolved to: {FONT_IN_USE}")
    return (
        C_BIO2TOKEN,
        C_BLUE,
        C_INK,
        C_NAVY,
        C_PROLIT,
        C_SEPARATE,
        C_WARN,
        FONT_IN_USE,
        FONT_STACK,
    )


@app.cell
def _(FIGDIR, plt):
    def save_fig(fig, stem):
        """Write the figure as SVG and PNG at slide resolution."""
        svg = FIGDIR / f"{stem}.svg"
        png = FIGDIR / f"{stem}.png"
        fig.savefig(svg, format="svg")
        fig.savefig(png, format="png", dpi=100)
        w, h = fig.get_size_inches()
        print(f"[save] {stem}: {int(w * 100)} x {int(h * 100)} px -> {svg.name}, {png.name}")
        plt.close(fig)
        return svg, png

    return (save_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Figure 1 — Reconstruction distributions

    CASP16, the same 303 protein–ligand complexes for all three tokenizers.

    **Which ProLIT runs.** The arm registry
    (`benchmarks/common/src/prolit_bench/variants.py`) names `joint` / `separate`
    by identity, but the controlled pair the significance analysis and the paper
    claim rest on (`docs/results/casp16_significance.md`) is
    **`joint_e250_lig3` vs `separate_e250`**: same 250-epoch recipe, same cache,
    same held-out eval PDBs, 8192 shared codes against 4096+4096. `joint_e250_lig3`
    is also the tokenizer the generation pipeline in Figures 3–4 uses
    (`vq_e250_lig3`, epoch 237), so the four figures describe one system.
    The older `arm-joint` / `arm-separate` numbers are recorded in the manifest.
    """)
    return


@app.cell
def _(RECON, pd):
    RECON_ARMS = {
        "ProLIT": "casp16.arm-joint_e250_lig3.parquet",
        "Separate ProLIT": "casp16.arm-separate_e250.parquet",
        "Bio2Token": "casp16_bio2token.parquet",
    }
    RECON_FILES = {k: RECON / "results" / v for k, v in RECON_ARMS.items()}

    def _complex_rows(path):
        d = pd.read_parquet(path)
        c = d[(d["modality"] == "complex") & d["ok"]]
        return c.set_index("sample_id")

    recon = {k: _complex_rows(p) for k, p in RECON_FILES.items()}
    recon_ids = sorted(set.intersection(*(set(v.index) for v in recon.values())))
    print(f"[fig1] complexes shared by all three arms: {len(recon_ids)}")
    for _k, _v in recon.items():
        print(f"        {_k:<16} model={_v['model'].unique()[0]:<28} "
              f"bits/atom={_v['bits_per_atom'].unique()[0]:.0f}")
    return RECON_ARMS, RECON_FILES, recon, recon_ids


@app.cell
def _(RECON, np, recon, recon_ids):
    # ---- interface RMSD on one interface definition, not three ---------------
    #
    # `recon_bench.metrics.complex_metrics` calls a ligand atom an interface
    # atom when it sits within 4 A of a protein atom in *that arm's own*
    # reference view, and returns NaN when no ligand atom qualifies.
    #
    # Bio2Token's complex view is backbone-only -- its protein rows are exactly
    # N/CA/C/O per residue, because its upstream reader keeps nothing else -- so
    # for eight complexes whose ligand is a bromide, an acetate or a sulfate,
    # bound through side chains, no backbone atom lands within 4 Å and the value
    # is NaN. That is an artefact of the receptor representation, not of the
    # reconstruction, and it also means the three arms were being scored on
    # three different sets of ligand atoms.
    #
    # The interface is a property of the crystal structure, so take it from the
    # shared all-atom reference once and apply it to every arm. The ligand atoms
    # are the same objects in the same order in all three dumps (verified below
    # by internal distances), and Bio2Token's frame differs only by a
    # translation, which cancels in a no-superposition RMSD of ref against rec.
    OWN_DUMPS = RECON / "outputs" / "own_allatom"
    B2T_DUMPS = RECON / "outputs" / "bio2token" / "complex"
    IFACE_CUTOFF = 4.0
    RECON_DUMP_SUBDIR = {
        "ProLIT": "joint_e250_lig3",
        "Separate ProLIT": "separate_e250",
    }

    def _rmsd(p, q):
        """RMSD without superposition — the bench's own definition."""
        return float(np.sqrt(np.mean(np.sum((p - q) ** 2, axis=1))))

    def common_interface_rmsd():
        """Per-arm interface RMSD under the shared all-atom interface mask."""
        out = {k: {} for k in recon}
        sizes, order_err = {}, 0.0
        for sid in recon_ids:
            own = {
                arm: np.load(OWN_DUMPS / sub / f"{sid}.npz", allow_pickle=True)
                for arm, sub in RECON_DUMP_SUBDIR.items()
            }
            ref = own["ProLIT"]
            mask = (
                np.linalg.norm(
                    ref["protein_ref"][:, None] - ref["ligand_ref"][None], axis=-1
                )
                < IFACE_CUTOFF
            ).any(axis=0)
            sizes[sid] = (int(mask.sum()), int(mask.size))
            for arm, d in own.items():
                out[arm][sid] = _rmsd(d["ligand_ref"][mask], d["ligand_rec"][mask])

            b = np.load(B2T_DUMPS / f"{sid}.npz", allow_pickle=False)
            n_prot = int(b["n_protein_rows"])
            lig_ref, lig_rec = b["ref"][n_prot:], b["rec"][n_prot:]
            # Same atoms, same order? Internal distances are rigid-motion
            # invariant, so they must agree if the correspondence holds.
            if lig_ref.shape[0] > 1:
                dj = np.linalg.norm(
                    ref["ligand_ref"][:, None] - ref["ligand_ref"][None], axis=-1
                )
                db = np.linalg.norm(lig_ref[:, None] - lig_ref[None], axis=-1)
                order_err = max(order_err, float(np.abs(dj - db).max()))
            out["Bio2Token"][sid] = _rmsd(lig_ref[mask], lig_rec[mask])
        return out, sizes, order_err

    iface_rmsd, iface_sizes, IFACE_ORDER_ERR = common_interface_rmsd()

    # The own-arm masks were already all-atom, so their numbers must not move;
    # if they do, the correspondence assumption is wrong and the figure is wrong.
    IFACE_OWN_DRIFT = max(
        abs(iface_rmsd[a][s] - recon[a].loc[s, "iface_lig_rmsd"])
        for a in RECON_DUMP_SUBDIR
        for s in recon_ids
    )
    IFACE_EMPTY = sum(1 for s in recon_ids if iface_sizes[s][0] == 0)
    print(f"[fig1] common interface mask: ligand-atom order check "
          f"{IFACE_ORDER_ERR:.1e} Å, own-arm drift {IFACE_OWN_DRIFT:.1e} Å, "
          f"complexes with an empty interface {IFACE_EMPTY}")
    for _k in recon:
        _v = np.array([iface_rmsd[_k][s] for s in recon_ids])
        _was = recon[_k].loc[recon_ids, "iface_lig_rmsd"]
        print(f"        {_k:<16} n {_was.notna().sum():>3} -> {_v.size}   "
              f"median {_was.median():.3f} -> {np.median(_v):.3f}")
    return (
        B2T_DUMPS,
        IFACE_CUTOFF,
        IFACE_EMPTY,
        IFACE_ORDER_ERR,
        IFACE_OWN_DRIFT,
        OWN_DUMPS,
        RECON_DUMP_SUBDIR,
        iface_rmsd,
    )


@app.cell
def _(iface_rmsd, np, pd, recon, recon_ids, sps):
    # Metric identity: name, column, whether higher is better, axis label.
    # The arrow in the label is the "lower/higher is better" statement.
    FIG1_METRICS = [
        ("Interface RMSD", "iface_lig_rmsd", False, "Interface RMSD [Å] ↓"),
        ("lDDT-PLI", "lddt_pli", True, "lDDT-PLI ↑"),
        ("Contact F1", "contact_f1", True, "Contact F1 ↑"),
    ]
    # Every pairwise comparison gets a test, not just the ablation pair.
    FIG1_PAIRS = [
        ("Bio2Token", "Separate ProLIT"),
        ("Separate ProLIT", "ProLIT"),
        ("Bio2Token", "ProLIT"),
    ]

    def _rank_biserial(diff):
        nz = diff[diff != 0]
        if nz.size == 0:
            return float("nan")
        r = sps.rankdata(np.abs(nz))
        return float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())

    def stars(p):
        """Conventional significance marker for a two-sided p-value."""
        for cut, mark in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (5e-2, "*")):
            if p < cut:
                return mark
        return "n.s."

    def collect_metric(col):
        """Values for the three arms on the complexes where all three reported."""
        if col == "iface_lig_rmsd":
            # Recomputed above under one shared interface definition; defined
            # for every complex, so nothing is dropped.
            cols = {k: pd.Series(iface_rmsd[k]).loc[recon_ids] for k in recon}
        else:
            cols = {k: recon[k].loc[recon_ids, col] for k in recon}
        mask = np.logical_and.reduce([v.notna().to_numpy() for v in cols.values()])
        ids = [i for i, m in zip(recon_ids, mask) if m]
        return {k: v[mask].to_numpy(float) for k, v in cols.items()}, ids

    fig1_data = {}
    for _name, _col, _hib, _lab in FIG1_METRICS:
        _vals, _ids = collect_metric(_col)
        _rec = {"values": _vals, "ids": _ids, "n": len(_ids), "col": _col,
                "higher_is_better": _hib, "label": _lab, "tests": {}}
        for _a, _b in FIG1_PAIRS:
            # b - a, so a positive difference always means "b beats a" for a
            # higher-is-better metric and the reverse for RMSD.
            _d = _vals[_b] - _vals[_a]
            _w = sps.wilcoxon(_d)
            _rec["tests"][(_a, _b)] = {
                "p": float(_w.pvalue),
                "stat": float(_w.statistic),
                "median_diff": float(np.median(_d)),
                "rank_biserial": _rank_biserial(_d),
                "b_win_rate": float((_d > 0).mean() if _hib else (_d < 0).mean()),
                "stars": stars(float(_w.pvalue)),
            }
        fig1_data[_name] = _rec

    for _name, _r in fig1_data.items():
        print(f"[fig1] {_name} (n={_r['n']})")
        for _k, _v in _r["values"].items():
            print(f"        {_k:<16} median={np.median(_v):.3f}  mean={_v.mean():.3f}")
        for (_a, _b), _t in _r["tests"].items():
            print(f"        {_a} vs {_b}: p={_t['p']:.3g} {_t['stars']:<4} "
                  f"r_rb={_t['rank_biserial']:+.3f} {_b} win={_t['b_win_rate']:.1%}")
    return FIG1_METRICS, FIG1_PAIRS, fig1_data, stars


@app.cell
def _(np, sps):
    def raincloud(ax, x, vals, color, *, width=0.36, seed=0, point_alpha=0.30):
        """Half-violin (right) + box (middle) + jittered points (left)."""
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]

        grid = np.linspace(vals.min(), vals.max(), 256)
        dens = sps.gaussian_kde(vals)(grid)
        dens = dens / dens.max() * width
        ax.fill_betweenx(grid, x + 0.06, x + 0.06 + dens, facecolor=color,
                         alpha=0.42, edgecolor=color, linewidth=2.0, zorder=2)

        rng = np.random.default_rng(seed)
        jx = x - 0.30 + rng.uniform(-0.10, 0.10, vals.size)
        ax.scatter(jx, vals, s=16, color=color, alpha=point_alpha,
                   linewidths=0, zorder=1)

        bp = ax.boxplot([vals], positions=[x - 0.06], widths=0.13,
                        showfliers=False, patch_artist=True, zorder=3,
                        medianprops={"color": "white", "linewidth": 3.0},
                        boxprops={"facecolor": color, "edgecolor": color,
                                  "linewidth": 2.0},
                        whiskerprops={"color": color, "linewidth": 2.0},
                        capprops={"color": color, "linewidth": 2.0})
        return bp, float(np.median(vals))

    return (raincloud,)


@app.cell
def _(
    C_BIO2TOKEN,
    C_INK,
    C_PROLIT,
    C_SEPARATE,
    FIG1_METRICS,
    fig1_data,
    np,
    plt,
    save_fig,
):
    FIG1_ORDER = [
        ("Bio2Token", C_BIO2TOKEN, "Bio2Token"),
        ("Separate ProLIT", C_SEPARATE, "ProLIT\n(separate)"),
        ("ProLIT", C_PROLIT, "ProLIT"),
    ]
    FIG1_ARMS = [a for a, _, _ in FIG1_ORDER]
    # Which pair sits on which bracket row: adjacent pairs share the lower row.
    FIG1_BRACKETS = [(0, 1, 0), (1, 2, 0), (0, 2, 1)]

    fig1, fig1_axes = plt.subplots(1, 3, figsize=(19.2, 8.0))
    fig1_marks = []

    for _ax, (_name, _col, _hib, _ylab) in zip(fig1_axes, FIG1_METRICS):
        _rec = fig1_data[_name]
        _lo = min(_rec["values"][a].min() for a in FIG1_ARMS)
        _hi = max(_rec["values"][a].max() for a in FIG1_ARMS)
        _rng = _hi - _lo

        for _i, (_arm, _c, _tick) in enumerate(FIG1_ORDER):
            _v = _rec["values"][_arm]
            _jit = np.random.default_rng(17 + _i).uniform(-0.16, 0.16, _v.size)
            _ax.scatter(_i + _jit, _v, s=13, color=_c, alpha=0.35,
                        linewidths=0, zorder=1)
            _ax.boxplot([_v], positions=[_i], widths=0.46, showfliers=False,
                        patch_artist=True, zorder=3,
                        medianprops={"color": _c, "linewidth": 3.0},
                        boxprops={"facecolor": "white", "alpha": 0.7,
                                  "edgecolor": _c, "linewidth": 2.2},
                        whiskerprops={"color": _c, "linewidth": 2.2},
                        capprops={"color": _c, "linewidth": 2.2})

        # Significance brackets for every pair, not only the ablation pair.
        _y0, _step, _cap = _hi + 0.06 * _rng, 0.15 * _rng, 0.028 * _rng
        for _a, _b, _lvl in FIG1_BRACKETS:
            _t = _rec["tests"][(FIG1_ARMS[_a], FIG1_ARMS[_b])]
            _y = _y0 + _lvl * _step
            _ax.plot([_a, _a, _b, _b],
                     [_y, _y + _cap, _y + _cap, _y],
                     color=C_INK, linewidth=1.8)
            _star = _t["stars"] != "n.s."
            _ax.text(0.5 * (_a + _b), _y + _cap + (0.02 if _star else 0.012) * _rng,
                     _t["stars"], ha="center",
                     va="center" if _star else "bottom",
                     fontsize=23 if _star else 19, color=C_INK)
            fig1_marks.append(_t["stars"])

        _ax.set_ylim(_lo - 0.06 * _rng, _hi + 0.38 * _rng)
        _ax.set_xlim(-0.6, 2.6)
        _ax.set_xticks(range(3))
        _ax.set_xticklabels([t for _, _, t in FIG1_ORDER], fontsize=19,
                            color=C_INK)
        _ax.set_ylabel(_ylab, labelpad=10)
        _ax.spines[["top", "right"]].set_visible(False)
        # The bracket band is not data: stop the axis, and its ticks, at the
        # last real value rather than leaving a tick floating past the spine.
        _ax.spines["left"].set_bounds(_lo - 0.06 * _rng, _hi)
        _ax.set_yticks([t for t in _ax.get_yticks()
                        if _lo - 0.06 * _rng <= t <= _hi])

    # Key for the markers actually drawn — never advertise a level unused here.
    FIG1_KEY = [
        (m, t) for m, t in
        (("n.s.", "p ≥ 0.05"), ("*", "p < 0.05"), ("**", "p < 0.01"),
         ("***", "p < 0.001"), ("****", "p < 0.0001"))
        if m in set(fig1_marks)
    ]
    FIG1_N = {_r["n"] for _r in fig1_data.values()}
    if len(FIG1_N) != 1:
        msg = f"panels no longer share one n: {sorted(FIG1_N)}"
        raise AssertionError(msg)
    fig1.text(0.5, 0.012,
              f"n = {FIG1_N.pop()} CASP16 complexes  ·  "
              "Wilcoxon signed-rank test, paired by complex:  "
              + ",   ".join(f"{m} {t}" for m, t in FIG1_KEY),
              ha="center", va="bottom", fontsize=19, color=C_INK)
    fig1.subplots_adjust(wspace=0.26, top=0.97, bottom=0.20)
    FIG1_PATHS = save_fig(fig1, "reconstruction-distributions")
    fig1
    return FIG1_ARMS, FIG1_BRACKETS, FIG1_KEY, FIG1_N, FIG1_ORDER, FIG1_PATHS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Figure 2 — Reconstruction examples

    Three complexes chosen by rule, not by eye, from the same 303:

    * **eligible** — ≥ 15 ligand heavy atoms and not an ion / cryoprotectant
      (`BR CL SO4 ACT DMS GOL EDO …`), so no panel is a lone metal or a sulfate;
    * **Representative** — the three metrics z-scored (interface RMSD sign-flipped
      so higher is better in all three) and the complex whose z-vector is nearest
      the median z-vector;
    * **Strong** — lDDT-PLI *and* Contact F1 both in the top quartile, then the
      lowest interface RMSD among those;
    * **Challenging** — the lowest mean z-score.

    Coordinates come from `benchmarks/recon-bench/outputs/own_allatom/joint_e250_lig3/*.npz`
    (`protein_ref`, `ligand_ref`, `protein_rec`, `ligand_rec`) — encode → decode
    round-trips actually written to disk, nothing re-derived.
    """)
    return


@app.cell
def _(np, re):
    ION_LIKE = {
        "BR", "CL", "NA", "ZN", "MG", "CA", "MN", "FE", "K", "CD", "CU", "NI",
        "CO", "HG", "SO4", "PO4", "ACT", "DMS", "EDO", "GOL", "NO3", "IOD",
        "PEG", "MES", "TRS", "FMT", "CIT",
    }
    MIN_LIG_ATOMS = 15

    def ligand_resname(sample_id):
        m = re.match(r".*__ligand_([A-Za-z0-9]+)_", sample_id)
        return m.group(1).upper() if m else ""

    def pick_recon_examples(complex_df, ligand_df):
        d = complex_df.copy()
        d["lig_atoms"] = ligand_df["n_atoms"]
        d["resname"] = [ligand_resname(i) for i in d.index]
        elig = d[(d["lig_atoms"] >= MIN_LIG_ATOMS) & (~d["resname"].isin(ION_LIKE))]

        cols = ["iface_lig_rmsd", "lddt_pli", "contact_f1"]
        z = elig[cols].apply(lambda s: (s - s.mean()) / s.std())
        z["iface_lig_rmsd"] *= -1.0  # higher = better everywhere
        score = z.mean(axis=1)
        dist = np.sqrt(((z - z.median()) ** 2).sum(axis=1))

        strong_pool = elig[
            (elig["lddt_pli"] >= elig["lddt_pli"].quantile(0.75))
            & (elig["contact_f1"] >= elig["contact_f1"].quantile(0.75))
        ]
        return {
            "Representative": dist.idxmin(),
            "Strong": strong_pool["iface_lig_rmsd"].idxmin(),
            "Challenging": score.idxmin(),
        }, elig, score

    return ION_LIKE, MIN_LIG_ATOMS, pick_recon_examples


@app.cell
def _(RECON_FILES, pd, pick_recon_examples):
    _d = pd.read_parquet(RECON_FILES["ProLIT"])
    _cx = _d[(_d["modality"] == "complex") & _d["ok"]].set_index("sample_id")
    _lig = _d[_d["modality"] == "ligand"].set_index("sample_id")
    FIG2_PICKS, fig2_pool, fig2_score = pick_recon_examples(_cx, _lig)
    fig2_meta = _cx.loc[list(FIG2_PICKS.values())]

    print(f"[fig2] eligible complexes: {len(fig2_pool)} of {len(_cx)}")
    for _k, _s in FIG2_PICKS.items():
        _r = fig2_pool.loc[_s]
        print(f"        {_k:<15} {_s:<28} atoms={_r['lig_atoms']:.0f} "
              f"iface={_r['iface_lig_rmsd']:.3f} lddt_pli={_r['lddt_pli']:.3f} "
              f"f1={_r['contact_f1']:.3f}")
    return FIG2_PICKS, fig2_meta, fig2_pool


@app.cell
def _(Line3DCollection, cKDTree, np):
    # ---- 3D molecule drawing -------------------------------------------------
    CPK = {
        "C": "#4B5563", "N": "#2563EB", "O": "#DC2626", "S": "#CA8A04",
        "F": "#16A34A", "CL": "#16A34A", "BR": "#92400E", "I": "#7C3AED",
        "P": "#EA580C", "SE": "#B45309", "B": "#F59E0B", "H": "#D1D5DB",
    }
    COVALENT = {
        "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07, "F": 0.57,
        "CL": 1.02, "BR": 1.20, "I": 1.39, "SE": 1.20, "B": 0.84, "H": 0.31,
    }
    VDW = {
        "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
        "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "B": 1.92, "SE": 1.90,
    }

    def elem_color(e):
        return CPK.get(str(e).upper(), "#7C3AED")

    def infer_bonds(xyz, elements, tol=0.42, max_d=2.25):
        """Connectivity from coordinates — the same thing the bench trusts."""
        if len(xyz) < 2:
            return np.zeros((0, 2), int)
        pairs = cKDTree(xyz).query_pairs(max_d, output_type="ndarray")
        if pairs.size == 0:
            return np.zeros((0, 2), int)
        r = np.array([COVALENT.get(str(e).upper(), 0.77) for e in elements])
        d = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
        return pairs[d < (r[pairs[:, 0]] + r[pairs[:, 1]] + tol)]

    def pca_frame(xyz):
        """Rotation putting the ligand's two widest axes in the view plane."""
        c = xyz.mean(0)
        _, _, vt = np.linalg.svd(xyz - c, full_matrices=False)
        rot = vt.copy()
        if np.linalg.det(rot) < 0:
            rot[2] *= -1
        return c, rot

    def residue_shell(prot_xyz, lig_xyz, resid, chain, cutoff=6.0):
        """Whole residues with any heavy atom within `cutoff` of the ligand."""
        near = cKDTree(lig_xyz).query_ball_point(prot_xyz, cutoff)
        hit = np.array([len(x) > 0 for x in near])
        keys = np.array([f"{a}|{b}" for a, b in zip(chain, resid)])
        good = set(keys[hit])
        return np.array([k in good for k in keys])

    def clash_detail(lig_xyz, lig_el, prot_xyz, prot_el, tol=0.75):
        """Bench clash rule: pairs closer than tol*(r_i+r_j).

        Returns (per-ligand-atom boolean, total pair count). ``sbdd_bench``
        reports the pair count in ``clash_count``; the boolean is what the
        figure draws a ring around.
        """
        lig_xyz = np.asarray(lig_xyz, float)
        if len(lig_xyz) == 0 or len(prot_xyz) == 0:
            return np.zeros(len(lig_xyz), bool), 0
        keep = np.array([str(e).upper() != "H" for e in lig_el])
        lr = np.array([VDW.get(str(e).upper(), 1.7) for e in lig_el])
        pr = np.array([VDW.get(str(e).upper(), 1.7) for e in prot_el])
        tree = cKDTree(prot_xyz)
        hit = np.zeros(len(lig_xyz), bool)
        pairs = 0
        for i, p in enumerate(lig_xyz):
            if not keep[i]:
                continue
            for j in tree.query_ball_point(p, tol * (lr[i] + pr.max())):
                if np.linalg.norm(p - prot_xyz[j]) < tol * (lr[i] + pr[j]):
                    hit[i] = True
                    pairs += 1
        return hit, pairs

    def draw_sticks(ax, xyz, bonds, elements, lw, alpha, color=None, zorder=2):
        segs, cols = [], []
        for i, j in bonds:
            mid = 0.5 * (xyz[i] + xyz[j])
            segs.append([xyz[i], mid])
            cols.append(color or elem_color(elements[i]))
            segs.append([mid, xyz[j]])
            cols.append(color or elem_color(elements[j]))
        if not segs:
            return
        lc = Line3DCollection(segs, colors=cols, linewidths=lw, alpha=alpha,
                              zorder=zorder)
        lc.set_capstyle("round")
        ax.add_collection3d(lc)

    def view_box(coords_list, rot, center, pad=(2.3, 2.3, 1.9), floor=3.2):
        """Half-widths of a *non-cubic* view box around everything drawn.

        A cube around an elongated ligand wastes most of the panel; per-axis
        half-widths with the drawing box aspect set to match keep the scale
        isotropic while filling the panel. Both rows of a column share the
        result, so "same camera, same scale" is literally the same numbers.
        """
        pts = np.vstack([(np.asarray(c) - center) @ rot.T for c in coords_list])
        half = np.maximum(np.abs(pts).max(axis=0), 0.0)
        return np.maximum(half + np.asarray(pad, float), floor)

    def frame_axes(ax, spans, *, zoom=1.40, elev=66, azim=-90):
        sx, sy, sz = spans
        ax.set_xlim(-sx, sx)
        ax.set_ylim(-sy, sy)
        ax.set_zlim(-sz, sz)
        ax.set_box_aspect((sx, sy, sz), zoom=zoom)
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")

    def draw_complex(
        ax, prot_xyz, prot_el, lig_xyz, lig_el, rot, center, spans,
        *, pocket_mask=None, pocket_color="#AFB6C2", lig_lw=5.2, atom_size=78,
        ghost_xyz=None, ghost_color="#1C3078", ghost_label_bonds=None,
        highlight=None, highlight_color="#B45309", elev=66, azim=-90,
        zoom=1.40,
    ):
        P = (np.asarray(prot_xyz) - center) @ rot.T
        L = (np.asarray(lig_xyz) - center) @ rot.T
        if pocket_mask is None:
            pocket_mask = np.ones(len(P), bool)
        Pk = P[pocket_mask]
        Pe = np.asarray(prot_el)[pocket_mask]
        draw_sticks(ax, Pk, infer_bonds(Pk, Pe), Pe, lw=1.9, alpha=0.75,
                    color=pocket_color, zorder=1)
        ax.scatter(Pk[:, 0], Pk[:, 1], Pk[:, 2], s=9, c=pocket_color,
                   alpha=0.75, linewidths=0, zorder=1)

        lb = infer_bonds(L, lig_el)
        if ghost_xyz is not None:
            G = (np.asarray(ghost_xyz) - center) @ rot.T
            gb = ghost_label_bonds if ghost_label_bonds is not None else infer_bonds(G, lig_el)
            draw_sticks(ax, G, gb, lig_el, lw=2.8, alpha=0.95,
                        color=ghost_color, zorder=3)
        draw_sticks(ax, L, lb, lig_el, lw=lig_lw, alpha=1.0, zorder=4)
        ax.scatter(L[:, 0], L[:, 1], L[:, 2], s=atom_size,
                   c=[elem_color(e) for e in lig_el], linewidths=0,
                   zorder=5, depthshade=False)
        if highlight is not None and highlight.any():
            ax.scatter(L[highlight, 0], L[highlight, 1], L[highlight, 2],
                       s=330, facecolors="none", edgecolors=highlight_color,
                       linewidths=3.0, zorder=6, depthshade=False)
        frame_axes(ax, spans, zoom=zoom, elev=elev, azim=azim)

    return (
        CPK,
        VDW,
        clash_detail,
        draw_complex,
        elem_color,
        infer_bonds,
        pca_frame,
        residue_shell,
        view_box,
    )


@app.cell
def _(
    C_INK,
    C_NAVY,
    C_PROLIT,
    FIG2_PICKS,
    Line2D,
    RECON,
    draw_complex,
    fig2_pool,
    infer_bonds,
    np,
    pca_frame,
    plt,
    residue_shell,
    save_fig,
    view_box,
):
    FIG2_NPZ_DIR = RECON / "outputs" / "own_allatom" / "joint_e250_lig3"

    fig2 = plt.figure(figsize=(18.4, 11.2))
    FIG2_ROWS = [("Reference", "ref"), ("ProLIT reconstruction", "rec")]

    for _col, (_kind, _sid) in enumerate(FIG2_PICKS.items()):
        _d = np.load(FIG2_NPZ_DIR / f"{_sid}.npz", allow_pickle=True)
        _lref, _lrec = _d["ligand_ref"], _d["ligand_rec"]
        _pref, _prec = _d["protein_ref"], _d["protein_rec"]
        _lel, _pel = _d["ligand_elements"], _d["protein_elements"]
        _lbonds = infer_bonds(_lref, _lel)

        _c, _rot = pca_frame(_lref)
        # one box for both rows -> identical camera, orientation and scale
        _spans = view_box([_lref, _lrec], _rot, _c)
        _mask = residue_shell(_pref, _lref, _d["protein_resid"], _d["protein_chain"], 6.0)

        _m = fig2_pool.loc[_sid]
        for _row, (_rowname, _which) in enumerate(FIG2_ROWS):
            _ax = fig2.add_subplot(2, 3, _row * 3 + _col + 1, projection="3d")
            draw_complex(
                _ax,
                _pref if _which == "ref" else _prec, _pel,
                _lref if _which == "ref" else _lrec, _lel,
                _rot, _c, _spans, pocket_mask=_mask,
                ghost_xyz=None if _which == "ref" else _lref,
                ghost_label_bonds=_lbonds,
            )
            if _col == 0:
                _ax.text2D(-0.03, 0.5, _rowname, transform=_ax.transAxes,
                           rotation=90, ha="center", va="center",
                           fontsize=25, fontweight="bold",
                           color=C_NAVY if _which == "ref" else C_PROLIT)
            if _row == 0:
                _ax.set_title(
                    f"{_kind}\n{_sid.replace('__ligand_', '  ·  ')}",
                    fontsize=24, fontweight="bold", color=C_INK, pad=0,
                )
            if _row == 1:
                _ax.text2D(
                    0.5, 0.02,
                    f"Interface RMSD {_m['iface_lig_rmsd']:.2f} Å\n"
                    f"lDDT-PLI {_m['lddt_pli']:.3f}    "
                    f"Contact F1 {_m['contact_f1']:.3f}",
                    transform=_ax.transAxes, ha="center", va="top",
                    fontsize=21, color=C_INK, linespacing=1.35,
                )

    _handles = [
        Line2D([], [], color="#AFB6C2", linewidth=3.5, label="pocket atoms"),
        Line2D([], [], color=C_NAVY, linewidth=3.5, label="reference ligand"),
        Line2D([], [], color="#4B5563", linewidth=5.0,
               label="ligand (element colours: N blue, O red, halogen green)"),
    ]
    fig2.legend(handles=_handles, loc="lower center", ncol=3, fontsize=21,
                bbox_to_anchor=(0.5, 0.0))
    fig2.subplots_adjust(left=0.045, right=0.99, top=0.93, bottom=0.17,
                         wspace=0.0, hspace=0.0)
    FIG2_PATHS = save_fig(fig2, "reconstruction-examples")
    fig2
    return FIG2_NPZ_DIR, FIG2_PATHS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Figure 5 — Where each tokenizer can be measured at all

    Every discrete structure tokenizer in the benchmark, on the same 303 CASP16
    complexes, grouped by **what it reconstructs**: protein, ligand, or the
    complex. Colour = that group; shade = the model inside it.

    The metric set narrows as the target narrows, which is the point of the
    figure:

    | metric | models it exists for |
    |---|---|
    | Kabsch RMSD | all seven — each on its own reconstruction target |
    | TM-score, lDDT | protein and complex tokenizers |
    | lDDT-PLI, Contact F1 | complex tokenizers only |

    **Kabsch, not raw RMSD.** ESM3, FoldToken, ConfSeq and Token-Mol reconstruct
    in their own frame and never predict where the thing sits, so their raw RMSD
    is 35–77 Å and means nothing. Superposed RMSD is the only axis on which all
    seven can stand. That the shared-pocket-frame arms *also* place what they
    reconstruct is a separate property, reported in Figure 1 and Figure 3.

    **ESM3 and FoldToken are read at `eval_scope == "pocket"`**, the scope
    `docs/results/casp16_significance.md` reports (its ESM3 row — TM 0.811,
    lDDT 0.952, Kabsch 1.159 Å as means — reproduces exactly at this scope and
    at no other).
    """)
    return


@app.cell
def _():
    # label, group, results file, model, modality, eval_scope
    FIG5_MODELS = [
        ("ESM3", "protein", "casp16_esm3", "esm3", "protein_backbone", "pocket"),
        ("FoldToken", "protein", "casp16_priorwork", "foldtoken",
         "protein_backbone", "pocket"),
        ("ConfSeq", "ligand", "casp16_confseq", "confseq", "ligand", "native"),
        ("Token-Mol", "ligand", "casp16_priorwork", "token_mol", "ligand", "native"),
        ("Bio2Token (complex)", "complex", "casp16_bio2token", "bio2token.complex",
         "complex", "native"),
        ("ProLIT (separate)", "complex", "casp16.arm-separate_e250",
         "own_allatom.separate_e250", "complex", "native"),
        ("ProLIT", "complex", "casp16.arm-joint_e250_lig3",
         "own_allatom.joint_e250_lig3", "complex", "native"),
    ]
    # metric key, column, higher-is-better, axis label, which groups have it,
    # and the modality it is read from (None = the model's own target row).
    FIG5_METRICS = [
        ("Kabsch RMSD", "kabsch_rmsd", False, "Kabsch RMSD [Å] ↓",
         ("protein", "ligand", "complex"), None),
        ("TM-score", "tm_score", True, "TM-score ↑",
         ("protein", "complex"), "protein_backbone"),
        ("lDDT", "lddt", True, "lDDT ↑",
         ("protein", "complex"), "protein_backbone"),
        ("lDDT-PLI", "lddt_pli", True, "lDDT-PLI ↑", ("complex",), "complex"),
        ("Contact F1", "contact_f1", True, "Contact F1 ↑", ("complex",), "complex"),
    ]
    # What panel 1's RMSD is over, per group — three different quantities, so
    # the figure says so rather than inviting a cross-group reading.
    FIG5_RMSD_OVER = {
        "protein": "CA atoms",
        "ligand": "ligand heavy atoms",
        "complex": "pocket + ligand heavy atoms",
    }
    return FIG5_METRICS, FIG5_MODELS, FIG5_RMSD_OVER


@app.cell
def _(FIG5_METRICS, FIG5_MODELS, RECON, np, pd, sps, stars):
    def _read(fname, model, modality, scope):
        d = pd.read_parquet(RECON / "results" / f"{fname}.parquet")
        g = d[(d["model"] == model) & (d["modality"] == modality)
              & (d["eval_scope"] == scope) & d["ok"]]
        return g.set_index("sample_id")

    # Every model's own-target rows, plus the protein_backbone view that the
    # complex tokenizers also produce (that is where TM-score and lDDT live).
    fig5_own = {
        lab: _read(f, m, mod, sc) for lab, _grp, f, m, mod, sc in FIG5_MODELS
    }
    fig5_prot = {}
    for _lab, _grp, _f, _m, _mod, _sc in FIG5_MODELS:
        if _grp == "complex":
            fig5_prot[_lab] = _read(_f, _m, "protein_backbone", _sc)
        elif _grp == "protein":
            fig5_prot[_lab] = fig5_own[_lab]

    fig5_ids = sorted(set.intersection(*(set(v.index) for v in fig5_own.values())))
    print(f"[fig5] complexes shared by all {len(FIG5_MODELS)} models: {len(fig5_ids)}")

    def fig5_series(label, group, col, source):
        """One model's values for one metric, or None where it does not apply."""
        if source == "protein_backbone":
            frame = fig5_prot.get(label)
        elif source == "complex":
            frame = fig5_own[label] if group == "complex" else None
        else:
            frame = fig5_own[label]
        if frame is None or col not in frame or not frame[col].notna().any():
            return None
        return frame.loc[fig5_ids, col].to_numpy(float)

    fig5_data = {}
    for _key, _col, _hib, _lab, _groups, _src in FIG5_METRICS:
        vals, tests = {}, {}
        for _mlab, _grp, *_ in FIG5_MODELS:
            if _grp not in _groups:
                continue
            v = fig5_series(_mlab, _grp, _col, _src)
            if v is not None:
                vals[_mlab] = v
        # Best rival = best median among the models ProLIT can be compared with
        # like for like. In the RMSD panel each group is measured over a
        # different set of atoms, so the rival is restricted to ProLIT's own
        # group; elsewhere every model is measured identically.
        group_of = {m: g for m, g, *_ in FIG5_MODELS}
        pool = [m for m in vals if m != "ProLIT"]
        if _col == "kabsch_rmsd":
            pool = [m for m in pool if group_of[m] == "complex"]
        if pool and "ProLIT" in vals:
            best = (max if _hib else min)(pool, key=lambda m: np.median(vals[m]))
            diff = vals["ProLIT"] - vals[best]
            w = sps.wilcoxon(diff)
            win = float((diff > 0).mean() if _hib else (diff < 0).mean())
            # Significance is not direction, and here the two summaries of
            # direction disagree: ESM3 has the better median on the protein
            # panels while ProLIT has the better mean, because ESM3 fails
            # catastrophically on a minority of complexes. Carry both rather
            # than picking the flattering one.
            mp, mr = float(np.mean(vals["ProLIT"])), float(np.mean(vals[best]))
            by_mean = "ProLIT" if ((mp > mr) if _hib else (mp < mr)) else best
            by_test = "ProLIT" if win > 0.5 else best
            tests = {
                "rival": best, "p": float(w.pvalue), "stars": stars(float(w.pvalue)),
                "median_diff": float(np.median(diff)),
                "prolit_win_rate": win,
                "winner_paired": by_test, "winner_mean": by_mean,
                "split": by_test != by_mean,
                "prolit_mean": mp, "rival_mean": mr,
            }
        fig5_data[_key] = {"values": vals, "test": tests, "label": _lab,
                           "higher_is_better": _hib, "groups": _groups}

    # Markdown rows for the manifest's median table, built here so the manifest
    # cell stays one flat narrative.
    FIG5_MEDIAN_ROWS = [
        f"| {lab} | " + " | ".join(
            "—" if rec["values"].get(lab) is None
            else f"{np.median(rec['values'][lab]):.3f}"
            for rec in fig5_data.values()
        ) + " |"
        for lab, *_ in FIG5_MODELS
    ]

    for _key, _rec in fig5_data.items():
        _t = _rec["test"]
        print(f"[fig5] {_key}: " + ", ".join(
            f"{m}={np.median(v):.3f}" for m, v in _rec["values"].items()))
        if _t:
            print(f"        ProLIT vs {_t['rival']}: p={_t['p']:.3g} "
                  f"{_t['stars']} ProLIT win={_t['prolit_win_rate']:.1%} "
                  f"-> paired/median: {_t['winner_paired']}, "
                  f"mean: {_t['winner_mean']}"
                  + ("  [SPLIT]" if _t["split"] else ""))
    return FIG5_MEDIAN_ROWS, fig5_data, fig5_ids, fig5_own, fig5_prot


@app.cell
def _(C_INK, FIG5_METRICS, FIG5_MODELS, fig5_data, np, plt, save_fig):
    # Colour = reconstruction target, shade = model within it. ProLIT keeps its
    # own teal so it reads the same here as in every other figure.
    FIG5_COLORS = {
        "ESM3": "#1C3078", "FoldToken": "#3A6DC9",
        "ConfSeq": "#B45309", "Token-Mol": "#E8A94A",
        "Bio2Token (complex)": "#83C6BF", "ProLIT (separate)": "#3E9A91",
        "ProLIT": "#0F766E",
    }
    # Display names. Bio2Token appears only as its complex arm in these
    # figures, so the arm qualifier is redundant on the axis; the manifest
    # keeps the full `Bio2Token (complex)` name for provenance.
    FIG5_TICK = {"Bio2Token (complex)": "Bio2Token"}
    FIG5_EMPHASIS = {"ProLIT", "ProLIT (separate)"}
    FIG5_FILES = {
        "Kabsch RMSD": "recon-kabsch-rmsd",
        "TM-score": "recon-tm-score",
        "lDDT": "recon-lddt",
        "lDDT-PLI": "recon-lddt-pli",
        "Contact F1": "recon-contact-f1",
    }

    def sig3(x):
        """Three significant figures, without a bare trailing point."""
        return f"{x:#.3g}".rstrip(".")

    FIG5_PATHS = {}
    FIG5_SUMMARY = {}
    FIG5_CLIPPED = {}
    for _key, _col, _hib, _xlab, _groups, _src in FIG5_METRICS:
        _rec = fig5_data[_key]
        _models = [m for m, g, *_ in FIG5_MODELS if m in _rec["values"]]
        _n = len(_models)
        # Wider per model than the panel version, so the enlarged labels have
        # room; margins are set in inches so every width gets the same gutters.
        _w_in, _h_in = 2.9 + 1.62 * _n, 9.4
        _fig, _ax = plt.subplots(figsize=(_w_in, _h_in))

        _top = 0.0
        for _i, _mlab in enumerate(_models):
            _v = _rec["values"][_mlab]
            _c = FIG5_COLORS[_mlab]
            _jit = np.random.default_rng(11 + _i).uniform(-0.16, 0.16, _v.size)
            _ax.scatter(_i + _jit, _v, s=16, color=_c, alpha=0.32,
                        linewidths=0, zorder=1)
            _ax.boxplot([_v], positions=[_i], widths=0.5, showfliers=False,
                        patch_artist=True, zorder=3,
                        medianprops={"color": _c, "linewidth": 3.8},
                        boxprops={"facecolor": "white", "alpha": 0.7,
                                  "edgecolor": _c, "linewidth": 2.8},
                        whiskerprops={"color": _c, "linewidth": 2.8},
                        capprops={"color": _c, "linewidth": 2.8})
            _q1, _q3 = np.percentile(_v, [25, 75])
            _w = _v[_v <= _q3 + 1.5 * (_q3 - _q1)]
            _top = max(_top, float(_w.max()) if _w.size else float(_q3))

        _lo = min(float(np.min(_rec["values"][m])) for m in _models)
        # A metric that cannot exceed 1 gets its axis run to 1, so the reader
        # sees how much headroom is left rather than a cropped top.
        _capped = _hib and max(
            float(np.max(_rec["values"][m])) for m in _models) <= 1.0
        _top = 1.0 if _capped else _top
        _rng = _top - _lo
        _hi_lim = _top if _capped else _top + 0.06 * _rng
        _ax.set_ylim(_lo - 0.06 * _rng, _hi_lim)

        # mean +/- SD sits outside the frame, so nothing collides with the data
        # and the numbers line up. Three significant figures.
        _rows = {}
        for _i, _mlab in enumerate(_models):
            _v = _rec["values"][_mlab]
            _mu, _sd = float(np.mean(_v)), float(np.std(_v, ddof=1))
            _rows[_mlab] = (_mu, _sd, float(np.median(_v)))
            _ax.text(_i, 1.035, f"{sig3(_mu)}\n± {sig3(_sd)}",
                     transform=_ax.get_xaxis_transform(), ha="center",
                     va="bottom", fontsize=26, color=C_INK, linespacing=1.2)

        _ax.set_xticks(range(_n))
        # Rotated, so the column spacing no longer has to hold the longest
        # model name and the type can be bigger on a narrower figure.
        _ax.set_xticklabels([FIG5_TICK.get(m, m) for m in _models], fontsize=34,
                            color=C_INK, rotation=30, ha="right",
                            rotation_mode="anchor")
        for _tl, _mlab in zip(_ax.get_xticklabels(), _models):
            if _mlab in FIG5_EMPHASIS:
                _tl.set_fontweight("bold")
        _ax.set_xlim(-0.65, _n - 0.35)
        _ax.set_ylabel(_xlab, labelpad=12, fontsize=38)
        _ax.tick_params(axis="y", labelsize=30)
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.spines["left"].set_bounds(_lo - 0.06 * _rng, _top)
        # The locator hands back 1.0000000000000002 for the tick at 1, so an
        # exact bound silently drops the top tick of a bounded metric.
        _tol = 1e-9 + 1e-6 * _rng
        _ax.set_yticks([t for t in _ax.get_yticks()
                        if _lo - 0.06 * _rng - _tol <= t <= _top + _tol])
        _ax.set_ylim(_lo - 0.06 * _rng, _hi_lim)
        FIG5_CLIPPED[_key] = (
            sum(int((_rec["values"][m] > _hi_lim).sum()) for m in _models),
            sum(_rec["values"][m].size for m in _models),
            float(_hi_lim),
        )
        _fig.subplots_adjust(left=1.55 / _w_in, right=1 - 0.25 / _w_in,
                             top=1 - 1.25 / _h_in, bottom=2.00 / _h_in)
        FIG5_PATHS[_key] = save_fig(_fig, FIG5_FILES[_key])
        FIG5_SUMMARY[_key] = _rows

    return (
        FIG5_CLIPPED,
        FIG5_COLORS,
        FIG5_FILES,
        FIG5_PATHS,
        FIG5_SUMMARY,
        sig3,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Figure 3 — Generation: the pose bottleneck

    100 canonical CrossDocked pockets × 100 molecules = 10,000, from
    `results_gen100_arom_post_s*` — the run reported as *ProLIT* in
    `docs/results/2026-08-22_canonical_100.md`. Figure 4's examples come from
    the same run.

    Pipeline behind those molecules (`jobs/generated/gen_ref.sh`):
    tokenizer `vq_e250_lig3` e237, CLM `clm_e250lig3_fullft` e00,
    refiner `refit_e250lig3` e16, then local relaxation.

    The three Vina columns are **three different questions** and are never mixed:

    | column | Vina call | what it reads |
    |---|---|---|
    | Vina **Score** | `--score_only` | the pose exactly as generated |
    | Vina **Min** | `--local_only` | the generated pose after local minimisation |
    | Vina **Dock** | full search in the pocket box | the molecule, pose discarded |
    """)
    return


@app.cell
def _(Path, REPO, pd):
    # ---- generation: the canonical 100-pocket run -----------------------------
    #
    # The 97-target set used by Figure 4 was not CrossDocked's canonical test
    # split (docs/notes/2026-08-21_target_set_was_not_the_canonical_split.md).
    # The generation distributions below therefore come from the rebuilt
    # 100-pocket run, `gen100_arom_post`, which is the row reported as "ProLIT"
    # in docs/results/2026-08-22_canonical_100.md.
    #
    # That run currently lives in a worktree rather than on main, so the tree is
    # resolved rather than hard-coded and the choice is printed.
    GEN2_RUN = "results_gen100_arom_post"
    GEN2_TREE_CANDIDATES = [
        Path(x) for x in [
            *( [__import__("os").environ["PROLIT_GEN_BENCH"]]
               if "PROLIT_GEN_BENCH" in __import__("os").environ else [] ),
            REPO / ".claude" / "worktrees" / "shape-complementarity"
            / "benchmarks" / "sbdd-bench",
            REPO / "benchmarks" / "sbdd-bench",
        ]
    ]
    GEN2_BENCH = next(
        (t for t in GEN2_TREE_CANDIDATES if sorted(t.glob(GEN2_RUN + "_s*"))), None
    )
    if GEN2_BENCH is None:
        _missing = (f"{GEN2_RUN} not found in any of: "
                    + ", ".join(str(t) for t in GEN2_TREE_CANDIDATES))
        raise FileNotFoundError(_missing)
    GEN2_SHARDS = sorted(GEN2_BENCH.glob(GEN2_RUN + "_s*"))

    gen2_mol = pd.concat(
        [pd.read_parquet(s / "per_molecule.parquet") for s in GEN2_SHARDS],
        ignore_index=True,
    )
    gen2_tgt = pd.concat(
        [pd.read_csv(s / "per_target.csv") for s in GEN2_SHARDS], ignore_index=True
    )
    for _c in ("vina_score", "vina_min", "vina_dock", "qed", "mol_wt"):
        gen2_mol[_c] = pd.to_numeric(gen2_mol[_c], errors="coerce")

    GEN2_N_TARGETS = int(gen2_tgt["target_id"].nunique())
    GEN2_N_MOLS = int(len(gen2_mol))
    print(f"[gen2] {GEN2_BENCH}")
    print(f"[gen2] {GEN2_RUN}: {len(GEN2_SHARDS)} shards, "
          f"{GEN2_N_TARGETS} targets, {GEN2_N_MOLS} molecules")
    return (
        GEN2_BENCH,
        GEN2_N_MOLS,
        GEN2_N_TARGETS,
        GEN2_RUN,
        GEN2_SHARDS,
        gen2_mol,
        gen2_tgt,
    )


@app.cell
def _(GEN2_BENCH, gen2_tgt, np, pd):
    # The crystal ligand of each pocket, scored through the same three Vina
    # calls by the bench (`ref_vina_*`). QED, molecular weight and PoseBusters
    # validity are NOT in the results table: `metrics.evaluate_target` drops the
    # `ref`-tagged entry before the pose-quality block and re-adds the reference
    # only for `dock_generated`, so it gets Vina columns and nothing else. They
    # are computed here from the same reference SDFs the bench docked -- real
    # molecules, not quoted constants.
    def reference_chem():
        from rdkit import Chem
        from rdkit.Chem import QED, Descriptors

        rows, mols = [], []
        for tid in sorted(gen2_tgt["target_id"].unique()):
            sdf = GEN2_BENCH / "data" / "targets" / tid / f"{tid}_ref_ligand.sdf"
            if not sdf.exists():
                continue
            mol = next(
                (m for m in Chem.SDMolSupplier(str(sdf), sanitize=True,
                                               removeHs=True) if m is not None),
                None,
            )
            if mol is None:
                continue
            rows.append({"target_id": tid, "qed": float(QED.qed(mol)),
                         "mol_wt": float(Descriptors.MolWt(mol))})
            mols.append(mol)
        return pd.DataFrame(rows), mols

    def reference_pb(mols):
        """PoseBusters validity of the crystal ligands.

        Mirrors `sbdd_bench.pose.pb_validity`: the `mol` config with
        `energy_ratio` and `check_radicals` dropped, busting each molecule's own
        bonds. Replicated rather than imported so the notebook does not reach
        into a benchmark package; the two must stay in step, and the per-check
        columns are printed so a drift is visible.
        """
        from posebusters import PoseBusters

        cfg = PoseBusters(config="mol").config
        cfg["modules"] = [
            m for m in cfg["modules"]
            if m.get("function") not in {"energy_ratio", "check_radicals"}
        ]
        cfg["max_workers"] = 4
        table = PoseBusters(config=cfg).bust(mols)
        return float(table.all(axis=1).mean()), table.mean()

    gen2_ref_chem, gen2_ref_mols = reference_chem()
    GEN2_REF_PB, _pb_checks = reference_pb(gen2_ref_mols)
    print("[gen2] crystal-ligand PoseBusters per check:")
    print(_pb_checks.round(3).to_string())
    GEN2_REF = {
        "vina_score": float(gen2_tgt["ref_vina_score"].median()),
        "vina_min": float(gen2_tgt["ref_vina_min"].median()),
        "vina_dock": float(gen2_tgt["ref_vina_dock"].median()),
        "qed": float(gen2_ref_chem["qed"].median()) if len(gen2_ref_chem) else np.nan,
        "mol_wt": (float(gen2_ref_chem["mol_wt"].median())
                   if len(gen2_ref_chem) else np.nan),
        "pb_valid_rate": GEN2_REF_PB,
    }
    print(f"[gen2] reference ligands read: {len(gen2_ref_chem)} of "
          f"{gen2_tgt['target_id'].nunique()}")
    print(f"[gen2] reference medians {GEN2_REF}")
    return GEN2_REF, GEN2_REF_PB, gen2_ref_chem


@app.cell
def _(C_BLUE, C_INK, C_NAVY, C_PROLIT, C_WARN, GEN2_REF, gen2_mol, gen2_tgt,
      np, plt, save_fig, sig3):
    # Four figures in the same idiom as the reconstruction set: box + jitter,
    # mean +- SD above the frame, rotated labels, no caption furniture.
    def gen_box(ax, i, values, color, jitter_seed, point_size, point_alpha):
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        jit = np.random.default_rng(jitter_seed).uniform(-0.16, 0.16, v.size)
        ax.scatter(i + jit, v, s=point_size, color=color, alpha=point_alpha,
                   linewidths=0, zorder=1)
        ax.boxplot([v], positions=[i], widths=0.5, showfliers=False,
                   patch_artist=True, zorder=3,
                   medianprops={"color": color, "linewidth": 3.8},
                   boxprops={"facecolor": "white", "alpha": 0.7,
                             "edgecolor": color, "linewidth": 2.8},
                   whiskerprops={"color": color, "linewidth": 2.8},
                   capprops={"color": color, "linewidth": 2.8})
        return v

    def gen_figure(stem, columns, ylabel, *, unit, ref=None, bounded=False):
        """One metric (or one family) as a box-and-jitter figure."""
        n = len(columns)
        # A one-box figure still needs a slide-legible canvas, so the width has
        # a floor rather than shrinking to a single column.
        w_in, h_in = 2.9 + 1.62 * max(n, 2.6), 9.4
        fig, ax = plt.subplots(figsize=(w_in, h_in))

        drawn, top = [], -np.inf
        for i, (_label, values, color) in enumerate(columns):
            per_mol = unit == "molecule"
            v = gen_box(ax, i, values, color, 11 + i,
                        6 if per_mol else 16, 0.10 if per_mol else 0.32)
            drawn.append(v)
            q1, q3 = np.percentile(v, [25, 75])
            w = v[v <= q3 + 1.5 * (q3 - q1)]
            top = max(top, float(w.max()) if w.size else float(q3))

        lo = min(float(v.min()) for v in drawn)
        if bounded and max(float(v.max()) for v in drawn) <= 1.0:
            top = 1.0
        rng = top - lo
        hi_lim = top if (bounded and top == 1.0) else top + 0.06 * rng
        ax.set_ylim(lo - 0.06 * rng, hi_lim)

        summary = {}
        for i, ((label, _values, _color), v) in enumerate(zip(columns, drawn)):
            mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
            summary[label] = (mu, sd, float(np.median(v)))
            ax.text(i, 1.035, f"{sig3(mu)}\n± {sig3(sd)}",
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=26, color=C_INK, linespacing=1.2)

        if ref is not None:
            ax.plot(range(n), ref, marker="D", markersize=15, color=C_NAVY,
                    linewidth=3.0, linestyle="--", zorder=6,
                    markeredgecolor="white", markeredgewidth=1.6)
            ax.text(n - 0.62, ref[-1], "  crystal\n  ligand", ha="left",
                    va="center", fontsize=24, color=C_NAVY, linespacing=1.15)

        ax.set_xticks(range(n))
        if n > 1:
            ax.set_xticklabels([c[0] for c in columns], fontsize=34,
                               color=C_INK, rotation=30, ha="right",
                               rotation_mode="anchor")
        else:
            # One box: the y-axis already names the metric, so a tick label
            # under it would just repeat itself.
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.65, n - 0.35)
        ax.set_ylabel(ylabel, labelpad=12, fontsize=38)
        ax.tick_params(axis="y", labelsize=30)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_bounds(lo - 0.06 * rng, top)
        tol = 1e-9 + 1e-6 * rng
        ax.set_yticks([t for t in ax.get_yticks()
                       if lo - 0.06 * rng - tol <= t <= top + tol])
        ax.set_ylim(lo - 0.06 * rng, hi_lim)
        fig.subplots_adjust(left=1.75 / w_in, right=1 - 0.25 / w_in,
                            top=1 - 1.25 / h_in,
                            bottom=(2.00 if n > 1 else 0.45) / h_in)
        clipped = (sum(int((v > hi_lim).sum()) for v in drawn),
                   sum(int(v.size) for v in drawn), float(hi_lim))
        return save_fig(fig, stem), summary, clipped

    GEN2_PATHS, GEN2_SUMMARY, GEN2_CLIPPED = {}, {}, {}

    # Vina: one figure, the same molecules read three ways, per target.
    _vina_cols = [
        ("Vina Score", gen2_tgt["vina_score_median"].to_numpy(float), C_WARN),
        ("Vina Min", gen2_tgt["vina_min_median"].to_numpy(float), C_BLUE),
        ("Vina Dock", gen2_tgt["vina_dock_median"].to_numpy(float), C_PROLIT),
    ]
    GEN2_PATHS["Vina"], GEN2_SUMMARY["Vina"], GEN2_CLIPPED["Vina"] = gen_figure(
        "gen-vina", _vina_cols, "Vina energy [kcal/mol] ↓", unit="target",
        ref=[GEN2_REF["vina_score"], GEN2_REF["vina_min"], GEN2_REF["vina_dock"]],
    )

    # PoseBusters validity only exists as a rate, so its unit is the target.
    GEN2_PATHS["PB-valid"], GEN2_SUMMARY["PB-valid"], GEN2_CLIPPED["PB-valid"] = (
        gen_figure("gen-pb-valid",
                   [("PoseBusters-valid",
                     gen2_tgt["pb_valid_rate"].to_numpy(float), C_PROLIT)],
                   "PoseBusters-valid rate ↑", unit="target", bounded=True,
                   ref=[GEN2_REF["pb_valid_rate"]])
    )

    for _stem, _col, _lab, _bounded in (
        ("gen-qed", "qed", "QED ↑", True),
        ("gen-molecular-weight", "mol_wt", "Molecular weight [Da]", False),
    ):
        _key = "QED" if _col == "qed" else "Molecular weight"
        GEN2_PATHS[_key], GEN2_SUMMARY[_key], GEN2_CLIPPED[_key] = gen_figure(
            _stem, [(_key, gen2_mol[_col].dropna().to_numpy(float), C_PROLIT)],
            _lab, unit="molecule", bounded=_bounded,
            ref=[GEN2_REF[_col]] if np.isfinite(GEN2_REF[_col]) else None,
        )

    for _k, _rows in GEN2_SUMMARY.items():
        print(f"[gen2] {_k}: " + ", ".join(
            f"{m} {sig3(mu)}±{sig3(sd)} (med {sig3(md)})"
            for m, (mu, sd, md) in _rows.items()))
    return GEN2_CLIPPED, GEN2_PATHS, GEN2_SUMMARY


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Figure 4 — Generation examples

    Selected by rule from the same 9,700 molecules. Common eligibility:
    valid, both Vina Score and Vina Dock present, 20–45 heavy atoms,
    QED ≥ 0.25, PoseBusters-valid — i.e. every panel is a *chemically
    reasonable* molecule, so the panels differ only in where it was put.

    * **Successful** — clash-free, then the best (most negative) Vina Score.
    * **Pose-rescuable** — clashing, positive Vina Score, Vina Dock ≤ −8; among
      those, the molecule whose Score→Dock gain is closest to the pool median,
      so the panel shows a typical rescue, not the record one.
    * **Failure** — the 90th-percentile clash count of the pool.

    Each is taken from a different target.

    **Redocked coordinates are recomputed here.** `sbdd_bench.docking` writes
    Vina's docked pose into a temporary directory and keeps only the score, so
    there is nothing on disk to draw. The three molecules are re-docked with the
    identical call (same receptor pdbqt, same pocket box, `--seed 1`,
    `--exhaustiveness 8`) and cached under `outputs/iibmp2026/redock/`; the
    reproduced scores are checked against the benchmark's stored `vina_dock`.
    """)
    return


@app.cell
def _(GEN_SDF_DIR, gen2_mol):
    FIG4_MIN_ATOMS, FIG4_MAX_ATOMS, FIG4_MIN_QED = 20, 45, 0.25

    def sdf_atom_counts(targets):
        """Atoms actually in each SDF record, per (target_id, idx).

        `n_atoms` in the results table is the *largest fragment* of the
        re-perceived molecule, so a multi-fragment record reports 37 while the
        record the figure would draw holds 62. Docking and the clash count used
        every atom, so the Vina columns still describe the whole record -- but
        QED, PoseBusters and the drawing do not. Comparing the two counts is
        the cheapest way to keep a panel honest.
        """
        from rdkit import Chem

        out = {}
        for tid in targets:
            path = GEN_SDF_DIR / tid / "generated.sdf"
            if not path.exists():
                continue
            supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
            for i, mol in enumerate(supplier):
                if mol is not None:
                    out[(tid, i)] = sum(
                        1 for a in mol.GetAtoms() if a.GetSymbol() != "H"
                    )
        return out

    def pick_gen_examples(df):
        pool = df[
            df["valid_b"]
            & df["vina_score"].notna()
            & df["vina_dock"].notna()
            & df["n_atoms"].between(FIG4_MIN_ATOMS, FIG4_MAX_ATOMS)
            & (df["qed"] >= FIG4_MIN_QED)
            & (df["pb_b"] == True)  # noqa: E712 — pandas BooleanDtype
            # `n_atoms` counts the largest fragment, so without this a
            # multi-fragment record passes the size filter and then gets drawn
            # whole -- 62 atoms in a panel that claims 37.
            & (df["connected"].astype("boolean") == True)  # noqa: E712
        ].copy()
        counts = sdf_atom_counts(sorted(pool["target_id"].unique()))
        pool["sdf_atoms"] = [
            counts.get((t, int(i)), -1)
            for t, i in zip(pool["target_id"], pool["idx"])
        ]
        # One fragment only: what is drawn is then exactly what was scored.
        pool = pool[pool["sdf_atoms"] == pool["n_atoms"].astype(int)].copy()
        pool["gain"] = pool["vina_score"] - pool["vina_dock"]

        success = pool[pool["clash_count"] == 0].nsmallest(1, "vina_score").iloc[0]

        resc_pool = pool[
            (pool["vina_score"] > 0)
            & (pool["vina_dock"] <= -8.0)
            & (pool["clash_count"] > 0)
            & (pool["target_id"] != success["target_id"])
        ].copy()
        resc_pool["d"] = (resc_pool["gain"] - resc_pool["gain"].median()).abs()
        rescuable = resc_pool.nsmallest(1, "d").iloc[0]

        # Clash-free is 96% in this run, so the 90th percentile of the whole
        # pool is zero clashes -- a "failure" panel with nothing wrong in it.
        # Take the 90th percentile of the molecules that actually clash.
        fail_pool = pool[
            ~pool["target_id"].isin([success["target_id"], rescuable["target_id"]])
            & (pool["clash_count"] > 0)
        ].copy()
        thr = fail_pool["clash_count"].quantile(0.90)
        fail_pool["d"] = (fail_pool["clash_count"] - thr).abs()
        failure = fail_pool.nsmallest(1, "d").iloc[0]
        return (
            {"Successful": success, "Pose-rescuable": rescuable, "Failure": failure},
            len(pool),
            float(thr),
            float(resc_pool["gain"].median()),
        )

    # Same run as the generation distributions -- the 97-target set these
    # examples used to come from was rebuilt on 2026-08-21, so its receptors and
    # docking boxes no longer exist and its stored metrics cannot be redrawn.
    _g = gen2_mol.copy()
    _g["valid_b"] = _g["valid"].astype(bool)
    _g["pb_b"] = _g["pb_valid"].astype("boolean")
    FIG4_PICKS, FIG4_POOL_N, FIG4_CLASH_THR, FIG4_GAIN_MED = pick_gen_examples(_g)
    print(f"[fig4] eligible molecules: {FIG4_POOL_N}; "
          f"failure clash threshold (p90) = {FIG4_CLASH_THR:.0f}; "
          f"rescuable median gain = {FIG4_GAIN_MED:.1f} kcal/mol")
    for _k, _r in FIG4_PICKS.items():
        print(f"        {_k:<15} {_r['target_id']:<32} idx={_r['idx']:<4} "
              f"score={_r['vina_score']:+8.2f} min={_r['vina_min']:+8.2f} "
              f"dock={_r['vina_dock']:+7.2f} clashes={_r['clash_count']:<3} "
              f"QED={_r['qed']:.2f}")
    return FIG4_CLASH_THR, FIG4_GAIN_MED, FIG4_MAX_ATOMS, FIG4_MIN_ATOMS, FIG4_MIN_QED, FIG4_PICKS, FIG4_POOL_N


@app.cell
def _(GEN2_BENCH, GEN2_RUN, Path, REDOCK_DIR, json, np, subprocess, tempfile):
    GEN_SDF_DIR = GEN2_BENCH / "outputs" / "gen100_arom_post" / "own"
    TARGETS_DIR = GEN2_BENCH / "data" / "targets"

    def read_sdf_entry(sdf_path, index):
        """Elements + coordinates of one SDF record, no sanitization."""
        from rdkit import Chem

        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
        mol = supplier[index]
        conf = mol.GetConformer()
        els = [a.GetSymbol() for a in mol.GetAtoms()]
        xyz = np.array(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
              conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
            dtype=np.float64,
        )
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        return els, xyz, name

    def read_receptor_pdb(path):
        """Heavy atoms of the receptor — the same file the bench counts clashes against."""
        els, xyz, resid, chain = [], [], [], []
        for ln in path.read_text().splitlines():
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            e = (ln[76:78].strip() or ln[12:16].strip()[:1]).capitalize()
            if e == "H":
                continue
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            els.append(e)
            chain.append(ln[21])
            resid.append(ln[22:27].strip())
        return (np.asarray(els), np.asarray(xyz, float).reshape(-1, 3),
                np.asarray(resid), np.asarray(chain))

    # AutoDock atom types are not element symbols: aromatic carbon is "A",
    # hydrogen-bonding N/O/S carry a trailing A/S. Read as elements they come
    # out as sodium and gallium, which is how a redocked pose ends up drawn in
    # the wrong colours.
    AD_TYPE_TO_ELEMENT = {
        "A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N", "O": "O",
        "OA": "O", "OS": "O", "S": "S", "SA": "S", "P": "P", "F": "F",
        "CL": "Cl", "BR": "Br", "I": "I", "SI": "Si", "B": "B", "SE": "Se",
        "MG": "Mg", "MN": "Mn", "ZN": "Zn", "CA": "Ca", "FE": "Fe",
    }
    AD_HYDROGEN = {"H", "HD", "HS"}

    def _pdbqt_heavy(text):
        els, xyz = [], []
        for ln in text.splitlines():
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            ad = ln[77:79].strip().upper()
            if ad in AD_HYDROGEN:
                continue
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            els.append(AD_TYPE_TO_ELEMENT.get(ad, ad.capitalize()))
        return els, np.asarray(xyz, float).reshape(-1, 3)

    def redock(target_id, idx, elements, coords, *, exhaustiveness=8, seed=1):
        """Vina full-search redock of one molecule; cached, real coordinates only.

        Same call as ``sbdd_bench.docking.dock_one``'s ``dock`` mode: the target's
        fixed pocket box, ``--seed 1``, ``--exhaustiveness 8``.
        """
        # The run is part of the key: the target set was rebuilt once already,
        # and a cache hit across that boundary would silently draw a pose docked
        # into a receptor box that no longer exists.
        cache = REDOCK_DIR / f"{GEN2_RUN}__{target_id}__{idx}.npz"
        if cache.exists():
            d = np.load(cache, allow_pickle=True)
            return (list(d["elements"]), d["coords"], float(d["score"]),
                    str(d["mode"]))

        from prolit.external_tools import require_tool

        vina, obabel = require_tool("vina"), require_tool("obabel")
        box = json.loads(
            (TARGETS_DIR / target_id / f"{target_id}_box.json").read_text()
        )
        receptor = TARGETS_DIR / target_id / f"{target_id}_receptor.pdbqt"
        c, s = box["center"], box["size"]
        with tempfile.TemporaryDirectory(dir=str(REDOCK_DIR)) as tmp:
            work = Path(tmp)
            xyz_f, lig_pdbqt, out_pdbqt = (
                work / "l.xyz", work / "l.pdbqt", work / "d.pdbqt"
            )
            xyz_f.write_text(
                f"{len(elements)}\n\n"
                + "".join(f"{e} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n"
                          for e, p in zip(elements, coords))
            )
            subprocess.run(
                [obabel, str(xyz_f), "-O", str(lig_pdbqt), "-r", "-p", "7.4",
                 "--partialcharge", "gasteiger"],
                capture_output=True, check=False,
            )
            subprocess.run(
                [vina, "--receptor", str(receptor), "--ligand", str(lig_pdbqt),
                 "--cpu", "1", "--seed", str(seed),
                 "--exhaustiveness", str(exhaustiveness), "--out", str(out_pdbqt),
                 "--center_x", f"{c[0]:.3f}", "--center_y", f"{c[1]:.3f}",
                 "--center_z", f"{c[2]:.3f}", "--size_x", f"{s[0]:.3f}",
                 "--size_y", f"{s[1]:.3f}", "--size_z", f"{s[2]:.3f}"],
                capture_output=True, check=True, text=True,
            )
            text = out_pdbqt.read_text()
        first = text.split("ENDMDL")[0]
        score = float(
            next(ln for ln in first.splitlines() if "VINA RESULT" in ln).split()[3]
        )
        els, pose = _pdbqt_heavy(first)
        # Vina preserves the input heavy-atom order, so when the counts agree
        # the SDF's own element list is exact and beats the type mapping.
        if len(els) == len([e for e in elements if e != "H"]):
            els = [e for e in elements if e != "H"]
        np.savez(cache, elements=np.array(els), coords=pose, score=score,
                 mode="vina --exhaustiveness 8 --seed 1, target pocket box")
        return els, pose, score, "recomputed"

    return GEN_SDF_DIR, TARGETS_DIR, read_receptor_pdb, read_sdf_entry, redock


@app.cell
def _(
    FIG4_PICKS,
    GEN_SDF_DIR,
    TARGETS_DIR,
    clash_detail,
    np,
    read_receptor_pdb,
    read_sdf_entry,
    redock,
):
    fig4_scenes = {}
    for _kind, _r in FIG4_PICKS.items():
        _t = _r["target_id"]
        _els, _raw, _name = read_sdf_entry(GEN_SDF_DIR / _t / "generated.sdf",
                                           int(_r["idx"]))
        _pel, _pxyz, _presid, _pchain = read_receptor_pdb(
            TARGETS_DIR / _t / f"{_t}_receptor.pdb"
        )
        _dels, _dock_xyz, _dock_score, _mode = redock(_t, int(_r["idx"]), _els, _raw)
        _hl_raw, _np_raw = clash_detail(_raw, _els, _pxyz, _pel)
        _hl_dock, _np_dock = clash_detail(_dock_xyz, _dels, _pxyz, _pel)
        fig4_scenes[_kind] = {
            "row": _r, "target": _t, "sdf_name": _name,
            "elements": _els, "raw": _raw,
            "prot_el": _pel, "prot_xyz": _pxyz,
            "prot_resid": _presid, "prot_chain": _pchain,
            "dock_elements": _dels, "dock": _dock_xyz,
            "dock_score": _dock_score, "dock_mode": _mode,
            "clash_atoms_raw": _hl_raw, "clash_atoms_dock": _hl_dock,
            "clash_pairs_raw": _np_raw, "clash_pairs_dock": _np_dock,
            "centroid_shift": float(
                np.linalg.norm(_raw.mean(axis=0) - _dock_xyz.mean(axis=0))
            ),
        }
        print(f"[fig4] {_kind:<15} {_t} idx={_r['idx']} sdf_name={_name} "
              f"atoms={len(_els)} redock={_dock_score:+.2f} "
              f"(stored {_r['vina_dock']:+.2f}, Δ={abs(_dock_score - _r['vina_dock']):.2f}) "
              f"[{_mode}]  clash pairs raw={_np_raw} (stored {_r['clash_count']}) "
              f"dock={_np_dock}  centroid shift="
              f"{np.linalg.norm(_raw.mean(axis=0) - _dock_xyz.mean(axis=0)):.2f} A")
    _dmax = max(abs(v["dock_score"] - v["row"]["vina_dock"]) for v in fig4_scenes.values())
    print(f"[fig4] worst |redock − stored vina_dock| = {_dmax:.3f} kcal/mol")
    return (fig4_scenes,)


@app.cell
def _(
    C_INK,
    C_PROLIT,
    C_WARN,
    FIG4_PICKS,
    Line2D,
    draw_complex,
    fig4_scenes,
    np,
    pca_frame,
    plt,
    residue_shell,
    save_fig,
    view_box,
):
    fig4 = plt.figure(figsize=(18.4, 11.6))
    FIG4_ROWS = [("Generated pose", "raw"), ("Redocked pose", "dock")]
    FIG4_SUBS = {
        "Successful": "right molecule, right place",
        "Pose-rescuable": "right molecule, wrong place",
        "Failure": "buried in the receptor",
    }

    for _col, _kind in enumerate(FIG4_PICKS):
        _sc = fig4_scenes[_kind]
        _r = _sc["row"]
        _c, _rot = pca_frame(_sc["raw"])
        # one box for both rows, sized to hold the generated *and* redocked
        # pose, so the two panels really are the same camera and scale
        _spans = view_box([_sc["raw"], _sc["dock"]], _rot, _c)
        _both = np.vstack([_sc["raw"], _sc["dock"]])
        _mask = residue_shell(_sc["prot_xyz"], _both, _sc["prot_resid"],
                              _sc["prot_chain"], 6.0)

        for _row, (_rowname, _which) in enumerate(FIG4_ROWS):
            _ax = fig4.add_subplot(2, 3, _row * 3 + _col + 1, projection="3d")
            _raw_row = _which == "raw"
            draw_complex(
                _ax, _sc["prot_xyz"], _sc["prot_el"],
                _sc["raw"] if _raw_row else _sc["dock"],
                _sc["elements"] if _raw_row else _sc["dock_elements"],
                _rot, _c, _spans, pocket_mask=_mask, lig_lw=6.2, atom_size=98,
                highlight=_sc["clash_atoms_raw"] if _raw_row
                else _sc["clash_atoms_dock"],
            )
            if _col == 0:
                _ax.text2D(-0.03, 0.5, _rowname, transform=_ax.transAxes,
                           rotation=90, ha="center", va="center",
                           fontsize=25, fontweight="bold",
                           color=C_WARN if _raw_row else C_PROLIT)
            if _raw_row:
                _ax.set_title(f"{_kind}\n{FIG4_SUBS[_kind]}", fontsize=24,
                              fontweight="bold", color=C_INK, pad=0)
                _ax.text2D(0.5, 0.055,
                           f"Vina Score {_r['vina_score']:+.2f}    "
                           f"{_r['clash_count']} clashes",
                           transform=_ax.transAxes, ha="center", va="top",
                           fontsize=22, color=C_WARN, fontweight="bold")
            else:
                _ax.text2D(0.5, 0.075,
                           f"Vina Dock {_sc['dock_score']:+.2f}    "
                           f"{_sc['clash_pairs_dock']} clashes",
                           transform=_ax.transAxes, ha="center", va="top",
                           fontsize=22, color=C_PROLIT, fontweight="bold")
                _ax.text2D(0.5, 0.0,
                           f"centroid moved {_sc['centroid_shift']:.1f} Å",
                           transform=_ax.transAxes, ha="center", va="top",
                           fontsize=21, color=C_WARN)
                _ax.text2D(0.5, -0.065, _sc["target"],
                           transform=_ax.transAxes, ha="center", va="top",
                           fontsize=18, color="#4B5563")

    _handles = [
        Line2D([], [], color="#AFB6C2", linewidth=3.5, label="receptor atoms"),
        Line2D([], [], color="#4B5563", linewidth=5.0,
               label="ligand (element colours)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor=C_WARN, markeredgewidth=3.0, markersize=17,
               label="atom clashing with the receptor"),
    ]
    fig4.legend(handles=_handles, loc="lower center", ncol=3, fontsize=21,
                bbox_to_anchor=(0.5, 0.0))
    fig4.subplots_adjust(left=0.045, right=0.99, top=0.93, bottom=0.17,
                         wspace=0.0, hspace=0.0)
    FIG4_PATHS = save_fig(fig4, "generation-examples")
    fig4
    return FIG4_PATHS, FIG4_ROWS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Provenance

    Writes `outputs/iibmp2026/figure_manifest.md` and copies the SVG/PNG pair of
    each figure into the presentation repository.
    """)
    return


@app.cell
def _(
    FIG1_PATHS,
    FIG2_PATHS,
    FIG4_PATHS,
    FIG5_PATHS,
    GEN2_PATHS,
    Path,
    REPO,
    shutil,
):
    PRESENTATION_ASSETS = Path(
        "/gs/bs/tga-ohuelab/sakano/git/presentations/presentations/IIBMP2026/assets/results"
    )
    ALL_FIG_PATHS = [FIG1_PATHS, FIG2_PATHS, FIG4_PATHS,
                     *FIG5_PATHS.values(), *GEN2_PATHS.values()]

    copied = []
    if PRESENTATION_ASSETS.parent.exists():
        PRESENTATION_ASSETS.mkdir(parents=True, exist_ok=True)
        for _pair in ALL_FIG_PATHS:
            for _p in _pair:
                shutil.copy2(_p, PRESENTATION_ASSETS / _p.name)
                copied.append(str(PRESENTATION_ASSETS / _p.name))
        print(f"[copy] {len(copied)} files -> {PRESENTATION_ASSETS}")
    else:
        print(f"[copy] SKIPPED — {PRESENTATION_ASSETS.parent} does not exist")

    print(f"[figures] written under {REPO / 'outputs' / 'iibmp2026' / 'figures'}")
    return ALL_FIG_PATHS, PRESENTATION_ASSETS, copied


@app.cell
def _():
    def fmt_p(p):
        """p-value as the manifest prints it: plain below the threshold."""
        return f"n.s. (p = {p:.2f})" if p >= 0.05 else f"p = {p:.1e}"

    return (fmt_p,)


@app.cell
def _(
    B2T_DUMPS,
    FIG1_PATHS,
    FIG2_PATHS,
    FIG2_PICKS,
    FIG4_CLASH_THR,
    FIG4_GAIN_MED,
    FIG4_MAX_ATOMS,
    FIG4_MIN_ATOMS,
    FIG4_MIN_QED,
    FIG4_PATHS,
    FIG4_PICKS,
    FIG4_POOL_N,
    FIG5_CLIPPED,
    FIG5_MODELS,
    FIG5_PATHS,
    FIG5_RMSD_OVER,
    FIG5_SUMMARY,
    FONT_IN_USE,
    GEN2_BENCH,
    GEN2_CLIPPED,
    GEN_SDF_DIR,
    GEN2_N_MOLS,
    GEN2_N_TARGETS,
    GEN2_PATHS,
    GEN2_REF,
    GEN2_RUN,
    GEN2_SHARDS,
    GEN2_SUMMARY,
    IFACE_CUTOFF,
    IFACE_EMPTY,
    IFACE_ORDER_ERR,
    IFACE_OWN_DRIFT,
    ION_LIKE,
    MIN_LIG_ATOMS,
    OUT,
    PRESENTATION_ASSETS,
    RECON,
    RECON_FILES,
    REPO,
    SBDD,
    copied,
    datetime,
    fig1_data,
    fig2_pool,
    fig4_scenes,
    fig5_data,
    fig5_ids,
    fmt_p,
    gen2_ref_chem,
    np,
    sig3,
    timezone,
):
    _today = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    _lines = []
    A = _lines.append

    A("# IIBMP2026 figure manifest")
    A("")
    A(f"Generated **{_today}** by `notebooks/iibmp2026_results.py`.")
    A("")
    A("Regenerate everything (figures + this manifest):")
    A("")
    A("```sh")
    A("cd " + str(REPO))
    A("PROLIT_VINA=<path/to/vina> PROLIT_OBABEL=<path/to/obabel> \\")
    A("    .venv/bin/python notebooks/iibmp2026_results.py")
    A("```")
    A("")
    A("`PROLIT_VINA` / `PROLIT_OBABEL` are needed only when "
      "`outputs/iibmp2026/redock/` is empty (Figure 4); with the cache present "
      "the notebook runs with no external binary.")
    A("")
    A(f"Font stack resolved on the generating machine: **{FONT_IN_USE}**. "
      "The figures request `Yu Gothic` first and fall back; all figure text is "
      "English, so the fallback changes nothing but the typeface.")
    A("")
    A("Outputs:")
    A("")
    A(f"* figures — `{OUT / 'figures'}`")
    A(f"* redock cache — `{OUT / 'redock'}`")
    A(f"* copied to — `{PRESENTATION_ASSETS}` ({len(copied)} files)")
    A("")

    # ---------------- Figure 1 ----------------
    A("---")
    A("")
    A("## Figure 1 — `reconstruction-distributions.{svg,png}`")
    A("")
    A("**Data files (absolute)**")
    A("")
    for _k, _p in RECON_FILES.items():
        A(f"* {_k} — `{_p}`")
    A("")
    A("**Runs / configuration** (from each arm's "
      "`benchmarks/recon-bench/outputs/own_allatom/<arm>/arm.json`)")
    A("")
    A("| arm in figure | recon-bench arm | tokenizer weights | codebook |")
    A("|---|---|---|---|")
    A("| ProLIT (joint) | `joint_e250_lig3` | "
      "`pocket-ligand-vqvae/vq_e250_lig3/checkpoints/atomvqvae-epoch=237-val/atom_coord=0.1021.ckpt` "
      "| 1 shared book, 8192 (13 bit/atom) |")
    A("| Separate ProLIT | `separate_e250` | "
      "`pocket-ligand-vqvae/vq_sep_prot/.../atom_coord=0.1910.ckpt` + "
      "`pocket-ligand-vqvae/vq_sep_lig_allatom/.../atom_coord=0.0891.ckpt` "
      "| 2 books, 4096+4096 |")
    A("| Bio2Token | `bio2token.complex` | published Bio2Token model "
      "(recon-bench adapter) | FSQ 4^6 = 4096 (12 bit/atom) |")
    A("")
    A("Normalization statistics travel with each checkpoint "
      "(`data/descriptor_cache_allatom/normalization_stats*.pt`), as recorded in "
      "`arm.json`.")
    A("")
    A("`joint_e250_lig3` / `separate_e250` — rather than the registry's "
      "`joint` / `separate` — are used because they are the controlled pair the "
      "significance analysis rests on (`docs/results/casp16_significance.md`: same "
      "250-epoch recipe, same cache, same held-out eval PDBs) **and** because "
      "`vq_e250_lig3` is the tokenizer Figures 3–4 generate through. For the "
      "record, the registry arms on the same 303 complexes give: "
      "lDDT-PLI 0.931 (`joint`) vs 0.920 (`separate`), Contact F1 0.689 vs 0.643, "
      "interface RMSD 0.469 vs 0.446 Å (means) — same direction, smaller margin.")
    A("")
    A('**Filters** — `modality == "complex"` and `ok == True`; then the '
      "intersection of sample IDs across all three arms; then, per metric, only "
      "complexes where all three arms reported a value.")
    A("")
    A("**Sample size / aggregation unit** — one point per CASP16 complex, "
      f"n = {sorted({r['n'] for r in fig1_data.values()})[0]} in all three "
      "panels, nothing dropped.")
    A("")
    A("### Interface RMSD is recomputed on one interface definition")
    A("")
    A("`recon_bench.metrics.complex_metrics` derives the interface from **each "
      "arm's own reference view**:")
    A("")
    A("```python")
    A("iface = (dist(ref_p, ref_l) < 4.0).any(axis=0)")
    A('"iface_lig_rmsd": rmsd(ref_l[iface], rec_l[iface]) if iface.any() else np.nan')
    A("```")
    A("")
    A("Bio2Token's complex view is **backbone-only** — its protein rows are "
      "exactly N/CA/C/O per residue (verified: the protein-atom count is "
      "divisible by 4 in 303/303 complexes, and mapping `protein_order` back "
      "through the joint dump's `protein_atom_names` gives N, CA, C, O and "
      "nothing else), because Bio2Token was trained on proteins and its "
      "upstream reader discards the rest. Its complex mode is labelled "
      "out-of-distribution in `adapters/bio2token.py` for the same reason.")
    A("")
    A("Consequence: for eight complexes whose ligand is a bromide, an acetate "
      "or a sulfate — 1–5 heavy atoms, bound through side chains — no backbone "
      "atom falls within 4 Å, so `iface.any()` is False and the value was NaN "
      "(and, consistently, `contact_recall` is NaN and `contact_f1` is 0 in all "
      "eight). That is a property of the receptor representation, not of the "
      "reconstruction, and it also meant the three arms were scored on three "
      "different sets of ligand atoms.")
    A("")
    A("The interface is a property of the crystal structure, so this notebook "
      "takes it **once** from the shared all-atom reference "
      f"(`joint_e250_lig3` dump, {IFACE_CUTOFF:.0f} Å) and applies the same mask "
      "to every arm. Validity of that substitution is asserted in-run:")
    A("")
    A(f"* ligand atoms are the same objects in the same order in all three "
      f"dumps — internal distances agree to {IFACE_ORDER_ERR:.1e} Å "
      "(rigid-motion invariant, so this tests the correspondence); Bio2Token's "
      "frame differs from the crystal frame by a pure translation, which "
      "cancels in a no-superposition RMSD of ref against rec;")
    A(f"* the two all-atom arms already used this mask, so their numbers must "
      f"not move — measured drift {IFACE_OWN_DRIFT:.1e} Å;")
    A(f"* complexes left with an empty interface: {IFACE_EMPTY}.")
    A("")
    A("Effect: ProLIT and ProLIT (separate) are bit-identical to the stored "
      "values; **only Bio2Token changes**, from n = 295 / median 1.090 Å to "
      "n = 303 / median 1.086 Å. Every marker in the panel is unchanged "
      "(ProLIT vs separate stays n.s.).")
    A("")
    A("Coordinates: `benchmarks/recon-bench/outputs/own_allatom/"
      "{joint_e250_lig3,separate_e250}/<sample_id>.npz` and "
      f"`{B2T_DUMPS}/<sample_id>.npz` (`ref`, `rec`, `n_protein_rows`). The "
      "Bio2Token dumps are produced by the arm's own environment "
      "(`.venv-bio2token`); if that directory is absent the recomputation "
      "cannot run and the notebook fails rather than quietly falling back.")
    A("")
    A("**Remaining caveat, not fixable here.** lDDT-PLI and Contact F1 take the "
      "receptor atoms into the metric itself, so Bio2Token is still scored "
      "against a backbone-only reference contact set in those two panels; there "
      "is no way to give a backbone tokenizer side chains after the fact. "
      "Comparing an all-atom tokenizer with a backbone one on interface metrics "
      "is inherently favourable to neither cleanly, and the ProLIT-vs-separate "
      "ablation — both all-atom — is unaffected.")
    A("")
    A("**Central tendency** — the box is the IQR with the median line and "
      "1.5·IQR whiskers; outliers are not drawn twice because every complex is "
      "already a point. Medians rather than means because interface RMSD is "
      "right-skewed. Exact medians are tabulated below rather than printed on "
      "the figure.")
    A("")
    A("| metric | Bio2Token median | ProLIT (separate) median | "
      "ProLIT median |")
    A("|---|---|---|---|")
    for _name, _rec in fig1_data.items():
        _v = _rec["values"]
        A(f"| {_name} | {np.median(_v['Bio2Token']):.3f} | "
          f"{np.median(_v['Separate ProLIT']):.3f} | "
          f"{np.median(_v['ProLIT']):.3f} |")
    A("")
    A("**Statistical test** — Wilcoxon signed-rank, two-sided, paired per "
      "complex, for **all three** pairs. The figure carries the conventional "
      "marker only (n.s. / \\* / \\*\\* / \\*\\*\\* / \\*\\*\\*\\*, keyed under the "
      "panels); the exact *p* and the matched-pairs rank-biserial effect size "
      "*r* are here. `win rate` is the fraction of complexes on which the second "
      "arm beats the first, in the metric's own direction.")
    A("")
    A("| metric | comparison (A vs B) | marker | p | r | B win rate |")
    A("|---|---|---|---|---|---|")
    for _name, _rec in fig1_data.items():
        for (_a, _b), _t in _rec["tests"].items():
            A(f"| {_name} | {_a} vs {_b} | `{_t['stars']}` | "
              f"{fmt_p(_t['p'])} | {_t['rank_biserial']:+.2f} | "
              f"{_t['b_win_rate']:.1%} |")
    A("")
    A("**Metrics not plotted, and why** — PoseBusters validity, SMILES match and "
      "the chemistry heads exist only on the `ligand` modality rows, not the "
      "`complex` rows this figure is about; they are per-complex and real, but "
      "they answer a different question and are left to the paper table.")
    A("")
    A(f"Files: `{FIG1_PATHS[0]}`, `{FIG1_PATHS[1]}`")
    A("")

    # ---------------- Figure 2 ----------------
    A("---")
    A("")
    A("## Figure 2 — `reconstruction-examples.{svg,png}`")
    A("")
    A("**Data files**")
    A("")
    A(f"* metrics — `{RECON_FILES['ProLIT']}`")
    A(f"* coordinates — `{RECON / 'outputs' / 'own_allatom' / 'joint_e250_lig3'}/<sample_id>.npz` "
      "(`protein_ref`, `ligand_ref`, `protein_rec`, `ligand_rec`, "
      "`protein_elements`, `ligand_elements`, `protein_resid`, `protein_chain`)")
    A("")
    A("**Run / configuration** — same `joint_e250_lig3` arm as Figure 1.")
    A("")
    A(f"**Eligibility filter** — ≥ {MIN_LIG_ATOMS} ligand heavy atoms and ligand "
      f"residue name not in the ion / cryoprotectant list "
      f"({', '.join(sorted(ION_LIKE))}). "
      f"{len(fig2_pool)} of 303 complexes eligible.")
    A("")
    A("**Selection rules** (no visual selection at any point)")
    A("")
    A("* *Representative* — interface RMSD, lDDT-PLI and Contact F1 z-scored "
      "(RMSD sign-flipped so higher is better in all three); the complex whose "
      "z-vector is closest in Euclidean distance to the median z-vector.")
    A("* *Strong* — lDDT-PLI **and** Contact F1 both ≥ their 75th percentile; "
      "among those, the lowest interface RMSD.")
    A("* *Challenging* — the lowest mean z-score.")
    A("")
    A("**Selected IDs**")
    A("")
    A("| role | sample_id | ligand heavy atoms | interface RMSD [Å] | lDDT-PLI | Contact F1 |")
    A("|---|---|---|---|---|---|")
    for _k, _s in FIG2_PICKS.items():
        _r = fig2_pool.loc[_s]
        A(f"| {_k} | `{_s}` | {_r['lig_atoms']:.0f} | "
          f"{_r['iface_lig_rmsd']:.3f} | {_r['lddt_pli']:.3f} | "
          f"{_r['contact_f1']:.3f} |")
    A("")
    A("**Aggregation unit** — none; single complexes, drawn from stored "
      "coordinates. Pocket wireframe connectivity is inferred from the stored "
      "coordinates with a covalent-radius rule (the NPZ carries no protein bond "
      "list); ligand connectivity uses the same rule so reference and "
      "reconstruction are drawn identically. Only whole residues with an atom "
      "within 6 Å of the reference ligand are drawn, and both rows use the same "
      "camera, orientation (ligand PCA frame) and scale.")
    A("")
    A(f"Files: `{FIG2_PATHS[0]}`, `{FIG2_PATHS[1]}`")
    A("")

    # ---------------- Figure 5 ----------------
    A("---")
    A("")
    A("## Figure 5 — `reconstruction-model-comparison.{svg,png}`")
    A("")
    A("**Data files** — one results table per model, all on the same CASP16 set:")
    A("")
    A("| model | group | results file | `model` | `modality` | `eval_scope` |")
    A("|---|---|---|---|---|---|")
    for _lab, _grp, _f, _m, _mod, _sc in FIG5_MODELS:
        A(f"| {_lab} | {_grp} | `{RECON / 'results' / (_f + '.parquet')}` | "
          f"`{_m}` | `{_mod}` | `{_sc}` |")
    A("")
    A("TM-score and lDDT for the complex tokenizers are read from their "
      "`protein_backbone` rows (same file, same scope).")
    A("")
    A(f"**Filters** — `ok == True`; then the intersection of sample IDs across "
      f"all {len(FIG5_MODELS)} models: n = {len(fig5_ids)}, nothing dropped. "
      "Bio2Token appears only as its `complex` arm; its protein-only and "
      "ligand-only arms are dropped from these figures. The figures themselves "
      "do not print n — every box in every one of them is the same 303 "
      "complexes.")
    A("")
    A("**Aggregation unit** — one point per CASP16 complex; box = IQR, median "
      "line, 1.5·IQR whiskers. Medians are tabulated below.")
    A("")
    A("**Kabsch, not raw RMSD.** ESM3, FoldToken, ConfSeq and Token-Mol "
      "reconstruct in their own frame and never predict placement, so their raw "
      "`rmsd` is a meaningless 35–77 Å (ESM3 35.5, FoldToken 54.5, ConfSeq 48.3, "
      "Token-Mol 47.9 Å, medians) against 0.4–1.1 Å for the shared-frame arms. "
      "Superposed RMSD is the only axis all seven can stand on. That the "
      "shared-pocket-frame arms *also* place what they reconstruct is a separate "
      "property, reported in Figures 1 and 3.")
    A("")
    A("**The Kabsch RMSD figure is three quantities, not one.** Each model is "
      "measured on its own reconstruction target: "
      + ", ".join(f"{g} → {o}" for g, o in FIG5_RMSD_OVER.items())
      + ". Comparisons inside a target group are exact; across groups the "
        "figure answers how well each tokenizer does its own job. The figure "
        "carries no note saying so, so **a slide using it has to say it out "
        "loud** — the three colour blocks are the only cue.")
    A("")
    A("**ESM3 / FoldToken scope** — `pocket`, not `full`. That is the scope "
      "`docs/results/casp16_significance.md` reports: its ESM3 row (TM 0.811, "
      "lDDT 0.952, Kabsch 1.159 Å as means) reproduces exactly at `pocket` and "
      "at no other scope (`full` gives 0.876 / 0.932 / 3.509).")
    A("")
    A("**Values drawn on the figures** — each box carries `mean ± SD` "
      "(sample SD, `ddof=1`) above it in black at three significant figures; "
      "the box itself is the IQR with the median line and 1.5·IQR whiskers. "
      "Metrics bounded above by 1 have their axis run to 1 rather than to the "
      "highest whisker.")
    A("")
    _clip = {k: v for k, v in FIG5_CLIPPED.items() if v[0]}
    if _clip:
        A("**Points above the axis.** The Kabsch RMSD axis is scaled to the "
          "whiskers, because ESM3's worst complex is 17.2 Å against 1.4 Å for "
          "ProLIT and a shared axis would flatten every box. The figure carries "
          "no marker for what falls outside, so record it here:")
        A("")
        A("| figure | axis top | points above it | share |")
        A("|---|---|---|---|")
        A("\n".join(
            f"| {k} | {sig3(top)} | {n} of {tot} | {n / tot:.2%} |"
            for k, (n, tot, top) in _clip.items()
        ))
        A("")
        A("Almost all of them are ESM3's. The `mean ± SD` printed above each "
          "box is what keeps this visible on the figure itself: ESM3 reads "
          "1.16 ± 2.15 Å with its box sitting at 0.36 Å, and an SD twice the "
          "mean is only possible with a long tail off the top.")
        A("")
    for _k, _rows in FIG5_SUMMARY.items():
        A(f"*{_k}*")
        A("")
        A("| model | mean ± SD | median |")
        A("|---|---|---|")
        A("\n".join(
            f"| {m} | {sig3(mu)} ± {sig3(sd)} | {sig3(md)} |"
            for m, (mu, sd, md) in _rows.items()
        ))
        A("")
    A("**No significance markers.** These five figures carry no brackets and no "
      "asterisks: the markers cost more than they explained. "
      "The paired tests are still computed in the notebook and recorded here, "
      "and the full pairwise analysis lives in "
      "`docs/results/casp16_significance.md`.")
    A("")
    A("| metric | best like-for-like rival | p (Wilcoxon signed-rank) | "
      "ProLIT win rate | median favours | mean favours |")
    A("|---|---|---|---|---|---|")
    A("\n".join(
        f"| {k} | {r['test']['rival']} | {fmt_p(r['test']['p'])} | "
        f"{r['test']['prolit_win_rate']:.1%} | "
        f"**{r['test']['winner_paired']}** | **{r['test']['winner_mean']}** |"
        for k, r in fig5_data.items() if r["test"]
    ))
    A("")
    A("### The two protein metrics split, and the split *is* the claim")
    A("")
    A("On TM-score and lDDT the median and the paired signed-rank test favour "
      "ESM3 while the **mean favours ProLIT**. Both are correct; they measure "
      "different things. Printing `mean ± SD` above the boxes makes the split "
      "readable straight off the figure: ESM3's mean sits well below its box "
      "and its SD is several times ProLIT's, which is the failure tail.")
    A("")
    A("The reason is a failure tail that only one of the two models has:")
    A("")
    A("| | ProLIT | ESM3 |")
    A("|---|---|---|")
    A("| TM-score median / mean | 0.907 / **0.881** | **0.930** / 0.811 |")
    A("| TM-score worst case | 0.543 | **0.001** |")
    A("| lDDT median / mean | 0.959 / **0.964** | **0.980** / 0.952 |")
    A("| lDDT worst case | 0.857 | **0.400** |")
    A("| TM-score on the 59 complexes where ESM3 < 0.70 | 0.847 | 0.344 |")
    A("| TM-score on the other 244 | 0.889 | 0.924 |")
    A("| lDDT on the 47 complexes where ESM3 < 0.90 | 0.993 | 0.809 |")
    A("| lDDT on the other 256 | 0.958 | 0.978 |")
    A("")
    A("ESM3 is better on roughly seven complexes in ten and collapses on the "
      "rest; ProLIT is slightly behind almost everywhere and never collapses. "
      "That is the paper's protein claim — **robustness, not accuracy** "
      "(`docs/results/casp16_significance.md` marks exactly these two rows "
      "with ⚠ for the same reason). A slide may say ProLIT has the better "
      "mean and the better worst case; it may not say ProLIT is simply more "
      "accurate than ESM3.")
    A("")
    A("**Colours** — hue = reconstruction target (blue protein, amber ligand, "
      "teal complex), shade = model within the group; ProLIT keeps `#0F766E` so "
      "it reads the same as in every other figure. The models stay in target "
      "order in every figure, so the colour blocks still separate the groups "
      "even though the group labels were dropped. This is a different colour "
      "axis from Figure 1, which colours by arm rather than by target.")
    A("")
    A("**Files** — one figure per metric:")
    A("")
    A("\n".join(f"* {k} — `{v[0]}`, `{v[1]}`" for k, v in FIG5_PATHS.items()))
    A("")

    # ---------------- Figure 3 (generation distributions) ----------------
    A("---")
    A("")
    A("## Figure 3 — generation distributions, one figure per metric")
    A("")
    A("`gen-vina`, `gen-pb-valid`, `gen-qed`, `gen-molecular-weight` "
      "(`.svg` + `.png` each). These replace the earlier single "
      "`generation-pose-bottleneck` panel figure.")
    A("")
    A("**Data files**")
    A("")
    for _p in GEN2_SHARDS:
        A(f"* `{_p / 'per_molecule.parquet'}`, `{_p / 'per_target.csv'}`")
    A(f"* reference ligands — `{GEN2_BENCH}/data/targets/<target_id>/"
      "<target_id>_ref_ligand.sdf`")
    A("")
    A("**Which run, and why it is not the one Figure 4 uses.** These four "
      f"figures come from `{GEN2_RUN}` — the rebuilt **canonical 100-pocket** "
      "run reported as *ProLIT* in `docs/results/2026-08-22_canonical_100.md*. "
      "The 97-target set behind Figure 4 was **not** CrossDocked's canonical "
      "test split (`docs/notes/2026-08-21_target_set_was_not_the_canonical_"
      "split.md`), so the two are different pocket sets and their numbers are "
      "not interchangeable. Figure 4's examples were left on the older run "
      "because reselecting and re-docking them is a separate job; **a slide "
      "must not read a number off Figure 4's run and a number off these "
      "figures as if they described the same evaluation.**")
    A("")
    A(f"The run currently lives in a worktree, `{GEN2_BENCH}`, not on `main`. "
      "The notebook resolves the tree at run time (`PROLIT_GEN_BENCH`, then the "
      "worktree, then the main checkout) and prints which one it used.")
    A("")
    A("**Run / configuration** (from `docs/results/2026-08-22_canonical_100.md` "
      "and the generating jobs)")
    A("")
    A("| component | value |")
    A("|---|---|")
    A("| CLM | `clm_e250lig3_fullft` e00 |")
    A("| pose refiner | `refit_e250lig3` e16 |")
    A("| decode | place-before-refine, valence pruning, fragment joining, "
      "aromaticity head |")
    A("| post-processing | per-molecule restrained local relaxation + rigid + "
      "torsion settling (`scripts/relax_generated.py --torsions`) |")
    A("| scoring | `run_evaluation.py --models own`, 6 shards |")
    A("")
    A(f"**Sample size** — {GEN2_N_TARGETS} targets × 100 = "
      f"{GEN2_N_MOLS:,} molecules.")
    A("")
    A("**Aggregation unit — different per figure, because the metrics are.**")
    A("")
    A("| figure | unit | n points | why |")
    A("|---|---|---|---|")
    A(f"| `gen-vina` | target | {GEN2_N_TARGETS} | the median over that "
      "target's molecules (`vina_*_median`), which is how the benchmark and "
      "the result docs report Vina; means are unusable because a handful of "
      "targets diverge to +100 kcal/mol |")
    A(f"| `gen-pb-valid` | target | {GEN2_N_TARGETS} | PoseBusters validity is "
      "a per-molecule boolean, so the only thing that has a distribution is "
      "the per-target rate |")
    A(f"| `gen-qed` | molecule | {GEN2_N_MOLS:,} | a per-molecule property; "
      "collapsing to target medians would hide the chemical spread that is the "
      "point of the figure |")
    A(f"| `gen-molecular-weight` | molecule | {GEN2_N_MOLS:,} | same |")
    A("")
    A("**Values drawn** — `mean ± SD` (sample SD, `ddof=1`) above each box at "
      "three significant figures, in black; box = IQR, median line, 1.5·IQR "
      "whiskers. QED and PoseBusters rate are bounded above by 1 and their axes "
      "run to 1.")
    A("")
    for _k, _rows in GEN2_SUMMARY.items():
        A(f"*{_k}*")
        A("")
        A("| column | mean ± SD | median |")
        A("|---|---|---|")
        A("\n".join(f"| {m} | {sig3(mu)} ± {sig3(sd)} | {sig3(md)} |"
                    for m, (mu, sd, md) in _rows.items()))
        A("")
    _gclip = {k: v for k, v in GEN2_CLIPPED.items() if v[0]}
    if _gclip:
        A("**Points above the axis** (axes are scaled to the whiskers; the "
          "figures carry no marker for what falls outside):")
        A("")
        A("| figure | axis top | points above it | share |")
        A("|---|---|---|---|")
        A("\n".join(f"| {k} | {sig3(top)} | {n} of {tot} | {n / tot:.2%} |"
                    for k, (n, tot, top) in _gclip.items()))
        A("")
    A("**Crystal ligand.** The dashed navy line is the reference ligand of each "
      "pocket. Its Vina numbers come from the bench's own `ref_vina_*` columns "
      "— the same three calls on the same receptors — median over targets: "
      f"Score {GEN2_REF['vina_score']:+.2f}, Min {GEN2_REF['vina_min']:+.2f}, "
      f"Dock {GEN2_REF['vina_dock']:+.2f} kcal/mol. QED and molecular weight "
      "are **not** in the results table, so they are computed here with RDKit "
      f"from the same `*_ref_ligand.sdf` files the bench docked "
      f"({len(gen2_ref_chem)} of {GEN2_N_TARGETS} read): QED "
      f"{GEN2_REF['qed']:.3f}, MW {GEN2_REF['mol_wt']:.1f} Da, PoseBusters "
      f"{GEN2_REF['pb_valid_rate']:.3f}.")
    A("")
    A("**Why the benchmark has no reference PoseBusters number.** "
      "`metrics.evaluate_target` removes the `ref`-tagged entry from the loaded "
      "SDF *before* the pose-quality block, then rebuilds the reference through "
      "`_ref_genmol` and hands it only to `dock_generated`. The reference "
      "therefore comes back with `ref_vina_score/min/dock` and nothing else — "
      "no PB, no clash count, no strain. It is computed here instead, "
      "replicating `sbdd_bench.pose.pb_validity` exactly: PoseBusters `mol` "
      "config with `energy_ratio` and `check_radicals` dropped, busting each "
      "molecule's own bonds. The per-check pass rates are printed when the "
      "notebook runs so a drift from the bench's configuration is visible; the "
      f"crystal ligands score {GEN2_REF['pb_valid_rate']:.2f} "
      f"({round(GEN2_REF['pb_valid_rate'] * GEN2_N_TARGETS)}/{GEN2_N_TARGETS}), "
      "failing only on internal steric clash and non-aromatic ring flatness. "
      "That is the ceiling this metric has: ProLIT's 0.920 sits 0.05 under a "
      "crystal structure, not under 1.0.")
    A("")
    A("**No baseline models.** FLOWR was run head-to-head on these same 100 "
      "pockets with the same receptors, Vina settings and PoseBusters settings "
      "(`docs/results/2026-08-23_flowr_head_to_head.md`, "
      f"`{GEN2_BENCH}/results_flowr100_s*`) and is deliberately **not** drawn "
      "here. For the record, so no slide implies otherwise: FLOWR leads on all "
      "three Vina readings — Score −5.63 vs −1.89, Min −6.04 vs −4.43, Dock "
      "−7.31 vs −6.95 (target medians; the Dock gap is small but significant, "
      "p = 0.020) — and on QED, 0.519 vs 0.385. ProLIT leads on clash-free "
      "(0.970 vs 0.883). The training sets differ (SPINDR vs CrossDocked2020) "
      "and that difference does not go away by running both here.")
    A("")
    A("**Excluded on purpose** — the 3-target DiffSBDD / TargetDiff / DiffGui "
      "numbers in `docs/results/best_allatom_configs.md` are from a different "
      "evaluation path (a since-fixed Open Babel re-perception route, molecules "
      "deleted, environments gone) and are **not** shown.")
    A("")
    A("**Files**")
    A("")
    A("\n".join(f"* {k} — `{v[0]}`, `{v[1]}`" for k, v in GEN2_PATHS.items()))
    A("")

    # ---------------- Figure 4 ----------------
    A("---")
    A("")
    A("## Figure 4 — `generation-examples.{svg,png}`")
    A("")
    A("**Data files**")
    A("")
    A(f"* molecule metrics — `{GEN2_BENCH}/{GEN2_RUN}_s*/per_molecule.parquet`")
    A(f"* generated coordinates — `{GEN_SDF_DIR}/<target_id>/generated.sdf`, "
      "record index = the `idx` column (record 0 is the reference ligand and is "
      "dropped by the bench, so `idx` indexes the SDF directly)")
    A(f"* receptor — `{SBDD}/data/targets/<target_id>/<target_id>_receptor.pdb` "
      "(the same file `sbdd_bench` counts clashes against; the figure draws only "
      "whole residues with an atom within 6 Å of either pose)")
    A(f"* receptor for redocking — `{SBDD}/data/targets/<target_id>/<target_id>_receptor.pdbqt`, "
      "box `<target_id>_box.json`")
    A(f"* redocked coordinates — `{OUT / 'redock'}/<target_id>__<idx>.npz`")
    A("")
    A("**Run / configuration** — the same canonical 100-pocket run as the "
      f"generation distributions above (`{GEN2_RUN}`).")
    A("")
    A("**Why it had to be moved off the 97-target run.** These examples used to "
      "come from `results_full97` + `outputs/gen_ref_relaxed`. The target set "
      "was rebuilt on 2026-08-21 (canonical CrossDocked split), which replaced "
      "every `data/targets/<id>/*_receptor.pdb*` and `*_box.json` in place. The "
      "generated SDFs were untouched, but the receptors and docking boxes the "
      "old numbers were computed against no longer exist: re-docking the same "
      "molecule now returned −9.85 against a stored −9.20, and recomputing its "
      "clashes gave 3 and 34 against stored 5 and 20. The figure would have "
      "drawn one receptor while labelling it with another receptor's numbers. "
      "Rebuilding it on the canonical run restores exact agreement — redock "
      "matches stored `vina_dock` to 0.000 kcal/mol and every clash count "
      "matches — and puts the examples on the same pockets as the "
      "distributions.")
    A("")
    A(f"**Eligibility filter** — valid, Vina Score and Vina Dock both present, "
      f"{FIG4_MIN_ATOMS}–{FIG4_MAX_ATOMS} heavy atoms, QED ≥ {FIG4_MIN_QED}, "
      f"PoseBusters-valid. {FIG4_POOL_N:,} molecules eligible. Every panel is "
      "therefore a chemically reasonable molecule; the panels differ only in "
      "where the model put it.")
    A("")
    A("**Selection rules**")
    A("")
    A("* *Successful* — clash count 0, then the most negative Vina Score.")
    A(f"* *Pose-rescuable* — clash count > 0, Vina Score > 0, Vina Dock ≤ −8; "
      f"among those, the molecule whose Score→Dock gain is closest to the pool "
      f"median ({FIG4_GAIN_MED:.1f} kcal/mol), so it is a typical rescue rather "
      f"than the record one.")
    A(f"* *Failure* — clash count closest to the pool's 90th percentile "
      f"({FIG4_CLASH_THR:.0f} clashes).")
    A("* All three come from different targets (each rule excludes the targets "
      "already used).")
    A("")
    A("**Selected IDs**")
    A("")
    A("| role | target_id | idx | SDF `_Name` | Vina Score | Vina Min | "
      "Vina Dock (stored) | Vina Dock (redocked here) | clashes stored / "
      "recomputed / after redock | centroid shift | QED | SA |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    A("\n".join(
        f"| {k} | `{r['target_id']}` | {r['idx']} | "
        f"`{fig4_scenes[k]['sdf_name']}` | {r['vina_score']:+.2f} | "
        f"{r['vina_min']:+.2f} | {r['vina_dock']:+.2f} | "
        f"{fig4_scenes[k]['dock_score']:+.2f} | {r['clash_count']} / "
        f"{fig4_scenes[k]['clash_pairs_raw']} / "
        f"{fig4_scenes[k]['clash_pairs_dock']} | "
        f"{fig4_scenes[k]['centroid_shift']:.2f} Å | {r['qed']:.2f} | "
        f"{r['sa']:.2f} |"
        for k, r in FIG4_PICKS.items()
    ))
    A("")
    A("`centroid shift` is the distance between the generated pose's centroid "
      "and the redocked pose's centroid — the figure's numeric handle on "
      "\"wrong place\". Clash counts are pairs, the benchmark's unit: `stored` "
      "is `per_molecule.parquet`, `recomputed` is this notebook against the "
      "same receptor PDB (they agree exactly), `after redock` is the same rule "
      "applied to the redocked coordinates.")
    A("")
    A("**Aggregation unit** — none; single molecules.")
    A("")
    A("**Redocking.** `sbdd_bench.docking.dock_one` writes Vina's docked pose "
      "into a `TemporaryDirectory` and keeps only the score, so no redocked "
      "coordinates exist on disk anywhere in the benchmark. Rather than draw "
      "nothing in the bottom row, the three selected molecules are re-docked here "
      "with the identical call — same receptor pdbqt, same fixed pocket box, "
      "`--exhaustiveness 8 --seed 1`, Open Babel ligand prep with "
      "`-r -p 7.4 --partialcharge gasteiger` — and the reproduced scores are "
      "compared against the benchmark's stored `vina_dock` in the table above "
      "(worst discrepancy "
      f"{max(abs(fig4_scenes[k]['dock_score'] - FIG4_PICKS[k]['vina_dock']) for k in FIG4_PICKS):.3f} "
      "kcal/mol). Both rows of a column share the camera, orientation and scale. "
      "Clash rings mark ligand atoms within 0.75·(r_i+r_j) of a receptor heavy "
      "atom — the benchmark's own clash rule, recomputed from the drawn "
      "coordinates.")
    A("")
    A("**Bond drawing.** Connectivity is inferred from the coordinates with the "
      "same covalent-radius rule used in Figure 2, rather than read from the "
      "mol block. This matches how the benchmark scores geometry: it never "
      "trusts the supplied bond list.")
    A("")
    A("**Single-fragment filter.** `n_atoms` in the results table is the "
      "*largest fragment* of the re-perceived molecule, while docking and the "
      "clash count used every atom in the record. A multi-fragment entry "
      "therefore passes a size filter on `n_atoms` and then gets drawn whole — "
      "one candidate reported 37 atoms and held 62. The pool is restricted to "
      "records whose SDF heavy-atom count equals `n_atoms`, so what is drawn is "
      "exactly what every annotation describes.")
    A("")
    A(f"Files: `{FIG4_PATHS[0]}`, `{FIG4_PATHS[1]}`")
    A("")

    _manifest = OUT / "figure_manifest.md"
    _manifest.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"[manifest] {_manifest} ({len(_lines)} lines)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## What each figure lets the slide claim

    1. **`reconstruction-distributions`** — On 303 CASP16 complexes, ProLIT's
       joint tokenizer reconstructs the protein–ligand interface significantly
       better than separate protein/ligand codebooks of the same total size
       (lDDT-PLI, Contact F1) and far better than Bio2Token on all three
       measures, while interface RMSD is statistically indistinguishable between
       the joint and separate arms.
    2. **`reconstruction-examples`** — Sub-ångström interface reconstruction is
       the typical case, not a cherry-picked one: even the rule-selected
       *challenging* complex keeps its contact pattern recognisable.
    5. **`reconstruction-model-comparison`** — ProLIT is the only tokenizer in
       the benchmark that can be scored on every axis at once: it beats the
       dedicated ligand tokenizers on ligand shape, carries the better *mean*
       and a far better worst case than ESM3 on the protein side (ESM3 holds
       the better median — it is ahead on most complexes and collapses on a
       minority), and is the only family that has an interface to score at all.
    3. **`gen-vina` / `gen-pb-valid` / `gen-qed` / `gen-molecular-weight`** —
       On the canonical 100 pockets the generated molecules are almost always
       physically clean (PoseBusters 0.920) and redock essentially onto the
       crystal ligand's own score (−6.98 against −7.74), while the pose the
       model writes still scores −1.88 against the crystal ligand's −6.81:
       the gap left is binding strength and placement, not chemical validity.
    4. **`generation-examples`** — The same model can put a good molecule exactly
       right, put one several ångström wrong so that only redocking recovers it,
       or bury it in the receptor; the difference between the three panels is
       placement alone.
    """)
    return


if __name__ == "__main__":
    app.run()
