# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
"""Multi-faceted geometry diagnostics — localize the distorted-shape problem.

Loads the dump from ``scripts/diagnose_geometry.py`` and compares three arms on
the same held-out test pockets:

- **GT**       (blue)  — real ligand, true coordinates (ceiling).
- **VQ-recon** (green) — real ligand -> VQ-VAE encode -> decode (decoder only).
- **LM-gen**   (red)   — pocket-conditioned LM samples codes -> decode (+ LM).

The central question each cell helps answer: is the bad geometry (clashes,
fragmentation) caused by the **VQ-VAE decoder/representation** (then VQ-recon
already looks bad) or by the **LM** sampling out-of-distribution codes (then
VQ-recon looks clean but LM-gen is bad)?

Regenerate the data with::

    uv run python scripts/diagnose_geometry.py \
        --lm-ckpt pocket-ligand-lm/<run>/checkpoints/<best>.ckpt --num-pockets 60
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

    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.titlesize": 21,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 15,
            "figure.dpi": 110,
        }
    )
    # arm -> (color, label)
    ARMS = {
        "gt": ("#3a7ca5", "GT (true coords)"),
        "recon": ("#66a182", "VQ-recon (decoder only)"),
        "gen": ("#d1495b", "LM-gen"),
    }
    return ARMS, Path, mo, np, os, plt


@app.cell
def _(Path, np, os):
    # Repository root, found from this file so the notebook runs from any
    # checkout; PROLIT_ROOT overrides it.
    _repo = Path(
        os.environ.get("PROLIT_ROOT") or Path(__file__).resolve().parent.parent
    )
    _cands = [
        Path.cwd() / "outputs/diagnostics/diag_data.npz",
        Path.cwd().parent / "outputs/diagnostics/diag_data.npz",
        _repo / "outputs/diagnostics/diag_data.npz",
    ]
    DATA_PATH = next(p for p in _cands if p.exists())
    D = np.load(DATA_PATH, allow_pickle=True)
    return D, DATA_PATH


@app.cell
def _(D, np):
    # Accessors over arm-prefixed arrays.
    def arr(arm, key):
        return D[f"{arm}_{key}"]

    def finite(a):
        a = np.asarray(a, float)
        return a[np.isfinite(a)]

    def clash_frac(arm, thr=1.2):
        a = finite(arr(arm, "min_pair_dist"))
        return 100 * (a < thr).mean() if len(a) else float("nan")

    methods = [str(x) for x in D["methods"]]
    return arr, clash_frac, finite, methods


@app.cell
def _(D, DATA_PATH, clash_frac, mo, np):
    _kab = np.asarray(D["recon_rmsd_kabsch"], float)
    _pa = np.asarray(D["recon_rmsd_peratom"], float)
    _real_u = 100 * (D["real_code_hist"] > 0).mean()
    _gen_u = 100 * (D["gen_code_hist"] > 0).mean()
    mo.md(
        f"""
        # 立体構造の課題・多面分析（原因の切り分け）

        **source**: `{DATA_PATH}`  ·  **ckpt**: `{D["lm_ckpt"]}`  ·  pockets: **{int(D["num_pockets"])}**
        ·  mols: GT **{len(D["gt_n_atoms"])}** / recon **{len(D["recon_n_atoms"])}** / gen **{len(D["gen_n_atoms"])}**

        | 指標 | GT | VQ-recon | LM-gen |
        |---|---:|---:|---:|
        | clash<1.2Å の割合 ↓ | {clash_frac("gt"):.0f}% | {clash_frac("recon"):.0f}% | {clash_frac("gen"):.0f}% |
        | geometric-valid ↑ | {100 * D["gt_v_geom_ok"].mean():.0f}% | {100 * D["recon_v_geom_ok"].mean():.0f}% | {100 * D["gen_v_geom_ok"].mean():.0f}% |
        | single-fragment ↑ | {100 * (D["gt_n_components"] == 1).mean():.0f}% | {100 * (D["recon_n_components"] == 1).mean():.0f}% | {100 * (D["gen_n_components"] == 1).mean():.0f}% |

        **VQ-VAE 再構成 RMSD**: Kabsch(内部形状) **{np.nanmean(_kab):.2f} Å** / per-atom(絶対配置) **{np.nanmean(_pa):.2f} Å**
        ·  **codebook 使用率**: real **{_real_u:.0f}%** / gen **{_gen_u:.0f}%**

        **読み方**: VQ-recon が GT 近く・LM-gen だけ悪ければ → 原因は**LM(OODコード)**。
        VQ-recon も悪ければ → 原因は**デコーダ/表現**。
        """
    )
    return


@app.cell
def _(ARMS, D, methods, plt):
    # 1. Three-way validity table — the localizer.
    _names = {
        "geom_ok": "Geometric(conn+noclash+bond)", "no_clash": "No clash(>=1.0A)",
        "connected": "Connected(1 frag)", "rdkit_chargesearch": "RDKit(charge search)",
        "rdkit_relaxed": "RDKit(UFF-relaxed)", "rdkit_charge0": "RDKit(charge0 strict)",
    }
    _arms = ["gt", "recon", "gen"]
    _rows = [
        [_names.get(_m, _m), *[f"{100 * D[f'{_a}_v_{_m}'].mean():.0f}%" for _a in _arms]]
        for _m in methods
    ]
    _fig, _ax = plt.subplots(figsize=(11, 5.5))
    _ax.axis("off")
    _ax.set_title("Validity by method: GT vs VQ-recon vs LM-gen", pad=22)
    _tab = _ax.table(
        cellText=_rows, colLabels=["method", *[ARMS[a][1].split(" ")[0] for a in _arms]],
        cellLoc="center", colColours=["#eeeeee", *[ARMS[a][0] for a in _arms]], loc="center",
    )
    _tab.auto_set_font_size(False)
    _tab.set_fontsize(13)
    _tab.scale(1, 2.3)
    for (_r, _c), _cell in _tab.get_celld().items():
        _cell.set_width(0.42 if _c == 0 else 0.19)
        if _r == 0:
            _cell.set_text_props(color="white", weight="bold")
        if _c == 0:
            _cell.set_text_props(ha="left")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(ARMS, arr, finite, np, plt):
    # 2. Min interatomic distance (clash) — does the DECODER itself clash?
    _fig, _ax = plt.subplots(figsize=(11, 6))
    _bins = np.linspace(0.5, 3.0, 60)
    for _arm in ("gt", "recon", "gen"):
        _c, _lbl = ARMS[_arm]
        _style = {"alpha": 0.35} if _arm == "gt" else {"histtype": "step", "lw": 2.6}
        _ax.hist(finite(arr(_arm, "min_pair_dist")), bins=_bins, density=True, color=_c, label=_lbl, **_style)
    _ax.axvline(1.2, ls="--", lw=2, color="black", label="clash < 1.2 Å")
    _ax.set_title("Min interatomic distance (clash localizer)")
    _ax.set_xlabel("nearest-atom distance (Å)")
    _ax.set_ylabel("density")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(D, np, plt):
    # 3. VQ-VAE reconstruction RMSD: per-atom (placement) vs Kabsch (internal shape).
    _pa = np.asarray(D["recon_rmsd_peratom"], float)
    _kab = np.asarray(D["recon_rmsd_kabsch"], float)
    _fig, _ax = plt.subplots(figsize=(11, 6))
    _bins = np.linspace(0, max(2.0, float(np.nanmax(_pa)) if len(_pa) else 2.0), 50)
    _ax.hist(_pa, bins=_bins, density=True, alpha=0.5, color="#b07aa1", label=f"per-atom (abs placement) μ={np.nanmean(_pa):.2f}Å")
    _ax.hist(_kab, bins=_bins, density=True, histtype="step", lw=2.6, color="#2a6f4e", label=f"Kabsch (internal shape) μ={np.nanmean(_kab):.2f}Å")
    _ax.set_title("VQ-VAE reconstruction RMSD (real ligand → encode → decode)")
    _ax.set_xlabel("RMSD (Å)")
    _ax.set_ylabel("density")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(ARMS, arr, np, plt):
    # 4. Bond-length distribution: GT vs recon vs gen.
    _fig, _ax = plt.subplots(figsize=(11, 6))
    _bins = np.linspace(0.8, 2.2, 70)
    for _arm in ("gt", "recon", "gen"):
        _c, _lbl = ARMS[_arm]
        _style = {"alpha": 0.35} if _arm == "gt" else {"histtype": "step", "lw": 2.6}
        _ax.hist(arr(_arm, "bond_lengths"), bins=_bins, density=True, color=_c, label=_lbl, **_style)
    for _x in (1.34, 1.54):
        _ax.axvline(_x, ls="--", lw=1.0, color="gray")
    _ax.set_title("Bond-length distribution")
    _ax.set_xlabel("inferred bond length (Å)")
    _ax.set_ylabel("density")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(ARMS, D, np, plt):
    # 5. Connectivity: single-fragment fraction + #components distribution.
    _arms = ["gt", "recon", "gen"]
    _single = [100 * (np.asarray(D[f"{a}_n_components"]) == 1).mean() for a in _arms]
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    _bars = _a1.bar([ARMS[a][1].split(" ")[0] for a in _arms], _single, color=[ARMS[a][0] for a in _arms], width=0.6)
    for _b, _v in zip(_bars, _single, strict=True):
        _a1.text(_b.get_x() + _b.get_width() / 2, _v + 1, f"{_v:.0f}%", ha="center", fontsize=15)
    _a1.set_ylim(0, 109)
    _a1.set_title("Single connected molecule")
    _a1.set_ylabel("fraction (%)")
    _hi = int(max(np.asarray(D[f"{a}_n_components"]).max() for a in _arms)) + 1
    _bins = np.arange(0.5, _hi + 1.5, 1)
    for _arm in _arms:
        _c, _lbl = ARMS[_arm]
        _style = {"alpha": 0.35} if _arm == "gt" else {"histtype": "step", "lw": 2.6}
        _a2.hist(np.asarray(D[f"{_arm}_n_components"]), bins=_bins, density=True, color=_c, label=_lbl, **_style)
    _a2.set_title("#connected components")
    _a2.set_xlabel("components per molecule")
    _a2.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(D, np, plt):
    # 6. Codebook usage: do LM-sampled codes match the real-code distribution?
    _real = np.asarray(D["real_code_hist"], float)
    _gen = np.asarray(D["gen_code_hist"], float)
    _rp = _real / max(_real.sum(), 1)
    _gp = _gen / max(_gen.sum(), 1)
    _ood_mass = 100 * _gp[_real == 0].sum()  # gen probability mass on codes never used by real ligands
    _both = ((_real > 0) & (_gen > 0)).sum()
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    # left: sorted code frequencies (rank plot)
    _a1.plot(np.sort(_rp)[::-1], color="#66a182", lw=2.2, label="real (VQ-recon)")
    _a1.plot(np.sort(_gp)[::-1], color="#d1495b", lw=2.2, label="LM-gen")
    _a1.set_yscale("log")
    _a1.set_title("Code-frequency rank plot")
    _a1.set_xlabel("code rank")
    _a1.set_ylabel("probability (log)")
    _a1.legend()
    # right: per-code real vs gen probability scatter (OOD = points on the gen axis where real=0)
    _a2.scatter(_rp + 1e-6, _gp + 1e-6, s=8, alpha=0.4, color="#444")
    _lim = [1e-6, max(_rp.max(), _gp.max()) * 1.3 + 1e-6]
    _a2.plot(_lim, _lim, ls="--", color="gray", lw=1)
    _a2.set_xscale("log")
    _a2.set_yscale("log")
    _a2.set_xlim(_lim)
    _a2.set_ylim(_lim)
    _a2.set_title(f"per-code real vs gen\n(OOD gen mass on real-unseen codes: {_ood_mass:.0f}%)")
    _a2.set_xlabel("real prob")
    _a2.set_ylabel("gen prob")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(ARMS, arr, np, plt):
    # 7. Heavy-atom count + bonds-per-atom (size & connectivity sanity).
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    _hi = int(max(arr(a, "n_atoms").max() for a in ("gt", "recon", "gen"))) + 2
    _b1 = np.arange(0, _hi, 2)
    _b2 = np.linspace(0, 4, 45)
    for _arm in ("gt", "recon", "gen"):
        _c, _lbl = ARMS[_arm]
        _style = {"alpha": 0.35} if _arm == "gt" else {"histtype": "step", "lw": 2.4}
        _a1.hist(arr(_arm, "n_atoms"), bins=_b1, density=True, color=_c, label=_lbl, **_style)
        _a2.hist(arr(_arm, "bonds_per_atom"), bins=_b2, density=True, color=_c, label=_lbl, **_style)
    _a1.set_title("Heavy-atom count")
    _a1.set_xlabel("# heavy atoms")
    _a1.legend()
    _a2.set_title("Mean bonds per atom")
    _a2.set_xlabel("2·#bonds / #atoms")
    _a2.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 解釈ガイド
    - **clash の局在**: cell 2 で VQ-recon の最近接距離分布が GT に重なるのに LM-gen だけ左(<1.2Å)に寄る
      → clash は**LMのコード列**起因。逆に VQ-recon も左寄りなら**デコーダ**起因。
    - **再構成 RMSD**(cell 3): Kabsch ≪ per-atom なら誤差は**剛体配置**(pocket fit)で、内部形状は良い。
      両方小さければデコーダの形状再現は良好 → 課題は LM か配置側。
    - **codebook**(cell 6): OOD gen mass が大きい = LM が「実分子では使われないコード」を多用
      → デコーダが未学習領域を引かされて clash。対策は (a) LM 改善、(b) デコーダの OOD ロバスト化/リファインメント。
    - **連結性**(cell 5): bonds は距離から事後推定なので、原子が離れすぎると分裂。
      recon で既に分裂が多いなら距離忠実度、gen だけ多いなら LM。
    """)
    return


if __name__ == "__main__":
    app.run()
