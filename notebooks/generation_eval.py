# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
"""3D ligand-generation quality evaluation.

Loads the metrics dumped by ``scripts/eval_generation.py`` (generated ligands
vs ground-truth, decoded to 3D) and renders one plot per cell. Regenerate the
data with::

    uv run python scripts/eval_generation.py \
        --lm-ckpt pocket-ligand-lm/<run>/checkpoints/<best>.ckpt \
        --num-pockets 60 --num-samples 3
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import Counter
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # Large fonts for readability.
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 16,
            "figure.dpi": 110,
        }
    )
    GEN_C, GT_C = "#d1495b", "#3a7ca5"  # generated / ground-truth colors
    return Counter, GEN_C, GT_C, Path, mo, np, plt


@app.cell
def _(Path, np):
    # Locate and load the eval dump (run from repo root or notebooks/).
    _cands = [
        Path.cwd() / "outputs/gen_eval/eval_data.npz",
        Path.cwd().parent / "outputs/gen_eval/eval_data.npz",
        Path(
            "/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen"
            "/outputs/gen_eval/eval_data.npz"
        ),
    ]
    DATA_PATH = next(p for p in _cands if p.exists())
    D = np.load(DATA_PATH, allow_pickle=True)
    return D, DATA_PATH


@app.cell
def _(D, DATA_PATH, mo):
    mo.md(
        f"""
        # 立体構造生成の品質評価

        **source**: `{DATA_PATH}`
        **checkpoint**: `{D["lm_ckpt"]}`

        - pockets: **{int(D["num_pockets"])}**
        - generated molecules: **{int(D["num_gen"])}**  ·  ground-truth: **{int(D["num_gt"])}**
        - **RDKit-valid (OpenBabel + largest-frag, DiffSBDD式)**: gen
          **{100 * D["gen_v_openbabel"].mean():.0f}%** / GT
          **{100 * D["gt_v_openbabel"].mean():.0f}%**
        - **PB-valid (PoseBusters geometry, DiffGui式)**: gen
          **{100 * D["gen_v_pb_valid"].mean():.0f}%** / GT
          **{100 * D["gt_v_pb_valid"].mean():.0f}%**
        - geometric-valid: gen **{100 * D["gen_v_geom_ok"].mean():.0f}%** /
          GT **{100 * D["gt_v_geom_ok"].mean():.0f}%**  （手法別は次ページの比較表）

        各セル＝1プロット。<span style="color:#d1495b">赤=生成</span> /
        <span style="color:#3a7ca5">青=正解(GT)</span>。
        """
    )
    return


@app.cell
def _(D, GEN_C, GT_C, np, plt):
    # 1. Bond-length distribution (geometry validity).
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _bins = np.linspace(0.8, 2.2, 60)
    _ax.hist(D["gen_bond_lengths"], bins=_bins, density=True, alpha=0.6, color=GEN_C, label="generated")
    _ax.hist(D["gt_bond_lengths"], bins=_bins, density=True, alpha=0.6, color=GT_C, label="ground-truth")
    for _x in (1.34, 1.54):
        _ax.axvline(_x, ls="--", lw=1.5, color="gray")
    _ax.set_title("Bond-length distribution")
    _ax.set_xlabel("inferred bond length (Å)")
    _ax.set_ylabel("density")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(D, GEN_C, GT_C, np, plt):
    # 2. Minimum interatomic distance (steric-clash proxy).
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _bins = np.linspace(0.5, 3.0, 50)
    _gen = D["gen_min_pair_dist"][np.isfinite(D["gen_min_pair_dist"])]
    _gt = D["gt_min_pair_dist"][np.isfinite(D["gt_min_pair_dist"])]
    _ax.hist(_gen, bins=_bins, density=True, alpha=0.6, color=GEN_C, label="generated")
    _ax.hist(_gt, bins=_bins, density=True, alpha=0.6, color=GT_C, label="ground-truth")
    _ax.axvline(1.2, ls="--", lw=2, color="black", label="clash < 1.2 Å")
    _ax.set_title("Min interatomic distance (clash check)")
    _ax.set_xlabel("nearest-atom distance (Å)")
    _ax.set_ylabel("density")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(D, GEN_C, GT_C, np, plt):
    # 3. Atom-count distribution.
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _hi = int(max(D["gen_n_atoms"].max(), D["gt_n_atoms"].max())) + 2
    _bins = np.arange(0, _hi, 2)
    _ax.hist(D["gen_n_atoms"], bins=_bins, density=True, alpha=0.6, color=GEN_C, label="generated")
    _ax.hist(D["gt_n_atoms"], bins=_bins, density=True, alpha=0.6, color=GT_C, label="ground-truth")
    _ax.set_title("Heavy-atom count per ligand")
    _ax.set_xlabel("number of heavy atoms")
    _ax.set_ylabel("density")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(Counter, D, GEN_C, GT_C, np, plt):
    # 4. Element composition (fraction of atoms).
    def _frac(arr):
        _c = Counter(str(e) for e in arr)
        return _c, (sum(_c.values()) or 1)

    _gc, _gtot = _frac(D["gen_elements"])
    _tc, _ttot = _frac(D["gt_elements"])
    _elems = sorted(set(_gc) | set(_tc), key=lambda e: -(_gc.get(e, 0) + _tc.get(e, 0)))[:8]
    _x = np.arange(len(_elems))
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _ax.bar(_x - 0.2, [_gc.get(e, 0) / _gtot for e in _elems], 0.4, color=GEN_C, label="generated")
    _ax.bar(_x + 0.2, [_tc.get(e, 0) / _ttot for e in _elems], 0.4, color=GT_C, label="ground-truth")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_elems)
    _ax.set_title("Element composition")
    _ax.set_xlabel("element")
    _ax.set_ylabel("fraction of atoms")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(D, GEN_C, GT_C, np, plt):
    # 5. Bonds-per-atom (mean connectivity degree).
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _bins = np.linspace(0, 4, 40)
    _ax.hist(D["gen_bonds_per_atom"], bins=_bins, density=True, alpha=0.6, color=GEN_C, label="generated")
    _ax.hist(D["gt_bonds_per_atom"], bins=_bins, density=True, alpha=0.6, color=GT_C, label="ground-truth")
    _ax.set_title("Connectivity: mean bonds per atom")
    _ax.set_xlabel("2·#bonds / #atoms")
    _ax.set_ylabel("density")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(D, GEN_C, GT_C, plt):
    # 6. Single-connected-molecule fraction (1 component = not fragmented).
    _gen_frac = float((D["gen_n_components"] == 1).mean())
    _gt_frac = float((D["gt_n_components"] == 1).mean())
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _bars = _ax.bar(["generated", "ground-truth"], [_gen_frac, _gt_frac], color=[GEN_C, GT_C], width=0.6)
    for _b, _v in zip(_bars, [_gen_frac, _gt_frac]):
        _ax.text(_b.get_x() + _b.get_width() / 2, _v + 0.02, f"{_v * 100:.0f}%", ha="center", fontsize=18)
    _ax.set_ylim(0, 1.1)
    _ax.set_title("Single connected molecule (no fragments)")
    _ax.set_ylabel("fraction")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(D, GEN_C, GT_C, plt):
    # 7. Validity comparison table: several definitions, generated vs GT.
    _names = {
        "connected": "Connected (1 fragment)",
        "no_clash": "No clash (≥1.0 Å)",
        "geom_ok": "Geometric (conn.+no-clash)",
        "rdkit_charge0": "RDKit (charge=0, strict)",
        "rdkit_chargesearch": "RDKit (charge search)",
        "rdkit_relaxed": "RDKit (UFF-relaxed)",
        "openbabel": "OpenBabel + largest-frag (DiffSBDD)",
        "pb_valid": "PoseBusters geom (DiffGui PB-valid)",
    }
    _rows = []
    for _m in [str(x) for x in D["methods"]]:
        _g = 100 * D[f"gen_v_{_m}"].mean()
        _t = 100 * D[f"gt_v_{_m}"].mean()
        _rows.append([_names.get(_m, _m), f"{_g:.0f}%", f"{_t:.0f}%"])
    _rows.append(["True SDF bonds (GT reference)", "—", f"{100 * D['gt_v_true_bonds'].mean():.0f}%"])

    _fig, _ax = plt.subplots(figsize=(12, 6))
    _ax.axis("off")
    _ax.set_title("Validity by method (generated vs ground-truth)", pad=24)
    _tab = _ax.table(
        cellText=_rows,
        colLabels=["validity method", "generated", "ground-truth"],
        cellLoc="center",
        colColours=["#eeeeee", GEN_C, GT_C],
        loc="center",
    )
    _tab.auto_set_font_size(False)
    _tab.set_fontsize(15)
    _tab.scale(1, 2.4)
    for (_r, _c), _cell in _tab.get_celld().items():
        _cell.set_width(0.48 if _c == 0 else 0.26)
        if _r == 0:
            _cell.set_text_props(color="white", weight="bold")
        if _c == 0:
            _cell.set_text_props(ha="left")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(D, GEN_C, GT_C, np, plt):
    # 8. Ligand-to-pocket-centroid distance (placement in the binding site).
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _hi = float(max(D["gen_centroid_dist"].max(), D["gt_centroid_dist"].max()))
    _bins = np.linspace(0, _hi, 40)
    _ax.hist(D["gen_centroid_dist"], bins=_bins, density=True, alpha=0.6, color=GEN_C, label="generated")
    _ax.hist(D["gt_centroid_dist"], bins=_bins, density=True, alpha=0.6, color=GT_C, label="ground-truth")
    _ax.set_title("Ligand centroid → pocket centroid distance")
    _ax.set_xlabel("distance (Å)")
    _ax.set_ylabel("density")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
