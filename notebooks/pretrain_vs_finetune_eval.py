# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
"""Pretrained vs fine-tuned LM: 3D ligand-generation quality comparison.

Overlays the per-molecule geometry/validity metrics of several LM checkpoints,
each dumped by ``scripts/eval_generation.py`` into its own ``.npz``. The point
is to check whether GEOM pretraining fixes the distorted-shape problem: a
fine-tuned (pretrained) model should match or beat the from-scratch baseline on
bond-length/clash/validity, and the pretrained model alone should already emit
valid ligand shapes.

Generate the dumps first (one per checkpoint), e.g.::

    # pretrained ligand-only model -- its native UNconditional mode
    uv run python scripts/eval_generation.py \
        --lm-ckpt pocket-ligand-lm/gdnesyzx/checkpoints/lm-e01-vl1.8593.ckpt \
        --empty-pocket --label "pretrain (GEOM)" \
        --out outputs/gen_cmp/pretrain.npz
    # fine-tuned model -- pocket-conditioned
    uv run python scripts/eval_generation.py \
        --lm-ckpt pocket-ligand-lm/<finetune-run>/checkpoints/<best>.ckpt \
        --label "finetune (GEOM->CrossDocked)" \
        --out outputs/gen_cmp/finetune.npz
    # from-scratch baseline -- pocket-conditioned (did pretraining help?)
    uv run python scripts/eval_generation.py \
        --lm-ckpt pocket-ligand-lm/g79let5b/checkpoints/lm-e09-vl1.4088.ckpt \
        --label "scratch (CrossDocked only)" \
        --out outputs/gen_cmp/scratch.npz

Missing dumps are skipped, so the notebook renders with whatever exists.
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from collections import Counter
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
            "legend.fontsize": 14,
            "figure.dpi": 110,
        }
    )
    GT_COLOR = "#3a7ca5"  # ground-truth reference (filled grey-blue)
    # Per-model colors, assigned in spec order.
    MODEL_COLORS = ["#d1495b", "#e0a32e", "#66a182", "#8d5fd3"]
    return Counter, GT_COLOR, MODEL_COLORS, Path, mo, np, os, plt


@app.cell
def _(MODEL_COLORS, Path, np, os):
    # Which dumps to compare. Edit labels/paths as needed; missing files are
    # skipped so the notebook works before every checkpoint has been evaluated.
    # Repository root, found from this file so the notebook runs from any
    # checkout; PROLIT_ROOT overrides it.
    _repo = Path(os.environ.get("PROLIT_ROOT") or Path(__file__).resolve().parent.parent)
    PROJECT = _repo
    SPECS = [
        ("finetune (GEOM->CrossDocked)", "outputs/gen_cmp/finetune.npz"),
        ("pretrain (GEOM, empty pocket)", "outputs/gen_cmp/pretrain.npz"),
        ("scratch (CrossDocked only)", "outputs/gen_cmp/scratch.npz"),
    ]

    def _resolve(rel):
        for base in (Path.cwd(), Path.cwd().parent, PROJECT):
            p = base / rel
            if p.exists():
                return p
        return None

    models = []
    for i, (label, rel) in enumerate(SPECS):
        path = _resolve(rel)
        if path is None:
            continue
        models.append(
            {
                "label": label,
                "color": MODEL_COLORS[i % len(MODEL_COLORS)],
                "path": path,
                "D": np.load(path, allow_pickle=True),
            }
        )
    # Ground-truth reference: prefer a pocket-conditioned dump (full pockets).
    gt_ref = next(
        (m["D"] for m in models if not bool(m["D"]["empty_pocket"])),
        models[0]["D"] if models else None,
    )
    return SPECS, gt_ref, models


@app.cell
def _(models, np):
    # Small accessors used by every plot.
    def gen(model, key):
        return model["D"][f"gen_{key}"]

    def gt(ref, key):
        return ref[f"gt_{key}"]

    def finite(arr):
        arr = np.asarray(arr, dtype=float)
        return arr[np.isfinite(arr)]

    n_models = len(models)
    return finite, gen, gt, n_models


@app.cell
def _(SPECS, mo, models, n_models):
    _lines = "\n".join(
        f"- <span style='color:{m['color']}'>**{m['label']}**</span> — "
        f"{int(m['D']['num_gen'])} gen mols, "
        f"{'unconditional' if bool(m['D']['empty_pocket']) else 'pocket-conditioned'}"
        f"  ·  `{m['path'].name}`"
        for m in models
    )
    _loaded_names = {m["path"].name for m in models}
    _missing = [rel for _, rel in SPECS if rel.split("/")[-1] not in _loaded_names]
    _missing_md = (
        f"\n\n_未生成（スキップ）_: {', '.join(_missing)}" if _missing else ""
    )
    mo.md(
        f"""
        # 事前学習 vs fine-tune: 生成リガンドの立体品質比較

        各モデルが生成したリガンドを VQ-VAE で 3D 復元し、形状・妥当性メトリクスを
        正解(GT, 青)に重ねて比較。狙い: **GEOM 事前学習で「形の崩れ」が改善したか**。

        **比較対象 ({n_models}):**

        {_lines}{_missing_md}

        - 形状メトリクス(結合長・クラッシュ・連結性・原子数・元素)は**フレーム不変**
          なので無条件生成(empty pocket)とも公平に比較可。
        - centroid 距離だけは pocket 条件付きモデルのみ意味を持つ(後段)。
        """
    )
    return


@app.cell
def _(gen, gt, gt_ref, models, plt):
    # 1. Validity-by-method comparison table (headline).
    _method_names = {
        "connected": "Connected (1 fragment)",
        "no_clash": "No clash (≥1.0 Å)",
        "geom_ok": "Geometric (conn.+no-clash+bonds)",
        "rdkit_charge0": "RDKit (charge=0, strict)",
        "rdkit_chargesearch": "RDKit (charge search)",
        "rdkit_relaxed": "RDKit (UFF-relaxed)",
    }
    _methods = [str(x) for x in gt_ref["methods"]]
    _col_labels = ["validity method", *[m["label"].split(" ")[0] for m in models], "GT"]
    _rows = [
        [
            _method_names.get(_meth, _meth),
            *[f"{100 * gen(_m, f'v_{_meth}').mean():.0f}%" for _m in models],
            f"{100 * gt(gt_ref, f'v_{_meth}').mean():.0f}%",
        ]
        for _meth in _methods
    ]
    _rows.append(
        ["True SDF bonds (GT ceiling)", *["—"] * len(models),
         f"{100 * gt(gt_ref, 'v_true_bonds').mean():.0f}%"]
    )

    _fig, _ax = plt.subplots(figsize=(4 + 2.4 * len(models), 5.5))
    _ax.axis("off")
    _ax.set_title("Validity by method (per model vs GT)", pad=22)
    _col_colors = ["#eeeeee", *[m["color"] for m in models], "#3a7ca5"]
    _tab = _ax.table(
        cellText=_rows, colLabels=_col_labels, cellLoc="center",
        colColours=_col_colors, loc="center",
    )
    _tab.auto_set_font_size(False)
    _tab.set_fontsize(13)
    _tab.scale(1, 2.3)
    for (_r, _c), _cell in _tab.get_celld().items():
        _cell.set_width(0.40 if _c == 0 else 0.18)
        if _r == 0:
            _cell.set_text_props(color="white", weight="bold")
        if _c == 0:
            _cell.set_text_props(ha="left")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(GT_COLOR, gen, gt, gt_ref, models, np, plt):
    # 2. Bond-length distribution (the core "shape" check).
    _fig, _ax = plt.subplots(figsize=(11, 6))
    _bins = np.linspace(0.8, 2.2, 70)
    _ax.hist(gt(gt_ref, "bond_lengths"), bins=_bins, density=True, alpha=0.35,
             color=GT_COLOR, label="ground-truth")
    for _m in models:
        _ax.hist(gen(_m, "bond_lengths"), bins=_bins, density=True, histtype="step",
                 lw=2.4, color=_m["color"], label=_m["label"].split(" ")[0])
    for _x in (1.34, 1.54):
        _ax.axvline(_x, ls="--", lw=1.2, color="gray")
    _ax.set_title("Bond-length distribution")
    _ax.set_xlabel("inferred bond length (Å)")
    _ax.set_ylabel("density")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(GT_COLOR, finite, gen, gt, gt_ref, models, np, plt):
    # 3. Minimum interatomic distance (steric-clash proxy).
    _fig, _ax = plt.subplots(figsize=(11, 6))
    _bins = np.linspace(0.5, 3.0, 55)
    _ax.hist(finite(gt(gt_ref, "min_pair_dist")), bins=_bins, density=True,
             alpha=0.35, color=GT_COLOR, label="ground-truth")
    for _m in models:
        _ax.hist(finite(gen(_m, "min_pair_dist")), bins=_bins, density=True,
                 histtype="step", lw=2.4, color=_m["color"],
                 label=_m["label"].split(" ")[0])
    _ax.axvline(1.2, ls="--", lw=2, color="black", label="clash < 1.2 Å")
    _ax.set_title("Min interatomic distance (clash check)")
    _ax.set_xlabel("nearest-atom distance (Å)")
    _ax.set_ylabel("density")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(GT_COLOR, gen, gt, gt_ref, models, np, plt):
    # 4. Connectivity: mean bonds per atom.
    _fig, _ax = plt.subplots(figsize=(11, 6))
    _bins = np.linspace(0, 4, 45)
    _ax.hist(gt(gt_ref, "bonds_per_atom"), bins=_bins, density=True, alpha=0.35,
             color=GT_COLOR, label="ground-truth")
    for _m in models:
        _ax.hist(gen(_m, "bonds_per_atom"), bins=_bins, density=True,
                 histtype="step", lw=2.4, color=_m["color"],
                 label=_m["label"].split(" ")[0])
    _ax.set_title("Connectivity: mean bonds per atom")
    _ax.set_xlabel("2·#bonds / #atoms")
    _ax.set_ylabel("density")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(GT_COLOR, gen, gt, gt_ref, models, np, plt):
    # 5. Heavy-atom count per ligand.
    _fig, _ax = plt.subplots(figsize=(11, 6))
    _hi = int(max(gt(gt_ref, "n_atoms").max(),
                  max((gen(m, "n_atoms").max() for m in models), default=10))) + 2
    _bins = np.arange(0, _hi, 2)
    _ax.hist(gt(gt_ref, "n_atoms"), bins=_bins, density=True, alpha=0.35,
             color=GT_COLOR, label="ground-truth")
    for _m in models:
        _ax.hist(gen(_m, "n_atoms"), bins=_bins, density=True, histtype="step",
                 lw=2.4, color=_m["color"], label=_m["label"].split(" ")[0])
    _ax.set_title("Heavy-atom count per ligand")
    _ax.set_xlabel("number of heavy atoms")
    _ax.set_ylabel("density")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(Counter, GT_COLOR, gen, gt, gt_ref, models, np, plt):
    # 6. Element composition (fraction of atoms).
    def _frac(arr):
        c = Counter(str(e) for e in arr)
        return c, (sum(c.values()) or 1)

    _gt_c, _gt_tot = _frac(gt(gt_ref, "elements"))
    _model_fracs = [(_m, *_frac(gen(_m, "elements"))) for _m in models]
    _elems = sorted(_gt_c, key=lambda e: -_gt_c[e])[:8]
    _x = np.arange(len(_elems))
    _ngrp = len(models) + 1
    _w = 0.8 / _ngrp
    _fig, _ax = plt.subplots(figsize=(12, 6))
    _ax.bar(_x - 0.4 + _w / 2, [_gt_c.get(e, 0) / _gt_tot for e in _elems], _w,
            color=GT_COLOR, label="GT")
    for _j, (_m, _c, _tot) in enumerate(_model_fracs, start=1):
        _ax.bar(_x - 0.4 + _w / 2 + _j * _w,
                [_c.get(e, 0) / _tot for e in _elems], _w,
                color=_m["color"], label=_m["label"].split(" ")[0])
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
def _(GT_COLOR, gen, gt, gt_ref, models, np, plt):
    # 7. Ligand→pocket-centroid distance — pocket-conditioned models only.
    _cond = [m for m in models if not bool(m["D"]["empty_pocket"])]
    _fig, _ax = plt.subplots(figsize=(11, 6))
    _hi = float(max(gt(gt_ref, "centroid_dist").max(),
                    *[gen(m, "centroid_dist").max() for m in _cond]))
    _bins = np.linspace(0, _hi, 40)
    _ax.hist(gt(gt_ref, "centroid_dist"), bins=_bins, density=True, alpha=0.35,
             color=GT_COLOR, label="ground-truth")
    for _m in _cond:
        _ax.hist(gen(_m, "centroid_dist"), bins=_bins, density=True,
                 histtype="step", lw=2.4, color=_m["color"],
                 label=_m["label"].split(" ")[0])
    _ax.set_title("Ligand centroid → pocket centroid (conditioned only)")
    _ax.set_xlabel("distance (Å)")
    _ax.set_ylabel("density")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
