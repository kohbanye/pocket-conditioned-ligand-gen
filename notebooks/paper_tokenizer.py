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
    # Tokenizer evaluation — data collection for the paper tables

    All-atom トークナイザの **joint（protein+ligand を 1 codebook）** と
    **separate（protein-only VQ + ligand-only VQ）** を同一の test 複合体上で
    比較し、論文の 2 つの表に必要な数値を全部集める notebook。

    両者は `data/descriptor_cache_allatom` の同じ cache・同じ codebook size
    (8192)・同じアーキテクチャで学習され、差分は `--modality` のみ（統制済み）。

    ## 集める指標（列グループ）

    | group | metric | 意味 |
    |---|---|---|
    | **Ligand** | `lig_rmsd_frame` | pocket frame での per-atom RMSD（**絶対配置**＝ポーズ） |
    | | `lig_rmsd_kabsch` | Kabsch 後 RMSD（内部形状のみ） |
    | | `lig_bond_mae` / `lig_bond_max` | 結合長誤差 平均 / 最悪（FoldToken の `L_r` / `max L_r`） |
    | | `lig_angle_mae` / `lig_angle_max` | 結合角誤差 平均 / 最悪（`L_a` / `max L_a`） |
    | | `lig_element_acc` | element head の復元率（FoldToken の `Rec` に相当） |
    | **Pocket** | `prot_rmsd_frame` / `prot_rmsd_kabsch` | 同上 |
    | | `prot_lddt` | pocket 内 lDDT（局所距離の保存度、alignment 不要） |
    | | `prot_aa_acc` | 残基型復元率 |
    | **Interface** | `lddt_pli` | protein–ligand 原子間距離の lDDT（CASP15 準拠, R0=6Å） |
    | | `contact_f1` | 4Å 接触集合の F1（precision/recall も保存） |
    | | `clash_frac` | vdW 半径和の 0.75 倍を下回る pair の割合 |
    | | `iface_lig_rmsd` | 参照で接触しているリガンド原子に限った RMSD |
    | **Cost** | `bits/atom`, `vocab` | rate 正規化用（表に必ず入れる列） |

    出力は `outputs/tokenizer_eval/` 以下に
    `per_complex.csv`（tidy, 1 行 = 1 複合体 × 1 arm）と集計済み
    `table1_main.csv` / `table2_ablation.csv` / `codebook_stats.csv`。
    """)
    return


@app.cell
def _():
    import json
    import os
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from prolit.model.vqvae_module import AtomVQVAEModule
    from prolit.tokenizers.descriptor_schema import (
        ATOM_DESCRIPTOR_DIM,
        ATOM_LAYOUT,
        LIGAND_ELEMENT_VOCAB,
        PROTEIN_AA_VOCAB,
        fields_by_name,
    )

    plt.rcParams["figure.dpi"] = 120
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 19,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    OUT_DIR = project_root / "outputs" / "tokenizer_eval"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"output: {OUT_DIR}")
    return (
        ATOM_DESCRIPTOR_DIM,
        ATOM_LAYOUT,
        AtomVQVAEModule,
        LIGAND_ELEMENT_VOCAB,
        OUT_DIR,
        PROTEIN_AA_VOCAB,
        device,
        fields_by_name,
        json,
        np,
        os,
        pd,
        plt,
        project_root,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Arm registry

    1 arm = 1 トークナイザ構成。`protein` / `ligand` それぞれに (checkpoint,
    normalization stats) を持たせる形にしてあるので、joint は同じ ckpt を両方に
    指すだけ。**新しい ablation arm（共有フレームのみ / KNN 抜き / codebook size
    違いなど）はここに 1 エントリ足せば表に載る。**

    `vocab` は下流 LM が持つべき語彙サイズ（= bits/atom の計算元）。separate は
    2 つの book を連結する必要があるので 8192×2 = 16384 になる点が本質的。
    """)
    return


@app.cell
def _(os, project_root):
    VQ_DIR = project_root / "pocket-ligand-vqvae"
    CACHE_DIR = project_root / os.environ.get(
        "ATOM_VQVAE_CACHE_DIR", "data/descriptor_cache_allatom"
    )

    MIN_EPOCH = int(os.environ.get("TOKENIZER_EVAL_MIN_EPOCH", "90"))

    def _best_ckpt(ckpt_dir, min_epoch=MIN_EPOCH):
        """Lowest-val-metric checkpoint of a FINISHED run, or None.

        Checkpoint filenames embed the monitored metric (``val/atom_coord``), and
        the '/' in the metric name makes each one its own directory, so the files
        live at ``<ckpt_dir>/atomvqvae-epoch=NN-val/atom_coord=X.ckpt``.

        ``min_epoch`` guards against silently picking up a checkpoint from a run
        that is still training: a half-trained VQ would land in the paper table
        looking like a real ablation result.
        """
        found = []
        for path in ckpt_dir.glob("*/atom_coord=*.ckpt"):
            epoch = int(path.parent.name.split("epoch=")[1].split("-")[0])
            if epoch >= min_epoch:
                found.append((float(path.stem.split("=")[-1]), epoch, path))
        if not found:
            return None
        metric, epoch, path = min(found)
        print(f"    ckpt {ckpt_dir.parent.name}: epoch {epoch}, val {metric:.4f}")
        return path

    JOINT_CKPT = VQ_DIR / "xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
    PROT_CKPT = VQ_DIR / "protein-vqvae/checkpoints/atomvqvae-epoch=98-val/atom_coord=0.1412.ckpt"

    ARMS = [
        {
            "name": "joint",
            "label": "ProLIT",
            "frame": "shared",
            "codebook": "1 shared book",
            "protein_ckpt": JOINT_CKPT,
            "protein_norm": CACHE_DIR / "normalization_stats.pt",
            "ligand_ckpt": JOINT_CKPT,
            "ligand_norm": CACHE_DIR / "normalization_stats.pt",
            "vocab": 8192,
        },
    ]

    # --- no-learning reference --------------------------------------------
    ARMS.append({
        "name": "binning",
        "label": "Coordinate binning (no training)",
        "frame": "shared",
        "codebook": "grid",
        "kind": "binning",
        "vocab": 8000 * 12,
    })

    # --- separate-tokenizer ablation --------------------------------------
    # 4096+4096 matches the joint arm on codebook vectors AND on LM vocabulary
    # rows, so the two differ only in whether the book is shared. This is the
    # arm the paper reports.
    SEP4K_PROT = _best_ckpt(VQ_DIR / "protein-vqvae-4096/checkpoints")
    SEP4K_LIG = _best_ckpt(VQ_DIR / "ligand-vqvae-4096/checkpoints")
    if SEP4K_PROT is not None and SEP4K_LIG is not None:
        ARMS.append({
            "name": "separate",
            "label": "ProLIT (separate tokenizers)",
            "frame": "shared",
            "codebook": "2 books",
            "protein_ckpt": SEP4K_PROT,
            "protein_norm": CACHE_DIR / "normalization_stats_protein.pt",
            "ligand_ckpt": SEP4K_LIG,
            "ligand_norm": CACHE_DIR / "normalization_stats_ligand.pt",
            "vocab": 8192,
        })
    else:
        print(f"separate VQs not past epoch {MIN_EPOCH} -- arm skipped")

    # --- ligand-own-frame arms -------------------------------------------
    # A ligand-only tokenizer encodes the molecule in ITS OWN canonical frame,
    # so its tokens are SE(3)-invariant and carry no pose. These arms use a VQ
    # trained on `data/descriptor_cache_ligand_localframe` and are placed back
    # into the pocket with a pose budget (None = oracle placement).
    LOCAL_CACHE = project_root / os.environ.get(
        "TOKENIZER_EVAL_LOCAL_CACHE", "data/descriptor_cache_ligand_localframe"
    )
    LOCAL_LIG_CKPT = _best_ckpt(
        VQ_DIR
        / os.environ.get("TOKENIZER_EVAL_LOCAL_RUN", "ligand-vqvae-localframe")
        / "checkpoints"
    )

    if LOCAL_LIG_CKPT is not None:
        # Sweep the pose budget so the paper can report the BREAK-EVEN point:
        # how many extra tokens a ligand-own-frame tokenizer must spend on the
        # rigid transform before its interface metrics match joint, which spends
        # zero. Picking a single budget would look arbitrary.
        pose_sweep = [(None, "oracle"), (39, "3 tok"), (26, "2 tok"),
                      (20, "1.5 tok"), (13, "1 tok")]
        for pose_bits, tag in pose_sweep:
            ARMS.append({
                "name": f"lig_localframe_{tag.replace(' ', '')}",
                "label": f"Ligand-own-frame + {tag} pose",
                "frame": "per-modality",
                "codebook": "2 books",
                "protein_ckpt": PROT_CKPT,
                "protein_norm": CACHE_DIR / "normalization_stats_protein.pt",
                "ligand_ckpt": LOCAL_LIG_CKPT,
                "ligand_norm": LOCAL_CACHE / "normalization_stats_ligand.pt",
                "vocab": 16384,
                "ligand_frame": "local",
                "pose_bits": pose_bits,
            })
    else:
        print(f"ligand-vqvae-localframe not past epoch {MIN_EPOCH} -- arms skipped")

    for _a in ARMS:
        _a.setdefault("ligand_frame", "pocket")
        _a.setdefault("pose_bits", None)
        _a.setdefault("kind", "vq")
        if _a["kind"] != "vq":
            continue
        for _k in ("protein_ckpt", "ligand_ckpt", "protein_norm", "ligand_norm"):
            assert _a[_k].exists(), f"{_a['name']}: missing {_k} -> {_a[_k]}"  # noqa: S101
    print(f"{len(ARMS)} arms registered, all checkpoints present")
    for _a in ARMS:
        print(f"  {_a['name']:>24}  frame={_a['ligand_frame']:<12} pose_bits={_a['pose_bits']}")
    return ARMS, CACHE_DIR


@app.cell
def _(ARMS, ATOM_DESCRIPTOR_DIM, AtomVQVAEModule, GridQuantizer, device, torch):
    def _load_arm(arm):
        loaded = {}
        cache = {}
        if arm["kind"] == "binning":
            grid = GridQuantizer()
            # Raw descriptors in, raw out: identity normalization.
            identity = {
                "vqvae": grid,
                "mean": torch.zeros(ATOM_DESCRIPTOR_DIM, device=device),
                "std": torch.ones(ATOM_DESCRIPTOR_DIM, device=device),
                "codebook_size": grid.codebook_size,
            }
            return {"protein": identity, "ligand": identity}
        for side in ("protein", "ligand"):
            ckpt = arm[f"{side}_ckpt"]
            if ckpt not in cache:
                mod = AtomVQVAEModule.load_from_checkpoint(str(ckpt), map_location=device)
                mod.eval().to(device)
                cache[ckpt] = mod
            mod = cache[ckpt]
            stats = torch.load(arm[f"{side}_norm"], weights_only=False)
            loaded[side] = {
                "vqvae": mod.vqvae,
                "mean": stats["atom_mean"].to(device).float(),
                "std": stats["atom_std"].to(device).float(),
                "codebook_size": mod.config.atom.codebook_size,
            }
        return loaded

    MODELS = {}
    for arm_spec in ARMS:
        MODELS[arm_spec["name"]] = _load_arm(arm_spec)
        sizes = {s: MODELS[arm_spec["name"]][s]["codebook_size"] for s in ("protein", "ligand")}
        print(f"{arm_spec['name']:>10}: codebook sizes {sizes}")
    return (MODELS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Metric implementations

    - **lDDT** — 参照構造で近い原子ペアの距離が復元後も保たれているかを
      4 閾値 (0.5/1/2/4 Å) の平均で測る。superposition 不要なので「界面が壊れて
      いないか」を見るのに適する。
    - **lDDT-PLI** — 上記を **protein 原子 × ligand 原子のペアだけ**に限定した
      CASP15 の標準指標。joint の主張（相対配置が保存される）が最も直接に出る。
    - **contact F1** — 4 Å 以内の protein–ligand heavy-atom pair 集合の一致度。
    - **clash** — vdW 半径和の 0.75 倍未満を衝突とみなす（PoseBusters 準拠）。

    再構成なので原子の対応は自明（同じ原子集合・同じ順序）であり、
    アラインメントや原子マッチングの曖昧さは無い。
    """)
    return


@app.cell
def _(np):
    from rdkit.Chem import GetPeriodicTable

    _PT = GetPeriodicTable()
    _VDW_CACHE: dict[str, float] = {}

    def vdw_radius(symbol: str) -> float:
        """vdW radius in Angstrom, cached; unknown elements fall back to carbon."""
        if symbol not in _VDW_CACHE:
            try:
                _VDW_CACHE[symbol] = float(_PT.GetRvdw(symbol))
            except RuntimeError:
                _VDW_CACHE[symbol] = 1.7
        return _VDW_CACHE[symbol]

    LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)

    def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
        """RMSD after optimal superposition (internal shape only)."""
        p_c, q_c = p - p.mean(0), q - q.mean(0)
        u, _, vt = np.linalg.svd(q_c.T @ p_c)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        q_al = q_c @ (vt.T @ np.diag([1.0, 1.0, d]) @ u.T).T
        return float(np.sqrt(np.mean(np.sum((p_c - q_al) ** 2, -1))))

    def frame_rmsd(p: np.ndarray, q: np.ndarray) -> float:
        """RMSD in the shared pocket frame — NO superposition (absolute placement)."""
        return float(np.sqrt(np.mean(np.sum((p - q) ** 2, -1))))

    def _lddt_from_dists(ref_d: np.ndarray, mdl_d: np.ndarray, mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        delta = np.abs(mdl_d[mask] - ref_d[mask])
        return float(np.mean([(delta < t).mean() for t in LDDT_THRESHOLDS]))

    def lddt_intra(
        ref: np.ndarray, mdl: np.ndarray, group: np.ndarray, inclusion_radius: float = 15.0
    ) -> float:
        """lDDT over intra-molecular pairs, excluding pairs inside the same group.

        ``group`` is a per-atom integer id (residue index for a pocket) so that
        trivially-rigid intra-residue distances do not inflate the score.
        """
        ref_d = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=-1)
        mdl_d = np.linalg.norm(mdl[:, None, :] - mdl[None, :, :], axis=-1)
        mask = (ref_d < inclusion_radius) & (group[:, None] != group[None, :])
        mask &= np.triu(np.ones_like(mask, dtype=bool), k=1)
        return _lddt_from_dists(ref_d, mdl_d, mask)

    def lddt_pli(
        ref_prot: np.ndarray,
        ref_lig: np.ndarray,
        mdl_prot: np.ndarray,
        mdl_lig: np.ndarray,
        inclusion_radius: float = 6.0,
    ) -> float:
        """CASP15-style lDDT-PLI: lDDT restricted to protein-ligand atom pairs."""
        ref_d = np.linalg.norm(ref_prot[:, None, :] - ref_lig[None, :, :], axis=-1)
        mdl_d = np.linalg.norm(mdl_prot[:, None, :] - mdl_lig[None, :, :], axis=-1)
        return _lddt_from_dists(ref_d, mdl_d, ref_d < inclusion_radius)

    def contact_prf(
        ref_prot: np.ndarray,
        ref_lig: np.ndarray,
        mdl_prot: np.ndarray,
        mdl_lig: np.ndarray,
        cutoff: float = 4.0,
    ) -> tuple[float, float, float]:
        """Precision / recall / F1 of the protein-ligand contact set."""
        ref_c = np.linalg.norm(ref_prot[:, None, :] - ref_lig[None, :, :], axis=-1) < cutoff
        mdl_c = np.linalg.norm(mdl_prot[:, None, :] - mdl_lig[None, :, :], axis=-1) < cutoff
        tp = float((ref_c & mdl_c).sum())
        prec = tp / mdl_c.sum() if mdl_c.sum() else float("nan")
        rec = tp / ref_c.sum() if ref_c.sum() else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) > 0 else 0.0
        return prec, rec, f1

    def clash_stats(
        prot: np.ndarray,
        lig: np.ndarray,
        prot_vdw: np.ndarray,
        lig_vdw: np.ndarray,
        tolerance: float = 0.75,
    ) -> tuple[float, float, float]:
        """(clashing-pair fraction, fraction of ligand atoms clashing, min dist ratio)."""
        dist = np.linalg.norm(prot[:, None, :] - lig[None, :, :], axis=-1)
        limit = tolerance * (prot_vdw[:, None] + lig_vdw[None, :])
        clash = dist < limit
        ratio = dist / (prot_vdw[:, None] + lig_vdw[None, :])
        return (
            float(clash.mean()),
            float(clash.any(axis=0).mean()),
            float(ratio.min()),
        )

    def bond_geometry(
        ref: np.ndarray, mdl: np.ndarray, bonds: list[tuple[int, int]]
    ) -> dict[str, float]:
        """Bond-length and bond-angle absolute errors (mean and worst-case)."""
        if not bonds:
            return {"bond_mae": np.nan, "bond_max": np.nan,
                    "angle_mae": np.nan, "angle_max": np.nan}
        idx_i = np.array([b[0] for b in bonds])
        idx_j = np.array([b[1] for b in bonds])
        ref_len = np.linalg.norm(ref[idx_i] - ref[idx_j], axis=-1)
        mdl_len = np.linalg.norm(mdl[idx_i] - mdl[idx_j], axis=-1)
        len_err = np.abs(mdl_len - ref_len)

        neighbors: dict[int, list[int]] = {}
        for i, j in bonds:
            neighbors.setdefault(i, []).append(j)
            neighbors.setdefault(j, []).append(i)

        def _angles(coords):
            out = []
            for center, nbrs in neighbors.items():
                for a_i in range(len(nbrs)):
                    for b_i in range(a_i + 1, len(nbrs)):
                        v1 = coords[nbrs[a_i]] - coords[center]
                        v2 = coords[nbrs[b_i]] - coords[center]
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        # Degenerate (coincident atoms) still emits a slot: the
                        # reference and model lists must stay index-aligned, or
                        # a tokenizer that collapses two atoms onto one point
                        # silently shifts every later angle. NaN drops out of
                        # the nan-aware reductions below.
                        out.append(
                            np.nan
                            if n1 < 1e-6 or n2 < 1e-6
                            else np.degrees(np.arccos(
                                np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                            ))
                        )
            return np.array(out)

        ref_ang, mdl_ang = _angles(ref), _angles(mdl)
        ang_err = np.abs(mdl_ang - ref_ang) if len(ref_ang) else np.array([np.nan])
        finite = ang_err[np.isfinite(ang_err)]
        return {
            "bond_mae": float(len_err.mean()),
            "bond_max": float(len_err.max()),
            "angle_mae": float(finite.mean()) if finite.size else np.nan,
            "angle_max": float(finite.max()) if finite.size else np.nan,
        }

    print("metric helpers ready")
    return (
        bond_geometry,
        clash_stats,
        contact_prf,
        frame_rmsd,
        kabsch_rmsd,
        lddt_intra,
        lddt_pli,
        vdw_radius,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2b. Pose quantization (for ligand-own-frame arms)

    リガンド自身のフレームで符号化するトークナイザ（Mol-StrucTok, Geo2Seq,
    および本 notebook の `ligand-own-frame` arm）のトークン列は **SE(3) 不変**で、
    ポケット内のどこに置かれるかの情報を持たない。復元物を受容体に戻すには
    6 自由度の剛体変換を**別途送る必要がある**。

    その予算をビットで表して量子化するのがここ。1 トークン = 13 bits
    (codebook 8192) なので、「配置に n トークン使う」がそのまま比較できる。
    joint はこの追加予算が **0** である点が主張の核。

    - 並進: ポケットの bounding box を一辺 `2**(bits/3)` 分割した立方格子
    - 回転: 決定的に生成した `2**bits` 個の単位四元数から最近傍（q と −q は同一視）
    """)
    return


@app.cell
def _(np):
    def _rotation_grid(n_rot, seed=0):
        """(n_rot, 4) unit quaternions, deterministic for a given (n_rot, seed)."""
        rng = np.random.default_rng(seed)
        q = rng.normal(size=(n_rot, 4))
        return q / np.linalg.norm(q, axis=1, keepdims=True)

    def _quat_to_matrix(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    def _matrix_to_quat(rot):
        trace = np.trace(rot)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q = np.array([
                0.25 / s,
                (rot[2, 1] - rot[1, 2]) * s,
                (rot[0, 2] - rot[2, 0]) * s,
                (rot[1, 0] - rot[0, 1]) * s,
            ])
        else:
            i = int(np.argmax(np.diag(rot)))
            j, k = (i + 1) % 3, (i + 2) % 3
            s = 2.0 * np.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k, k])
            q = np.zeros(4)
            q[0] = (rot[k, j] - rot[j, k]) / s
            q[i + 1] = 0.25 * s
            q[j + 1] = (rot[j, i] + rot[i, j]) / s
            q[k + 1] = (rot[k, i] + rot[i, k]) / s
        return q / np.linalg.norm(q)

    def quantize_pose(centroid, rotation, box_origin, box_size, pose_bits, seed=0):
        """Quantize a rigid transform to ``pose_bits`` total bits.

        Half the budget goes to translation, half to rotation. ``pose_bits=None``
        returns the transform unchanged (the oracle-placement upper bound).
        """
        if pose_bits is None:
            return centroid, rotation
        trans_bits = pose_bits // 2
        rot_bits = pose_bits - trans_bits
        steps = max(int(round(2 ** (trans_bits / 3.0))), 1)
        cell = box_size / steps
        idx = np.clip(np.floor((centroid - box_origin) / cell), 0, steps - 1)
        centroid_q = box_origin + (idx + 0.5) * cell
        grid = _rotation_grid(2**rot_bits, seed)
        best = int(np.argmax(np.abs(grid @ _matrix_to_quat(rotation))))
        return centroid_q, _quat_to_matrix(grid[best])

    print("pose quantization ready")
    return (quantize_pose,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2c. Coordinate-binning baseline (no training)

    「学習した codebook は、単に空間を格子で区切るのと比べて本当に得か」を
    同じレートで測るための下界の参照。**学習が要らない**ので先行研究行の中で
    唯一自前で完全に再現できる。

    1 原子 = (元素, 空間セル) の組を 1 トークンにする。予算 13 bits を
    元素 3 bits (8 種) + 位置 10 bits (10×10×10 セル) に配分し、ポケット
    canonical frame の一辺 32 Å の立方体を等分する。VQ と同じ
    `encode` / `decode_to_outputs` インタフェースを持たせてあるので、
    arm registry からは他のトークナイザと全く同じように扱える。
    """)
    return


@app.cell
def _(ATOM_LAYOUT, LIGAND_ELEMENT_VOCAB, fields_by_name, np, torch):
    from prolit.tokenizers.geometry import (
        cartesian_to_spherical_np,
        spherical_to_cartesian_np,
    )

    class GridQuantizer:
        """Naive (element, spatial cell) tokenizer -- the no-learning reference.

        Presents the same surface as a trained VQ (``encode`` /
        ``decode_to_outputs``) so it drops into the arm registry unchanged.
        Operates on RAW descriptors, so its arm feeds identity mean/std.
        """

        def __init__(self, cells_per_axis=10, box=32.0):
            self.cells = cells_per_axis
            self.box = box
            self.n_elements = len(LIGAND_ELEMENT_VOCAB)
            fields = fields_by_name(ATOM_LAYOUT)
            self._coord = fields["coord"]
            self._element = fields["element"].start

        @property
        def codebook_size(self):
            return self.cells**3 * self.n_elements

        def _cell_index(self, cartesian):
            half = self.box / 2.0
            idx = np.floor((cartesian + half) / self.box * self.cells)
            return np.clip(idx, 0, self.cells - 1).astype(np.int64)

        def encode(self, x):
            desc = x.cpu().numpy()
            sph = desc[:, self._coord.start : self._coord.end].astype(np.float64)
            grid = self._cell_index(spherical_to_cartesian_np(sph))
            flat = (grid[:, 0] * self.cells + grid[:, 1]) * self.cells + grid[:, 2]
            element = desc[:, self._element].astype(np.int64)
            return torch.from_numpy(flat * self.n_elements + element).to(x.device)

        def decode_to_outputs(self, indices):
            codes = indices.cpu().numpy()
            element = codes % self.n_elements
            flat = codes // self.n_elements
            grid = np.stack(
                [flat // self.cells**2, (flat // self.cells) % self.cells,
                 flat % self.cells],
                axis=-1,
            ).astype(np.float64)
            half, step = self.box / 2.0, self.box / self.cells
            cartesian = (grid + 0.5) * step - half
            coord = cartesian_to_spherical_np(cartesian)
            logits = np.zeros((len(codes), self.n_elements), dtype=np.float32)
            logits[np.arange(len(codes)), element] = 1.0
            return {
                "coord": torch.from_numpy(coord).float().to(indices.device),
                "element": torch.from_numpy(logits).to(indices.device),
            }

    print(
        f"GridQuantizer ready: vocab {GridQuantizer().codebook_size} "
        f"({np.log2(GridQuantizer().codebook_size):.2f} bits/atom)"
    )
    return (GridQuantizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Test complex stream

    `visualization_atom.py` と同じサンプリング（cdonly fold0 test, label==1,
    `_min.sdf.gz`）。descriptor は **1 回だけ**計算し、全 arm で使い回すので
    どの arm も完全に同一の複合体・同一の入力を見る。
    """)
    return


@app.cell
def _(os, project_root):
    import gzip
    import re
    import tarfile
    from collections import defaultdict
    from functools import lru_cache

    import numpy as _np
    import pyarrow.parquet as pq

    from prolit.config import PocketExtractionConfig
    from prolit.tokenizers.atom import (
        LigandAtomDescriptor,
        ProteinAtomDescriptor,
        atom_descriptor_to_coords,
        precompute_receptor_atom_features,
    )
    from prolit.tokenizers.ligand import parse_sdf_text
    from prolit.tokenizers.protein import (
        compute_canonical_frame,
        extract_pocket_atoms_from_candidates,
        precompute_pocket_atom_candidates,
    )

    N_SAMPLES = int(os.environ.get("TOKENIZER_EVAL_N", "500"))
    SEED = 42

    pocket_config = PocketExtractionConfig(max_residues=50)
    prot_desc_calc = ProteinAtomDescriptor()
    lig_desc_calc = LigandAtomDescriptor()
    hub_cache = project_root / "data" / "hub_cache"
    receptor_dir = hub_cache / "receptors"
    ligand_repo = hub_cache / "repo" / "ligands"

    manifest = pq.read_table(hub_cache / "repo" / "manifest.parquet").to_pandas()
    test_df = manifest[
        (manifest["source_type"] == "cdonly")
        & (manifest["cdonly_fold0"] == "test")
        & (manifest["label"] == 1)
        & (manifest["ligand_sdf_gz"].str.endswith("_min.sdf.gz"))
    ].reset_index(drop=True)
    print(f"test pool: {len(test_df)} complexes; sampling up to {N_SAMPLES}")

    @lru_cache(maxsize=512)
    def load_receptor(rec_rel):
        path = receptor_dir / rec_rel
        if not path.exists():
            return None
        return (
            precompute_pocket_atom_candidates(path),
            precompute_receptor_atom_features(path),
        )

    _rng = _np.random.default_rng(SEED)
    _cand = test_df.iloc[
        _rng.choice(len(test_df), min(N_SAMPLES * 5, len(test_df)), replace=False)
    ]
    _shard_to_pairs = defaultdict(dict)
    for _row in _cand.itertuples(index=False):
        _shard_to_pairs[int(_row.shard_idx)][int(_row.pair_idx)] = (
            f"{_row.complex_dir}/{_row.receptor_pdb}"
        )
    _shard_order = list(_shard_to_pairs)
    _rng.shuffle(_shard_order)
    _member_re = re.compile(r"(\d+)\.sdf\.gz$")

    def iter_sampled():
        """Yield (complex_id, receptor_rel_path, parsed_mol) from the tar shards."""
        for shard_idx in _shard_order:
            wanted = _shard_to_pairs[shard_idx]
            tar_path = ligand_repo / f"{shard_idx:06d}.tar"
            if not tar_path.exists():
                continue
            with tarfile.open(tar_path, "r|") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    match = _member_re.search(member.name.rsplit("/", 1)[-1])
                    if match is None:
                        continue
                    pair_idx = int(match.group(1))
                    rec_rel = wanted.get(pair_idx)
                    if rec_rel is None:
                        continue
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        continue
                    mols = parse_sdf_text(
                        gzip.decompress(fobj.read()).decode("utf-8", "replace")
                    )
                    if mols:
                        yield f"{shard_idx}_{pair_idx}", rec_rel, mols[0]
    return (
        N_SAMPLES,
        compute_canonical_frame,
        load_receptor,
        atom_descriptor_to_coords,
        extract_pocket_atoms_from_candidates,
        iter_sampled,
        lig_desc_calc,
        pocket_config,
        prot_desc_calc,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Encode → decode → 3D

    各 arm について descriptor を正規化 → `encode` → `decode_to_outputs` →
    coord head を逆正規化して Cartesian に戻す。categorical head の argmax も
    同時に取り出して復元率（`Rec`）に使う。
    """)
    return


@app.cell
def _(ATOM_DESCRIPTOR_DIM, ATOM_LAYOUT, atom_descriptor_to_coords, fields_by_name, np, torch):
    _COORD_F = fields_by_name(ATOM_LAYOUT)["coord"]
    _CAT_SLOTS = {
        name: fields_by_name(ATOM_LAYOUT)[name].start
        for name in ("element", "charge", "hybrid", "aromatic", "ring", "numH", "aa", "bb_sc")
    }

    @torch.no_grad()
    def encode_decode(side_model, desc_np, meta, frame=None):
        """Return (reconstructed 3D coords, {head: accuracy}, codebook indices).

        ``frame`` overrides the (centroid, rotation) used to map the decoded
        canonical coordinates back to global space. Ligand-own-frame arms pass a
        quantized frame here; everything else decodes in the frame the
        descriptor was built in.
        """
        vq = side_model["vqvae"]
        mean_t, std_t = side_model["mean"], side_model["std"]
        desc_t = torch.from_numpy(desc_np).to(mean_t.device).float()
        norm = (desc_t - mean_t) / std_t
        indices = vq.encode(norm)
        outs = vq.decode_to_outputs(indices)

        coord = outs["coord"] * std_t[_COORD_F.start : _COORD_F.end] + mean_t[
            _COORD_F.start : _COORD_F.end
        ]
        recon = np.zeros((desc_np.shape[0], ATOM_DESCRIPTOR_DIM), dtype=np.float32)
        recon[:, _COORD_F.start : _COORD_F.end] = coord.cpu().numpy()
        coords_3d = atom_descriptor_to_coords(recon, meta, pocket_frame=frame)

        acc = {}
        for name, col in _CAT_SLOTS.items():
            if name not in outs:
                continue
            pred = outs[name].argmax(dim=-1).cpu().numpy()
            target = desc_np[:, col].astype(np.int64)
            acc[name] = float((pred == target).mean())
        return coords_3d, acc, indices.cpu().numpy()

    print("encode_decode ready")
    return (encode_decode,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Main evaluation loop

    1 複合体につき全 arm を回して 1 行ずつ記録する。コードブック統計用に
    indices も蓄積する。
    """)
    return


@app.cell
def _(
    ARMS,
    MODELS,
    N_SAMPLES,
    compute_canonical_frame,
    load_receptor,
    bond_geometry,
    clash_stats,
    contact_prf,
    encode_decode,
    extract_pocket_atoms_from_candidates,
    frame_rmsd,
    iter_sampled,
    kabsch_rmsd,
    lddt_intra,
    lddt_pli,
    lig_desc_calc,
    np,
    pd,
    pocket_config,
    prot_desc_calc,
    quantize_pose,
    vdw_radius,
):
    records = []
    pose_records = []
    code_usage = {a["name"]: {"protein": [], "ligand": []} for a in ARMS}
    n_done = 0
    n_skipped = 0

    for complex_id, rec_rel, mol in iter_sampled():
        if n_done >= N_SAMPLES:
            break
        rec = load_receptor(rec_rel)
        if rec is None:
            n_skipped += 1
            continue
        precomp, feats = rec
        heavy = np.array(
            [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"], dtype=np.float32
        )
        if len(heavy) == 0:
            n_skipped += 1
            continue
        pocket = extract_pocket_atoms_from_candidates(precomp, heavy, pocket_config)
        if pocket is None or pocket.atom_coords.shape[0] == 0:
            n_skipped += 1
            continue

        centroid, rotation = compute_canonical_frame(pocket.ca_coords.astype(np.float64))
        frame = (centroid, rotation)

        pdesc, pmeta = prot_desc_calc.compute(pocket, feats, frame)
        ldesc, _elems, lmeta = lig_desc_calc.compute(mol["atoms"], mol["bonds"], frame)
        if len(ldesc) == 0 or len(pdesc) == 0:
            n_skipped += 1
            continue

        # Ligand-own-frame view: same molecule, canonical frame built from the
        # ligand's own heavy atoms (what a single-modality ligand tokenizer sees).
        lig_frame = compute_canonical_frame(heavy.astype(np.float64))
        ldesc_local, _e2, lmeta_local = lig_desc_calc.compute(
            mol["atoms"], mol["bonds"], lig_frame
        )
        # Pocket bounding box: the placement search space both sides can derive
        # from the receptor, so quantizing the pose inside it is well defined.
        box_origin = pocket.atom_coords.astype(np.float64).min(axis=0)
        box_size = float(
            (pocket.atom_coords.astype(np.float64).max(axis=0) - box_origin).max()
        )

        prot_ref = pocket.atom_coords.astype(np.float64)
        heavy_to_orig = list(lmeta["heavy_to_orig"])
        lig_ref = np.array(
            [
                (mol["atoms"][i][1], mol["atoms"][i][2], mol["atoms"][i][3])
                for i in heavy_to_orig
            ],
            dtype=np.float64,
        )

        # Bonds re-indexed into the heavy-atom ordering used by the descriptor.
        orig_to_heavy = {o: h for h, o in enumerate(heavy_to_orig)}
        lig_bonds = [
            (orig_to_heavy[b[0]], orig_to_heavy[b[1]])
            for b in mol["bonds"]
            if b[0] in orig_to_heavy and b[1] in orig_to_heavy
        ]
        lig_bonds_typed = [
            (orig_to_heavy[b[0]], orig_to_heavy[b[1]], b[2])
            for b in mol["bonds"]
            if b[0] in orig_to_heavy and b[1] in orig_to_heavy
        ]
        lig_elements = [mol["atoms"][i][0] for i in heavy_to_orig]
        # Reference geometry gets a PB row too, as the achievable ceiling.
        pose_records.append({
            "complex_id": complex_id,
            "arm": "reference",
            "elements": lig_elements,
            "bonds": lig_bonds_typed,
            "coords": lig_ref,
        })

        # Residue grouping for pocket lDDT + vdW radii for clash detection.
        res_group = np.array(
            [hash((c, r)) % (2**31) for c, r in zip(pocket.atom_chain, pocket.atom_resseq, strict=True)]
        )
        prot_vdw = np.array([vdw_radius(e) for e in pocket.atom_elements])
        # Backbone-only view: residue-level protein tokenizers (ESM3 / FoldSeek /
        # FoldToken) only reconstruct N, CA, C, O, so an all-heavy-atom pocket
        # RMSD would not be comparable to their published numbers.
        bb_sel = np.array([n.strip() in ("N", "CA", "C", "O") for n in pocket.atom_names])
        lig_vdw = np.array([vdw_radius(mol["atoms"][i][0]) for i in heavy_to_orig])

        # Reference contact mask (used to define "interface ligand atoms").
        ref_cross = np.linalg.norm(prot_ref[:, None, :] - lig_ref[None, :, :], axis=-1)
        iface_lig = (ref_cross < 4.0).any(axis=0)

        for arm in ARMS:
            models = MODELS[arm["name"]]
            prot_mdl, prot_acc, prot_idx = encode_decode(models["protein"], pdesc, pmeta)
            if arm["ligand_frame"] == "local":
                # Decode in the ligand's own frame, then pay for the placement:
                # the token sequence itself is SE(3)-invariant and carries none.
                placed_frame = quantize_pose(
                    lmeta_local["centroid"],
                    lmeta_local["rotation"],
                    box_origin,
                    box_size,
                    arm["pose_bits"],
                )
                lig_mdl, lig_acc, lig_idx = encode_decode(
                    models["ligand"], ldesc_local, lmeta_local, frame=placed_frame
                )
            else:
                lig_mdl, lig_acc, lig_idx = encode_decode(
                    models["ligand"], ldesc, lmeta
                )
            code_usage[arm["name"]]["protein"].append(prot_idx)
            code_usage[arm["name"]]["ligand"].append(lig_idx)

            prec, recall, f1 = contact_prf(prot_ref, lig_ref, prot_mdl, lig_mdl)
            clash_pair, clash_atom, min_ratio = clash_stats(
                prot_mdl, lig_mdl, prot_vdw, lig_vdw
            )
            geom = bond_geometry(lig_ref, lig_mdl, lig_bonds)
            pose_records.append({
                "complex_id": complex_id,
                "arm": arm["name"],
                "elements": lig_elements,
                "bonds": lig_bonds_typed,
                "coords": lig_mdl,
            })

            records.append({
                "complex_id": complex_id,
                "arm": arm["name"],
                "n_prot_atoms": len(prot_ref),
                "n_lig_atoms": len(lig_ref),
                # --- Ligand ---
                "lig_rmsd_frame": frame_rmsd(lig_ref, lig_mdl),
                "lig_rmsd_kabsch": kabsch_rmsd(lig_ref, lig_mdl),
                "lig_bond_mae": geom["bond_mae"],
                "lig_bond_max": geom["bond_max"],
                "lig_angle_mae": geom["angle_mae"],
                "lig_angle_max": geom["angle_max"],
                "lig_element_acc": lig_acc.get("element", np.nan),
                # --- Pocket ---
                "prot_rmsd_frame": frame_rmsd(prot_ref, prot_mdl),
                "prot_rmsd_kabsch": kabsch_rmsd(prot_ref, prot_mdl),
                "prot_lddt": lddt_intra(prot_ref, prot_mdl, res_group),
                "prot_bb_rmsd_frame": (
                    frame_rmsd(prot_ref[bb_sel], prot_mdl[bb_sel]) if bb_sel.any() else np.nan
                ),
                "prot_bb_rmsd_kabsch": (
                    kabsch_rmsd(prot_ref[bb_sel], prot_mdl[bb_sel]) if bb_sel.any() else np.nan
                ),
                "prot_bb_lddt": (
                    lddt_intra(prot_ref[bb_sel], prot_mdl[bb_sel], res_group[bb_sel])
                    if bb_sel.any()
                    else np.nan
                ),
                "prot_element_acc": prot_acc.get("element", np.nan),
                "prot_aa_acc": prot_acc.get("aa", np.nan),
                # --- Interface ---
                "lddt_pli": lddt_pli(prot_ref, lig_ref, prot_mdl, lig_mdl),
                "contact_precision": prec,
                "contact_recall": recall,
                "contact_f1": f1,
                "clash_pair_frac": clash_pair,
                "clash_lig_atom_frac": clash_atom,
                "min_dist_ratio": min_ratio,
                "iface_lig_rmsd": (
                    frame_rmsd(lig_ref[iface_lig], lig_mdl[iface_lig])
                    if iface_lig.any()
                    else np.nan
                ),
            })
        n_done += 1

    metric_records = pd.DataFrame(records)
    print(f"evaluated {n_done} complexes ({n_skipped} skipped) x {len(ARMS)} arms")
    print(f"rows: {len(metric_records)}")
    return code_usage, metric_records, n_done, pose_records


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5b. PB-validity of the reconstruction

    再構成したリガンド座標に **元の結合グラフ**を載せて RDKit mol を作り、
    PoseBusters の `mol` チェック（sanitization / bond length / bond angle /
    internal clash / ring flatness / internal energy 等）を全通過したものを
    PB-valid とする。生成ではなく再構成なので結合認識は不要で、
    「トークナイザがどれだけ分子として壊すか」を純粋に測れる。

    `reference` 行は結晶座標そのものなので、達成可能な上限を与える
    （1.0 にならない分はデータ側の問題であり、tokenizer の責任ではない）。
    """)
    return


@app.cell
def _(OUT_DIR, os, pd, pose_records):
    from posebusters import PoseBusters
    from rdkit import Chem, RDLogger
    from rdkit.Geometry import Point3D

    RDLogger.DisableLog("rdApp.*")

    _BOND_ORDER = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }

    def build_mol(elements, bonds, coords):
        """RDKit mol from (elements, bond graph, coords); None if unbuildable."""
        rw = Chem.RWMol()
        for symbol in elements:
            rw.AddAtom(Chem.Atom(symbol))
        for i, j, order in bonds:
            if rw.GetBondBetweenAtoms(i, j) is None:
                rw.AddBond(i, j, _BOND_ORDER.get(order, Chem.BondType.SINGLE))
        mol = rw.GetMol()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for idx, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))
        mol.AddConformer(conf, assignId=True)
        try:
            Chem.SanitizeMol(mol)
        except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
            return None
        return mol

    # The PoseBusters energy-ratio check re-generates conformers and dominates
    # runtime (~1 s/mol), so PB is (a) capped at PB_N complexes, (b) cached per
    # (complex_id, arm), and (c) flushed every chunk so an interrupted run keeps
    # its work and a rerun resumes instead of restarting.
    PB_CACHE = OUT_DIR / "pb_checks.csv"
    PB_N = int(os.environ.get("TOKENIZER_EVAL_PB_N", "300"))
    PB_CHUNK = 25

    pb_complexes = []
    for pose in pose_records:
        if pose["complex_id"] not in pb_complexes:
            pb_complexes.append(pose["complex_id"])
    pb_complexes = set(pb_complexes[:PB_N])

    cached = pd.read_csv(PB_CACHE, dtype={"complex_id": str}) if PB_CACHE.exists() else None
    done = (
        set(zip(cached["complex_id"], cached["arm"], strict=True))
        if cached is not None
        else set()
    )
    todo = [
        pose
        for pose in pose_records
        if pose["complex_id"] in pb_complexes
        and (pose["complex_id"], pose["arm"]) not in done
    ]
    print(f"PB: {len(pb_complexes)} complexes, {len(done)} cached, {len(todo)} to compute")

    buster = PoseBusters(config="mol")

    def _pb_row(pose):
        built = build_mol(pose["elements"], pose["bonds"], pose["coords"])
        if built is None:
            return {
                "complex_id": pose["complex_id"], "arm": pose["arm"],
                "pb_valid": 0.0, "pb_sanitization": 0.0,
            }
        checks = buster.bust([built], None, None).iloc[0]
        return {
            "complex_id": pose["complex_id"],
            "arm": pose["arm"],
            "pb_valid": float(bool(checks.all())),
            **{f"pb_{name}": float(bool(val)) for name, val in checks.items()},
        }

    pb_frames = [cached] if cached is not None else []
    for start in range(0, len(todo), PB_CHUNK):
        chunk = pd.DataFrame([_pb_row(p) for p in todo[start : start + PB_CHUNK]])
        pb_frames.append(chunk)
        pd.concat(pb_frames, ignore_index=True).to_csv(PB_CACHE, index=False)
        print(f"  PB {min(start + PB_CHUNK, len(todo))}/{len(todo)}", flush=True)

    pb_df = pd.concat(pb_frames, ignore_index=True)
    pb_df["complex_id"] = pb_df["complex_id"].astype(str)
    pb_df = pb_df.drop_duplicates(subset=["complex_id", "arm"], keep="last")
    print(f"PB checks: {len(pb_df)} rows ({PB_CACHE})")
    print(pb_df.groupby("arm")["pb_valid"].agg(["mean", "count"]).to_string())
    return (pb_df,)


@app.cell
def _(metric_records, pb_df):
    per_complex = metric_records.merge(
        pb_df.drop(columns=["pb_sanitize"], errors="ignore"),
        on=["complex_id", "arm"],
        how="left",
    )
    print(f"per_complex: {len(per_complex)} rows, {len(per_complex.columns)} columns")
    return (per_complex,)


@app.cell
def _(OUT_DIR, per_complex):
    _path = OUT_DIR / "per_complex.csv"
    per_complex.to_csv(_path, index=False)
    print(f"wrote {_path}  ({len(per_complex)} rows)")
    per_complex.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Codebook statistics

    utilization / perplexity / bits-per-atom、および joint のみ意味を持つ
    **protein-ligand コード共有数**。bits/atom は表の rate 列にそのまま入る。
    """)
    return


@app.cell
def _(ARMS, MODELS, OUT_DIR, code_usage, np, pd):
    def _usage_stats(idx_list, codebook_size):
        flat = np.concatenate(idx_list) if idx_list else np.array([], dtype=np.int64)
        counts = np.bincount(flat, minlength=codebook_size).astype(float)
        probs = counts / max(counts.sum(), 1.0)
        nz = probs[probs > 0]
        return {
            "active": int((counts > 0).sum()),
            "util": float((counts > 0).sum() / codebook_size),
            "perplexity": float(np.exp(-(nz * np.log(nz)).sum())) if len(nz) else 0.0,
            "codes": set(np.unique(flat).tolist()),
            "n_atoms": len(flat),
        }

    cb_rows = []
    for cb_arm in ARMS:
        usage = code_usage[cb_arm["name"]]
        p_size = MODELS[cb_arm["name"]]["protein"]["codebook_size"]
        l_size = MODELS[cb_arm["name"]]["ligand"]["codebook_size"]
        p_stats = _usage_stats(usage["protein"], p_size)
        l_stats = _usage_stats(usage["ligand"], l_size)
        shared = (
            len(p_stats["codes"] & l_stats["codes"])
            if cb_arm["codebook"] == "1 shared book"
            else 0
        )
        cb_rows.append({
            "arm": cb_arm["name"],
            "label": cb_arm["label"],
            "codebook": cb_arm["codebook"],
            "vocab": cb_arm["vocab"],
            # Rate, NOT log2(combined vocab): the <p>..</p><l>..</l> block
            # structure makes the modality known, so a protein atom costs
            # log2(protein book) and a ligand atom log2(ligand book). Weighted
            # by the actual atom counts. `vocab` stays as the separate cost of
            # the LM's embedding/softmax rows.
            "bits_per_atom": float(
                (p_stats["n_atoms"] * np.log2(p_size)
                 + l_stats["n_atoms"] * np.log2(l_size))
                / max(p_stats["n_atoms"] + l_stats["n_atoms"], 1)
            ),
            "protein_codebook_size": p_size,
            "ligand_codebook_size": l_size,
            "protein_active": p_stats["active"],
            "protein_util": p_stats["util"],
            "protein_perplexity": p_stats["perplexity"],
            "ligand_active": l_stats["active"],
            "ligand_util": l_stats["util"],
            "ligand_perplexity": l_stats["perplexity"],
            "shared_codes": shared,
            "protein_only_codes": len(p_stats["codes"] - l_stats["codes"]) if shared else p_stats["active"],
            "ligand_only_codes": len(l_stats["codes"] - p_stats["codes"]) if shared else l_stats["active"],
            "n_protein_atoms": p_stats["n_atoms"],
            "n_ligand_atoms": l_stats["n_atoms"],
        })

    codebook_stats = pd.DataFrame(cb_rows)
    codebook_stats.to_csv(OUT_DIR / "codebook_stats.csv", index=False)
    print(f"wrote {OUT_DIR / 'codebook_stats.csv'}")
    codebook_stats
    return (codebook_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Table 1 — main comparison

    FoldToken の表に倣い、列グループを **Ligand / Pocket / Interface / Cost**
    にする。平均値だけでなく **success rate（閾値を満たした複合体の割合）** と
    **worst-case（max 系）** を持たせるのがポイント。

    先行研究（Mol-StrucTok / ESM3-struct / FoldToken など）の行は、それぞれを
    このパイプラインで評価できるようになった時点で `ARMS` に追加すれば
    同じ表に載る。現時点では joint / separate の 2 行。
    """)
    return


@app.cell
def _(ARMS, codebook_stats, np, pd, per_complex):
    SUCCESS_THRESHOLDS = {
        "lig_rmsd_kabsch": [("Lig shape<0.25A", 0.25), ("Lig shape<0.5A", 0.5)],
        "lig_rmsd_frame": [("Lig pose<0.5A", 0.5), ("Lig pose<1.0A", 1.0)],
        "prot_rmsd_frame": [("Pocket<1.0A", 1.0)],
        "lddt_pli": [("lDDT-PLI>0.8", 0.8)],
    }

    def _aggregate(df):
        row = {}
        for col in df.columns:
            if col in ("complex_id", "arm"):
                continue
            row[f"{col}_mean"] = float(np.nanmean(df[col]))
            row[f"{col}_median"] = float(np.nanmedian(df[col]))
        for col, specs in SUCCESS_THRESHOLDS.items():
            for name, thr in specs:
                series = df[col].to_numpy()
                hit = series > thr if col.startswith("lddt") else series < thr
                row[name] = float(np.nanmean(hit.astype(float)))
        return row

    agg_rows = []
    for agg_arm in ARMS:
        sub = per_complex[per_complex["arm"] == agg_arm["name"]]
        stats = _aggregate(sub)
        cb = codebook_stats[codebook_stats["arm"] == agg_arm["name"]].iloc[0]
        agg_rows.append({
            "arm": agg_arm["name"],
            "label": agg_arm["label"],
            "frame": agg_arm["frame"],
            "codebook": agg_arm["codebook"],
            "vocab": agg_arm["vocab"],
            "pose_bits": agg_arm["pose_bits"] or 0,
            "bits_per_atom": cb["bits_per_atom"],
            "n_complexes": len(sub),
            **stats,
        })

    aggregate = pd.DataFrame(agg_rows)

    TABLE1_COLUMNS = [
        ("label", "Method"),
        ("bits_per_atom", "bits/atom"),
        ("pose_bits", "pose bits"),
        # Ligand
        ("lig_rmsd_frame_mean", "Lig RMSD (frame)"),
        ("lig_rmsd_kabsch_mean", "Lig RMSD (Kabsch)"),
        ("Lig shape<0.25A", "Lig<0.25A"),
        ("pb_valid_mean", "PB-valid"),
        ("lig_bond_mae_mean", "L_r"),
        ("lig_bond_max_mean", "max L_r"),
        ("lig_angle_mae_mean", "L_a"),
        ("lig_element_acc_mean", "Lig Rec"),
        # Pocket
        ("prot_rmsd_frame_mean", "Pocket RMSD"),
        ("prot_lddt_mean", "Pocket lDDT"),
        ("prot_bb_rmsd_frame_mean", "BB RMSD"),
        ("prot_bb_lddt_mean", "BB lDDT"),
        ("prot_aa_acc_mean", "Res Rec"),
        # Interface
        ("lddt_pli_mean", "lDDT-PLI"),
        ("contact_f1_mean", "Contact F1"),
        ("clash_lig_atom_frac_mean", "Clash"),
        ("iface_lig_rmsd_mean", "Iface RMSD"),
    ]

    table1 = aggregate[[c for c, _ in TABLE1_COLUMNS]].copy()
    table1.columns = [n for _, n in TABLE1_COLUMNS]
    table1
    return TABLE1_COLUMNS, aggregate, table1


@app.cell
def _(OUT_DIR, aggregate, table1):
    aggregate.to_csv(OUT_DIR / "aggregate_full.csv", index=False)
    table1.to_csv(OUT_DIR / "table1_main.csv", index=False)
    print(f"wrote {OUT_DIR / 'table1_main.csv'} and aggregate_full.csv")
    print()
    print(table1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Table 2 — design ablation (cumulative)

    「separate から joint へ何を変えたか」を 1 行ずつ積み上げる表。設計軸
    (`frame` / `codebook`) を明示列にし、指標は Table 1 の部分集合に絞る。

    現在は 2 行だが、`ARMS` に共有フレームのみ・KNN 抜き・codebook size 違いの
    arm を足せば行が増える。
    """)
    return


@app.cell
def _(OUT_DIR, aggregate):
    TABLE2_COLUMNS = [
        ("label", "Config"),
        ("frame", "Frame"),
        ("codebook", "Codebook"),
        ("bits_per_atom", "bits/atom"),
        ("lig_rmsd_frame_mean", "Lig RMSD"),
        ("lig_rmsd_frame_median", "med"),
        ("prot_rmsd_frame_mean", "Pocket RMSD"),
        ("prot_rmsd_frame_median", "med"),
        ("lddt_pli_mean", "lDDT-PLI"),
        ("contact_f1_mean", "Contact F1"),
        ("clash_lig_atom_frac_mean", "Clash"),
    ]
    table2 = aggregate[[c for c, _ in TABLE2_COLUMNS]].copy()
    table2.columns = [n for _, n in TABLE2_COLUMNS]
    table2.to_csv(OUT_DIR / "table2_ablation.csv", index=False)
    print(f"wrote {OUT_DIR / 'table2_ablation.csv'}")
    print()
    print(table2.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    table2
    return (table2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Paired significance tests

    同一複合体上での対比較なので、arm 間の差は **paired Wilcoxon** で検定できる。
    論文では「joint が sep より lDDT-PLI で有意に良い (p=...)」と書ける。
    """)
    return


@app.cell
def _(ARMS, OUT_DIR, np, pd, per_complex):
    from scipy.stats import wilcoxon

    PAIRED_METRICS = [
        "lig_rmsd_frame", "lig_rmsd_kabsch", "prot_rmsd_frame", "prot_lddt",
        "lddt_pli", "contact_f1", "clash_lig_atom_frac", "iface_lig_rmsd",
        "lig_bond_mae", "lig_angle_mae",
    ]

    base_arm = ARMS[0]["name"]
    test_rows = []
    for test_arm in ARMS[1:]:
        pivot_a = per_complex[per_complex["arm"] == base_arm].set_index("complex_id")
        pivot_b = per_complex[per_complex["arm"] == test_arm["name"]].set_index("complex_id")
        common = pivot_a.index.intersection(pivot_b.index)
        for test_metric in PAIRED_METRICS:
            x = pivot_a.loc[common, test_metric].to_numpy()
            y = pivot_b.loc[common, test_metric].to_numpy()
            ok = ~(np.isnan(x) | np.isnan(y))
            stat, pval = wilcoxon(x[ok], y[ok]) if ok.sum() > 10 else (np.nan, np.nan)
            test_rows.append({
                "metric": test_metric,
                "arm_a": base_arm,
                "arm_b": test_arm["name"],
                "mean_a": float(np.nanmean(x[ok])),
                "mean_b": float(np.nanmean(y[ok])),
                "delta": float(np.nanmean(x[ok]) - np.nanmean(y[ok])),
                "n": int(ok.sum()),
                "wilcoxon_p": float(pval),
            })

    paired_tests = pd.DataFrame(test_rows)
    paired_tests.to_csv(OUT_DIR / "paired_tests.csv", index=False)
    print(f"wrote {OUT_DIR / 'paired_tests.csv'}")
    paired_tests
    return (paired_tests,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Distributions

    平均だけでは差が見えにくいので、joint / separate の分布を重ねて確認する。
    論文の図にするなら lDDT-PLI と ligand pose RMSD の 2 枚が主役。
    """)
    return


@app.cell
def _(ARMS, np, per_complex, plt):
    PLOT_METRICS = [
        ("lig_rmsd_frame", "Ligand RMSD in pocket frame (A)", False),
        ("lig_rmsd_kabsch", "Ligand RMSD after Kabsch (A)", False),
        ("prot_rmsd_frame", "Pocket RMSD (A)", False),
        ("lddt_pli", "lDDT-PLI", True),
        ("contact_f1", "Contact F1 (4A)", True),
        ("clash_lig_atom_frac", "Fraction of clashing ligand atoms", False),
    ]

    dist_fig, dist_axes = plt.subplots(2, 3, figsize=(21, 11))
    for dist_ax, (dist_metric, xlabel, higher_better) in zip(
        dist_axes.ravel(), PLOT_METRICS, strict=True
    ):
        for plot_arm in ARMS:
            vals = per_complex.loc[
                per_complex["arm"] == plot_arm["name"], dist_metric
            ].to_numpy()
            vals = vals[~np.isnan(vals)]
            dist_ax.hist(
                vals, bins=40, alpha=0.55,
                label=f"{plot_arm['name']} (med={np.median(vals):.3f})",
            )
        dist_ax.set_xlabel(xlabel)
        dist_ax.set_ylabel("Count")
        dist_ax.set_title(f"{dist_metric}  ({'higher' if higher_better else 'lower'} is better)")
        dist_ax.legend()
    dist_fig.tight_layout()
    dist_fig
    return


@app.cell
def _(ARMS, np, per_complex, plt):
    scatter_fig, scatter_axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for sc_ax, sc_metric in zip(scatter_axes, ["lddt_pli", "lig_rmsd_frame"], strict=True):
        pivot = per_complex.pivot(index="complex_id", columns="arm", values=sc_metric)
        cols = [a["name"] for a in ARMS][:2]
        sc_ax.scatter(pivot[cols[1]], pivot[cols[0]], s=12, alpha=0.5)
        lo = float(np.nanmin(pivot[cols].to_numpy()))
        hi = float(np.nanmax(pivot[cols].to_numpy()))
        sc_ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        sc_ax.set_xlabel(f"{cols[1]}")
        sc_ax.set_ylabel(f"{cols[0]}")
        sc_ax.set_title(f"per-complex {sc_metric}")
    scatter_fig.tight_layout()
    scatter_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. LaTeX export

    そのまま論文に貼れる形（booktabs）で吐く。数値の丸めと bold は最終的に
    手で調整する前提。
    """)
    return


@app.cell
def _(OUT_DIR, json, n_done, table1, table2):
    def _to_latex(df, caption, label):
        return df.to_latex(
            index=False,
            escape=True,
            float_format="%.3f",
            caption=caption,
            label=label,
            column_format="l" + "c" * (len(df.columns) - 1),
        )

    latex_1 = _to_latex(
        table1,
        "All-atom tokenizer quality on held-out CrossDocked complexes.",
        "tab:tokenizer_main",
    )
    latex_2 = _to_latex(
        table2,
        "Design ablation: from separate single-modality tokenizers to the joint tokenizer.",
        "tab:tokenizer_ablation",
    )
    (OUT_DIR / "table1_main.tex").write_text(latex_1)
    (OUT_DIR / "table2_ablation.tex").write_text(latex_2)
    (OUT_DIR / "meta.json").write_text(
        json.dumps({"n_complexes": int(n_done), "split": "cdonly fold0 test, label==1, _min"}, indent=2)
    )
    print(f"wrote LaTeX to {OUT_DIR}")
    print(latex_1)
    return


if __name__ == "__main__":
    app.run()
