import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3D Structure & Token Sequence Visualization

    CrossDocked2020 から実際の複合体を1つ取り出し、ポケット構造・リガンド構造を py3Dmol で表示するとともに、VQ-VAE によるトークン列を確認する。

    **内容:**
    1. 複合体の3Dビューア（ポケット + リガンド）
    2. トークン列の可視化
    3. Codebook index 分布
    4. VQ-VAE 再構成精度の可視化（元構造 vs 復元構造）
    """)
    return


@app.cell
def _():
    import gzip
    import re
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import py3Dmol
    import torch

    # Add project root to path (resolve from this file, not cwd)
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.config import PocketExtractionConfig
    from src.data.descriptors import _parse_types_file
    from src.model.vqvae_module import VQVAEModule
    from src.tokenizers.ligand import LigandDescriptor, parse_sdf
    from src.tokenizers.protein import (
        PocketDescriptor,
        extract_full_sequence,
        extract_pocket,
    )
    from src.tokenizers.sequence import TokenSequenceAssembler

    # Alias for cross-cell access (marimo treats _-prefixed names as cell-local)
    parse_types_file = _parse_types_file

    plt.rcParams["figure.dpi"] = 120
    return (
        LigandDescriptor,
        PocketDescriptor,
        PocketExtractionConfig,
        TokenSequenceAssembler,
        VQVAEModule,
        extract_full_sequence,
        extract_pocket,
        gzip,
        np,
        parse_sdf,
        parse_types_file,
        plt,
        project_root,
        py3Dmol,
        re,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load checkpoint and data
    """)
    return


@app.cell
def _(project_root):
    # --- Checkpoint path (edit here) ---
    ckpt_dir = project_root / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.rglob("*.ckpt"))
        for p in ckpts:
            print(p.relative_to(project_root))
    else:
        print("No checkpoints/ directory found.")
        ckpts = sorted(project_root.rglob("*.ckpt"))
        for p in ckpts[-5:]:
            print(p.relative_to(project_root))
    return


@app.cell
def _(VQVAEModule, torch):
    CKPT_PATH = "/home/sakano/git/pocket-conditioned-ligand-gen/pocket-ligand-vqvae/42tfb6kx/checkpoints/vqvae-epoch=19-val/protein_recon=0.1512.ckpt"
    print(f"Loading: {CKPT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = VQVAEModule.load_from_checkpoint(str(CKPT_PATH), map_location=device)
    module.eval()
    module.to(device)

    protein_vqvae = module.protein_vqvae
    ligand_vqvae = module.ligand_vqvae
    print(f"Device: {device}")
    return device, ligand_vqvae, protein_vqvae


@app.cell
def _(project_root, torch):
    # Load normalization stats
    norm_stats = torch.load(
        project_root / "data" / "descriptor_cache" / "normalization_stats.pt",
        weights_only=True,
    )
    print("Norm stats loaded")
    print(f"  protein_mean: {norm_stats['protein_mean'].shape}")
    print(f"  ligand_mean:  {norm_stats['ligand_mean'].shape}")
    return (norm_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Pick a sample complex
    """)
    return


@app.cell
def _(parse_types_file, project_root):
    types_file = project_root / "data" / "types" / "cdonly_it2_tt_v1.3_0_test0.types"
    pairs = parse_types_file(types_file)
    crossdocked_dir = project_root / "data" / "CrossDocked2020"

    # Try a few pairs until we find one that works
    SAMPLE_IDX = 0
    for _try_idx in range(min(20, len(pairs))):
        rec_rel, lig_rel = pairs[SAMPLE_IDX + _try_idx]
        rec_path = crossdocked_dir / rec_rel
        lig_path = crossdocked_dir / lig_rel
        if rec_path.exists() and lig_path.exists():
            SAMPLE_IDX += _try_idx
            break

    print(f"Sample #{SAMPLE_IDX}")
    print(f"  Receptor: {rec_rel}")
    print(f"  Ligand:   {lig_rel}")
    return crossdocked_dir, lig_path, pairs, rec_path


@app.cell
def _(
    LigandDescriptor,
    PocketDescriptor,
    PocketExtractionConfig,
    TokenSequenceAssembler,
    device,
    extract_full_sequence,
    extract_pocket,
    lig_path,
    ligand_vqvae,
    norm_stats,
    np,
    parse_sdf,
    protein_vqvae,
    rec_path,
    torch,
):
    # --- Extract pocket and compute token sequence ---
    pocket_config = PocketExtractionConfig()
    molecules = parse_sdf(lig_path)
    mol = molecules[0]
    lig_coords = np.array(
        [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"], dtype=np.float32
    )

    pocket_result = extract_pocket(rec_path, lig_coords, pocket_config)
    backbone_coords, pocket_seq = pocket_result
    full_seq = extract_full_sequence(rec_path)

    # Compute protein descriptors → VQ-VAE tokens
    protein_desc_calc = PocketDescriptor()
    prot_desc_raw, prot_meta = protein_desc_calc.compute(backbone_coords)

    prot_t = torch.from_numpy(prot_desc_raw).to(device)
    prot_t = (prot_t - norm_stats["protein_mean"].to(device)) / norm_stats[
        "protein_std"
    ].to(device)

    with torch.no_grad():
        prot_code_indices = protein_vqvae.encode(prot_t).cpu().tolist()

    pocket_tokens = [
        f"{aa}_{code}" for aa, code in zip(pocket_seq, prot_code_indices, strict=True)
    ]

    # Compute ligand descriptors → VQ-VAE tokens
    ligand_desc_calc = LigandDescriptor()
    lig_desc_raw, elements, lig_meta = ligand_desc_calc.compute(
        mol["atoms"], mol["bonds"]
    )

    lig_t = torch.from_numpy(lig_desc_raw).to(device)
    lig_t = (lig_t - norm_stats["ligand_mean"].to(device)) / norm_stats[
        "ligand_std"
    ].to(device)

    with torch.no_grad():
        lig_code_indices = ligand_vqvae.encode(lig_t).cpu().tolist()

    ligand_tokens = [
        f"{elem}_{code}" for elem, code in zip(elements, lig_code_indices, strict=True)
    ]

    # Assemble full token sequence
    assembler = TokenSequenceAssembler()
    token_sequence = assembler.assemble(pocket_tokens, full_seq, ligand_tokens)

    print(f"Pocket residues: {len(pocket_seq)}")
    print(f"Ligand atoms:    {len(elements)}")
    print(f"Full sequence:   {len(full_seq)} AAs")
    return (
        elements,
        full_seq,
        lig_code_indices,
        lig_coords,
        lig_desc_raw,
        lig_meta,
        ligand_desc_calc,
        ligand_tokens,
        mol,
        pocket_config,
        pocket_seq,
        pocket_tokens,
        prot_code_indices,
        prot_desc_raw,
        prot_meta,
        protein_desc_calc,
        token_sequence,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 3D Viewer: Pocket (cartoon) + Ligand (stick)

    - **タンパク質**: 全体を薄い cartoon で表示し、ポケット残基をハイライト
    - **リガンド**: stick 表示（元素ごとに色分け）
    """)
    return


@app.cell
def _(lig_coords, np, pocket_config, rec_path):
    # --- Identify pocket residue IDs for highlighting ---
    from Bio.PDB import PDBParser as _PDBParser

    from src.tokenizers.protein import AA_3TO1, BACKBONE_ATOMS

    _parser = _PDBParser(QUIET=True)
    _structure = _parser.get_structure("rec", str(rec_path))
    _model = _structure[0]

    pocket_residue_ids = []  # (chain_id, resi)
    for chain in _model:
        for residue in chain:
            if residue.get_resname() not in AA_3TO1:
                continue
            if not all(a in residue for a in BACKBONE_ATOMS):
                continue
            ca = residue["CA"].get_vector().get_array()
            if (
                np.min(np.linalg.norm(lig_coords - ca, axis=1))
                <= pocket_config.distance_cutoff
            ):
                pocket_residue_ids.append((chain.id, residue.get_id()[1]))

    print(f"Pocket residue IDs: {len(pocket_residue_ids)}")
    return (pocket_residue_ids,)


@app.cell
def _(gzip, lig_path, pocket_residue_ids, py3Dmol, rec_path):
    print("=== 3D viewer cell STARTED ===")
    pdb_block = rec_path.read_text()

    with gzip.open(str(lig_path), "rt") as f:
        sdf_full = f.read()
    sdf_block = sdf_full.split("$$$$")[0] + "$$$$\n"

    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_block, "pdb")
    view.setStyle({"model": 0}, {"cartoon": {"color": "white", "opacity": 0.3}})
    pocket_resi = [r[1] for r in pocket_residue_ids]
    view.setStyle(
        {"model": 0, "resi": pocket_resi},
        {
            "cartoon": {"color": "skyblue", "opacity": 0.9},
            "stick": {"color": "skyblue", "radius": 0.1},
        },
    )
    view.addModel(sdf_block, "sdf")
    view.setStyle({"model": 1}, {"stick": {"colorscheme": "default", "radius": 0.2}})
    view.zoomTo({"model": 1})
    print(f"PDB: {len(pdb_block):,} bytes, SDF: {len(sdf_block):,} bytes")
    return (sdf_block,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Token Sequence

    上記の複合体をトークナイズした結果。各セクションを色分けして表示する。
    - <span style="color:#4a90d9">**青**: ポケット構造トークン `AA_code ...
    `</span>
    - <span style="color:#2ecc71">**緑**: タンパク質配列 `<s>MKTII...</s>`</span>
    - <span style="color:#e74c3c">**赤**: リガンド構造トークン `<l>elem_code ...</l>`</span>
    """)
    return


@app.cell
def _(full_seq, ligand_tokens, mo, pocket_tokens, re, token_sequence):
    # --- Render token sequence with color-coded sections ---
    def render_token_sequence(seq: str) -> str:
        """Render token sequence as color-coded HTML."""
        m = re.match("<p>(.*?)</p><s>(.*?)</s><l>(.*?)</l>", seq, re.DOTALL)
        if not m:
            return seq
        pocket_str, seq_str, ligand_str = m.groups()
        pocket_toks = pocket_str.split()
        ligand_toks = ligand_str.split()

        def _tok_span(tok: str, color: str) -> str:
            return f'<span style="background:{color};padding:1px 3px;margin:1px;border-radius:3px;font-size:12px">{tok}</span>'

        pocket_html = " ".join(_tok_span(t, "#d6eaf8") for t in pocket_toks)
        ligand_html = " ".join(_tok_span(t, "#fadbd8") for t in ligand_toks)
        max_display_len = 80
        seq_display = (
            seq_str
            if len(seq_str) <= max_display_len
            else seq_str[:40] + "..." + seq_str[-40:]
        )
        seq_html = f'<span style="background:#d5f5e3;padding:1px 3px;border-radius:3px;font-size:12px;word-break:break-all">{seq_display}</span>'
        return f'<div style="font-family:monospace;line-height:2.2"><b style="color:#4a90d9">&lt;p&gt;</b> {pocket_html} <b style="color:#4a90d9">&lt;/p&gt;</b><br><b style="color:#2ecc71">&lt;s&gt;</b> {seq_html} <b style="color:#2ecc71">&lt;/s&gt;</b><br><b style="color:#e74c3c">&lt;l&gt;</b> {ligand_html} <b style="color:#e74c3c">&lt;/l&gt;</b></div>'

    print(
        f"Total tokens: {len(pocket_tokens)} (pocket) + {len(full_seq)} (seq) + {len(ligand_tokens)} (ligand) = {len(pocket_tokens) + len(full_seq) + len(ligand_tokens)}"
    )
    mo.Html(render_token_sequence(token_sequence))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Codebook Index Distribution per Atom/Residue

    この複合体におけるトークン（codebook index）の分布を確認する。
    """)
    return


@app.cell
def _(elements, lig_code_indices, plt, pocket_seq, prot_code_indices):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 3.5))
    _axes[0].bar(
        range(len(prot_code_indices)), prot_code_indices, width=0.8, color="steelblue"
    )
    # Protein pocket: codebook index per residue
    _axes[0].set_xlabel("Residue index in pocket")
    _axes[0].set_ylabel("Codebook index")
    _axes[0].set_title(f"Pocket structure tokens ({len(prot_code_indices)} residues)")
    _axes[0].set_xticks(range(len(pocket_seq)))
    _axes[0].set_xticklabels(list(pocket_seq), fontsize=5, rotation=90)
    colors = {
        "C": "#666666",
        "N": "#3050F8",
        "O": "#FF0D0D",
        "S": "#FFFF30",
        "F": "#90E050",
        "Cl": "#1FF01F",
        "Br": "#A62929",
        "P": "#FF8000",
    }
    # Annotate amino acids on x-axis
    bar_colors = [colors.get(e, "#AAAAAA") for e in elements]
    _axes[1].bar(
        range(len(lig_code_indices)), lig_code_indices, width=0.8, color=bar_colors
    )
    _axes[1].set_xlabel("Atom index (BFS order)")
    # Ligand: codebook index per atom
    _axes[1].set_ylabel("Codebook index")
    _axes[1].set_title(f"Ligand structure tokens ({len(lig_code_indices)} atoms)")
    _axes[1].set_xticks(range(len(elements)))
    _axes[1].set_xticklabels(elements, fontsize=6, rotation=90)
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. VQ-VAE Reconstruction Accuracy Analysis

    VQ-VAE でエンコード→デコードした記述子から3D座標を復元し、元の構造とのズレを可視化する。

    ### 分析項目
    - 記述子レベルの比較（元 vs 復元）
    - 原子ごとの RMSD
    - 元構造と復元構造の重ね合わせ3Dビューア
    - 残基ごとのバックボーン RMSD（タンパク質）
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1 Ligand: Descriptor-level comparison
    """)
    return


@app.cell
def _(
    device,
    lig_code_indices,
    lig_desc_raw,
    ligand_vqvae,
    norm_stats,
    np,
    prot_code_indices,
    prot_desc_raw,
    protein_vqvae,
    torch,
):
    # --- Decode VQ-VAE codebook indices back to descriptors ---
    with torch.no_grad():
        lig_indices_t = torch.tensor(lig_code_indices, device=device)
        lig_recon_norm = ligand_vqvae.decode(lig_indices_t).cpu()

        prot_indices_t = torch.tensor(prot_code_indices, device=device)
        prot_recon_norm = protein_vqvae.decode(prot_indices_t).cpu()

    # Denormalize
    lig_recon_desc = (
        lig_recon_norm * norm_stats["ligand_std"] + norm_stats["ligand_mean"]
    ).numpy()
    prot_recon_desc = (
        prot_recon_norm * norm_stats["protein_std"] + norm_stats["protein_mean"]
    ).numpy()

    print("Ligand descriptors:")
    print(f"  Original shape:      {lig_desc_raw.shape}")
    print(f"  Reconstructed shape: {lig_recon_desc.shape}")
    print(f"  MSE (descriptor):    {np.mean((lig_desc_raw - lig_recon_desc) ** 2):.6f}")
    print(
        f"  MAE (descriptor):    {np.mean(np.abs(lig_desc_raw - lig_recon_desc)):.6f}"
    )
    print()
    print("Protein descriptors:")
    print(f"  Original shape:      {prot_desc_raw.shape}")
    print(f"  Reconstructed shape: {prot_recon_desc.shape}")
    print(
        f"  MSE (descriptor):    {np.mean((prot_desc_raw - prot_recon_desc) ** 2):.6f}"
    )
    print(
        f"  MAE (descriptor):    {np.mean(np.abs(prot_desc_raw - prot_recon_desc)):.6f}"
    )
    return lig_recon_desc, prot_recon_desc


@app.cell
def _(lig_desc_raw, lig_recon_desc, np, plt):
    # --- Per-dimension descriptor comparison (Ligand) ---
    dim_labels = ["bond_length", "bond_angle", "sin_dihedral", "cos_dihedral"]
    _fig, _axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for _d in range(4):
        _ax = _axes[_d]
        orig = lig_desc_raw[:, _d]
        recon = lig_recon_desc[:, _d]
        _ax.scatter(orig, recon, alpha=0.6, s=20, edgecolors="none")
        lims = [min(orig.min(), recon.min()), max(orig.max(), recon.max())]
        _ax.plot(lims, lims, "r--", linewidth=0.8)
        _ax.set_xlabel("Original")
        _ax.set_ylabel("Reconstructed")
        _ax.set_title(f"{dim_labels[_d]}\nMAE={np.mean(np.abs(orig - recon)):.4f}")
        _ax.set_aspect("equal")
    _fig.suptitle("Ligand: Original vs Reconstructed Descriptors", fontsize=13)
    _fig.tight_layout()
    plt.show()
    return (dim_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 Ligand: 3D coordinate reconstruction & per-atom error
    """)
    return


@app.cell
def _(LigandDescriptor, elements, lig_desc_raw, lig_meta, lig_recon_desc, np):
    # --- Reconstruct 3D coordinates from original and VQ-VAE descriptors ---
    # Original coordinates (from descriptors, not raw SDF, to ensure same frame)
    lig_coords_orig = LigandDescriptor.descriptor_to_coords(lig_desc_raw, lig_meta)
    lig_coords_recon = LigandDescriptor.descriptor_to_coords(lig_recon_desc, lig_meta)

    # Per-atom distance error
    per_atom_dist = np.linalg.norm(lig_coords_orig - lig_coords_recon, axis=1)

    # DFS order mapping: elements are in DFS order, coords are in original order
    # lig_meta["order"] maps DFS position -> original atom index
    dfs_order = lig_meta["order"]
    elements_orig_order = [""] * len(elements)
    for dfs_pos, orig_idx in enumerate(dfs_order):
        elements_orig_order[orig_idx] = elements[dfs_pos]

    print(f"Ligand RMSD (all atoms): {np.sqrt(np.mean(per_atom_dist**2)):.4f} Å")
    print(f"Mean per-atom error:     {np.mean(per_atom_dist):.4f} Å")
    print(
        f"Max per-atom error:      {np.max(per_atom_dist):.4f} Å (atom {np.argmax(per_atom_dist)}, {elements_orig_order[np.argmax(per_atom_dist)]})"
    )
    print(f"Median per-atom error:   {np.median(per_atom_dist):.4f} Å")
    return dfs_order, elements_orig_order, lig_coords_recon, per_atom_dist


@app.cell
def _(elements_orig_order, np, per_atom_dist, plt):
    # --- Per-atom error bar chart ---
    _fig, _ax = plt.subplots(figsize=(max(8, len(per_atom_dist) * 0.4), 4))
    atom_colors = {
        "C": "#666666",
        "N": "#3050F8",
        "O": "#FF0D0D",
        "S": "#FFFF30",
        "F": "#90E050",
        "Cl": "#1FF01F",
        "Br": "#A62929",
        "P": "#FF8000",
    }
    bar_c = [atom_colors.get(e, "#AAAAAA") for e in elements_orig_order]
    _ax.bar(range(len(per_atom_dist)), per_atom_dist, color=bar_c, width=0.8)
    _ax.axhline(
        y=np.mean(per_atom_dist),
        color="red",
        linestyle="--",
        linewidth=0.8,
        label=f"Mean: {np.mean(per_atom_dist):.3f} Å",
    )
    _ax.set_xlabel("Atom index (original order)")
    _ax.set_ylabel("Distance error (Å)")
    _ax.set_title("Ligand: Per-atom reconstruction error")
    _ax.set_xticks(range(len(elements_orig_order)))
    _ax.set_xticklabels(
        [f"{e}\n{i}" for i, e in enumerate(elements_orig_order)], fontsize=6
    )
    _ax.legend()
    _fig.tight_layout()
    plt.show()
    return (atom_colors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 Ligand: 3D overlay (Original vs Reconstructed)

    - **元構造**: stick (通常の元素カラー)
    - **復元構造**: stick (オレンジ)
    - 対応する原子間を赤い線で接続し、ズレの大きさを可視化
    """)
    return


@app.cell
def _(
    elements_orig_order,
    lig_coords_recon,
    mo,
    mol,
    per_atom_dist,
    py3Dmol,
    sdf_block,
):
    def coords_to_xyz_block(coords: "numpy.ndarray", elements_list: list[str]) -> str:
        """Convert coordinates and elements to XYZ format string."""
        lines = [str(len(coords)), "reconstructed"]
        for (x, y, z), elem in zip(coords, elements_list, strict=True):
            lines.append(f"{elem}  {x:.6f}  {y:.6f}  {z:.6f}")
        return "\n".join(lines)

    view2 = py3Dmol.view(width=800, height=500)
    view2.addModel(sdf_block, "sdf")
    view2.setStyle({"model": 0}, {"stick": {"colorscheme": "default", "radius": 0.15}})
    xyz_block = coords_to_xyz_block(lig_coords_recon, elements_orig_order)
    # Original ligand from SDF
    view2.addModel(xyz_block, "xyz")
    view2.setStyle({"model": 1}, {"stick": {"color": "orange", "radius": 0.15}})
    orig_heavy_atoms = [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"]
    # Reconstructed ligand as XYZ
    for i in range(len(lig_coords_recon)):
        ox, oy, oz = orig_heavy_atoms[i]
        rx, ry, rz = lig_coords_recon[i]
        dist = per_atom_dist[i]
        # Draw lines between corresponding atoms to show displacement
        # Use original SDF coords (heavy atoms only, in file order)
        if dist > 0.1:  # noqa: PLR2004
            intensity = min(1.0, dist / 2.0)
            _color = f"rgb({int(255 * intensity)}, {int(50 * (1 - intensity))}, {int(50 * (1 - intensity))})"
            view2.addLine(
                {
                    "start": {"x": float(ox), "y": float(oy), "z": float(oz)},
                    "end": {"x": float(rx), "y": float(ry), "z": float(rz)},
                    "color": _color,
                    "dashed": True,
                }
            )
    view2.zoomTo({"model": 0})
    print("Gray/colored sticks = original, Orange sticks = VQ-VAE reconstructed")
    print("Dashed lines = atom displacement (redder = larger error)")
    mo.iframe(view2._make_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.4 Ligand: Error vs. graph distance from root

    自己回帰的な記述子（Z-matrix）では、ルート原子から遠い原子ほど誤差が蓄積しやすい。
    DFS 順序における位置と復元誤差の関係を確認する。
    """)
    return


@app.cell
def _(atom_colors, dfs_order, elements, np, per_atom_dist, plt):
    # Per-atom error in DFS traversal order
    per_atom_dist_dfs = np.array([per_atom_dist[orig_idx] for orig_idx in dfs_order])
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))
    _axes[0].bar(
        range(len(per_atom_dist_dfs)),
        per_atom_dist_dfs,
        color=[atom_colors.get(e, "#AAAAAA") for e in elements],
        width=0.8,
    )
    _axes[0].set_xlabel("DFS traversal position")
    # Left: error vs DFS position
    _axes[0].set_ylabel("Distance error (Å)")
    _axes[0].set_title("Error vs DFS order (later = farther from root)")
    _axes[0].set_xticks(range(len(elements)))
    _axes[0].set_xticklabels([f"{e}" for e in elements], fontsize=6, rotation=90)
    sorted_errors = np.sort(per_atom_dist)
    _axes[1].step(
        sorted_errors,
        np.arange(1, len(sorted_errors) + 1) / len(sorted_errors),
        where="post",
    )
    _axes[1].set_xlabel("Distance error (Å)")
    _axes[1].set_ylabel("Cumulative fraction of atoms")
    _axes[1].set_title("CDF of per-atom reconstruction error")
    _axes[1].axvline(
        x=0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="0.5 Å"
    )
    _axes[1].axvline(
        x=1.0, color="orange", linestyle="--", linewidth=0.8, alpha=0.7, label="1.0 Å"
    )
    _axes[1].axvline(
        x=2.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="2.0 Å"
    )
    # Right: cumulative error distribution
    _axes[1].legend()
    _axes[1].grid(visible=True, alpha=0.3)
    _fig.tight_layout()
    plt.show()
    for threshold in [0.25, 0.5, 1.0, 2.0]:
        frac = np.mean(per_atom_dist < threshold)
        # Print fraction of atoms within thresholds
        print(
            f"  Atoms within {threshold:.2f} Å: {frac * 100:.1f}% ({int(frac * len(per_atom_dist))}/{len(per_atom_dist)})"
        )
    return (per_atom_dist_dfs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.5 Protein pocket: Backbone reconstruction error
    """)
    return


@app.cell
def _(PocketDescriptor, np, prot_desc_raw, prot_meta, prot_recon_desc):
    # --- Reconstruct protein backbone from VQ-VAE descriptors ---
    prot_backbone_orig = PocketDescriptor.descriptor_to_backbone_coords(
        prot_desc_raw, prot_meta
    )
    prot_backbone_recon = PocketDescriptor.descriptor_to_backbone_coords(
        prot_recon_desc, prot_meta
    )
    per_res_diff = prot_backbone_orig - prot_backbone_recon
    per_res_rmsd = np.sqrt(np.mean(per_res_diff**2, axis=(1, 2)))
    atom_names = ["N", "CA", "C"]
    for _a_idx, _a_name in enumerate(atom_names):
        _diff = np.linalg.norm(
            prot_backbone_orig[:, _a_idx] - prot_backbone_recon[:, _a_idx], axis=1
        )
        # Per-residue RMSD over (N, CA, C)
        print(
            f"  {_a_name}: mean error = {np.mean(_diff):.4f} Å, max = {np.max(_diff):.4f} Å"
        )  # (L, 3, 3)
    print(f"\nOverall backbone RMSD: {np.sqrt(np.mean(per_res_diff**2)):.4f} Å")  # (L,)
    # Per-atom-type error
    print(
        f"Per-residue RMSD: mean = {np.mean(per_res_rmsd):.4f} Å, max = {np.max(per_res_rmsd):.4f} Å"
    )
    return atom_names, per_res_rmsd, prot_backbone_orig, prot_backbone_recon


@app.cell
def _(
    atom_names,
    np,
    per_res_rmsd,
    plt,
    pocket_seq,
    prot_backbone_orig,
    prot_backbone_recon,
):
    # --- Per-residue backbone RMSD ---
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))
    bar_colors_res = plt.cm.RdYlGn_r(per_res_rmsd / max(per_res_rmsd.max(), 0.01))
    # Left: per-residue RMSD
    _axes[0].bar(
        range(len(per_res_rmsd)), per_res_rmsd, color=bar_colors_res, width=0.8
    )
    _axes[0].set_xlabel("Pocket residue index")
    _axes[0].set_ylabel("Backbone RMSD (Å)")
    _axes[0].set_title(f"Per-residue backbone RMSD ({len(per_res_rmsd)} residues)")
    _axes[0].set_xticks(range(len(pocket_seq)))
    _axes[0].set_xticklabels(list(pocket_seq), fontsize=5, rotation=90)
    _axes[0].axhline(
        y=np.mean(per_res_rmsd),
        color="red",
        linestyle="--",
        linewidth=0.8,
        label=f"Mean: {np.mean(per_res_rmsd):.3f} Å",
    )
    _axes[0].legend()
    x_pos = np.arange(len(pocket_seq))
    width = 0.25
    for _a_idx, (_a_name, _color) in enumerate(
        zip(atom_names, ["#3050F8", "#666666", "#FF0D0D"], strict=True)
    ):
        _diff = np.linalg.norm(
            prot_backbone_orig[:, _a_idx] - prot_backbone_recon[:, _a_idx], axis=1
        )
        _axes[1].bar(
            x_pos + _a_idx * width,
            _diff,
            width=width,
            color=_color,
            alpha=0.7,
            label=_a_name,
        )
    _axes[1].set_xlabel("Pocket residue index")
    _axes[1].set_ylabel("Distance error (Å)")
    _axes[1].set_title("Per-atom-type backbone error")
    # Right: per-atom-type comparison
    _axes[1].set_xticks(x_pos + width)
    _axes[1].set_xticklabels(list(pocket_seq), fontsize=5, rotation=90)
    _axes[1].legend()
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.6 Ligand: Descriptor error breakdown

    各記述子次元（bond_length, bond_angle, sin/cos_dihedral）ごとの誤差を原子ごとに分解し、
    どの次元の誤差が3D座標のズレに最も寄与しているかを分析する。
    """)
    return


@app.cell
def _(
    atom_colors,
    dim_labels,
    elements,
    lig_desc_raw,
    lig_recon_desc,
    np,
    per_atom_dist_dfs,
    plt,
):
    desc_error = np.abs(lig_desc_raw - lig_recon_desc)  # (N, 4)
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 8))
    for _d in range(4):
        _ax = _axes[_d // 2][_d % 2]
        bar_c_dfs = [atom_colors.get(e, "#AAAAAA") for e in elements]
        _ax.bar(range(len(desc_error)), desc_error[:, _d], color=bar_c_dfs, width=0.8)
        _ax.set_xlabel("Atom index (DFS order)")
        _ax.set_ylabel("Absolute error")
        _ax.set_title(f"{dim_labels[_d]} (MAE={np.mean(desc_error[:, _d]):.4f})")
        _ax.set_xticks(range(len(elements)))
        _ax.set_xticklabels([f"{e}" for e in elements], fontsize=6, rotation=90)
    _fig.suptitle("Ligand: Per-atom descriptor error by dimension", fontsize=13)
    _fig.tight_layout()
    plt.show()
    print("\nCorrelation between descriptor MAE and 3D error:")
    for _d in range(4):
        corr = np.corrcoef(desc_error[:, _d], per_atom_dist_dfs)[0, 1]
        # Correlation between descriptor error and 3D coordinate error
        print(f"  {dim_labels[_d]}: r = {corr:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.7 Multiple samples: Reconstruction error distribution

    1サンプルだけでは偏りがあるので、複数の複合体で RMSD 分布を確認する。
    """)
    return


@app.cell
def _(
    LigandDescriptor,
    PocketDescriptor,
    crossdocked_dir,
    device,
    extract_pocket,
    ligand_desc_calc,
    ligand_vqvae,
    norm_stats,
    np,
    pairs,
    parse_sdf,
    pocket_config,
    protein_desc_calc,
    protein_vqvae,
    torch,
):
    N_SAMPLES = 50

    lig_rmsds = []
    prot_rmsds = []
    lig_sizes = []
    failed = 0

    for idx in range(min(N_SAMPLES, len(pairs))):
        rec_r, lig_r = pairs[idx]
        rp = crossdocked_dir / rec_r
        lp = crossdocked_dir / lig_r
        if not (rp.exists() and lp.exists()):
            continue

        try:
            mols = parse_sdf(lp)
            m = mols[0]
            lc = np.array(
                [(a[1], a[2], a[3]) for a in m["atoms"] if a[0] != "H"],
                dtype=np.float32,
            )

            bc, _ps = extract_pocket(rp, lc, pocket_config)

            # Protein
            pd_raw, pm = protein_desc_calc.compute(bc)
            pt = torch.from_numpy(pd_raw).to(device)
            pt = (pt - norm_stats["protein_mean"].to(device)) / norm_stats[
                "protein_std"
            ].to(device)
            with torch.no_grad():
                pi = protein_vqvae.encode(pt)
                pr = protein_vqvae.decode(pi).cpu()
            pr_desc = (
                pr * norm_stats["protein_std"] + norm_stats["protein_mean"]
            ).numpy()
            bb_orig = PocketDescriptor.descriptor_to_backbone_coords(pd_raw, pm)
            bb_recon = PocketDescriptor.descriptor_to_backbone_coords(pr_desc, pm)
            prot_rmsds.append(np.sqrt(np.mean((bb_orig - bb_recon) ** 2)))

            # Ligand
            ld_raw, elems, lm = ligand_desc_calc.compute(m["atoms"], m["bonds"])
            lt = torch.from_numpy(ld_raw).to(device)
            lt = (lt - norm_stats["ligand_mean"].to(device)) / norm_stats[
                "ligand_std"
            ].to(device)
            with torch.no_grad():
                li = ligand_vqvae.encode(lt)
                lr = ligand_vqvae.decode(li).cpu()
            lr_desc = (
                lr * norm_stats["ligand_std"] + norm_stats["ligand_mean"]
            ).numpy()
            co = LigandDescriptor.descriptor_to_coords(ld_raw, lm)
            cr = LigandDescriptor.descriptor_to_coords(lr_desc, lm)
            lig_rmsds.append(np.sqrt(np.mean((co - cr) ** 2)))
            lig_sizes.append(len(elems))
        except Exception:  # noqa: BLE001 — skip malformed complexes during batch eval
            failed += 1
            continue

    print(f"Computed {len(lig_rmsds)} samples ({failed} failed)")
    print(
        f"\nLigand RMSD:  mean={np.mean(lig_rmsds):.3f}, median={np.median(lig_rmsds):.3f}, max={np.max(lig_rmsds):.3f} Å"
    )
    print(
        f"Protein RMSD: mean={np.mean(prot_rmsds):.3f}, median={np.median(prot_rmsds):.3f}, max={np.max(prot_rmsds):.3f} Å"
    )
    return lig_rmsds, lig_sizes, prot_rmsds


@app.cell
def _(lig_rmsds, lig_sizes, np, plt, prot_rmsds):
    _fig, _axes = plt.subplots(1, 3, figsize=(16, 4))
    _axes[0].hist(lig_rmsds, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    # Ligand RMSD distribution
    _axes[0].axvline(
        np.median(lig_rmsds),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(lig_rmsds):.2f} Å",
    )
    _axes[0].set_xlabel("RMSD (Å)")
    _axes[0].set_ylabel("Count")
    _axes[0].set_title(f"Ligand reconstruction RMSD (n={len(lig_rmsds)})")
    _axes[0].legend()
    _axes[1].hist(prot_rmsds, bins=20, color="darkorange", edgecolor="white", alpha=0.8)
    _axes[1].axvline(
        np.median(prot_rmsds),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(prot_rmsds):.2f} Å",
    )
    _axes[1].set_xlabel("RMSD (Å)")
    _axes[1].set_ylabel("Count")
    _axes[1].set_title(f"Protein backbone RMSD (n={len(prot_rmsds)})")
    _axes[1].legend()
    _axes[2].scatter(lig_sizes, lig_rmsds, alpha=0.6, s=30, edgecolors="none")
    # Protein RMSD distribution
    _axes[2].set_xlabel("Number of heavy atoms")
    _axes[2].set_ylabel("RMSD (Å)")
    _axes[2].set_title("Ligand RMSD vs molecule size")
    z = np.polyfit(lig_sizes, lig_rmsds, 1)
    x_line = np.linspace(min(lig_sizes), max(lig_sizes), 100)
    _axes[2].plot(
        x_line,
        np.polyval(z, x_line),
        "r--",
        linewidth=0.8,
        label=f"slope={z[0]:.3f} Å/atom",
    )
    _axes[2].legend()
    _fig.tight_layout()
    # Ligand RMSD vs molecule size
    # Add trend line
    plt.show()
    return


if __name__ == "__main__":
    app.run()
