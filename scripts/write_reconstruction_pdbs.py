"""Dump original and VQ-VAE-reconstructed pocket+ligand complexes as PDBs.

For each sampled complex from the cdonly fold0 test split, writes three files:
  {tag}_orig.pdb         — full receptor (every chain/atom from the source
                           PDB) plus the original ligand heavy atoms as
                           HETATM, so the ligand sits in its native pose.
  {tag}_orig_pocket.pdb  — pocket backbone (N/CA/C of selected residues)
                           plus the original ligand, matching the residue
                           subset that the VQ-VAE actually sees.
  {tag}_recon.pdb        — same pocket residues + ligand after passing
                           through the protein and ligand VQ-VAEs
                           (descriptor → encode → decode → NeRF-style 3D
                           reconstruction).

`_orig_pocket.pdb` and `_recon.pdb` share residue numbering and ligand atom
names so loading them together (PyMOL: ``load A.pdb``, ``load B.pdb``) lines
them up directly for visual diffing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CrossDockedConfig,
    PocketExtractionConfig,
    VQVAETrainingConfig,
)
from src.data.descriptors import ComplexDescriptorDataModule
from src.model.vqvae_module import VQVAEModule
from src.tokenizers.ligand import LigandDescriptor, parse_sdf
from src.tokenizers.protein import (
    AA_3TO1,
    BACKBONE_ATOMS,
    BackboneZMatrixDescriptor,
    _compute_canonical_frame,
    extract_pocket,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

AA_1TO3: dict[str, str] = {v: k for k, v in AA_3TO1.items()}


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """Per-atom RMSD after optimal rigid-body alignment of q onto p."""
    p_c = p - p.mean(axis=0)
    q_c = q - q.mean(axis=0)
    h = q_c.T @ p_c
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    q_aligned = q_c @ rot.T
    return float(np.sqrt(np.mean(np.sum((p_c - q_aligned) ** 2, axis=-1))))


def per_atom_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((p - q) ** 2, axis=-1))))


METRIC_FN = {
    "lig-kabsch": lambda r: kabsch_rmsd(
        r["ligand_coords_orig"], r["ligand_coords_recon"]
    ),
    "lig-per-atom": lambda r: per_atom_rmsd(
        r["ligand_coords_orig"], r["ligand_coords_recon"]
    ),
    "prot-per-atom": lambda r: per_atom_rmsd(
        r["backbone_orig"].reshape(-1, 3), r["backbone_recon"].reshape(-1, 3)
    ),
    "joint-kabsch": lambda r: kabsch_rmsd(
        np.vstack([r["backbone_orig"].reshape(-1, 3), r["ligand_coords_orig"]]),
        np.vstack([r["backbone_recon"].reshape(-1, 3), r["ligand_coords_recon"]]),
    ),
}


def _fmt_atom(
    record: str,
    serial: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    res_seq: int,
    xyz: np.ndarray,
    element: str,
) -> str:
    """Format a single PDB ATOM/HETATM line per the column spec."""
    name_field = f" {atom_name:<3s}" if len(atom_name) < 4 else atom_name[:4]
    return (
        f"{record:<6s}{serial:>5d} {name_field}"
        f" {res_name:>3s} {chain_id:1s}{res_seq:>4d}    "
        f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}"
        f"{1.00:>6.2f}{0.00:>6.2f}          {element:>2s}\n"
    )


def _conect_lines(
    bonds: list[tuple[int, int]], *, start_serial: int
) -> list[str]:
    """Emit CONECT records so viewers don't have to guess bonds from distance."""
    lines: list[str] = []
    for a, b in bonds:
        # PDB CONECT is 1-indexed by atom serial.
        lines.append(f"CONECT{start_serial + a:>5d}{start_serial + b:>5d}\n")
    return lines


def _ligand_lines(
    ligand_elements: list[str],
    ligand_coords: np.ndarray,
    *,
    start_serial: int,
    bonds: list[tuple[int, int]] | None = None,
) -> list[str]:
    """Format the ligand heavy atoms as HETATM (+ optional CONECT) records."""
    lines: list[str] = []
    serial = start_serial
    for k, (elem, xyz) in enumerate(
        zip(ligand_elements, ligand_coords, strict=True)
    ):
        atom_name = f"{elem}{k + 1}"[:4]
        lines.append(
            _fmt_atom("HETATM", serial, atom_name, "LIG", "L", 1, xyz, elem)
        )
        serial += 1
    if bonds:
        lines.extend(_conect_lines(bonds, start_serial=start_serial))
    return lines


def write_full_protein_pdb(
    out_path: Path,
    receptor_pdb_path: Path,
    ligand_elements: list[str],
    ligand_coords: np.ndarray,
    ligand_bonds: list[tuple[int, int]] | None = None,
) -> None:
    """Copy the source receptor PDB verbatim and append the ligand HETATMs."""
    raw = receptor_pdb_path.read_text().splitlines(keepends=True)
    # Drop trailing END/ENDMDL so we can append ligand records before it.
    keep: list[str] = []
    for line in raw:
        head = line[:6].strip()
        if head in {"END", "ENDMDL", "MASTER"}:
            continue
        keep.append(line)
    last_serial = 0
    for line in keep:
        if line.startswith(("ATOM", "HETATM")):
            try:
                last_serial = max(last_serial, int(line[6:11]))
            except ValueError:
                pass
    keep.append("TER\n")
    keep.extend(
        _ligand_lines(
            ligand_elements,
            ligand_coords,
            start_serial=last_serial + 1,
            bonds=ligand_bonds,
        )
    )
    keep.append("END\n")
    out_path.write_text("".join(keep))


def write_complex_pdb(
    out_path: Path,
    backbone_coords: np.ndarray,
    pocket_seq: str,
    residue_ids: list[tuple[str, int]],
    ligand_elements: list[str],
    ligand_coords: np.ndarray,
    ligand_bonds: list[tuple[int, int]] | None = None,
) -> None:
    """Write pocket backbone (ATOM) + ligand (HETATM) as a single PDB."""
    lines: list[str] = []
    serial = 1
    last_chain: str | None = None
    for i, (res_one, (chain_id, res_seq)) in enumerate(
        zip(pocket_seq, residue_ids, strict=True)
    ):
        if last_chain is not None and chain_id != last_chain:
            lines.append("TER\n")
        res_three = AA_1TO3.get(res_one, "UNK")
        for j, atom_name in enumerate(BACKBONE_ATOMS):
            element = atom_name[0]
            lines.append(
                _fmt_atom(
                    "ATOM",
                    serial,
                    atom_name,
                    res_three,
                    chain_id,
                    res_seq,
                    backbone_coords[i, j],
                    element,
                )
            )
            serial += 1
        last_chain = chain_id
    lines.append("TER\n")
    lines.extend(
        _ligand_lines(
            ligand_elements,
            ligand_coords,
            start_serial=serial,
            bonds=ligand_bonds,
        )
    )
    lines.append("END\n")
    out_path.write_text("".join(lines))


@torch.no_grad()
def reconstruct_one(
    rec_path: Path,
    lig_path: Path,
    *,
    pocket_config: PocketExtractionConfig,
    protein_desc_calc: BackboneZMatrixDescriptor,
    ligand_desc_calc: LigandDescriptor,
    protein_vqvae,
    ligand_vqvae,
    norm_stats: dict[str, torch.Tensor],
    device: torch.device,
) -> dict | None:
    """Run one complex through the encode→decode pipeline; return both poses."""
    molecules = parse_sdf(lig_path)
    if not molecules:
        return None
    mol = molecules[0]
    if not mol["atoms"]:
        return None
    # Pocket extraction needs heavy-atom coords for the distance cutoff.
    heavy_for_pocket = np.array(
        [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"],
        dtype=np.float32,
    )
    if len(heavy_for_pocket) == 0:
        return None
    pocket_result = extract_pocket(
        rec_path, heavy_for_pocket, pocket_config
    )
    if pocket_result is None:
        return None
    backbone_coords_orig, pocket_seq, residue_ids = pocket_result

    ca_coords = backbone_coords_orig[:, 1].astype(np.float64)
    centroid, rotation = _compute_canonical_frame(ca_coords)
    pocket_frame = (centroid, rotation)

    prot_desc, prot_meta = protein_desc_calc.compute(
        backbone_coords_orig, residue_ids, pocket_frame=pocket_frame
    )
    prot_t = torch.from_numpy(prot_desc).to(device)
    prot_norm = (prot_t - norm_stats["protein_mean"]) / norm_stats["protein_std"]
    pi = protein_vqvae.encode(prot_norm)
    prot_recon_norm = protein_vqvae.decode(pi)
    prot_recon_desc = (
        prot_recon_norm * norm_stats["protein_std"] + norm_stats["protein_mean"]
    ).cpu().numpy()
    backbone_recon = BackboneZMatrixDescriptor.descriptor_to_backbone_coords(
        prot_recon_desc, prot_meta
    )

    lig_desc, lig_elements, lig_meta = ligand_desc_calc.compute(
        mol["atoms"], mol["bonds"], pocket_frame=pocket_frame
    )
    if len(lig_desc) == 0:
        return None
    lig_t = torch.from_numpy(lig_desc).to(device)
    lig_norm = (lig_t - norm_stats["ligand_mean"]) / norm_stats["ligand_std"]
    li = ligand_vqvae.encode(lig_norm)
    lig_recon_norm = ligand_vqvae.decode(li)
    lig_recon_desc = (
        lig_recon_norm * norm_stats["ligand_std"] + norm_stats["ligand_mean"]
    ).cpu().numpy()
    lig_coords_recon = LigandDescriptor.descriptor_to_coords(
        lig_recon_desc, lig_meta, pocket_frame=pocket_frame
    )

    # `descriptor_to_coords` returns coords in **raw SDF order** (indexed by
    # raw atom index), while `lig_elements` is in DFS order.  Reorder both
    # the original and reconstructed coords into DFS order so they line up
    # element-by-element, then drop hydrogens from all three for cleaner PDB.
    raw_coords = np.array(
        [(a[1], a[2], a[3]) for a in mol["atoms"]], dtype=np.float64
    )
    dfs_order = lig_meta["order"]
    lig_coords_orig_dfs = raw_coords[dfs_order]
    lig_coords_recon_dfs = lig_coords_recon.astype(np.float64)[dfs_order]
    keep = [i for i, e in enumerate(lig_elements) if e != "H"]
    # Remap SDF bonds (raw atom indices) → final ligand atom indices in the
    # heavy-only DFS order, dropping any bond that touched a hydrogen.
    raw_to_final: dict[int, int] = {
        dfs_order[dfs_pos]: final_pos
        for final_pos, dfs_pos in enumerate(keep)
    }
    bonds_remapped: list[tuple[int, int]] = []
    for a, b, *_ in mol["bonds"]:
        if a in raw_to_final and b in raw_to_final:
            bonds_remapped.append((raw_to_final[a], raw_to_final[b]))
    return {
        "pocket_seq": pocket_seq,
        "residue_ids": residue_ids,
        "backbone_orig": backbone_coords_orig.astype(np.float64),
        "backbone_recon": backbone_recon.astype(np.float64),
        "ligand_elements": [lig_elements[i] for i in keep],
        "ligand_coords_orig": lig_coords_orig_dfs[keep],
        "ligand_coords_recon": lig_coords_recon_dfs[keep],
        "ligand_bonds": bonds_remapped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path(
            "pocket-ligand-vqvae/29sx3ezx/checkpoints/"
            "vqvae-epoch=98-val/protein_recon=0.1197.ckpt"
        ),
        help="VQ-VAE checkpoint (relative paths resolve from project root).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "descriptor_cache_v3",
        help="Descriptor cache (only used to load normalization stats).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reconstruction_pdbs",
    )
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Cap on candidate complexes to scan (default: 5 × n_samples for "
        "random mode, 100 × n_samples for worst-* modes).",
    )
    parser.add_argument(
        "--mode",
        choices=["random", *(f"worst-{m}" for m in METRIC_FN)],
        default="random",
        help="random: first n_samples valid candidates. worst-*: scan a larger "
        "pool, rank by the chosen RMSD metric (descending), and write the "
        "top n_samples worst examples.",
    )
    args = parser.parse_args()

    ckpt_path = args.ckpt if args.ckpt.is_absolute() else PROJECT_ROOT / args.ckpt
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading checkpoint: %s", ckpt_path)
    module = VQVAEModule.load_from_checkpoint(str(ckpt_path), map_location=device)
    module.eval().to(device)
    protein_vqvae = module.protein_vqvae
    ligand_vqvae = module.ligand_vqvae

    config = VQVAETrainingConfig()
    data_config = CrossDockedConfig(data_dir=PROJECT_ROOT / "data")
    dm = ComplexDescriptorDataModule(config, data_config)
    if args.cache_dir is not None:
        dm.cache_dir = args.cache_dir
    dm.setup()
    norm_stats = {k: v.to(device) for k, v in dm.norm_stats.items()}

    hub_cache_dir = PROJECT_ROOT / "data" / "hub_cache"
    manifest_df = pq.read_table(hub_cache_dir / "repo" / "manifest.parquet").to_pandas()
    test_df = manifest_df[
        (manifest_df["source_type"] == "cdonly")
        & (manifest_df["cdonly_fold0"] == "test")
    ].reset_index(drop=True)
    logger.info("cdonly fold0 test pool: %d entries", len(test_df))

    receptor_dir = hub_cache_dir / "receptors"
    ligand_dir = hub_cache_dir / "ligands"

    rng = np.random.default_rng(args.seed)
    is_worst = args.mode != "random"
    metric_key = args.mode.removeprefix("worst-") if is_worst else None
    default_attempts = args.n_samples * (100 if is_worst else 5)
    max_attempts = args.max_attempts or default_attempts
    sample_indices = rng.choice(
        len(test_df), size=min(max_attempts, len(test_df)), replace=False
    )

    pocket_config = PocketExtractionConfig()
    protein_desc_calc = BackboneZMatrixDescriptor()
    ligand_desc_calc = LigandDescriptor()

    # Scan candidates; in random mode stop after n_samples valid hits, in
    # worst mode keep going through the whole pool to rank.
    candidates: list[tuple[float, int, dict]] = []
    for cand_idx in sample_indices:
        if not is_worst and len(candidates) >= args.n_samples:
            break
        row = test_df.iloc[int(cand_idx)]
        rec_path = receptor_dir / f"{row['complex_dir']}/{row['receptor_pdb']}"
        lig_path = ligand_dir / f"{int(row['pair_idx']):07d}.sdf.gz"
        if not rec_path.exists() or not lig_path.exists():
            continue
        try:
            result = reconstruct_one(
                rec_path,
                lig_path,
                pocket_config=pocket_config,
                protein_desc_calc=protein_desc_calc,
                ligand_desc_calc=ligand_desc_calc,
                protein_vqvae=protein_vqvae,
                ligand_vqvae=ligand_vqvae,
                norm_stats=norm_stats,
                device=device,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("skip %s: %s", row["complex_dir"], e)
            continue
        if result is None:
            continue
        score = METRIC_FN[metric_key](result) if metric_key else 0.0
        candidates.append((score, int(cand_idx), result))

    if is_worst:
        candidates.sort(key=lambda x: -x[0])
        logger.info(
            "Scanned %d valid candidates, taking top %d by %s RMSD",
            len(candidates), args.n_samples, metric_key,
        )
    selected = candidates[: args.n_samples]

    n_done = 0
    for score, cand_idx, result in selected:
        row = test_df.iloc[cand_idx]
        rec_path = receptor_dir / f"{row['complex_dir']}/{row['receptor_pdb']}"
        score_tag = f"_{metric_key}{score:.2f}" if is_worst else ""
        tag = f"{n_done:04d}_pair{int(row['pair_idx']):07d}{score_tag}"
        orig_full_pdb = args.out_dir / f"{tag}_orig.pdb"
        orig_pocket_pdb = args.out_dir / f"{tag}_orig_pocket.pdb"
        recon_pdb = args.out_dir / f"{tag}_recon.pdb"
        write_full_protein_pdb(
            orig_full_pdb,
            rec_path,
            result["ligand_elements"],
            result["ligand_coords_orig"],
            ligand_bonds=result["ligand_bonds"],
        )
        write_complex_pdb(
            orig_pocket_pdb,
            result["backbone_orig"],
            result["pocket_seq"],
            result["residue_ids"],
            result["ligand_elements"],
            result["ligand_coords_orig"],
            ligand_bonds=result["ligand_bonds"],
        )
        write_complex_pdb(
            recon_pdb,
            result["backbone_recon"],
            result["pocket_seq"],
            result["residue_ids"],
            result["ligand_elements"],
            result["ligand_coords_recon"],
            ligand_bonds=result["ligand_bonds"],
        )
        score_str = f" {metric_key}={score:.2f} Å" if is_worst else ""
        logger.info(
            "[%d/%d] %s%s — pocket %d res, ligand %d atoms → %s, %s, %s",
            n_done + 1,
            args.n_samples,
            row["complex_dir"],
            score_str,
            len(result["pocket_seq"]),
            len(result["ligand_elements"]),
            orig_full_pdb.name,
            orig_pocket_pdb.name,
            recon_pdb.name,
        )
        n_done += 1

    logger.info("Wrote %d complex(es) to %s", n_done, args.out_dir)


if __name__ == "__main__":
    main()
