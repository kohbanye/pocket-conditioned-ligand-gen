"""Drive the own model's ``reconstruct_one`` on explicit (receptor, ligand) pairs.

Run by the **own model's** interpreter (it imports that repo's code), not the
bench env. Loads the upstream ``write_reconstruction_pdbs.py`` by path to reuse
its tested ``reconstruct_one`` + PDB writers, then for each pair writes
``{id}_orig_pocket.pdb`` and ``{id}_recon.pdb`` (pocket backbone + ligand,
original vs VQ-VAE reconstruction) into the output directory.

Self-contained: stdlib + torch + the own repo only. Invoked as:

    <own_venv_python> own_reconstruct_cli.py \
        --workdir <own_repo> --ckpt <ckpt> --cache-dir <descriptor_cache> \
        --out-dir <out> --pairs <pairs.json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_own_module(workdir: Path):
    if str(workdir) not in sys.path:
        sys.path.insert(0, str(workdir))
    script = workdir / "scripts" / "write_reconstruction_pdbs.py"
    spec = importlib.util.spec_from_file_location("own_wrp", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    args = p.parse_args()

    import torch

    W = load_own_module(args.workdir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = W.VQVAEModule.load_from_checkpoint(str(args.ckpt), map_location=device)
    module.eval().to(device)
    protein_vqvae = module.protein_vqvae
    ligand_vqvae = module.ligand_vqvae

    stats = torch.load(args.cache_dir / "normalization_stats.pt", map_location=device)
    norm_stats = {k: v.to(device) for k, v in stats.items()}

    pocket_config = W.PocketExtractionConfig()
    protein_desc_calc = W.BackboneSphericalDescriptor()
    ligand_desc_calc = W.LigandDescriptor()

    pairs = json.loads(args.pairs.read_text())
    n_ok = 0
    summary = []
    for pair in pairs:
        tag = pair["id"]
        try:
            result = W.reconstruct_one(
                Path(pair["receptor"]),
                Path(pair["ligand"]),
                pocket_config=pocket_config,
                protein_desc_calc=protein_desc_calc,
                ligand_desc_calc=ligand_desc_calc,
                protein_vqvae=protein_vqvae,
                ligand_vqvae=ligand_vqvae,
                norm_stats=norm_stats,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            summary.append({"id": tag, "ok": False, "error": repr(exc)})
            print(f"[own] skip {tag}: {exc}", file=sys.stderr)
            continue
        if result is None:
            summary.append({"id": tag, "ok": False, "error": "reconstruct_one returned None"})
            continue

        W.write_complex_pdb(
            args.out_dir / f"{tag}_orig_pocket.pdb",
            result["backbone_orig"],
            result["pocket_seq"],
            result["residue_ids"],
            result["ligand_elements"],
            result["ligand_coords_orig"],
            ligand_bonds=result["ligand_bonds"],
        )
        n_res = len(result["pocket_seq_pred"])
        recon_residue_ids = [("A", i + 1) for i in range(n_res)]
        W.write_complex_pdb(
            args.out_dir / f"{tag}_recon.pdb",
            result["backbone_recon"],
            result["pocket_seq_pred"],
            recon_residue_ids,
            result["ligand_elements_pred"],
            result["ligand_coords_recon"],
            ligand_bonds=result["ligand_bonds_pred"],
        )
        summary.append(
            {
                "id": tag,
                "ok": True,
                "n_pocket_res": len(result["pocket_seq"]),
                "n_ligand_atoms": len(result["ligand_elements"]),
            }
        )
        n_ok += 1

    (args.out_dir / "own_recon_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[own] reconstructed {n_ok}/{len(pairs)} complexes -> {args.out_dir}")


if __name__ == "__main__":
    main()
