"""Reconstruct pockets, ligands and complexes with Bio2Token (subprocess).

Bio2Token pins torch 2.4.1+cu121 and mamba-ssm, so it runs in its own venv
(``scripts/setup_bio2token_env.sh``) and the adapter drives it from outside --
the same arrangement FoldToken uses.

Three modes, all fed from the all-atom NPZ dumps the own tokenizer already
produced, so every model sees the identical pocket and ligand in the identical
frame:

``protein``  pocket heavy atoms      -> prot2token / bio2token
``ligand``   ligand heavy atoms      -> mol2token
``complex``  pocket + ligand at once -> bio2token

The complex mode is **out of distribution**: Bio2Token was trained on proteins,
RNA and small molecules as separate structures, and its upstream PDB reader
silently drops every non-standard residue, so no protein-ligand complex ever
reached it during training. It is reported to show what a state-of-the-art
all-atom tokenizer does when asked for a complex, not as a like-for-like number.

Bio2Token consumes coordinates and nothing else -- no elements, no bond graph.
Small-molecule atoms are all ``BB_CLASS`` with a single residue id, exactly as
its NablaDFT loader does; protein atoms get BB / C_REF / SC from their atom
names, ordered backbone-then-sidechain within each residue to match the
convention the model was trained on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_BB_ORDER = ["N", "CA", "C", "O"]


def load_bio2token(repo: Path):
    """Import Bio2Token and return the handles this CLI needs."""
    src = repo / "src"
    for p in (str(src), str(repo)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from bio2token.data.utils.molecule_conventions import SC_ATOMS_AA
    from bio2token.data.utils.tokens import BB_CLASS, C_REF_CLASS, SC_CLASS
    from bio2token.data.utils.utils import compute_masks
    from bio2token.models.autoencoder import Autoencoder, AutoencoderConfig
    from bio2token.utils.configs import pi_instantiate, utilsyaml_to_dict

    return {
        "Autoencoder": Autoencoder,
        "AutoencoderConfig": AutoencoderConfig,
        "utilsyaml_to_dict": utilsyaml_to_dict,
        "pi_instantiate": pi_instantiate,
        "compute_masks": compute_masks,
        "BB_CLASS": BB_CLASS,
        "C_REF_CLASS": C_REF_CLASS,
        "SC_CLASS": SC_CLASS,
        "SC_ATOMS_AA": SC_ATOMS_AA,
    }


def order_pocket_atoms(b2t, atom_names, chains, resids, aa1):
    """Order pocket atoms backbone-then-sidechain per residue, with token classes.

    Bio2Token's own loader emits atoms in this canonical order, so feeding a raw
    PDB ordering would put the model off its training distribution for no reason.
    Returns (index array into the input arrays, token_class, residue_index).
    """
    order, classes, res_index = [], [], []
    seen: dict[tuple, int] = {}
    for pos, key in enumerate(zip(chains, resids, strict=True)):
        seen.setdefault(key, len(seen))
        del pos
    by_res: dict[tuple, list[int]] = {}
    for i, key in enumerate(zip(chains, resids, strict=True)):
        by_res.setdefault(key, []).append(i)

    for key, idxs in sorted(by_res.items(), key=lambda kv: seen[kv[0]]):
        names = {atom_names[i].strip(): i for i in idxs}
        res1 = aa1.get(key, "X")
        sc_names = b2t["SC_ATOMS_AA"].get(res1, [])
        for name in _BB_ORDER:
            if name in names:
                order.append(names[name])
                classes.append(b2t["C_REF_CLASS"] if name == "CA" else b2t["BB_CLASS"])
                res_index.append(seen[key])
        for name in sc_names:
            if name in names:
                order.append(names[name])
                classes.append(b2t["SC_CLASS"])
                res_index.append(seen[key])
    return np.array(order, dtype=int), np.array(classes, dtype=int), np.array(res_index, dtype=int)


def run_model(b2t, model, device, coords, token_class, residue_ids):
    """One forward pass; returns (reconstructed coords, token indices)."""
    import torch

    n = coords.shape[0]
    batch = {
        "structure": torch.tensor(coords, dtype=torch.float32),
        "residue_ids": torch.tensor(residue_ids, dtype=torch.long),
        "token_class": torch.tensor(token_class, dtype=torch.long),
        "unknown_structure": torch.zeros(n, dtype=torch.bool),
    }
    batch = b2t["compute_masks"](batch, structure_track=True)
    batch = {k: v[None].to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model(batch)
    rec = out["all_atom_coords"][0].detach().cpu().numpy()[:n]
    idx = out["indices"][0].detach().cpu().numpy().reshape(-1)[:n]
    return rec.astype(np.float64), idx


def main() -> None:  # noqa: PLR0915
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True, help="third_party/bio2token")
    p.add_argument("--dumps", type=Path, required=True, help="dir of all-atom NPZ dumps")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["protein", "ligand", "complex"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--aa-map", type=Path, default=None, help="optional {sample: {chain_resid: aa1}}")
    args = p.parse_args()

    import torch

    b2t = load_bio2token(args.repo)
    # utilsyaml_to_dict resolves its argument against "configs/" in the *current*
    # directory, so the model config only loads from inside the Bio2Token repo.
    cwd = Path.cwd()
    os.chdir(args.repo)
    try:
        cfg = b2t["utilsyaml_to_dict"]("test_pdb.yaml")
    finally:
        os.chdir(cwd)
    model = b2t["Autoencoder"](
        b2t["pi_instantiate"](b2t["AutoencoderConfig"], yaml_dict=cfg["model"])
    )
    state = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict({k.replace("model.", ""): v for k, v in state.items()})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    aa_map = json.loads(args.aa_map.read_text()) if args.aa_map else {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary, n_ok = [], 0

    for dump in sorted(args.dumps.glob("*.npz")):
        tag = dump.stem
        try:
            d = np.load(dump, allow_pickle=False)
            prot = d["protein_ref"].astype(np.float64)
            lig = d["ligand_ref"].astype(np.float64)
            names = [str(x) for x in d["protein_atom_names"]]
            chains = [str(x) for x in d["protein_chain"]]
            resids = [int(x) for x in d["protein_resid"]]
            aa1 = {
                (c, r): aa_map.get(tag, {}).get(f"{c}_{r}", "X")
                for c, r in zip(chains, resids, strict=True)
            }

            if args.mode == "ligand":
                # Bio2Token's small-molecule convention: every atom backbone,
                # one residue, recentred (its NablaDFT loader does the same).
                ref = lig
                centre = ref.mean(axis=0)
                coords = ref - centre
                tclass = np.full(len(ref), b2t["BB_CLASS"], dtype=int)
                rids = np.zeros(len(ref), dtype=int)
                rec, idx = run_model(b2t, model, device, coords, tclass, rids)
                out = {"ref": coords, "rec": rec, "n_tokens": np.int64(len(idx))}
            else:
                order, tclass, rids = order_pocket_atoms(b2t, names, chains, resids, aa1)
                prot_ord = prot[order]
                if args.mode == "protein":
                    ref = prot_ord
                    centre = ref.mean(axis=0)
                    coords = ref - centre
                else:  # complex: one sequence, one shared centre, so the
                    # ligand's placement relative to the pocket is preserved.
                    ref = np.vstack([prot_ord, lig])
                    centre = ref.mean(axis=0)
                    coords = ref - centre
                    tclass = np.concatenate(
                        [tclass, np.full(len(lig), b2t["BB_CLASS"], dtype=int)]
                    )
                    rids = np.concatenate(
                        [rids, np.full(len(lig), int(rids.max()) + 1, dtype=int)]
                    )
                rec, idx = run_model(b2t, model, device, coords, tclass, rids)
                out = {
                    "ref": ref - centre,
                    "rec": rec,
                    "n_tokens": np.int64(len(idx)),
                    "n_protein_rows": np.int64(len(prot_ord)),
                    "protein_order": order,
                }
            np.savez_compressed(args.out_dir / f"{tag}.npz", **out)
            summary.append({"id": tag, "ok": True})
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - one bad complex must not stop the run
            summary.append({"id": tag, "ok": False, "error": repr(exc)})
            print(f"[bio2token] skip {tag}: {exc}", file=sys.stderr)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[bio2token] {args.mode}: {n_ok} reconstructed -> {args.out_dir}")


if __name__ == "__main__":
    main()
