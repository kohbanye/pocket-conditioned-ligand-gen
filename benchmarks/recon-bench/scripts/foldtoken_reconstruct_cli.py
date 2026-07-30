"""Reconstruct PDBs with FoldToken4, one structure per forward pass.

Run by the FoldToken venv. The upstream reconstruct.py batches 32 structures
per forward; with length-heterogeneous inputs that batching corrupts the
reconstruction (e.g. a structure that scores ~1.8 A alone blows up to ~15 A in a
32-batch). This driver loads the model once and reconstructs each structure in a
batch of 1, which matches the single-structure quality the paper reports.

Writes ``{title}_pred.pdb`` + ``vqids.json`` into ``{path_out}_level{level}/``,
same layout as reconstruct.py, so the adapter reads it unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True, help="FoldToken_open repo root")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--path_in", required=True)
    p.add_argument("--path_out", required=True)
    p.add_argument("--level", type=int, default=12)
    args = p.parse_args()

    # foldtoken/ must be on the path for `model_interface` and `src.*`.
    sys.path.insert(0, str(Path(args.workdir) / "foldtoken"))
    sys.path.insert(0, str(args.workdir))

    import torch
    from model_interface import MInterface
    from omegaconf import OmegaConf
    from prolit.chroma.data import Protein

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    model = MInterface(**config)
    ckpt = torch.load(args.checkpoint, map_location=torch.device("cuda"))
    for key in list(ckpt.keys()):
        if "_forward_module." in key:
            ckpt[key.replace("_forward_module.", "")] = ckpt.pop(key)
    model.load_state_dict(ckpt)
    model = model.to("cuda").eval()

    out_dir = Path(f"{args.path_out}_level{args.level}")
    out_dir.mkdir(parents=True, exist_ok=True)

    vqids: dict[str, list] = {}
    files = sorted(f for f in os.listdir(args.path_in) if f.endswith((".pdb", ".cif")))
    for fname in files:
        title = fname.split(".")[0]
        try:
            protein = Protein(os.path.join(args.path_in, fname), device="cuda")
            torch.manual_seed(0)
            with torch.no_grad():
                batch = model.batch_proteins([protein])
                preds, codes = model.sample(batch=batch, level=args.level)
            preds[0].to(
                str(out_dir / f"{title}_pred.pdb"),
                mask_indices=None,
                seq=codes[0].tolist(),
            )
            vqids[title] = codes[0].tolist()
        except Exception as exc:  # noqa: BLE001 - skip a bad structure, keep going
            print(f"[foldtoken] skip {title}: {exc}", file=sys.stderr)

    (out_dir / "vqids.json").write_text(json.dumps(vqids))
    print(f"[foldtoken] reconstructed {len(vqids)}/{len(files)} -> {out_dir}")


if __name__ == "__main__":
    main()
