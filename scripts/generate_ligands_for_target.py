"""Generate many ligands conditioned on a single docking target's pocket.

Given a prepared target (see ``scripts/prepare_target.py``) — a protein-only
receptor PDB and a reference ligand SDF defining the pocket — this:

1. Extracts the pocket around the reference ligand, computes its canonical
   frame, and encodes the backbone with the protein VQ-VAE -> protein codes.
2. Prompts the LM with ``<bos><p> pocket-codes </p><l>`` and autoregressively
   samples ligand structure codes, in batches, until ``--num-samples`` ligands
   are produced.
3. Decodes each sample with the ligand VQ-VAE, places atoms in the real pocket
   frame (global coords matching the receptor), and infers bonds.

Outputs (under ``--out-dir``):
    generated.sdf      every decoded ligand as a heavy-atom V2000 mol block
    generated.jsonl    one JSON record per ligand {idx, elements, coords, ...}
                       (the docking script's input — coords are the source of
                       truth; bonds are re-perceived by Open Babel at dock time)
    generated_meta.csv per-ligand summary (atom count, formula, flags)

The reference ligand's own pose is also written (idx=-1, tag "ref") so the
known inhibitor can be docked as a positive control alongside the generations.

Run on a GPU node with the venv python directly (``uv run`` rebuilds the
editable package, which is very slow here)::

    PYTHONPATH=$PWD .venv/bin/python scripts/generate_ligands_for_target.py \
        --receptor data/targets/2ity/2ity_receptor.pdb \
        --ref-ligand data/targets/2ity/2ity_ref_ligand.sdf \
        --lm-ckpt pocket-ligand-lm/g79let5b/checkpoints/lm-e09-vl1.4088.ckpt \
        --num-samples 10000 --batch-size 128 --out-dir outputs/egfr_2ity
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np  # noqa: TC002
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_ligands_3d import _decode_ligand, _pocket_codes  # noqa: E402
from scripts.write_reconstruction_pdbs import infer_bonds  # noqa: E402
from src.config import (  # noqa: E402
    CrossDockedConfig,
    LMTrainingConfig,
    PocketExtractionConfig,
    VQVAETrainingConfig,
)
from src.data.descriptors import ComplexDescriptorDataModule  # noqa: E402
from src.model.lm_module import LigandLMModule  # noqa: E402
from src.model.vqvae_module import VQVAEModule  # noqa: E402
from src.tokenizers.ligand import parse_sdf  # noqa: E402
from src.tokenizers.lm_vocab import (  # noqa: E402
    BOS_ID,
    L_CLOSE_ID,
    L_OPEN_ID,
    P_CLOSE_ID,
    P_OPEN_ID,
    PAD_ID,
    LMVocab,
)
from src.tokenizers.protein import (  # noqa: E402
    BackboneSphericalDescriptor,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Real chemical elements we are willing to dock; "X" is the VQ-VAE OTHER
# catch-all and cannot be written to XYZ / docked.
_REAL_ELEMENTS = {
    "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "H",
}
_MIN_ATOMS = 5
_MAX_ATOMS = 80


def _n_fragments(n_atoms: int, bonds: list[tuple[int, int]]) -> int:
    """Number of connected components given a bond list (union-find)."""
    parent = list(range(n_atoms))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in bonds:
        parent[find(a)] = find(b)
    return len({find(i) for i in range(n_atoms)})


def _molblock(elements: list[str], coords: np.ndarray, bonds: list[tuple[int, int]],
              title: str) -> str:
    """Minimal V2000 mol block (single bonds) for storage / visualisation."""
    n_atoms, n_bonds = len(elements), len(bonds)
    counts = f"{n_atoms:>3d}{n_bonds:>3d}  0  0  0  0  0  0  0  0999 V2000"
    lines = [title, "  pclg", "", counts]
    for (x, y, z), el in zip(coords, elements, strict=True):
        sym = el if el != "X" else "*"
        lines.append(
            f"{x:>10.4f}{y:>10.4f}{z:>10.4f} {sym:<3s} "
            "0  0  0  0  0  0  0  0  0  0  0  0"
        )
    for a, b in bonds:
        lines.append(f"{a + 1:>3d}{b + 1:>3d}  1  0  0  0  0")
    lines.append("M  END")
    lines.append("$$$$")
    return "\n".join(lines) + "\n"


@torch.no_grad()
def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor", type=Path, required=True)
    parser.add_argument("--ref-ligand", type=Path, required=True)
    parser.add_argument("--lm-ckpt", type=str, required=True)
    parser.add_argument(
        "--vqvae-ckpt",
        type=str,
        default=(
            "pocket-ligand-vqvae/3dvcbp0h/checkpoints/"
            "vqvae-epoch=99-val/ligand_coord=0.1501.ckpt"
        ),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "descriptor_cache_v4"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = LMVocab()
    lig_lo = vocab.ligand_offset

    # ---- models ----
    vqvae_ckpt = (
        args.vqvae_ckpt
        if Path(args.vqvae_ckpt).is_absolute()
        else PROJECT_ROOT / args.vqvae_ckpt
    )
    vqvae = (
        VQVAEModule.load_from_checkpoint(str(vqvae_ckpt), map_location=device)
        .eval()
        .to(device)
    )
    lm = (
        LigandLMModule.load_from_checkpoint(
            args.lm_ckpt, config=LMTrainingConfig(), map_location=device
        )
        .eval()
        .to(device)
    )
    model = lm.model

    # ---- normalization stats (identical path to the validated 3D script) ----
    dm = ComplexDescriptorDataModule(
        VQVAETrainingConfig(), CrossDockedConfig(data_dir=PROJECT_ROOT / "data")
    )
    dm.cache_dir = args.cache_dir
    dm.setup()
    norm_stats = {k: v.to(device) for k, v in dm.norm_stats.items()}

    # ---- pocket conditioning from the target ----
    mols = parse_sdf(args.ref_ligand)
    if not mols:
        msg = f"Could not parse reference ligand {args.ref_ligand}"
        raise SystemExit(msg)
    res = _pocket_codes(
        args.receptor, mols[0], PocketExtractionConfig(),
        BackboneSphericalDescriptor(), vqvae.protein_vqvae, norm_stats, device,
    )
    if res is None:
        msg = "Pocket extraction failed for the target."
        raise SystemExit(msg)
    prot_codes, frame, ref_coords, ref_elems = res
    logger.info(
        "Pocket: %d residues -> %d protein codes | reference ligand %d atoms",
        len(prot_codes), len(prot_codes), len(ref_elems),
    )

    prompt = [
        BOS_ID, P_OPEN_ID,
        *(vocab.protein_offset + c for c in prot_codes),
        P_CLOSE_ID, L_OPEN_ID,
    ]
    prompt_t = torch.tensor([prompt], device=device)
    prompt_len = len(prompt)

    sdf_path = args.out_dir / "generated.sdf"
    jsonl_path = args.out_dir / "generated.jsonl"
    meta_rows: list[dict] = []

    sdf_f = sdf_path.open("w")
    jsonl_f = jsonl_path.open("w")

    def emit(idx: int, elements: list[str], coords: np.ndarray,  # noqa: PLR0913
             bonds: list[tuple[int, int]], *, terminated: bool, n_codes: int) -> None:
        comp = Counter(elements)
        formula = "".join(f"{e}{n}" for e, n in sorted(comp.items()))
        has_unknown = any(e not in _REAL_ELEMENTS for e in elements)
        n_frag = _n_fragments(len(elements), bonds)
        dockable = (
            not has_unknown
            and _MIN_ATOMS <= len(elements) <= _MAX_ATOMS
        )
        tag = "ref" if idx < 0 else f"gen_{idx}"
        sdf_f.write(_molblock(elements, coords, bonds, tag))
        jsonl_f.write(json.dumps({
            "idx": idx,
            "tag": tag,
            "elements": elements,
            "coords": [[round(float(c), 4) for c in xyz] for xyz in coords],
            "dockable": dockable,
        }) + "\n")
        meta_rows.append({
            "idx": idx, "tag": tag, "n_atoms": len(elements), "formula": formula,
            "terminated": terminated, "n_fragments": n_frag,
            "has_unknown_element": has_unknown, "n_codes": n_codes,
            "dockable": dockable,
        })

    # Reference ligand as positive control (idx = -1).
    ref_bonds = infer_bonds(ref_elems, ref_coords)
    emit(-1, ref_elems, ref_coords, ref_bonds, terminated=True, n_codes=0)

    # ---- batched sampling ----
    produced = 0
    terminated_count = 0
    dockable_count = 0
    n_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        bs = min(args.batch_size, args.num_samples - produced)
        if bs <= 0:
            break
        prompt_ids = prompt_t.repeat(bs, 1)
        gen = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=L_CLOSE_ID,
            pad_token_id=PAD_ID,
        )
        for k in range(gen.shape[0]):
            out_tokens = gen[k].tolist()[prompt_len:]
            terminated = L_CLOSE_ID in out_tokens
            lig_tok = (
                out_tokens[: out_tokens.index(L_CLOSE_ID)] if terminated else out_tokens
            )
            codes = [
                t - lig_lo
                for t in lig_tok
                if lig_lo <= t < lig_lo + vocab.ligand_codebook_size
            ]
            idx = produced
            produced += 1
            terminated_count += int(terminated)
            if not codes:
                meta_rows.append({
                    "idx": idx, "tag": f"gen_{idx}", "n_atoms": 0, "formula": "",
                    "terminated": terminated, "n_fragments": 0,
                    "has_unknown_element": False, "n_codes": 0, "dockable": False,
                })
                continue
            coords, elems = _decode_ligand(
                codes, vqvae.ligand_vqvae, norm_stats, frame, device
            )
            bonds = infer_bonds(elems, coords)
            emit(idx, elems, coords, bonds, terminated=terminated, n_codes=len(codes))
            if meta_rows[-1]["dockable"]:
                dockable_count += 1
        sdf_f.flush()
        jsonl_f.flush()
        logger.info(
            "batch %d/%d | produced %d | terminated %d | dockable %d",
            b + 1, n_batches, produced, terminated_count, dockable_count,
        )

    sdf_f.close()
    jsonl_f.close()

    import csv  # noqa: PLC0415

    meta_path = args.out_dir / "generated_meta.csv"
    with meta_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meta_rows)

    logger.info(
        "\nDone. %d generated (%d terminated, %d dockable) + 1 reference.\n"
        "  %s\n  %s\n  %s",
        produced, terminated_count, dockable_count, sdf_path, jsonl_path, meta_path,
    )


if __name__ == "__main__":
    main()
