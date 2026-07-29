"""Generate many ligands conditioned on a single docking target's pocket.

Given a prepared target (see ``scripts/prepare_target.py``) — a protein-only
receptor PDB and a reference ligand SDF defining the pocket — this:

1. Extracts the pocket around the reference ligand, computes its canonical
   frame, and encodes the pocket with the protein VQ-VAE -> protein codes.
2. Prompts the LM with ``<bos><p> pocket-codes </p><l>`` and autoregressively
   samples ligand structure codes, in batches, until ``--num-samples`` ligands
   are produced.
3. Decodes each sample with the ligand VQ-VAE, places atoms in the real pocket
   frame (global coords matching the receptor), and infers bonds.

Two tokenizer arms are supported (both share the sampling / output code, so the
emitted ``generated.sdf`` is identical in layout for either — this is what the
sbdd-bench ``own`` adapter consumes):

* the joint tokenizer (default): one all-atom VQ over a shared codebook
  (``--vqvae-ckpt`` is an :class:`AtomVQVAEModule`).
* ``--separate-protein-ckpt`` (+ the other ``--separate-*`` flags): the ablation
  arm — pocket encoded by a protein-only VQ, ligand decoded by a ligand-only VQ
  over one combined ``2*--codebook-size`` space (:class:`SeparateVQVAE`); the LM
  is the matching separate LM.

Both encode/decode branches mirror ``scripts/generate_ligands_3d.py`` exactly
(that script is the validated reference); only the driver here differs (single
external target -> ``generated.sdf`` instead of an internal test-pool sweep ->
per-pocket PDBs).

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

    # joint tokenizer (single 8192 codebook)
    PYTHONPATH=$PWD .venv/bin/python scripts/generate_ligands_for_target.py \
        --receptor data/targets/2ity/2ity_receptor.pdb \
        --ref-ligand data/targets/2ity/2ity_ref_ligand.sdf \
        --vqvae-ckpt <atom-vqvae.ckpt> --codebook-size 8192 \
        --lm-ckpt pocket-ligand-lm/p6lpk7br/checkpoints/lm-e02-vl1.0029.ckpt \
        --num-samples 100 --batch-size 100 --out-dir outputs/egfr_2ity

    # separate 4096+4096 tokenizers (combined 8192 space)
    ... --separate-protein-ckpt <p.ckpt> --separate-protein-norm <p.pt> \
        --separate-ligand-ckpt <l.ckpt> --separate-ligand-norm <l.pt> \
        --codebook-size 4096 ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prolit.chem.pdb_io import infer_bonds  # noqa: E402
from prolit.config import (  # noqa: E402
    PocketExtractionConfig,
)
from prolit.tokenizers.atom import ProteinAtomDescriptor  # noqa: E402
from prolit.tokenizers.ligand import parse_sdf  # noqa: E402
from prolit.tokenizers.lm_vocab import (  # noqa: E402
    L_CLOSE_ID,
    PAD_ID,
    AtomLMVocab,
)
from scripts.generate_ligands_3d import (  # noqa: E402
    _decode_ligand_atom,
    _pocket_codes_atom,
    load_atom_lm,
    load_atom_norm_stats,
    load_atom_vqvae,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Real chemical elements we are willing to dock; "X" is the VQ-VAE OTHER
# catch-all and cannot be written to XYZ / docked.
_REAL_ELEMENTS = {
    "C",
    "N",
    "O",
    "S",
    "F",
    "Cl",
    "Br",
    "I",
    "P",
    "B",
    "Si",
    "H",
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


def _molblock(
    elements: list[str], coords: np.ndarray, bonds: list[tuple[int, int]], title: str
) -> str:
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


def _pocket_context(receptor_path: Path, ref_mol: dict, frame: tuple) -> tuple | None:
    """Pocket atoms (canonical coords + node features) for the pose refiner.

    Extracted independently of the LM's backbone protein VQ-VAE, in the SAME
    canonical ``frame`` the ligand is decoded into, so refiner sees ligand and
    pocket in one coordinate system. Only atoms within the refiner's cutoff of
    the ligand end up mattering (the collate filters by radius).
    """
    from prolit.model.pose_refiner import pocket_feats_from_descriptor  # noqa: PLC0415
    from prolit.tokenizers.atom import (  # noqa: PLC0415
        ProteinAtomDescriptor,
        precompute_receptor_atom_features_from_text,
    )
    from prolit.tokenizers.protein import (  # noqa: PLC0415
        extract_pocket_atoms_from_candidates,
        precompute_pocket_atom_candidates_from_text,
    )

    rec_text = Path(receptor_path).read_text()
    ref_heavy = np.array(
        [(a[1], a[2], a[3]) for a in ref_mol["atoms"] if a[0] != "H"], dtype=np.float32
    )
    precomp = precompute_pocket_atom_candidates_from_text(rec_text)
    pocket = extract_pocket_atoms_from_candidates(
        precomp, ref_heavy, PocketExtractionConfig()
    )
    if pocket is None or pocket.atom_coords.shape[0] == 0:
        return None
    feats = precompute_receptor_atom_features_from_text(rec_text)
    prot_desc, _ = ProteinAtomDescriptor().compute(pocket, feats, frame)
    if prot_desc.shape[0] != pocket.atom_coords.shape[0]:
        return None
    centroid, rotation = frame
    pkt_canon = (
        (pocket.atom_coords.astype(np.float64) - centroid) @ rotation.T
    ).astype(np.float32)
    return pkt_canon, pocket_feats_from_descriptor(prot_desc)


def _build_generator(args, device):  # noqa: ANN001
    """Load the LM + VQ-VAE(s) for the selected tokenizer path and return the
    pieces the sampling loop needs.

    Returns ``(model, vocab, code_lo, code_hi, code_base, encode_pocket,
    decode_codes)`` where ``encode_pocket(rec_path, mol)`` yields
    ``(protein_codes, frame, ref_coords, ref_elems)`` and
    ``decode_codes(codes, frame, refiner, pocket_ctx)`` yields
    ``(coords, elements)``. Both branches mirror ``generate_ligands_3d.py``'s
    ``main`` exactly.
    """
    vqvae_ckpt = (
        args.vqvae_ckpt
        if Path(args.vqvae_ckpt).is_absolute()
        else PROJECT_ROOT / args.vqvae_ckpt
    )
    pocket_config = PocketExtractionConfig()
    receptor_cache: dict[str, tuple] = {}

    if args.separate_protein_ckpt is not None:
        # ABLATION separate-tokenizers mode: pocket encoded by a protein-only VQ,
        # ligand decoded by a ligand-only VQ, unified into one combined code space
        # (protein codes [0, Pc), ligand codes [Pc, 2*Pc)) by SeparateVQVAE. The
        # LM is the separate LM trained over an AtomLMVocab of 2*codebook-size.
        from prolit.tokenizers.separate_vqvae import SeparateVQVAE  # noqa: PLC0415

        separate_vqvae = SeparateVQVAE.from_checkpoints(
            args.separate_protein_ckpt,
            args.separate_protein_norm,
            args.separate_ligand_ckpt,
            args.separate_ligand_norm,
            device,
            codebook_size=args.codebook_size,
        )
        combined_codebook_size = 2 * args.codebook_size
        model = load_atom_lm(args.lm_ckpt, combined_codebook_size, device, split=False)
        protein_norm = load_atom_norm_stats(args.separate_protein_norm, device)
        ligand_norm = separate_vqvae.ligand_norm_stats
        vocab = AtomLMVocab(codebook_size=combined_codebook_size)
        code_lo = vocab.offset + args.codebook_size
        code_hi = vocab.offset + vocab.codebook_size
        code_base = vocab.offset
        prot_atom_desc = ProteinAtomDescriptor()

        def encode_pocket(rec_path, mol):  # noqa: ANN001, ANN202
            return _pocket_codes_atom(
                rec_path,
                mol,
                pocket_config,
                prot_atom_desc,
                separate_vqvae.protein,
                protein_norm,
                device,
                receptor_cache=receptor_cache,
            )

        def decode_codes(codes, frame, refiner=None, pocket_ctx=None):  # noqa: ANN001, ANN202
            return _decode_ligand_atom(
                codes,
                separate_vqvae,
                ligand_norm,
                frame,
                device,
                refiner=refiner,
                pocket_ctx=pocket_ctx,
            )

    else:
        atom_vqvae = load_atom_vqvae(vqvae_ckpt, args.codebook_size, device)
        model = load_atom_lm(args.lm_ckpt, args.codebook_size, device)
        norm_stats = load_atom_norm_stats(args.norm_stats, device)
        vocab = AtomLMVocab(codebook_size=args.codebook_size)
        code_lo, code_hi = vocab.offset, vocab.offset + vocab.codebook_size
        code_base = code_lo
        prot_atom_desc = ProteinAtomDescriptor()

        def encode_pocket(rec_path, mol):  # noqa: ANN001, ANN202
            return _pocket_codes_atom(
                rec_path,
                mol,
                pocket_config,
                prot_atom_desc,
                atom_vqvae,
                norm_stats,
                device,
                receptor_cache=receptor_cache,
            )

        def decode_codes(codes, frame, refiner=None, pocket_ctx=None):  # noqa: ANN001, ANN202
            return _decode_ligand_atom(
                codes,
                atom_vqvae,
                norm_stats,
                frame,
                device,
                refiner=refiner,
                pocket_ctx=pocket_ctx,
            )

    return model, vocab, code_lo, code_hi, code_base, encode_pocket, decode_codes


@torch.no_grad()
def main() -> None:  # noqa: C901, PLR0915
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
    parser.add_argument(
        "--min-atoms-abs",
        type=int,
        default=0,
        help="Absolute floor on the generated heavy-atom count, combined with "
        "--min-atoms-frac as max(frac * n_ref, abs). Reference-tied sizing alone "
        "loses affinity on small-reference pockets: the baselines regress ligand "
        "size toward their own corpus mean (DiffSBDD averages 20.9 heavy atoms "
        "and runs 1.48x the reference on the smallest-reference third of the "
        "CrossDocked test set), while this LM's length tracks the reference "
        "almost deterministically. This floor matches that behaviour.",
    )
    parser.add_argument(
        "--min-atoms-frac",
        type=float,
        default=0.0,
        help="Condition the generated ligand LENGTH on the reference ligand: "
        "suppress the </l> stop token until at least round(frac * n_ref_heavy) "
        "ligand codes have been emitted. One ligand token decodes to exactly one "
        "heavy atom, so this is a per-target minimum heavy-atom count -- the same "
        "kind of size conditioning TargetDiff/DiffSBDD apply by sampling N from a "
        "pocket-conditioned distribution. 0 disables it (unconstrained length).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--refine-ckpt",
        type=str,
        default=None,
        help="PoseRefinerModule checkpoint. When set, each decoded pose is "
        "refined by the E(3)-equivariant pose refiner before bond inference "
        "(removes clashes/strain from the raw pose). Default off = unchanged.",
    )
    # --- tokenizer-path flags (mirror scripts/generate_ligands_3d.py) ---
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument(
        "--separate-protein-ckpt",
        type=Path,
        default=None,
        help="ABLATION separate-tokenizers mode: protein-only VQ ckpt. When set "
        "(with the other --separate-* flags), decode uses a SeparateVQVAE over a "
        "combined 2*codebook-size code space and the LM must be the separate LM "
        "(AtomLMVocab sized 2*codebook-size). Implies the all-atom encode path.",
    )
    parser.add_argument("--separate-protein-norm", type=Path, default=None)
    parser.add_argument("--separate-ligand-ckpt", type=Path, default=None)
    parser.add_argument("--separate-ligand-norm", type=Path, default=None)
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "descriptor_cache_allatom"
        / "normalization_stats.pt",
        help="All-atom normalization stats (.pt with atom_mean/atom_std).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- models + tokenizer setup (legacy / all-atom / separate) ----
    (
        model,
        vocab,
        code_lo,
        code_hi,
        code_base,
        encode_pocket,
        decode_codes,
    ) = _build_generator(args, device)

    def build_prompt(prot_codes: list[int]) -> list[int]:
        # build_sequence(prot, [])[:-2] drops the trailing </l><eos>, leaving
        # <bos><p> prot </p><l>; identical for LMVocab and AtomLMVocab.
        return vocab.build_sequence(prot_codes, [])[:-2]

    # ---- pocket conditioning from the target ----
    mols = parse_sdf(args.ref_ligand)
    if not mols:
        msg = f"Could not parse reference ligand {args.ref_ligand}"
        raise SystemExit(msg)
    res = encode_pocket(args.receptor, mols[0])
    if res is None:
        msg = "Pocket extraction failed for the target."
        raise SystemExit(msg)
    prot_codes, frame, ref_coords, ref_elems = res
    logger.info(
        "Pocket: %d protein codes | reference ligand %d atoms",
        len(prot_codes),
        len(ref_elems),
    )

    # ---- optional pose refiner ----
    refiner = None
    pocket_ctx = None
    if args.refine_ckpt is not None:
        from prolit.model.pose_refiner import PoseRefinerModule  # noqa: PLC0415

        refiner = (
            PoseRefinerModule.load_from_checkpoint(
                args.refine_ckpt, map_location=device
            )
            .eval()
            .to(device)
        )
        pocket_ctx = _pocket_context(args.receptor, mols[0], frame)
        if pocket_ctx is None:
            logger.warning(
                "Pocket-context extraction failed; pose refinement disabled."
            )
            refiner = None
        else:
            logger.info(
                "Pose refiner loaded (%d pocket atoms).", pocket_ctx[0].shape[0]
            )

    prompt = build_prompt(prot_codes)
    prompt_t = torch.tensor([prompt], device=device)
    prompt_len = len(prompt)

    sdf_path = args.out_dir / "generated.sdf"
    jsonl_path = args.out_dir / "generated.jsonl"
    meta_rows: list[dict] = []

    sdf_f = sdf_path.open("w")
    jsonl_f = jsonl_path.open("w")

    def emit(  # noqa: PLR0913
        idx: int,
        elements: list[str],
        coords: np.ndarray,
        bonds: list[tuple[int, int]],
        *,
        terminated: bool,
        n_codes: int,
    ) -> None:
        comp = Counter(elements)
        formula = "".join(f"{e}{n}" for e, n in sorted(comp.items()))
        has_unknown = any(e not in _REAL_ELEMENTS for e in elements)
        n_frag = _n_fragments(len(elements), bonds)
        dockable = not has_unknown and _MIN_ATOMS <= len(elements) <= _MAX_ATOMS
        tag = "ref" if idx < 0 else f"gen_{idx}"
        sdf_f.write(_molblock(elements, coords, bonds, tag))
        jsonl_f.write(
            json.dumps(
                {
                    "idx": idx,
                    "tag": tag,
                    "elements": elements,
                    "coords": [[round(float(c), 4) for c in xyz] for xyz in coords],
                    "dockable": dockable,
                }
            )
            + "\n"
        )
        meta_rows.append(
            {
                "idx": idx,
                "tag": tag,
                "n_atoms": len(elements),
                "formula": formula,
                "terminated": terminated,
                "n_fragments": n_frag,
                "has_unknown_element": has_unknown,
                "n_codes": n_codes,
                "dockable": dockable,
            }
        )

    # Reference ligand as positive control (idx = -1).
    # Reference-conditioned minimum ligand length (see --min-atoms-frac). The LM
    # under-generates size relative to the crystal ligand, and Vina is not
    # size-normalised, so an unconstrained length systematically loses affinity.
    min_new_tokens = max(
        int(round(args.min_atoms_frac * len(ref_elems)))
        if args.min_atoms_frac > 0
        else 0,
        args.min_atoms_abs,
    )
    min_new_tokens = min(min_new_tokens, args.max_new_tokens - 1)
    if min_new_tokens > 0:
        logger.info(
            "min_new_tokens=%d (max of %.2f x %d reference heavy atoms, abs %d)",
            min_new_tokens,
            args.min_atoms_frac,
            len(ref_elems),
            args.min_atoms_abs,
        )

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
            min_new_tokens=min_new_tokens,
            eos_token_id=L_CLOSE_ID,
            pad_token_id=PAD_ID,
        )
        for k in range(gen.shape[0]):
            out_tokens = gen[k].tolist()[prompt_len:]
            terminated = L_CLOSE_ID in out_tokens
            lig_tok = (
                out_tokens[: out_tokens.index(L_CLOSE_ID)] if terminated else out_tokens
            )
            codes = [t - code_base for t in lig_tok if code_lo <= t < code_hi]
            idx = produced
            produced += 1
            terminated_count += int(terminated)
            if not codes:
                meta_rows.append(
                    {
                        "idx": idx,
                        "tag": f"gen_{idx}",
                        "n_atoms": 0,
                        "formula": "",
                        "terminated": terminated,
                        "n_fragments": 0,
                        "has_unknown_element": False,
                        "n_codes": 0,
                        "dockable": False,
                    }
                )
                continue
            coords, elems = decode_codes(
                codes, frame, refiner=refiner, pocket_ctx=pocket_ctx
            )
            bonds = infer_bonds(elems, coords)
            emit(idx, elems, coords, bonds, terminated=terminated, n_codes=len(codes))
            if meta_rows[-1]["dockable"]:
                dockable_count += 1
        sdf_f.flush()
        jsonl_f.flush()
        logger.info(
            "batch %d/%d | produced %d | terminated %d | dockable %d",
            b + 1,
            n_batches,
            produced,
            terminated_count,
            dockable_count,
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
        produced,
        terminated_count,
        dockable_count,
        sdf_path,
        jsonl_path,
        meta_path,
    )


if __name__ == "__main__":
    main()
