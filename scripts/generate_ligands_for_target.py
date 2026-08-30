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
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Sibling module, imported by bare name: Python puts the script's own
# directory on sys.path[0], so this resolves no matter where it is run from
# (a ``scripts.`` prefix would need the repository root on the path).
from generate_ligands_3d import (
    _decode_ligand_atom,
    _perceive_bonds,
    _pocket_codes_atom,
    load_atom_lm,
    load_atom_norm_stats,
    load_atom_vqvae,
)
from rdkit import Chem
from transformers import LogitsProcessor, LogitsProcessorList

from prolit.chem.bond_orders import (
    connect_fragments,
    mol_from_decoded,
    prune_to_valence,
)
from prolit.chem.pdb_io import infer_bonds
from prolit.chem.rigid_fit import vdw_radii
from prolit.config import (
    PocketExtractionConfig,
)
from prolit.model.mlm_decode import refine_codes
from prolit.provenance import write_manifest
from prolit.seeding import add_seed_argument, seed_from_args
from prolit.tokenizers.atom import ProteinAtomDescriptor
from prolit.tokenizers.ligand import parse_sdf
from prolit.tokenizers.lm_vocab import (
    L_CLOSE_ID,
    PAD_ID,
    AtomLMVocab,
)

# Repository root, used only for default data/output locations. ``prolit`` is
# an installed package, so nothing needs to be put on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    """Minimal V2000 mol block, **all bonds single** -- the last-resort writer.

    Only used when :func:`prolit.chem.bond_orders.mol_from_decoded` cannot make
    a molecule out of the decoded chemistry. Prefer that: writing an aromatic
    ring as a saturated one keeps the coordinates honest but makes the chemistry
    a lie, and every downstream consumer then reads 1.39 A bonds and 120 degree
    angles as a *geometry* error rather than a bond-order one.
    """
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
    # The radii ride along because the same pocket is what a rigid steric
    # placement is measured against, and re-deriving them downstream would be
    # a second place for the element list to be got wrong.
    return (
        pkt_canon,
        pocket_feats_from_descriptor(prot_desc),
        vdw_radii(list(pocket.atom_elements)),
        # second radii set, for --scoring-radii (see rigid_fit.VINA_RADII)
        vdw_radii(list(pocket.atom_elements), scoring=True),
    )


def _build_generator(args, device) -> tuple:  # noqa: ANN001
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

        def decode_codes(codes, frame, refiner=None, pocket_ctx=None, bond_head=None, reference_codes=None):  # noqa: ANN001, ANN202, E501, PLR0913
            return _decode_ligand_atom(
                codes,
                separate_vqvae,
                ligand_norm,
                frame,
                device,
                refiner=refiner,
                pocket_ctx=pocket_ctx,
                place_first=args.place_before_refine,
                scoring_radii=args.scoring_radii,
                refine_rounds=args.refine_rounds,
                bond_head=bond_head,
                reference_codes=reference_codes,
                reconcile_mode=args.reconcile,
                refine_project=args.refine_project,
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

        def decode_codes(codes, frame, refiner=None, pocket_ctx=None, bond_head=None, reference_codes=None):  # noqa: ANN001, ANN202, E501, PLR0913
            return _decode_ligand_atom(
                codes,
                atom_vqvae,
                norm_stats,
                frame,
                device,
                refiner=refiner,
                pocket_ctx=pocket_ctx,
                place_first=args.place_before_refine,
                scoring_radii=args.scoring_radii,
                refine_rounds=args.refine_rounds,
                bond_head=bond_head,
                reference_codes=reference_codes,
                reconcile_mode=args.reconcile,
                refine_project=args.refine_project,
            )

    return model, vocab, code_lo, code_hi, code_base, encode_pocket, decode_codes



class _AnchorTemperature(LogitsProcessor):
    """Sample the first ``n_anchor`` generated tokens at a different temperature.

    ``generate`` applies one temperature to the whole sequence. Rescaling the
    logits here, before that warper runs, gives an effective temperature of
    ``base / factor`` for the tokens this covers.
    """

    def __init__(self, prompt_len: int, n_anchor: int, factor: float) -> None:
        self.prompt_len = prompt_len
        self.n_anchor = n_anchor
        self.factor = factor

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):  # noqa: ANN204
        if input_ids.shape[1] - self.prompt_len < self.n_anchor:
            return scores * self.factor
        return scores


@torch.no_grad()
def main() -> None:  # noqa: C901, PLR0912, PLR0915
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
    parser.add_argument(
        "--anchor-temperature",
        type=float,
        default=None,
        help="sampling temperature for the first --anchor-atoms ligand tokens. "
        "The first atom is where the molecule gets anchored in the pocket, and "
        "its spread propagates: measured across 99 targets, the spread of atom "
        "0 is 2.81 A, the spread of the whole molecule's centroid is 2.20 A, "
        "and the two correlate at Spearman +0.74. Sampling the anchor colder "
        "than the rest buys placement without flattening the molecular "
        "diversity that the later atoms carry. Default: same as --temperature.",
    )
    parser.add_argument(
        "--anchor-atoms",
        type=int,
        default=3,
        help="how many leading ligand tokens --anchor-temperature applies to.",
    )
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
    add_seed_argument(parser, default=0)
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
    parser.add_argument(
        "--place-before-refine",
        action="store_true",
        help="slide the decoded ligand off the pocket wall as a rigid body "
        "before the refiner sees it, so the refiner is handed the local error "
        "it was trained on rather than a 2 A global displacement",
    )
    parser.add_argument(
        "--refine-rounds",
        type=int,
        default=1,
        help="how many place-then-refine rounds to run",
    )
    parser.add_argument(
        "--mlm-ckpt",
        type=Path,
        default=None,
        help="complex MLM (ligand-only masking). Given, the sampled ligand "
        "codes are re-decided by the MLM before decoding: it re-masks the "
        "positions it is least sure of and re-predicts them with the rest of "
        "the molecule visible. The causal model has to choose the anchor while "
        "it is still 5.30 A uncertain and every later atom inherits that; the "
        "bidirectional model is 0.65 A. Measured, RMSD 1.070 -> 0.857.",
    )
    parser.add_argument(
        "--iter-rounds",
        type=int,
        default=8,
        help="how many re-mask/re-predict rounds --mlm-ckpt runs",
    )
    parser.add_argument(
        "--refine-project",
        choices=("none", "rigid", "torsion"),
        default="none",
        help="'rigid' keeps only the rigid part of the refiner's displacement. "
        "Every refiner in this family buys contact by shrinking the molecule "
        "(bonds out of tolerance 10.0%% without one, 48.1%% with); a rigid "
        "motion cannot change a bond length, so this keeps the placement it "
        "predicts and drops the compression. 'torsion' additionally keeps the "
        "dihedral each rotatable bond turned through, which is still exactly "
        "bond- and angle-preserving but follows the refiner further: Vina's "
        "own local optimiser moves in that same space and finds 9.91 kcal "
        "where the rigid part alone finds 4.40.",
    )
    parser.add_argument(
        "--iter-mode",
        choices=("warm", "cold"),
        default="warm",
        help="'warm' revises the causal model's codes; 'cold' throws them away "
        "and lets the MLM fill an all-masked ligand of the same length, so no "
        "position is ever committed on a left prefix alone. Two thirds of the "
        "clash slope along the token order is the causal decode order.",
    )
    parser.add_argument(
        "--iter-order",
        choices=("confidence", "late_first"),
        default="confidence",
        help="which positions each refinement round re-decides. 'confidence' is "
        "MaskGIT's least-confident-first. 'late_first' sweeps blocks backwards "
        "from the end, because the clash rate climbs 11.4%% -> 33.7%% along the "
        "decode order while FLOWR stays flat at ~8%%.",
    )
    parser.add_argument(
        "--reconcile",
        choices=("off", "align", "splice"),
        default="off",
        help="what to do about the decoder reacting globally to a local code "
        "edit. Measured, editing one code moves the edited atom 2.20 A (the "
        "point) and every other atom 0.24 A (not the point), and only 16% of "
        "that is a rigid move. 'align' superimposes on the unedited atoms; "
        "'splice' also puts them back exactly.",
    )
    parser.add_argument(
        "--iter-frac",
        type=float,
        default=0.25,
        help="fraction of the ligand re-masked each round",
    )
    parser.add_argument(
        "--bond-ckpt",
        type=Path,
        default=None,
        help="trained bond head (pipelines/train/bond_head.py). Given, the "
        "bond graph is read off the decoded chemistry instead of off the "
        "decoded distances -- perception recovers 31% of the true bonds at "
        "the error the decoder makes, the head 72%. Molecule identity is read "
        "off that graph, so this is not only about the bond list.",
    )
    parser.add_argument(
        "--scoring-radii",
        action="store_true",
        help="give --place-before-refine the radii Vina scores with "
        "(rigid_fit.VINA_RADII) instead of Bondi radii, so what the rigid "
        "placement pushes apart is what the scoring function charges for. "
        "Diagnostic only: the placement itself is a numerical optimiser and "
        "does not belong in an ML-only comparison.",
    )
    args = parser.parse_args()
    seed_from_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Which weights produced these molecules is not recoverable from the SDF,
    # and a benchmark tree outlives the shell that wrote it. Training runs
    # already drop a run.json beside their checkpoints; a generation run is
    # every bit as much a thing whose numbers get reported, so it drops one
    # too -- reconstructing the checkpoint set for the 97-target tree from
    # file mtimes cost an afternoon exactly once.
    write_manifest(args.out_dir, seed=getattr(args, "seed", None))
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

    # ---- optional iterative decoder ----
    mlm = mlm_mask_id = None
    if args.mlm_ckpt is not None:
        from prolit.model.mlm_decode import load_mlm  # noqa: PLC0415

        mlm, mlm_mask_id = load_mlm(str(args.mlm_ckpt), device)
        logger.info("Iterative decoder loaded from %s", args.mlm_ckpt)

    # ---- optional bond head ----
    bond_head = None
    if args.bond_ckpt is not None:
        from prolit.model.bond_head import load_bond_head  # noqa: PLC0415

        bond_head = load_bond_head(str(args.bond_ckpt), device)
        logger.info("Bond head loaded from %s", args.bond_ckpt)

    # ---- optional pose refiner ----
    refiner = None
    pocket_ctx = None
    if args.refine_ckpt is not None:
        # Which module class the checkpoint belongs to is decided by what is
        # IN it, not by a flag: a torsion refiner has a torsion head and a
        # free-displacement one does not. Loading the wrong class silently
        # drops the head and refines with an untrained backbone.
        import torch as _torch  # noqa: PLC0415

        from prolit.model.pose_refiner import PoseRefinerModule  # noqa: PLC0415
        from prolit.model.torsion_refiner import TorsionRefinerModule  # noqa: PLC0415

        _sd = _torch.load(args.refine_ckpt, map_location="cpu", weights_only=False)
        _keys = _sd.get("state_dict", {})
        _cls = (
            TorsionRefinerModule
            if any(k.startswith("net.torsion_head") for k in _keys)
            else PoseRefinerModule
        )
        logger.info("Pose refiner: %s", _cls.__name__)
        refiner = (
            _cls.load_from_checkpoint(args.refine_ckpt, map_location=device)
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
        mol: object | None = None,
    ) -> None:
        comp = Counter(elements)
        formula = "".join(f"{e}{n}" for e, n in sorted(comp.items()))
        has_unknown = any(e not in _REAL_ELEMENTS for e in elements)
        n_frag = _n_fragments(len(elements), bonds)
        dockable = not has_unknown and _MIN_ATOMS <= len(elements) <= _MAX_ATOMS
        tag = "ref" if idx < 0 else f"gen_{idx}"
        if mol is not None:
            mol.SetProp("_Name", tag)
            sdf_f.write(Chem.MolToMolBlock(mol) + "$$$$\n")
        else:
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
        round(args.min_atoms_frac * len(ref_elems))
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
        procs = LogitsProcessorList()
        if args.anchor_temperature and args.anchor_atoms > 0:
            procs.append(
                _AnchorTemperature(
                    prompt_len,
                    args.anchor_atoms,
                    args.temperature / args.anchor_temperature,
                )
            )
        gen = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            logits_processor=procs,
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
            codes_before = list(codes)
            if mlm is not None and codes and args.iter_mode == "cold":
                # Keep only the LENGTH the causal model chose, so this arm
                # differs from the control in the decode order and nothing else.
                from prolit.model.mlm_decode import cold_decode  # noqa: PLC0415

                codes = cold_decode(
                    mlm,
                    mlm_mask_id,
                    prot_codes,
                    len(codes),
                    codebook_size=args.codebook_size,
                    rounds=args.iter_rounds,
                    temperature=args.temperature,
                    device=device,
                )
                codes_before = list(codes)
            elif mlm is not None and codes:
                # The causal model chose these left to right; let the
                # bidirectional one re-decide the ones it is least sure of now
                # that the whole molecule is on the table.
                codes = refine_codes(
                    mlm,
                    mlm_mask_id,
                    prot_codes,
                    codes,
                    codebook_size=args.codebook_size,
                    rounds=args.iter_rounds,
                    frac=args.iter_frac,
                order=args.iter_order,
                    device=device,
                )
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
            coords, elems, charges, num_h, aromatic, lig_feat = decode_codes(
                codes, frame, refiner=refiner, pocket_ctx=pocket_ctx,
                bond_head=bond_head,
                reference_codes=codes_before if mlm is not None else None,
            )
            # Perception knows distances, not valences, so two atoms the
            # decoder placed too close arrive as an extra bond and the whole
            # molecule falls out of the chemistry-aware path. Prune first.
            # Join first, prune second: the joins are bridges and the pruning
            # will not cut a bridge, so the two repairs compose.
            # The head reads the graph again from the *refined* coordinates:
            # perception improves as the pose does, and this is the graph the
            # molecule's identity is built from.
            perceived = _perceive_bonds(bond_head, coords, lig_feat, elems, device)
            bonds = prune_to_valence(
                elems, charges, num_h,
                connect_fragments(elems, perceived, coords),
                coords,
            )
            mol = mol_from_decoded(
                elems, charges, num_h, coords, bonds,
                perceived=True, aromatic=aromatic,
            )
            emit(
                idx, elems, coords, bonds,
                terminated=terminated, n_codes=len(codes), mol=mol,
            )
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
