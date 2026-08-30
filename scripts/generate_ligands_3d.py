"""Generate ligands for real test pockets and decode them to 3D PDB.

Pipeline per test complex:
1. Extract the pocket from the receptor PDB, compute its canonical frame, and
   encode the backbone with the protein VQ-VAE -> protein structure codes.
2. Prompt the LM with ``<bos><p> pocket-codes </p><l>`` and autoregressively
   sample ligand structure codes until ``</l>``.
3. Decode the generated ligand codes with the ligand VQ-VAE (codes ->
   spherical coord head + element head), place atoms in the *real pocket frame*
   (global coords), and write a PDB with the receptor + generated ligand.

Writes, per pocket:
    {tag}_gt.pdb     receptor + native (ground-truth) ligand, for reference
    {tag}_s{k}.pdb   receptor + generated-ligand sample k, docked in the pocket

Run on a GPU node::

    uv run python scripts/generate_ligands_3d.py \
        --lm-ckpt <path/to/lm.ckpt> --num-pockets 3 --num-samples 2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
import torch

from prolit.chem.pdb_io import (
    infer_bonds,
    write_full_protein_pdb,
)
from prolit.config import (
    CLMTrainingConfig,
    PocketExtractionConfig,
)
from prolit.data.descriptors import read_mol_from_tar
from prolit.model.clm_module import ProLITCLMModule
from prolit.seeding import add_seed_argument, seed_from_args
from prolit.tokenizers.atom import (
    ProteinAtomDescriptor,
    precompute_receptor_atom_features,
)
from prolit.tokenizers.descriptor_schema import (
    ATOM_LAYOUT,
    LIGAND_CHARGE_VOCAB,
    LIGAND_ELEMENT_VOCAB,
    LIGAND_NUMH_VOCAB,
    fields_by_name,
)
from prolit.tokenizers.geometry import spherical_to_cartesian_np
from prolit.tokenizers.lm_vocab import (
    L_CLOSE_ID,
    PAD_ID,
    AtomLMVocab,
)
from prolit.tokenizers.loaders import load_atom_vqvae as _load_atom_vqvae
from prolit.tokenizers.protein import (
    compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    precompute_pocket_atom_candidates,
)

# Repository root, used only for default data/output locations. ``prolit`` is
# an installed package, so nothing needs to be put on sys.path.
if TYPE_CHECKING:
    from collections.abc import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# All-atom path (unified single-codebook VQ-VAE + AtomLMVocab)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _pocket_codes_atom(  # noqa: PLR0913
    rec_path: Path,
    mol: dict,
    pocket_config: PocketExtractionConfig,
    prot_desc: ProteinAtomDescriptor,
    atom_vqvae: object,
    norm_stats: dict[str, torch.Tensor],
    device: torch.device,
    *,
    receptor_cache: dict[str, tuple] | None = None,
) -> tuple[list[int], tuple[np.ndarray, np.ndarray], np.ndarray, list[str]] | None:
    """All-atom counterpart of :func:`_pocket_codes`.

    Encodes every heavy atom of the pocket residues with the unified atom
    VQ-VAE (one codebook shared with the ligand). Mirrors the encode side of
    :func:`prolit.data.atom_descriptors._atom_process_pose` so the codes match the
    training-time tokenization exactly.
    """
    if not mol["atoms"]:
        return None
    heavy = np.array(
        [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"], dtype=np.float32
    )
    if len(heavy) == 0:
        return None

    key = str(rec_path)
    cached = receptor_cache.get(key) if receptor_cache is not None else None
    if cached is None:
        precomputed = precompute_pocket_atom_candidates(rec_path)
        feats = precompute_receptor_atom_features(rec_path)
        if receptor_cache is not None:
            receptor_cache[key] = (precomputed, feats)
    else:
        precomputed, feats = cached

    pocket = extract_pocket_atoms_from_candidates(precomputed, heavy, pocket_config)
    if pocket is None or pocket.atom_coords.shape[0] == 0:
        return None
    centroid, rotation = compute_canonical_frame(pocket.ca_coords.astype(np.float64))
    frame = (centroid, rotation)

    prot_arr, _ = prot_desc.compute(pocket, feats, frame)
    prot_t = torch.from_numpy(prot_arr).to(device)
    prot_norm = (prot_t - norm_stats["atom_mean"]) / norm_stats["atom_std"]
    codes = atom_vqvae.encode(prot_norm).cpu().tolist()

    gt_elems = [a[0] for a in mol["atoms"] if a[0] != "H"]
    return codes, frame, heavy.astype(np.float64), gt_elems


@torch.no_grad()
def _perceive_bonds(
    bond_head: object | None,
    canonical: np.ndarray,
    lig_feat: np.ndarray,
    elements: list[str],
    device: torch.device,
) -> list[tuple[int, int]]:
    """The bond graph the refiner is given, from the head when there is one.

    Distance perception recovers 31% of the true bonds at the error the decoder
    makes; the head, reading the same atoms' decoded chemistry, recovers 72%
    (:mod:`prolit.model.bond_head`). Falling back to distance keeps every
    existing checkpoint runnable without one.
    """
    if bond_head is None:
        return infer_bonds(elements, canonical)
    from prolit.model.bond_head import bonds_from_head  # noqa: PLC0415

    return bonds_from_head(bond_head, canonical, lig_feat, device=device)


def _rigid_part(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Keep only the rigid part of the displacement the refiner predicted.

    The refiner cannot compress what it is only allowed to move. Measured over
    60 targets, every refiner in this family trades chemistry for contact:
    bonds out of tolerance go 10.0% (no refiner) -> 48.1% (``refit_press0.6``)
    / 50.3% (``refit_deploy``) while the clash rate goes 36.0% -> 25.0% /
    28.3%. Both numbers are the same act -- a smaller molecule overlaps less --
    and press corruption is not the cause, since the non-press refiner
    compresses slightly harder.

    This consults no objective function: it is a closed-form projection of the
    model's own output onto SE(3), not an optimisation against Vina or a force
    field. A rigid motion cannot change a bond length or angle, so the molecule
    keeps the chemistry the decoder gave it and the placement the refiner
    predicted.
    """
    from prolit.model.mlm_decode import kabsch_onto  # noqa: PLC0415

    if before.shape != after.shape:
        return after
    return kabsch_onto(before, after).astype(np.float32)


def _rigid_torsion_part(
    before: np.ndarray, after: np.ndarray, bonds: np.ndarray
) -> np.ndarray:
    """Project the refiner's displacement onto rigid motion + tree torsions.

    ``_rigid_part`` throws away everything the refiner said about internal
    motion, and that is most of what it said: over 94 targets the SE(3) part
    alone recovers 4.40 of ``refit_press0.6``'s 6.66 kcal. The remaining 2.26
    is bought by breaking bonds -- but Vina's own local optimiser, which moves
    nothing except translation, rotation and torsions, recovers 9.91 kcal from
    these very poses. So the missing motion is *expressible* without touching a
    bond; the free-displacement head simply does not express it that way.

    Like ``_rigid_part`` this consults no objective function. The angles are
    read off the model's own output in closed form -- the dihedral each
    rotatable bond turned through between the pose the refiner was given and
    the pose it returned -- and re-applied to the original. Bond lengths and
    angles are unchanged by construction, so the molecule keeps the decoder's
    chemistry while following the refiner's placement as far as a real molecule
    can follow it.
    """
    from prolit.chem.torsions import rotatable_bonds, torsion_delta  # noqa: PLC0415
    from prolit.model.mlm_decode import kabsch_onto  # noqa: PLC0415
    from prolit.model.torsion_transform import apply_torsions  # noqa: PLC0415

    if before.shape != after.shape:
        return after
    pairs, masks = rotatable_bonds(bonds, before.shape[0])
    if len(pairs) == 0:
        return _rigid_part(before, after)
    # Torsions first, rigid second. Dihedrals are rotation-invariant, so the
    # angles can be read straight off the pair; but superposing first would fit
    # the rigid part to a difference the torsions have not yet removed, and the
    # two motions would compose into neither.
    angles = torsion_delta(before, after, bonds, pairs)
    turned = (
        apply_torsions(  # the training-time transform, so it speaks torch
            torch.from_numpy(np.ascontiguousarray(before)).float(),
            torch.from_numpy(np.ascontiguousarray(pairs)).long(),
            torch.from_numpy(np.ascontiguousarray(masks)).bool(),
            torch.from_numpy(np.ascontiguousarray(angles)).float(),
        )
        .numpy()
        .astype(np.float64)
    )
    return kabsch_onto(turned, after).astype(np.float32)


def _project_displacement(
    mode: str,
    before: np.ndarray,
    after: np.ndarray,
    bonds_of_before: Callable[[], np.ndarray],
) -> np.ndarray:
    """Dispatch to the requested projection of the refiner's displacement."""
    if mode == "rigid":
        return _rigid_part(before, after)
    if mode == "torsion":
        return _rigid_torsion_part(before, after, bonds_of_before())
    return after


def _decode_ligand_atom(  # noqa: PLR0913
    codes: list[int],
    atom_vqvae: object,
    norm_stats: dict[str, torch.Tensor],
    frame: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    *,
    refiner: object | None = None,
    pocket_ctx: tuple | None = None,
    place_first: bool = False,
    scoring_radii: bool = False,
    refine_rounds: int = 1,
    bond_head: object | None = None,
    reference_codes: list[int] | None = None,
    reconcile_mode: str = "off",
    refine_project: str = "none",
) -> tuple[np.ndarray, list[str], list[int], list[int], list[bool], np.ndarray]:
    """Decode codes to global coords, elements, charges, numH and aromaticity.

    ``charge`` and ``numH`` come back alongside the elements because they are
    what turns a bare connectivity graph into a molecule: the orders of atom
    *i*'s bonds must sum to ``valence(element, charge) - numH(i)``, which is
    exactly what :func:`prolit.chem.bond_orders.assign_bond_orders` solves.
    Dropping them forces every bond to be written single, and an aromatic ring
    written as a saturated one fails PoseBusters three separate ways -- its
    bonds are 1.39 A where a single bond wants 1.53, its angles are 120 where
    sp3 wants 109.5, and a ring declared non-aromatic must not be planar.

    The unified coord head is the same 4-D spherical ``(r, θ, sin φ, cos φ)`` in
    the pocket canonical frame as the ligand VQ-VAE, so reconstruction mirrors
    :func:`prolit.tokenizers.atom.atom_descriptor_to_coords`. When ``refiner`` +
    ``pocket_ctx`` (``(pocket canonical coords, pocket node features)``) are
    given, the E(3)-equivariant pose refiner cleans the pose before the global
    transform.

    ``place_first`` slides the decoded ligand off the pocket wall as a rigid
    body *before* handing it to the refiner. The refiner was trained on crystal
    ligands with a local jitter, but the decoder hands it a chemically fine
    molecule about 2 A out of place -- so without this it is asked to undo an
    error it never saw. Removing the global part first puts its input back in
    the distribution it was fitted on, and leaves it the local part it can
    actually correct. See :mod:`prolit.chem.rigid_fit`.

    ``refine_rounds`` repeats place-then-refine. Feeding a flow-matching model
    its own output is normally out of distribution, but the placement in front
    of each round is exactly what puts it back in -- so the question of whether
    a second pass helps is an empirical one rather than a broken one.
    """
    idx = torch.tensor(codes, dtype=torch.long, device=device)
    outputs = atom_vqvae.decode_to_outputs(idx)
    coord_field = fields_by_name(ATOM_LAYOUT)["coord"]
    cmean = norm_stats["atom_mean"][coord_field.start : coord_field.end]
    cstd = norm_stats["atom_std"][coord_field.start : coord_field.end]
    coord_denorm = (outputs["coord"] * cstd + cmean).cpu().numpy()
    canonical = spherical_to_cartesian_np(coord_denorm)
    centroid, rotation = frame
    from prolit.model.pose_refiner import (  # noqa: PLC0415
        LIG_CHEM_HEADS,
        ligand_feats_from_heads,
        refine_ligand_canonical,
    )

    # The node-feature block is the decoder's six chemistry heads laid out the
    # way both the refiner and the bond head read them. Built once here rather
    # than inside the refiner branch, because the bond head needs it whether or
    # not a refiner ran.
    chem = {h: outputs[h].argmax(dim=-1).cpu().numpy() for h in LIG_CHEM_HEADS}
    lig_feat = ligand_feats_from_heads(chem, canonical.shape[0])
    canonical_before = canonical.copy()
    if refiner is not None and pocket_ctx is not None:
        elems_r = [
            LIGAND_ELEMENT_VOCAB[i] if LIGAND_ELEMENT_VOCAB[i] != "OTHER" else "X"
            for i in chem["element"]
        ]
        pkt_canon, pkt_feat, *pkt_rest = pocket_ctx
        place = None
        if place_first and pkt_rest:
            from prolit.chem.rigid_fit import (  # noqa: PLC0415
                rigid_pocket_fit,
                vdw_radii,
            )

            # Bondi radii make the objective zero 0.4 A inside the surface Vina
            # rewards, and its gradient dies there -- so the fitter stops with
            # every atom pressed too deep. Measured over 99 targets: Vina's
            # repulsion term is 7.50 for these poses against 1.64 for FLOWR,
            # while every attractive term is better than FLOWR's.
            lig_radii = vdw_radii(elems_r, scoring=scoring_radii)
            pkt_coords = np.asarray(pkt_canon, dtype=np.float64)
            pkt_radii = pkt_rest[0]
            if scoring_radii and len(pkt_rest) > 1:
                pkt_radii = pkt_rest[1]

            def place(xyz: np.ndarray) -> np.ndarray:
                fit = rigid_pocket_fit(
                    xyz.astype(np.float64), lig_radii, pkt_coords, pkt_radii
                )
                return fit.apply(xyz.astype(np.float64))

        for _ in range(max(1, refine_rounds)):
            if place is not None:
                canonical = place(canonical)
            bonds_r = np.asarray(
                _perceive_bonds(bond_head, canonical, lig_feat, elems_r, device),
                dtype=np.int64,
            ).reshape(-1, 2)
            canonical = refine_ligand_canonical(
                refiner,
                canonical.astype(np.float32),
                lig_feat,
                pkt_canon,
                pkt_feat,
                bonds=bonds_r,
                device=device,
            )
    if refiner is not None and pocket_ctx is not None and refine_project != "none":
        canonical = _project_displacement(
            refine_project,
            canonical_before,
            canonical,
            # Bonds are perceived on the pose the refiner was GIVEN, not the
            # one it returned: a distorted output perceives a different bond
            # graph (52.2% of molecules change SMILES under press0.6), and the
            # projection has to move the molecule the decoder actually made.
            lambda: np.asarray(
                _perceive_bonds(
                    bond_head, canonical_before, lig_feat, elems_r, device
                ),
                dtype=np.int64,
            ).reshape(-1, 2),
        )
    if reference_codes is not None and reconcile_mode != "off":
        # The decoder is contextual, so editing one code moves every atom. Only
        # the edited atoms are meant to move (2.20 A); the rest drift 0.24 A.
        from prolit.model.mlm_decode import reconcile  # noqa: PLC0415

        ref_out = atom_vqvae.decode_to_outputs(
            torch.tensor(reference_codes, dtype=torch.long, device=device)
        )
        ref_xyz = spherical_to_cartesian_np(
            (ref_out["coord"] * cstd + cmean).cpu().numpy()
        )
        changed = [
            i for i, (a, b) in enumerate(zip(reference_codes, codes, strict=False))
            if a != b
        ]
        if changed and ref_xyz.shape == canonical.shape:
            canonical = reconcile(
                ref_xyz, canonical, changed, mode=reconcile_mode
            ).astype(np.float32)

    coords = canonical @ rotation + centroid
    elem_idx = outputs["element"].argmax(dim=-1).cpu().numpy()
    elements = [
        LIGAND_ELEMENT_VOCAB[i] if LIGAND_ELEMENT_VOCAB[i] != "OTHER" else "X"
        for i in elem_idx
    ]
    charges = [
        LIGAND_CHARGE_VOCAB[i] for i in outputs["charge"].argmax(dim=-1).cpu().numpy()
    ]
    num_h = [
        LIGAND_NUMH_VOCAB[i] for i in outputs["numH"].argmax(dim=-1).cpu().numpy()
    ]
    # The aromatic head rides out with the rest. It used to stop at the
    # refiner's node features, which is how a model that predicts aromaticity
    # ended up emitting molecules whose median fsp3 was 1.00.
    aromatic = [bool(i) for i in outputs["aromatic"].argmax(dim=-1).cpu().numpy()]
    return coords, elements, charges, num_h, aromatic, lig_feat


def load_atom_norm_stats(
    path: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load the unified all-atom normalization stats (atom_mean / atom_std)."""
    stats = torch.load(path, weights_only=False)
    return {k: v.to(device) for k, v in stats.items()}


def load_atom_vqvae(
    ckpt: Path,
    codebook_size: int,
    device: torch.device,
) -> object:
    """Load the frozen all-atom VQ-VAE (returns the inner TransformerVQVAE)."""
    module = _load_atom_vqvae(str(ckpt), device, codebook_size=codebook_size)
    return module.vqvae


def load_atom_lm(ckpt: str, codebook_size: int, device: torch.device) -> object:
    """Load an all-atom LM checkpoint over a single ``codebook_size`` range.

    A run trained with an auxiliary head (``--centroid-loss-weight``) carries
    that head's weights in the checkpoint. Generation only ever uses
    ``self.model``, so those keys are surplus rather than a mismatch -- but a
    strict load rejects the whole file over them. The head is only built when
    the config asks for it, so recreate it from what the checkpoint has.
    """
    config = CLMTrainingConfig()
    config.model.atom_codebook_size = codebook_size
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if any(k.startswith("centroid_head.") for k in state.get("state_dict", {})):
        saved = state.get("hyper_parameters", {}).get("config")
        config.centroid_loss_weight = float(
            getattr(saved, "centroid_loss_weight", 1.0) or 1.0
        )
        config.code_mean_coords = str(getattr(saved, "code_mean_coords", "") or "")
    return (
        ProLITCLMModule.load_from_checkpoint(ckpt, config=config, map_location=device)
        .eval()
        .to(device)
        .model
    )


def main() -> None:  # noqa: PLR0915, C901
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "generated_3d"
    )
    parser.add_argument("--num-pockets", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    add_seed_argument(parser, default=0)
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
    seed_from_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- models + tokenizer setup (joint tokenizer, or the separate ablation) ----
    vqvae_ckpt = (
        args.vqvae_ckpt
        if Path(args.vqvae_ckpt).is_absolute()
        else PROJECT_ROOT / args.vqvae_ckpt
    )
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
        model = load_atom_lm(args.lm_ckpt, combined_codebook_size, device)
        # Encode uses the protein VQ with the protein modality's stats; decode uses
        # the ligand VQ with the ligand modality's stats (exposed by SeparateVQVAE).
        protein_norm = load_atom_norm_stats(args.separate_protein_norm, device)
        ligand_norm = separate_vqvae.ligand_norm_stats
        vocab = AtomLMVocab(codebook_size=combined_codebook_size)
        # Ligand block tokens occupy [offset + Pc, offset + 2*Pc). Filter to that
        # window but subtract only ``offset`` so the extracted codes stay in
        # COMBINED space -- exactly what SeparateVQVAE.decode_to_outputs expects.
        code_lo = vocab.offset + args.codebook_size
        code_hi = vocab.offset + vocab.codebook_size
        code_base = vocab.offset
        prot_atom_desc = ProteinAtomDescriptor()

        def encode_pocket(rec_path: Path, mol: dict):  # noqa: ANN202
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

        def decode_codes(codes, frame):  # noqa: ANN001, ANN202
            return _decode_ligand_atom(
                codes, separate_vqvae, ligand_norm, frame, device
            )
    else:
        atom_vqvae = load_atom_vqvae(vqvae_ckpt, args.codebook_size, device)
        model = load_atom_lm(args.lm_ckpt, args.codebook_size, device)
        norm_stats = load_atom_norm_stats(args.norm_stats, device)
        vocab = AtomLMVocab(codebook_size=args.codebook_size)
        code_lo, code_hi = vocab.offset, vocab.offset + vocab.codebook_size
        code_base = code_lo
        prot_atom_desc = ProteinAtomDescriptor()

        def encode_pocket(rec_path: Path, mol: dict):  # noqa: ANN202
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

        def decode_codes(codes, frame):  # noqa: ANN001, ANN202
            return _decode_ligand_atom(codes, atom_vqvae, norm_stats, frame, device)

    def build_prompt(prot_codes: list[int]) -> list[int]:
        # build_sequence(prot, [])[:-2] drops the trailing </l><eos>, leaving
        # <bos><p> prot </p><l>.
        return vocab.build_sequence(prot_codes, [])[:-2]

    # ---- test pool ----
    hub = PROJECT_ROOT / "data" / "hub_cache"
    mdf = pq.read_table(hub / "repo" / "manifest.parquet").to_pandas()
    test_df = mdf[
        (mdf["source_type"] == "cdonly") & (mdf["cdonly_fold0"] == "test")
    ].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(test_df))
    receptor_dir = hub / "receptors"
    repo_dir = hub / "repo"

    pocket_config = PocketExtractionConfig()

    done = 0
    valid = 0
    total = 0
    for oi in order:
        if done >= args.num_pockets:
            break
        row = test_df.iloc[int(oi)]
        rec_path = receptor_dir / f"{row['complex_dir']}/{row['receptor_pdb']}"
        if not rec_path.exists():
            continue
        mol = read_mol_from_tar(repo_dir, int(row["shard_idx"]), int(row["pair_idx"]))
        if mol is None:
            continue
        try:
            res = encode_pocket(rec_path, mol)
        except Exception as e:  # noqa: BLE001
            logger.warning("skip %s: %s", row["complex_dir"], e)
            continue
        if res is None:
            continue
        prot_codes, frame, gt_coords, gt_elems = res

        prompt = build_prompt(prot_codes)
        prompt_ids = torch.tensor([prompt], device=device).repeat(args.num_samples, 1)
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

        tag = f"{done:03d}_pair{int(row['pair_idx']):07d}"
        write_full_protein_pdb(
            args.out_dir / f"{tag}_gt.pdb",
            rec_path,
            gt_elems,
            gt_coords,
            ligand_bonds=infer_bonds(gt_elems, gt_coords),
        )
        logger.info(
            "=== pocket %d (%s) | %d res | GT %d atoms ===",
            done,
            row["complex_dir"],
            len(prot_codes),
            len(gt_elems),
        )
        for k in range(gen.shape[0]):
            out_tokens = gen[k].tolist()[len(prompt) :]
            terminated = L_CLOSE_ID in out_tokens
            lig_tok = (
                out_tokens[: out_tokens.index(L_CLOSE_ID)] if terminated else out_tokens
            )
            codes = [t - code_base for t in lig_tok if code_lo <= t < code_hi]
            total += 1
            if not codes:
                logger.info("  s%d: EMPTY/invalid", k)
                continue
            coords, elems, _charges, _num_h, _arom, _feat = decode_codes(codes, frame)
            bonds = infer_bonds(elems, coords)
            write_full_protein_pdb(
                args.out_dir / f"{tag}_s{k}.pdb",
                rec_path,
                elems,
                coords,
                ligand_bonds=bonds,
            )
            valid += int(terminated)
            from collections import Counter  # noqa: PLC0415

            comp = ",".join(f"{e}{n}" for e, n in sorted(Counter(elems).items()))
            logger.info(
                "  s%d: %d atoms [%s] %d bonds -> %s_s%d.pdb",
                k,
                len(elems),
                comp,
                len(bonds),
                tag,
                k,
            )
        done += 1

    logger.info(
        "\nWrote %d pockets (%d gen samples, %d cleanly terminated) to %s",
        done,
        total,
        valid,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
