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
import gzip
import logging
import sys
import tarfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.write_reconstruction_pdbs import (  # noqa: E402
    _reconstruct_descriptor_from_coord_head,
    infer_bonds,
    write_full_protein_pdb,
)
from src.config import (  # noqa: E402
    AtomVQVAETrainingConfig,
    CrossDockedConfig,
    LMTrainingConfig,
    PocketExtractionConfig,
    VQVAETrainingConfig,
)
from src.data.descriptors import ComplexDescriptorDataModule  # noqa: E402
from src.model.lm_module import LigandLMModule  # noqa: E402
from src.model.vqvae_module import AtomVQVAEModule, VQVAEModule  # noqa: E402
from src.tokenizers.atom import (  # noqa: E402
    ProteinAtomDescriptor,
    precompute_receptor_atom_features,
)
from src.tokenizers.descriptor_schema import (  # noqa: E402
    ATOM_LAYOUT,
    LIGAND_DESCRIPTOR_DIM,
    LIGAND_ELEMENT_VOCAB,
    LIGAND_LAYOUT,
    SOURCE_LIGAND_IDX,
    fields_by_name,
)
from src.tokenizers.geometry import spherical_to_cartesian_np  # noqa: E402
from src.tokenizers.ligand import (  # noqa: E402
    LigandDescriptor,
    parse_sdf_text,
    solve_ligand_coords,
)
from src.tokenizers.lm_vocab import (  # noqa: E402
    L_CLOSE_ID,
    PAD_ID,
    AtomLMVocab,
    LMVocab,
)
from src.tokenizers.protein import (  # noqa: E402
    BackboneSphericalDescriptor,
    _compute_canonical_frame,
    extract_pocket,
    extract_pocket_atoms_from_candidates,
    precompute_pocket_atom_candidates,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _read_mol_from_tar(repo_dir: Path, shard_idx: int, pair_idx: int) -> dict | None:
    """Read one ligand SDF straight from its packed tar shard (no extraction)."""
    tar_path = repo_dir / "ligands" / f"{int(shard_idx):06d}.tar"
    if not tar_path.exists():
        return None
    member = f"{int(pair_idx):07d}.sdf.gz"
    with tarfile.open(tar_path, "r") as tf:
        try:
            info = tf.getmember(member)
        except KeyError:
            info = next(
                (m for m in tf.getmembers() if m.name.rsplit("/", 1)[-1] == member),
                None,
            )
        if info is None:
            return None
        fileobj = tf.extractfile(info)
        if fileobj is None:
            return None
        text = gzip.decompress(fileobj.read()).decode("utf-8", "replace")
    mols = parse_sdf_text(text)
    return mols[0] if mols else None


@torch.no_grad()
def _pocket_codes(  # noqa: PLR0913
    rec_path: Path,
    mol: dict,
    pocket_config: PocketExtractionConfig,
    protein_desc_calc: BackboneSphericalDescriptor,
    protein_vqvae: object,
    norm_stats: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[list[int], tuple[np.ndarray, np.ndarray], np.ndarray, list[str]] | None:
    """Return (protein_codes, pocket_frame, gt_ligand_coords, gt_elements)."""
    if not mol["atoms"]:
        return None
    heavy = np.array(
        [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"], dtype=np.float32
    )
    if len(heavy) == 0:
        return None
    pocket_result = extract_pocket(rec_path, heavy, pocket_config)
    if pocket_result is None:
        return None
    backbone, pocket_seq, residue_ids = pocket_result
    centroid, rotation = _compute_canonical_frame(backbone[:, 1].astype(np.float64))
    frame = (centroid, rotation)

    prot_desc, _ = protein_desc_calc.compute(
        backbone,
        residue_ids,
        pocket_frame=frame,
        residue_names_one_letter=list(pocket_seq),
    )
    prot_t = torch.from_numpy(prot_desc).to(device)
    prot_norm = (prot_t - norm_stats["protein_mean"]) / norm_stats["protein_std"]
    codes = protein_vqvae.encode(prot_norm).cpu().tolist()

    gt_elems = [a[0] for a in mol["atoms"] if a[0] != "H"]
    return codes, frame, heavy.astype(np.float64), gt_elems


@torch.no_grad()
def _decode_ligand(  # noqa: PLR0913
    codes: list[int],
    ligand_vqvae: object,
    norm_stats: dict[str, torch.Tensor],
    frame: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    *,
    use_solve: bool = False,
    refiner: object | None = None,
    pocket_ctx: tuple | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Decode generated ligand codes to (coords in global frame, elements).

    By default uses ONLY the absolute coord head (the knn_offsets head, if the
    VQ-VAE has one, acts purely as a training-time regulariser). Set
    ``use_solve=True`` to instead reconstruct via the absolute+relative geometry
    solve (:func:`solve_ligand_coords`).

    When ``refiner`` (a ``PoseRefinerModule``) and ``pocket_ctx`` (``(pocket
    canonical coords, pocket node features)`` in this ``frame``) are given, the
    per-atom coord head is refined by the E(3)-equivariant pose refiner before
    the global-frame transform -- a learned, pocket-aware replacement for
    ``solve_ligand_coords`` that removes clashes/strain from the raw pose.
    """
    idx = torch.tensor(codes, dtype=torch.long, device=device)
    outputs = ligand_vqvae.decode_to_outputs(idx)
    fields = fields_by_name(LIGAND_LAYOUT)
    coord_field = fields["coord"]
    cmean = norm_stats["ligand_mean"][coord_field.start : coord_field.end]
    cstd = norm_stats["ligand_std"][coord_field.start : coord_field.end]
    coord_denorm = outputs["coord"] * cstd + cmean
    if refiner is not None and pocket_ctx is not None:
        from src.model.pose_refiner import (  # noqa: PLC0415
            LIG_CHEM_HEADS,
            ligand_feats_from_heads,
            refine_ligand_canonical,
        )

        canonical = spherical_to_cartesian_np(coord_denorm.cpu().numpy())
        chem = {h: outputs[h].argmax(dim=-1).cpu().numpy() for h in LIG_CHEM_HEADS}
        lig_feat = ligand_feats_from_heads(chem, canonical.shape[0])
        elems_r = [
            LIGAND_ELEMENT_VOCAB[i] if LIGAND_ELEMENT_VOCAB[i] != "OTHER" else "X"
            for i in chem["element"]
        ]
        bonds_r = np.asarray(infer_bonds(elems_r, canonical), dtype=np.int64).reshape(
            -1, 2
        )
        pkt_canon, pkt_feat = pocket_ctx
        refined = refine_ligand_canonical(
            refiner,
            canonical,
            lig_feat,
            pkt_canon,
            pkt_feat,
            bonds=bonds_r,
            device=device,
        )
        centroid, rotation = frame
        coords = refined @ rotation + centroid
    elif use_solve and "knn_offsets" in outputs:
        ko_field = fields["knn_offsets"]
        ko_mean = norm_stats["ligand_mean"][ko_field.start : ko_field.end]
        ko_std = norm_stats["ligand_std"][ko_field.start : ko_field.end]
        ko_denorm = outputs["knn_offsets"] * ko_std + ko_mean
        coords = solve_ligand_coords(
            coord_denorm.cpu().numpy(), ko_denorm.cpu().numpy(), frame
        )
    else:
        desc = _reconstruct_descriptor_from_coord_head(
            coord_denorm, LIGAND_DESCRIPTOR_DIM, coord_field.start, coord_field.length
        )
        coords = LigandDescriptor.descriptor_to_coords(desc, {}, pocket_frame=frame)
    elem_idx = outputs["element"].argmax(dim=-1).cpu().numpy()
    elements = [
        LIGAND_ELEMENT_VOCAB[i] if LIGAND_ELEMENT_VOCAB[i] != "OTHER" else "X"
        for i in elem_idx
    ]
    return coords, elements


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
    :func:`src.data.atom_descriptors._atom_process_pose` so the codes match the
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
    centroid, rotation = _compute_canonical_frame(pocket.ca_coords.astype(np.float64))
    frame = (centroid, rotation)

    prot_arr, _ = prot_desc.compute(pocket, feats, frame)
    prot_t = torch.from_numpy(prot_arr).to(device)
    prot_norm = (prot_t - norm_stats["atom_mean"]) / norm_stats["atom_std"]
    codes = atom_vqvae.encode(prot_norm).cpu().tolist()

    gt_elems = [a[0] for a in mol["atoms"] if a[0] != "H"]
    return codes, frame, heavy.astype(np.float64), gt_elems


@torch.no_grad()
def _decode_ligand_atom(  # noqa: PLR0913
    codes: list[int],
    atom_vqvae: object,
    norm_stats: dict[str, torch.Tensor],
    frame: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    *,
    source_idx: int | None = None,
    refiner: object | None = None,
    pocket_ctx: tuple | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Decode generated ligand codes (all-atom VQ-VAE) to global coords + elements.

    The unified coord head is the same 4-D spherical ``(r, θ, sin φ, cos φ)`` in
    the pocket canonical frame as the ligand VQ-VAE, so reconstruction mirrors
    :func:`src.tokenizers.atom.atom_descriptor_to_coords`. Pass ``source_idx``
    (ligand) for a split-codebook VQ so the ligand book is used. When ``refiner``
    + ``pocket_ctx`` (``(pocket canonical coords, pocket node features)``) are
    given, the E(3)-equivariant pose refiner cleans the pose before the global
    transform (same as the legacy :func:`_decode_ligand`).
    """
    idx = torch.tensor(codes, dtype=torch.long, device=device)
    outputs = atom_vqvae.decode_to_outputs(idx, source_idx)
    coord_field = fields_by_name(ATOM_LAYOUT)["coord"]
    cmean = norm_stats["atom_mean"][coord_field.start : coord_field.end]
    cstd = norm_stats["atom_std"][coord_field.start : coord_field.end]
    coord_denorm = (outputs["coord"] * cstd + cmean).cpu().numpy()
    canonical = spherical_to_cartesian_np(coord_denorm)
    centroid, rotation = frame
    if refiner is not None and pocket_ctx is not None:
        from src.model.pose_refiner import (  # noqa: PLC0415
            LIG_CHEM_HEADS,
            ligand_feats_from_heads,
            refine_ligand_canonical,
        )

        chem = {h: outputs[h].argmax(dim=-1).cpu().numpy() for h in LIG_CHEM_HEADS}
        lig_feat = ligand_feats_from_heads(chem, canonical.shape[0])
        elems_r = [
            LIGAND_ELEMENT_VOCAB[i] if LIGAND_ELEMENT_VOCAB[i] != "OTHER" else "X"
            for i in chem["element"]
        ]
        bonds_r = np.asarray(infer_bonds(elems_r, canonical), dtype=np.int64).reshape(
            -1, 2
        )
        pkt_canon, pkt_feat = pocket_ctx
        canonical = refine_ligand_canonical(
            refiner,
            canonical.astype(np.float32),
            lig_feat,
            pkt_canon,
            pkt_feat,
            bonds=bonds_r,
            device=device,
        )
    coords = canonical @ rotation + centroid
    elem_idx = outputs["element"].argmax(dim=-1).cpu().numpy()
    elements = [
        LIGAND_ELEMENT_VOCAB[i] if LIGAND_ELEMENT_VOCAB[i] != "OTHER" else "X"
        for i in elem_idx
    ]
    return coords, elements


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
    *,
    split: bool = False,
    ligand_codebook_size: int = 4096,
) -> object:
    """Load the frozen unified all-atom VQ-VAE (returns the inner TransformerVQVAE).

    ``split`` loads the split-codebook variant (protein ``codebook_size`` +
    ligand ``ligand_codebook_size`` books).
    """
    config = AtomVQVAETrainingConfig()
    config.atom.codebook_size = codebook_size
    if split:
        config.atom.split_codebook = True
        config.atom.ligand_codebook_size = ligand_codebook_size
    module = (
        AtomVQVAEModule.load_from_checkpoint(
            str(ckpt), config=config, map_location=device
        )
        .eval()
        .to(device)
    )
    return module.vqvae


def load_atom_lm(
    ckpt: str,
    codebook_size: int,
    device: torch.device,
    *,
    split: bool = False,
    ligand_codebook_size: int = 4096,
) -> object:
    """Load an all-atom LM checkpoint.

    Default = single-range atom vocab (specials + one atom codebook). ``split``
    = 2-range vocab (specials + protein ``codebook_size`` + ligand
    ``ligand_codebook_size``), matching the split-codebook tokenizer.
    """
    config = LMTrainingConfig()
    if split:
        config.model.protein_codebook_size = codebook_size
        config.model.ligand_codebook_size = ligand_codebook_size
    else:
        config.model.atom_codebook_size = codebook_size
    return (
        LigandLMModule.load_from_checkpoint(ckpt, config=config, map_location=device)
        .eval()
        .to(device)
        .model
    )


def main() -> None:  # noqa: PLR0912, PLR0915, C901
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--all-atom",
        action="store_true",
        help="Use the unified single-codebook all-atom tokenizer (AtomLMVocab + "
        "AtomVQVAEModule) instead of the legacy protein+ligand 2-codebook path.",
    )
    parser.add_argument(
        "--split-codebook",
        action="store_true",
        help="All-atom VQ with SPLIT codebooks (protein + ligand books, one "
        "shared descriptor/encoder/decoder) -> 2-range LMVocab. Implies all-atom.",
    )
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--ligand-codebook-size", type=int, default=4096)
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

    # ---- models + tokenizer setup (branch: all-atom vs legacy 2-codebook) ----
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
        from src.tokenizers.separate_vqvae import SeparateVQVAE  # noqa: PLC0415

        separate_vqvae = SeparateVQVAE.from_checkpoints(
            args.separate_protein_ckpt,
            args.separate_protein_norm,
            args.separate_ligand_ckpt,
            args.separate_ligand_norm,
            device,
            codebook_size=args.codebook_size,
        )
        combined_codebook_size = 2 * args.codebook_size
        model = load_atom_lm(
            args.lm_ckpt, combined_codebook_size, device, split=False
        )
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
                codes, separate_vqvae, ligand_norm, frame, device, source_idx=None
            )
    elif args.all_atom or args.split_codebook:
        split = args.split_codebook
        atom_vqvae = load_atom_vqvae(
            vqvae_ckpt,
            args.codebook_size,
            device,
            split=split,
            ligand_codebook_size=args.ligand_codebook_size,
        )
        model = load_atom_lm(
            args.lm_ckpt,
            args.codebook_size,
            device,
            split=split,
            ligand_codebook_size=args.ligand_codebook_size,
        )
        norm_stats = load_atom_norm_stats(args.norm_stats, device)
        if split:
            vocab = LMVocab(
                protein_codebook_size=args.codebook_size,
                ligand_codebook_size=args.ligand_codebook_size,
            )
            code_lo, code_hi = (
                vocab.ligand_offset,
                vocab.ligand_offset + vocab.ligand_codebook_size,
            )
            dec_source = SOURCE_LIGAND_IDX
        else:
            vocab = AtomLMVocab(codebook_size=args.codebook_size)
            code_lo, code_hi = vocab.offset, vocab.offset + vocab.codebook_size
            dec_source = None
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
            return _decode_ligand_atom(
                codes, atom_vqvae, norm_stats, frame, device, source_idx=dec_source
            )
    else:
        vqvae = (
            VQVAEModule.load_from_checkpoint(str(vqvae_ckpt), map_location=device)
            .eval()
            .to(device)
        )
        model = (
            LigandLMModule.load_from_checkpoint(
                args.lm_ckpt, config=LMTrainingConfig(), map_location=device
            )
            .eval()
            .to(device)
            .model
        )
        # ---- normalization stats (v4 = what the VQ-VAE/LM were built on) ----
        dm = ComplexDescriptorDataModule(
            VQVAETrainingConfig(), CrossDockedConfig(data_dir=PROJECT_ROOT / "data")
        )
        dm.cache_dir = args.cache_dir
        dm.setup()
        norm_stats = {k: v.to(device) for k, v in dm.norm_stats.items()}
        vocab = LMVocab()
        code_lo, code_hi = (
            vocab.ligand_offset,
            vocab.ligand_offset + vocab.ligand_codebook_size,
        )
        code_base = code_lo
        protein_desc_calc = BackboneSphericalDescriptor()

        def encode_pocket(rec_path: Path, mol: dict):  # noqa: ANN202
            return _pocket_codes(
                rec_path,
                mol,
                pocket_config,
                protein_desc_calc,
                vqvae.protein_vqvae,
                norm_stats,
                device,
            )

        def decode_codes(codes, frame):  # noqa: ANN001, ANN202
            return _decode_ligand(codes, vqvae.ligand_vqvae, norm_stats, frame, device)

    def build_prompt(prot_codes: list[int]) -> list[int]:
        # build_sequence(prot, [])[:-2] drops the trailing </l><eos>, leaving
        # <bos><p> prot </p><l>; identical for LMVocab and AtomLMVocab.
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
        mol = _read_mol_from_tar(repo_dir, int(row["shard_idx"]), int(row["pair_idx"]))
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
            coords, elems = decode_codes(codes, frame)
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
