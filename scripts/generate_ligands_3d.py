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
    CrossDockedConfig,
    LMTrainingConfig,
    PocketExtractionConfig,
    VQVAETrainingConfig,
)
from src.data.descriptors import ComplexDescriptorDataModule  # noqa: E402
from src.model.lm_module import LigandLMModule  # noqa: E402
from src.model.vqvae_module import VQVAEModule  # noqa: E402
from src.tokenizers.descriptor_schema import (  # noqa: E402
    LIGAND_DESCRIPTOR_DIM,
    LIGAND_ELEMENT_VOCAB,
    LIGAND_LAYOUT,
    fields_by_name,
)
from src.tokenizers.ligand import LigandDescriptor, parse_sdf_text  # noqa: E402
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
    _compute_canonical_frame,
    extract_pocket,
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
def _decode_ligand(
    codes: list[int],
    ligand_vqvae: object,
    norm_stats: dict[str, torch.Tensor],
    frame: tuple[np.ndarray, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """Decode generated ligand codes to (coords in global frame, elements)."""
    idx = torch.tensor(codes, dtype=torch.long, device=device)
    outputs = ligand_vqvae.decode_to_outputs(idx)
    coord_field = fields_by_name(LIGAND_LAYOUT)["coord"]
    cmean = norm_stats["ligand_mean"][coord_field.start : coord_field.end]
    cstd = norm_stats["ligand_std"][coord_field.start : coord_field.end]
    coord_denorm = outputs["coord"] * cstd + cmean
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


def main() -> None:  # noqa: PLR0915
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

    # ---- normalization stats (v4 = what the VQ-VAE/LM were built on) ----
    dm = ComplexDescriptorDataModule(
        VQVAETrainingConfig(), CrossDockedConfig(data_dir=PROJECT_ROOT / "data")
    )
    dm.cache_dir = args.cache_dir
    dm.setup()
    norm_stats = {k: v.to(device) for k, v in dm.norm_stats.items()}

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
    protein_desc_calc = BackboneSphericalDescriptor()

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
            res = _pocket_codes(
                rec_path,
                mol,
                pocket_config,
                protein_desc_calc,
                vqvae.protein_vqvae,
                norm_stats,
                device,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("skip %s: %s", row["complex_dir"], e)
            continue
        if res is None:
            continue
        prot_codes, frame, gt_coords, gt_elems = res

        prompt = [
            BOS_ID,
            P_OPEN_ID,
            *(vocab.protein_offset + c for c in prot_codes),
            P_CLOSE_ID,
            L_OPEN_ID,
        ]
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
            codes = [
                t - lig_lo
                for t in lig_tok
                if lig_lo <= t < lig_lo + vocab.ligand_codebook_size
            ]
            total += 1
            if not codes:
                logger.info("  s%d: EMPTY/invalid", k)
                continue
            coords, elems = _decode_ligand(
                codes, vqvae.ligand_vqvae, norm_stats, frame, device
            )
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
