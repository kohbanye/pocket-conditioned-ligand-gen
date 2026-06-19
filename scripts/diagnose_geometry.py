"""Multi-faceted geometry diagnostics: localize the distorted-shape problem.

Three arms, evaluated on the SAME held-out test pockets so they are directly
comparable, and dumped into one ``.npz`` for ``notebooks/geometry_diagnostics.py``:

1. **GT**         — the real ligand's true SDF coordinates (the ceiling).
2. **VQ-recon**   — real ligand -> descriptor -> ligand VQ-VAE encode -> decode.
                    Isolates the *tokenizer/decoder* error (no LM involved).
3. **LM-gen**     — pocket-conditioned LM samples codes -> decode.
                    Adds the *LM*'s contribution on top of the decoder.

The key question: is the ~78% clash a decoder/representation problem (VQ-recon
already clashes) or an LM problem (VQ-recon is clean but LM-sampled codes are
out-of-distribution)? Plus reconstruction RMSD (per-atom vs Kabsch = rigid-shift
vs internal-shape) and codebook usage (do LM codes match the real-code
distribution?).

Run on a GPU node::

    uv run python scripts/diagnose_geometry.py \
        --lm-ckpt pocket-ligand-lm/<finetune-run>/checkpoints/<best>.ckpt \
        --num-pockets 60 --num-samples 3
"""
# ruff: noqa: PLR2004, E501

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_generation import VALIDITY_METHODS, _metrics  # noqa: E402
from scripts.generate_ligands_3d import (  # noqa: E402
    _decode_ligand,
    _pocket_codes,
    _read_mol_from_tar,
)
from scripts.write_reconstruction_pdbs import kabsch_rmsd, per_atom_rmsd  # noqa: E402
from src.config import (  # noqa: E402
    CrossDockedConfig,
    LMTrainingConfig,
    PocketExtractionConfig,
    VQVAETrainingConfig,
)
from src.data.descriptors import ComplexDescriptorDataModule  # noqa: E402
from src.model.lm_module import LigandLMModule  # noqa: E402
from src.model.vqvae_module import VQVAEModule  # noqa: E402
from src.tokenizers.ligand import LigandDescriptor, _build_rdkit_mol  # noqa: E402
from src.tokenizers.lm_vocab import (  # noqa: E402
    BOS_ID,
    L_CLOSE_ID,
    L_OPEN_ID,
    P_CLOSE_ID,
    P_OPEN_ID,
    PAD_ID,
    LMVocab,
)
from src.tokenizers.protein import BackboneSphericalDescriptor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _pack(items: list[dict], methods: list[str]) -> dict:
    """Pack per-molecule metric dicts into flat arrays for the npz."""
    if not items:
        items = []
    out = {
        "n_atoms": np.array([m["n_atoms"] for m in items]),
        "min_pair_dist": np.array([m["min_pair_dist"] for m in items]),
        "bonds_per_atom": np.array([m["bonds_per_atom"] for m in items]),
        "n_components": np.array([m["n_components"] for m in items]),
        "centroid_dist": np.array([m["centroid_dist"] for m in items]),
        "bond_lengths": (
            np.concatenate([np.asarray(m["bond_lengths"]) for m in items])
            if items
            else np.array([])
        ),
        "elements": np.array([e for m in items for e in m["elements"]], dtype=object),
    }
    for meth in methods:
        out[f"v_{meth}"] = np.array(
            [bool(m.get(f"v_{meth}", False)) for m in items], dtype=bool
        )
    return out


@torch.no_grad()
def _reconstruct(  # noqa: PLR0913
    mol: dict,
    frame: tuple[np.ndarray, np.ndarray],
    lig_desc: LigandDescriptor,
    ligand_vqvae: object,
    norm_stats: dict[str, torch.Tensor],
    lig_mean: np.ndarray,
    lig_std: np.ndarray,
    device: torch.device,
    *,
    use_solve: bool = False,
) -> tuple[np.ndarray, list[str], list[int]] | None:
    """Real ligand -> descriptor -> encode -> decode. Returns (coords, elems, codes)."""
    desc, _elems, _meta = lig_desc.compute(mol["atoms"], mol["bonds"], pocket_frame=frame)
    if len(desc) == 0:
        return None
    desc_norm = torch.from_numpy((desc - lig_mean) / lig_std).float().to(device)
    codes = ligand_vqvae.encode(desc_norm)
    code_list = codes.cpu().tolist()
    coords, elems = _decode_ligand(
        code_list, ligand_vqvae, norm_stats, frame, device, use_solve=use_solve
    )
    return coords, elems, code_list


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm-ckpt", type=str, default=None)
    parser.add_argument(
        "--recon-only",
        action="store_true",
        help="Skip the LM/gen arm (GT + VQ-recon only). Use when evaluating a "
        "newly-trained VQ-VAE whose codebook no longer matches the old LM.",
    )
    parser.add_argument(
        "--use-solve",
        action="store_true",
        help="Reconstruct coords via the absolute+relative geometry solve. "
        "Default (off) uses absolute coords only (knn_offsets head = pure "
        "training regulariser).",
    )
    parser.add_argument(
        "--vqvae-ckpt",
        type=str,
        default=(
            "pocket-ligand-vqvae/3dvcbp0h/checkpoints/"
            "vqvae-epoch=99-val/ligand_coord=0.1501.ckpt"
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "descriptor_cache_v4")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "diagnostics" / "diag_data.npz")
    parser.add_argument("--num-pockets", type=int, default=60)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = LMVocab()
    lig_lo = vocab.ligand_offset
    lig_cb = vocab.ligand_codebook_size

    recon_only = args.recon_only or args.lm_ckpt is None
    vqvae_ckpt = args.vqvae_ckpt if Path(args.vqvae_ckpt).is_absolute() else PROJECT_ROOT / args.vqvae_ckpt
    vqvae = VQVAEModule.load_from_checkpoint(str(vqvae_ckpt), map_location=device).eval().to(device)
    model = None
    if not recon_only:
        lm = LigandLMModule.load_from_checkpoint(args.lm_ckpt, config=LMTrainingConfig(), map_location=device).eval().to(device)
        model = lm.model

    dm = ComplexDescriptorDataModule(VQVAETrainingConfig(), CrossDockedConfig(data_dir=PROJECT_ROOT / "data"))
    dm.cache_dir = args.cache_dir
    dm.setup()
    norm_stats = {k: v.to(device) for k, v in dm.norm_stats.items()}
    lig_mean = dm.norm_stats["ligand_mean"].numpy()
    lig_std = dm.norm_stats["ligand_std"].numpy()

    hub = PROJECT_ROOT / "data" / "hub_cache"
    mdf = pq.read_table(hub / "repo" / "manifest.parquet").to_pandas()
    test_df = mdf[(mdf["source_type"] == "cdonly") & (mdf["cdonly_fold0"] == "test")].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(test_df))
    receptor_dir, repo_dir = hub / "receptors", hub / "repo"
    pocket_config = PocketExtractionConfig()
    protein_desc_calc = BackboneSphericalDescriptor()
    lig_desc = LigandDescriptor()

    gt: list[dict] = []
    recon: list[dict] = []
    gen: list[dict] = []
    recon_pa: list[float] = []   # per-atom RMSD (recon vs true)
    recon_kab: list[float] = []  # Kabsch RMSD (internal shape only)
    real_codes: list[int] = []
    gen_codes: list[int] = []
    done = 0
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
            res = _pocket_codes(rec_path, mol, pocket_config, protein_desc_calc, vqvae.protein_vqvae, norm_stats, device)
        except Exception as e:  # noqa: BLE001
            logger.warning("skip %s: %s", row["complex_dir"], e)
            continue
        if res is None:
            continue
        prot_codes, frame, gt_coords, gt_elems = res
        centroid = frame[0]
        gt_true = _build_rdkit_mol(mol["atoms"], mol["bonds"]) is not None
        gt.append(_metrics(gt_elems, gt_coords, centroid, ref_valid=gt_true))

        # --- VQ-VAE reconstruction arm (no LM) ---
        rc = _reconstruct(mol, frame, lig_desc, vqvae.ligand_vqvae, norm_stats, lig_mean, lig_std, device, use_solve=args.use_solve)
        if rc is not None:
            recon_coords, recon_elems, code_list = rc
            real_codes.extend(code_list)
            recon.append(_metrics(recon_elems, recon_coords, centroid))
            if len(recon_coords) == len(gt_coords) and len(gt_coords) >= 3:
                recon_pa.append(per_atom_rmsd(gt_coords, recon_coords))
                recon_kab.append(kabsch_rmsd(gt_coords, recon_coords))

        # --- LM generation arm (pocket-conditioned) ---
        if not recon_only:
            prompt = [BOS_ID, P_OPEN_ID, *(vocab.protein_offset + c for c in prot_codes), P_CLOSE_ID, L_OPEN_ID]
            prompt_ids = torch.tensor([prompt], device=device).repeat(args.num_samples, 1)
            with torch.no_grad():
                out = model.generate(
                    input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                    do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens, eos_token_id=L_CLOSE_ID, pad_token_id=PAD_ID,
                )
            for k in range(out.shape[0]):
                toks = out[k].tolist()[len(prompt):]
                lig_tok = toks[: toks.index(L_CLOSE_ID)] if L_CLOSE_ID in toks else toks
                codes = [t - lig_lo for t in lig_tok if lig_lo <= t < lig_lo + lig_cb]
                if len(codes) < 2:
                    continue
                gen_codes.extend(codes)
                coords, elems = _decode_ligand(codes, vqvae.ligand_vqvae, norm_stats, frame, device, use_solve=args.use_solve)
                gen.append(_metrics(elems, coords, centroid))
        done += 1
        if done % 10 == 0:
            logger.info("processed %d pockets (recon %d, gen %d)", done, len(recon), len(gen))

    methods = VALIDITY_METHODS
    real_hist = np.bincount(np.asarray(real_codes, dtype=np.int64), minlength=lig_cb) if real_codes else np.zeros(lig_cb, dtype=np.int64)
    gen_hist = np.bincount(np.asarray(gen_codes, dtype=np.int64), minlength=lig_cb) if gen_codes else np.zeros(lig_cb, dtype=np.int64)

    np.savez(
        args.out,
        num_pockets=done,
        lm_ckpt=str(args.lm_ckpt),
        methods=np.array(methods, dtype=object),
        ligand_codebook_size=lig_cb,
        recon_rmsd_peratom=np.array(recon_pa),
        recon_rmsd_kabsch=np.array(recon_kab),
        real_code_hist=real_hist,
        gen_code_hist=gen_hist,
        **{f"gt_{k}": v for k, v in _pack(gt, [*methods, "true_bonds"]).items()},
        **{f"recon_{k}": v for k, v in _pack(recon, methods).items()},
        **{f"gen_{k}": v for k, v in _pack(gen, methods).items()},
    )
    logger.info(
        "Saved %s | %d pockets | recon %d / gen %d mols | recon Kabsch %.2f A / per-atom %.2f A | code util real %.0f%% gen %.0f%%",
        args.out, done, len(recon), len(gen),
        float(np.mean(recon_kab)) if recon_kab else float("nan"),
        float(np.mean(recon_pa)) if recon_pa else float("nan"),
        100 * (real_hist > 0).mean(), 100 * (gen_hist > 0).mean(),
    )


if __name__ == "__main__":
    main()
