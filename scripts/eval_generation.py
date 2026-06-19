"""Generate ligands for many test pockets and dump 3D-quality metrics.

Produces a compact ``.npz`` that ``notebooks/generation_eval.py`` loads for
plotting. For each held-out-test pocket we sample several ligands from the LM,
decode them to 3D via the VQ-VAE, and compute per-molecule geometry stats for
both the generated molecules and the ground-truth ligand (reference).

Metrics per molecule: atom count, element symbols, inferred-bond lengths,
minimum interatomic distance (clash proxy), mean bonds-per-atom, connected-
component count, RDKit validity (DetermineBonds + sanitize), and ligand-to-
pocket-centroid distance (placement).

Run on a GPU node::

    uv run python scripts/eval_generation.py \
        --lm-ckpt <path/to/lm.ckpt> --num-pockets 60 --num-samples 3
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

from scripts.generate_ligands_3d import (  # noqa: E402
    _decode_ligand,
    _pocket_codes,
    _read_mol_from_tar,
)
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
from src.tokenizers.ligand import _build_rdkit_mol  # noqa: E402
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


def _n_components(n_atoms: int, bonds: list[tuple[int, int]]) -> int:
    """Connected-component count of the bond graph (isolated atoms count)."""
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in bonds:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n_atoms)})


# Validity is computed several ways so we can see how much the strict
# geometry-only RDKit check undercounts vs fairer definitions. Order = columns
# of the comparison table in notebooks/generation_eval.py.
VALIDITY_METHODS = [
    "connected",            # single connected component (distance bonds)
    "no_clash",             # no atom pair closer than 1.0 A
    "geom_ok",              # connected AND no_clash AND bond lengths sane
    "rdkit_charge0",        # DetermineBonds(charge=0) + sanitize  (strict)
    "rdkit_chargesearch",   # DetermineBonds over a few total charges
    "rdkit_relaxed",        # connectivity -> UFF relax -> DetermineBonds
]
_CHARGES = (0, 1, -1, 2, -2)
_CLASH = 1.0  # A; nearest-atom distance below this counts as a steric clash


def _safe(fn, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return None


def _xyz_block(elements: list[str], coords: np.ndarray) -> str:
    body = "\n".join(
        f"{e} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
        for e, c in zip(elements, coords, strict=True)
    )
    return f"{len(elements)}\n\n{body}"


def _determine_mol(elements: list[str], coords: np.ndarray, charge: int):  # noqa: ANN202
    """MolFromXYZ -> DetermineBonds(charge) -> sanitize; returns mol or raises."""
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import rdDetermineBonds  # noqa: PLC0415

    mol = Chem.MolFromXYZBlock(_xyz_block(elements, coords))
    if mol is None:
        msg = "MolFromXYZBlock failed"
        raise ValueError(msg)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    Chem.SanitizeMol(mol)
    return mol


def _charge_search_valid(elements: list[str], coords: np.ndarray) -> bool:
    return any(_safe(_determine_mol, elements, coords, q) is not None for q in _CHARGES)


def _relaxed_valid(elements: list[str], coords: np.ndarray) -> bool:
    """Perceive connectivity, UFF-minimize, then re-perceive bonds on relaxed xyz."""
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem, rdDetermineBonds  # noqa: PLC0415

    mol = Chem.MolFromXYZBlock(_xyz_block(elements, coords))
    if mol is None:
        return False
    rdDetermineBonds.DetermineConnectivity(mol)
    Chem.SanitizeMol(mol, catchErrors=True)
    _safe(AllChem.UFFOptimizeMolecule, mol, maxIters=200)
    relaxed = mol.GetConformer().GetPositions()
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    return _charge_search_valid(syms, relaxed)


def _validity_flags(
    elements: list[str], coords: np.ndarray, min_pair: float, n_comp: int, blens: list[float]
) -> dict[str, bool]:
    too_few = len(elements) < 2 or any(e == "X" for e in elements)
    connected = n_comp == 1 and len(elements) >= 1
    no_clash = (not np.isnan(min_pair)) and min_pair >= _CLASH
    lens_ok = bool(blens) and all(0.9 <= b <= 1.9 for b in blens)
    rdkit_c0 = (not too_few) and _safe(_determine_mol, elements, coords, 0) is not None
    return {
        "connected": connected,
        "no_clash": no_clash,
        "geom_ok": connected and no_clash and lens_ok,
        "rdkit_charge0": bool(rdkit_c0),
        "rdkit_chargesearch": (not too_few) and _charge_search_valid(elements, coords),
        "rdkit_relaxed": (not too_few) and bool(_safe(_relaxed_valid, elements, coords)),
    }


def _metrics(
    elements: list[str],
    coords: np.ndarray,
    pocket_centroid: np.ndarray,
    *,
    ref_valid: bool | None = None,
) -> dict:
    coords = np.asarray(coords, dtype=np.float64)
    n = len(elements)
    bonds = infer_bonds(elements, coords)
    blens = [float(np.linalg.norm(coords[a] - coords[b])) for a, b in bonds]
    if n >= 2:
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        d[np.diag_indices(n)] = np.inf
        min_pair = float(d.min())
    else:
        min_pair = float("nan")
    n_comp = _n_components(n, bonds)
    flags = _validity_flags(elements, coords, min_pair, n_comp, blens)
    # GT only: validity from the molecule's real SDF bonds (reference ceiling).
    if ref_valid is not None:
        flags["true_bonds"] = bool(ref_valid)
    return {
        "n_atoms": n,
        "elements": list(elements),
        "coords": coords.astype(np.float32),
        "bond_lengths": blens,
        "min_pair_dist": min_pair,
        "bonds_per_atom": (2 * len(bonds) / n) if n else 0.0,
        "n_components": n_comp,
        "centroid_dist": float(np.linalg.norm(coords.mean(axis=0) - pocket_centroid)),
        **{f"v_{k}": v for k, v in flags.items()},
    }


def main() -> None:  # noqa: C901, PLR0915
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
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "descriptor_cache_v4")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "gen_eval" / "eval_data.npz")
    parser.add_argument("--num-pockets", type=int, default=60)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--empty-pocket",
        action="store_true",
        help="Generate ligands UNconditionally (prompt <bos><p></p><l>), the "
        "native mode of the GEOM-pretrained ligand-only LM. Shape metrics are "
        "frame-invariant so they stay comparable; centroid_dist is not "
        "meaningful for these (the ligand is not placed in the pocket).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Human label stored in the npz (defaults to the checkpoint stem).",
    )
    args = parser.parse_args()
    label = args.label or Path(args.lm_ckpt).stem

    torch.manual_seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = LMVocab()
    lig_lo = vocab.ligand_offset

    vqvae_ckpt = args.vqvae_ckpt if Path(args.vqvae_ckpt).is_absolute() else PROJECT_ROOT / args.vqvae_ckpt
    vqvae = VQVAEModule.load_from_checkpoint(str(vqvae_ckpt), map_location=device).eval().to(device)
    lm = LigandLMModule.load_from_checkpoint(args.lm_ckpt, config=LMTrainingConfig(), map_location=device).eval().to(device)
    model = lm.model

    dm = ComplexDescriptorDataModule(VQVAETrainingConfig(), CrossDockedConfig(data_dir=PROJECT_ROOT / "data"))
    dm.cache_dir = args.cache_dir
    dm.setup()
    norm_stats = {k: v.to(device) for k, v in dm.norm_stats.items()}

    hub = PROJECT_ROOT / "data" / "hub_cache"
    mdf = pq.read_table(hub / "repo" / "manifest.parquet").to_pandas()
    test_df = mdf[(mdf["source_type"] == "cdonly") & (mdf["cdonly_fold0"] == "test")].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(test_df))
    receptor_dir, repo_dir = hub / "receptors", hub / "repo"
    pocket_config = PocketExtractionConfig()
    protein_desc_calc = BackboneSphericalDescriptor()

    gen: list[dict] = []
    gt: list[dict] = []
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

        if args.empty_pocket:
            prompt = [BOS_ID, P_OPEN_ID, P_CLOSE_ID, L_OPEN_ID]
        else:
            prompt = [BOS_ID, P_OPEN_ID, *(vocab.protein_offset + c for c in prot_codes), P_CLOSE_ID, L_OPEN_ID]
        prompt_ids = torch.tensor([prompt], device=device).repeat(args.num_samples, 1)
        with torch.no_grad():
            out = model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=L_CLOSE_ID,
                pad_token_id=PAD_ID,
            )
        for k in range(out.shape[0]):
            toks = out[k].tolist()[len(prompt):]
            terminated = L_CLOSE_ID in toks
            lig_tok = toks[: toks.index(L_CLOSE_ID)] if terminated else toks
            codes = [t - lig_lo for t in lig_tok if lig_lo <= t < lig_lo + vocab.ligand_codebook_size]
            if len(codes) < 2:
                continue
            coords, elems = _decode_ligand(codes, vqvae.ligand_vqvae, norm_stats, frame, device)
            m = _metrics(elems, coords, centroid)
            m["terminated"] = terminated
            gen.append(m)
        done += 1
        if done % 10 == 0:
            logger.info("processed %d pockets (%d gen mols)", done, len(gen))

    def _pack(items: list[dict], *, with_true: bool) -> dict:
        methods = [*VALIDITY_METHODS, *(["true_bonds"] if with_true else [])]
        out = {
            "n_atoms": np.array([m["n_atoms"] for m in items]),
            "min_pair_dist": np.array([m["min_pair_dist"] for m in items]),
            "bonds_per_atom": np.array([m["bonds_per_atom"] for m in items]),
            "n_components": np.array([m["n_components"] for m in items]),
            "centroid_dist": np.array([m["centroid_dist"] for m in items]),
            "bond_lengths": np.concatenate([np.array(m["bond_lengths"]) for m in items]) if items else np.array([]),
            "elements": np.array([e for m in items for e in m["elements"]], dtype=object),
            "coords_list": np.array([m["coords"] for m in items] + [None], dtype=object)[:-1],
            "elements_list": np.array([m["elements"] for m in items] + [None], dtype=object)[:-1],
        }
        for meth in methods:
            out[f"v_{meth}"] = np.array([bool(m.get(f"v_{meth}", False)) for m in items], dtype=bool)
        return out

    gp, tp = _pack(gen, with_true=False), _pack(gt, with_true=True)
    np.savez(
        args.out,
        num_pockets=done,
        num_gen=len(gen),
        num_gt=len(gt),
        lm_ckpt=str(args.lm_ckpt),
        label=label,
        empty_pocket=bool(args.empty_pocket),
        methods=np.array(VALIDITY_METHODS, dtype=object),
        **{f"gen_{k}": v for k, v in gp.items()},
        **{f"gt_{k}": v for k, v in tp.items()},
    )
    logger.info(
        "Saved %s | %d pockets, %d gen / %d gt mols | gen rdkit(c0) %.0f%%",
        args.out,
        done,
        len(gen),
        len(gt),
        100 * gp["v_rdkit_charge0"].mean() if len(gen) > 0 else 0.0,
    )


if __name__ == "__main__":
    main()
