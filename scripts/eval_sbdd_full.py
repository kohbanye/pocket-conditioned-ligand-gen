"""Multi-faceted SBDD evaluation: what is wrong with the generated ligands?

For each held-out CrossDocked test pocket, generate ligands from one or more LM
checkpoints and score them along three axes, side by side with the reference
(native) ligand as the ceiling:

A. **Is it a valid molecule?** RDKit parse, connectivity, drug-likeness
   (QED, MW, ring count, rotatable bonds).
B. **Is the 3D geometry clean?** PoseBusters per-check (bond lengths, bond
   ANGLES, aromatic-ring flatness, double-bond flatness, internal steric clash)
   -- the parts we had not been measuring.
C. **Does it bind?** AutoDock Vina ``score_only`` (as generated) and
   ``local_only`` (after local minimisation). The score_only -> local_only gap
   isolates how much bad *geometry* costs vs the molecule/placement itself.

Writes one row per (pocket, model, sample) to a parquet for the comparison
notebook. Run with posebusters available::

    uv run --with posebusters python scripts/eval_sbdd_full.py \
        --lm-ckpts scratch:pocket-ligand-lm/g79let5b/checkpoints/lm-e09-vl1.4088.ckpt \
                   finetune:pocket-ligand-lm/cjp7e60q/checkpoints/lm-e02-vl1.4965.ckpt \
        --num-pockets 100 --num-samples 3 --out outputs/sbdd_eval/results.parquet
"""
# ruff: noqa: PLR2004, E501, PLR0915, PLR0912, C901

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dock_vina import (  # noqa: E402
    DEFAULT_OBABEL,
    DEFAULT_VINA,
    _heavy_rmsd,
    _parse_score,
    _read_pdbqt_heavy,
    _run,
    _write_xyz,
)
from scripts.eval_posebusters import _reconstruct  # noqa: E402
from scripts.generate_ligands_3d import (  # noqa: E402
    _decode_ligand,
    _decode_ligand_atom,
    _pocket_codes,
    _pocket_codes_atom,
    _read_mol_from_tar,
    load_atom_lm,
    load_atom_norm_stats,
    load_atom_vqvae,
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
from src.tokenizers.atom import ProteinAtomDescriptor  # noqa: E402
from src.tokenizers.descriptor_schema import SOURCE_LIGAND_IDX  # noqa: E402
from src.tokenizers.lm_vocab import (  # noqa: E402
    BOS_ID,
    L_CLOSE_ID,
    L_OPEN_ID,
    P_CLOSE_ID,
    P_OPEN_ID,
    PAD_ID,
    AtomLMVocab,
    LMVocab,
)
from src.tokenizers.protein import BackboneSphericalDescriptor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PREPARE_RECEPTOR = "/home/5/uq02055/usr/app/ADFRsuite/bin/prepare_receptor"
PB_CHECKS = [
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "double_bond_flatness",
    "all_atoms_connected",
]


def _prepare_receptor(rec_pdb: Path, out_pdbqt: Path, tool: str, obabel: str) -> bool:
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    r = _run([tool, "-r", str(rec_pdb), "-o", str(out_pdbqt),
              "-A", "checkhydrogens", "-U", "nphs_lps_waters_nonstdres"])
    if out_pdbqt.exists():
        return True
    # Fallback to obabel.
    _run([obabel, str(rec_pdb), "-O", str(out_pdbqt), "-xr"])
    logger.warning("prepare_receptor failed for %s (%s); used obabel", rec_pdb.name, r.stderr.strip()[:80])
    return out_pdbqt.exists()


def _dock(rec: dict, vina: str, obabel: str, box_size: float, tmp_dir: str) -> dict:
    """Vina score_only + local_only for one ligand against its receptor pdbqt."""
    elements, coords = rec["elements"], rec["coords"]
    out = {"score_as_is": None, "score_opt": None, "opt_rmsd": None, "n_atoms_docked": None}
    n = len(elements)
    if n < 2:
        return out
    xs, ys, zs = (np.asarray([c[i] for c in coords]) for i in range(3))
    cx, cy, cz = float(xs.mean()), float(ys.mean()), float(zs.mean())
    box = [
        "--center_x", f"{cx:.3f}", "--center_y", f"{cy:.3f}", "--center_z", f"{cz:.3f}",
        "--size_x", f"{max(box_size, float(np.ptp(xs)) + 8):.3f}",
        "--size_y", f"{max(box_size, float(np.ptp(ys)) + 8):.3f}",
        "--size_z", f"{max(box_size, float(np.ptp(zs)) + 8):.3f}",
    ]
    common = ["--receptor", rec["receptor"], "--cpu", "1", "--seed", "1", *box]
    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        tdp = Path(td)
        xyz, pdbqt = tdp / "lig.xyz", tdp / "lig.pdbqt"
        _write_xyz(xyz, elements, [list(c) for c in coords])
        _run([obabel, str(xyz), "-O", str(pdbqt), "-r", "-p", "7.4", "--partialcharge", "gasteiger"])
        if not pdbqt.exists() or not any(
            ln.startswith(("ATOM", "HETATM")) for ln in pdbqt.read_text().splitlines()
        ):
            return out
        out["n_atoms_docked"] = len(_read_pdbqt_heavy(pdbqt))
        try:
            out["score_as_is"] = _parse_score(_run([vina, "--ligand", str(pdbqt), "--score_only", *common]).stdout)
            optp = tdp / "opt.pdbqt"
            out["score_opt"] = _parse_score(_run([vina, "--ligand", str(pdbqt), "--local_only", "--out", str(optp), *common]).stdout)
            if optp.exists():
                out["opt_rmsd"] = _heavy_rmsd(_read_pdbqt_heavy(pdbqt), _read_pdbqt_heavy(optp))
        except Exception as e:  # noqa: BLE001
            logger.debug("dock failed: %s", e)
    return out


def _chem_and_geom(coords: np.ndarray, bonds: list[tuple[int, int]]) -> dict:
    n = len(coords)
    blens = [float(np.linalg.norm(coords[a] - coords[b])) for a, b in bonds]
    if n >= 2:
        d = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
        d[np.diag_indices(n)] = np.inf
        minp = float(d.min())
    else:
        minp = float("nan")
    # connected components over the bond graph
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in bonds:
        parent[find(a)] = find(b)
    ncomp = len({find(i) for i in range(n)})
    return {"n_atoms": n, "min_pair_dist": minp, "n_components": ncomp,
            "bond_len_mean": float(np.mean(blens)) if blens else float("nan")}


def _rdkit_props(mol) -> dict:  # noqa: ANN001
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import QED, Descriptors  # noqa: PLC0415
    out = {"rdkit_valid": False, "qed": None, "mw": None, "n_rings": None, "n_rot": None}
    if mol is None:
        return out
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m)
        out["rdkit_valid"] = True
        out["qed"] = float(QED.qed(m))
        out["mw"] = float(Descriptors.MolWt(m))
        out["n_rings"] = int(Descriptors.RingCount(m))
        out["n_rot"] = int(Descriptors.NumRotatableBonds(m))
    except Exception as e:  # noqa: BLE001
        logger.debug("rdkit props failed: %s", e)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm-ckpts", type=str, nargs="+", required=True, help="name:path pairs")
    parser.add_argument("--vqvae-ckpt", type=str, default="pocket-ligand-vqvae/3dvcbp0h/checkpoints/vqvae-epoch=99-val/ligand_coord=0.1501.ckpt")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "descriptor_cache_v4")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "sbdd_eval" / "results.parquet")
    parser.add_argument("--num-pockets", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--all-atom",
        action="store_true",
        help="Use the unified all-atom tokenizer (AtomLMVocab + AtomVQVAEModule); "
        "every --lm-ckpts entry is then an all-atom LM sharing one atom VQ-VAE.",
    )
    parser.add_argument(
        "--split-codebook",
        action="store_true",
        help="All-atom VQ with SPLIT codebooks (protein + ligand) -> 2-range "
        "LMVocab. Implies all-atom.",
    )
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--ligand-codebook-size", type=int, default=4096)
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=PROJECT_ROOT / "data" / "descriptor_cache_allatom"
        / "normalization_stats.pt",
    )
    parser.add_argument("--dock-workers", type=int, default=32)
    parser.add_argument("--vina", type=str, default=DEFAULT_VINA)
    parser.add_argument("--obabel", type=str, default=DEFAULT_OBABEL)
    parser.add_argument("--prepare-receptor", type=str, default=DEFAULT_PREPARE_RECEPTOR)
    parser.add_argument("--box-size", type=float, default=22.5)
    parser.add_argument("--tmp-dir", type=str, default=str(Path.home() / "tmpdir"))
    args = parser.parse_args()

    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rec_cache_dir = args.out.parent / "receptors_pdbqt"
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pc = PocketExtractionConfig()
    receptor_cache: dict[str, tuple] = {}

    vqvae_ckpt = args.vqvae_ckpt if Path(args.vqvae_ckpt).is_absolute() else PROJECT_ROOT / args.vqvae_ckpt
    if args.all_atom or args.split_codebook:
        split = args.split_codebook
        atom_vqvae = load_atom_vqvae(vqvae_ckpt, args.codebook_size, device, split=split, ligand_codebook_size=args.ligand_codebook_size)
        norm = load_atom_norm_stats(args.norm_stats, device)
        prot_atom_desc = ProteinAtomDescriptor()
        if split:
            vocab = LMVocab(protein_codebook_size=args.codebook_size, ligand_codebook_size=args.ligand_codebook_size)
            code_lo, code_hi = vocab.ligand_offset, vocab.ligand_offset + vocab.ligand_codebook_size
            dec_source = SOURCE_LIGAND_IDX
        else:
            vocab = AtomLMVocab(codebook_size=args.codebook_size)
            code_lo, code_hi = vocab.offset, vocab.offset + vocab.codebook_size
            dec_source = None
        models = {}
        for spec in args.lm_ckpts:
            name, path = spec.split(":", 1)
            models[name] = load_atom_lm(path, args.codebook_size, device, split=split, ligand_codebook_size=args.ligand_codebook_size)

        def encode_pocket(rec_pdb: Path, mol: dict):  # noqa: ANN202
            return _pocket_codes_atom(
                rec_pdb, mol, pc, prot_atom_desc, atom_vqvae, norm, device,
                receptor_cache=receptor_cache,
            )

        def decode_codes(codes, frame):  # noqa: ANN001, ANN202
            return _decode_ligand_atom(codes, atom_vqvae, norm, frame, device, source_idx=dec_source)

        def build_prompt(prot_codes: list[int]) -> list[int]:
            return vocab.build_sequence(prot_codes, [])[:-2]  # drop </l><eos>
    else:
        vqvae = VQVAEModule.load_from_checkpoint(str(vqvae_ckpt), map_location=device).eval().to(device)
        vocab = LMVocab()
        code_lo, code_hi = vocab.ligand_offset, vocab.ligand_offset + vocab.ligand_codebook_size
        pdc = BackboneSphericalDescriptor()
        models = {}
        for spec in args.lm_ckpts:
            name, path = spec.split(":", 1)
            models[name] = LigandLMModule.load_from_checkpoint(path, config=LMTrainingConfig(), map_location=device).eval().to(device).model
        dm = ComplexDescriptorDataModule(VQVAETrainingConfig(), CrossDockedConfig(data_dir=PROJECT_ROOT / "data"))
        dm.cache_dir = args.cache_dir
        dm.setup()
        norm = {k: v.to(device) for k, v in dm.norm_stats.items()}

        def encode_pocket(rec_pdb: Path, mol: dict):  # noqa: ANN202
            return _pocket_codes(rec_pdb, mol, pc, pdc, vqvae.protein_vqvae, norm, device)

        def decode_codes(codes, frame):  # noqa: ANN001, ANN202
            return _decode_ligand(codes, vqvae.ligand_vqvae, norm, frame, device)

        def build_prompt(prot_codes: list[int]) -> list[int]:
            return [BOS_ID, P_OPEN_ID, *(vocab.protein_offset + c for c in prot_codes), P_CLOSE_ID, L_OPEN_ID]

    hub = PROJECT_ROOT / "data" / "hub_cache"
    mdf = pq.read_table(hub / "repo" / "manifest.parquet").to_pandas()
    tdf = mdf[(mdf["source_type"] == "cdonly") & (mdf["cdonly_fold0"] == "test")].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(tdf))
    rec_pdbqt_cache: dict[str, str] = {}

    # ---- Phase 1: generation (GPU) ----------------------------------------
    records: list[dict] = []  # one per ligand (GT + each model sample)
    done = 0
    for oi in order:
        if done >= args.num_pockets:
            break
        row = tdf.iloc[int(oi)]
        rec_pdb = hub / "receptors" / f"{row['complex_dir']}/{row['receptor_pdb']}"
        if not rec_pdb.exists():
            continue
        mol = _read_mol_from_tar(hub / "repo", int(row["shard_idx"]), int(row["pair_idx"]))
        if mol is None:
            continue
        try:
            res = encode_pocket(rec_pdb, mol)
        except Exception as e:  # noqa: BLE001
            logger.warning("skip %s: %s", row["complex_dir"], e)
            continue
        if res is None:
            continue
        prot_codes, frame, gt_coords, gt_elems = res
        # receptor pdbqt (cached per receptor path)
        key = str(rec_pdb)
        if key not in rec_pdbqt_cache:
            outp = rec_cache_dir / f"{done:04d}.pdbqt"
            if _prepare_receptor(rec_pdb, outp, args.prepare_receptor, args.obabel):
                rec_pdbqt_cache[key] = str(outp)
            else:
                rec_pdbqt_cache[key] = ""
        recp = rec_pdbqt_cache[key]
        base = {"pocket": done, "complex_dir": row["complex_dir"], "receptor": recp}
        records.append({**base, "model": "GT", "sample": 0, "coords": gt_coords, "elements": gt_elems})
        prompt = build_prompt(prot_codes)
        for name, model in models.items():
            pids = torch.tensor([prompt], device=device).repeat(args.num_samples, 1)
            with torch.no_grad():
                gen = model.generate(input_ids=pids, attention_mask=torch.ones_like(pids), do_sample=True,
                                     temperature=args.temperature, top_p=args.top_p,
                                     max_new_tokens=args.max_new_tokens, eos_token_id=L_CLOSE_ID, pad_token_id=PAD_ID)
            for k in range(gen.shape[0]):
                toks = gen[k].tolist()[len(prompt):]
                lig = toks[: toks.index(L_CLOSE_ID)] if L_CLOSE_ID in toks else toks
                codes = [t - code_lo for t in lig if code_lo <= t < code_hi]
                if len(codes) < 2:
                    continue
                coords, elems = decode_codes(codes, frame)
                records.append({**base, "model": name, "sample": k, "coords": coords, "elements": elems})
        done += 1
        if done % 10 == 0:
            logger.info("generated %d pockets (%d ligands)", done, len(records))

    logger.info("Phase 1 done: %d pockets, %d ligands. Receptors prepared: %d", done, len(records), len(rec_pdbqt_cache))

    # free GPU before threaded docking
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Phase 2: docking (threaded subprocess) ---------------------------
    def dock_rec(rec: dict) -> dict:
        if not rec["receptor"]:
            return {"score_as_is": None, "score_opt": None, "opt_rmsd": None, "n_atoms_docked": None}
        return _dock(rec, args.vina, args.obabel, args.box_size, args.tmp_dir)

    with ThreadPoolExecutor(max_workers=args.dock_workers) as ex:
        dock_results = list(ex.map(dock_rec, records))
    logger.info("Phase 2 (docking) done")

    # ---- Phase 3: chemistry + geometry + PoseBusters ----------------------
    coords_list = [r["coords"] for r in records]
    elements_list = [r["elements"] for r in records]
    mols = _reconstruct(coords_list, elements_list)  # {idx: rdkit mol}

    from posebusters import PoseBusters  # noqa: PLC0415

    # Drop energy_ratio (slow conformer embedding) and check_radicals (open
    # valences from heavy-atom OpenBabel reconstruction flag ~every molecule,
    # incl. real GT, as radicals -- a reconstruction artefact). Mirrors
    # eval_posebusters.pb_valid so GT scores realistically (~85% PB-valid).
    _base = PoseBusters(config="mol")
    _cfg = _base.config
    _cfg["modules"] = [m for m in _cfg["modules"] if m.get("function") not in {"energy_ratio", "check_radicals"}]
    _cfg["max_workers"] = 4
    pb = PoseBusters(config=_cfg)
    idxs = sorted(mols)
    pb_rows: dict[int, dict] = {}
    if idxs:
        pb_df = pb.bust([mols[i] for i in idxs])
        cols = [c for c in PB_CHECKS if c in pb_df.columns]
        valid = pb_df.all(axis=1).to_numpy()  # PB-valid = all remaining mol checks pass
        for j, i in enumerate(idxs):
            d = {f"pb_{c}": bool(pb_df.iloc[j][c]) for c in cols}
            d["pb_valid"] = bool(valid[j]) if j < len(valid) else False
            pb_rows[i] = d

    out_rows = []
    for i, rec in enumerate(records):
        mol = mols.get(i)
        bonds = []
        if mol is not None:
            bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        cg = _chem_and_geom(np.asarray(rec["coords"], float), bonds)
        out_rows.append({
            "pocket": rec["pocket"], "complex_dir": rec["complex_dir"],
            "model": rec["model"], "sample": rec["sample"],
            **cg, **_rdkit_props(mol), **dock_results[i], **pb_rows.get(i, {}),
        })

    df = pd.DataFrame(out_rows)
    df.to_parquet(args.out)
    logger.info("Saved %d rows to %s", len(df), args.out)
    # quick headline per model
    for name in df["model"].unique():
        sub = df[df["model"] == name]
        pb = sub["pb_valid"] if "pb_valid" in sub else pd.Series([False] * len(sub))
        logger.info("  %-10s n=%d | rdkit_valid %.0f%% | pb_valid %.0f%% | score_opt med %.2f",
                    name, len(sub), 100 * sub["rdkit_valid"].mean(),
                    100 * pb.mean(), sub["score_opt"].median(skipna=True))


if __name__ == "__main__":
    main()
