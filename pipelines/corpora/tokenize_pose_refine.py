"""Build the pose-refiner training set: (corrupted pose, native pose, pocket).

The refiner learns to undo the *exact* geometry error a given generation decoder
emits. So for each CASF/sbdd-excluded BioLIP2 native complex we manufacture the
deployment corruption directly: run the crystal ligand through the ligand VQ-VAE
(encode -> decode -> spherical coord head -> Cartesian), which reproduces the
pairwise-blind, clash-prone reconstruction the LM decode path produces. Its atoms
are in 1:1 correspondence with the crystal pose, giving a clean supervised
(x0 -> x1) pair. Graded corruption levels (small rigid + isotropic jitter on top
of the VQ round-trip) widen the source support toward the more-corrupted LM
regime.

The corruption is defined by the deployed all-atom VQ-VAE (``AtomVQVAEModule`` +
``LigandAtomDescriptor``), so the refiner learns to repair exactly the error the
generation path emits.

The pocket half (coordinates + per-atom chemistry) is decoder-independent and
always comes from the all-atom receptor parse (``ProteinAtomDescriptor``).
Everything is stored in the pocket canonical frame; output is the concatenated
memmaps documented in :mod:`prolit.data.pose_refine_dataset` (pocket stored ONCE per
complex, referenced by pointer -> inode-safe).

Run (single GPU; use the venv python directly -- ``uv run`` rebuilds the editable
package, which is very slow here)::

    PYTHONPATH=$PWD .venv/bin/python pipelines/corpora/tokenize_pose_refine.py \
        --ckpt "pocket-ligand-vqvae/3dvcbp0h/checkpoints/vqvae-epoch=99-val/\
ligand_coord=0.1501.ckpt" \
        --cache-dir data/descriptor_cache_v4 \
        --n-complexes 12000 --n-corrupt 4 --out-dir data/pose_refine_legacy
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from pipelines.corpora.tokenize_biolip import (
    _bucket_code,
    _cd_test_pdbs,
    _load_ccd_smiles,
    _parse_biolip_txt,
    _read_needed,
)
from pipelines.corpora.tokenize_decoys import _perturb
from prolit.model.pose_refiner import (
    FEATURE_FIELDS,
    LIG_CHEM_HEADS,
    ligand_feats_from_heads,
    pocket_feats_from_descriptor,
)
from prolit.tokenizers.atom import (
    LigandAtomDescriptor,
    ProteinAtomDescriptor,
    precompute_receptor_atom_features_from_text,
)
from prolit.tokenizers.descriptor_schema import (
    ATOM_LAYOUT,
    fields_by_name,
)
from prolit.tokenizers.geometry import spherical_to_cartesian_np
from prolit.tokenizers.ligand import parse_ligand_pdb_text
from prolit.tokenizers.protein import (
    compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    precompute_pocket_atom_candidates_from_text,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# sbdd-bench evaluation targets (kept out of training, like CASF).
SBDD_PDBS = frozenset({"1iep", "2ity", "3pbl"})


# ---------------------------------------------------------------------------
# Ligand codecs: descriptor + VQ round-trip, one per deployment decoder.
# ---------------------------------------------------------------------------
class _LigandCodec:
    """Shared VQ round-trip: normalize -> encode -> decode -> canonical coords."""

    vq: object
    desc_fn: object
    mean: np.ndarray
    std: np.ndarray
    cmean: np.ndarray
    cstd: np.ndarray
    device: torch.device
    codebook_size: int = 0

    def descriptor(self, atoms: list, bonds: list, frame: tuple) -> np.ndarray:
        return self.desc_fn.compute(atoms, bonds, pocket_frame=frame)[0]

    @torch.no_grad()
    def roundtrip(
        self,
        desc: np.ndarray,
        resample_frac: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """VQ round-trip. ``resample_frac`` > 0 replaces that fraction of the
        ligand codes with random codebook entries BEFORE decoding, producing
        the intramolecularly-inconsistent geometry (bad bonds/angles/clashes)
        the LM decode emits -- the corruption the refiner must learn to repair.
        The chem features returned are always the base (frac=0) decode's, so a
        resampled x0 keeps the molecule's real features + native target."""
        x = torch.from_numpy((desc - self.mean) / self.std).float().to(self.device)
        codes = self.vq.encode(x)
        if resample_frac > 0 and rng is not None and self.codebook_size > 0:
            n = int(codes.shape[0])
            k = round(resample_frac * n)
            if k > 0:
                pos = rng.choice(n, k, replace=False)
                repl = rng.integers(0, self.codebook_size, k)
                codes = codes.clone()
                codes[torch.as_tensor(pos, device=codes.device)] = torch.as_tensor(
                    repl, dtype=codes.dtype, device=codes.device
                )
        out = self.vq.decode_to_outputs(codes)
        coord = out["coord"].cpu().numpy() * self.cstd + self.cmean
        x0 = spherical_to_cartesian_np(coord).astype(np.float32)
        chem = {h: out[h].argmax(dim=-1).cpu().numpy() for h in LIG_CHEM_HEADS}
        return x0, chem


class _AtomCodec(_LigandCodec):
    """The deployed all-atom decoder."""

    def __init__(
        self, ckpt: Path, norm_stats: Path, codebook_size: int, device: torch.device
    ) -> None:
        from prolit.config import AtomVQVAETrainingConfig  # noqa: PLC0415
        from prolit.model.vqvae_module import AtomVQVAEModule  # noqa: PLC0415

        cfg = AtomVQVAETrainingConfig()
        cfg.atom.codebook_size = codebook_size
        module = (
            AtomVQVAEModule.load_from_checkpoint(ckpt, config=cfg, map_location=device)
            .eval()
            .to(device)
        )
        norm = torch.load(norm_stats, weights_only=False)
        module.vqvae.set_normalization(norm["atom_mean"], norm["atom_std"])
        self.vq = module.vqvae
        self.device = device
        self.mean, self.std = norm["atom_mean"].numpy(), norm["atom_std"].numpy()
        f = fields_by_name(ATOM_LAYOUT)["coord"]
        self.cmean, self.cstd = self.mean[f.start : f.end], self.std[f.start : f.end]
        self.desc_fn = LigandAtomDescriptor()
        self.codebook_size = self.vq.codebook.num_codes


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------
def _augment(
    x0: np.ndarray, scale: float, rng: np.random.Generator, sigma_max: float
) -> np.ndarray:
    """Graded corruption on top of the VQ round-trip (capped rigid + jitter)."""
    if scale <= 0:
        return x0
    # small rigid perturbation (0.25 factor caps it at ~22 deg / 1.5 A so the
    # binding mode is preserved) + isotropic jitter (more VQ-like local error).
    x = _perturb(x0, rng, scale * 0.25)[0]
    return x + rng.normal(0.0, scale * sigma_max, size=x.shape)


class _PoseRefineWriter:
    """Streams the per-complex + per-record memmaps for the refiner dataset."""

    _STREAMS = (
        "lig_x1",
        "lig_feat",
        "lig_bonds",
        "lig_bond_ref",
        "pkt_x",
        "pkt_feat",
        "lig_x0",
        "records",
        "record_scale",
        "complexes",
    )

    def __init__(self, out_dir: Path, split: str) -> None:
        self._f = {
            name: (out_dir / f"{split}.{name}").open("wb") for name in self._STREAMS
        }
        self.n_complexes = 0
        self.n_records = 0

    def add_complex(  # noqa: PLR0913
        self,
        x1: np.ndarray,
        lig_feat: np.ndarray,
        bonds: np.ndarray,
        bond_ref: np.ndarray,
        pkt_x: np.ndarray,
        pkt_feat: np.ndarray,
    ) -> int:
        self._f["lig_x1"].write(x1.astype(np.float32).tobytes())
        self._f["lig_feat"].write(lig_feat.astype(np.int16).tobytes())
        self._f["lig_bonds"].write(bonds.astype(np.int32).tobytes())
        self._f["lig_bond_ref"].write(bond_ref.astype(np.float32).tobytes())
        self._f["pkt_x"].write(pkt_x.astype(np.float32).tobytes())
        self._f["pkt_feat"].write(pkt_feat.astype(np.int16).tobytes())
        self._f["complexes"].write(
            np.asarray(
                [x1.shape[0], pkt_x.shape[0], bonds.shape[0]], dtype=np.int64
            ).tobytes()
        )
        cid = self.n_complexes
        self.n_complexes += 1
        return cid

    def add_record(self, cid: int, x0: np.ndarray, scale: float) -> None:
        self._f["lig_x0"].write(x0.astype(np.float32).tobytes())
        self._f["records"].write(np.asarray([cid], dtype=np.int64).tobytes())
        self._f["record_scale"].write(np.asarray([scale], dtype=np.float32).tobytes())
        self.n_records += 1
        if self.n_records % 512 == 0:
            for fh in self._f.values():
                fh.flush()

    def close(self) -> None:
        for fh in self._f.values():
            fh.close()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="atom VQ-VAE ckpt.")
    parser.add_argument(
        "--norm-stats",
        type=Path,
        required=True,
        help="normalization_stats.pt that accompanies --ckpt.",
    )
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--biolip-dir", type=Path, default=Path("data/biolip"))
    parser.add_argument(
        "--cd-manifest", type=Path, default=Path("data/hub_cache/repo/manifest.parquet")
    )
    parser.add_argument(
        "--casf-pdbs", type=Path, default=Path("data/casf2016_pdbs.txt")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/pose_refine"))
    parser.add_argument("--max-residues", type=int, default=50)
    parser.add_argument("--n-complexes", type=int, default=12000)
    parser.add_argument(
        "--n-corrupt",
        type=int,
        default=4,
        help="Corruption records per complex (incl. base VQ round-trip).",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=0.7,
        help="Max jitter std (A) at corruption scale 1.",
    )
    parser.add_argument(
        "--resample-frac",
        type=float,
        default=0.0,
        help="Max fraction of ligand VQ codes resampled before decode (LM-like "
        "intramolecular corruption). 0 = jitter-only (legacy behaviour).",
    )
    parser.add_argument("--min-heavy", type=int, default=6)
    parser.add_argument("--max-heavy", type=int, default=50)
    parser.add_argument("--val-frac", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from prolit.config import PocketExtractionConfig  # noqa: PLC0415

    codec = _AtomCodec(args.ckpt, args.norm_stats, args.codebook_size, device)

    prot_desc_fn = ProteinAtomDescriptor()
    pocket_cfg = PocketExtractionConfig(max_residues=args.max_residues)

    sites = _parse_biolip_txt(args.biolip_dir / "BioLiP.txt.gz")
    ccd_smiles = _load_ccd_smiles(args.biolip_dir / "ligand.tsv.gz")
    excluded = _cd_test_pdbs(args.cd_manifest) | set(SBDD_PDBS)
    if args.casf_pdbs.exists():
        excluded |= {p.lower() for p in args.casf_pdbs.read_text().split() if p.strip()}
    rng = np.random.default_rng(args.seed)
    seen: set = set()
    uniq = []
    for s in sites:
        if s[0] in excluded:
            continue
        key = (s[0], s[2])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    rng.shuffle(uniq)
    uniq = uniq[: args.n_complexes]
    logger.info(
        "refiner source (all-atom decoder): %d native complexes (x%d records)",
        len(uniq),
        args.n_corrupt,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "train": _PoseRefineWriter(args.out_dir, "train"),
        "val": _PoseRefineWriter(args.out_dir, "val"),
    }
    val_pdbs = {s[0] for s in uniq[: int(len(uniq) * args.val_frac)]}

    by_bucket: dict[str, list[tuple]] = {}
    for s in uniq:
        by_bucket.setdefault(_bucket_code(s[0]), []).append(s)

    from tqdm import tqdm  # noqa: PLC0415

    import pipelines.corpora.tokenize_biolip as tb  # noqa: PLC0415

    n_ok = 0
    for code in tqdm(sorted(by_bucket), desc="buckets"):
        site_list = by_bucket[code]
        needed_rec = {f"{p}{rc}.pdb" for p, rc, _c, _l, _s in site_list}
        needed_lig = {f"{p}_{cc}_{lc}_{s}.pdb" for p, _rc, cc, lc, s in site_list}
        tb._w_biolip_dir = args.biolip_dir  # noqa: SLF001
        receptors = _read_needed("receptor", code, needed_rec)
        ligands = _read_needed("ligand", code, needed_lig)
        for pdb, rchain, ccd, ligchain, serial in site_list:
            rec = receptors.get(f"{pdb}{rchain}.pdb")
            lig = ligands.get(f"{pdb}_{ccd}_{ligchain}_{serial}.pdb")
            if rec is None or lig is None:
                continue
            try:
                mol = parse_ligand_pdb_text(
                    lig.decode("utf-8", "replace"), ccd_smiles.get(ccd)
                )
                if mol is None:
                    continue
                heavy_idx = [i for i, a in enumerate(mol["atoms"]) if a[0] != "H"]
                if not (args.min_heavy <= len(heavy_idx) <= args.max_heavy):
                    continue
                heavy = np.array(
                    [
                        (mol["atoms"][i][1], mol["atoms"][i][2], mol["atoms"][i][3])
                        for i in heavy_idx
                    ],
                    dtype=np.float64,
                )

                rec_text = rec.decode("utf-8", "replace")
                precomp = precompute_pocket_atom_candidates_from_text(rec_text)
                pocket = extract_pocket_atoms_from_candidates(
                    precomp, heavy.astype(np.float32), pocket_cfg
                )
                if pocket is None or pocket.atom_coords.shape[0] == 0:
                    continue
                feats = precompute_receptor_atom_features_from_text(rec_text)
                centroid, rotation = compute_canonical_frame(
                    pocket.ca_coords.astype(np.float64)
                )
                frame = (centroid, rotation)
                prot_desc, _ = prot_desc_fn.compute(pocket, feats, frame)
                if prot_desc.shape[0] != pocket.atom_coords.shape[0]:
                    continue
                pkt_x = (
                    (pocket.atom_coords.astype(np.float64) - centroid) @ rotation.T
                ).astype(np.float32)
                pkt_feat = pocket_feats_from_descriptor(prot_desc)

                desc = codec.descriptor(mol["atoms"], mol["bonds"], frame)
                if desc.shape[0] != len(heavy_idx):
                    continue
                x0_base, chem = codec.roundtrip(desc)
                n = x0_base.shape[0]
                if n != len(heavy_idx):
                    continue
                x1 = ((heavy - centroid) @ rotation.T).astype(np.float32)
                lig_feat = ligand_feats_from_heads(chem, n)

                orig_to_heavy = {orig: h for h, orig in enumerate(heavy_idx)}
                hb = [
                    (orig_to_heavy[a], orig_to_heavy[b])
                    for a, b, _t in mol["bonds"]
                    if a in orig_to_heavy and b in orig_to_heavy and a != b
                ]
                bonds = np.asarray(hb, dtype=np.int32).reshape(-1, 2)
                bond_ref = (
                    np.linalg.norm(x1[bonds[:, 0]] - x1[bonds[:, 1]], axis=1).astype(
                        np.float32
                    )
                    if bonds.shape[0]
                    else np.zeros(0, dtype=np.float32)
                )

                split = "val" if pdb in val_pdbs else "train"
                w = writers[split]
                cid = w.add_complex(x1, lig_feat, bonds, bond_ref, pkt_x, pkt_feat)
                w.add_record(cid, x0_base, 0.0)
                for k in range(1, args.n_corrupt):
                    scale = k / args.n_corrupt
                    # codebook-resampled decode (LM-like intramolecular errors)
                    # + a light rigid/jitter on top for placement diversity.
                    if args.resample_frac > 0:
                        x0_k = codec.roundtrip(desc, scale * args.resample_frac, rng)[0]
                    else:
                        x0_k = x0_base
                    x0_k = _augment(x0_k, scale * 0.5, rng, args.sigma_max)
                    w.add_record(cid, x0_k.astype(np.float32), scale)
                n_ok += 1
            except Exception:
                logger.exception("failed %s_%s", pdb, ccd)
                continue

    meta = {
        "source": "biolip2_vq_bridge",
        "decoder": "atom",
        "n_corrupt": args.n_corrupt,
        "sigma_max": args.sigma_max,
        "feature_fields": [name for name, _ in FEATURE_FIELDS],
        "complexes_used": n_ok,
        "splits": {},
    }
    for split, w in writers.items():
        meta["splits"][split] = {
            "num_complexes": w.n_complexes,
            "num_records": w.n_records,
        }
        logger.info("%s: %d complexes, %d records", split, w.n_complexes, w.n_records)
        w.close()
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("wrote pose-refine set to %s", args.out_dir)


if __name__ == "__main__":
    main()
