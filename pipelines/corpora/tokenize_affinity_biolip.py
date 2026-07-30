"""Binding-affinity corpus: BioLIP native complexes labelled with pK.

The RMSD head answers "is this pose native-like"; it has no affinity signal by
construction (CASF scoring power R = -0.04). This builds the corpus for a second
head that answers "how tightly does it bind": each BioLIP complex that carries an
experimental Kd/Ki/IC50 is tokenized ONCE (crystal pose only -- no decoys needed)
and labelled with pK = -log10(molar).

Affinity is taken from BioLIP's curated columns, preferring the better-curated
source: PDBbind-CN > Binding MOAD > literature > BindingDB. The CASF-2016 core is
held out (it is the benchmark) as are CrossDocked fold0-test PDBs.

Output reuses the RescoreDataset layout (``{split}.bin`` / ``.len`` / ``.rmsd``),
with pK stored in the label stream, so ``train_rescore.py`` trains on it unchanged.

Run (single GPU)::

    uv run python pipelines/corpora/tokenize_affinity_biolip.py \
        --ckpt <atom-vqvae>.ckpt \
        --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
        --out-dir data/lm_tokens_affinity
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Sibling modules in this directory, imported by bare name: Python puts a
# script's own directory on sys.path[0], so this resolves from any cwd.
from tokenize_biolip import (
    _bucket_code,
    _load_ccd_smiles,
    _read_needed,
)
from tokenize_decoys import _cd_test_pdbs, _RmsdWriter

from prolit.config import AtomVQVAETrainingConfig, PocketExtractionConfig
from prolit.model.vqvae_module import AtomVQVAEModule
from prolit.tokenizers.ligand import parse_ligand_pdb_text
from prolit.tokenizers.lm_vocab import AtomLMVocab
from prolit.tokenizers.pose_encoder import PoseEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BioLIP columns (0-indexed): 13 literature, 14 Binding MOAD, 15 PDBbind-CN,
# 16 BindingDB. Prefer the better-curated sources first.
_AFF_COLS = ((15, "pdbbind"), (14, "moad"), (13, "literature"), (16, "bindingdb"))
_UNITS = {"mm": 1e-3, "um": 1e-6, "nm": 1e-9, "pm": 1e-12, "fm": 1e-15, "m": 1.0}
_AFF_RE = re.compile(
    r"(Kd|Ki|IC50)\s*[=~<>]\s*([0-9.eE+-]+)\s*([munpf]?M)", re.IGNORECASE
)


def _to_pk(text: str) -> float | None:
    """'Kd=289uM' -> pK (= -log10 molar). None if unparseable."""
    m = _AFF_RE.search(text or "")
    if not m:
        return None
    try:
        val = float(m.group(2))
    except ValueError:
        return None
    unit = _UNITS.get(m.group(3).lower())
    if not unit or val <= 0:
        return None
    return -math.log10(val * unit)


def _parse_affinity_sites(
    path: Path, *, pk_min: float, pk_max: float, kinds: set[str] | None = None
) -> list[tuple[str, str, str, str, str, float, str]]:
    """(pdb, rchain, ccd, ligchain, serial, pK, uniprot) for sites with a value.

    ``kinds`` restricts the measurement type (e.g. {"KD", "KI"}). CASF-2016 is
    labelled with Kd/Ki only -- zero IC50 -- so training on IC50 (44% of BioLIP's
    entries, and assay-condition dependent) is a train/test distribution mismatch.
    Measured empirically: the Kd/Ki-only filter halves the corpus and costs 0.044
    scoring R, so volume wins and IC50 is kept by default.

    The UniProt id (column 17) groups ligands by protein, which is what the
    within-protein ranking loss needs -- 1,437 proteins carry 2+ ligands.
    """
    import gzip  # noqa: PLC0415

    out: list[tuple[str, str, str, str, str, float, str]] = []
    seen: set[tuple[str, str]] = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 18:  # noqa: PLR2004  need col 17 (UniProt)
                continue
            pk = None
            for idx, _src in _AFF_COLS:
                if kinds is not None:
                    m = _AFF_RE.search(c[idx] or "")
                    if m and m.group(1).upper() not in kinds:
                        continue  # wrong measurement type -> try next source
                pk = _to_pk(c[idx])
                if pk is not None:
                    break
            if pk is None or not (pk_min <= pk <= pk_max):
                continue
            pdb, ccd = c[0].lower(), c[4]
            if (pdb, ccd) in seen:  # one entry per (pdb, ligand)
                continue
            seen.add((pdb, ccd))
            out.append((pdb, c[1], ccd, c[5], c[6], pk, c[17].strip()))
    return out


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--separate-protein-ckpt", type=Path, default=None)
    p.add_argument("--separate-protein-norm", type=Path, default=None)
    p.add_argument("--separate-ligand-ckpt", type=Path, default=None)
    p.add_argument("--separate-ligand-norm", type=Path, default=None)
    p.add_argument("--norm-stats", type=Path, default=None)
    p.add_argument("--biolip-dir", type=Path, default=Path("data/biolip"))
    p.add_argument(
        "--cd-manifest", type=Path, default=Path("data/hub_cache/repo/manifest.parquet")
    )
    p.add_argument("--casf-pdbs", type=Path, default=Path("data/casf2016_pdbs.txt"))
    p.add_argument("--out-dir", type=Path, default=Path("data/lm_tokens_affinity"))
    p.add_argument("--codebook-size", type=int, default=8192)
    p.add_argument("--max-residues", type=int, default=50)
    p.add_argument("--min-heavy", type=int, default=6)
    p.add_argument("--max-heavy", type=int, default=60)
    p.add_argument(
        "--affinity-types",
        default="KD,KI",
        help="Measurement types to keep (CASF-2016 is Kd/Ki only; IC50 is "
        "assay-dependent). Use 'KD,KI,IC50' for everything.",
    )
    p.add_argument("--pk-min", type=float, default=2.0)
    p.add_argument("--pk-max", type=float, default=13.0)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    cfg = AtomVQVAETrainingConfig()
    cfg.atom.codebook_size = args.codebook_size
    if args.separate_protein_ckpt is not None:
        # ABLATION separate-tokenizers mode: protein-only VQ + ligand-only VQ
        # unified into one code space. Feed RAW descriptors (identity external
        # norm) via PoseEncoder -- SeparateVQVAE normalizes per modality
        # internally. Combined single-range AtomLMVocab over 2*codebook_size
        # codes. PoseEncoder encodes pocket + ligand in separate (single-
        # modality) encode_batch calls, which SeparateVQVAE requires.
        from prolit.tokenizers.descriptor_schema import (  # noqa: PLC0415
            ATOM_DESCRIPTOR_DIM,
        )
        from prolit.tokenizers.separate_vqvae import SeparateVQVAE  # noqa: PLC0415

        module = SeparateVQVAE.from_checkpoints(
            args.separate_protein_ckpt,
            args.separate_protein_norm,
            args.separate_ligand_ckpt,
            args.separate_ligand_norm,
            device,
            codebook_size=args.codebook_size,
        )
        mean = np.zeros(ATOM_DESCRIPTOR_DIM, dtype=np.float32)
        std = np.ones(ATOM_DESCRIPTOR_DIM, dtype=np.float32)
        vocab = AtomLMVocab(codebook_size=2 * args.codebook_size)
    else:
        module = AtomVQVAEModule.load_from_checkpoint(
            args.ckpt, config=cfg, map_location=device
        )
        module.eval().to(device)
        norm = torch.load(args.norm_stats, weights_only=False)
        module.vqvae.set_normalization(norm["atom_mean"], norm["atom_std"])
        mean = norm["atom_mean"].numpy()
        std = norm["atom_std"].numpy()
        vocab = AtomLMVocab(codebook_size=args.codebook_size)
    enc = PoseEncoder(
        module.vqvae,
        mean,
        std,
        vocab,
        device,
        PocketExtractionConfig(max_residues=args.max_residues),
    )

    kinds = {k.strip().upper() for k in args.affinity_types.split(",") if k.strip()}
    sites = _parse_affinity_sites(
        args.biolip_dir / "BioLiP.txt.gz",
        pk_min=args.pk_min,
        pk_max=args.pk_max,
        kinds=kinds or None,
    )
    ccd_smiles = _load_ccd_smiles(args.biolip_dir / "ligand.tsv.gz")
    excluded = _cd_test_pdbs(args.cd_manifest)
    if args.casf_pdbs.exists():
        excluded |= {x.lower() for x in args.casf_pdbs.read_text().split() if x.strip()}
    sites = [s for s in sites if s[0] not in excluded]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(sites)
    logger.info(
        "affinity sites (%s, CASF/CD-test excluded, pK %.1f-%.1f): %d",
        ",".join(sorted(kinds)),
        args.pk_min,
        args.pk_max,
        len(sites),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "train": _RmsdWriter(args.out_dir, "train"),
        "val": _RmsdWriter(args.out_dir, "val"),
    }

    # Protein groups: ligands sharing a UniProt id are ranked against each other
    # by the within-protein ranking loss. A site with no UniProt id becomes its
    # own singleton group -- it still trains the regression, just carries no pair.
    gid_of: dict[str, int] = {}
    site_gid: dict[tuple[str, str], int] = {}
    for s in sites:
        key = s[6] or f"__nouniprot_{s[0]}_{s[2]}"
        site_gid[(s[0], s[2])] = gid_of.setdefault(key, len(gid_of))
    per_group = Counter(site_gid.values())
    # Split val by PROTEIN, not by PDB: the same protein under a different PDB id
    # would otherwise sit on both sides and make early stopping optimistic.
    order = rng.permutation(len(gid_of)).tolist()
    target = len(sites) * args.val_frac
    val_gids: set[int] = set()
    acc = 0
    for g in order:
        if acc >= target:
            break
        val_gids.add(g)
        acc += per_group[g]
    n_multi = sum(1 for v in per_group.values() if v >= 2)  # noqa: PLR2004
    logger.info(
        "protein groups: %d (%d with 2+ ligands) | val groups %d (~%d sites)",
        len(gid_of),
        n_multi,
        len(val_gids),
        acc,
    )
    gids_out: dict[str, list[int]] = {"train": [], "val": []}

    by_bucket: dict[str, list] = {}
    for s in sites:
        by_bucket.setdefault(_bucket_code(s[0]), []).append(s)

    import tokenize_biolip as tb  # noqa: PLC0415

    tb._w_biolip_dir = args.biolip_dir  # noqa: SLF001

    from tqdm import tqdm  # noqa: PLC0415

    n_ok = 0
    pks: list[float] = []
    for code in tqdm(sorted(by_bucket), desc="buckets"):
        site_list = by_bucket[code]
        rec_names = {f"{p_}{rc}.pdb" for p_, rc, _c, _l, _s, _k, _u in site_list}
        lig_names = {
            f"{p_}_{cc}_{lc}_{sr}.pdb" for p_, _rc, cc, lc, sr, _k, _u in site_list
        }
        receptors = _read_needed("receptor", code, rec_names)
        ligands = _read_needed("ligand", code, lig_names)
        for pdb, rchain, ccd, ligchain, serial, pk, _uniprot in site_list:
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
                heavy = np.array(
                    [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"],
                    np.float32,
                )
                if not (args.min_heavy <= heavy.shape[0] <= args.max_heavy):
                    continue
                # Pocket is carved around THIS ligand, so it cannot be cached per
                # receptor -- two ligands in one chain have different pockets.
                setup = enc.setup_pocket(rec.decode("utf-8", "replace"), heavy)
                if setup is None:
                    continue
                p_codes, frame = setup
                seq = enc.ligand_seq(p_codes, mol, frame)
                if seq is None:
                    continue
                gid = site_gid[(pdb, ccd)]
                split = "val" if gid in val_gids else "train"
                writers[split].write(seq, float(pk))
                gids_out[split].append(gid)
                pks.append(pk)
                n_ok += 1
            except Exception:
                logger.exception("failed %s_%s", pdb, ccd)
                continue

    meta = {
        "vocab_size": vocab.vocab_size,
        # Separate-tokenizers mode doubles the code space (protein then ligand).
        "atom_codebook_size": (
            2 * args.codebook_size
            if args.separate_protein_ckpt is not None
            else args.codebook_size
        ),
        "source": "biolip2_affinity_pk",
        "label": "pK (-log10 molar)",
        "n_complexes": n_ok,
        "train_docs": writers["train"].num_docs,
        "val_docs": writers["val"].num_docs,
        "max_len": max(writers["train"].max_len, writers["val"].max_len),
    }
    if args.separate_protein_ckpt is not None:
        meta["separate_tokenizers"] = True
    for w in writers.values():
        w.close()
    for split, gl in gids_out.items():
        np.asarray(gl, dtype=np.int32).tofile(args.out_dir / f"{split}.grp")
    meta["train_pair_groups"] = sum(
        1 for v in Counter(gids_out["train"]).values() if v >= 2  # noqa: PLR2004
    )
    torch.save(meta, args.out_dir / "meta.pt")
    a = np.array(pks) if pks else np.zeros(1)
    logger.info(
        "DONE: %d complexes -> train %d / val %d | pK med %.2f std %.2f | max_len %d",
        n_ok,
        writers["train"].num_docs,
        writers["val"].num_docs,
        float(np.median(a)),
        float(a.std()),
        meta["max_len"],
    )


if __name__ == "__main__":
    main()
