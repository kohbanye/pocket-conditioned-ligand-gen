"""Build a pose-refiner training set that distils the analytic clash relief.

Why distillation. `vina_score` is scored on the pose as generated, and the thing
that moves it is ligand-pocket overlap: an analytic relief (scipy L-BFGS on a
one-sided repulsion against a rigid pocket) took the 100-pocket mean from -2.92 to
-5.59, while swapping between the three trained refiner checkpoints moved it by at
most 0.2. But running that optimiser at evaluation time makes the comparison with
DiffSBDD/DiffGui unfair -- they report poses straight out of the model. Distilling
it into the refiner keeps the gain and restores the comparison: training-time
physics, inference-time network.

Input is a pair of arm directories under ``sbdd-bench/outputs``: the generated
poses (``--src-arm``) and their relaxed counterparts (``--dst-arm``), written by
``scripts/relax_in_pocket.py`` so atom order is preserved. Output is the memmap
layout ``src/data/pose_refine_dataset.py`` reads, so the source repo's
``pipelines/train/refiner.py`` consumes it unchanged.

Coordinates stay in the receptor's own frame. The refiner is E(3)-equivariant --
it reads only relative edge vectors -- so a model trained in the PDB frame applies
unchanged to the pocket canonical frame used at generation time.

Usage (source-repo interpreter, it owns ``prolit.tokenizers``)::

    <src-repo>/.venv/bin/python scripts/build_distill_refine_set.py \
        --src-arm sep4096_cs --dst-arm sep4096_cs_rx105 \
        --out-dir <src-repo>/data/pose_refine_distill --val-targets 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SOURCE_REPO = Path(__file__).resolve().parents[3]
SBDD_BENCH = Path(__file__).resolve().parents[2] / "sbdd-bench"
sys.path.insert(0, str(SOURCE_REPO))

from prolit.config import PocketExtractionConfig  # noqa: E402
from prolit.model.pose_refiner import (  # noqa: E402
    ligand_feats_from_heads,
    pocket_feats_from_descriptor,
)
from prolit.tokenizers.atom import (  # noqa: E402
    ProteinAtomDescriptor,
    precompute_receptor_atom_features_from_text,
)
from prolit.tokenizers.descriptor_schema import (  # noqa: E402
    LIGAND_CHARGE_TO_IDX,
    LIGAND_ELEMENT_TO_IDX,
    LIGAND_HYBRID_VOCAB,
    LIGAND_OTHER_IDX,
    LIGAND_RING_NONE_IDX,
)
from prolit.tokenizers.protein import (  # noqa: E402
    extract_pocket_atoms_from_candidates,
    precompute_pocket_atom_candidates_from_text,
)
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_HYBRID_IDX = {name: i for i, name in enumerate(LIGAND_HYBRID_VOCAB)}
_RING_IDX = {3: 0, 4: 1, 5: 2}  # R3 / R4 / R5; anything larger is R6+ (3)
_MAX_H = 4
_MAX_CHARGE = 2


def ligand_feats_from_mol(mol: Chem.Mol) -> np.ndarray:
    """The refiner's (n, 9) node-feature block, derived from an RDKit molecule.

    At generation time these come from the VQ decoder's chemistry heads; here the
    molecule is already perceived, so they are read off the graph instead. The
    field order and vocabularies are the schema's, not ours.
    """
    ring_info = mol.GetRingInfo()
    chem: dict[str, list[int]] = {k: [] for k in
                                  ("element", "charge", "hybrid", "aromatic", "ring", "numH")}
    for atom in mol.GetAtoms():
        chem["element"].append(
            LIGAND_ELEMENT_TO_IDX.get(atom.GetSymbol(), LIGAND_OTHER_IDX)
        )
        charge = int(np.clip(atom.GetFormalCharge(), -_MAX_CHARGE, _MAX_CHARGE))
        chem["charge"].append(LIGAND_CHARGE_TO_IDX[charge])
        if atom.GetIsAromatic():
            hyb = _HYBRID_IDX["AROM"]
        else:
            hyb = _HYBRID_IDX.get(str(atom.GetHybridization()), _HYBRID_IDX["OTHER"])
        chem["hybrid"].append(hyb)
        chem["aromatic"].append(int(atom.GetIsAromatic()))
        sizes = [s for s in (3, 4, 5, 6) if ring_info.IsAtomInRingOfSize(atom.GetIdx(), s)]
        if not sizes:
            ring = 3 if atom.IsInRing() else LIGAND_RING_NONE_IDX  # in a ring > 6 -> R6+
        else:
            ring = _RING_IDX.get(min(sizes), 3)
        chem["ring"].append(ring)
        chem["numH"].append(int(np.clip(atom.GetTotalNumHs(), 0, _MAX_H)))
    arrays = {k: np.asarray(v, dtype=np.int64) for k, v in chem.items()}
    return ligand_feats_from_heads(arrays, mol.GetNumAtoms())


def pocket_context(receptor_pdb: Path, ref_sdf: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Pocket heavy-atom coordinates (receptor frame) + their (M, 9) features."""
    ref = Chem.MolFromMolFile(str(ref_sdf), sanitize=False)
    if ref is None or ref.GetNumConformers() == 0:
        return None
    ref_heavy = np.asarray(
        [
            list(ref.GetConformer().GetAtomPosition(i))
            for i, a in enumerate(ref.GetAtoms())
            if a.GetAtomicNum() > 1
        ],
        dtype=np.float32,
    )
    text = receptor_pdb.read_text()
    pocket = extract_pocket_atoms_from_candidates(
        precompute_pocket_atom_candidates_from_text(text),
        ref_heavy,
        PocketExtractionConfig(),
    )
    if pocket is None or pocket.atom_coords.shape[0] == 0:
        return None
    feats = precompute_receptor_atom_features_from_text(text)
    # Identity frame: only the categorical columns are read out, and they do not
    # depend on the frame.
    frame = (np.zeros(3), np.eye(3))
    desc, _ = ProteinAtomDescriptor().compute(pocket, feats, frame)
    if desc.shape[0] != pocket.atom_coords.shape[0]:
        return None
    return pocket.atom_coords.astype(np.float32), pocket_feats_from_descriptor(desc)


class Writer:
    """Streams the memmaps ``PoseRefineDataset`` expects."""

    STREAMS = (
        "lig_x1", "lig_feat", "lig_bonds", "lig_bond_ref", "pkt_x", "pkt_feat",
        "lig_x0", "records", "record_scale", "complexes",
    )

    def __init__(self, out_dir: Path, split: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._f = {n: (out_dir / f"{split}.{n}").open("wb") for n in self.STREAMS}
        self.n_complexes = 0
        self.n_records = 0

    def add(  # noqa: PLR0913
        self,
        x0: np.ndarray,
        x1: np.ndarray,
        lig_feat: np.ndarray,
        bonds: np.ndarray,
        pkt_x: np.ndarray,
        pkt_feat: np.ndarray,
    ) -> None:
        bond_ref = (
            np.linalg.norm(x1[bonds[:, 0]] - x1[bonds[:, 1]], axis=1).astype(np.float32)
            if bonds.shape[0]
            else np.zeros(0, dtype=np.float32)
        )
        self._f["lig_x1"].write(x1.astype(np.float32).tobytes())
        self._f["lig_feat"].write(lig_feat.astype(np.int16).tobytes())
        self._f["lig_bonds"].write(bonds.astype(np.int32).tobytes())
        self._f["lig_bond_ref"].write(bond_ref.tobytes())
        self._f["pkt_x"].write(pkt_x.astype(np.float32).tobytes())
        self._f["pkt_feat"].write(pkt_feat.astype(np.int16).tobytes())
        self._f["complexes"].write(
            np.asarray([x1.shape[0], pkt_x.shape[0], bonds.shape[0]], dtype=np.int64).tobytes()
        )
        self._f["lig_x0"].write(x0.astype(np.float32).tobytes())
        self._f["records"].write(np.asarray([self.n_complexes], dtype=np.int64).tobytes())
        self._f["record_scale"].write(np.asarray([0.0], dtype=np.float32).tobytes())
        self.n_complexes += 1
        self.n_records += 1

    def close(self) -> None:
        for fh in self._f.values():
            fh.close()


def main() -> None:  # noqa: C901, PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="src_arm:dst_arm pairs, e.g. sep4096_cs:sep4096_cs_rx105")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--index", type=Path,
                    default=SBDD_BENCH / "data" / "targets" / "index.json")
    ap.add_argument(
        "--only-targets",
        type=Path,
        default=None,
        help="Restrict the training set to these target ids (one per line). "
        "The distillation poses come from the benchmark's own pockets, so "
        "the refiner must never be trained on a pocket it is later scored "
        "on -- pass a split that is disjoint from the evaluation set.",
    )
    ap.add_argument("--val-targets", type=int, default=8)
    ap.add_argument("--max-per-target", type=int, default=100)
    # Drop pairs whose teacher moved the ligand implausibly far. On pockets
    # taken straight from hub_cache the reference ligand is occasionally
    # incomplete, pocket extraction then misses atoms and the relaxation runs
    # away -- those pairs teach the student nothing but noise (observed mean
    # displacement 1.43 A with a 19.9 A tail, vs 0.52 A on clean pockets).
    ap.add_argument("--max-disp", type=float, default=2.5)
    args = ap.parse_args()

    index = json.loads(args.index.read_text())
    targets = index["targets"] if isinstance(index, dict) and "targets" in index else index
    by_id = {t["target_id"]: t for t in targets}
    if args.only_targets:
        keep = set(args.only_targets.read_text().split())
        by_id = {k: v for k, v in by_id.items() if k in keep}
        targets = [t for t in targets if t["target_id"] in keep]
    tdir = args.index.parent

    val_ids = {t["target_id"] for t in targets[: args.val_targets]}
    writers = {s: Writer(args.out_dir, s) for s in ("train", "val")}
    n_pairs_used = n_skipped = 0

    for spec in args.pairs:
        src_arm, dst_arm = spec.split(":")
        for tid, meta in by_id.items():
            src = SBDD_BENCH / "outputs" / src_arm / "own" / tid / "generated.sdf"
            dst = SBDD_BENCH / "outputs" / dst_arm / "own" / tid / "generated.sdf"
            if not src.exists() or not dst.exists():
                continue
            ctx = pocket_context(tdir / meta["receptor_pdb"], tdir / meta["ref_ligand_sdf"])
            if ctx is None:
                continue
            pkt_x, pkt_feat = ctx
            w = writers["val" if tid in val_ids else "train"]
            a_iter = Chem.SDMolSupplier(str(src), sanitize=True, removeHs=True)
            b_iter = Chem.SDMolSupplier(str(dst), sanitize=True, removeHs=True)
            for i, (a, b) in enumerate(zip(a_iter, b_iter, strict=False)):
                if i >= args.max_per_target:
                    break
                if a is None or b is None or a.GetNumAtoms() != b.GetNumAtoms():
                    n_skipped += 1
                    continue
                x0 = np.asarray(a.GetConformer().GetPositions(), dtype=np.float32)
                x1 = np.asarray(b.GetConformer().GetPositions(), dtype=np.float32)
                bonds = np.asarray(
                    [(bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()) for bd in a.GetBonds()],
                    dtype=np.int32,
                ).reshape(-1, 2)
                try:
                    feat = ligand_feats_from_mol(a)
                except Exception:  # noqa: BLE001
                    n_skipped += 1
                    continue
                if args.max_disp > 0:
                    disp = float(np.linalg.norm(x1 - x0, axis=1).max())
                    if disp > args.max_disp:
                        n_skipped += 1
                        continue
                w.add(x0, x1, feat, bonds, pkt_x, pkt_feat)
                n_pairs_used += 1
            print(f"[distill] {src_arm}->{dst_arm} {tid}: total pairs {n_pairs_used}", flush=True)

    meta_out = {
        "source": "distill_relaxed_poses",
        "pairs": args.pairs,
        "feature_fields": ["source", "element", "charge", "hybrid", "aromatic",
                           "ring", "numH", "aa", "bb_sc"],
        "n_skipped": n_skipped,
        "splits": {
            s: {"num_complexes": w.n_complexes, "num_records": w.n_records}
            for s, w in writers.items()
        },
    }
    for w in writers.values():
        w.close()
    (args.out_dir / "meta.json").write_text(json.dumps(meta_out, indent=1))
    print(json.dumps(meta_out, indent=1))


if __name__ == "__main__":
    main()
