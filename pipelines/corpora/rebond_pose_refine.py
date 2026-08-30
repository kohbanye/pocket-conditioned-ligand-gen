"""Re-derive a pose-refine corpus's bonds the way deployment derives them.

The refiner takes the ligand's bonds as edge features and as the reference for
its bond-length loss. The corpus stores the *crystal* connectivity. Deployment
has no crystal to read: ``generate_ligands_for_target.py`` calls ``infer_bonds``
on the decoded coordinates, and those coordinates are wrong by 2.5 A RMSD, so
the perception is wrong with them. Measured on the CLM-corruption corpus:

    perceived from            Jaccard   bonds missed   bonds invented
    crystal coordinates        1.000        0.11            0.00
    CLM-decoded coordinates    0.313        0.44            0.63

Not one molecule's connectivity survives intact. A refiner fitted on the first
row is asked at generation time to work from the second.

This rewrites the bond streams from the corpus's own corrupted coordinates --
the scale-0 record, which is the decode itself with no augmentation on top --
so that training sees the graph deployment will hand it. The *targets* do not
move: ``lig_bond_ref`` stays the distance between those two atoms in the
crystal pose, so an invented bond is supervised to the true (large) distance
rather than pulled to a bond length. Everything else is hard-linked, not
copied.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

from prolit.api import LIGAND_ELEMENT_VOCAB
from prolit.chem.pdb_io import infer_bonds
from prolit.model.pose_refiner import FEATURE_FIELDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#: Streams the rewrite does not touch; hard-linked so a second corpus costs no
#: space (they are the same bytes on the same filesystem).
_SHARED = (
    "lig_x0",
    "lig_x1",
    "lig_feat",
    "pkt_x",
    "pkt_feat",
    "records",
    "record_scale",
)


def _rebond_split(src: Path, dst: Path, split: str, meta: dict) -> int:
    n_complexes = meta["splits"][split]["num_complexes"]
    n_records = meta["splits"][split]["num_records"]
    if n_complexes == 0:
        for name in (*_SHARED, "complexes", "lig_bonds", "lig_bond_ref"):
            p = src / f"{split}.{name}"
            if p.exists():
                os.link(p, dst / f"{split}.{name}")
        return 0

    comp = np.fromfile(src / f"{split}.complexes", dtype=np.int64).reshape(-1, 3)
    comp = comp[:n_complexes].copy()
    x1 = np.fromfile(src / f"{split}.lig_x1", dtype=np.float32).reshape(-1, 3)
    x0 = np.fromfile(src / f"{split}.lig_x0", dtype=np.float32).reshape(-1, 3)
    feat = np.fromfile(src / f"{split}.lig_feat", dtype=np.int16).reshape(
        -1, len(FEATURE_FIELDS)
    )
    rec = np.fromfile(src / f"{split}.records", dtype=np.int64)[:n_records]
    scale = np.fromfile(src / f"{split}.record_scale", dtype=np.float32)[:n_records]

    # The scale-0 record of each complex is the decode with nothing added.
    off_x1 = np.concatenate([[0], np.cumsum(comp[:, 0])])
    x0_at = {}
    pos = 0
    for i, cid in enumerate(rec):
        n = int(comp[cid, 0])
        if scale[i] == 0.0 and cid not in x0_at:
            x0_at[int(cid)] = x0[pos : pos + n]
        pos += n

    elem_col = [name for name, _ in FEATURE_FIELDS].index("element")
    changed = 0
    with (
        (dst / f"{split}.lig_bonds").open("wb") as fb,
        (dst / f"{split}.lig_bond_ref").open("wb") as fr,
    ):
        for cid in range(n_complexes):
            n = int(comp[cid, 0])
            coords = x0_at.get(cid)
            els = [
                LIGAND_ELEMENT_VOCAB[k] if LIGAND_ELEMENT_VOCAB[k] != "OTHER" else "C"
                for k in feat[off_x1[cid] : off_x1[cid] + n, elem_col]
            ]
            pairs = (
                np.asarray(
                    [
                        (u, v)
                        for u, v, *_ in infer_bonds(els, coords.astype(np.float64))
                    ],
                    dtype=np.int32,
                ).reshape(-1, 2)
                if coords is not None
                else np.zeros((0, 2), dtype=np.int32)
            )
            tgt = x1[off_x1[cid] : off_x1[cid] + n]
            ref = (
                np.linalg.norm(tgt[pairs[:, 0]] - tgt[pairs[:, 1]], axis=1)
                if pairs.shape[0]
                else np.zeros(0)
            )
            fb.write(pairs.astype(np.int32).tobytes())
            fr.write(ref.astype(np.float32).tobytes())
            changed += int(pairs.shape[0] != comp[cid, 2])
            comp[cid, 2] = pairs.shape[0]
    comp.tofile(dst / f"{split}.complexes")
    for name in _SHARED:
        p = src / f"{split}.{name}"
        if p.exists():
            os.link(p, dst / f"{split}.{name}")
    logger.info(
        "%s: %d complexes, %d with a different bond count", split, n_complexes, changed
    )
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    meta = json.loads((a.src / "meta.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        if split in meta["splits"]:
            _rebond_split(a.src, a.out, split, meta)
    meta["bonds"] = "perceived_from_corrupted_coords"
    meta["rebonded_from"] = str(a.src)
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
