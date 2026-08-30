"""Concatenate sharded pose-refine corpora into one loadable directory.

``tokenize_pose_refine.py --shard I/N`` writes N independent corpora. Every
array in the format is a flat append-only stream indexed by cumulative counts,
so merging is concatenation -- with one exception: ``records`` stores a complex
id, and each shard numbered its complexes from zero. Those ids are the only
thing that needs rewriting, by the number of complexes the shards before it
contributed.

Bond indices are ligand-local and pocket pointers are resolved through the same
per-complex counts, so neither moves.

    .venv/bin/python pipelines/corpora/merge_pose_refine.py \
        --out data/pose_refine_clm data/pose_refine_clm_s*
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#: Streams that concatenate untouched, and the dtype each is stored in.
_PLAIN = {
    "lig_x1": np.float32,
    "lig_feat": np.int16,
    "lig_bonds": np.int32,
    "lig_bond_ref": np.float32,
    "pkt_x": np.float32,
    "pkt_feat": np.int16,
    "complexes": np.int64,
    "lig_x0": np.float32,
    "record_scale": np.float32,
}


def _merge_split(shards: list[Path], out: Path, split: str) -> dict:
    metas = [json.loads((d / "meta.json").read_text()) for d in shards]
    n_complexes = n_records = 0
    for name, dtype in _PLAIN.items():
        with (out / f"{split}.{name}").open("wb") as fh:
            for d in shards:
                src = d / f"{split}.{name}"
                if src.exists():
                    fh.write(np.fromfile(src, dtype=dtype).tobytes())
    with (out / f"{split}.records").open("wb") as fh:
        base = 0
        for d, m in zip(shards, metas, strict=True):
            src = d / f"{split}.records"
            if src.exists():
                rec = np.fromfile(src, dtype=np.int64)
                fh.write((rec + base).tobytes())
                n_records += len(rec)
            base += m["splits"].get(split, {}).get("num_complexes", 0)
        n_complexes = base
    return {"num_complexes": n_complexes, "num_records": n_records}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    shards = sorted(a.shards)
    a.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((shards[0] / "meta.json").read_text())
    meta["splits"] = {}
    meta["complexes_used"] = 0
    for split in ("train", "val"):
        meta["splits"][split] = _merge_split(shards, a.out, split)
        logger.info(
            "%s: %d complexes, %d records",
            split,
            meta["splits"][split]["num_complexes"],
            meta["splits"][split]["num_records"],
        )
    meta["complexes_used"] = sum(
        json.loads((d / "meta.json").read_text())["complexes_used"] for d in shards
    )
    meta["merged_from"] = [str(d) for d in shards]
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("merged %d shards into %s", len(shards), a.out)


if __name__ == "__main__":
    main()
