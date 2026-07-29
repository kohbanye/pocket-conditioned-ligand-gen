"""Concatenate sharded decoy token sets (``tokenize_decoys.py --num-shards``).

The shards are flat streams (``.bin`` uint16 tokens, ``.len`` uint16 lengths,
``.rmsd`` float32), so a byte-wise concatenation is a valid corpus as long as
every shard used the same vocabulary and the same val-split rule -- which they
do, being the same command with a different ``--shard-id``.

    python scripts/concat_decoy_shards.py data/lm_tokens_decoys_v8
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEAR_NATIVE = 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--pattern", default="shard*")
    args = parser.parse_args()

    shards = sorted(p for p in args.out_dir.glob(args.pattern) if p.is_dir())
    if not shards:
        msg = f"no shards matching {args.pattern} under {args.out_dir}"
        raise SystemExit(msg)
    logger.info("merging %d shards", len(shards))

    meta = json.loads((shards[0] / "meta.json").read_text())
    meta["splits"] = {}
    meta["complexes_used"] = 0
    for s in shards:
        m = json.loads((s / "meta.json").read_text())
        meta["complexes_used"] += m["complexes_used"]

    for split in ("train", "val"):
        n_docs = n_tok = max_len = 0
        # disp/dlen carry the per-ligand-atom displacement labels; they only
        # exist for corpora built with that supervision, hence the skip below.
        for ext in ("bin", "len", "rmsd", "disp", "dlen"):
            if not any((s / f"{split}.{ext}").exists() for s in shards):
                continue
            with (args.out_dir / f"{split}.{ext}").open("wb") as out:
                for s in shards:
                    f = s / f"{split}.{ext}"
                    if f.exists() and f.stat().st_size:
                        with f.open("rb") as fh:
                            shutil.copyfileobj(fh, out, 1 << 22)
        lens = np.fromfile(args.out_dir / f"{split}.len", dtype=np.uint16)
        rmsd = np.fromfile(args.out_dir / f"{split}.rmsd", dtype=np.float32)
        if len(lens) != len(rmsd):
            msg = f"{split}: {len(lens)} lens vs {len(rmsd)} rmsds -- truncated shard?"
            raise SystemExit(msg)
        n_docs, n_tok, max_len = len(lens), int(lens.sum()), int(lens.max(initial=0))
        meta["splits"][split] = {
            "num_docs": n_docs,
            "num_tokens": n_tok,
            "max_len": max_len,
        }
        logger.info(
            "%s: %d docs, %d tokens, max_len %d, frac<2A %.3f",
            split, n_docs, n_tok, max_len, float((rmsd < NEAR_NATIVE).mean()),
        )
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(
        "wrote %s/meta.json (%d complexes)", args.out_dir, meta["complexes_used"]
    )


if __name__ == "__main__":
    main()
