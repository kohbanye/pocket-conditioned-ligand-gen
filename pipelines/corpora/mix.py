"""Concatenate packed token caches into one mixed-corpus cache.

Used to build the LM pretraining corpus by merging the protein-only (PLINDER)
and ligand-only (GEOM) all-atom token caches. All inputs must share the same
vocab; the packed format (``{split}.bin`` uint16 stream + ``{split}.len`` uint16
doc lengths) concatenates trivially because docs are delimited by ``.len``.

Block-level shuffling at train time interleaves the (homogeneous) protein and
ligand blocks across each epoch, so no doc-level interleaving is needed here.

Run::

    uv run python pipelines/corpora/mix.py \
        --inputs data/lm_tokens_protein_plinder data/lm_tokens_geom_allatom \
        --out-dir data/lm_tokens_pretrain_mixed
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _cat(srcs: list[Path], dst: Path) -> bool:
    present = [s for s in srcs if s.exists() and s.stat().st_size > 0]
    if not present:
        return False
    with dst.open("wb") as out:
        for s in present:
            with s.open("rb") as f:
                shutil.copyfileobj(f, out)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["train", "val"]
    )
    args = parser.parse_args()

    metas = [json.loads((d / "meta.json").read_text()) for d in args.inputs]
    vocab = metas[0]["vocab_size"]
    if any(m["vocab_size"] != vocab for m in metas):
        msg = f"vocab mismatch across inputs: {[m['vocab_size'] for m in metas]}"
        raise ValueError(msg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "vocab_size": vocab,
        "atom_codebook_size": metas[0].get("atom_codebook_size"),
        "atom_offset": metas[0].get("atom_offset"),
        "all_atom": True,
        "mixed_from": [str(d) for d in args.inputs],
        "splits": {},
    }
    for split in args.splits:
        bin_ok = _cat(
            [d / f"{split}.bin" for d in args.inputs], args.out_dir / f"{split}.bin"
        )
        _cat([d / f"{split}.len" for d in args.inputs], args.out_dir / f"{split}.len")
        if not bin_ok:
            continue
        lengths = np.fromfile(args.out_dir / f"{split}.len", dtype=np.uint16).astype(
            np.int64
        )
        meta["splits"][split] = {
            "num_docs": int(lengths.size),
            "num_tokens": int(lengths.sum()),
            "max_len": int(lengths.max()) if lengths.size else 0,
        }
        print(
            f"{split}: {lengths.size} docs, {int(lengths.sum())} tokens, "
            f"max_len={int(lengths.max()) if lengths.size else 0}"
        )

    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote mixed cache to {args.out_dir}")


if __name__ == "__main__":
    main()
