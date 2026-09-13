"""Concatenate packed token caches into one mixed-corpus cache.

Used to build the LM pretraining corpus by merging the protein-only (PLINDER)
and ligand-only (GEOM) all-atom token caches. All inputs must share the same
vocab; the packed format (``{split}.bin`` uint16 stream + ``{split}.len`` uint16
doc lengths) concatenates trivially because docs are delimited by ``.len``.

Block-level shuffling at train time interleaves the (homogeneous) protein and
ligand blocks across each epoch, so no doc-level interleaving is needed here.

Two cache layouts exist and this handles both, because merging one of them
without its sidecars loses the labels silently -- the documents survive, the
supervision does not:

* **LM caches** carry ``meta.json`` and nothing but ``.bin`` / ``.len``.
* **Scoring caches** (``tokenize_affinity_*``, ``tokenize_decoys``) carry
  ``meta.pt`` and per-document sidecars: ``.rmsd`` (the label), ``.grp`` (the
  group a listwise or ranking loss forms its lists within), and the optional
  ``.comp`` / ``.dlen`` / ``.disp``.

Group ids are LOCAL to a cache, so concatenating them unchanged would merge two
corpora's unrelated proteins into one group and let a ranking loss compare
ligands that never shared a target. Each input's ids are therefore offset past
every id already written.

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

#: Per-document sidecars a scoring cache carries next to ``.bin`` / ``.len``.
#: ``.grp`` is handled separately because its values need offsetting.
_LABEL_SIDECARS = ("rmsd", "comp", "dlen", "disp")


def _cat(srcs: list[Path], dst: Path) -> bool:
    present = [s for s in srcs if s.exists() and s.stat().st_size > 0]
    if not present:
        return False
    with dst.open("wb") as out:
        for s in present:
            with s.open("rb") as f:
                shutil.copyfileobj(f, out)
    return True


def _cat_groups(srcs: list[Path], counts: list[int], dst: Path) -> bool:
    """Concatenate ``.grp`` streams, offsetting each input past the last.

    Ids are local to a cache: two corpora both number their first protein 0.
    Concatenated as-is, a grouped loss would build lists spanning two corpora
    and rank ligands of different targets against each other.

    An input with no ``.grp`` at all (``tokenize_affinity_pdbbind`` writes none)
    contributes one singleton group per document, which is what "this document
    shares a target with nothing I know of" means -- a grouped loss draws no
    pair from it. Skipping the input instead would leave the stream shorter than
    the documents and shift every later group onto the wrong one.
    """
    if not any(s.exists() and s.stat().st_size > 0 for s in srcs):
        return False
    out, offset = [], 0
    for src, n_docs in zip(srcs, counts, strict=True):
        if src.exists() and src.stat().st_size > 0:
            ids = np.fromfile(src, dtype=np.int32).astype(np.int64)
        else:
            ids = np.arange(n_docs, dtype=np.int64)
        if ids.size:
            out.append(ids + offset)
            offset += int(ids.max()) + 1
    np.concatenate(out).astype(np.int32).tofile(dst)
    return True


def _doc_count(d: Path, split: str) -> int:
    """Documents in one cache's split, from its ``.len`` stream."""
    path = d / f"{split}.len"
    return path.stat().st_size // 2 if path.exists() else 0


def _read_meta(d: Path) -> dict:
    """A cache's metadata, from whichever of the two layouts it uses."""
    if (d / "meta.json").exists():
        return json.loads((d / "meta.json").read_text())
    if (d / "meta.pt").exists():
        import torch  # noqa: PLC0415  (only the scoring layout needs it)

        return dict(torch.load(d / "meta.pt", weights_only=False))
    msg = f"{d}: no meta.json or meta.pt -- not a token cache"
    raise FileNotFoundError(msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["train", "val"]
    )
    args = parser.parse_args()

    metas = [_read_meta(d) for d in args.inputs]
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
        for name in _LABEL_SIDECARS:
            _cat(
                [d / f"{split}.{name}" for d in args.inputs],
                args.out_dir / f"{split}.{name}",
            )
        _cat_groups(
            [d / f"{split}.grp" for d in args.inputs],
            [_doc_count(d, split) for d in args.inputs],
            args.out_dir / f"{split}.grp",
        )
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
        # A per-document sidecar that came out shorter than the document stream
        # means an input was missing it, which would silently shift every later
        # value onto the wrong document rather than fail.
        for name, itemsize in (("rmsd", 4), ("grp", 4)):
            path = args.out_dir / f"{split}.{name}"
            if not path.exists():
                continue
            n = path.stat().st_size // itemsize
            if n != lengths.size:
                msg = (
                    f"{split}: {lengths.size} docs but {n} .{name} entries. An "
                    f"input is missing its .{name} sidecar; merging would "
                    "misalign it against the documents."
                )
                raise ValueError(msg)
        print(
            f"{split}: {lengths.size} docs, {int(lengths.sum())} tokens, "
            f"max_len={int(lengths.max()) if lengths.size else 0}"
        )

    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote mixed cache to {args.out_dir}")


if __name__ == "__main__":
    main()
