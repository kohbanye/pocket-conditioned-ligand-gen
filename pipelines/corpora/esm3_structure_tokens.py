"""Cache ESM3 structure tokens for every residue of a set of receptors.

The stapled baseline (``prolit.tokenizers.stapled``) conditions on ESM3's
structure tokens where ProLIT conditions on its own pocket-atom tokens.

**Run this with an interpreter that has ``esm`` installed, which is not the one
the rest of the pipeline uses.** ESM3's package pins a fork of ``transformers``
that must not sit beside the one ProLIT's language models need, so the two
environments have to stay apart. This script needs nothing from that
environment except the ``esm`` package itself -- it imports no evaluation code
and shells into nothing -- and what it writes is read back through
:class:`prolit.data.esm3_tokens.Esm3TokenCache`, which needs only numpy. That
is the whole reason the cache exists: it is the seam between the two
environments::

    <interpreter with esm> pipelines/corpora/esm3_structure_tokens.py \\
        --manifest data/esm3_manifests/biolip.jsonl \\
        --out-dir data/esm3_tokens_biolip

``--manifest`` is JSON lines -- one file, or a directory of ``*.jsonl.gz``,
which is what ``tokenize_decoys.py --dump-receptors`` writes. One structure per
line, in any of three shapes::

    {"id": "1abcA", "pdb": "ATOM ...\\n..."}          # text inline
    {"id": "1abc", "path": "/.../1abc_receptor.pdb"}
    {"id": "5xyz", "tar": "/.../shard_003.tar", "member": "5xyz/receptor.pdb"}

**The whole structure is encoded, not the pocket.** Encoding a pocket alone
renumbers discontiguous residues 1..L, presenting residues angstroms apart as
chain neighbours; measured on PoseBusters that costs 8.48 A of backbone Kabsch
against 0.30 A for the same residues read out of a full-chain encoding. Caching
per structure also means several corpora that cut different pockets from one
receptor share the expensive half.

Resumable: a shard already on disk is skipped, so a job that hits its walltime
costs only the shard in flight.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import tarfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:  # bench interpreter has no prolit
    sys.path.insert(0, str(REPO_ROOT / "src"))

from prolit.data.esm3_tokens import write_shard  # noqa: E402
from prolit.seeding import add_seed_argument, seed_from_args  # noqa: E402
from prolit.tokenizers.esm3_layout import chain_break_layout  # noqa: E402
from prolit.tokenizers.protein import (  # noqa: E402
    precompute_pocket_atom_candidates_from_text,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_BACKBONE = ("N", "CA", "C")


def _read_text(rec: dict, tar_cache: dict[str, tarfile.TarFile]) -> str | None:
    """Structure text from a plain path or from inside a tar, without unpacking.

    Receptor sets run to hundreds of thousands of files and the group filesystem
    has a limited inode budget, so archives are streamed rather than extracted.
    """
    if "pdb" in rec:
        return rec["pdb"]
    if "path" in rec:
        p = Path(rec["path"])
        return p.read_text() if p.exists() else None
    tar_path = rec["tar"]
    tf = tar_cache.get(tar_path)
    if tf is None:
        tf = tarfile.open(tar_path)  # noqa: SIM115 -- kept open across the shard
        tar_cache[tar_path] = tf
    try:
        f = tf.extractfile(rec["member"])
    except KeyError:
        return None
    return None if f is None else f.read().decode("utf-8", "replace")


def _backbone(text: str) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]] | None:
    """(L, 3, 3) N/CA/C, chain ids, and (chain, author resid) per residue.

    Read through ProLIT's own receptor parser, so the residue keys here are the
    keys its pocket extraction will later ask for. A residue missing N or C
    falls back to its CA, which is what the reconstruction benchmark's reader
    does -- ESM3 is robust to it and dropping the residue instead would put a
    hole in the chain that its relative-position embedding would read as a gap.
    """
    pre = precompute_pocket_atom_candidates_from_text(text)
    if len(pre.ca_coords) == 0:
        return None
    coords = np.empty((len(pre.ca_coords), 3, 3), dtype=np.float64)
    for i, atoms in enumerate(pre.residue_atoms):
        by_name = {name.strip(): coord for name, _elem, coord in atoms}
        ca = by_name.get("CA", pre.ca_coords[i])
        coords[i] = np.stack([by_name.get(n, ca) for n in _BACKBONE])
    keys = list(zip(pre.chain_ids, pre.residue_indices, strict=True))
    return coords, np.asarray(pre.chain_ids), [(str(c), int(r)) for c, r in keys]


def _load_manifest(
    path: Path, part_k: int | None = None, part_n: int | None = None
) -> list[dict]:
    """One JSON-lines file, or every ``*.jsonl.gz`` in a directory.

    Sorted by filename so a resumed run shards the same structures into the same
    npz files; the shard index is part of the cache key, and reshuffling it
    would silently orphan every id an earlier run already wrote.
    """
    files = sorted(path.glob("*.jsonl.gz")) if path.is_dir() else [path]
    if part_k is not None and part_n:
        files = files[part_k::part_n]
    records: list[dict] = []
    for f in files:
        opener = gzip.open if f.suffix == ".gz" else open
        with opener(f, "rt") as fh:
            records.extend(json.loads(line) for line in fh if line.strip())
    return records


class _Encoder:
    """ESM3's structure encoder, loaded once."""

    def __init__(self, device: str | None = None) -> None:
        import torch  # noqa: PLC0415
        from esm.pretrained import (  # noqa: PLC0415
            ESM3_structure_encoder_v0,
        )

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ESM3_structure_encoder_v0(self.device)

    def encode(self, coords: np.ndarray, chain_ids: np.ndarray) -> np.ndarray | None:
        """One structure token per residue, in input row order."""
        laid, residue_index, is_residue, order = chain_break_layout(coords, chain_ids)
        torch = self.torch
        x = torch.from_numpy(laid).float().unsqueeze(0).to(self.device)
        ridx = torch.from_numpy(residue_index).long().unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, tokens = self.model.encode(x, residue_index=ridx)
        tok = tokens[0].detach().cpu().numpy()[is_residue]
        # ``order`` is the input row each emitted residue came from; invert it so
        # the cache is keyed the way the caller reads its residues.
        back = np.empty(order.size, dtype=np.int64)
        back[order] = np.arange(order.size)
        return tok[back]


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="structures per npz. Sharded because one file per receptor would "
        "spend hundreds of thousands of inodes on a shared filesystem.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after this many structures"
    )
    parser.add_argument(
        "--part",
        default=None,
        metavar="K/N",
        help="take every Nth manifest FILE, starting at K, so several GPUs can "
        "fill one cache at once. Each part writes its own index and its own "
        "shard files (see --shard-offset); the loader merges every index it "
        "finds, so no separate merge step is needed.",
    )
    parser.add_argument(
        "--shard-offset",
        type=int,
        default=None,
        help="first shard number this part writes (default: K * 1000, which "
        "keeps parts from colliding without either knowing the other's size)",
    )
    parser.add_argument("--device", default=None)
    add_seed_argument(parser)
    args = parser.parse_args()
    seed_from_args(args)

    part_k = None
    if args.part is not None:
        part_k, part_n = (int(v) for v in args.part.split("/"))
        if not 0 <= part_k < part_n:
            msg = f"--part {args.part}: need 0 <= K < N"
            raise SystemExit(msg)
    records = _load_manifest(args.manifest, part_k, args.part and part_n)
    shard_base = (
        args.shard_offset
        if args.shard_offset is not None
        else (0 if part_k is None else part_k * 1000)
    )
    if args.limit:
        records = records[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("%d structures -> %s", len(records), args.out_dir)

    encoder: _Encoder | None = None
    tar_cache: dict[str, tarfile.TarFile] = {}
    index: dict[str, list[int]] = {}
    n_shards = 0
    failures = 0
    for shard_idx, start in enumerate(range(0, len(records), args.shard_size)):
        chunk = records[start : start + args.shard_size]
        shard_no = shard_base + shard_idx
        shard_path = args.out_dir / f"shard_{shard_no:04d}.npz"
        n_shards = shard_idx + 1
        if shard_path.exists():
            # Resume: recover this shard's ids without re-encoding it.
            with np.load(shard_path, allow_pickle=True) as z:
                for slot, sid in enumerate(z["struct_ids"]):
                    index[str(sid)] = [shard_no, slot]
            logger.info("shard %d: exists, skipped", shard_idx)
            continue
        if encoder is None:
            encoder = _Encoder(args.device)
        entries = []
        for rec in chunk:
            text = _read_text(rec, tar_cache)
            if text is None:
                failures += 1
                continue
            parsed = _backbone(text)
            if parsed is None:
                failures += 1
                continue
            coords, chain_ids, keys = parsed
            try:
                tokens = encoder.encode(coords, chain_ids)
            except (RuntimeError, ValueError) as exc:
                logger.warning("%s: encode failed: %s", rec["id"], exc)
                failures += 1
                continue
            index[str(rec["id"])] = [shard_no, len(entries)]
            entries.append((str(rec["id"]), keys, tokens))
        write_shard(shard_path, entries)
        logger.info(
            "shard %d: %d structures -> %s", shard_idx, len(entries), shard_path
        )

    name = "index.json" if part_k is None else f"index.part{part_k}.json"
    (args.out_dir / name).write_text(json.dumps({"shards": n_shards, "ids": index}))
    logger.info("cached %d structures, %d failed", len(index), failures)
    for tf in tar_cache.values():
        tf.close()


if __name__ == "__main__":
    main()
