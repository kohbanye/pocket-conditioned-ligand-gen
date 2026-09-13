"""Tokenize CrossDocked poses as ESM3 x ConfSeq streams (the stapled baseline).

The counterpart of ``tokenize_crossdocked.py`` for the baseline the paper
compares against, and the corpus ProLIT-CLM's opposite number is trained on.

**It cannot read the descriptor cache.** ``tokenize_crossdocked.py`` starts from
``data/descriptor_cache_*``, whose entries hold 33-D per-atom descriptors and
nothing else -- no bonds, no element-resolved coordinates -- while ConfSeq needs
an RDKit molecule. So this streams the ligand tars the descriptor cache was
itself built from (:func:`prolit.data.atom_tar_prep.iter_tar_poses`), and the
two builders share the walk rather than each having one.

What is held constant with the ProLIT corpus, because the models trained on the
two are compared directly:

* the pocket split, via :func:`prolit.data.holdout.crossdocked_pocket_split` --
  same source types, same CASF exclusion, same generation-benchmark pocket
  exclusion, same seed, same held-out fraction;
* ProLIT's own pocket residues, extracted the same way with the same
  ``max_residues``, so both arms condition on the same residues of the same
  receptor and differ only in how those residues are written down.

What is not, and is reported rather than hidden: no rotation augmentation. ESM3
codes and ConfSeq tokens are both frame-invariant, so a rotated copy of a
complex is a duplicate document, not a second view of one.

Parallelism is over ``(tar shard, slice of pairs)`` rather than over tar shards
alone. There are only 35 tars and ~17M poses, so a shard per worker caps the
build at 35 cores and roughly ninety hours; slicing the pairs inside a shard
lets a run use a whole node. Each slice re-reads its tar, which is cheap
relative to encoding and keeps the workers independent.

Run (CPU, one node)::

    .venv/bin/python pipelines/corpora/tokenize_crossdocked_stapled.py \\
        --esm3-cache data/esm3_tokens_crossdocked \\
        --stapled-vocab data/stapled/confseq_vocab.json \\
        --num-partitions 8 --partition-index 0 --num-workers 160 \\
        --out-dir data/lm_tokens_stapled_cd/p0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from prolit.config import PocketExtractionConfig
from prolit.data.atom_tar_prep import iter_tar_poses
from prolit.data.esm3_tokens import Esm3TokenCache
from prolit.data.holdout import (
    casf_pdbs,
    crossdocked_pocket_split,
    sbdd_bench_pockets,
)
from prolit.data.token_io import SplitWriter
from prolit.data.work_budget import WorkBudget
from prolit.seeding import add_seed_argument, seed_from_args
from prolit.tokenizers.protein import precompute_pocket_atom_candidates
from prolit.tokenizers.stapled import NUM_SPECIAL, ConfSeqVocab, StapledVocab
from prolit.tokenizers.stapled_encoder import StapledEncoder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

#: The source types the published ProLIT CLM corpus was cut from.
DEFAULT_SOURCE_TYPES = ("cdonly", "it0", "it2_redocked")

#: Per-worker receptor precomputation cache. Small on purpose: a worker walks
#: one slice of one tar, whose pairs cluster on a handful of receptors, and 160
#: workers on a node each holding a large cache is how a build runs out of
#: memory rather than out of time.
_RECEPTOR_CACHE = 8

_G: dict[str, Any] = {}

#: pair_idx -> "train"/"val", inherited by fork rather than sent per task.
_SPLIT_MAP: dict[int, str] = {}


def _worker_init(cfg: dict) -> None:
    """One StapledEncoder per worker; the vocabulary and cache are read-only."""
    _G["enc"] = StapledEncoder(
        cache=Esm3TokenCache(Path(cfg["esm3_cache"])),
        confseq_repo=Path(cfg["confseq_repo"]),
        vocab=StapledVocab(confseq=ConfSeqVocab.load(Path(cfg["vocab"]))),
        pocket_cfg=PocketExtractionConfig(**cfg["pocket_cfg"]),
    )
    _G["cfg"] = cfg
    _G["get_receptor"] = lru_cache(maxsize=_RECEPTOR_CACHE)(_precompute_receptor)


def _precompute_receptor(rec_path: str) -> Any | None:  # noqa: ANN401
    try:
        return precompute_pocket_atom_candidates(Path(rec_path))
    except Exception:
        logger.exception("receptor %s", rec_path)
        return None


def _struct_id(rec_path: str) -> str:
    """``.../POCKET_0/2bq0_A_rec.pdb`` -> ``POCKET_0/2bq0_A_rec``.

    The ESM3 cache is keyed this way by ``data/esm3_manifests/crossdocked.jsonl``;
    a receptor filename alone is not unique across pockets.
    """
    p = Path(rec_path)
    return f"{p.parent.name}/{p.stem}"


def _heavy(mol: dict) -> np.ndarray:
    return np.array(
        [i for i, a in enumerate(mol["atoms"]) if a[0] != "H"], dtype=int
    )


def _process_task(task: tuple[int, int, int, dict]) -> tuple[dict, dict]:
    """Encode one (tar shard, slice) -> ``{split: [sequences]}`` plus a tally.

    The pocket is built once per *pair* from that pair's first pose, not once
    per pose: every pose of a pair sits in the same site, and the receptor parse
    plus pocket extraction dominates the per-pose cost otherwise.
    """
    shard_idx, slice_j, n_slices, cfg = task
    # NOT carried in the task: the pair -> split map has 1.55M entries and
    # there are hundreds of tasks. Sending it per task pickles ~150 MB that
    # many times over, and holding a private copy in each of 160 workers is
    # tens of gigabytes -- the same shape as the 1.195T vmem OOM the BioLiP
    # builder carries a comment about. It is set on the module before the pool
    # forks, so the children inherit one copy-on-write copy and pickle none.
    split_map: dict[int, str] = _SPLIT_MAP
    enc: StapledEncoder = _G["enc"]
    get_receptor = _G["get_receptor"]
    # One symmetric ligand must cost one pose, not the task it sits in. With
    # 17M poses over a few hundred tasks the question is not whether a build
    # meets one but how many; a decoy shard already spent eleven hours on this
    # exact failure and wrote nothing after its sixth.
    budget = WorkBudget(int(cfg["pose_timeout"]))

    out: dict[str, list[list[int]]] = {"train": [], "val": []}
    tally: dict[str, int] = {}
    pocket = None
    pocket_pair = -1

    for pair_idx, _pose_idx, mol, rec_path in iter_tar_poses(
        Path(cfg["repo_dir"]),
        Path(cfg["manifest"]),
        Path(cfg["receptors_dir"]),
        list(cfg["source_types"]),
        shard_idx,
        good_poses_only=False,
        min_only=False,
    ):
        if pair_idx % n_slices != slice_j:
            continue
        split = split_map.get(pair_idx)
        if split is None:
            continue
        tally["poses"] = tally.get("poses", 0) + 1
        hidx = _heavy(mol)
        if hidx.size == 0:
            tally["failed_empty"] = tally.get("failed_empty", 0) + 1
            continue
        budget.arm()
        try:
            if pair_idx != pocket_pair:
                pocket_pair = pair_idx
                precomp = get_receptor(rec_path)
                ref = np.array(
                    [(a[1], a[2], a[3]) for a in mol["atoms"] if a[0] != "H"],
                    dtype=np.float32,
                )
                pocket = (
                    None
                    if precomp is None
                    else enc.setup_pocket_precomputed(
                        _struct_id(rec_path), precomp, ref
                    )
                )
            if pocket is None:
                tally["failed_pocket"] = tally.get("failed_pocket", 0) + 1
                continue
            seq, why = enc.ligand_seq_with_reason(
                pocket, mol["atoms"], mol["bonds"], hidx
            )
        except TimeoutError:
            # The pocket may have been half-built when the clock fired, so the
            # next pose of this pair rebuilds it rather than trusting it.
            pocket_pair = -1
            tally["failed_timeout"] = tally.get("failed_timeout", 0) + 1
            continue
        except Exception:  # noqa: BLE001 -- one bad pose must not end the task
            tally["failed_error"] = tally.get("failed_error", 0) + 1
            continue
        finally:
            budget.disarm()
        if seq is None:
            tally[f"failed_{why}"] = tally.get(f"failed_{why}", 0) + 1
            continue
        out[split].append(seq)
    return out, tally


def _tasks(
    shards: list[int], n_slices: int, n_parts: int, part: int
) -> list[tuple[int, int]]:
    """(shard, slice) pairs for this partition, interleaved so parts are even."""
    all_tasks = [(s, j) for s in shards for j in range(n_slices)]
    return [t for i, t in enumerate(all_tasks) if i % n_parts == part]


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path("data/hub_cache/repo"))
    parser.add_argument(
        "--receptors-dir", type=Path, default=Path("data/hub_cache/receptors")
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--source-types", type=str, nargs="+", default=list(DEFAULT_SOURCE_TYPES)
    )
    parser.add_argument("--esm3-cache", type=Path, required=True)
    parser.add_argument(
        "--confseq-repo", type=Path, default=Path("third_party/ConfSeq")
    )
    parser.add_argument("--stapled-vocab", type=Path, required=True)
    parser.add_argument("--max-residues", type=int, default=50)
    parser.add_argument("--pocket-val-frac", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--casf-pdbs", type=Path, default=Path("data/casf2016_pdbs.txt")
    )
    parser.add_argument(
        "--exclude-pockets", type=Path, default=Path("data/sbdd_bench_pockets.txt")
    )
    parser.add_argument("--num-shards", type=int, default=35)
    parser.add_argument(
        "--slices-per-shard",
        type=int,
        default=16,
        help="pair-modulo slices each tar is cut into; total tasks = shards x this",
    )
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=10800,
        help="seconds one (shard, slice) task may take before it is abandoned. "
        "The last line of defence: --pose-timeout misses anything wedged "
        "inside C, and without this the whole run waits on it.",
    )
    parser.add_argument(
        "--pose-timeout",
        type=int,
        default=20,
        help="seconds one pose may take before it is abandoned (0 disables). "
        "See prolit.data.work_budget.WorkBudget for what it does and does not "
        "catch.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    add_seed_argument(parser)
    args = parser.parse_args()
    seed_from_args(args)

    manifest = args.manifest or (args.repo_dir / "manifest.parquet")
    split = crossdocked_pocket_split(
        manifest,
        args.source_types,
        args.pocket_val_frac,
        args.split_seed,
        casf_pdbs(args.casf_pdbs) if args.casf_pdbs.exists() else None,
        sbdd_bench_pockets(args.exclude_pockets)
        if args.exclude_pockets.exists()
        else None,
    )
    logger.info(
        "pocket split: %d pockets (%d val), %d pairs in the corpus",
        len(split.pocket_split),
        sum(v == "val" for v in split.pocket_split.values()),
        len(split.pair_to_pocket),
    )
    split_map = {
        pair: split.pocket_split[pocket]
        for pair, pocket in split.pair_to_pocket.items()
    }

    cfg = {
        "repo_dir": str(args.repo_dir),
        "manifest": str(manifest),
        "receptors_dir": str(args.receptors_dir),
        "source_types": list(args.source_types),
        "esm3_cache": str(args.esm3_cache),
        "confseq_repo": str(args.confseq_repo),
        "vocab": str(args.stapled_vocab),
        "pocket_cfg": asdict(PocketExtractionConfig(max_residues=args.max_residues)),
        "pose_timeout": args.pose_timeout,
    }
    tasks = _tasks(
        list(range(args.num_shards)),
        args.slices_per_shard,
        args.num_partitions,
        args.partition_index,
    )
    logger.info(
        "partition %d/%d: %d tasks over %d workers",
        args.partition_index,
        args.num_partitions,
        len(tasks),
        args.num_workers,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {s: SplitWriter(args.out_dir, s) for s in ("train", "val")}
    totals: dict[str, int] = {}
    vocab = StapledVocab(confseq=ConfSeqVocab.load(args.stapled_vocab))

    def _meta() -> None:
        """Written after every task, not only at the end.

        A build this long ends at its walltime as often as it ends by finishing,
        and a corpus whose meta.json was never written cannot be read back even
        though every token is on disk.
        """
        meta = {
            "vocab_size": vocab.vocab_size,
            "atom_codebook_size": vocab.vocab_size - NUM_SPECIAL,
            "atom_offset": NUM_SPECIAL,
            "all_atom": True,
            "source": "crossdocked",
            "source_types": list(args.source_types),
            "pretrain": {"num_rotations": 1},
            "stapled": {
                "protein_tokenizer": "esm3_structure_v0",
                "ligand_tokenizer": "confseq",
                "pose_tokens": vocab.n_pose_tokens,
                "pose_bits": vocab.pose_bits,
                "esm3_cache": str(args.esm3_cache),
                "confseq_vocab_path": str(args.stapled_vocab),
                "failed": {k: v for k, v in sorted(totals.items()) if k != "poses"},
                "poses_seen": totals.get("poses", 0),
            },
            "split": {
                "pockets": len(split.pocket_split),
                "val_pockets": sum(
                    v == "val" for v in split.pocket_split.values()
                ),
                "val_frac": args.pocket_val_frac,
                "seed": args.split_seed,
            },
            "partition": {
                "index": args.partition_index,
                "count": args.num_partitions,
                "tasks_done": _meta.done,  # type: ignore[attr-defined]
                "tasks_total": len(tasks),
            },
            "splits": {
                s: {
                    "num_docs": w.num_docs,
                    "num_tokens": w.num_tokens,
                    "max_len": w.max_len,
                }
                for s, w in writers.items()
            },
        }
        (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    _meta.done = 0  # type: ignore[attr-defined]

    global _SPLIT_MAP  # noqa: PLW0603 -- read by the forked children
    _SPLIT_MAP = split_map
    payload = [(s, j, args.slices_per_shard, cfg) for s, j in tasks]
    if args.num_workers > 0:
        import multiprocessing  # noqa: PLC0415

        # apply_async + get(timeout) rather than imap_unordered, because a
        # worker CAN still wedge: WorkBudget's signal handler runs between
        # bytecodes, so a call that stays inside C is not interrupted by it.
        # imap_unordered offers no escape from that -- it blocks on the wedged
        # task until the walltime and the whole run dies with it, which is
        # exactly how a decoy shard lost eleven hours. Here the task is
        # abandoned, its workers' siblings keep going, and the run ends
        # cleanly with its meta written.
        ctx = multiprocessing.get_context("fork")
        n_timeout = 0
        with ctx.Pool(
            args.num_workers, initializer=_worker_init, initargs=(cfg,)
        ) as pool:
            pending = [
                (t[0], t[1], pool.apply_async(_process_task, (t,))) for t in payload
            ]
            for shard_idx, slice_j, ar in pending:
                try:
                    out, tally = ar.get(timeout=args.task_timeout)
                except multiprocessing.TimeoutError:
                    n_timeout += 1
                    logger.warning(
                        "task shard=%d slice=%d exceeded %ds; abandoned",
                        shard_idx,
                        slice_j,
                        args.task_timeout,
                    )
                    _meta.done += 1  # type: ignore[attr-defined]
                    _meta()
                    continue
                _consume(out, tally, writers, totals)
                _meta.done += 1  # type: ignore[attr-defined]
                _meta()
        if n_timeout:
            logger.warning("%d of %d tasks abandoned", n_timeout, len(payload))
    else:
        _worker_init(cfg)
        for task in payload:
            out, tally = _process_task(task)
            _consume(out, tally, writers, totals)
            _meta.done += 1  # type: ignore[attr-defined]
            _meta()

    for w in writers.values():
        w.close()
    _meta()
    logger.info("tally: %s", dict(sorted(totals.items())))
    for s, w in writers.items():
        logger.info("%s: %d docs, %d tokens", s, w.num_docs, w.num_tokens)


def _consume(
    out: dict, tally: dict, writers: dict, totals: dict
) -> None:
    for split_name, seqs in out.items():
        if seqs:
            writers[split_name].write(seqs)
    for k, v in tally.items():
        totals[k] = totals.get(k, 0) + v


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
