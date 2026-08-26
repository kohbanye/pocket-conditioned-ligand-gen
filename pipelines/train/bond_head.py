"""Train the bond head: decoded chemistry + geometry -> the bond graph.

The generation path reads bonds off the decoded coordinates, and at the error
the language model makes that recovers 31% of them (see
:mod:`prolit.model.bond_head` for the measurements and why no threshold fixes
it). This trains the replacement on exactly the input it will meet at
generation time: a pose-refine corpus holds the LM's own decoded coordinates
(``lig_x0``) beside the crystal molecule's true bonds.

Every corruption level is used, and the crystal pose is added as a level of its
own. A head fitted on the deployed error alone learns not to trust distance,
and then scores 0.789 on coordinates where plain distance perception scores
1.000 -- fine while the pose stays 2.5 A out, wrong the moment the refiner
improves it. Training across the range keeps it right in both regimes.

    .venv/bin/python pipelines/train/bond_head.py \
        --data-dir data/pose_refine_clm --run-name bond_clm --max-epochs 40
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from prolit.config import BondHeadTrainingConfig
from prolit.model.bond_head import (
    BondHead,
    _chem_columns,
    bond_capacity,
    bond_jaccard,
    pair_features,
)
from prolit.provenance import write_manifest
from prolit.seeding import add_seed_argument, rng_for, seed_from_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#: Bonds are ~7% of all atom pairs, so the loss is reweighted rather than the
#: data resampled -- every pair of a molecule has to be scored together anyway.
POS_WEIGHT = 8.0


def load_split(
    data_dir: Path, split: str, *, evaluation_only: bool = False
) -> list[tuple]:
    from prolit.model.pose_refiner import FEATURE_FIELDS  # noqa: PLC0415

    meta = json.loads((data_dir / "meta.json").read_text())
    n_c = meta["splits"][split]["num_complexes"]
    n_r = meta["splits"][split]["num_records"]
    width = len(FEATURE_FIELDS)
    comp = np.fromfile(data_dir / f"{split}.complexes", dtype=np.int64).reshape(-1, 3)
    comp = comp[:n_c]
    x0 = np.fromfile(data_dir / f"{split}.lig_x0", dtype=np.float32).reshape(-1, 3)
    feat = np.fromfile(data_dir / f"{split}.lig_feat", dtype=np.int16)
    feat = feat.reshape(-1, width)
    bonds = np.fromfile(data_dir / f"{split}.lig_bonds", dtype=np.int32).reshape(-1, 2)
    rec = np.fromfile(data_dir / f"{split}.records", dtype=np.int64)[:n_r]
    scale = np.fromfile(data_dir / f"{split}.record_scale", dtype=np.float32)[:n_r]
    off_a = np.concatenate([[0], np.cumsum(comp[:, 0])])
    off_b = np.concatenate([[0], np.cumsum(comp[:, 2])])
    cols = _chem_columns()

    x1 = np.fromfile(data_dir / f"{split}.lig_x1", dtype=np.float32).reshape(-1, 3)
    out, pos = [], 0
    min_atoms = 3
    seen_crystal: set[int] = set()

    def example(coords: np.ndarray, feats: np.ndarray, true: set) -> tuple:
        cont, cat, idx_i, idx_j = pair_features(coords, feats, cols)
        y = np.array(
            [
                1.0 if (int(a), int(b)) in true else 0.0
                for a, b in zip(idx_i, idx_j, strict=True)
            ],
            dtype=np.float32,
        )
        return (cont, cat, y, idx_i, idx_j, bond_capacity(feats, cols), true)

    for k, cid in enumerate(rec):
        n = int(comp[cid, 0])
        coords = np.asarray(x0[pos : pos + n], dtype=np.float32)
        pos += n
        if n < min_atoms:
            continue
        true = {
            tuple(sorted(t))
            for t in bonds[off_b[cid] : off_b[cid] + comp[cid, 2]].tolist()
        }
        if not true:
            continue
        feats = feat[off_a[cid] : off_a[cid] + n].astype(np.int64)
        # The scale-0 record is the decode itself and is what generation hands
        # the head; the graded ones and the crystal pose widen the range of
        # input quality it stays calibrated over.
        if evaluation_only and scale[k] != 0.0:
            continue
        out.append(example(coords, feats, true))
        if not evaluation_only and cid not in seen_crystal:
            seen_crystal.add(int(cid))
            crystal = np.asarray(x1[off_a[cid] : off_a[cid] + n], dtype=np.float32)
            out.append(example(crystal, feats, true))
    return out


@torch.no_grad()
def evaluate(
    head: nn.Module, data: list[tuple], device: torch.device
) -> tuple[float, float]:
    head.eval()
    scores = []
    for cont, cat, _y, idx_i, idx_j, cap, true in data:
        prob = (
            torch.sigmoid(
                head(
                    torch.from_numpy(cont).to(device),
                    torch.from_numpy(cat).to(device),
                )
            )
            .cpu()
            .numpy()
        )
        degree = np.zeros(len(cap), dtype=np.float32)
        got = []
        for k in np.argsort(-prob):
            if prob[k] < 0.5:  # noqa: PLR2004
                break
            a, b = int(idx_i[k]), int(idx_j[k])
            if degree[a] < cap[a] and degree[b] < cap[b]:
                got.append((a, b))
                degree[a] += 1
                degree[b] += 1
        scores.append(bond_jaccard(got, list(true)))
    arr = np.array(scores)
    return float(np.median(arr)), float(np.mean(arr == 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("pocket-ligand-bond"))
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-molecules", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=256)
    add_seed_argument(parser, default=0)
    args = parser.parse_args()
    seed_from_args(args)

    config = BondHeadTrainingConfig(
        data_dir=str(args.data_dir),
        max_epochs=args.max_epochs,
        batch_molecules=args.batch_molecules,
        learning_rate=args.lr,
        embedding_dim=args.dim,
        hidden_dim=args.hidden,
    )
    config.seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_split(args.data_dir, "train")
    # Validation is scored on the decode alone: that is the input generation
    # hands the head, and averaging it with easier levels would hide a
    # regression there.
    val = load_split(args.data_dir, "val", evaluation_only=True)
    logger.info("train %d molecules, val %d", len(train), len(val))

    head = BondHead(dim=config.embedding_dim, hidden=config.hidden_dim).to(device)
    opt = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    pos_weight = torch.tensor(POS_WEIGHT, device=device)
    rng = rng_for(config.seed, "bond-head-shuffle")

    out_dir = args.out_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir, seed=config.seed)

    best = -1.0
    for epoch in range(config.max_epochs):
        head.train()
        order = rng.permutation(len(train))
        total = 0.0
        for start in range(0, len(order), config.batch_molecules):
            batch = order[start : start + config.batch_molecules]
            def stack(column: int, batch: np.ndarray = batch) -> torch.Tensor:
                return torch.from_numpy(
                    np.concatenate([train[i][column] for i in batch])
                ).to(device)

            cont, cat, y = stack(0), stack(1), stack(2)
            loss = nn.functional.binary_cross_entropy_with_logits(
                head(cont, cat), y, pos_weight=pos_weight
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(batch)
        jaccard, exact = evaluate(head, val, device)
        logger.info(
            "epoch %d  loss %.4f  val Jaccard %.3f  exact %.3f",
            epoch,
            total / max(len(order), 1),
            jaccard,
            exact,
        )
        if jaccard > best:
            best = jaccard
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "config": asdict(config),
                    "jaccard": jaccard,
                    "epoch": epoch,
                },
                out_dir / "bond_head.pt",
            )
    logger.info("best val Jaccard %.3f -> %s", best, out_dir / "bond_head.pt")


if __name__ == "__main__":
    main()
