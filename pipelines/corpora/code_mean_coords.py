"""Mean decoded coordinate for each codebook entry.

The decoder is contextual -- a code's coordinate depends on the sequence it
sits in -- so no exact code-to-point map exists. But a *mean* is well defined
and is enough to compute a document's ligand centroid from its tokens alone,
which is what an auxiliary centroid-regression target needs. Building it here
means the LM corpus does not have to be rebuilt to carry coordinates.

Measured on reference ligands, averaging this table over a molecule's codes
reproduces the decoded centroid to well under the 2.1 A error the language
model currently makes on placement, which is the only accuracy that matters
for the target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from generate_ligands_3d import load_atom_norm_stats, load_atom_vqvae  # noqa: E402

from prolit.api import ATOM_LAYOUT, fields_by_name  # noqa: E402
from prolit.seeding import add_seed_argument, rng_for, seed_from_args  # noqa: E402
from prolit.tokenizers.atom import spherical_to_cartesian_np  # noqa: E402

#: A document with fewer ligand codes than this is a fragment, not a molecule,
#: and its decode would be dominated by the missing context.
MIN_LIGAND_CODES = 3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vqvae-ckpt", required=True)
    p.add_argument("--norm-stats", required=True)
    p.add_argument("--token-dir", type=Path, required=True)
    p.add_argument("--codebook-size", type=int, default=8192)
    p.add_argument("--docs", type=int, default=40000)
    p.add_argument("--out", type=Path, required=True)
    add_seed_argument(p)
    a = p.parse_args()
    seed_from_args(a)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq = load_atom_vqvae(a.vqvae_ckpt, a.codebook_size, dev)
    ns = load_atom_norm_stats(a.norm_stats, dev)
    cf = fields_by_name(ATOM_LAYOUT)["coord"]
    cm, cs = ns["atom_mean"][cf.start : cf.end], ns["atom_std"][cf.start : cf.end]

    from prolit.tokenizers.lm_vocab import (  # noqa: PLC0415
        L_CLOSE_ID,
        L_OPEN_ID,
        NUM_SPECIAL,
    )

    toks = np.memmap(a.token_dir / "train.bin", dtype=np.uint16, mode="r")
    lens = np.fromfile(a.token_dir / "train.len", dtype=np.uint16).astype(np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)])[:-1]
    rng = rng_for(a.seed, "code-mean-coords")
    pick = rng.choice(len(lens), size=min(a.docs, len(lens)), replace=False)

    total = np.zeros((a.codebook_size, 3), dtype=np.float64)
    count = np.zeros(a.codebook_size, dtype=np.int64)
    for n, i in enumerate(pick):
        doc = np.asarray(toks[offs[i] : offs[i] + int(lens[i])])
        lo = np.flatnonzero(doc == L_OPEN_ID)
        hi = np.flatnonzero(doc == L_CLOSE_ID)
        if lo.size == 0 or hi.size == 0 or hi[-1] <= lo[0] + 1:
            continue
        codes = doc[int(lo[0]) + 1 : int(hi[-1])].astype(np.int64) - NUM_SPECIAL
        codes = codes[(codes >= 0) & (codes < a.codebook_size)]
        if len(codes) < MIN_LIGAND_CODES:
            continue
        with torch.no_grad():
            out = vq.decode_to_outputs(
                torch.tensor(codes, dtype=torch.long, device=dev)
            )
            c = (out["coord"] * cs + cm).detach().cpu().numpy()
        xyz = spherical_to_cartesian_np(c)
        np.add.at(total, codes, xyz)
        np.add.at(count, codes, 1)
        if (n + 1) % 5000 == 0:
            seen = int((count > 0).sum())
            print(f"  {n + 1}/{len(pick)} docs, {seen}/{a.codebook_size} codes",
                  flush=True)

    seen = count > 0
    table = np.zeros((a.codebook_size, 3), dtype=np.float32)
    table[seen] = (total[seen] / count[seen, None]).astype(np.float32)
    torch.save({"table": torch.from_numpy(table),
                "count": torch.from_numpy(count)}, a.out)
    print(f"wrote {a.out}: {int(seen.sum())}/{a.codebook_size} codes observed")


if __name__ == "__main__":
    main()
