"""Does deciding the codes all at once beat deciding them left to right?

The pose error is the language model's code choice: on the same 100 reference
molecules the quantizer costs 1.28 kcal/mol and the LM's argmax costs 3.81 more,
landing atoms 1.90 A off. That error is *front-loaded* -- the first atom is off
by 3.19 A and later ones by ~1.5 -- and the reason is visible in the model's own
uncertainty. Measured as the spatial spread of its predictive distribution:

    conditioning                                       spread
    pocket only (what the first atom gets)              5.30 A
    pocket + the atoms already emitted (autoregressive) 1.40 A
    pocket + every other atom (bidirectional)           0.65 A

An autoregressive decoder must commit to the anchor while it is still in the
5.30 A regime, and everything downstream inherits that. A bidirectional model
scoring the same token with the rest of the molecule visible is more than twice
as sharp and gets the code exactly right 75% of the time against the causal
model's 48%.

So this asks whether iterative decoding with the complex MLM actually converts
that sharpness into a better pose, on the metric that matters -- the decoded
coordinates -- against the causal model on the same documents:

``cold``  every ligand code starts masked; each round commits the most
          confident positions. Out of distribution at the first step (the MLM
          was trained at 15% masking), which is exactly what this measures.
``warm``  start from the causal model's own argmax and let the MLM revise:
          re-mask the least confident fraction, re-predict, repeat. In
          distribution, and the natural "CLM proposes, MLM revises" pipeline.

The reference for every arm is the decode of the *true* codes, so the number is
comparable across arms and isolates code choice from quantizer error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

BENCH = Path(__file__).resolve().parent.parent
REPO = BENCH.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from generate_ligands_3d import (  # noqa: E402
    load_atom_lm,
    load_atom_norm_stats,
    load_atom_vqvae,
)
from prolit.api import ATOM_LAYOUT, fields_by_name  # noqa: E402
from prolit.model.mlm_module import ProLITMLMModule  # noqa: E402
from prolit.seeding import add_seed_argument, rng_for, seed_from_args  # noqa: E402
from prolit.tokenizers.geometry import spherical_to_cartesian_np  # noqa: E402
from prolit.tokenizers.lm_vocab import (  # noqa: E402
    L_CLOSE_ID,
    L_OPEN_ID,
    NUM_SPECIAL,
)

#: A document shorter than this is a fragment, not a molecule.
MIN_LIGAND_CODES = 6


def _decode_xyz(vq: object, codes: np.ndarray, cmean, cstd, device) -> np.ndarray:
    with torch.no_grad():
        out = vq.decode_to_outputs(
            torch.tensor(codes, dtype=torch.long, device=device)
        )
        coord = (out["coord"] * cstd + cmean).cpu().numpy()
    return spherical_to_cartesian_np(coord)


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


@torch.no_grad()
def _mlm_probs(mlm, ids: np.ndarray, device, n_codes: int) -> torch.Tensor:
    t = torch.tensor([ids], dtype=torch.long, device=device)
    out = mlm.model(input_ids=t, attention_mask=torch.ones_like(t))
    logits = out.logits if hasattr(out, "logits") else out
    return torch.softmax(logits[0, :, NUM_SPECIAL : NUM_SPECIAL + n_codes].float(), -1)


def _cold(mlm, doc, lo, hi, mask_id, device, n_codes, rounds):
    ids = doc.copy()
    ids[lo:hi] = mask_id
    todo = np.ones(hi - lo, dtype=bool)
    for r in range(rounds):
        p = _mlm_probs(mlm, ids, device, n_codes)[lo:hi]
        conf, best = p.max(-1)
        conf = conf.cpu().numpy()
        best = best.cpu().numpy()
        # Commit an equal share each round, most confident first.
        k = max(1, int(np.ceil(todo.sum() / (rounds - r))))
        order = np.argsort(-np.where(todo, conf, -np.inf))[:k]
        ids[lo + order] = best[order] + NUM_SPECIAL
        todo[order] = False
        if not todo.any():
            break
    return ids[lo:hi] - NUM_SPECIAL


def _warm(mlm, doc, lo, hi, start, mask_id, device, n_codes, rounds, frac):
    ids = doc.copy()
    ids[lo:hi] = start + NUM_SPECIAL
    n = hi - lo
    for _ in range(rounds):
        p = _mlm_probs(mlm, ids, device, n_codes)[lo:hi]
        cur = torch.tensor(ids[lo:hi] - NUM_SPECIAL, device=device)
        held = p.gather(1, cur[:, None]).squeeze(1).cpu().numpy()
        k = max(1, int(round(frac * n)))
        weak = np.argsort(held)[:k]
        ids[lo + weak] = mask_id
        p = _mlm_probs(mlm, ids, device, n_codes)[lo:hi]
        ids[lo + weak] = p[weak].argmax(-1).cpu().numpy() + NUM_SPECIAL
    return ids[lo:hi] - NUM_SPECIAL


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", type=Path, required=True)
    ap.add_argument("--vqvae-ckpt", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--lm-ckpt", required=True)
    ap.add_argument("--mlm-ckpt", required=True)
    ap.add_argument("--codebook-size", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--warm-frac", type=float, default=0.25)
    ap.add_argument("--out", type=Path, required=True)
    add_seed_argument(ap, default=0)
    a = ap.parse_args()
    seed_from_args(a)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq = load_atom_vqvae(a.vqvae_ckpt, a.codebook_size, dev)
    lm = load_atom_lm(a.lm_ckpt, a.codebook_size, dev)
    mlm = ProLITMLMModule.load_from_checkpoint(a.mlm_ckpt, map_location=dev)
    mlm = mlm.eval().to(dev)
    mask_id = mlm.config.model.mask_token_id
    ns = load_atom_norm_stats(a.norm_stats, dev)
    f = fields_by_name(ATOM_LAYOUT)["coord"]
    cmean, cstd = ns["atom_mean"][f.start : f.end], ns["atom_std"][f.start : f.end]

    toks = np.memmap(a.token_dir / "train.bin", dtype=np.uint16, mode="r")
    lens = np.fromfile(a.token_dir / "train.len", dtype=np.uint16).astype(np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)])[:-1]
    rng = rng_for(a.seed, "iterative-decode-probe")

    rows, n = [], 0
    for i in rng.permutation(len(lens)):
        doc = np.asarray(toks[offs[i] : offs[i] + int(lens[i])]).astype(np.int64)
        o = np.flatnonzero(doc == L_OPEN_ID)
        c = np.flatnonzero(doc == L_CLOSE_ID)
        if o.size == 0 or c.size == 0 or c[-1] <= o[0] + 1:
            continue
        lo, hi = int(o[0]) + 1, int(c[-1])
        true = doc[lo:hi] - NUM_SPECIAL
        if len(true) < MIN_LIGAND_CODES or true.min() < 0 or true.max() >= a.codebook_size:
            continue

        with torch.no_grad():
            logits = lm(torch.tensor([doc], dtype=torch.long, device=dev)).logits[0]
        causal = (
            logits[lo - 1 : hi - 1, NUM_SPECIAL : NUM_SPECIAL + a.codebook_size]
            .argmax(-1)
            .cpu()
            .numpy()
        )
        cold = _cold(mlm, doc, lo, hi, mask_id, dev, a.codebook_size, a.rounds)
        warm = _warm(
            mlm, doc, lo, hi, causal, mask_id, dev, a.codebook_size,
            a.rounds, a.warm_frac,
        )

        ref = _decode_xyz(vq, true, cmean, cstd, dev)
        row = {"n_atoms": int(len(true))}
        for tag, codes in (("causal", causal), ("cold", cold), ("warm", warm)):
            row[f"{tag}_rmsd"] = _rmsd(_decode_xyz(vq, codes, cmean, cstd, dev), ref)
            row[f"{tag}_exact"] = float((codes == true).mean())
        rows.append(row)
        n += 1
        if n >= a.limit:
            break

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows))
    print(f"=== コード選択のやり方と復号ポーズの誤差 (n={n}) ===")
    print(f"{'方式':28s} {'RMSD 中央':>10s} {'平均':>8s} {'コード一致':>10s}")
    for tag, label in (
        ("causal", "自己回帰 (現行)"),
        ("cold", "反復・全マスクから"),
        ("warm", "反復・自己回帰を修正"),
    ):
        r = np.array([x[f"{tag}_rmsd"] for x in rows])
        e = np.array([x[f"{tag}_exact"] for x in rows])
        print(f"{label:28s} {np.median(r):10.3f} {r.mean():8.3f} {e.mean():10.3f}")
    for tag in ("cold", "warm"):
        d = np.array([x[f"{tag}_rmsd"] - x["causal_rmsd"] for x in rows])
        print(f"  {tag:6s} 対 causal: 差の中央 {np.median(d):+.3f}  "
              f"良い分子 {(d < 0).mean():.1%}")


if __name__ == "__main__":
    main()
