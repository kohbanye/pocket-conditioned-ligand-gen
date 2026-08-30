"""生成ポーズを、論文自身の rescorer で選び直す。

ずれの 79% はばらつきで、生成物の中には重心 0.47 A のものがある。
だが幾何 (衝突・PB・ポケット重心) では見分けられなかった。
見分けるのが ProLIT-MLM + rescoring head の役目 -- pose rescoring 用に
学習してある。

Vina もドッキングも参照リガンドの座標も使わない。使うのはポケットと
生成分子だけで、どちらも生成側が既に持っているもの。FLOWR も
--max_sample_iter で自分の判定によりサンプルを引き直すので、対等。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path("/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/"
            ".claude/worktrees/shape-complementarity")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "benchmarks/sbdd-bench"))
sys.path.insert(0, str(REPO / "benchmarks/pose-rescoring-bench"))

from generate_ligands_3d import load_atom_norm_stats, load_atom_vqvae  # noqa: E402
from prolit.api import (  # noqa: E402
    AtomLMVocab,
    PoseEncoder,
    ProteinAtomDescriptor,
    compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    ligand_mask,
    load_masked_lm,
    load_scoring_head,
    parse_sdf,
    precompute_pocket_atom_candidates,
    precompute_receptor_atom_features,
)
from prolit.config import PocketExtractionConfig  # noqa: E402

from sbdd_bench import datasets, molio  # noqa: E402


def main() -> None:  # noqa: PLR0915
    p = argparse.ArgumentParser()
    p.add_argument("--vqvae-ckpt", required=True)
    p.add_argument("--mlm-ckpt", required=True)
    p.add_argument("--head-ckpt", required=True)
    p.add_argument("--norm-stats", required=True)
    p.add_argument("--tree", default="benchmarks/sbdd-bench/outputs/gen100_arom_post/own")
    p.add_argument("--results", default="benchmarks/sbdd-bench/results_gen100_arom_post_s*")
    p.add_argument("--codebook-size", type=int, default=8192)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--shard", default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq = load_atom_vqvae(a.vqvae_ckpt, a.codebook_size, dev)
    mlm, mask_id = load_masked_lm(a.mlm_ckpt, a.codebook_size, dev)
    head = load_scoring_head(a.head_ckpt, a.codebook_size, dev, pooling="mean")
    ns = load_atom_norm_stats(a.norm_stats, dev)
    vocab = AtomLMVocab(codebook_size=a.codebook_size)
    cfg = PocketExtractionConfig()
    pdesc = ProteinAtomDescriptor()
    enc = PoseEncoder(
        vq,
        ns["atom_mean"].detach().cpu().numpy(),
        ns["atom_std"].detach().cpu().numpy(),
        vocab, dev, cfg,
    )

    ts = {x.target_id: x for x in datasets.load_targets()[: a.limit]}
    tree = Path(a.tree)
    tids = sorted({p.parent.name for p in tree.glob("*/generated.sdf")} & set(ts))
    if a.shard:
        k, n = (int(v) for v in a.shard.split("/"))
        tids = tids[k::n]

    rows = []
    for tid in tids:
        tg = ts[tid]
        try:
            sdf = tree / tid / "generated.sdf"
            gens = [g for g in molio.load_generated(sdf)
                    if g.tag != "ref" and g.mol is not None]
            # parse_sdf gives the dict shape the tokenizer expects; building it
            # by hand from an RDKit mol drops the bond types it needs.
            parsed = parse_sdf(sdf)
            if len(gens) < 5 or len(parsed) < len(gens):
                continue
            by_idx = {g.idx: g for g in gens}
            heavy = np.array([(x[1], x[2], x[3]) for x in parsed[0]["atoms"]
                              if x[0] != "H"], dtype=np.float32)
            pre = precompute_pocket_atom_candidates(Path(tg.receptor_pdb))
            feats = precompute_receptor_atom_features(Path(tg.receptor_pdb))
            pocket = extract_pocket_atoms_from_candidates(pre, heavy, cfg)
            if pocket is None or pocket.atom_coords.shape[0] == 0:
                continue
            frame = compute_canonical_frame(pocket.ca_coords.astype(np.float64))
            parr, _ = pdesc.compute(pocket, feats, frame)
            with torch.no_grad():
                pt = ((torch.from_numpy(parr).to(dev) - ns["atom_mean"])
                      / ns["atom_std"])
                pcodes = vq.encode(pt).detach().cpu().tolist()
            for pos, mol in enumerate(parsed):
                if pos not in by_idx:
                    continue
                g = by_idx[pos]
                seq = enc.ligand_seq(pcodes, mol, frame)
                if seq is None:
                    continue
                ids = torch.tensor([seq], device=dev)
                batch = {"input_ids": ids,
                         "attention_mask": torch.ones_like(ids),
                         "ligand_mask": torch.tensor(
                             ligand_mask(np.asarray(seq)), device=dev).unsqueeze(0)}
                with torch.no_grad():
                    s = -float(head(batch).item())
                rows.append({"target_id": tid, "idx": g.idx, "head": s})
            print(f"{tid[:26]:26s} {len(gens)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {tid}: {exc!r}", flush=True)
    a.out.write_text(json.dumps(rows))
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
