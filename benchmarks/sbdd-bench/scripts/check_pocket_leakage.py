"""ベンチマークのポケットが、LM の訓練コーパスに入っているか。

トークン化には run.json が残らないので、コーパス側から確かめる。
各文書の先頭は <bos><p> の直後からポケットのコードが並ぶので、
その先頭 24 個の署名を全文書について集め、ベンチマーク 100 ポケットの
署名と照合する。
"""
import sys
from pathlib import Path

import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
# Derived from this file, not written in: a checkout that moves must still run.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "benchmarks/sbdd-bench"))
import torch  # noqa: E402
from generate_ligands_3d import load_atom_norm_stats, load_atom_vqvae  # noqa: E402
from prolit.api import (  # noqa: E402
    ProteinAtomDescriptor,
    compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    parse_sdf,
    precompute_pocket_atom_candidates,
    precompute_receptor_atom_features,
)
from prolit.config import PocketExtractionConfig  # noqa: E402

from sbdd_bench import datasets  # noqa: E402

R = Path("/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vq = load_atom_vqvae(str(R/"pocket-ligand-vqvae/vq_e250_lig3/checkpoints/atomvqvae-epoch=244-val/atom_coord=0.1022.ckpt"), 8192, dev)
ns = load_atom_norm_stats(str(R/"data/descriptor_cache_allatom/normalization_stats.pt"), dev)
cfg = PocketExtractionConfig()
pdesc = ProteinAtomDescriptor()
K = 24
want = {}
for t in datasets.load_targets():
    try:
        mols = parse_sdf(Path(t.ref_ligand_sdf))
        heavy = np.array([(x[1],x[2],x[3]) for x in mols[0]["atoms"] if x[0]!="H"],
                         dtype=np.float32)
        pre = precompute_pocket_atom_candidates(Path(t.receptor_pdb))
        feats = precompute_receptor_atom_features(Path(t.receptor_pdb))
        pocket = extract_pocket_atoms_from_candidates(pre, heavy, cfg)
        if pocket is None or pocket.atom_coords.shape[0] < K:
            continue
        frame = compute_canonical_frame(pocket.ca_coords.astype(np.float64))
        parr, _ = pdesc.compute(pocket, feats, frame)
        with torch.no_grad():
            pt = (torch.from_numpy(parr).to(dev) - ns["atom_mean"]) / ns["atom_std"]
            codes = vq.encode(pt).detach().cpu().numpy()[:K] + 7   # + NUM_SPECIAL
        want[t.target_id] = tuple(int(x) for x in codes)
    except Exception:
        pass
print(f"署名を作れたベンチマークポケット {len(want)}", flush=True)

D = R/"data/lm_tokens_e250lig3_full"
toks = np.memmap(D/"train.bin", dtype=np.uint16, mode="r")
lens = np.fromfile(D/"train.len", dtype=np.uint16).astype(np.int64)
offs = np.concatenate([[0], np.cumsum(lens)])[:-1]
print(f"訓練文書 {len(lens)}", flush=True)
# 各文書の先頭 2 個は <bos><p> なので、ポケットは offset+2 から
# Hash each document's pocket prefix rather than materialising 16.5M tuples:
# a set of tuples that size is tens of GB, and the whole point is a membership
# test.
targets = {}
for tid, sig in want.items():
    targets.setdefault(hash(sig), []).append((tid, sig))
hit = set()
CH = 500_000
ar = np.arange(2, 2 + K)
for st in range(0, len(lens), CH):
    en = min(st + CH, len(lens))
    sig = np.stack([np.asarray(toks[o + 2 : o + 2 + K]) for o in offs[st:en]])
    for row in sig:
        t = tuple(int(x) for x in row)
        for tid, s2 in targets.get(hash(t), ()):
            if s2 == t:
                hit.add(tid)
    print(f"  {en}/{len(lens)} 走査  ヒット {len(hit)}", flush=True)
hit = sorted(hit)
print(f"\n=== 訓練コーパスに現れたベンチマークポケット: {len(hit)}/{len(want)} ===")
for h in hit[:15]:
    print("  ", h)
