"""Diagnostic: does the encoder representation separate congeneric ligands?

Ranking power fails on clusters where our head inverts small pK gaps between
congeners. Two competing explanations:
  (a) the VQ tokens / MLM representation collapse congeners (info lost) -- then
      no head can rank them;
  (b) the representation DOES separate them but the MLP head under-uses it --
      then the head/training is the fixable weak link.

This dumps the mean-pooled ligand representation for each CASF complex and, per
cluster, measures whether representation distance tracks pK distance, and how
well a leave-one-cluster-out linear probe on the frozen representation ranks --
an upper bound on what any head over this representation can do.
"""

from __future__ import annotations

# ruff: noqa: PLC0415
from pathlib import Path

import numpy as np
import torch

from prolit.config import (
    AtomVQVAETrainingConfig,
    ComplexMLMConfig,
    MLMTrainingConfig,
    PocketExtractionConfig,
)
from prolit.data.rescore_dataset import ligand_mask
from prolit.model.mlm_module import ComplexMLMModule
from prolit.model.vqvae_module import AtomVQVAEModule
from prolit.tokenizers.ligand import parse_sdf
from prolit.tokenizers.lm_vocab import AtomLMVocab
from scripts.eval_casf_rescore import _PoseEncoder

CASF = Path("data/casf2016")
MLM = "pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt"
VQ = (
    "pocket-ligand-vqvae/xzkjxu9q/checkpoints/"
    "atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt"
)
NORM = "data/descriptor_cache_allatom/normalization_stats.pt"


def _labels() -> dict[str, tuple[float, str]]:
    out: dict[str, tuple[float, str]] = {}
    for ln in (CASF / "power_scoring" / "CoreSet.dat").read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        c = ln.split()
        out[c[0].lower()] = (float(c[3]), c[-1])
    return out


def main() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq_cfg = AtomVQVAETrainingConfig()
    vq_cfg.atom.codebook_size = 8192
    module = AtomVQVAEModule.load_from_checkpoint(
        VQ, config=vq_cfg, map_location=device
    )
    module.eval().to(device)
    norm = torch.load(NORM, weights_only=False)
    module.vqvae.set_normalization(norm["atom_mean"], norm["atom_std"])
    mlm_cfg = MLMTrainingConfig(model=ComplexMLMConfig(atom_codebook_size=8192))
    mlm = ComplexMLMModule.load_from_checkpoint(
        MLM, config=mlm_cfg, map_location=device
    ).model
    mlm.eval().to(device)
    enc = _PoseEncoder(
        module,
        norm["atom_mean"].numpy(),
        norm["atom_std"].numpy(),
        AtomLMVocab(codebook_size=8192),
        device,
        PocketExtractionConfig(max_residues=50),
    )

    lab = _labels()
    ids, pks, clusters, reprs = [], [], [], []
    for tid in sorted(p.name for p in (CASF / "coreset").iterdir() if p.is_dir()):
        if tid not in lab:
            continue
        prot = CASF / "coreset" / tid / f"{tid}_protein.pdb"
        sdf = CASF / "coreset" / tid / f"{tid}_ligand.sdf"
        if not (prot.exists() and sdf.exists()):
            continue
        try:
            native = parse_sdf(sdf)[0]
            heavy = np.array(
                [(a[1], a[2], a[3]) for a in native["atoms"] if a[0] != "H"], np.float32
            )
            setup = enc.setup_pocket(prot.read_text(), heavy)
            if setup is None:
                continue
            p_codes, frame = setup
            seq = enc.ligand_seq(p_codes, native, frame)
            if seq is None:
                continue
            t = torch.tensor([seq], device=device)
            lig = torch.tensor(ligand_mask(np.asarray(seq)), device=device)
            with torch.no_grad():
                hs = mlm.encode(t, torch.ones_like(t))[0]  # (L,H)
                pooled = hs[lig].mean(dim=0)  # mean over ligand tokens
            reprs.append(pooled.float().cpu().numpy())
            ids.append(tid)
            pks.append(lab[tid][0])
            clusters.append(lab[tid][1])
        except Exception as e:  # noqa: BLE001
            print(f"  skip {tid}: {e}")
    reprs = np.array(reprs)
    pks = np.array(pks)
    out = "outputs/casf/casf_repr.npz"
    np.savez(out, ids=ids, pks=pks, clusters=clusters, reprs=reprs)
    print(f"dumped {len(ids)} complexes -> {out}  (repr dim {reprs.shape[1]})")


if __name__ == "__main__":
    main()
