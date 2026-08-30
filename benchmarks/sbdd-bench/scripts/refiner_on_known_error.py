"""Does the refiner recover the error the tokenizer makes?

The quantizer displaces a reference ligand by RMSD 0.350 A, and 54% of that is
a rigid whole-molecule shift. Vina's own local optimisation recovers nearly all
of it -- the round-tripped pose scores -5.60 as written and -6.49 after
``--local_only``. That -6.49 is above FLOWR's published -6.29, so whether the
ceiling actually binds depends entirely on how much of the rigid part our own
refinement recovers.

So test the refinement on an error we know the shape of, rather than on
generated poses where the error is unknown and confounded with molecule
quality. The molecule is a reference ligand throughout; only its coordinates
move.

Arms, and which of them are fair to report against a generative baseline:

- ``crystal``   the deposited pose. Upper bound, not achievable.
- ``roundtrip`` encode and decode. The tokenizer ceiling as it stands.
- ``refiner``   the flow-matching pose refiner. A learned model -- fair.
- ``rigid``     ``rigid_pocket_fit``, a numerical optimiser against a vdW
                objective. NOT a learned model. Reported as a diagnostic
                ceiling for local repair, not as a result.
- ``rigid+refiner`` what the deployed pipeline actually runs.

``--source`` picks the error being repaired:

- ``vq``   encode/decode the reference ligand. RMSD 0.35, mostly rigid.
- ``clm``  additionally replace each code with the language model's own
           teacher-forced argmax. RMSD 1.86, internal 1.52, translation 0.76 --
           the error the refiner actually meets at generation time, but with a
           ground truth, which generated molecules do not have.

``--refine-ckpt`` may be given more than once to compare refiners on the same
poses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

BENCH = Path(__file__).resolve().parent.parent
REPO = BENCH.parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from generate_ligands_3d import load_atom_norm_stats, load_atom_vqvae  # noqa: E402
from prolit.api import (  # noqa: E402
    ATOM_LAYOUT,
    LIGAND_ELEMENT_VOCAB,
    AtomLMVocab,
    LigandAtomDescriptor,
    ProteinAtomDescriptor,
    compute_canonical_frame,
    extract_pocket_atoms_from_candidates,
    fields_by_name,
    infer_bonds,
    load_pose_refiner,
    parse_sdf,
    precompute_pocket_atom_candidates,
    precompute_receptor_atom_features,
)
from prolit.chem.rigid_fit import rigid_pocket_fit, vdw_radii  # noqa: E402
from prolit.config import PocketExtractionConfig  # noqa: E402
from prolit.tokenizers.atom import spherical_to_cartesian_np  # noqa: E402

from sbdd_bench import datasets, docking, molio  # noqa: E402


def main() -> None:  # noqa: PLR0915
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vqvae-ckpt", required=True)
    p.add_argument("--refine-ckpt", required=True, action="append",
                   help="repeatable; each becomes its own arm")
    p.add_argument("--lm-ckpt", default=None,
                   help="required for --source clm")
    p.add_argument("--source", choices=("vq", "clm"), default="vq")
    p.add_argument("--norm-stats", required=True)
    p.add_argument("--codebook-size", type=int, default=8192)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--shard", default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq = load_atom_vqvae(a.vqvae_ckpt, a.codebook_size, dev)
    refiners = {}
    for i, ck in enumerate(a.refine_ckpt):
        # A run directory names its checkpoints ``<run>/checkpoints/<file>``, so
        # the run is two levels up -- but only when the middle level really is
        # ``checkpoints``. Pinned copies live side by side under one directory
        # and used to collapse to a single name, which is worse than an error:
        # the dict silently kept the last checkpoint loaded and reported it
        # under a label that looked like all five.
        path = Path(ck)
        name = (
            path.parent.parent.name
            if path.parent.name == "checkpoints"
            else path.stem
        ) or f"ref{i}"
        if name in refiners:
            msg = f"two --refine-ckpt resolve to the same arm name {name!r}"
            raise SystemExit(msg)
        refiners[name] = load_pose_refiner(ck, dev)
    lm = None
    if a.source == "clm":
        from generate_ligands_3d import load_atom_lm  # noqa: PLC0415

        if not a.lm_ckpt:
            msg = "--source clm needs --lm-ckpt"
            raise SystemExit(msg)
        lm = load_atom_lm(a.lm_ckpt, a.codebook_size, dev)
    ns = load_atom_norm_stats(a.norm_stats, dev)
    cfg = PocketExtractionConfig()
    ldesc, pdesc = LigandAtomDescriptor(), ProteinAtomDescriptor()
    vocab = AtomLMVocab(codebook_size=a.codebook_size)
    cf = fields_by_name(ATOM_LAYOUT)["coord"]
    cm, cs = ns["atom_mean"][cf.start : cf.end], ns["atom_std"][cf.start : cf.end]

    from prolit.model.pose_refiner import (  # noqa: PLC0415
        LIG_CHEM_HEADS,
        ligand_feats_from_heads,
        pocket_feats_from_descriptor,  # noqa: PLC0415
        refine_ligand_canonical,
    )

    targets = datasets.load_targets()[: a.limit]
    if a.shard:
        k, n = (int(v) for v in a.shard.split("/"))
        targets = targets[k::n]

    rows = []
    for t in targets:
        if not t.receptor_pdbqt:
            continue
        try:
            mols = parse_sdf(Path(t.ref_ligand_sdf))
            if not mols:
                continue
            mol = mols[0]
            heavy = np.array([(x[1], x[2], x[3]) for x in mol["atoms"]
                              if x[0] != "H"], dtype=np.float32)
            if len(heavy) < 5:
                continue
            pre = precompute_pocket_atom_candidates(Path(t.receptor_pdb))
            feats = precompute_receptor_atom_features(Path(t.receptor_pdb))
            pocket = extract_pocket_atoms_from_candidates(pre, heavy, cfg)
            if pocket is None or pocket.atom_coords.shape[0] == 0:
                continue
            frame = compute_canonical_frame(pocket.ca_coords.astype(np.float64))
            centroid, rot = frame
            bonds = mol.get("bonds") or infer_bonds(
                [x[0] for x in mol["atoms"]],
                np.array([(x[1], x[2], x[3]) for x in mol["atoms"]]))
            larr, _, meta = ldesc.compute(mol["atoms"], bonds, frame)
            order = meta["heavy_to_orig"]
            orig = np.array([(mol["atoms"][i][1], mol["atoms"][i][2],
                              mol["atoms"][i][3]) for i in order], dtype=np.float64)
            els = [mol["atoms"][i][0] for i in order]

            with torch.no_grad():
                x = (torch.from_numpy(larr).to(dev) - ns["atom_mean"]) / ns["atom_std"]
                codes = vq.encode(x)
                if lm is not None:
                    # Replace every ligand code with the LM's own teacher-forced
                    # argmax. The molecule is unchanged and the crystal pose is
                    # still the target, but the coordinates now carry exactly
                    # the error the refiner meets at generation time.
                    parr0, _ = pdesc.compute(pocket, feats, frame)
                    pt0 = ((torch.from_numpy(parr0).to(dev) - ns["atom_mean"])
                           / ns["atom_std"])
                    pcodes = vq.encode(pt0).detach().cpu().tolist()
                    lcodes = codes.detach().cpu().tolist()
                    seq = vocab.build_sequence(pcodes, lcodes)
                    logits = lm(
                        torch.tensor([seq], dtype=torch.long, device=dev)
                    ).logits[0]
                    start = 2 + len(pcodes) + 2
                    off = vocab.offset
                    pred = [
                        int(logits[start + j - 1][off : off + a.codebook_size]
                            .argmax().item())
                        for j in range(len(lcodes))
                    ]
                    codes = torch.tensor(pred, dtype=torch.long, device=dev)
                out = vq.decode_to_outputs(codes)
                c = (out["coord"] * cs + cm).detach().cpu().numpy()
                chem = {h: out[h].argmax(dim=-1).cpu().numpy() for h in LIG_CHEM_HEADS}
            canon = spherical_to_cartesian_np(c)

            prot_arr, _ = pdesc.compute(pocket, feats, frame)
            pkt_feat = pocket_feats_from_descriptor(prot_arr)
            pkt_canon = (pocket.atom_coords.astype(np.float64) - centroid) @ rot.T
            pkt_r = vdw_radii(list(pocket.atom_elements))
            lig_feat = ligand_feats_from_heads(chem, canon.shape[0])
            elems_r = [LIGAND_ELEMENT_VOCAB[i] if LIGAND_ELEMENT_VOCAB[i] != "OTHER"
                       else "X" for i in chem["element"]]
            lig_r = vdw_radii(elems_r)

            # Bind the per-target arrays as defaults: these helpers are only
            # ever called inside this iteration, and the linter is right that a
            # late-bound closure over a loop variable is a trap waiting to
            # happen if anyone stores one.
            def to_world(z, rot=rot, centroid=centroid):  # noqa: ANN001, ANN202
                return z @ rot + centroid

            def do_rigid(z, lig_r=lig_r, pkt_canon=pkt_canon,  # noqa: ANN001, ANN202
                         pkt_r=pkt_r):
                fit = rigid_pocket_fit(z.astype(np.float64), lig_r, pkt_canon, pkt_r)
                return fit.apply(z.astype(np.float64))

            def do_refine(z, model, elems_r=elems_r,  # noqa: ANN001, ANN202
                          lig_feat=lig_feat, pkt_canon=pkt_canon,
                          pkt_feat=pkt_feat):
                # The deployed path derives bonds from the pose it is about to
                # refine and passes them in; the refiner was trained with them,
                # so omitting them is out of distribution and the model moves
                # atoms it should leave alone -- measured, it turned a +0.2
                # improvement into a 6 kcal/mol loss.
                zf = np.asarray(z, dtype=np.float32)
                bb = np.asarray(infer_bonds(elems_r, zf),
                                dtype=np.int64).reshape(-1, 2)
                return refine_ligand_canonical(
                    model, zf, lig_feat, pkt_canon, pkt_feat,
                    bonds=bb, device=dev)

            arms = {
                "crystal": orig,
                "start": to_world(canon),
                "rigid": to_world(do_rigid(canon)),
            }
            for nm, model in refiners.items():
                arms[f"ref:{nm}"] = to_world(do_refine(canon, model))
                arms[f"rigid+ref:{nm}"] = to_world(
                    do_refine(do_rigid(canon), model))

            gens, key = [], {}
            for i, (nm, xyz) in enumerate(arms.items()):
                gens.append(molio.GenMol(idx=i, elements=list(els),
                                         coords=np.asarray(xyz), tag=nm))
                key[i] = nm
            sc = docking.dock_generated(gens, t.receptor_pdbqt, t.box,
                                        modes=("score", "min"), workers=2,
                                        exhaustiveness=8)
            row = {"tid": t.target_id, "n_atoms": len(els)}
            for r in sc:
                nm = key[r["idx"]]
                row[f"{nm}_score"] = r.get("vina_score")
                row[f"{nm}_min"] = r.get("vina_min")
                row[f"{nm}_rmsd"] = float(np.sqrt(
                    ((np.asarray(arms[nm]) - orig) ** 2).sum(1).mean()))
            rows.append(row)
            print(f"{t.target_id[:24]:24s} " + " ".join(
                f"{k}={row.get(k + '_score')}" for k in arms), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {t.target_id}: {exc!r}", flush=True)

    a.out.write_text(json.dumps(rows))
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
