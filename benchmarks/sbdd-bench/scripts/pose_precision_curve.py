"""How accurate does a pose have to be, in angstroms, to score like a good one?

Every other number here is an outcome -- a Vina score, a validity rate. This
one is a conversion: it says what a given per-atom placement error is *worth*
in kcal/mol, and so what accuracy a generator has to reach before its poses
stop being the thing that limits it.

Isotropic Gaussian noise of a known sigma is added to each *reference* ligand
-- a crystal pose of a real molecule -- and all three Vina numbers are measured.
The molecule is held perfect by construction, so the only thing moving is
placement accuracy, and the resulting curve reads directly as
"error of X angstroms costs Y kcal/mol".

Note the noise is unbiased and independent per atom, which is the friendliest
possible shape for a given magnitude: it distorts local geometry but does not
translate or rotate the molecule as a whole. Real generator error is not this
kind, so the curve is a *lower* bound on what a given error costs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path("/gs/bs/tga-ohuelab/sakano/git/pocket-conditioned-ligand-gen/"
            ".claude/worktrees/shape-complementarity")
sys.path.insert(0, str(REPO / "benchmarks/sbdd-bench"))
sys.path.insert(0, str(REPO / "src"))
from prolit.seeding import rng_for  # noqa: E402

from sbdd_bench import datasets, docking, molio  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--shard", default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    targets = datasets.load_targets()[: a.limit]
    if a.shard:
        k, n = (int(v) for v in a.shard.split("/"))
        targets = targets[k::n]
    rows = []
    for t in targets:
        if not t.receptor_pdbqt:
            continue
        try:
            supp = Chem.SDMolSupplier(str(t.ref_ligand_sdf), removeHs=True)
            ref = next((m for m in supp if m is not None), None)
            if ref is None or ref.GetNumAtoms() < 5:
                continue
            base = ref.GetConformer().GetPositions()
            els = [at.GetSymbol() for at in ref.GetAtoms()]
            gens, key = [], {}
            i = 0
            for s in a.sigmas:
                reps = 1 if s == 0.0 else a.repeats
                for r in range(reps):
                    rng = rng_for(a.seed, f"noise-{t.target_id}-{s}-{r}")
                    xyz = base + (rng.normal(0.0, s, base.shape) if s > 0 else 0.0)
                    gens.append(molio.GenMol(idx=i, elements=list(els),
                                             coords=np.asarray(xyz)))
                    key[i] = (s, r, float(np.linalg.norm(xyz - base, axis=1).mean()))
                    i += 1
            out = docking.dock_generated(gens, t.receptor_pdbqt, t.box,
                                         modes=("score", "min", "dock"),
                                         workers=a.workers, exhaustiveness=8)
            for row in out:
                s, r, disp = key[row["idx"]]
                rows.append({"tid": t.target_id, "sigma": s, "rep": r,
                             "mean_disp": disp, "n_atoms": len(els),
                             "vina_score": row["vina_score"],
                             "vina_min": row["vina_min"],
                             "vina_dock": row["vina_dock"]})
            print(f"{t.target_id[:30]:30s} {len(out)} rows", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {t.target_id}: {exc!r}", flush=True)
    a.out.write_text(json.dumps(rows))
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
