"""ProLIT against the ESM3 x ConfSeq baseline, paired, at matched scope.

Separate from ``significance.py`` because this comparison pairs two DIFFERENT
modalities: ESM3 reconstructs backbone atoms only, so the baseline's ``complex``
rows are backbone-scope already, and ProLIT has to be read from its
``complex_backbone`` rows rather than its all-atom ``complex`` ones. Putting
them in one column would compare two different quantities -- most real
protein-ligand contacts are to side chains, which the baseline never sees.

The pose-budget arms are one rate curve, so they are corrected together: Holm
runs over every (arm, metric) pair at once rather than per arm, which is the
difference between five independent tests and one family of twenty.

Run:
    .venv/bin/python scripts/stapled_significance.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "common" / "src"))
from prolit_bench.stats import (  # noqa: E402
    bootstrap_ci_paired_diff,
    holm_correction,
    wilcoxon_paired,
)

R = ROOT / "benchmarks/recon-bench/results"
stap = pd.read_parquet(R / "posebusters_stapled.parquet")
own = pd.read_parquet(R / "posebusters_ownbb.parquet")

def s(df, model, modality, metric):
    sub = df[(df.model == model) & (df.modality == modality) & df.ok]
    sub = sub.drop_duplicates(subset="sample_id")
    return sub.set_index("sample_id")[metric].dropna()

METRICS = [("lddt_pli", True), ("contact_f1", True),
           ("clash_lig_atom_frac", False), ("iface_lig_rmsd", False)]
ARMS = ["stapled_pose0", "stapled_pose13", "stapled_pose26",
        "stapled_pose39", "stapled_oracle"]
POSE_BITS = {"stapled_pose0": 0, "stapled_pose13": 13, "stapled_pose26": 26,
             "stapled_pose39": 39, "stapled_oracle": float("inf")}

rows, pvals = [], []
for arm in ARMS:
    for metric, higher in METRICS:
        a = s(own, "own_allatom.joint_e250_lig3", "complex_backbone", metric)
        b = s(stap, arm, "complex", metric)
        idx = a.index.intersection(b.index)
        x, y = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
        diff = x - y
        ci = bootstrap_ci_paired_diff(x, y, seed=0)
        w = wilcoxon_paired(x, y)
        nz = diff != 0
        win = float(((diff > 0) if higher else (diff < 0))[nz].mean()) if nz.any() else float("nan")
        rows.append({"arm": arm, "pose_bits": POSE_BITS[arm], "metric": metric,
                     "prolit": x.mean(), "stapled": y.mean(), "diff": diff.mean(),
                     "ci_lo": ci.low, "ci_hi": ci.high, "win%": 100 * win,
                     "ties": int((~nz).sum()), "n": len(idx)})
        pvals.append(w.pvalue)
adj = holm_correction(pvals)
for r, p in zip(rows, adj, strict=True):
    r["holm_p"] = p
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for metric, _ in METRICS:
    print(f"\n### {metric}")
    print(df[df.metric == metric][["arm","pose_bits","prolit","stapled","diff","ci_lo","ci_hi","win%","ties","holm_p","n"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
