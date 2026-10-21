"""Paired comparisons between ProLIT and every other model in a results table.

Reads any parquet ``run_reconstruction.py`` wrote and prints, per rival, the
mean of each metric, the paired bootstrap CI of the difference, a tie-aware win
rate, and a Holm-corrected Wilcoxon p-value.

Two things this deliberately does NOT let you misread:

* **Ties.** ``pb_valid`` and ``smiles_match`` are binary, and the two arms agree
  on most complexes. A win rate counting ties as losses reads 10% where ours
  actually wins 3 of every 4 complexes the arms disagree on, so the rate here is
  over discordant pairs only and the tie count is printed next to it.
* **Scope.** ESM3 and FoldToken reconstruct the full chain and are scored both
  on it and on our pocket residues. Only the pocket rows are comparable to a
  pocket-native tokenizer, so those are the ones taken.

What it cannot fix is that the pocket rows still compare ~30 reconstructed
residues against ~390, and that the baselines record no bits -- read the
``bits_per_atom`` column in the summary before reading any RMSD.

Which set the table came from matters for how the n reads. CASP16's 303
complexes are four proteins, so a paired test there treats 227 near-replicates
of autotaxin as independent; PoseBusters' 428 are 426 distinct proteins, and the
n is the n.

Run:
    .venv/bin/python scripts/significance.py results/posebusters.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# prolit_bench is a path dependency of this bench, but the script is also run
# straight from a checkout, so derive the location instead of hard-coding a site.
BENCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH / "common" / "src"))

from prolit_bench.stats import (  # noqa: E402
    bootstrap_ci_paired_diff,
    holm_correction,
    wilcoxon_paired,
)

OURS = "own_allatom.joint_e250_lig3"

# (metric, modality, higher_is_better) -- the columns the paper reports.
METRICS = [
    ("lddt_pli", "complex", True),
    ("contact_f1", "complex", True),
    ("clash_lig_atom_frac", "complex", False),
    ("iface_lig_rmsd", "complex", False),
    ("lddt", "protein_allatom", True),
    ("kabsch_rmsd", "protein_allatom", False),
    ("tm_score", "protein_backbone", True),
    ("kabsch_rmsd", "protein_backbone", False),
    ("kabsch_rmsd", "ligand", False),
    ("pb_valid", "ligand", True),
    ("smiles_match", "ligand", True),
    ("bond_mae", "ligand", False),
]


def series(df, model, metric, modality):
    """One metric for one model, indexed by sample_id.

    ``eval_scope`` matters: ESM3 and FoldToken reconstruct the full chain and
    are scored both on it and on our pocket residues. Only the pocket rows are
    comparable to a pocket-native tokenizer, so prefer those and fall back to
    whatever scope the model does report.
    """
    sub = df[(df.model == model) & (df.modality == modality) & df.ok]
    if sub.empty or metric not in sub:
        return None
    for scope in ("pocket", "native", "full"):
        s = sub[sub.eval_scope == scope]
        if not s.empty:
            return s.set_index("sample_id")[metric].dropna()
    return None


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/posebusters.parquet")
    df = pd.read_parquet(path)
    print("# Reconstruction benchmark — paired comparisons\n")
    print(f"source: `{path}`  rows: {len(df)}  ok: {int(df.ok.sum())}\n")

    n_fail = int((~df.ok).sum())
    if n_fail:
        print(f"**{n_fail} failed reconstructions**:\n")
        print(df[~df.ok].groupby("model").size().to_string(), "\n")

    print("## Per-model summary (mean over complexes)\n")
    summ = (
        df[df.ok]
        .groupby(["model", "modality", "eval_scope"])
        .agg(
            n=("sample_id", "nunique"),
            kabsch=("kabsch_rmsd", "mean"),
            lddt=("lddt", "mean"),
            tm=("tm_score", "mean"),
            lddt_pli=("lddt_pli", "mean"),
            contact_f1=("contact_f1", "mean"),
            pb_valid=("pb_valid", "mean"),
            bits_per_atom=("bits_per_atom", "mean"),
        )
    )
    print(summ.to_string(float_format=lambda v: f"{v:.3f}"), "\n")

    rivals = sorted({m for m in df.model.unique() if m != OURS})
    for rival in rivals:
        rows, pvals = [], []
        for metric, modality, higher in METRICS:
            a = series(df, OURS, metric, modality)
            b = series(df, rival, metric, modality)
            if a is None or b is None:
                continue
            common = a.index.intersection(b.index)
            if len(common) < 10:
                continue
            x, y = a.loc[common].to_numpy(), b.loc[common].to_numpy()
            ok = ~(np.isnan(x) | np.isnan(y))
            x, y = x[ok], y[ok]
            if len(x) < 10 or np.allclose(x, y):
                continue
            t = wilcoxon_paired(x, y)
            ci = bootstrap_ci_paired_diff(x, y)
            # Win rate among DISCORDANT pairs. Counting ties as losses makes a
            # binary metric like pb_valid look like a 10% win rate when both
            # arms agree on 86% of complexes and ours wins 3 of every 4 they
            # disagree on -- which is the opposite reading.
            better = (x > y) if higher else (x < y)
            worse = (x < y) if higher else (x > y)
            n_disc = int(better.sum() + worse.sum())
            wins = float(better.sum() / n_disc) if n_disc else float("nan")
            rows.append(
                {
                    "metric": f"{metric} ({modality})",
                    "ours": float(np.mean(x)),
                    "rival": float(np.mean(y)),
                    "diff": float(np.mean(x) - np.mean(y)),
                    "ci": f"[{ci.low:+.3f}, {ci.high:+.3f}]",
                    "win%": 100 * wins,
                    "ties": len(x) - n_disc,
                    "n": len(x),
                    "p": t.pvalue,
                }
            )
            pvals.append(t.pvalue)
        if not rows:
            continue
        for row, p in zip(rows, holm_correction(pvals), strict=True):
            row["holm_p"] = p
        print(f"## ProLIT (joint_e250_lig3) vs {rival}\n")
        out = pd.DataFrame(rows)
        print(
            out.to_string(
                index=False,
                float_format=lambda v: f"{v:.4f}" if abs(v) < 1000 else f"{v:.3e}",
            ),
            "\n",
        )


if __name__ == "__main__":
    main()
