"""Score Boltz-2's CASF-2016 affinity predictions and compare to the field.

Boltz-2 is run in native mode (protein sequence + ligand SMILES -> predicted
structure + affinity), so unlike our head / GenScore / Vina it never sees the
crystal pose -- it is a reference column, not a same-input comparison.

``affinity_pred_value`` is log10(IC50 in uM); pK = 6 - value. We correlate the
predicted pK with the experimental logKa (power_scoring/CoreSet.dat) and report
scoring power (Pearson R) and ranking power (mean within-cluster Spearman),
alongside GenScore, our best ensemble, and Vina, with paired significance tests.

Run (CPU, after the array job finishes)::

    uv run python scripts/eval_boltz_casf.py
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

CASF = Path("data/casf2016")
PRED = Path("outputs/boltz_casf/predict")
GEN = Path(
    os.environ.get(
        "PROLIT_BASELINES_DIR", "../baselines"
    )
    + "/casf_work/scoring_power_genscore.csv"
)
OURS = [
    "affinity_power_lf",
    "affinity_all_attn",
    "affinity_kdki_mean",
    "affinity_kdki_attn",
    "affinity_kdki_meanmax",
]


def _labels() -> dict[str, tuple[float, str]]:
    """pdb -> (logKa, cluster) from the scoring-power CoreSet."""
    out: dict[str, tuple[float, str]] = {}
    with (CASF / "power_scoring" / "CoreSet.dat").open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            c = line.split()
            out[c[0].lower()] = (float(c[3]), c[-1])
    return out


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _boltz_pk() -> dict[str, float]:
    """pdb -> predicted pK (= 6 - affinity_pred_value)."""
    out: dict[str, float] = {}
    for j in PRED.glob("boltz_results_*/predictions/*/affinity_*.json"):
        pdb = j.stem.replace("affinity_", "").lower()
        try:
            with j.open() as f:
                y = json.load(f)["affinity_pred_value"]
        except (KeyError, json.JSONDecodeError):
            continue
        out[pdb] = 6.0 - float(y)
    return out


def _ours_ensemble() -> dict[str, float]:
    """Fixed z-sum ensemble of the leak-free heads (no test selection)."""
    d: dict[str, list[float]] = {}
    for f in OURS:
        for r in _read_csv(Path(f"outputs/casf/{f}.csv")):
            d.setdefault(r["pdbid"], []).append(float(r["head"]))
    # z-normalise each head across complexes, then sum
    keys = sorted(d)
    arr = np.array([d[k] for k in keys])  # (N, nheads)
    z = (arr - arr.mean(0)) / arr.std(0)
    s = z.sum(1)
    return dict(zip(keys, s.tolist(), strict=True))


def _genscore() -> dict[str, float]:
    return {r["pdbid"].lower(): float(r["score"]) for r in _read_csv(GEN)}


def _vina() -> dict[str, float]:
    p = Path("outputs/casf/vina_scoring.csv")
    if not p.exists():
        return {}
    return {
        r["pdbid"].lower(): float(r["vina_score"])
        for r in _read_csv(p)
        if r.get("vina_score")
    }


def _power(
    pred: dict[str, float], lab: dict[str, tuple[float, str]], common: list[str]
) -> tuple[float, float, int]:
    p = np.array([pred[t] for t in common])
    y = np.array([lab[t][0] for t in common])
    r = float(np.corrcoef(p, y)[0, 1])
    by = defaultdict(list)
    for t in common:
        by[lab[t][1]].append((pred[t], lab[t][0]))
    sps = [
        np.corrcoef(
            np.argsort(np.argsort([a[0] for a in v])),
            np.argsort(np.argsort([a[1] for a in v])),
        )[0, 1]
        for v in by.values()
        if len(v) >= 3 and len({a[0] for a in v}) > 1  # noqa: PLR2004
    ]
    return r, float(np.nanmean(sps)), len(common)


def main() -> None:
    lab = _labels()
    methods = {
        "Boltz-2": _boltz_pk(),
        "OUR ensemble": _ours_ensemble(),
        "GenScore": _genscore(),
        "Vina": _vina(),
    }
    print(f"CASF-2016 scoring/ranking power (labels: {len(lab)} complexes)\n")
    print(f"{'method':16s} {'n':>4s} {'scoring R':>10s} {'ranking rho':>12s}")
    print("-" * 46)
    for name, pred in methods.items():
        common = sorted(set(pred) & set(lab))
        if not common:
            print(f"{name:16s} {'--- no predictions yet ---':>30s}")
            continue
        r, rho, n = _power(pred, lab, common)
        print(f"{name:16s} {n:4d} {r:10.3f} {rho:12.3f}")

    # Boltz completeness
    nb = len(methods["Boltz-2"])
    print(f"\nBoltz-2 predictions present: {nb}/285")
    if nb < 285:  # noqa: PLR2004
        print("  (partial -- rerun after the array job finishes for the full table)")

    # sanity: pK conversion example
    if methods["Boltz-2"]:
        ex = next(iter(methods["Boltz-2"]))
        print(
            f"\nexample {ex}: Boltz pK={methods['Boltz-2'][ex]:.2f} "
            f"vs logKa={lab.get(ex, (float('nan'),))[0]:.2f}"
        )


if __name__ == "__main__":
    main()
