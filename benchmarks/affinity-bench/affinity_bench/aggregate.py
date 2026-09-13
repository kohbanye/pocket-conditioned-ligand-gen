"""Per-complex dumps -> a metrics table and a significance verdict.

Method-agnostic by construction: every function takes a
``dict[method_name -> frame]``, so the same code produces the comparison
against existing methods and the comparison between our own arms. The project's
question is never "is this point estimate higher" -- the differences at stake
are a few hundredths on 285 complexes -- so a table is only ever reported next
to a p-value against a named reference, Holm-corrected across the set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from prolit_bench import stats

from affinity_bench import metrics as A


def affinity_metrics(
    preds: dict[str, pd.DataFrame],
    pred_col: str = A.PRED,
) -> pd.DataFrame:
    """Scoring R and ranking rho per method, one row each."""
    rows = [
        {
            "method": name,
            "scoring_R": A.scoring_r(df, pred_col),
            "ranking_rho": A.ranking_rho(df, pred_col),
            "n": int(df.dropna(subset=["logka", pred_col]).shape[0]),
        }
        for name, df in preds.items()
    ]
    return pd.DataFrame(rows).set_index("method")


def affinity_pairwise(
    preds: dict[str, pd.DataFrame],
    reference: str,
    pred_col: str = A.PRED,
) -> pd.DataFrame:
    """Significance of every method against ``reference``.

    Scoring R is compared with the Williams/Steiger test for two dependent
    correlations: both share the measured-pK variable, so the ordinary test for
    two independent correlations does not apply and would overstate the
    evidence. Ranking rho is a paired per-cluster quantity, so Wilcoxon.
    """
    ref = preds[reference].dropna(subset=["logka", pred_col]).set_index("pdbid")
    ref_clusters = A.cluster_rho(preds[reference], pred_col).set_index("cluster")["rho"]
    scoring_p, ranking_p = [], []
    others = [m for m in preds if m != reference]
    for name in others:
        cur = preds[name].dropna(subset=["logka", pred_col]).set_index("pdbid")
        shared = ref.index.intersection(cur.index)
        sr = stats.compare_scoring_r(
            ref.loc[shared, "logka"].to_numpy(),
            cur.loc[shared, pred_col].to_numpy(),
            ref.loc[shared, pred_col].to_numpy(),
        )
        cur_clusters = A.cluster_rho(preds[name], pred_col).set_index("cluster")["rho"]
        common = ref_clusters.index.intersection(cur_clusters.index)
        rr = stats.wilcoxon_paired(
            cur_clusters.loc[common].to_numpy(),
            ref_clusters.loc[common].to_numpy(),
        )
        scoring_p.append(sr.pvalue)
        ranking_p.append(rr.pvalue)
    return pd.DataFrame(
        {
            "vs_reference": reference,
            "d_scoring_R": [
                A.scoring_r(preds[m], pred_col)
                - A.scoring_r(preds[reference], pred_col)
                for m in others
            ],
            "scoring_p": scoring_p,
            "scoring_p_holm": stats.holm_correction(scoring_p),
            "d_ranking_rho": [
                A.ranking_rho(preds[m], pred_col)
                - A.ranking_rho(preds[reference], pred_col)
                for m in others
            ],
            "ranking_p": ranking_p,
            "ranking_p_holm": stats.holm_correction(ranking_p),
        },
        index=pd.Index(others, name="method"),
    )


def beats_reference(
    preds: dict[str, pd.DataFrame],
    method: str,
    reference: str = "GenScore",
    pred_col: str = A.PRED,
) -> pd.Series:
    """The goal, stated as data: does ``method`` beat ``reference`` on both?

    Reported alongside the table so the target cannot drift to whichever metric
    happens to have won. Both deltas must be positive; the p-values say whether
    a positive delta is worth anything.
    """
    metrics = affinity_metrics(preds, pred_col)
    sig = affinity_pairwise(preds, reference=reference, pred_col=pred_col)
    return pd.Series(
        {
            "scoring_R": metrics.loc[method, "scoring_R"],
            "ranking_rho": metrics.loc[method, "ranking_rho"],
            "d_scoring_R": sig.loc[method, "d_scoring_R"],
            "d_ranking_rho": sig.loc[method, "d_ranking_rho"],
            "scoring_p": sig.loc[method, "scoring_p"],
            "ranking_p": sig.loc[method, "ranking_p"],
            "beats_both": bool(
                sig.loc[method, "d_scoring_R"] > 0
                and sig.loc[method, "d_ranking_rho"] > 0
            ),
        },
    )


def seed_spread(
    frames: dict[int, pd.DataFrame],
    pred_col: str = A.PRED,
) -> pd.Series:
    """Mean and sd of both metrics over the training seeds of ONE arm.

    Reported instead of a single run because retraining this head with nothing
    but a different seed moves scoring R by ~0.05 and ranking rho by ~0.04 --
    larger than every architectural difference measured so far. A one-run number
    is a draw from that distribution, not a property of the arm.
    """
    rs = [A.scoring_r(f, pred_col) for f in frames.values()]
    rhos = [A.ranking_rho(f, pred_col) for f in frames.values()]
    return pd.Series(
        {
            "scoring_R": float(np.mean(rs)),
            "scoring_R_sd": float(np.std(rs, ddof=1)) if len(rs) > 1 else float("nan"),
            "ranking_rho": float(np.mean(rhos)),
            "ranking_rho_sd": (
                float(np.std(rhos, ddof=1)) if len(rhos) > 1 else float("nan")
            ),
            "n_seeds": len(frames),
        },
    )


def paired_by_seed(
    arms: dict[str, dict[int, pd.DataFrame]],
    reference: str,
    pred_col: str = A.PRED,
) -> pd.DataFrame:
    """Compare arms seed by seed, against ``reference``, on the SHARED seeds.

    Two arms trained at the same seeds are paired: the seed's effect is common
    to both and cancels in the difference, so three runs can resolve a gap that
    would need dozens of unpaired ones. Comparing an arm's best run against
    another's is the mistake this exists to prevent.

    ``n_better`` is reported next to the p-value because with three seeds a
    paired t-test has almost no power -- "3 of 3 seeds moved the same way" is
    the more honest summary, and a split like 2/3 says the effect is not there.
    """
    ref = arms[reference]
    rows = []
    for name, frames in arms.items():
        if name == reference:
            continue
        seeds = sorted(set(frames) & set(ref))
        if not seeds:
            continue
        d_r = np.array(
            [
                A.scoring_r(frames[s], pred_col) - A.scoring_r(ref[s], pred_col)
                for s in seeds
            ]
        )
        d_rho = np.array(
            [
                A.ranking_rho(frames[s], pred_col) - A.ranking_rho(ref[s], pred_col)
                for s in seeds
            ]
        )
        row = {
            "vs_reference": reference,
            "n_seeds": len(seeds),
            "d_scoring_R": float(d_r.mean()),
            "d_scoring_R_sd": float(d_r.std(ddof=1)) if d_r.size > 1 else float("nan"),
            "scoring_n_better": int((d_r > 0).sum()),
            "d_ranking_rho": float(d_rho.mean()),
            "d_ranking_rho_sd": (
                float(d_rho.std(ddof=1)) if d_rho.size > 1 else float("nan")
            ),
            "ranking_n_better": int((d_rho > 0).sum()),
        }
        if d_r.size > 1:
            row["scoring_p"] = stats.paired_ttest(d_r, np.zeros_like(d_r)).pvalue
            row["ranking_p"] = stats.paired_ttest(d_rho, np.zeros_like(d_rho)).pvalue
        rows.append((name, row))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [r for _, r in rows], index=pd.Index([n for n, _ in rows], name="method")
    )
