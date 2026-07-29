"""Aggregate per-sample dumps into comparison tables + significance verdicts.

Every function is *method-agnostic*: it takes a ``dict[method_name -> frame]`` so
the exact same code produces the existing-method comparison (ours vs
GenScore/Vina/...) and the tokenizer ablation (joint_nocasf vs separate).
Significance is always computed against a chosen reference method and
Holm-corrected across the set of pairwise comparisons.
"""

from __future__ import annotations

import pandas as pd
from prolit_bench import stats

from ctbench.metrics import affinity as A
from ctbench.metrics import rescoring as R

# ----------------------------------------------------------------------------
# Affinity (CASF scoring & ranking power)
# ----------------------------------------------------------------------------


def affinity_metrics(
    preds: dict[str, pd.DataFrame],
    pred_col: str = A.PRED,
) -> pd.DataFrame:
    """Scoring R and ranking rho per method (rows = methods)."""
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
    """Significance of each method vs ``reference`` (Steiger + Wilcoxon)."""
    ref = preds[reference].dropna(subset=["logka", pred_col]).set_index("pdbid")
    ref_clusters = A.cluster_rho(preds[reference], pred_col).set_index("cluster")["rho"]
    rows, scoring_p, ranking_p = [], [], []
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
        rows.append(name)
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


# ----------------------------------------------------------------------------
# Pose rescoring (CASF docking power)
# ----------------------------------------------------------------------------


def rescoring_metrics(scored: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Docking power @2Å/@1Å and ranking rho per method (frames already oriented)."""
    rows = [
        {
            "method": name,
            "DP@2A": R.docking_power(df, 2.0),
            "DP@1A": R.docking_power(df, 1.0),
            "ranking_rho": R.ranking_rho(df),
            "n_targets": int(df["pdbid"].nunique()),
        }
        for name, df in scored.items()
    ]
    return pd.DataFrame(rows).set_index("method")


def rescoring_pairwise(
    scored: dict[str, pd.DataFrame],
    reference: str,
    cut: float = 2.0,
) -> pd.DataFrame:
    """Significance vs ``reference``: McNemar (docking), Wilcoxon (per-target rho)."""
    ref_succ = R.target_success(scored[reference], cut).set_index("pdbid")["success"]
    ref_rho = R.target_rho(scored[reference]).set_index("pdbid")["rho"]
    ref_dp = R.docking_power(scored[reference], cut)
    ref_rrho = R.ranking_rho(scored[reference])
    others = [m for m in scored if m != reference]
    dp_p, rho_p, d_dp, d_rho = [], [], [], []
    for name in others:
        cur_succ = R.target_success(scored[name], cut).set_index("pdbid")["success"]
        shared = ref_succ.index.intersection(cur_succ.index)
        mc = stats.mcnemar(
            cur_succ.loc[shared].to_numpy(),
            ref_succ.loc[shared].to_numpy(),
        )
        cur_rho = R.target_rho(scored[name]).set_index("pdbid")["rho"]
        common = ref_rho.index.intersection(cur_rho.index)
        wl = stats.wilcoxon_paired(
            cur_rho.loc[common].to_numpy(),
            ref_rho.loc[common].to_numpy(),
        )
        dp_p.append(mc.pvalue)
        rho_p.append(wl.pvalue)
        d_dp.append(R.docking_power(scored[name], cut) - ref_dp)
        d_rho.append(R.ranking_rho(scored[name]) - ref_rrho)
    return pd.DataFrame(
        {
            "vs_reference": reference,
            "d_DP": d_dp,
            "DP_mcnemar_p": dp_p,
            "DP_p_holm": stats.holm_correction(dp_p),
            "d_ranking_rho": d_rho,
            "ranking_p": rho_p,
            "ranking_p_holm": stats.holm_correction(rho_p),
        },
        index=pd.Index(others, name="method"),
    )


def orient_pose_dumps(
    dumps: dict[str, pd.DataFrame],
    raw_col: str = "head",
) -> dict[str, pd.DataFrame]:
    """Orient a set of raw per-pose dumps (higher = more native-like) for comparison."""
    return {name: R.orient(df, raw_col=raw_col) for name, df in dumps.items()}


# ----------------------------------------------------------------------------
# Generation (Vina + molecular quality vs SBDD baselines)
# ----------------------------------------------------------------------------

_GEN_COLUMNS = (
    "vina_score_mean",
    "vina_min_mean",
    "pb_valid_rate",
    "clash_free_rate",
    "qed_mean",
    "sa_mean",
    "div_scaffold_diversity",
)


def generation_table(
    per_model_rows: dict[str, pd.Series],
    columns: tuple[str, ...] = _GEN_COLUMNS,
) -> pd.DataFrame:
    """Assemble a generation comparison table from one per-model row per method."""
    return pd.DataFrame(
        {name: row[list(columns)] for name, row in per_model_rows.items()},
    ).T


def generation_pairwise(
    per_target: dict[str, pd.DataFrame],
    reference: str,
    metric: str = "vina_score_mean",
) -> pd.DataFrame:
    """Paired t-test of ``metric`` over shared targets, each method vs ``reference``."""
    ref = per_target[reference].set_index("target_id")[metric]
    others = [m for m in per_target if m != reference]
    pvals, diffs = [], []
    for name in others:
        cur = per_target[name].set_index("target_id")[metric]
        # Align on shared targets and drop pairs where either side is NaN (targets
        # with no dockable molecule) so the paired t-test isn't poisoned to NaN.
        paired = pd.concat({"ref": ref, "cur": cur}, axis=1).dropna()
        res = stats.paired_ttest(paired["cur"].to_numpy(), paired["ref"].to_numpy())
        pvals.append(res.pvalue)
        diffs.append(float((paired["cur"] - paired["ref"]).mean()))
    return pd.DataFrame(
        {
            "vs_reference": reference,
            f"d_{metric}": diffs,
            "ttest_p": pvals,
            "ttest_p_holm": stats.holm_correction(pvals),
        },
        index=pd.Index(others, name="method"),
    )
