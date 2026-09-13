"""Aggregate per-sample dumps into comparison tables + significance verdicts.

Every function is *method-agnostic*: it takes a ``dict[method_name -> frame]`` so
the exact same code produces the existing-method comparison (ours vs
RTMScore/GenScore/Vina/...) and the tokenizer ablation (joint_nocasf vs separate).

Affinity lived here too until it became its own benchmark. It shares an
architecture with pose rescoring and nothing else -- not a corpus, not a label,
not a backbone -- and one registry describing two models under one name is the
drift ``prolit_bench.variants`` exists to prevent. See ``benchmarks/affinity-bench``.
Significance is always computed against a chosen reference method and
Holm-corrected across the set of pairwise comparisons.
"""

from __future__ import annotations

import pandas as pd
from prolit_bench import stats

from pose_rescoring_bench.metrics import rescoring as R

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
