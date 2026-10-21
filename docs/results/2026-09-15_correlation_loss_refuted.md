# Optimising the correlation directly changes nothing on CASF

2026-09-15 11:55 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_correlation_loss.md` before the arm was
implemented or trained; applied by
`experiments/affinity_loop_20260914/correlation_verdict.py`, written before the
heads finished. Raw output in `correlation_verdict_output.txt`. All 12 jobs
verified first (`qacct` exit 0, clean logs, 285/285 non-NaN).

## Why it was worth trying

Read out of the baselines' source, not their papers:

* **OnionNet-2** (`retrain/train.py:12-27, 70, 105`): `loss = alpha * (1 - PCC)
  + (1 - alpha) * RMSE`, `alpha` default **0.7**.
* **GenScore** (`GenScore/model/utils.py:222, 225`): a `corrcoef` term in the
  loss.

Both models that beat us on scoring power optimise the correlation directly; we
optimised per-complex smooth-L1 and then reported a correlation. The weight was
copied from OnionNet-2's default, not swept.

## Verdict: DOES NOT WORK (C1, C2, C3 all False)

| f32 family ensemble | R | between | within | rho | shrink slope |
|---|---:|---:|---:|---:|---:|
| treatment `e250_unionnc_corr` (w=0.7) | 0.8110 | 0.3742 | 0.4368 | 0.6965 | -0.1812 |
| control `e250_union_noconf` (w=0) | 0.8105 | 0.3778 | 0.4328 | 0.7193 | -0.1737 |

* **C1** dR **+0.0004**, paired 2/4 seeds (-0.0049, +0.0121, +0.0037, -0.0012).
  The bar was +0.010 on >= 3/4.
* **C2** drho **-0.0228**.
* **C3** slope -0.174 -> -0.181, i.e. unchanged (-0.0075).
* f16 agrees: dR +0.0006, 2/4 seeds.

## What the training logs say, and why it matters

`train/corr` fell to **0.006-0.020**, i.e. a within-batch Pearson of
**0.98-0.99**. The term was not inert: the model drove the training correlation
almost to its ceiling. CASF's correlation did not move at all.

**Maximising the correlation on the training distribution does not transfer.**
The objective was already effectively aligned with the metric through the
regression term; making the alignment explicit added nothing. This also closes
the reading that our loss function was the thing separating us from OnionNet-2
and GenScore -- their advantage is elsewhere.

## Where this leaves the diagnosis

Two rounds in a row have now shown that the between-target deficit cannot be
reached from the loss:

* the group-mean term made the shrinkage **worse** (slope -0.248 -> -0.338, R
  -0.041),
* the correlation term left it **unchanged** (-0.174 -> -0.181, R +0.0004).

The remaining reading is the one the refutations point at: our per-target
estimate is **noisier** than GIGN's, and no reweighting of the objective fixes
an estimate that is not there. That is a representation problem, not a
calibration or objective problem.

Sixteen levers are now refuted or neutral.
