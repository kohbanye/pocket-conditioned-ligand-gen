# Pre-registration: optimise the correlation, like the two strongest baselines

2026-09-15 ~11:20 JST. Written BEFORE the arm is implemented, trained, or
scored.

## Why this lever

Read out of the baselines' own source, not their papers:

* **OnionNet-2** (`third_party/OnionNet-2/retrain/train.py:12-27, 70, 105`)
  trains on `loss = alpha * (1 - PCC) + (1 - alpha) * RMSE` with `alpha`
  defaulting to **0.7**. PCC is the Pearson correlation across the batch.
* **GenScore** (`third_party/GenScore/GenScore/model/utils.py:222, 225`) adds
  `affi = corrcoef(y, labels)[1,0]` to its loss with its own weight.

Both of the models that beat us on CASF scoring power **optimise the
correlation directly**. We optimise per-complex smooth-L1 only, and then report
a correlation. Nothing in our objective asks the predictions to line up on a
common scale across complexes.

This costs nothing in inputs: no contact pair, no interaction edge, no
distance bin, no contact count. It is one term in the loss, no new parameters,
and the tokenizer and head are untouched.

It also bears on the measured deficit. The diagnosis found the target-level
spread compressed (slope of cluster-mean error against cluster-mean pK: ours
-0.190, GIGN -0.048). A correlation term is scale-free: it cannot be satisfied
by shrinking toward the mean, which is exactly what a squared-error objective
rewards when the signal is uncertain.

## The arm

| | treatment `e250_unionnc_corr` | control `e250_union_noconf` |
|---|---|---|
| corpus | `data/lm_tokens_e250aff_union_noconf` | the same |
| `--corr-weight` | **0.7** | 0.0 (the flag did not exist) |
| batching | default (random, 32 per batch) | the same |

**The control already exists and is already scored** -- the same 4 seeds (7, 11,
12, 13), the same driver, the same recipe (`--label-cap 13 --max-epochs 20
--early-stop-patience 4 --block-size 640`, backbone `mlm_e250lig3` e02, ligand
readout), dumps at f16 and f32. It is the control the concat round used. So this
round trains 4 heads and runs 8 inferences.

`alpha = 0.7` is **OnionNet-2's published default, copied rather than chosen**.
No sweep: picking a weight by CASF score would be selection on the test set.

**Disclosed in advance:** adding the term changes what `val/loss` means, so it
changes which epoch is selected. That is inherent to the lever (OnionNet-2
selects on the same mixed loss) and is not a confound to be removed -- but it
does mean the comparison is "this objective, selected on this objective"
against "that objective, selected on that objective".

## Predictions, fixed now

* **C1 (scoring).** f32 family-ensemble dR >= +0.010 AND treatment R above
  control on >= 3/4 paired seeds.
* **C2 (ranking guard).** f32 family-ensemble drho >= -0.010.
* **C3 (mechanism).** The shrinkage slope moves toward 0: treatment slope
  >= -0.12 (ours is -0.190; GIGN -0.048), measured exactly as in the
  diagnosis, after a least-squares calibration onto pK.

**Lever verdict:** works only if C1 and C2 hold. C3 is the mechanism check: if
C1 passes while C3 does not, the gain came from somewhere else and the
explanation in this note is wrong.

## What this round does not do

No headline. If the lever works, a headline needs its own pre-registration,
written before the ensemble is scored.

## Cost

4 head jobs (gpu_1, 2:00) ~= 1.6 pt + 8 inference jobs (gpu_1, 0:30) ~= 0.8 pt
~= 2.4 pt.
