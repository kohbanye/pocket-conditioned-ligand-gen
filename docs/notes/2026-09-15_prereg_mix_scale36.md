# Pre-registration: the three-recipe mix at 36 models

2026-09-15 ~22:05 JST. Written BEFORE the 12 new heads exist and before any CASF
number for them exists.

## Why, and the disclosure that comes with it

The pre-registered headline round scored the three confirmed recipes at 8 seeds
each (24 models): f32 R 0.8244 / rho 0.7228, which beats GenScore's single
released checkpoint on R and loses on rho
(`docs/results/2026-09-15_three_recipe_headline.md`).

An exploratory pass afterwards, built as the size-matched *control* for a
different question, put the same three recipes at 32 models and scored
**R 0.8230 / rho 0.7333** -- above the single checkpoint on **both**. That number
was seen after the fact and is not a claim. This round asks whether it survives
a pre-registered test.

**Stated in advance, so a pass cannot be oversold:**

* rho moves about +-0.01 between configurations of this size in our own
  measurements, and the exploratory margin over the bar was 0.005. This is close
  to a coin flip, not a robust effect.
* **The 24 existing models are reused**; only 12 seeds are new. This is a
  scale-up of an already-scored mix, not a fresh-seed replication.
* 36 models of ours against 1 of theirs is not equal cost, and against
  GenScore's own 3-checkpoint ensemble (R 0.8387 / rho 0.7211) we expect to stay
  behind on R.

## Arms

Four fresh seeds per recipe, recipes unchanged:

| family | corpus | recipe flag | new seeds |
|---|---|---|---|
| `e250_unionnc_corr_v4` | `union_noconf` | `--corr-weight 0.7` | 98-101 |
| `e250_unionr8_ens_v4` | `union_r8` | `--rotations-per-complex 8` | 102-105 |
| `e250_unionheavy_v4` | `union_heavy` | (none; heavy pocket at inference) | 106-109 |

Common: `--label-cap 13 --max-epochs 20 --early-stop-patience 4 --block-size 640`,
backbone `mlm_e250lig3` e02, ligand readout.

**The judged model is the z-sum of all 36** (the 24 v3 models plus these 12):
no seed selection, no weighting, no blending, no frame mixing.

## Claims (f32 basis; f16 reported beside)

* **W1.** R(36) > 0.8127 **AND** rho(36) > 0.7281 -- beats the single released
  GenScore checkpoint on both metrics.
* **W2.** R(36) > R(GenScore 3 checkpoints) = 0.8387 AND rho(36) > 0.7211.
* **W3.** R(36) > 0.83.

Reported beside: twin-clean columns (19-complex set, n=266), the 24-model mix,
the 36-model mix's components, and our mean single seed.

A W1 pass is "a 36-model ensemble beats one released GenScore checkpoint on both
metrics", and must be written that way, with the model counts and the
coin-flip caveat above.

## Cost

12 heads (gpu_1, 1:00) ~= 2.4 pt + 24 inferences (gpu_1, 0:30) ~= 2.4 pt
~= **4.8 pt**, taking the ledger to about 160 of the 200 pt cap.
