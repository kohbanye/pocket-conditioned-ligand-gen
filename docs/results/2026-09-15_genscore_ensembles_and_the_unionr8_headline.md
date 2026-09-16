# The GenScore number to beat depends on how many GenScore models you run

2026-09-15 ~16:30 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_unionr8_headline.md` (written before the arm's
dumps existed); applied by
`experiments/affinity_loop_20260914/unionr8_headline_verdict.py`.

## The new fact

GenScore ships three released checkpoints of the configuration this bench used
(`GatedGCN_ft_1.0_{1,2,3}.pth`). They were never scored here -- every GenScore
number in this project is checkpoint 1. Scored now with the same CLI, the same
inputs (`casf_work` pockets/ligands) and the same two-complex repair, checkpoint
1 reproduces the existing dump to **max |diff| 3.2e-05** (R 0.8127), so the
three are directly comparable:

| GenScore | R | rho | R clean (n=266) | rho clean |
|---|---|---|---|---|
| 1 released checkpoint | 0.8127 | 0.7281 | 0.8184 | 0.7589 |
| **3 checkpoints, z-sum** | **0.8387** | 0.7211 | 0.8421 | 0.7411 |

Ensembling GenScore's own three seeds moves R **+0.026** and rho **-0.007**.
That is the like-for-like bar for any ensemble of ours, and it says the loop's
"R > 0.83" target is an ensemble-scale number: GenScore reaches it by running
three of its own models.

## The pre-registered headline (rotation-ensemble arm)

| f32 | R | rho | R clean | rho clean |
|---|---|---|---|---|
| ours `e250_unionr8_ens`, 8 seeds | 0.8093 | 0.6930 | 0.8149 | 0.7375 |
| GenScore, 1 checkpoint | 0.8127 | 0.7281 | 0.8184 | 0.7589 |
| GenScore, 3 checkpoints | 0.8387 | 0.7211 | 0.8421 | 0.7411 |

**H1 False against both references. H2 (R > 0.83) False.** Our mean single seed
is R 0.7702 / rho 0.6329 against GenScore's single-model 0.8127 / 0.7281, so the
single-model gap is large and the ensemble closes most but not all of it.

## Consequences for the loop

* Any future headline must name the ensemble size on both sides. An 8-seed
  ensemble of ours against one GenScore checkpoint is not like-for-like, and the
  3-checkpoint row is now available, so there is no excuse for reporting only
  the single-checkpoint comparison.
* rho is the metric where we are closest: the exploratory two-recipe mix
  (`corr 8 + unionr8 8`, R 0.8165 / rho 0.7246 at f32) already exceeds the
  3-checkpoint GenScore ensemble's rho 0.7211, while losing 0.022 on R.
* Twin-clean columns move both sides up by about the same amount, so the leak
  screen does not change any comparison here.

## Cost

3 CPU jobs (cpu_8, wall ~15-30 min each), 0.18 pt.
