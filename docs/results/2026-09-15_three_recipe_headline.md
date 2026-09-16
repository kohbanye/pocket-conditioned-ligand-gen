# The three-recipe mix on fresh seeds: best R we have, and still short of the goal

2026-09-15 ~20:10 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_three_recipe_headline.md` before any of the 24
fresh heads existed. Applied by `three_recipe_headline_verdict.py`. All 72 jobs
exited 0; every head was checked for corpus, recipe flag and seed via run.json.

## Result (f32 is the pre-registered basis)

| method | R | rho | R clean (n=266) | rho clean |
|---|---|---|---|---|
| **ours: 3 recipes x 8 fresh seeds (24 models)** | **0.8244** | 0.7228 | 0.8274 | 0.7571 |
| GenScore, 1 released checkpoint | 0.8127 | **0.7281** | 0.8184 | 0.7589 |
| GenScore, 3 released checkpoints | **0.8387** | 0.7211 | 0.8421 | 0.7411 |
| (component) `e250_unionnc_corr_v3` | 0.8129 | 0.7105 | 0.8168 | 0.7446 |
| (component) `e250_unionr8_ens_v3` | 0.8135 | 0.7246 | 0.8180 | 0.7571 |
| (component) `e250_unionheavy_v3` | 0.8045 | 0.7105 | 0.8071 | 0.7429 |
| [exploratory] same recipes, already-scored seeds | 0.8205 | 0.7246 | 0.8243 | 0.7643 |

Our mean single seed: R 0.7923 / rho 0.6708. At f16 the mix is R 0.8249 /
rho 0.7246, the same picture.

**Y1 (beat the single checkpoint on both): FALSE** -- R +0.0117, rho -0.0053.
**Y2 (beat the 3-checkpoint ensemble on both): FALSE** -- rho +0.0017, R -0.0143.
**Y3 (R > 0.83): FALSE** -- 0.8244, short by 0.0056.

## What the two references say

The comparison flips depending on how many GenScore models you run:

* against **one** released checkpoint we win scoring and lose ranking;
* against **three** we win ranking and lose scoring.

Neither is "both", and that is the honest summary of where this model stands.
On the twin-screened subset our ranking margin over the 3-model ensemble is
larger (rho 0.7571 against 0.7411), while the scoring gap persists.

## What this round does establish

* The fresh-seed mix is **better than the already-scored mix** (0.8244 against
  0.8205), so the earlier mixes were not a favourable draw -- the composition
  that two passed pre-registrations implied really is our best configuration.
* It is the **highest R this project has measured** for its own model, up from
  0.7710 for the published single-model arm and 0.8118 for an 8-seed
  single-recipe ensemble.
* The single-model gap is unchanged: **0.7923 against GenScore's 0.8127**.
  Ensembling three complementary recipes closes most of it but not all, and the
  remaining deficit against a like-for-like ensemble is 0.0143 R.

## Cost

24 heads + 48 inferences, ~9.6 pt. Ledger 148.92/200.
