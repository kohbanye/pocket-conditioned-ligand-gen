# 36 models beat one released GenScore checkpoint on both metrics

2026-09-15 ~22:35 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_mix_scale36.md` before the 12 scale-up heads
existed, including the caveats repeated below. Applied by `mix36_verdict.py`.
All 36 jobs exited 0; every head checked for corpus, recipe flag and seed.

## Result (f32 is the pre-registered basis)

| method | R | rho | R clean (n=266) | rho clean |
|---|---|---|---|---|
| **ours: 3 recipes, 36 models** | **0.8229** | **0.7316** | **0.8261** | **0.7643** |
| (reference) the 24-model mix | 0.8244 | 0.7228 | 0.8274 | 0.7571 |
| GenScore, 1 released checkpoint | 0.8127 | 0.7281 | 0.8184 | 0.7589 |
| GenScore, 3 released checkpoints | 0.8387 | 0.7211 | 0.8421 | 0.7411 |

**W1 (beat the single released checkpoint on R AND rho): TRUE** -- R +0.0102,
rho +0.0035, and both margins survive the twin screen (+0.0077 / +0.0054 on the
266 complexes with no training twin). At f16 the same mix is R 0.8235 /
rho 0.7351, the same picture.

**W2 (beat GenScore's own 3-checkpoint ensemble on both): FALSE** -- rho +0.0105
but R -0.0158.

**W3 (R > 0.83): FALSE** -- 0.8229, short by 0.0071.

## What this is, stated as pre-registered

* **36 models of ours against 1 of theirs is not equal cost.** The like-for-like
  comparison is the 3-checkpoint row, and we lose it on R.
* **The rho margin is inside the noise we declared.** +0.0035 against a stated
  +-0.01 spread at this ensemble size: the pre-registration called this close to
  a coin flip, and the result does not change that.
* **The 24 models of the earlier round were reused**; only 12 seeds are new.
* Our mean single model is R 0.7920 / rho 0.6718 against GenScore's single
  0.8127 / 0.7281. The ensemble is doing the work.

## The mechanism of the flip

Going from 24 to 36 models **traded R for rho**: R 0.8244 -> 0.8229 while rho
0.7228 -> 0.7316. W1 passed because of the rho half of that trade. The added
seeds are individually weaker on R (v4 components 0.8019-0.8108 against v3's
0.8045-0.8135) but push the ensemble's ranking up.

That is consistent with everything measured in this loop: R and rho are bought
from different places, and this model family has no lever that buys both at
once.

## Where the loop's goal stands

Two of the three conditions are met (beat GenScore on R, beat it on rho), both
against the single released checkpoint and both surviving the leak screen. The
third, R > 0.83, is 0.0071 away and no measured lever closes it: 24 refuted
levers, and the only two that work (mixing complementary recipes, adding the
heavy-pocket recipe) are already in this number.

## Cost

12 heads + 24 inferences, ~4.8 pt. Ledger 160.12/200.
