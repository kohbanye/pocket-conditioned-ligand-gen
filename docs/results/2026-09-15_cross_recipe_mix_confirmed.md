# Mixing two recipes works -- confirmed on fresh seeds. The headline still fails.

2026-09-15 ~17:00 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_cross_recipe_mix.md` (lever) and
`docs/notes/2026-09-15_prereg_crossmix_headline.md` (headline), both written
before seeds 18-25 existed. Applied by `crossmix_verdict.py` and
`crossmix_headline_verdict.py`. All 48 jobs exited 0.

## The lever: WORKS (the first one in this loop)

Fresh seeds 18-25 of each recipe, nothing reused from the exploratory pass:

| f32 | n | R | between | within | rho |
|---|---|---|---|---|---|
| `e250_unionnc_corr_v2` (no rotation) | 8 | 0.8073 | 0.3765 | 0.4308 | 0.6982 |
| `e250_unionr8_ens_v2` (rotation) | 8 | 0.8031 | 0.3971 | 0.4060 | 0.6982 |
| **mix** | 16 | **0.8152** | 0.3915 | 0.4237 | **0.7123** |

**M1: dR +0.0080 >= +0.004 PASS. M2: drho +0.0140 >= 0 PASS.** At f16:
dR +0.0064, drho +0.0088, mix R 0.8161 / rho 0.7263.

The mechanism is the one the pre-registration named: the rotation recipe carries
the highest between-target R measured anywhere here (0.3971) and the worst
within (0.4060); the plain recipe is the mirror image; the mix keeps most of
both (between 0.3915, within 0.4237). Same-family seeds cannot do this -- 4 -> 8
seeds of one recipe buys +0.0008.

## The headline: X1, X2, X3 all FAIL

| f32 | R | rho | R clean (n=266) | rho clean |
|---|---|---|---|---|
| ours, 16 models (2 recipes x 8 fresh) | 0.8152 | 0.7123 | 0.8197 | 0.7339 |
| GenScore, 1 released checkpoint | 0.8127 | 0.7281 | 0.8184 | 0.7589 |
| GenScore, 3 released checkpoints | 0.8387 | 0.7211 | 0.8421 | 0.7411 |

* **X2 (vs one GenScore checkpoint): FAIL.** R clears it (+0.0025) but rho does
  not (-0.0158). At f16 the mix is R 0.8161 / rho 0.7263 -- rho misses by
  **0.0018**, which is inside one seed's worth of noise, so it is a miss, not a
  win.
* **X1 (vs GenScore's own 3-checkpoint ensemble): FAIL**, and not close on R
  (-0.0235). This is the like-for-like comparison and the honest bar.
* **X3 (R > 0.83): FAIL** (0.8152).
* Our mean single model is R 0.7828 / rho 0.6547 against GenScore's single
  0.8127 / 0.7281. **The gap is in the single model, and ensembling hides about
  two thirds of it.**

## What this changes

* Mixing recipes is a real, confirmed lever, the only one of 20 tried in this
  loop. It is worth +0.008 R and +0.014 rho over the best single recipe at
  matched training cost per model.
* It does not reach the goal, and stacking more of the same will not: the
  remaining deficit against the fair reference is 0.024 R, three times what the
  confirmed lever delivers.
* Next, already pre-registered and building:
  `docs/notes/2026-09-15_prereg_heavy_third_recipe.md` -- a third recipe whose
  errors decorrelate more (residual corr 0.82-0.88 against 0.945 here), judged
  at fixed ensemble size (24 vs 24).

## Cost

16 heads + 32 inferences, ~6.4 pt. Ledger 128.12/200.
