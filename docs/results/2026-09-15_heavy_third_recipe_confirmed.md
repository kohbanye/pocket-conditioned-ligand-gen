# A heavy-atom-pocket third recipe works -- and its profile is the opposite of what was predicted

2026-09-15 ~18:50 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_heavy_third_recipe.md` before the corpora were
assembled and before any head existed. Applied by `heavy_verdict.py`. All 24
jobs exited 0; all 8 heads trained on `union_heavy` at 466 steps/epoch with the
default ligand readout (run.json).

## Result (f32 basis, 24 models on both sides)

| | n | R | between | within | rho |
|---|---|---|---|---|---|
| fresh mix (2 recipes x 8) | 16 | 0.8152 | 0.3915 | 0.4237 | 0.7123 |
| `e250_unionheavy` alone | 8 | 0.8096 | 0.3676 | **0.4421** | 0.7140 |
| **TREAT** fresh 16 + heavy 8 | 24 | **0.8205** | 0.3867 | 0.4337 | 0.7246 |
| **CTRL** fresh 16 + same-recipe 8 | 24 | 0.8159 | 0.3879 | 0.4280 | 0.7228 |

**T1: dR +0.0045 >= +0.004 PASS. T2: drho +0.0018 >= -0.005 PASS. LEVER WORKS.**
At f16: dR +0.0049, drho -0.0035, treatment R 0.8214 / rho 0.7333.

## The mechanism was the opposite of the prediction

The pre-registration argued heavy pockets would add a *between*-target
complementary family (that is what the dev-cohort residual correlations
suggested). The trained family is the reverse: **within 0.4421 is the highest of
any family measured here, between 0.3676 the lowest.** It works because the mix
was short of within, not because it was short of between -- the rotation recipe
had already taken the between corner, which is exactly why the all-atom recipe
(between 0.4032, the other between-heavy family) added nothing.

So the rule that survives three rounds is: **a new recipe helps only if it
occupies a corner the mix does not already hold.** Two recipes with the same
complementarity direction do not stack (all-atom round), two with opposite
directions do (crossmix and this round).

## Where this leaves the goal

| f32 | R | rho |
|---|---|---|
| ours, 24 models, 3 recipes | 0.8205 | 0.7246 |
| GenScore, 1 released checkpoint | 0.8127 | 0.7281 |
| GenScore, 3 released checkpoints | 0.8387 | 0.7211 |

* R clears the single GenScore checkpoint by +0.0078; rho misses by 0.0035.
* **At f16 the same mix is R 0.8214 / rho 0.7333, which clears both** -- but f32
  was the pre-registered basis and these f16 numbers were seen as a by-product
  of the verdict, so they are exploratory. Claiming them needs a fresh-seed
  round of its own.
* R > 0.83 is still 0.0095 away, and the like-for-like ensemble reference
  (0.8387) is 0.018 away.

## Next

The heavy heads reached their best validation loss at epochs 13-18 and stopped
at 17-19 against a cap of 20, while the ca recipes peak at epochs 3-9. That is
evidence of under-training specific to this recipe, and it is the cheapest
remaining mechanism with a concrete rationale:
`docs/notes/2026-09-15_prereg_heavy_longer_training.md`.

## Cost

8 heads + 16 inferences, ~3.2 pt (plus 1.0 pt of corpus builds, 0.6 of which was
a rebuild of an existing corpus that was killed once noticed). Ledger 134.52/200.
