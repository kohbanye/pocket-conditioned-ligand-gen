# Training the heavy recipe past 20 epochs: refuted

2026-09-15 ~19:30 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_heavy_longer_training.md`, including the void
condition, before any head existed. Applied by `heavylong_verdict.py`. All 24
jobs exited 0.

## The premise held, the lever did not

Treatment heads (last epoch / best-checkpoint epoch), cap 40:
s50 18/14, s51 21/17, s52 20/16, s53 22/18, s54 16/12, s55 23/19, s56 13/9,
s57 14/10. **4 of 8 trained past the old cap of 20**, so the round is testable,
not void.

| f32, 24 models a side | R | between | within | rho |
|---|---|---|---|---|
| heavy-long family alone (8) | 0.8037 | 0.3576 | 0.4462 | 0.6930 |
| heavy-standard family alone (8) | 0.8096 | 0.3676 | 0.4421 | 0.7140 |
| **TREAT** fresh 16 + heavy-long 8 | 0.8194 | 0.3837 | 0.4356 | 0.7298 |
| **CTRL** fresh 16 + heavy-standard 8 | 0.8205 | 0.3867 | 0.4337 | 0.7246 |

**L1: dR -0.0011 < +0.004 FAIL. L2: drho +0.0053 PASS. LEVER DOES NOT WORK.**
At f16: dR -0.0011, drho -0.0140.

## Reading

* The extra epochs did not buy quality: **the longer-trained family is worse on
  its own** (R 0.8037 against 0.8096). Best-checkpoint epochs stayed at 9-19
  even with a 40-epoch cap, so the epochs past 20 went to the patience window,
  not to improvement. The "cap was binding" reading of the earlier epoch counts
  was wrong -- early stopping was already choosing a peak inside the old cap,
  and the few seeds that ran longer were drifting, not still learning.
* Recorded for the next time this pattern appears: **a run stopping near its cap
  is not evidence of under-training; the best-checkpoint epoch is.** Here the
  best epochs (12-19) sat below the cap all along.

## The by-product worth keeping

Both 24-model mixes now clear GenScore's single released checkpoint
(R 0.8127 / rho 0.7281) on at least one frame setting:

| mix | f32 R / rho | f16 R / rho |
|---|---|---|
| fresh 16 + heavy-standard 8 | 0.8205 / 0.7246 | **0.8214 / 0.7333** |
| fresh 16 + heavy-long 8 | **0.8194 / 0.7298** | 0.8203 / 0.7193 |

Each of those "both bars cleared" cells was seen after the fact, inside a
verdict run for a different question, so none of them is a claim. The
composition that follows from the two *passed* pre-registrations -- corr +
rotation + heavy, 8 seeds each -- is now being run on fresh seeds and judged
once: `docs/notes/2026-09-15_prereg_three_recipe_headline.md`.

## Cost

8 heads (2:00 slots) + 16 inferences, ~4.8 pt. Ledger 139.31/200.
