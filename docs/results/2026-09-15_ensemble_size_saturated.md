# Ensemble size is saturated by 4-8 seeds

2026-09-15 ~16:00 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_ensemble_size.md` before seeds 14-17 were trained;
applied by `experiments/affinity_loop_20260914/ensemble_size_verdict.py`.

## Result

`e250_unionnc_corr`, seeds 7, 11, 12, 13 extended with 14, 15, 16, 17 (same
recipe, driver, corpus, `--corr-weight 0.7`). All 16 dumps present; all 12 new
jobs exited 0.

| family z-sum | R | between | within | rho |
|---|---|---|---|---|
| f32, 4 seeds | 0.8110 | 0.3742 | 0.4368 | 0.6965 |
| f32, 8 seeds | 0.8118 | 0.3716 | 0.4402 | 0.6947 |
| f16, 4 seeds | 0.8125 | 0.3766 | 0.4359 | 0.7140 |
| f16, 8 seeds | 0.8132 | 0.3734 | 0.4398 | 0.7070 |

**S1 (f32): dR +0.0008 < +0.005 -> saturated by 4-8 seeds.** S2: drho -0.0018
(f16 -0.0070).

Nested curve (R/rho, seeds added in order 7, 11, ..., 17), f32:
1: 0.8042/0.7000, 2: 0.8083/0.7000, 3: 0.8106/0.6982, 4: 0.8110/0.6965,
5: 0.8092/0.6947, 6: 0.8102/0.6965, 7: 0.8117/0.6947, 8: 0.8118/0.6947.
At f16 the fifth seed alone moved rho 0.7140 -> 0.6825.

## Reading

* The AK-Score path (single net < 0.74 -> 30 nets 0.827) does not transfer:
  our single seeds already sit at R ~0.80 and the ensemble gain is spent by
  3-4 seeds.
* The 4-seed family's f16 rho 0.7140 was a favourable draw; the 8-seed value
  0.7070 is the better estimate. rho moves +-0.02 with a single seed, so any
  rho comparison between families of <= 8 seeds carries that much noise.
* Exploratory goal line (8-seed ensemble): f16 R 0.8132 clears GenScore's
  0.8127; rho 0.7070 does not clear 0.7281; R 0.83 is not reached. Not a
  headline (the pre-registration says so).

## Cost

4 heads (wall ~10 min each) + 8 inferences, ~2.4 pt reserved.
