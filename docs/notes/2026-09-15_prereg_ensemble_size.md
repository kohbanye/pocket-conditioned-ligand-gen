# Pre-registration: does the ensemble keep improving with more seeds?

2026-09-15, written BEFORE seeds 14-17 of `e250_unionnc_corr` are trained.

## Why

The literature pass found ensemble size is the largest measured lever in this
model family: AK-Score goes from single-network R < 0.74 to 0.80 at five
networks and 0.827 at thirty. Our best family (`e250_unionnc_corr`, 4 seeds)
sits at R 0.8125 / rho 0.7140 at f16. It has never been measured at more than 4
seeds.

## Arm

`e250_unionnc_corr`, extended from seeds (7, 11, 12, 13) to (7, 11, 12, 13, 14,
15, 16, 17). Seeds 14-17 use the identical recipe, driver, corpus and weight
(`--corr-weight 0.7`); nothing else changes.

## Predictions, fixed now

On the f32 family ensemble:

* **S1.** R(8 seeds) - R(4 seeds) >= +0.005 -> "ensemble size still helps at
  this scale". Below that -> "saturated by 4-8 seeds", recorded as such.
* **S2.** The same difference for rho is reported beside it.

This is descriptive, not a lever verdict: the 4-seed set is nested inside the
8-seed set, so the difference measures what the added seeds contribute. No
per-seed claim is made.

The 8-seed family is also printed against the goal (R > 0.8127, rho > 0.7281,
R > 0.83) as an **exploratory line, not a headline** -- a headline built on it
would need its own pre-registration and would have to state it is an 8-seed
ensemble.

## Cost

4 head jobs (gpu_1, 2:00) ~= 1.6 pt + 8 inference jobs ~= 0.8 pt ~= 2.4 pt.
