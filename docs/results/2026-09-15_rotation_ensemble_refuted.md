# Rotation augmentation on the union corpus: refuted at family level too

2026-09-15 ~16:10 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_rotation_ensemble.md` before the corpus was built;
applied by `experiments/affinity_loop_20260914/rotation_ensemble_verdict.py`.

## What was run

`data/lm_tokens_e250aff_union_r8`: union_noconf plus 7 extra orientations per
training complex, built by `build_union_r8.py` and verified before training --
all 15,351 k=0 documents byte-identical to union_noconf, labels equal, val
byte-identical. 8 heads (`--rotations-per-complex 8`), 479 steps/epoch, the same
as the control, so the epoch is complex-sized and the batching matches.
All 24 jobs exited 0.

## Result

| f32 family z-sum | R | between | within | rho | shrink slope | per-seed mean R |
|---|---|---|---|---|---|---|
| `e250_unionr8_ens` (8 seeds) | 0.8093 | **0.3852** | 0.4241 | 0.6930 | -0.159 | 0.7702 |
| `e250_union_noconf` (6 seeds) | 0.8101 | 0.3801 | 0.4301 | 0.6930 | -0.169 | 0.7929 |

**E1 (dR >= +0.010): False (-0.0008). E2 (drho >= +0.010): False (0.0000).
LEVER DOES NOT WORK.** At f16: dR -0.0033, drho +0.0193.

## Reading

* The BioLiP-only round's family gain (dR +0.0143, drho +0.0193) did not
  reproduce on the union corpus. The f16 drho here is +0.0193 again, but f32 is
  0.0000 and f32 was the pre-registered basis -- the f16/f32 split is the size
  of the effect, which is what "rho moves +-0.02 per seed" already said.
* **Rotation sampling makes each model worse** (per-seed mean R 0.7702 vs
  0.7929, -0.023) and the ensemble only recovers to parity. On the union corpus
  the extra orientations cost more than the diversity buys.
* The one real signal: **between-target R rises to 0.3852, the highest measured
  for any of our arms** (control 0.3801), while within falls 0.4301 -> 0.4241.
  Rotation attacks frame/quantization noise, which is a between-target error
  source; it pays for it in within-target discrimination. Both halves of that
  are consistent with the diagnosis, and they cancel.
* The shrinkage slope improves slightly (-0.169 -> -0.159) -- the first
  intervention to move it in the right direction at all -- but the movement is
  far too small to matter.

## Cost

2 corpus builds + 8 heads + 16 inferences, ~5 pt reserved. Ledger ~120.7/200.
