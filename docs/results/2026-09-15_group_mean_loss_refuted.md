# The group-mean loss is refuted, and its mechanism ran backwards

2026-09-15 11:45 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_group_mean_loss.md` (with Amendment 1) before any
head was trained; applied by
`experiments/affinity_loop_20260914/group_mean_verdict.py`, written while the
heads were still queued. Raw output in `group_mean_verdict_output.txt`.

All 36 jobs verified before the verdict: `qacct` exit 0, clean log ends,
checkpoints present, 285/285 non-NaN in every dump.

## Verdict: DOES NOT WORK (G1 False, G2 False, G3 True)

Three arms, because Amendment 1 caught a batching confound before any number
was read (see below).

| f32 family ensemble | R | between | within | rho | shrink slope |
|---|---:|---:|---:|---:|---:|
| **C** treatment: grouped batching, weight 1.0 | 0.7520 | 0.3260 | 0.4260 | 0.6807 | **-0.3384** |
| **B** control: grouped batching, weight 0 | 0.7931 | 0.3516 | 0.4415 | 0.6772 | -0.2476 |
| **A** reference: random batching, weight 0 | 0.8063 | 0.3816 | 0.4247 | 0.7035 | -0.1696 |

* **C - B (the loss term):** dR **-0.0410**, paired **0/4** seeds
  (-0.0145, -0.0546, -0.0461, -0.0376). G2 fails.
* **G1 (the mechanism) fails in the wrong direction.** The slope of cluster-mean
  error against cluster-mean pK went from -0.248 to **-0.338**. Penalising the
  group's mean residual made the compression of target-level differences
  *worse*, not better. The pre-registration asked for >= -0.12.
* **B - A (the batching):** dR **-0.0133**, drho -0.0263. Group batching hurts
  on its own, consistent with the val loss measured before any CASF number
  (0.8165 -> 0.8252 on the identical objective).
* G3 held (drho +0.0035) but decides nothing when R falls by 0.041.
* f16 agrees throughout: dR -0.0443, 0/4 seeds.

## Why the mechanism inverted, stated as a hypothesis

The diagnosis measured a -0.19 slope and read it as miscalibration: the model
"shrinks" target-level differences and should be pushed to stop. The refuted
prediction says that reading was wrong. If the per-target mean prediction is
NOISY, shrinking toward the global mean is the error-minimising response, not a
bias to be corrected -- and a term that forces the group mean onto the label
mean adds variance to a quantity the model cannot estimate well, which is
exactly what a larger negative slope and a lower R look like.

That is also what the external analysis predicted before these numbers existed:
scaling up the between-target component was measured to be counterproductive
(optimal rescale a = 0.80, i.e. MORE shrinkage, not less).

**Consequence for the diagnosis.** "We compress target-level differences by 19%"
stays true as a description, but it is not an actionable defect. The remaining
reading is that our per-target estimate is noisier than GIGN's -- a signal
problem, not a calibration one.

## Amendment 1 earned its cost

The first control passed `--group-batch-partition` but the loader only switched
to group batches when a grouped LOSS was active, so it trained on random batches
(479 steps/epoch against the treatment's 722). Caught from the step counts
before any CASF number was read, fixed in
`RescoreDataModule._loader`, and a proper control (B) was trained.

Had the original control stood, this round would have reported dR **-0.054**,
mixing the loss term (-0.041) with the batching (-0.013). The three-arm split is
what separates them.

Fifteen levers are now refuted or neutral.
