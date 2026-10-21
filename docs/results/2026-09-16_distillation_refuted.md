# Distilling our own ensemble: refuted, and the mechanism went backwards

2026-09-16 ~00:40 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_distillation.md` before the teacher predictions
were computed. Applied by `kd_verdict.py` via an unattended collector job. All
24 jobs exited 0.

## The teacher was real, not a copy of the labels

The pre-registered void condition (teacher-vs-label pearson > 0.99 **and** rmse
< 0.10) was not met: the 8-head ensemble teacher agrees with the training labels
at pearson **0.9885**, rmse **0.4079**, and the heads disagree with each other by
**0.4948 pK** per training document on average. There was something to transfer.

Per-head agreement varied widely (pearson 0.8856-0.9971, rmse 0.41-0.92), i.e.
the heads memorise the training set to very different degrees.

## Result (f32 is the pre-registered basis)

| | n | R | between | within | rho |
|---|---|---|---|---|---|
| **CTRL** the teacher's own family | 8 | **0.8129** | 0.3761 | 0.4369 | **0.7105** |
| **TREAT** distilled students | 8 | 0.8102 | 0.3702 | 0.4400 | 0.6965 |

**D1: dR -0.0027 FAIL. D2: drho -0.0140 FAIL.** At f16: -0.0027 and +0.0000.

**The mechanism check failed too, in the wrong direction.** Mean single-seed R
went **0.7978 -> 0.7930** (-0.0048). Distillation was supposed to move a single
model toward the ensemble; it moved it away, closing **-32%** of the
single-to-ensemble gap.

## Why, and it was predicted

The same night's structural measurement
(`2026-09-16_target_bias_is_shared_and_unexplained.md`) found that 83% of our
per-target bias is shared across every family in this architecture. The teacher
is drawn from that same family, so **it carries the identical 0.660 pK shared
bias**. Its 0.41 pK of disagreement with the labels is mostly its own
memorisation, not knowledge that generalises to CASF.

Distilling a teacher that shares your systematic error transfers the error, not
the cure. That closes the remaining route of "raise the single model, and every
ensemble built from it rises with it" -- at least for any teacher drawn from
this family.

## Cost

8 teacher-scoring jobs + 8 heads + 16 inferences + collector, ~4.0 pt.
Ledger 173.78/200.
