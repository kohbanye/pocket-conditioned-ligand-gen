# Pre-registration: distil the ensemble into a single head

2026-09-15 ~23:40 JST. Written BEFORE the teacher predictions are computed and
before any student head exists.

## Why this, chosen after a flat re-read of what is actually measured

Two in-house numbers, both far outside the +-0.01 R noise band:

* a single model averages **R 0.792**, an 8-seed family reaches **0.8129**, the
  36-model mix **0.8229** -- ensembling is worth ~+0.03 R and we pay for it with
  36 forward passes;
* freezing the encoder collapses **between**-target R (0.376 -> 0.272) while
  **within** slightly improves (0.437 -> 0.457) -- the cross-target affinity
  scale is produced by fine-tuning, not read off the pretrained representation.

Distillation is the one documented way to move ensemble-level accuracy into a
single model, and a better single model raises every ensemble built from it.
Self-distillation can exceed the teacher (Furlanello et al., *ICML* 2018,
"Born-Again Networks") -- vision benchmarks, so a mechanism-existence proof, not
a magnitude estimate for our R.

## Teacher

The mean predicted pK of the **8 `e250_unionnc_corr_v3` heads** on each training
document of `union_noconf`, written as a float32 sidecar aligned to the document
stream. That family scores R 0.8129 on CASF against a single seed's ~0.79, so it
is a stronger teacher than the student can be alone.

## Student

Same corpus, same recipe as the control, plus a distillation term:

`loss = (1 - lambda) * base_loss(pred, label) + lambda * SmoothL1(pred, teacher)`
with **lambda = 0.5**, one value, not swept (a sweep chosen on CASF would be
selecting on the test set).

* **Treatment** `e250_unionnc_kd` (NEW, 8 seeds 126-133).
* **Control** `e250_unionnc_corr_v3` (8 seeds 58-65), already trained and scored.

## Predictions (f32 basis, 8-seed family ensembles)

* **D1 (scoring).** R(KD 8) - R(control 8) >= **+0.005**.
* **D2 (ranking).** rho(KD 8) >= rho(control 8) - 0.005.

**Works only if D1 and D2 hold.** f16 reported beside.

**Void condition, fixed now.** The teacher was trained on these very labels, so
its training-set predictions may simply *be* the labels, leaving nothing to
distil. If teacher-vs-label agreement on the training set is **Pearson r > 0.99
AND RMSE < 0.10 pK**, the round is reported **void**, not refuted, and the
agreement statistics are recorded.

Reported beside, not part of the verdict: teacher-vs-label agreement, and the
**mean single-seed R** of the KD family against the control's -- the documented
gap this technique is supposed to close is a single-model gap, so that number is
the mechanism check even if the ensemble comparison does not move.

## Cost

8 teacher-scoring jobs (gpu_1, 0:30) ~= 0.8 pt + 8 heads (gpu_1, 1:00) ~= 1.6 pt
+ 16 inferences ~= 1.6 pt + collector ~= 0.03 pt ~= **4.0 pt**, ledger to about
174 of the 200 cap.
