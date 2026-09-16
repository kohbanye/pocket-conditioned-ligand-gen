# Pre-registration: let the layer weights actually learn

2026-09-15 ~21:20 JST. Written BEFORE the two options exist and before any head
using them is trained.

## What the last round established, and what it did not

`docs/results/2026-09-15_layer_mix_uniform_refuted.md`: the mixing weights never
moved from their uniform initialisation (every layer 0.071-0.073 across all 8
seeds), so the round measured **the uniform average of all 14 states against the
last layer** and the average lost badly (dR -0.0173, drho -0.0281).

The question the round was meant to ask -- *is some layer other than the last a
better place to read this representation?* -- is still open, because the head
never got to choose. Fourteen scalars behind a frozen encoder, trained at the
head's learning rate, do not move.

## The two changes

* **`--layer-mix-init-last L`** (treatment uses 4.0): the mixing logits start at
  0 except the last state, which starts at L. With 14 states that is a softmax
  of 0.808 on the last layer and 0.015 on each of the others, so training
  **starts approximately where the control already is** and can move away only
  if the data asks it to.
* **`--layer-mix-lr-scale S`** (treatment uses 100): the 14 mixing scalars get
  their own optimizer group at `S x` the head's learning rate. Nothing else
  changes group, and the encoder stays frozen.

Neither knob touches inference: the learned weights live in the checkpoint, and
`layer_mix=True` alone rebuilds them. Both are recorded in the run's config, so
a checkpoint says how it was trained.

## Arms

* **Treatment** `e250_unionnc_layermix_lr` (NEW, 8 seeds 90-97): `union_noconf`,
  `--corr-weight 0.7 --layer-mix --layer-mix-init-last 4.0 --layer-mix-lr-scale 100`,
  otherwise the standard recipe.
* **Control** `e250_unionnc_corr_v3` (8 seeds 58-65): same corpus, same recipe,
  last layer only. Already trained and scored.

## Predictions (f32 basis, 8-seed family ensembles)

* **ZL1 (scoring).** R(treatment) - R(control) >= **+0.005**.
* **ZL2 (ranking).** rho(treatment) >= rho(control) - 0.005.

**Works only if ZL1 and ZL2 hold.** f16 reported beside.

**Void condition, fixed now (the previous round's lesson applied in advance):**
if the mean absolute change of the softmax weights from their initial values is
**< 0.02** averaged over seeds, the weights again failed to learn; the round is
reported **void**, not refuted, and the observed movement is recorded.

Reported beside, not part of the verdict: the final weight per layer for every
seed (does the head move toward the middle layers, or stay on the last one?),
and the 24-model headline mix with this family swapped in.

## Cost

8 heads (gpu_1, 1:00) ~= 1.6 pt + 16 inferences ~= 1.6 pt ~= **3.2 pt**, ledger
to about 155 of the 200 pt cap.
