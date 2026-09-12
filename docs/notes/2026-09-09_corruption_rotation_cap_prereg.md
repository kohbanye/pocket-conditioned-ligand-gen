# Pre-registration — the refiner is trained to undo a rotation nothing ever makes

Written 2026-09-09, before training. Arrived at by asking what the refiner is
shown rather than how it is built, after seven settings of target, head,
magnitude, step count and schedule all capped at 0.482 kcal.

## The defect

`tokenize_pose_refine.py` manufactures the training input by round-tripping the
crystal ligand through the VQ and adding a graded rigid displacement plus
isotropic jitter. That is the right *shape* of training data — a manual
displacement to be refined — and this arm keeps it. What is wrong is the
distribution.

Measured on 2,000 training records, the rigid rotation between the stored
corrupted pose and the crystal pose:

| percentile | rotation | centroid | RMSD |
|---|---|---|---|
| 10% | 7.5° | 0.496 Å | 1.702 |
| 25% | 12.0° | 0.762 Å | 2.098 |
| **50%** | **20.6°** | 1.134 Å | 2.898 |
| 75% | **45.6°** | 1.658 Å | 3.956 |
| 90% | **109.6°** | 2.238 Å | 4.989 |
| 99% | **175.6°** | 4.031 Å | 7.284 |

**25.4% of records exceed 45°, 13.8% exceed 90°, 7.2% exceed 135°.**

What deployment asks for, measured on generated poses: the closed-form fit moves
them 18.55° and Vina's rigid optimum 18.42°, with centroid shifts of 1.04 and
0.73 Å. The median of the training corruption (20.6°, 1.134 Å) matches that
well. **The tail does not exist at inference.**

Two consequences, both consistent with what the deployed net does:

1. Capacity spent unrolling a flipped molecule is not spent placing a
   nearly-right one.
2. The conditional mean of a multimodal large rotation is short — and the
   deployed refiner emits 3.96°, closing **16%** of the distance to its own
   target.

## The arm

`refiner_rotcap30`: identical to `refiner_overlap_t0` — same corpus
`data/pose_refine_clm`, same overlap target, same `--t-schedule zero`, same
warm start from `refiner_overlap/refine-e05-r2.0691.ckpt`, 6 epochs,
micro-batch 8, lr 3e-4, seed 7 — plus **`--max-corrupt-rot-deg 30`**.

The filter is a view over the same records: 21,484 → **13,778** on train
(64.1%) and 668 → 415 on val, and `tests/test_refine_projection.py` asserts the
survivors come back byte for byte. No pose is altered and no data is
manufactured.

Generation and evaluation: same VQ, same LM `clm_dfs_full_scratch`, seed 0,
T=0.7, top-p 0.95, `--refine-project rigid`, one round. Control:
`refoverlap_t0`; noref control `dfs_full_noref`.

## The confound, declared in advance

Filtering removes **36% of the training records**. A gain could therefore be
the rotation cap or could be an accident of training on less data, and this arm
cannot separate them. If it wins, the separating control is a second arm
trained on a *random* 64.1% of the records — same count, same everything, cap
off. **That control is not run first**, because if the cap loses there is
nothing to attribute.

## Registered predictions

- **Score**: paired median **−0.2 to −0.6 kcal** against `refoverlap_t0`,
  p < 0.05, on both bases. The band's top is set by the gap between what the
  refiner delivers (0.482) and what its own target is worth applied directly
  (1.131): closing a third of that is what "stop wasting capacity on a regime
  that never occurs" could plausibly buy. Paired ≥ 0, or p > 0.05, falsifies.
- **Mechanism, checked before generation**: the trained net's median rigid
  rotation on the canon100 poses must **rise above the deployed 3.96°**. The
  whole argument is that the hedge is short because the training distribution
  is long-tailed; if the step does not lengthen, the argument is wrong whatever
  the score does.
- **Mediator**: clash-free must rise (control 0.3213).
- **Costs are structurally zero**: same LM, same seed, same molecules, so
  uniqueness 0.5470, scaffold 0.3340, novelty 0.8202 and validity 0.9940 must
  return unchanged to within one molecule in ten thousand.
- **Disqualification**: PoseBusters validity falling more than 0.03 against
  `refoverlap_t0`.

## Cost

~2 h training on the reduced set, ~45 min generation, ~1.5 h evaluation, after
the `pctx + MLM` chain frees the GPUs.

---

## Addendum, 17:55 — the validation RMSD is not comparable to the control's

The filter applies to **both** splits: val goes 668 → 415 records, and the 253
it drops are exactly the large-rotation ones. So this arm's val RMSD is
measured on an easier set than the control's, and would read better even for an
identical model.

For the record, not for comparison: this arm reads 1.9315 / 1.9131 / 1.9197
across epochs 0–2 on **415** records; `refiner_overlap_t0` read 1.9364 / 1.9464
/ 1.9315 on **668**. Neither pair says anything about the other.

This is the same class of trap as the anchor arm's weighted validation loss two
days ago, and it is flagged here for the same reason: the registered mechanism
check (median rigid rotation of the trained net on the canon100 poses, read
before the score) is computed identically for both arms and is the instrument
that counts.

**One registered condition is relaxed.** The prereg says the mechanism check
runs "before generation"; measuring the net's rotation on canon100 poses
requires applying the refiner to them, which is what generation does. It
therefore runs **after generation and before the score**, which preserves its
purpose — it is still a prediction rather than a post-hoc explanation — and the
chain prints it above the evaluation step so the ordering is on the record.

---

## Mechanism check, 19:40 — FAILED, and in the opposite direction

Read from the finished dumps before the evaluation, as registered. Kabsch
against the paired `dfs_full_noref` dump, 10,000 molecules:

| | `refrotcap30` | `refoverlap_t0` (control) |
|---|---|---|
| **rigid rotation, median** | **3.715°** | 3.964° |
| displacement RMSD, median | 0.701 Å | 0.754 Å |
| centroid translation, median | 0.641 Å | 0.688 Å |
| non-rigid residual | 0.000 | 0.000 |

The registered condition was that the median rotation must **rise above
3.96°**. It fell, and so did every other measure of step length.

**The reasoning is falsified, independently of the score.** The argument had
two parts:

1. capacity spent unrolling a flipped molecule is not spent placing a
   nearly-right one — untested, and this arm cannot test it;
2. the conditional mean of a multimodal large rotation is short, so removing
   the tail should lengthen the step.

Part 2 is wrong. Removing the tail also lowered the *mean magnitude* of the
training targets, and the net's step tracked that down rather than tracking the
multimodality up. **The hedge follows the scale of what it is shown, not its
shape.** That is a more parsimonious account of the 16%-of-target step than the
one this arm was built on, and it also explains why α-scaling and extra rounds
both failed: they change the output's scale without changing what the training
distribution taught the net to expect.

The evaluation runs anyway — it is already queued and the number is cheap
against having generated 10,000 molecules — but the registered score prediction
(−0.2 to −0.6 kcal) is now expected to fail, and the confound control (a random
64.1% subsample) will not be needed.
