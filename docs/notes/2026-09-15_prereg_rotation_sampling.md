# Pre-registration: rotation sampling at the dataset level (2026-09-15 ~01:30)

Written BEFORE any head is trained and before any CASF number for these arms
exists. The implementation landed first (config field, dataset option, CLI flag,
6 tests, verified on the real `e250aff_r8` corpus); no arm has been run.

## Why this lever

* **A measured train/test mismatch.** The pocket-canonical frame makes tokens
  frame-dependent. Training sees ONE fixed orientation per complex; inference
  averages 16 or 32 random frames. Nothing has ever trained against the
  distribution the benchmark scores.
* **The obvious version of this already failed, for a reason that does not
  apply here.** `e250_rot8` trained on the 8-rotation corpus as 8x the
  documents: 65,200 docs, the encoder reached train/loss 0.034 against val/loss
  0.922 with its best checkpoint at epoch 0, and it cost 0.079 ranking rho.
  Here the index holds one entry per COMPLEX (8,150, exactly the 1-rotation
  count) and each access draws one of its 8 orientations, so an epoch is the
  same size and only the frame varies across epochs.
* **It adds no contact, edge, distance-bin or contact-count input feature.**
  The peer-group claim (winning against methods that do not construct explicit
  contacts) is preserved: the only change is which of 8 pre-tokenized
  orientations a training step sees.
* It also does not complicate the head: zero new parameters.

## Arms (one flag differs: `--rotations-per-complex`)

| | treatment `e250_rot8_sampled` | control `e250_rot1_ctl` |
|---|---|---|
| corpus | `data/lm_tokens_e250aff_r8` (8 orientations/complex) | `data/lm_tokens_e250aff_r1` |
| flag | `--rotations-per-complex 8` | (default 1) |
| train units | 8,150 complexes, 1 of 8 frames per draw | 8,150 docs |
| val | 431 docs, 1 orientation | 431 docs, same file |

`r1/val.*` and `r8/val.*` are **byte-identical** (verified: sha256 of `val.bin`,
`val.len`, `val.rmsd`, `val.grp` all match), so checkpoint selection is the same
signal on both sides.

Both arms are trained NOW with `pipelines/train/scoring_head.py`, same recipe as
every recent arm (`--label-cap 13 --max-epochs 20 --early-stop-patience 4
--block-size 640`, backbone `mlm_e250lig3` e02, default `ligand` readout),
seeds 7, 11, 12, 13. The existing 6-seed `e250_rot1` family is NOT the control:
it was trained by a different driver, and equal seed numbers across drivers do
not pair trajectories (that error is what produced and then retracted the
"head recipe lead"). f16 and f32 inference for every arm.

## Predictions, fixed now

* **S1 (scoring).** f32 family-ensemble dR >= +0.010 AND treatment R above
  control on >= 3/4 paired seeds.
* **S2 (ranking).** f32 family-ensemble drho >= +0.010 AND treatment rho above
  control on >= 3/4 paired seeds. rho is the metric closest to GenScore
  (-0.014 on the best pool), so a ranking gain is the point of this lever.
* **Lever verdict:** works only if S1 AND S2 hold. One of two is reported as a
  partial result with its numbers, not as a win.
* **M1 (mechanism, recorded not decisive).** The treatment's final train/loss
  should be HIGHER than the control's (augmentation makes the training
  distribution harder to memorise; the control sits at 0.018 train vs 0.695
  val). If M1 fails while S1/S2 pass, the effect is not the mechanism claimed.

## What this round does NOT do

`r1`/`r8` are BioLiP-only corpora whose families score R 0.75-0.79, far below
the 0.83 bar. **No headline is computed here** -- this round only decides
whether the lever works. If it does, Phase 2 rebuilds the union corpus with
`--num-rotations 8` (the PDBbind builder needs the flag added, which it lacks)
and a headline is pre-registered separately before it is scored.

## Cost

8 head jobs (gpu_1, 1:30) ~= 2.4 pt + 16 inference jobs (gpu_1, 0:30) ~= 1.6 pt
~= 4.0 pt, taking the ledger to ~108 of the 200 pt cap.
