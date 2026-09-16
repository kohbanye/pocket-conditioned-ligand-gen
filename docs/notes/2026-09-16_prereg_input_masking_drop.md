# Pre-registration: input masking with `drop` pooling (2026-09-16, second round)

Written after the `keep` round was refuted and before any head of this round
exists. Required by that round's own rule: a follow-up is a **new**
pre-registration on **new seeds**, never a re-judgement of another setting
against the control whose realisation has already been seen.

## What is changed, and why a fresh control

    treatment  ... --input-mask-prob 0.15 --input-mask-side ligand --input-mask-pool drop
    control    ... the same recipe with no masking flags at all

`keep` leaves a masked ligand row inside `ligand_mask`, so training pools a
mixture of observed and INFERRED rows while inference pools observed rows only.
That train/test mismatch was named in the first pre-registration as the most
likely way it would lose, and the measured signature agreed: between 0.3761 ->
0.3991, within 0.4369 -> 0.4066, dR -0.0072, drho -0.0263.

`drop` removes the masked position from the pooled mean, so the pooled statistic
is "mean over observed ligand atoms" in training and at inference alike. It is
then plain input dropout on the encoder's input, with no mismatch to pay for.

**Seeds 134-141, both arms, including a newly trained control.** The existing
control (`e250_unionnc_corr_v3`, seeds 58-65) has been read: its per-seed R is
known, so a second treatment judged against it is a second test against a fixed
realisation, and `n_better` stops being a fresh statistic.

| | family | stem | seeds |
|---|---|---|---|
| treatment | `e250_unionnc_imaskdrop` | `aff_e250_unionncimdrop` | 134-141 |
| control | `e250_unionnc_ctl_v5` | `aff_e250_unionncctl5` | 134-141 |

## Prior: low, and stated before the fact

`drop` removes information from the pooled mean and adds none. The most likely
outcome is the rotation-sampling shape -- a mild regulariser with family-ensemble
gains and no per-seed gain -- which this project has already refused once. It is
also a recipe-side lever, and 83% of the per-target bias is common to every
recipe family, so its reach is the 17% that recipe changes have always moved.

It is worth 6.4 pt anyway because it is the one reading that decides **why** the
`keep` round lost: mismatch, or masking itself. That answer closes the family.

## Decision rules

**M0 (void check).** The instrument is fixed from the last round, where "final
`train/loss`" turned out to record *which epoch early stopping landed on* (the
runs stopped anywhere from epoch 5 to 17). Here: treatment `train/loss` **at
matched epochs 3 and 5** must exceed the control's on **>= 6/8** seeds at each.
If not, VOID -- reported as unmeasured, not as a negative.

**D1 (primary: per-model quality).** Mean single-seed f32 CASF R, treatment minus
control, **>= +0.005**, AND paired **n_better >= 6/8**.

**D2 (secondary: family).** 8-model z-sum, **dR >= +0.004** AND **drho >= -0.005**.

Adoption requires **D1**. D2 alone is "ensemble-only" and is not adoption.

## Declared in advance

* **Descriptive, not decisive**: (drop - its own control) is compared against
  (keep - its own control, -0.0083 single-seed / -0.0072 family). Each delta is
  measured against a seed-matched control of its own, so the two deltas are
  comparable even though the seed sets differ. Pre-stated reading: if the
  train/test mismatch was the cause of the `keep` loss, `drop`'s delta should sit
  near 0; if `drop` loses by about the same amount, the cause is the masking
  itself and the whole ligand-masking family is closed.
* One rate, one side, one pooling. **No sweep.** Whatever the outcome, no third
  setting is judged against these seeds.
* The reported configuration stays the pre-registered 36-model mix. This round
  cannot change the headline.
* No new corpus, so the `union_noconf` leak audit (19 twins) applies unchanged.
* Budget: 16 heads x 1 h + 32 inferences x 0.5 h on `gpu_1` = **6.4 pt**
  (ledger 182.38 -> 188.78 / 200).
