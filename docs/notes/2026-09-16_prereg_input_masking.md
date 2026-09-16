# Pre-registration: input masking during head fine-tuning (2026-09-16)

Written before any treatment head exists. The option (`--input-mask-prob`,
`--input-mask-side`, `--input-mask-pool`) was implemented today; no head has been
trained with it, and no number below has been seen.

## What is changed

One flag group on the existing head recipe. A share of the TRAIN split's
codebook tokens is replaced by the encoder's `<mask>` id before the batch is
built. Validation is never masked (it is the checkpoint-selection signal) and
inference never masks, so a treatment head is scored by exactly the path every
existing arm is scored by -- no `Variant` field, no loader branch, no way for
the two to drift apart.

    control    pipelines/train/scoring_head.py --token-dir data/lm_tokens_e250aff_union_noconf
                 --mlm-ckpt .../mlm-e02-vl0.8009.ckpt --label-cap 13 --max-epochs 20
                 --early-stop-patience 4 --block-size 640 --corr-weight 0.7
    treatment  ... the same, plus
                 --input-mask-prob 0.15 --input-mask-side ligand --input-mask-pool keep

## Why this and not the twenty-six refuted levers

Two things separate it from what has already failed.

1. **It stays inside the pretraining distribution.** The recorded reason the
   input-repair round won +0.0303 on dev and lost 0.0015 on CASF is that it moved
   the encoder off the distribution the VQ and MLM learnt. `<mask>` is the one
   corruption that does not: the MLM saw it at 15% of positions, at every
   position type. The rate is 0.15 for that reason and no other.
2. **It attacks the measured memorisation.** A 105M encoder fine-tuned on ~15k
   complexes ends at train/loss 0.007 (seed 58) against val/loss 0.46. The whole
   R deficit is per-model quality -- ours 0.7920 single against GenScore's
   0.8127, while our ensembling gain is already the larger of the two (+0.031
   against +0.026).

With `--input-mask-pool keep` a masked ligand row stays in the pooled mean, so
the readout averages the state the encoder INFERRED there from the pocket and
the rest of the ligand. That is the only route to the interface that does not
construct a contact pair, a distance bin or a contact count, which the project
has excluded by choice.

**Prior on it is not high.** The nearest precedent, rotation sampling, reduced
memorisation on 4/4 seeds and still failed: it bought family-ensemble gains
(+0.019 R, +0.025 rho) with no per-seed gain (2/4, 1/4). The block-mask arm of
the dev round -- `ligand_block_masking`, a contiguous-in-token-order ligand span
during a 300-step MLM continuation -- came out at dPearson -0.0019 against its
control on the dev 492. That is weak evidence (dev deltas do not predict CASF,
the continuation itself is known to cost affinity, and the span was not spatial),
but it is evidence and it points down.

There is also a known cost: with `keep`, training pools a mixture of observed
and inferred rows while inference pools observed rows only. That train/test
mismatch is the most likely way this loses, and it is why `drop` exists. `drop`
is NOT judged here.

## Arms

| | arm family | seeds | heads |
|---|---|---|---|
| control | `e250_unionnc_corr_v3` (already trained and scored) | 58-65 | 8 |
| treatment | `e250_unionnc_imask` | 58-65 | 8 |

Same driver, same corpus, same recipe, same seed numbers -- so the comparison is
paired. (Seed pairing only removes variance within one driver; both sides here
are `pipelines/train/scoring_head.py` invoked identically but for the three
flags.) Basis f32, f16 reported beside, as fixed on 2026-09-14.

## Decision rules

**M0 (void check, evaluated first).** The final `train/loss` in the treatment
log must exceed the control's in **>= 6/8** paired seeds. The control sits at
0.007; a change that leaves that untouched has not changed training, and the
round is then reported as VOID -- not as a negative.

**B1 (primary: per-model quality).** Mean single-seed CASF scoring R over the 8
treatment heads minus the control's, **>= +0.005**, AND paired **n_better >= 6/8**.

**B2 (secondary: family).** 8-model z-sum, **dR >= +0.004** AND **drho >= -0.005**.

Adoption requires **B1**. B2 alone is recorded as "ensemble-only, the rotation
sampling shape" and is not adopted, not promoted into the reported mix, and not
used to justify further spending.

## Declared in advance

* One rate, one side, one pooling. **No sweep.** 0.15 / ligand / keep is fixed by
  the argument above, not by any measurement.
* If the round fails, the pocket side, `drop`, and other rates are **not**
  re-judged on these seeds and reported as this round's result. A follow-up is a
  new pre-registration with new seeds.
* The reported mix stays the pre-registered 36 (W1). This round cannot change
  the headline; at most it earns a follow-up.
* No new corpus, so the `union_noconf` leak audit (19 twins) applies unchanged,
  and clean columns use the same twin set as `mix36_verdict.py`.
* Budget: 8 heads x 1 h + 16 inferences x 0.5 h on `gpu_1` = **3.2 pt**.

## Companion (separate, free, already decided)

`dpll_probe.py` measures `PLL(ligand | pocket) - PLL(ligand | masked pocket)` on
CASF and asks whether it correlates with pK or with the 36-mix residual. It
trains nothing and is judged descriptively -- the bar taken from the offset-
predictor round is |rho| ~ 0.27 against the per-target bias, where 18 label-free
covariates together reached 0.088. Its in-sample gain is reported as a CEILING
beside a leave-one-cluster-out value, because fitting a coefficient on the test
statistic is how that round produced rho 0.442 for something worth 0.088.
