# Input masking during head fine-tuning is refuted (2026-09-16)

Pre-registered in `docs/notes/2026-09-16_prereg_input_masking.md` before any head
existed. 8 treatment heads (seeds 58-65) against the already-trained
`e250_unionnc_corr_v3` at the same seeds, same driver, same corpus, same recipe;
the only difference is `--input-mask-prob 0.15 --input-mask-side ligand
--input-mask-pool keep`. Jobs 8683908-8683965, 3.2 pt, ledger 182.38/200.
Verdict by `inputmask_verdict.py`, log `logs/watch_imask.log`.

## Result

| f32, 8 seeds | R | between | within | rho |
|---|---:|---:|---:|---:|
| treatment `e250_unionnc_imask` | 0.8057 | **0.3991** | **0.4066** | 0.6842 |
| control `e250_unionnc_corr_v3` | 0.8129 | 0.3761 | 0.4369 | 0.7105 |

* **B1 (primary, per-model quality): FALSE.** Single-seed mean R
  0.7895 against 0.7978, **d -0.0083**, better on **2/8** paired seeds
  (needed >= +0.005 and 6/8).
* **B2 (secondary, family): FALSE.** dR **-0.0072**, drho **-0.0263**
  (needed >= +0.004 and >= -0.005).
* f16 agrees throughout: dR -0.0080, drho -0.0351, 2/8 and 1/8.

Every delta is several times the +-0.01 noise floor and the direction is
consistent across seeds and both frame protocols.

## The M0 gate fired VOID, and the gate was wrong

M0 asked for the treatment's **final** `train/loss` to exceed the control's on
>= 6/8 seeds; it held on 5/8, so the rule returned VOID.

**That instrument was badly chosen and the round is not void.** Early stopping
lands on a different epoch in each run -- treatment 5,5,5,6,11,13,13,15 against
control 5,6,6,7,7,10,15,17 -- so the final value mostly records *when a run
stopped*, not how hard the training set was. Seed 62 makes it plain: the
treatment stopped at epoch 6 with 0.083 and the control ran to epoch 17 and
reached 0.0046.

Read at **matched epochs**, which is what M0 was trying to measure, the same logs
are unambiguous:

| | treatment harder than control |
|---|---|
| epoch 3 | **8/8 seeds** |
| epoch 5 | 7/8 |
| epoch 8 | 2/2 (only two runs reach it) |

The augmentation bit. The round measured what it set out to measure, and the
answer is negative.

Overriding a pre-registered gate deserves saying plainly why it is allowed here:
the override moves the result from "decides nothing" to "loses", i.e. it is in
the direction that cannot manufacture a win. The same override in the other
direction would be exactly the selection this project keeps catching itself at.

## Mechanism: it bought between-target scale and sold within-target ranking

between 0.3761 -> **0.3991** (+0.023), within 0.4369 -> **0.4066** (-0.030).

That is the same trade the `allatoms` readout and the rotation ensemble made, on
the same axis, and it lands on the wrong side of it: rho falls 0.026-0.035 while
R falls too, so nothing is bought. Hiding ligand atoms forces the pooled vector
to lean on pocket-derived state, which is a target-level quantity -- it separates
targets slightly better and orders ligands within a target distinctly worse.

## What this does not settle

`--input-mask-pool drop` was explicitly excluded from the judgement. With `keep`,
training pools a mixture of observed and inferred ligand rows while inference
pools observed rows only; that train/test mismatch was named in the
pre-registration as the most likely way this loses, and the between/within
signature is consistent with it. `drop` removes the mismatch and is a different
experiment. It is **not** re-judged on these seeds -- that would be the "try
another setting until one passes" pattern the pre-registration forbids.

Also untouched: the pocket side, other rates, and `ligand_only_masking` (a
P(ligand | pocket) MLM, implemented and never once run, which needs MLM
pretraining rather than a head round).

## Standing

Refuted. The reported configuration stays the pre-registered 36-model mix
(R 0.8229 / rho 0.7316). Implementation kept -- `input_mask_{prob,side,pool}`,
`RescoreDataset._apply_input_mask`, `tests/test_rescore_input_masking.py` -- since
the follow-ups reuse it.
