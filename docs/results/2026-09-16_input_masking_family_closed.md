# `drop` loses too: the ligand input-masking family is closed (2026-09-16)

Pre-registered in `docs/notes/2026-09-16_prereg_input_masking_drop.md` before any
head existed. Seeds 134-141, **both arms newly trained**, because the `keep`
round's control had already been read. Jobs 8684679-8684726, 6.4 pt, ledger
188.78/200. Verdict by `imaskdrop_verdict.py`, log `logs/watch_imaskdrop.log`.

## Result

| f32, 8 seeds | R | between | within | rho |
|---|---:|---:|---:|---:|
| treatment `e250_unionnc_imaskdrop` | 0.8048 | **0.3962** | **0.4086** | 0.6754 |
| control `e250_unionnc_ctl_v5` | 0.8132 | 0.3767 | 0.4365 | 0.7193 |

* **D1 (primary): FALSE.** Single-seed mean R 0.7904 against 0.7994,
  **d -0.0090**, better on **2/8** (needed >= +0.005 and 6/8).
* **D2 (secondary): FALSE.** dR **-0.0084**, drho **-0.0439**.
* f16 agrees: -0.0096 single, -0.0089 / -0.0579 family, 2/8 and 1/8.

## The pre-stated question is answered: the masking is the cause, not the mismatch

| against its own seed-matched control | single | family dR | family drho |
|---|---:|---:|---:|
| `keep` (seeds 58-65) | -0.0083 | -0.0072 | -0.0263 |
| `drop` (seeds 134-141) | **-0.0090** | **-0.0084** | **-0.0439** |

The pre-registration stated the reading before the fact: `drop` near 0 would mean
the train/test pooling mismatch caused the `keep` loss; `drop` losing by about
the same would mean the masking itself does. **It loses by the same, slightly
more.** Removing the mismatch changed nothing.

The between/within signature is the same on both, which is the mechanism saying
the same thing a second way:

| | between | within |
|---|---:|---:|
| `keep` - control | +0.023 | -0.030 |
| `drop` - control | +0.020 | -0.028 |

Hiding ligand atoms pushes the pooled vector onto pocket-derived state -- a
target-level quantity. Targets separate a little better, ligands within a target
order distinctly worse, and the net is negative on both metrics. Two poolings,
two independent seed sets, two frame protocols, one answer.

**The ligand input-masking family is closed.** No third setting is judged.

## The M0 gate fired VOID again, and this time the gate was the wrong idea

Epoch 3: treatment harder on **8/8**. Epoch 5: **4/8**. The rule asked for >= 6/8
at both, so it returned VOID.

Epoch 3 at 8/8 already settles the only question M0 exists to ask -- the
augmentation changed training. The epoch-5 dilution is a real observation (the
model adapts to the corruption within a few epochs), not a confound; the
matched-epoch instrument is sound, unlike the `keep` round's final-value one.

Owning the pattern, because it is now twice: **a void check earns its keep only
when the treatment effect might be near zero, and here it never was.** dR -0.009
with 2/8 seeds, replicated across two poolings and two seed sets, is many times
the +-0.01 noise floor. Certifying "the intervention did something" through the
training loss was redundant from the start, and writing a gate that is then
overridden twice makes the gate decorative. The overrides are defensible only
because both moved a result from "decides nothing" to "loses", which cannot
manufacture a win; the same move in the other direction would be the selection
this project keeps catching. Future rounds: put the void check on the
*intervention's own mechanism* (did the masked positions actually change, are the
adapter weights non-zero), not on a downstream training statistic.

## Side result: the control recipe replicates at fresh seeds

`e250_unionnc_ctl_v5` was trained only to be a clean control, and it reproduces
`e250_unionnc_corr_v3` closely at eight seeds that had never been run:

| f32 | family R | single-seed mean R | rho |
|---|---:|---:|---:|
| `corr_v3` (seeds 58-65) | 0.8129 | 0.7978 | 0.7105 |
| `ctl_v5` (seeds 134-141) | 0.8132 | 0.7994 | 0.7193 |

The recipe's numbers were not a lucky draw. This is worth more than it cost to
learn: every delta in this campaign is measured against that recipe.

## What is left, and why it was not started

`ligand_only_masking` (a P(ligand | pocket) MLM) is implemented and has never
been run. It is a *pretraining* change, not a head-time input change, so today's
three results lower its prior without settling it. The prior art that bears on it
directly is stronger: changing the backbone's objective is a family that already
retired -- `mlm_e250lig3_lo` improved val loss 0.8009 -> 0.7867 and moved affinity
the wrong way (dR -0.0040, drho -0.0237, 1/4 seeds each).

Cost, measured rather than guessed: `mlm_e250lig3`'s checkpoint mtimes give
~2 h/epoch on `node_f` (03:02 start, e00 05:05, e01 07:09, e02 09:12 on
2026-08-18), so ~6.2 h ~= 6.2 pt for the backbone alone, plus 3.2 pt of treatment
heads and 3.2 pt for the fresh control this project's rules require -- **~12.6 pt
against 11.22 remaining.** It also cannot join the reported mix: all 36 of those
arms sit on `mlm_e250lig3`, so a new backbone can only be judged as its own
family.

## Standing

Refuted; family closed. Reported configuration stays the pre-registered 36-model
mix (R 0.8229 / rho 0.7316). Ledger 188.78/200, 11.22 pt left.
