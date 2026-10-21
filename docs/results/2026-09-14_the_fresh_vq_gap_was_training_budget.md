# The fresh tokenizer's geometry gap was training budget

2026-09-14. A tokenizer trained from scratch at 30,000 steps loses badly to the
shipped `vq_e250_lig3`. At 120,000 steps -- same config, same seed, same packed
cache, only `--steps` changed -- it closes most of that gap and passes the
shipped tokenizer on ligand geometry. Held-out evaluation, 492 validation
complexes, `experiments/affinity_loop_20260914/fresh_vq_eval492_budget`.

This matters beyond curiosity: the reason to be able to retrain a tokenizer
from scratch is to retrain it on a **leak-cleaned cohort**
(`2026-09-14_leak_sensitivity_and_the_gign_bar.md`). Until now the from-scratch
route looked like it cost 2.2x on protein geometry, which would have made the
cleaned-cohort retrain unaffordable in quality. It does not.

## All five models, same inputs, own normalization each

`original` = `vq_e250_lig3` epoch 237. `geom` / `geombal` = fresh runs from
`packed_train_eligible` (`geombal` adds `balanced_chem_loss`). Every model is
paired with the normalization statistics it was trained under -- mismatching
them does not raise, it silently returns plausible but wrongly scaled
coordinates.

### protein

| metric | original | geom 30k | geom 120k | geombal 30k | geombal 120k |
|---|---:|---:|---:|---:|---:|
| coord RMSD (A) | **0.2962** | 0.5700 | 0.4292 | 0.6095 | 0.4011 |
| element accuracy | 0.9999 | 0.9989 | 1.0000 | 0.8730 | 0.9996 |
| charge accuracy | 1.0000 | 1.0000 | 1.0000 | 0.9998 | 1.0000 |
| effective codes | 4249 | 5729 | 5891 | 5471 | 4777 |

### ligand

| metric | original | geom 30k | geom 120k | geombal 30k | geombal 120k |
|---|---:|---:|---:|---:|---:|
| coord RMSD (A) | 0.5041 | 0.5985 | **0.4837** | 0.6708 | 0.5187 |
| interface distance MAE (A) | **0.2349** | 0.3503 | 0.2705 | 0.3777 | 0.2792 |
| contact recall @4.5 A | **0.8616** | 0.7885 | 0.8357 | 0.7725 | 0.8406 |
| non-contact FP @4.5 A | **0.0048** | 0.0077 | 0.0059 | 0.0080 | 0.0063 |
| element accuracy | 0.9920 | 0.9698 | 0.9902 | 0.9706 | **0.9947** |
| charge accuracy | 0.9879 | 0.9881 | 0.9883 | 0.9851 | **0.9981** |
| effective codes | 3494 | 3543 | 3733 | 3199 | 3713 |

## What 4x the steps bought

| | 30k -> 120k |
|---|---|
| protein coord RMSD | geom -0.1408, geombal -0.2084 |
| ligand coord RMSD | geom -0.1148, geombal -0.1521 |
| interface MAE | geom -0.0798, geombal -0.0985 |
| contact recall | geom +0.0472, geombal +0.0681 |
| protein element accuracy (geombal) | 0.8730 -> 0.9996 |

**30k was simply undertrained**, and `geombal`'s protein element accuracy of
0.873 at 30k -- against 0.9996 at 120k -- is the clearest sign of it. 30k x
B128 is ~3.8M complex-observations against the shipped tokenizer's ~47.7M, so
even 120k remains ~4x short.

## Where it stands against the shipped tokenizer

**Passes it:** ligand coordinate RMSD (0.4837 vs 0.5041), ligand element
accuracy (0.9947 vs 0.9920), ligand charge accuracy (0.9981 vs 0.9879). The
charge result is consistent with the earlier one-boolean `balanced_chem_loss`
A/B, which recovered charged-atom accuracy from 0.0 to 0.984.

**Still behind it:** protein coordinate RMSD (0.4011 vs 0.2962), interface
distance MAE (0.2705 vs 0.2349), contact recall (0.8406 vs 0.8616),
non-contact false positives (0.0059 vs 0.0048).

So the remaining deficit is **protein-side and interface geometry**, not ligand
chemistry or ligand geometry. Both fresh arms are still improving steeply at
120k, so the gap is not obviously a floor.

`geombal` beats `geom` on protein coord at 120k (0.4011 vs 0.4292) while losing
at 30k (0.6095 vs 0.5700) -- the balanced loss costs early and pays later, which
is a reason not to judge a loss change at a short budget.

## Caveats

* The fresh training runs log **no validation at all**
  (`limit_val_batches=0`, `"validation_used": false` in each `complete.json`),
  so their training curves say nothing about generalization. This evaluation is
  the first held-out measurement of them.
* `seed7` (the other 30k fresh run) is deliberately excluded: it trained from
  `packed_train_development`, a different cache, so including it would confound
  budget with training corpus. All four arms here share
  `packed_train_eligible`.
* Reconstruction quality is not affinity performance. Nothing here says a
  retrained tokenizer scores better on CASF; it says a from-scratch retrain is
  not disqualified on reconstruction, which is what the cleaned-cohort plan
  needs.
* "Effective codes" rises for every fresh arm over `original` on the protein
  side (4777-5891 vs 4249). Not interpreted here.
