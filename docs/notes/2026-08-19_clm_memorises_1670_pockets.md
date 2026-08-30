# The CLM memorises its 1670 training pockets (2026-08-19)

`clm_e250lig3_fullft` ends at **train loss 1.32-1.37, val loss 5.46** -- with
`--mask-prompt` the loss is over ligand tokens only, so that is perplexity 3.7
on training pockets against 235 on held-out ones. A 298M model cannot overfit
4.09B tokens in two epochs, so the number needed explaining.

## It is not the val set

Same packed format, no malformed docs in a 300-doc sample, pocket 184 atoms vs
train's 210 and ligand 21 vs 26 -- slightly smaller, same kind of data. The
CrossDocked tokenizer splits **by pocket**, so val pockets are genuinely unseen.

## It is the corpus

`data/lm_tokens_e250lig3_full` is CrossDocked only: 16.5M docs over ~**1670
distinct pockets** (1724 fold0-train pockets minus the 54 that overlap the
generation benchmark), i.e. ~9900 docs per pocket. Those docs are largely the
same ligand redocked into the same pocket, so memorising the pocket -> ligand
association is the cheapest way to drive the loss down, and the model takes it.

This is exactly the failure the generation table sees: the 97 evaluation pockets
are unseen, and on unseen pockets the model places atoms badly -- clash-free
0.124 against the tokenizer's own 0.800 round-trip floor.

**Annealing the LR does not fix this** and was the wrong diagnosis (the cosine
was also truncated by walltime, which is true but secondary). A lower LR fits
the 1670 pockets harder.

## What was sitting unused

| corpus | mode | systems | train docs | in pretrain? | in finetune? |
|---|---|---|---|---|---|
| `..._geom` | ligand only | 1.47M confs | 11.6M | yes | no |
| `..._plinder_protein` | protein only | 288k | 2.30M | yes | no |
| `..._biolip` | **complex** | **127,210** | 125k | **no** | **no** |
| `..._plinder_complex` | **complex** | **213,676** | 426k | **no** | **no** |
| `..._full` (CrossDocked) | complex | ~1670 pockets | 16.5M | no | yes |

The pretrain mixed ligand-only with protein-only; the finetune used CrossDocked
alone. The two **complex** corpora -- ~340k distinct systems, some 200x the
pocket diversity of CrossDocked -- were never used in either stage.

## Leak status

Both were built against `data/eval_holdout_pdbs.txt`, which at the time held
1707 ids and covered 103 of the generation benchmark's 104 receptors. The gap
was `2pqw`, and BioLiP has one entry for it. **No reported number is affected**
-- neither corpus was used by any trained model. `prolit.data.holdout` has been
fixed (`SBDD_BENCH_PDBS` still listed only the three original targets, so PDB-id
keyed corpora were effectively unprotected for the 97-target set) and the list
regenerated to 9329 ids, now including sibling receptors of every eval pocket.
Both corpora are being re-tokenized against it.
