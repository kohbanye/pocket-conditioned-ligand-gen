# The head moves weight off the last layer when allowed to — and that makes it worse

2026-09-15 ~21:45 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_layer_mix_learnable.md` before the options existed,
including the void condition. Applied by `layermixlr_verdict.py`. All 24 jobs
exited 0.

## The premise held this time

Round 1 left the mixing weights uniform, so it never tested a learned choice.
With a last-layer initialisation (0.808 on the last state) and a dedicated
learning rate for those 14 scalars, **the weights moved in every seed**: mean
|change| 0.0318 against the pre-registered void threshold of 0.02.

Where they moved to, averaged over 8 seeds:

| state | L0 (emb) | L1 | L2-L7 | L8-L10 | L11-L12 | **L13 (last)** |
|---|---|---|---|---|---|---|
| weight | 0.048 | 0.043 | 0.024-0.034 | ~0.033 | 0.023-0.028 | **0.589** |

Per seed the last layer lands between 0.437 and 0.706. The head keeps the last
layer as its main component and spreads the rest mostly onto the shallow states
and the L8-L10 band.

## Result (f32 is the pre-registered basis)

| | n | R | between | within | rho |
|---|---|---|---|---|---|
| **TREAT** `e250_unionnc_layermix_lr` | 8 | 0.8064 | 0.3733 | 0.4331 | 0.7018 |
| **CTRL** `e250_unionnc_corr_v3` (last layer only) | 8 | 0.8129 | 0.3761 | 0.4369 | 0.7105 |

**ZL1: dR -0.0065 FAIL. ZL2: drho -0.0088 FAIL. LEVER DOES NOT WORK.**
At f16: dR -0.0066, drho -0.0018.

## The two rounds together answer the question

| last-layer weight | R (f32, 8 seeds) |
|---|---|
| 1.000 (last layer only) | **0.8129** |
| 0.589 (learned mix, this round) | 0.8064 |
| 0.071 (uniform mix, round 1) | 0.7957 |

Monotone: **the more of any other layer is mixed in, the worse CASF gets.** For
this representation the last layer is the right place to read, and that is now a
measured answer rather than an untested default. The head does move weight away
from it when it can -- it is minimising a validation loss, and that movement does
not transfer to CASF.

## What it closes

With this, the "use the MLM representation better" axis is systematically
covered: layer choice (these two rounds), the four pooling readouts
(`ligand`/`balanced`/`allatoms`/`concat`), learned attention pooling,
ligand->pocket cross-attention and GenScore-style pairwise sums (recorded as
built and measured in `_pool`'s docstring), encoder fine-tuning (dev diagnostic),
and continued MLM pretraining (validation improves, affinity worsens).

## Recorded, not claimed

Swapping this family into the 24-model mix raises rho and lowers R: f32
0.8222 / 0.7333 against the pre-registered mix's 0.8244 / 0.7228 (f16
0.8228 / 0.7281 against 0.8249 / 0.7246). It is rho-complementary, but the R
cost is larger than the rho gain for the loop's goal, and these numbers were
seen inside a verdict for a different question.

## Cost

8 heads + 16 inferences, ~3.2 pt. Ledger 155.31/200.
