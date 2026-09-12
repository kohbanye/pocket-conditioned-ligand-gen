# The generator produces one pose per molecule, and the other poses are worth 1 kcal

Uniqueness is 0.69, so about a third of the 100 samples per target repeat a
SMILES. That makes a control nothing here has used: **the same molecule, sampled
twice**, holds chemistry exactly fixed and lets the pose vary on its own.

## The naive reading, and why it is circular

323 SMILES seen at least three times, over 2388 molecules and 73 clean targets:

| | |
|---|---|
| within-SMILES `vina_score` spread (max − min), median | **0.71 kcal** |
| within-SMILES `vina_score` sd, median | 0.32 |
| sd **across different molecules** in a target | **3.89** |

Read alone this says the pose is a deterministic function of the molecule. But a
repeated SMILES is a repeated *code sequence*, and codes carry chemistry and
coordinates together, so "same molecule, same pose" may just be "same tokens".

## The control: how different are the poses actually?

1937 same-SMILES pairs, heavy-atom RMSD between the two samples:

| quantile | RMSD |
|---|---|
| 25% | 0.000 A |
| 50% | 0.231 A |
| 90% | 1.054 A |

| | |
|---|---|
| pairs that are the **same pose** (< 0.1 A) | **39%** |
| pairs **genuinely differently posed** (> 1.0 A) | **11%** |

So the naive number was 39% circular — and the remaining 11% carries the finding:

| among the genuinely different poses (n=215) | |
|---|---|
| RMSD, median | 1.52 A |
| **absolute score difference, median** | **1.115 kcal** |
| pairs differing by more than 1 kcal | **55%** |

(Among the near-identical pairs the score difference is 0.000, which is the
sanity check that the two halves are being separated correctly.)

## What it means

**When the model does place the same molecule two ways, the two poses score over
a kcal apart — and it does that only 11% of the time.** Conditional on the
chemistry it has chosen, the generator has almost no pose entropy: 89% of repeats
land within 1 A of each other.

So there is real, kcal-scale headroom in pose space that sampling does not reach.
This is the pose-side quantification of what
`2026-09-11_temperature_widens_the_pool_the_ranker_cannot_reach.md` found on the
chemistry side: widening the pool is not the constraint, reaching into it is.

**Untested idea, recorded rather than run:** the MLM pass is the only stochastic
step after the codes are fixed, so k independent MLM passes over one code sequence
would give k poses of one molecule, and picking among them by the model's own
likelihood would be ML rather than an objective function. Whether the MLM's
likelihood prefers the better-scoring pose is the question that decides it, and it
is not answered here.

## The likelihood cannot pick the better pose either

The idea above was the only untested lever left, so it was tested. 200 pairs of
genuinely different poses (RMSD > 1 A) of the same molecule, each scored by the
complex MLM's **pseudo-likelihood** over the ligand block — every ligand position
masked in turn and the log-probability of the code actually there summed, because
an MLM with nothing masked can copy its input.

| | |
|---|---|
| MLM picks the better-scoring pose | **55.0%** (chance 50%) |
| binomial p | **0.179** |
| paired `vina_score`(chosen) − (other) | −0.307, p=0.042 |
| **gain over a random pick** | **−0.154 kcal** |
| the gap available in a pair | 1.217 kcal |

By how far apart the two poses are:

| RMSD | n | picks better | chosen | best available |
|---|---|---|---|---|
| 1.0–1.5 | 95 | 55.8% | −2.06 | −2.60 |
| 1.5–2.5 | 48 | 58.3% | −0.78 | −0.87 |
| **2.5+** | 57 | **50.9%** | −1.78 | −1.92 |

**At the widest separation it is exactly chance.** So the model's own likelihood
is close to blind to pose quality: generating k poses per molecule and choosing by
it would buy about **0.15 kcal** for a k-fold generation cost, with a selector
that is not significantly better than a coin.

This is the same shape as
`2026-09-11_temperature_widens_the_pool_the_ranker_cannot_reach.md` on the
chemistry side, and it follows from `2026-09-11_the_model_is_confidently_wrong.md`:
a model whose top-1 is 22.6% at entropy 0.925 has a likelihood that does not track
quality. **Widening the pool is never the constraint here; every selector that has
been tried cannot reach into it.**

**Lever closed.**
