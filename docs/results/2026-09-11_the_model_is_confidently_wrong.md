# Why the validation curve has been anti-predictive: confidence, not accuracy

Four times in this project a learning metric has pointed the opposite way to the
score. This is the mechanism, and it is the same one each time.

`clm_dfs_full_scratch`, 400 held-out documents, ligand-block positions only:

| checkpoint | cross-entropy | **model's own entropy** | top-1 | top-10 |
|---|---|---|---|---|
| `lm-e01` (best monitored loss) | 3.856 | **3.110** | **24.3%** | 57.5% |
| **`last.ckpt`** (what generation uses) | **8.446** | **0.925** | 22.6% | 49.3% |

Uniform over the 8199-token vocabulary would be 9.01.

**`last.ckpt` is not diverged. It is extremely confident and 77% wrong.** Its
predictive entropy of 0.925 is about two-and-a-half effective choices per token,
and it names the data's own code 22.6% of the time — within 1.7 points of the
checkpoint whose cross-entropy is less than half as large. The cross-entropy gap
is overconfidence, not ignorance: log-loss punishes a confident miss enormously.

## It explains every observation at once

1. **Its codes assemble 100% of the time** while genuinely near-uniform codes
   assemble 0.6%: a peaked model is self-consistent.
2. **Its cross-entropy is terrible**: it is sure and wrong, which is the worst
   case for log-loss and says nothing about coherence.
3. **It generates 5.359 kcal better than the low-loss checkpoint**: sampled at
   temperature 0.7, a diffuse distribution (entropy 3.110) wanders and a peaked
   one does not.

Training past epoch 2 made the model **more confident without making it much
more accurate**. That is ruinous for cross-entropy and helpful for sampling,
which is exactly the anti-correlation this project kept rediscovering.

## The selection rule this implies

**Cross-entropy conflates accuracy with calibration and must not be used to pick
a checkpoint for a sampler.** Nor does top-1 rescue it: `lm-e01` is 1.7 points
*more* accurate and generates 5.4 kcal worse. The quantity that tracked
generation quality here was **predictive entropy**.

That is one observation on two checkpoints, so it is a hypothesis about what to
measure next, not a rule to select on — and given the record, the only safe rule
remains: **generate and score.**

## Correction

An earlier note in this session read the cross-entropy of 8.14 as a diverged
model and called it "possibly the biggest defect in the pipeline". It is not a
defect at all; `last.ckpt` is the better generator and the right choice. The
reasoning that led there was: high CE -> near-uniform -> should not produce
molecules. The middle step was wrong, and the entropy measurement is what shows
where.
