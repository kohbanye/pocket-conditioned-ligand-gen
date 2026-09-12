# Both stacks, each against its own noref control

`noref` = no refiner, no projection, no MLM, no iterative decode. Clean 79.

| arm | score | min | score−min | PB | strain | unique SMILES | heavy |
|---|---|---|---|---|---|---|---|
| **dfs** noref (`dfs_full_noref`) | +3.394 | −4.688 | 8.082 | 0.525 | 876 | 0.656 | 22.73 |
| **dfs** best (MLM `+` clash `+` pruning) | **−1.308** | −4.759 | 3.451 | 0.515 | **0.691** ← | 870 | 22.07 |
| **pctx** noref (`canon100_pctx`) | +3.618 | −4.037 | 7.655 | **0.779** | **326** | 0.791 | 22.72 |
| **pctx** best (clash+geom `+` pruning) | **−1.403** | −4.771 | 3.368 | 0.730 | 437 | 0.677 | 22.03 |

| stack | total gain vs its own noref | p | better on |
|---|---|---|---|
| **dfs** | **−5.127** | 7.8e-13 | 87% |
| pctx | −4.276 | 1.6e-13 | 89% |

## The method is worth more on dfs, and costs nothing there

| cost | dfs | pctx |
|---|---|---|
| unique SMILES | **+0.035 (improves)** | −0.114 |
| PoseBusters | −0.010 | −0.049 |
| strain | −6 | +111 |
| heavy atoms | −0.66 | −0.69 |

Scores are effectively tied (−1.308 against −1.403, 0.095 apart) and the dfs
ceiling is 2.0 kcal higher. **Report dfs.**

## Correction: the diversity cost is the stack's, not the method's

This session reported the placement stack as costing **11 points of unique
SMILES**. That is true on pctx (0.791 -> 0.677) and false on dfs, where the same
steps *raise* it (0.656 -> 0.691). The method does not collapse variety; the
pctx stack does, and it starts from a much higher base.

The same holds for chemistry. pctx's PoseBusters advantage (0.779 against 0.525
at noref) is a property of its **baseline**, and it comes with a tokenizer
ceiling 2.0 kcal lower. Its molecules look better and cannot be placed as well.

## What to report

- **Score and headroom: the dfs stack.** −1.308 against its noref's +3.394, a
  **−5.127 kcal** gain, with no diversity or chemistry cost and 4.2 kcal of
  ceiling left.
- **PoseBusters in absolute terms: pctx still wins that column** (0.730 against
  0.515) and should be said so if the table carries it — but not as the arm the
  score claim rests on.
