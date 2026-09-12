# The tokenizer ceiling, and why it reverses the stack recommendation

The crystal ligand, encoded and decoded through each tokenizer and scored the
same way every generated pose is scored. No language model is involved: this is
what a **perfect** model could score.

| tokenizer | round-trip score | round-trip min | cost vs crystal | RMSD | bond MAE |
|---|---|---|---|---|---|
| **pctx** (what the best arm uses) | −3.519 | −4.971 | **3.352** | 0.876 | 0.201 |
| **dfs / `vq_ord_buriedfirst`** | **−5.496** | **−6.549** | **1.375** | **0.402** | **0.058** |
| crystal reference | −6.871 | −6.912 | — | — | — |

Clean 79 basis. **The dfs tokenizer's ceiling is 1.977 kcal higher**, and its
bond lengths come back 3.5x more accurately.

## What this does to the gap

The remaining 5.468 kcal of the best arm was being read as a placement problem.
It is not, in the main:

| component | kcal | share |
|---|---|---|
| **the tokenizer's round trip** | **3.352** | **61%** |
| everything the model and the placement do | 2.116 | 39% |

## Headroom, which is the number that should drive the choice

| stack | arm | score | its ceiling | headroom |
|---|---|---|---|---|
| pctx | best (clash+geom+pruning) | **−1.403** | −3.519 | 2.115 |
| dfs | deployed `refoverlap_t0` | +0.470 | −5.496 | 5.966 |
| dfs | `+` pruning | −0.875 | −5.496 | 4.621 |
| dfs | `+` MLM `+` clash order | −0.794 | −5.496 | 4.702 |
| | FLOWR (recorded, same evaluator) | −5.730 | | |

**The pctx stack cannot reach a competitive score.** Its ceiling is 2.2 kcal
below FLOWR, so a perfect language model on it still loses. The dfs stack's
ceiling is 0.23 *above* FLOWR, and it has 4.7 kcal of headroom left.

**This reverses a recommendation made twice in this session.** The pctx stack
was preferred for chemistry (PoseBusters 0.759 against 0.553, strain 432 against
865) and it is ahead on score today (−1.403 against −0.794). Neither survives
the ceiling: a stack that cannot reach the number is not the one to report the
number from.

## What follows

Carry the clash-ordered decode and the pruning onto the **dfs** stack. Both
already exist there (`canon100_dfs_mlm_clash` at −0.794); only the pruning step
is missing.

The chemistry cost on the dfs stack is real and must be reported with it —
PoseBusters 0.553 against pctx's 0.759, because that stack's molecules already
carry a broken bond in 19.9% of cases before anything is done. That is a
separate defect and the ceiling measurement points at its cause: bond MAE 0.058
for dfs vs 0.201 for pctx says the *tokenizer* is not what breaks them there.
