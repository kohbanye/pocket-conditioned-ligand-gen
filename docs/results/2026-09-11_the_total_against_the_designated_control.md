# The whole placement stack against the control it should be judged by

`noref` in this repository means **no refiner, no projection**
(`arm_verdict.py`, `compare_atom_order_arms.py`), not "no reference ligand". The
designated control for a placement claim is therefore the arm with no placement
repair at all: `canon100_pctx` — same tokenizer, same causal LM, no refiner, no
MLM, no iterative decode.

Every comparison in this session had been against an arm that *already* had the
refiner, which answers "does this one component help" but never "what is the
whole thing worth". Clean 79 basis.

| | `canon100_pctx` (noref) | best (refiner + MLM + clash order + pruning) | difference |
|---|---|---|---|
| **vina_score** | +3.618 | **−1.403** | **−4.276** (p=1.6e-13, 89% of targets) |
| vina_min | −4.037 | −4.771 | −0.734 |
| **score − min** | 7.655 | **3.368** | −4.287 |
| PoseBusters | **0.779** | 0.730 | −0.049 |
| strain energy | **326** | 437 | +111 |
| **unique SMILES** | **0.791** | 0.677 | **−0.110** (p=7.7e-06, worse on 77%) |
| internal diversity | 0.778 | 0.744 | −0.012 (p=0.13, n.s.) |
| validity | 0.989 | **1.000** | +0.011 |
| heavy atoms | 22.72 | 22.03 | −0.69 |

**The placement stack is worth 4.276 kcal**, and what it does is exactly what it
is named for: `score − min` — the distance from the pose to its own local
optimum — is more than halved, 7.655 to 3.368.

## Correction: there IS a diversity cost

This session reported "no diversity cost" for clash-ordered decoding and for
pruning. That was measured against `pctx + MLM (confidence)`, and against that
baseline it is true (+0.003 and +0.010 on unique SMILES).

Against the **designated** control it is not. The stack costs **11 points of
unique SMILES** (0.791 -> 0.677, p=7.7e-06). Internal diversity moves only
−0.012 and is not significant, so the loss is in *repeats*, not in chemotype
spread: the refiner, the MLM re-decode and the clash-ordered re-decode all pull
different draws toward the same solutions.

That is the number the fairness rule asks for, and it only appears against the
right baseline. The earlier statement was true of its comparison and wrong as a
claim about the method.

## What it costs, stated once

**+4.276 kcal of score, for 11 points of unique SMILES, 0.049 of PoseBusters and
111 of strain — at 0.69 fewer heavy atoms and higher validity.**
