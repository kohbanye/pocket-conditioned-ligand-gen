# Raising the temperature widens the pool; the ranker cannot reach the new part

Licensed by the entropy measurement: the LM's predictive entropy is **0.925** —
about two and a half effective choices per token — while the data's own code sits
in its top-10 **49.3%** of the time and is top-1 only 22.6%. So there are 27
points of signal between rank 1 and rank 10 that sampling at temperature 0.7 from
a peaked distribution rarely reaches. The recorded temperature work went
**downward** (0.3–0.7, for aromaticity, on the published stack); upward had never
been tried.

`canon100_dfs_t10`: the best arm's command with `--temperature 1.0` and nothing
else changed. Clean 79.

| arm | median | geometry top-1 | oracle best of 100 | min | PB | unique SMILES | heavy |
|---|---|---|---|---|---|---|---|
| temperature 0.7 | −1.308 | **−4.124** | −4.829 | −4.759 | 0.515 | 0.691 | 22.08 |
| temperature 1.0 | −1.058 | −4.002 | **−5.095** | −4.912 | 0.504 | **0.824** | 22.08 |

| quantity | paired | p | T=1.0 better on |
|---|---|---|---|
| median | +0.032 | 0.64 | 43% |
| **geometry top-1** | **+0.000** | **0.22** | 41% |
| **oracle best of 100** | **−0.147** | **0.014** | 62% |

## The mechanism worked and the outcome did not

The pool really did widen — measured before the scores came back, on 60 targets:
unique SMILES 0.625 -> 0.724, internal diversity 0.642 -> 0.683, same median size
with a wider spread. And its **best member improved** (oracle −0.147, p=0.014),
which is what the entropy argument predicted.

**The receptor-only ranker cannot find it.** Its top-1 pick is unchanged
(p=0.22), because it selects on van der Waals overlap and the molecules that
temperature 1.0 newly reaches are not better *in overlap terms*. The reported
number does not move.

This sharpens the ranking result: the geometric ranker captures 88% of the
headroom **in a given pool**. Widening the pool adds headroom it cannot reach.

## Verdict

**Keep temperature 0.7.** Down was already measured as ineffective and up buys
nothing the pipeline can deliver. The temperature axis closes.

**Co-report:** temperature 1.0 gives **+0.133 unique SMILES** (0.691 -> 0.824)
for +0.032 kcal of median (not significant), an unchanged ranked value and
−0.011 PoseBusters. If diversity is a reported column that is a favourable
trade; for score it is neutral.
