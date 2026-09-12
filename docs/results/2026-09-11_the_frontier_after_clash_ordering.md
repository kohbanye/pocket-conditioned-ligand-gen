# Where every arm sits after clash-ordered decoding

All on the canonical 100, scored through one evaluation path, `--dock-modes score`.

| arm | vina (median) | vina (mean) | PB | strain | QED | SA | heavy |
|---|---|---|---|---|---|---|---|
| deployed `refoverlap_t0` | −0.504 | +1.303 | 0.572 | 859 | 0.412 | 4.06 | 22.31 |
| deployed `+` pruning | −1.240 | −0.796 | 0.584 | 819 | 0.438 | 3.99 | 21.34 |
| deployed `+` MLM (confidence) | −0.494 | +1.421 | 0.605 | 756 | 0.419 | 3.95 | 22.30 |
| deployed `+` MLM (clash) | −1.224 | −0.011 | 0.553 | 865 | 0.406 | 4.12 | 22.30 |
| pctx `+` MLM (confidence) | −0.005 | +1.704 | **0.805** | **346** | 0.439 | **3.55** | 22.30 |
| pctx `+` clash, wall only | −1.246 | −0.636 | 0.718 | 503 | 0.419 | 3.88 | 22.28 |
| pctx `+` clash `+` bond | −1.164 | −0.417 | 0.740 | 465 | 0.424 | 3.84 | 22.28 |
| pctx `+` clash `+` geometry | −1.156 | −0.239 | 0.759 | 432 | 0.424 | 3.80 | 22.28 |
| **pctx `+` clash+geom `+` pruning** | **−1.623** | **−1.271** | 0.767 | 418 | 0.438 | 3.79 | 21.66 |
| reference ligand | −6.806 | | | | | | 22 |

**The best arm beats the deployed one on every axis**: score −0.504 -> −1.623,
PoseBusters 0.572 -> 0.767, strain 859 -> 418, QED and SA both better. The only
cost is 0.65 heavy atoms (21.66 against the reference's 22). The gap to the
reference narrows from 6.30 to 5.18 kcal, **18%**.

## Clash ordering carries across stacks; the chemistry cost does not

| stack | clash rate before | after | score gain (paired) | PB before | after |
|---|---|---|---|---|---|
| pctx | 16.6% | 9.3% | **−1.161** (p=2e-16, 90%) | 0.805 | 0.759 |
| deployed (dfs) | 16.6% | 9.8% | **−0.955** (p=5.5e-15, 89%) | 0.605 | 0.553 |

The mechanism is the same size on both. The chemistry cost is not: on the pctx
stack the geometry windows hold PoseBusters at 0.759, on the deployed stack the
same windows leave it at 0.553, **below the arm they started from**. The reason
is in the baseline, not the method — the deployed stack's molecules already have
a broken bond in 19.9% of cases against pctx's 6.2%, so there is far less good
geometry for a constraint to preserve.

**So the arm to report is the pctx one.** The deployed stack scores marginally
better before pruning (−1.224 vs −1.156) and is far worse everywhere else.

## What each step is worth

| step | paired score | what it costs |
|---|---|---|
| MLM with confidence ordering | +0.019 (p=0.37) | nothing; PB `+`0.030, strain −16 |
| `+` clash ordering | **−1.161** | PB −0.046, strain `+`86 |
| `+` clash-leaf pruning | −0.184 | 0.62 heavy atoms; PB `+`0.008 |

Pruning alone on the deployed arm was −0.954. On top of clash ordering it is
−0.184: the two are aimed at the same atoms and are sub-additive by a factor of
five, which the firing rate had already predicted (50% of molecules -> 27%).

## The rounds axis closes, and not for the reason expected

`--iter-rounds 6` against the same arm at 2, everything else identical. Clash
count falls 1.18 -> 0.78, a further 34%. What it buys:

| statistic | 6 rounds vs 2 |
|---|---|
| **paired per-target median** | **−0.032** (p=3.8e-11, better on 57%) |
| median over targets | −1.156 -> −1.379 |
| mean over targets | −0.239 -> −0.916 |

All three are true and they say different things. The typical target barely
moves; the aggregate statistics shift because a minority of targets improve a
lot. **The paired median is the effect**; quoting the 0.677 as the gain would be
quoting the difference of two summaries.

So the relationship between clash count and score is strongly non-linear: **the
first 45% of clash reduction bought 1.16 kcal, the next 34% bought 0.03.** Six
rounds also costs chemistry (PB 0.759 -> 0.747, strain 432 -> 457). **Two rounds
stays the default.**

### Why it is not a reachability limit

Re-running the codebook probe on the atoms this arm still leaves in the wall:

| | residual (clash+geom) | original (deployed) |
|---|---|---|
| atoms with **no** escape at all | **23%** | 16% |
| escapes available, median | 7 | 8 |

The residual is enriched in unreachable atoms but 77% of them could still be
moved. The axis is not blocked — **what is left is simply not worth much.** That
is the closure, and it is a different statement from "we cannot fix more".
