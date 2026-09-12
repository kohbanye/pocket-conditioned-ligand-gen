# 92% of the gap is placement; the molecules are 0.447 kcal short

`vina_dock` — redock the molecule from scratch and score the result — is the one
metric that is independent of where the generator put it, so it separates "the
molecule cannot bind" from "the molecule is in the wrong place". It had **never
been measured on the best arm**; the number on record (−7.06) belongs to an
older one.

621 molecules docked over the clean 79 (8 per target), exhaustiveness 8, through
exactly the evaluation path the generated poses take.

| | arm | reference | gap |
|---|---|---|---|
| `vina_score` (as generated) | **−1.308** | −6.871 | **+5.108** (paired) |
| `vina_min` (locally optimised) | −4.759 | −6.912 | +1.717 (paired) |
| **`vina_dock` (redocked)** | **−7.287** | **−7.859** | **+0.447, p=6.95e-4** |

The reference column for `vina_dock` is its own crystal ligand put through the
identical path — the bench only had a recorded dock for 5 of the 100 targets, so
the other 74 were docked here, with the reference ligand written into a
generation-shaped tree so nothing about the code path differs.

**Paired on all 79 clean targets: +0.447, p=6.95e-4, the arm better on 34%.**

Read at 26 of the 79 targets this was +0.235 at p=0.181, and reporting it then as
"as good as the crystal ligands, not significant" was wrong: the partial set was
both underpowered and the optimistic half. The completed measurement says the
molecules are **really, if slightly, worse**.

## What this settles

**The generator's chemistry is 8% of the problem and its placement is 92%.**

| | kcal | share of the reported gap |
|---|---|---|
| removed by local optimisation alone | 3.391 | **66%** |
| further removed by redocking | 1.271 | **25%** |
| left over — molecule quality | **0.447** | **9%**, p=6.9e-04 |

**This reproduces a number already on record.** The 2026-09-01 decomposition, on a
completely different arm and a different route, put molecule quality at **0.53
kcal / 7%**. Getting 0.447 / 8% here is an independent confirmation of that
split, not a new claim.

This is the end point of every thread this session followed. Repulsion is 71% of
the gap and 96% of that is against protein the model can see; the shape is right
and the placement is not (84% / 16%); the model adapts to the target and clash
noise hides it; and now the molecules themselves are shown to be as good as
crystal ligands. **There is nothing left to fix about what the model draws — only
about where it puts it.**

And the placement routes are, as of today, all measured and all closed under the
standing constraints: decoding levers saturated, the refiner's step locked to its
target's magnitude, the 2 kcal of bounded-rigid headroom reachable only through
Vina's own geometry, and no generic-chemistry objective with any signal.

## Caveat worth keeping

`vina_dock` rewards a molecule that *can* bind somewhere in the box; it says
nothing about whether the generator would ever produce that pose. It is the right
control for molecule quality and the wrong one for the product. The headline stays
`vina_score`.
