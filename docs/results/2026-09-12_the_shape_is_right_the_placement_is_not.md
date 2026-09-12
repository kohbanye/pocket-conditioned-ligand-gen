# The molecules are the right shape. They are badly placed.

Five routes this session converged on "the generated molecule's conformation does
not fit the pocket": interior atoms carry 59.5% of the repulsion, the per-atom
clash rate doubles by 20 heavy atoms, every leaf-level tool saturated, torsion
projection is three times worse than rigid, and the pose axis needs a teacher.
**None of them tested the claim.** This does, and it is false.

## The measurement

`rigid_pocket_fit` minimises summed squared van der Waals overlap over a bounded
rigid move — pure geometry, no energy function. Apply it to each molecule and
report the repulsion that **survives**. What survives is shape no placement can
fix. The crystal ligand is the control: same kind of molecule, same pocket.

474 molecules over 79 clean targets, Vina repulsion in weighted kcal:

| | before | after the best bounded move | removed | shift |
|---|---|---|---|---|
| **reference** | 1.834 | **1.748** | **4.7%** | 0.29 A |
| **generated** | 5.755 | **2.368** | **58.9%** | 0.83 A |

**The crystal pose is already at its rigid optimum** — 4.7% is noise, and that is
the sanity check the whole measurement rests on. The generated pose is not: a
**0.83 A** rigid move removes **2.63 kcal**, well over half its repulsion.

| | |
|---|---|
| residual excess over the reference | **+0.620** |
| the repulsion gap before any fit | +3.921 |
| **share of the gap that is shape** | **16%** |
| **share that is placement** | **84%** |

## By size, where shape does start to matter

| heavy atoms | generated: before | after | removed | reference: after |
|---|---|---|---|---|
| 0–20 | 4.424 | 1.487 | 66.4% | 1.304 |
| 20–25 | 7.194 | 2.126 | 70.4% | 1.335 |
| 25–30 | 7.203 | 2.650 | 63.2% | 2.270 |
| **30+** | 7.445 | **4.492** | **39.7%** | 2.721 |

Below 30 atoms a rigid move recovers two thirds of the repulsion and leaves the
molecule near the reference's own residual. Above 30 it recovers 40% and leaves
1.77 kcal more than the reference. **So the conformational story is true only for
the largest molecules**, and they are a minority.

## The radii mistake that had to be fixed first

Run with Bondi radii, the fit made the repulsion **worse** — the reference went
1.834 -> 6.633, which a fit that includes the identity transform cannot do to its
own objective. The objective was not the readout: Bondi radii zero the overlap
0.4 A inside the surface Vina charges from, so the fitter settles with every atom
pressed too deep. `generate_ligands_3d.py` documents exactly this; I walked into
it anyway. Fitter and readout must share radii or the number is meaningless.

## What this licenses, and the fairness question it raises

It does **not** license quietly adding the fit as an arm. Minimising van der
Waals overlap on **Vina's own X-S radii** is close to optimising Vina's repulsion
term directly, and `prune.py` records that ranking by that term was rejected as
circular — "it selects atoms with the function the result is scored by, the same
defect that disqualified Vina terms as a refiner teacher". With Bondi radii the
operation is generic chemistry and does not work (above). That tension is a
judgement for whoever owns the paper, not something to bury in an arm.

What it does establish, independent of that call:

**The deployed placement is leaving 2.63 kcal of *bounded rigid* headroom on the
table** — inside the motion class the refiner already uses and the record already
accepts (`2026-08-30_the_rigid_part_was_free.md`: the SE(3) projection is ML and
fair, and buys 4.40 kcal). The refiner is not finding the steric optimum of the
motion it is allowed to make. Improving *that* needs no new class of operation
and no Vina radii — it is the refiner's own objective that is wrong, which is the
open item `2026-09-10_the_refiner_target_is_half_noise.md` already names.

## Correction: it is not gaming, and it is worth about 2 kcal

Two further controls change the reading above.

**It moves away from the crystal ligand.** Centroid distance to the reference
goes 1.089 -> 1.388 A, paired **+0.227** (p=6.5e-19), closer on only 29% of
molecules, and the repulsion it removes is uncorrelated with any distance closed
(rho +0.016, p=0.74). I read that as gaming the scorer. **That inference was
wrong**: the generated molecule is a *different molecule*, so there is no reason
its best position should be the crystal ligand's centroid.

**The five-term total is the test, and it improves.** Same implementation as the
validated decomposition, weighted kcal:

| | repulsion before | after | **five-term before** | **after** |
|---|---|---|---|---|
| **reference** | 1.834 | 1.748 | **−10.795** | **−10.126** |
| **generated** | 5.755 | 2.368 | **−4.936** | **−7.287** |

paired, per molecule:

| | median | p | improved on |
|---|---|---|---|
| generated, repulsion | −2.631 | 2.8e-50 | 82% |
| **generated, five-term total** | **−1.971** | **1.3e-43** | **77%** |
| **reference, five-term total** | **+0.449** | | **10%** |

Of the 2.631 kcal of repulsion the move removes, the attractive terms give back
only 0.660 — **1.971 survives in the full intermolecular energy**. And the same
operation makes the **crystal pose worse**, on 90% of targets. A uniform
scoring-function artefact would help both; this helps only the poses that are
badly placed, which is what a real correction looks like.

## The decision this needs, which is not mine

`rigid_pocket_fit` on scoring radii minimises `sum max(0, r_i + r_j - d)^2`,
which **is** Vina's repulsion term up to its weight and cutoff. `prune.py`
records that ranking by that term was rejected as circular, and the same rule
disqualified Vina terms as a refiner teacher.

Against that: the five-term total improves by 1.971 of the 2.631, and the
reference control moves the other way. Neither is what term-specific gaming
produces. Whether the rule should bite here is a call about what the paper can
claim, so it is recorded rather than run:

* **worth**: about 2 kcal of intermolecular energy, the largest single lever
  measured in many iterations, on a 0.83 A rigid move that changes no bond
  length or angle;
* **cost**: the objective is one of the five terms the result is scored by;
* **the fair alternative**: the same 2 kcal sits inside the refiner's own motion
  class, and a refiner that found it would need no Vina term at inference — what
  it would need is a better training target than the one
  `2026-09-10_the_refiner_target_is_half_noise.md` describes.

Nothing here is in an arm. The number is a measurement of how much the deployed
placement leaves behind, and it stands whichever way the fairness call goes.

## There is no fair version of the 2 kcal

The obvious escape from the circularity is to minimise overlap on a radius set
that owes Vina nothing. Bondi radii are the standard crystallographic van der
Waals set. Same fit, same bounds, same readout — the validated five-term energy:

| fit radii | arm | five-term paired | p | improved on |
|---|---|---|---|---|
| **Vina X-S** (circular) | generated | **−1.971** | 1.3e-43 | 77% |
| Vina X-S | **reference** | +0.449 | 2.3e-07 | 10% |
| **Bondi** (fair) | generated | **−0.399** | 1.8e-04 | 55% |
| Bondi | **reference** | **+3.157** | 3.1e-12 | **4%** |

**The fair objective buys 0.4 kcal and destroys the crystal pose by 3.2.** It even
makes the generated molecules' repulsion *worse* (+0.562) while moving them 0.89 A.
Bondi radii are small enough that zero overlap is easy and the objective goes
flat, so the fit wanders with nothing to follow.

The asymmetry is the whole answer:

* on **Vina's** radii the operation helps generated poses (−1.971) far more than
  it hurts crystal ones (+0.449) — an asymmetry a uniform artefact cannot produce,
  so it is detecting something real about pose quality;
* on **Bondi** radii it hurts crystal poses eight times more than it helps
  generated ones — no signal at all.

**The only geometry that separates a good pose from a bad one here is Vina's
own.** So the 2 kcal cannot be bought with generic chemistry, and the fairness
question from the previous section resolves against it: there is no non-circular
steric objective that finds the move.

What survives: the measurement that the deployed placement is 2 kcal from the
optimum *of Vina's geometry*, and that a refiner would have to learn pose quality
from crystal structures rather than from any steric rule to reach it.

## And the refiner cannot be pushed into it either

The section above left "train a better refiner" as the fair route to the same
2 kcal. `2026-09-09_the_step_follows_the_scale_of_the_target.md` already closed
that, with a mechanism: **the refiner's step is a roughly fixed fraction of its
training target's magnitude** (0.75/1.96 = 0.38 for the control, 0.70/1.75 = 0.40
for the arm). Four independent attempts to lengthen it:

| what was changed | outcome |
|---|---|
| α-scale the output | flat to 1.5x, collapses at 2x |
| more rounds | null, p=0.52 |
| a better teacher | **−0.276 worse**, p=5.7e-09 |
| cut the corruption tail | **−0.168 worse**, p=3.7e-05 |

Lengthening the step needs a larger training target, and every arm that changed
the target made the result worse.

**So the chain is complete.** The 2 kcal of bounded-rigid headroom is real and
measured; it is reachable only by minimising Vina's own repulsion geometry; the
generic-chemistry version of that objective has no signal; and the refiner's step
cannot be pushed into it without a larger target, which is separately measured to
hurt.
