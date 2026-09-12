# The pose is 1.1 A from its own optimum, and that 1.1 A is 3.2 kcal

`min_rmsd` — how far Vina's local optimiser moves a pose — has been in every
evaluation dump all along and was never read. With the crystal ligand put
through the identical path as a control, it gives the sharpest single statement
of what is wrong.

7754 generated molecules and 79 crystal ligands, clean 79 targets. Vina's
`--local_only` moves translation, rotation **and torsions**.

| | p25 | median | p75 | p90 | kcal it gains |
|---|---|---|---|---|---|
| **generated** | 0.816 | **1.222 A** | 1.939 | 3.982 | **3.199** |
| **crystal ligand** | 0.088 | **0.136 A** | 0.201 | 0.291 | **0.017** |

Paired per target:

| | generated | reference | difference | p |
|---|---|---|---|---|
| displacement | 1.238 A | **0.136 A** | **+1.077** | 2.2e-13 |
| kcal recovered | 3.204 | **0.017** | **+3.243** | 1.9e-13 |

The generated pose moves **9x further** and gains **188x more**. Generated moves
more on **99% of targets**.

**So this is not a property of Vina's optimiser.** The crystal pose is already at
its local optimum — it moves 0.136 A and gains 0.017 kcal, which is the sanity
check the whole measurement rests on: a solved structure *is* a local minimum,
and Vina agrees that it is.

> **The generator places each molecule about 1.1 A from where that same molecule
> wants to sit, and that 1.1 A costs 3.2 kcal — 66% of the paired gap of 5.108.**

1.1 A is shorter than a bond. It costs this much because Vina's repulsion rises
steeply inside contact.

## The gain tracks the distance

| displacement | n | score | -> min | gained |
|---|---|---|---|---|
| 0.0–0.3 A | 88 | −1.22 | −1.22 | **0.00** |
| 0.3–0.6 A | 739 | −3.87 | −5.33 | 1.58 |
| 0.6–1.0 A | 2073 | −2.71 | −5.28 | 2.67 |
| 1.0–2.0 A | 3027 | −1.04 | −4.86 | 3.80 |
| **2.0 A +** | 1823 | **+0.99** | −3.87 | **4.70** |

rho(distance moved, kcal gained) = **+0.453**. The molecules that score worst are
exactly the ones furthest from their own optimum; the 88 that are already there
score −1.22 and gain nothing.

## It is a correlated displacement, not per-atom noise

The decoder's own per-atom error is 0.366 A at the median
(`2026-09-12_the_decoder_is_uniformly_imprecise_not_selectively_wrong.md`). If
those errors were independent, a 23-atom molecule's rigid displacement would be
about 0.366/sqrt(23) = **0.08 A**, not 1.1. So the 1.1 A is the **whole molecule
shifted together**, which per-atom precision cannot explain and a rigid move can
remove.

The tokenizer contributes part of it and not most: the round trip's centroid
movement is 0.191 A on record, under a fifth of the 1.1.

## How it reconciles with the rest of today

| operation | motion class | kcal found | displacement |
|---|---|---|---|
| bounded rigid fit, Vina radii | translation + rotation | 1.971 | 0.829 A |
| **Vina `--local_only`** | **+ torsions** | **3.451** | **1.222 A** |

Rigid alone is **57%** of what local optimisation finds, which matches the 66%
the record measured for the SE(3) projection of the refiner. The remaining 43%
needs torsions, and torsion projection of the refiner's output was measured to be
three times *worse* than rigid — the refiner does not know which torsions to turn.

**Nothing here reopens a closed lever.** It renames the problem precisely: not
"the poses are bad" but "the poses are one correlated Angstrom off, and that
Angstrom is 58% of the reported number".

## The direction is molecule-specific, so there is no cheap global fix

If the 1.1 A were a systematic bias — always too deep, say — a single global
nudge would collect most of it. The rigid correction's translation, projected on
the one axis with a meaning common to every target (pocket centre of mass to
ligand centroid; positive is outward, away from the protein's interior):

| | median \|shift\| | median outward | outward / \|shift\| | p | outward on |
|---|---|---|---|---|---|
| **generated** | 0.829 A | **+0.085 A** | **0.169** | 2.6e-04 | 58% |
| **reference** | 0.288 A | **+0.000 A** | 0.078 | 0.80 | **47%** |

Mean outward **+0.095 A** against a mean magnitude of **1.013 A** — a ratio of
**0.09**.

**Only about a tenth of the correction points outward; the rest has no direction
in common between molecules.** The systematic "too deep" component is real
(p=2.6e-04, 58% of molecules) and small. The reference control sits at exactly
chance (47%, p=0.80), which is what an already-optimal pose must give and is the
check that the axis is being read correctly.

Contacts within 5 A move the way that implies: the fit costs generated molecules
4 contacts (−2.0%) and *gains* the reference 3.

**So the 1.1 A cannot be collected by a global offset.** It has to be computed
per molecule from the pocket — which is exactly the refiner's job, and the
refiner's step is separately measured to be locked at 0.38–0.40 of its training
target's magnitude with four failed attempts to lengthen it.

Hypothesis closed: the placement error is not a calibration bug.
