# Why every leaf-based lever saturated: the repulsion moved inside the molecule

Repulsion is 71% of the remaining gap and 96% of that is against protein atoms
the model can see. Clash-ordered decoding, more rounds, more candidates, a finer
codebook and pruning are all saturated against it. This says why.

Vina's repulsion, attributed to the ligand atom that carries it, binned by how
many heavy-atom bonds that atom has. Clean 79, 632 molecules per arm, X-S radii.

| degree | **after prune** rep/mol | share | | **before prune** rep/mol | share |
|---|---|---|---|---|---|
| 1 terminal | 2.345 | 24.0% | | 4.596 | 35.4% |
| **2 chain** | **5.821** | **59.5%** | | 6.033 | 46.5% |
| 3 branch | 1.377 | 14.1% | | 1.927 | 14.8% |
| 4 | 0.224 | 2.3% | | 0.378 | 2.9% |
| total | 9.776 | | | 12.976 | |

**Pruning removed half the repulsion on terminal atoms** (4.596 -> 2.345) and did
not touch the rest, so terminal atoms fell from 35.4% of the burden to 24.0%.
The tool worked and then ran out of things it is allowed to touch.

**59.5% of what is left sits on degree-2 atoms — the middle of a chain.** A leaf
can be deleted (one bond to satisfy) or substituted (one bond to preserve); an
interior atom has two, so both operations are far more constrained there. That is
a geometric fact about the tools, not a tuning problem, and it predicts exactly
the saturation observed: pruning cannot reach these atoms at all, and a code
substitution that must keep two bond lengths and two angles almost never clears
a wall.

## What it licenses, and why that is closed too

Moving interior atoms needs an operation that moves several at once: rigid motion
or torsions. Both were already measured.

`--refine-project torsion` exists and was run (`2026-08-30_the_rigid_part_was_free.md`),
on bond-preserving arms only:

| arm | bond deviation | repulsion median | vs uncorrected |
|---|---|---|---|
| no refiner | 10.3% | 12.86 | — |
| TORSION | 10.3% | 11.51 | −10.5% |
| **RIGID** | **10.3%** | **8.75** | **−32%** |

**Torsion projection is three times worse than rigid**, so it is not an unused
lever. The 9.91 kcal that motivates it is what **Vina's own local optimiser**
finds in translation + rotation + torsion space; it is not what projecting the
refiner's displacement into that space recovers. The refiner does not know which
torsions to turn, and a projection cannot invent them.

So the interior repulsion is reachable only by an optimiser with an objective in
that space — which is the force field, excluded by standing decision. This is the
same wall as the pose axis (4.03 kcal reachable by rigid motion, 1.46 captured).

**Recorded so it is not re-derived a third time:** interior repulsion is not a
decoding problem and the decoding tools are correctly saturated.
