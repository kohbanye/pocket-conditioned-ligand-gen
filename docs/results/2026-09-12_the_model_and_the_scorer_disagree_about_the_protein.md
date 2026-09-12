# Where the gap is now, and a pocket that is smaller than it looks

## The five terms, re-measured on the best arm

The gap decomposition on record was taken on the deployed arm, before
clash-ordered decoding took 5 kcal off it. Re-run on `dfs + MLM + clash + prune`
against the crystal reference, clean 79, weighted kcal, per-target medians:

| term | best | deployed | reference | best − reference | share |
|---|---|---|---|---|---|
| **repulsion** | 7.328 | 10.450 | 1.834 | **+4.705** | **71.0%** |
| **hbond** | −1.082 | −1.302 | −1.753 | **+0.612** | **9.2%** |
| gauss1 | −2.079 | −2.194 | −2.448 | +0.330 | 5.0% |
| gauss2 | −5.410 | −5.649 | −5.555 | +0.139 | 2.1% |
| hydrophobic | −2.353 | −2.703 | −2.448 | −0.008 | −0.1% |
| total | −3.605 | −1.018 | −10.795 | +6.625 | |

**Repulsion is still the whole story at 71%**, down from 84% on the deployed arm
— the 3.12 weighted kcal it lost is exactly the session's clash work. The axis is
not exhausted; the *tools* for it are (clash ordering, more rounds, more
candidates, pruning, a finer codebook).

## hbond went backwards

The deployed arm scores −1.302 and the best arm −1.082, against a reference of
−1.753. **Some of the second-largest term was given away while repulsion was
being bought**, and pruning is the mechanism:

| element | atoms deleted | of all deletions | **of that element's atoms** |
|---|---|---|---|
| C | 4638 | 45.6% | 3.12% |
| **O** | 4045 | 39.8% | **8.73%** |
| N | 918 | 9.0% | 3.56% |

Oxygen is deleted at **2.8x the rate of carbon**, and Vina's `hbond` term fires
only on N/O pairs. The pruner ranks by Bondi overlap, which is not element-aware;
terminal oxygens (carbonyls, hydroxyls) are simply most of what a degree-one
filter leaves. N+O share of heavy atoms falls 0.315 → 0.307 across the step.

An element-aware prune rule — prefer deleting a non-polar leaf when the overlaps
are comparable — is the obvious response, and it has to be priced against the
repulsion it would stop buying. Not run.

## The model is conditioned on a smaller pocket than the name suggests

`extract_pocket_atoms_from_candidates` — "Select pocket residues **by CA
distance**". A residue is in the prompt only if its **alpha carbon** is within
`distance_cutoff` = 8 A of a ligand atom, and if it is not, *every* atom of that
residue is dropped. An arginine reaching into the site with its CA at 9 A
contributes nothing.

Counted against the reference ligand of each target — heavy atoms within 8 A of
the ligand that are **not** in the pocket the model sees:

| | |
|---|---|
| targets with at least one | **100 of 100** |
| such atoms per target | median **40**, mean 44.3, max 158 |

It is universal, not a handful of odd structures.

**It is mostly ordinary protein.** Cofactors, metals and modified residues are
excluded too (`resname not in AA_3TO1`), but they are rare: across all 100
targets that is 39 atoms in total — MSE 12, ZN 8, MG 8, CA 4, CSO 4, CU/CL/CO 1
each, and no water at all. The bulk of the 40 per target is side-chain atoms of
standard residues whose CA missed the cut.

## Which parts of the pipeline know

| step | receptor it sees |
|---|---|
| LM prompt (pocket tokens) | **residues with CA within 8 A** |
| pose refiner (`_pocket_context`) | **the same** |
| `rigid_pocket_fit` (only under `--place-before-refine`, not in the best arm) | **the same** |
| clash-ordered decoding probe | every ATOM/HETATM |
| pruner | every ATOM/HETATM |
| Vina, and every clash metric | every ATOM/HETATM |

`read_heavy_atoms`'s own docstring says cofactors and metals "count as part of
the wall a ligand must not walk through — which is what the clash metrics compare
against". So the two halves of the repository hold different definitions of the
protein, and the half that *places atoms* holds the smaller one.

## What this is worth is NOT yet settled

Two measurements of the same thing disagree in a way that matters:

* Over pairs that clash by the deployed 0.75 criterion: 29% of them are with
  atoms the model never saw, carrying **38.2%** of that repulsion, and 92.5% of
  those are inside the 8 A cut (so it is the CA rule, not the distance).
* Over **every** overlapping pair (which is what Vina's repulsion actually
  charges for): the median molecule has **zero** repulsion against invisible
  atoms, for the generated arm and the reference alike.

Both are right: the contribution is sparse and heavy-tailed — most molecules
never touch these atoms, a few touch them hard. A median-over-targets headline
cannot move much on this; a mean can. **The reference ligand clashes with them
too** (0.11 invisible pairs per molecule carrying 77% of its own small clash
repulsion), so part of this is shared and not a deficit at all.

Pricing it properly needs Vina's own radii rather than Bondi — the two differ by
a factor of five on the absolute repulsion here — and that is the next
measurement, not a conclusion.

## Priced with Vina's own radii: the blind spot is not where the score is

The two disagreeing measurements above were both computed with Bondi radii, which
are not what Vina charges against. Redone with the X-S radii and the 8 A cutoff
of the *validated* decomposition (`vina_term_breakdown.py`, whose sum is checked
against Vina's own intermolecular energy), clean 79, weighted kcal:

| | visible | **invisible** | total |
|---|---|---|---|
| reference | med 1.665, mean 2.367 | med 0.000, mean 0.390 | 1.834 |
| generated (best arm) | med 6.117, mean 7.602 | med 0.000, mean 0.613 | 6.786 |

The total reproduces the breakdown's repulsion (6.786 here against 7.328 there,
on a different molecule sample), so this is the same quantity.

Paired per target, generated minus reference:

| | median |
|---|---|
| **visible atoms** | **+4.016** |
| invisible atoms | **+0.000** (mean +0.223; only 14 of 79 targets exceed 0.5) |

**96% of the repulsion excess is against protein atoms the model was shown.**

So the CA-distance rule is a genuine representational gap — 40 heavy atoms per
target, in every target, dropped because their residue's alpha carbon missed an
8 A cut — and it is **not** where the score is. Widening it would cost a corpus
rebuild, a tokenizer retrain and an LM retrain to move a median-over-targets
headline by zero.

**Axis closed.** And it redirects: the model clashes with protein it can see.
That is a modelling failure, not a representation one, and it is the 4.016 kcal
that everything else in the gap is smaller than.
