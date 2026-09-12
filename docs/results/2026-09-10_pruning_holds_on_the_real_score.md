# Pruning holds on the real Vina score

The intermolecular estimate was −1.201 kcal
(`2026-09-10_two_atoms_the_molecule_cannot_place.md`). That figure carries
neither the intra term nor the torsion normalisation, both of which deleting an
atom changes, so the whole dump was pruned and put through the ordinary
evaluation against its own unpruned source in one invocation — same code path,
same Vina, same PoseBusters.

`canon100_refoverlap_t0` vs `canon100_refoverlap_t0_prune`, 9948 molecules each,
100 targets, paired per target.

## Score

| convention | deployed | pruned | change |
|---|---|---|---|
| median over targets (the −6.806 convention) | −0.504 | **−1.240** | −0.736 |
| mean over targets of medians (the bench column) | +1.303 | **−0.796** | −2.099 |
| **paired per target** | | | **−0.954**, p=5.5e-16, better on **87%** |

## Everything else moved the right way

| metric | deployed | pruned | paired | p |
|---|---|---|---|---|
| PoseBusters valid | 0.572 | 0.584 | +0.004 | — |
| strain energy | 858.9 | 819.1 | −5.1 | 5.3e-08 |
| clash count | 2.28 | 0.79 | −1.00 | 1.6e-12 |
| QED | 0.412 | 0.438 | +0.011 | 1.2e-08 |
| SA | 4.064 | 3.988 | −0.025 | 3.4e-05 |
| validity | 0.994 | 1.000 | — | 3e-05 |
| **internal diversity** | 0.638 | 0.657 | +0.003 | 8.3e-07 |
| **unique SMILES** | 0.547 | 0.573 | +0.019 | 8.1e-12 |

**No chemistry is broken and no diversity is paid.** Both diversity measures
*rise* — different molecules lose different atoms, so pruning separates near
duplicates more often than it merges them.

## The cost, stated plainly

| | deployed | pruned | reference |
|---|---|---|---|
| heavy atoms | 22.31 | **21.35** | 22 |
| molecular weight | 330.0 | **314.9** | — |

One heavy atom and 12.5 Da, which takes the median molecule from just above the
reference's size to just below it.

## The random control: the rule does the work

The reference-ligand control rules out "the scoring function likes smaller
molecules". It does not rule out "these molecules improve on losing *any*
terminal atom". So a matched control was run: the **same number of terminal
deletions per molecule** (0.99 over 50% of molecules -- identical by
construction), the atom chosen uniformly at random instead of by clash, seeded
through `prolit.seeding.rng_for`.

| arm | vina_score | vs deployed | p | better on | heavy atoms |
|---|---|---|---|---|---|
| deployed | +1.303 | — | — | — | 22.31 |
| **clash-pruned** | **−0.796** | **−0.954** | 5.5e-16 | 87% | 21.34 |
| random-pruned | +0.552 | −0.160 | 1.7e-11 | 71% | 21.34 |

**Head to head, clash beats random by −0.740 kcal, p=3.7e-16, on 87% of
targets.** So 0.16 of the 0.95 is "deleting a terminal atom helps a little" and
**0.79 — 83% — is choosing the right one.**

The two pruned arms are indistinguishable on chemistry (paired difference
+0.000 on PoseBusters, strain and QED alike) and differ only where they should:
clash count 0.79 against 1.74. The rule removes clashes; random deletion does
not.

**The lever stands.**

## Provenance

`prolit.chem.prune_clashing_leaves` (9 tests), applied by
`benchmarks/sbdd-bench/scripts/prune_arm.py` (`--random` for the control, seeded
through `prolit.seeding.rng_for`). The pruning reads the **receptor only** — no
reference ligand — so it does not change whatever reference information the
source arm already uses.
