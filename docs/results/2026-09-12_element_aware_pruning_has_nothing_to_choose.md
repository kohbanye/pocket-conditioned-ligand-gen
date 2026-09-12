# Element-aware pruning has nothing to choose between

`hbond` is the second largest term in the remaining gap (+0.612 kcal, 9.2%), and
the best arm is **worse** on it than the deployed arm (−1.082 against −1.302).
Pruning is the suspect: it deletes only terminal heavy atoms, and terminal heavy
atoms in these molecules are largely oxygens.

## The rate, measured one-to-one

An earlier count differenced two dumps with different molecule counts (10100
against 9954), so molecules that failed to parse landed in the difference and the
ratio came out as 2.8x. Re-run by matching each molecule to its own pruned self,
5858 molecules:

| element | atoms present | deleted | **rate** |
|---|---|---|---|
| C | 77073 | 499 | **0.65%** |
| N | 13424 | 127 | 0.95% |
| **O** | 26533 | 1417 | **5.34%** |

Oxygen goes at **8.2x the rate of carbon**, not 2.8x. The effect is larger than
first reported.

## And the obvious fix does nothing

`prune_clashing_leaves` was given a temporary `polar_margin`: a nitrogen or
oxygen leaf is deleted only when its overlap beats the best non-polar leaf's by
that margin (squared Angstroms), with 0 reproducing the published rule exactly.

| margin | molecules pruned | atoms | C | N | O | F | Cl | S |
|---|---|---|---|---|---|---|---|---|
| 0.0 (published) | 2999 | 2142 | 499 | 127 | 1417 | 43 | 27 | 13 |
| 0.1 | 2999 | 2142 | 499 | 127 | 1417 | 43 | 27 | 13 |
| 0.5 | 2999 | 2142 | 499 | 127 | 1417 | 43 | 27 | 13 |
| **2.0** | 2999 | 2142 | **499** | **127** | **1417** | 43 | 27 | 13 |

**Identical, atom for atom, even at a margin that makes a polar atom all but
untouchable.** The rule never has a non-polar candidate to pick instead: when a
terminal atom is in the wall it is the only clashing leaf, and it is an oxygen.

So the `hbond` cost of pruning is not a preference that can be tuned away. The
only way to keep those oxygens is not to prune them, which gives back the
repulsion they carry — and pruning's repulsion gain is already measured at 2.25
weighted kcal per molecule on terminal atoms (4.596 -> 2.345).

**The parameter is not kept.** It changes nothing, so leaving it in would be a
knob that cannot move the result; the rejection is recorded in
`prune_clashing_leaves`'s docstring instead, beside the two other rules that
were measured and dropped. Re-deriving it is a ten-line patch and this table.
