# Matching the coverage moved the target, and nothing beats it yet

2026-09-14. Two corrections to the comparison itself, then the standings they
produce. Neither changes a model; both change what the numbers mean.

## 1. GenScore was being scored on 283 of 285

Its local sweep fails to build graphs for **4k18** and **4mme** and records
FAILED, so its R and rho came from 283 complexes while ours came from 285.
`aggregate.affinity_pairwise` intersects for its p-value but computes
`d_scoring_R` from each method's own rows, so the headline delta differenced
two different sets -- and `beats_both` keyed off that delta.

Both were recovered with GenScore's own checkpoint
(`GatedGCN_ft_1.0_1.pth`), by repairing only the inputs. No coordinate, label,
model or line of GenScore was touched:

* **4mme** -- the supplied MOL2 states a +1 nitrogen that the distributed SDF
  records as neutral, leaving a 4-valent N that cannot sanitize. Restored with a
  single `M  CHG` line on the already-converted SDF.
* **4k18** -- GLU A70 OE1 and ARG A73 NH1 sit **1.58 A** apart in the deposited
  coordinates, so RDKit's proximity bonding invents a covalent bond between
  them and the carbonyl oxygen becomes 3-valent. Exactly that one inferred bond
  is removed. It must be handed over as a pickled `Chem.Mol`: writing the fix to
  PDB and re-reading regenerates the same wrong bond.

| GenScore | R | rho |
|---|---:|---:|
| as run (283) | 0.8159 | 0.7351 |
| **matched coverage (285)** | **0.8127** | **0.7281** |

The two recovered complexes are ones it scores relatively poorly, so matching
coverage *lowered* the bar slightly. That is the honest target. The original
dump is preserved at `results/genscore/scoring_283_as_run.csv`, and
`results/genscore/coverage.json` records the merge.

## 2. Three contaminated arms had climbed back into the table

`e250_union`, `e250_union_s11` and `e250_union_s12` trained against a validation
set that was 60% contaminated. Their registry entries were deleted on 2026-09-13
"so they cannot be picked up later as a best value" -- but `discover_arms` finds
arms by **directory**, not by registry membership, and their dumps were still in
`results/`. Today they reappeared in the standings with `e250_union_s12` leading
on scoring R.

Moved to `results/archive/contaminated_val_60pct/`, which `_NOT_ARMS` excludes.
Deleting a registry entry is not enough; the dump has to move too.

## 3. Where we actually stand

Every arm, CASF f16, all 285 complexes, against GenScore 0.8127 / 0.7281.
Single seeds are draws, so family means are the unit that means anything.

| family | seeds | R | rho |
|---|---:|---:|---:|
| `e250_union_noconf` | 3 | **0.7971** (−0.0156) | 0.6819 (−0.0462) |
| `e250_union_clean` | 3 | 0.7896 (−0.0231) | 0.6632 (−0.0649) |
| `e250_dev_ca_legacy` | 4 | 0.7770 (−0.0356) | 0.6636 (−0.0645) |
| `e250_dev_heavy_validated` | 4 | 0.7755 (−0.0372) | 0.6526 (−0.0754) |
| `e250_rot1` | 3 | 0.7544 (−0.0582) | **0.6953** (−0.0327) |
| `e250_union_lw` | 3 | 0.7453 (−0.0673) | 0.6322 (−0.0959) |
| `e250_pdbbind` | 3 | 0.7403 (−0.0724) | 0.6345 (−0.0936) |

**No arm beats GenScore on both metrics, at any seed.** The closest single draw
is `e250_union_noconf` seed 7 at 0.8096 / 0.7158 (−0.0030 / −0.0123), but its
family mean is 0.7971 / 0.6819, so reporting that seed would be selection on the
test set.

Two further facts the table makes plain:

* **The metrics are won by different corpora.** R is best under
  `union_noconf`, rho under `rot1`, and the ordering reverses between them.
  Adding PDBbind raises scoring and costs ranking, consistently.
* **The new 7,714-site development cohort is not better than the old 8,150 one.**
  Its own control lands at 0.7770 / 0.6636, below `union_noconf` and below
  `rot1` on rho. The cohort was built for the input-repair comparison, and that
  comparison is now closed; it is not a step forward on its own.

## Still running

The M14 parameter-free readout arms (8 heads, balanced and allatoms) and the
paired reconstruction of the two geometry-preserving fresh VQ arms against the
shipped tokenizer on the fixed 492 development complexes.
