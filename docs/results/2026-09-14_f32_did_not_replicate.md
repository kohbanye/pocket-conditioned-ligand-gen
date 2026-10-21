# 32 frames bought nothing, and the pilot family was the whole effect

2026-09-14. Doubling inference frames from 16 to 32 was justified on one
family's six seeds (drho +0.0155, 5/6 arms) and extended to every registry arm
for 18 pt. Across all 49 arms it is worth **nothing measurable**, and R is
slightly worse.

## Measured, all arms that have both dumps

| level | mean dR | better on | mean drho | better on |
|---|---:|---:|---:|---:|
| family (12 complete) | **-0.0013** | 1/12 | **+0.0028** | 8/12 |
| arm (49) | -0.0008 | 14/49 | +0.0025 (sd 0.0155) | 27/49 |

Per-arm sign test on drho: **27 up, 22 down**. The mean sits inside its own
standard error (0.0155 / sqrt(49) = 0.0022). R moves down on 35 of 49 arms.

Per family:

| family | arms | dR | drho |
|---|---:|---:|---:|
| `e250_dev_ca_legacy` | 4 | -0.0013 | **+0.0211** |
| `e250_union_noconf` | 6 | -0.0006 | **+0.0193** |
| `e250_unionnc_balanced` | 4 | -0.0010 | +0.0123 |
| `e250_pdbbind` | 3 | +0.0000 | +0.0088 |
| `e250_unionnc_allatoms` | 4 | -0.0006 | +0.0053 |
| `e250_rot1` | 6 | -0.0030 | +0.0035 |
| `e250_dev_allatoms` | 4 | -0.0018 | +0.0018 |
| `e250_dev_heavy_validated` | 4 | -0.0013 | +0.0018 |
| `e250_dev_balanced` | 4 | -0.0011 | -0.0018 |
| `e250_unionnc_lo` | 4 | -0.0016 | -0.0105 |
| `e250_union_clean` | 3 | -0.0013 | -0.0140 |
| `e250_union_lw` | 3 | -0.0023 | -0.0140 |

Two families hold nearly all the positive mass. `union_noconf` -- the pilot --
is one of them.

## Why the pilot misled

The pilot was six seeds of ONE family. Seed-level drho has sd 0.0155, so a
6-arm family mean has a standard error near 0.0063; +0.0155 is about 2.5 of
those, which reads as real. What it could not show is **family-to-family**
variation, and that turns out to dominate: family drho ranges -0.0140 to
+0.0211.

This is `feedback_single_run_comparisons` one level up. That rule says do not
compare arms on one training seed because seed moves R by ~0.05. The same
argument applies to families: **do not judge an inference-protocol change on one
family**, because the family effect has its own spread and picking the family
that moved is selection.

Cost of the lesson: 18 pt for the 45-arm extension, plus 2.4 pt for the pilot.

## What this does to the pre-registration

`docs/notes/2026-09-14_prereg_final_ensemble.md` rule 2 fixed **f32 as the
headline with f16 beside it**, written before any all-families result existed.
That stands, and is honored:

* The pre-registered choice now costs us slightly on R (-0.0013 at family
  level), so honoring it is not self-serving.
* The principled argument for f32 is unchanged and independent of this result:
  frame averaging estimates an expectation over rotations, so 32 samples is a
  strictly better estimate of the same model than 16. The estimator is better
  even when the metric does not move.
* What changes is the CLAIM. f32 is not a lever, is not an improvement, and
  must not be presented as one. It is a variance-reduction detail of the
  evaluation protocol.

Both tables get reported. Neither is chosen after the fact.

## Live levers after this

Frame count is out. What remains, from
`2026-09-14_the_deficit_is_not_interface_information.md`:

* **Family count** -- ensemble rho still climbing at 12 families (+0.002/family,
  no saturation). Untouched by this result, since it is about how many families
  exist, not how each is scored.
* **Training data** -- R is flat in family count but responds to source union
  (+0.02 over either source alone). The IC50 matched pair is building; the union
  scale-up is blocked (PDBbind raw data gone, val split unpinnable).

Nine levers have now been tested and refuted in one day: interface contacts,
pocket context, within-cluster compression, congeneric difficulty, metal
coordination, zero-shot `pll`, cross-generation ensembling, MLM continuation,
and frame count.
