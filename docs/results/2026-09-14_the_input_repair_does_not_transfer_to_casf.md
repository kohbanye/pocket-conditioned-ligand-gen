# The input repair was real, helped development, and does not transfer to CASF

2026-09-14. The 2026-09-13 development round found two genuine input defects,
repaired both, and measured **ΔR +0.0303 on all four training seeds** against
its own control (`experiments/affinity_loop_20260914/RESULT_JA.md`). That was
the most promising affinity result on the table. Measured on CASF-2016 with the
same four seeds and the same paired control, **it is gone.**

## The measurement

CASF-2016 core set, 285 complexes, 16 frames, seeds 7/11/12/13, each arm scored
under the inputs its own head was trained on. Control is the same round's
`ca_legacy` arm -- same corpus, same cohort, same head, same schedule -- so the
only difference is the two input repairs.

| seed | R control | R repaired | ΔR | rho control | rho repaired | Δrho |
|---|---:|---:|---:|---:|---:|---:|
| 7 | 0.7881 | 0.7843 | −0.0038 | 0.6842 | 0.6596 | −0.0246 |
| 11 | 0.7886 | 0.7864 | −0.0022 | 0.6491 | 0.6439 | −0.0053 |
| 12 | 0.7888 | 0.7759 | −0.0128 | 0.6807 | 0.6684 | −0.0123 |
| 13 | 0.7426 | 0.7553 | **+0.0127** | 0.6404 | 0.6386 | −0.0018 |
| **mean** | **0.7770** | **0.7755** | **−0.0015** | **0.6636** | **0.6526** | **−0.0110** |

`n_better` is **1/4 on scoring and 0/4 on ranking**. The pre-registered rule for
this round was `n_better`/`n_seeds`, so the hypothesis is **rejected**. GenScore
on the same benchmark is 0.8159 / 0.7351.

The one seed that improved, 13, is also the control's worst draw (R 0.7426
against 0.7881/0.7886/0.7888) -- the same regression-to-the-mean shape that
produced the single apparent win in H8 and H9. It is not evidence.

## The repair itself worked

This is not a null result from a no-op. The sidecars show 282 of 285 complexes
took the validated route, with exactly the three expected fallbacks (3ary,
3bv9, 4eky -- the complexes whose SDF and MOL2 disagree on coordinates), and
the route recovers **257 charged atoms against 14** for the rebuilt path. Heavy
crop likewise recovers the contacting side chains that CA selection drops in
124 of 283 complexes. Both defects were real and both are fixed; they just do
not move CASF.

## What this costs and what it saves

Spent: 8 GPU jobs, 3.2 pt of the 200 pt ceiling.

Saved: the clean 7,666-site retrain was queued behind exactly this question. Its
purpose was to turn a leak-flagged +0.030 into a headline number. There is no
+0.030 to launder, so that retrain is not worth its compute, and neither is
promoting these inputs into the shared tokenizer -- which would also have cost
protein reconstruction (+0.0139 A common-atom RMSD from heavy crop).

## What it means

Development ΔR on this cohort's 492 validation documents **does not predict CASF
ΔR**. The two sets differ in difficulty (dev R ~0.58-0.62, CASF ~0.78), in pK
range and in target composition, and the dev set was additionally used for
checkpoint selection. Any future arm justified by a dev delta has to be
confirmed on a held-out set before it earns compute -- which is what just
happened here, cheaply, and is the reason to keep doing it in this order.

Still outstanding and NOT answered by this: the remaining measured dev gain is
M14's parameter-free readout change (+0.0171 dev, 3/4 seeds). Its prior is now
lower, because the one dev effect that has been checked against CASF did not
survive.

## Fairness note carried forward

These arms cover all 285 complexes; the local GenScore dump covers 283 (it
fails to build graphs for 4k18 and 4mme). `aggregate.affinity_pairwise`
intersects for its p-value but computes `d_scoring_R` from each method's own
rows, so a 285-complex R is being differenced against a 283-complex one. The
numbers above are our-arm-vs-our-control and so unaffected, but no
"beats GenScore" statement should be made until coverage is matched. The
recovered inputs for both complexes already exist
(`genscore_graph_diagnostic/4k18_topology_corrected.pdb`,
`4mme_charge_restored.sdf`), so the fix is cheap.

## Addendum (2026-09-14 evening): the full 2x2, all four corners on CASF

The round originally scored only two corners (`ca_legacy` control and the full
`heavy_validated` repair). The other two were scored later so every registered
family could enter the pre-registered ensemble
(`docs/notes/2026-09-14_prereg_final_ensemble.md`, rule 3). Their heads existed
and were not re-selected. 4 seeds per corner, all 285 complexes, paired by seed
against `ca_legacy`.

| vs `ca_legacy` | f16 dR | f16 drho | f32 dR | f32 drho |
|---|---:|---:|---:|---:|
| chemistry only (`ca_validated`) | +0.0080 (2/4) | -0.0083 (1/4) | +0.0094 (3/4) | -0.0114 (1/4) |
| pocket only (`heavy_legacy`) | -0.0022 (3/4) | -0.0149 (1/4) | -0.0045 (3/4) | -0.0232 (0/4) |
| both (`heavy_validated`) | -0.0015 (1/4) | -0.0110 (0/4) | -0.0018 (1/4) | -0.0158 (1/4) |

Main effects on seed means:

| | pocket heavy - ca | chemistry validated - legacy | interaction |
|---|---:|---:|---:|
| R, f16 | -0.0059 | +0.0043 | -0.0073 |
| rho, f16 | -0.0088 | -0.0022 | +0.0123 |
| R, f32 | -0.0079 | +0.0061 | -0.0068 |
| rho, f32 | -0.0138 | -0.0020 | +0.0189 |

**The non-transfer is not two helpful factors cancelling.** Heavy-atom pocket
selection, which kept the side chains that dev said mattered, *costs* CASF
ranking; validated chemistry nudges R up and leaves rho flat; their positive
rho interaction means chemistry partly offsets the pocket loss, which is why the
full repair read as "no effect" rather than "harmful".

Four seeds is thin and no cell is 4/4, so this is a direction, not an
established effect. It adds no lever. All four corners share the same
leak-flagged 7,714-site cohort, so the paired contrasts are unaffected by it.
