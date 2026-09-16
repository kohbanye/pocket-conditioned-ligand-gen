# Same-protein ligand-kNN blend: refuted (val loved it, CASF did not)

2026-09-15 ~16:30 JST. Pre-registered in
`docs/notes/2026-09-15_prereg_ligand_knn_blend.md`; applied by
`experiments/affinity_loop_20260914/knn_blend_verdict.py`.

## Result

Selection on PDBbind val (198 complexes: the builder's val draw minus pdbids in
the BioLiP training half; 8 `e250_unionnc_corr` seeds, f16). Ensemble alone
R 0.6638. Every (k, w) helped on val, monotonically in w:

| k \ w | 0.1 | 0.2 | 0.3 | 0.5 | 0.8 |
|---|---|---|---|---|---|
| 1 | +0.0208 | +0.0358 | +0.0462 | +0.0573 | **+0.0603** |
| 3 | +0.0203 | +0.0347 | +0.0444 | +0.0542 | +0.0552 |
| 5 | +0.0198 | +0.0338 | +0.0433 | +0.0528 | +0.0540 |

Selected k=1, w=0.8. Applied once to CASF:

| | R | rho |
|---|---|---|
| f32 ensemble (8 seeds) | 0.8118 | 0.6947 |
| f32 blend | 0.7973 (**-0.0145**) | 0.6561 (**-0.0386**) |
| f16 ensemble | 0.8132 | 0.7070 |
| f16 blend | 0.7979 (-0.0153) | 0.6509 (-0.0561) |

**KN1 False, KN2 False -> DOES NOT WORK.**

## Reading

* The exploratory CASF scan's ceiling (+0.011 R at w 0.2) was real only at
  small weights chosen on CASF; val asked for four times that weight.
* Val and CASF disagree about the retrieval signal, in a direction that makes
  sense: a random PDBbind validation draw sits among its own congeneric series
  (the same papers deposited the training neighbours), while the CASF core was
  assembled to be diverse -- median best Tanimoto to a same-protein training
  ligand is 0.186. A prior that is strong on the first is noise on the second.
* The ensemble is much weaker on this val (R 0.66) than on CASF (R 0.81), so
  any blend weight chosen on it is pulled toward the prior. Another reason a
  random PDBbind split is the wrong selection set for CASF.
* Consequence for every future lever that needs a tuned weight: tune on
  something CASF-like (diverse, cluster-structured), not a random split.

## Cost

8 val inferences (~95 s wall each), ~1.2 pt reserved. CPU otherwise.
