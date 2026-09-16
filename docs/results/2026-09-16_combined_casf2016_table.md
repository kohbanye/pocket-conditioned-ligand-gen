# CASF-2016 scoring power: measured here + literature-reported, in one table

2026-09-16. **Mixed-provenance table, produced on request.** The `src` column is
load-bearing: `meas` rows were scored here on the same crystal poses and the
same labels (285 complexes); `lit` rows are copied from papers, on their
authors' own inputs and frequently on a different complex count. Rows are sorted
by R, which deliberately interleaves them — that is what makes the provenance
column necessary rather than decorative.

| # | model | R | rho | RMSE | n | src | int | pose | models |
|---|---|---:|---:|---:|---:|---|---|---|---|
| 1 | OctSurf | 0.875 | — | — | ? | lit² | no | full | ? |
| 2 | TopBP | 0.861 | — | 1.65 | 290 | lit | **yes** | full | GBT+50DL ×1000 |
| 3 | GIGN | **0.8550** | **0.7702** | 1.147 | 285 | **meas** | **yes** | full | 3 |
| 4 | HAC-Net | 0.846 | — | 1.205 | 290 | lit | part | full | 3 |
| 5 | OnionNet-2 | **0.8513** | **0.7175** | 1.215 | 285 | **meas** | **yes** | full | 1 |
| 6 | PointTransformer | 0.852 / 0.833 | — | 1.19 / 1.26 | 285 | lit | no | full | **5** / 1 |
| 7 | AK-Score v1 | 0.827 | — | ~1.22 | 285 | lit | no | full | **30** (single 0.760) |
| 8 | DeepAtom | 0.83 | — | 1.23 | 290 | lit | no | full | 5 runs |
| 9 | AEScore95 | 0.83 | — | 1.22 | 285 | lit | no | full | **5** (plain 0.80) |
| 10 | **OURS 36-model mix** | **0.8229** | **0.7316** | *1.334* | 285 | **meas** | **no** | full | **36** |
| 11 | KDEEP | 0.82 | — | 1.27 | ? | lit² | no | full | ? |
| 12 | GenScore | **0.8127** | **0.7281** | — | 285 | **meas** | **yes** | full | 1 |
| 13 | FAST | 0.810 | — | 1.308 | 290 | lit | **yes** | full | 2 |
| 14 | OURS single seed | **0.7920** | **0.6718** | 1.411 | 285 | **meas** | **no** | full | 1 |
| 15 | **Sfcnn** | **0.7880** | **0.6509** | 1.338 | 285 | **meas** | **no** | full | 1 |
| 16 | **PLANET** | **0.7870** | **0.5965** | 1.407 | 285 | **meas** | **no** | **none** | 1 |
| 17 | Pafnucy | 0.78 | — | 1.42 | 290 | lit | no | full | 1 |
| 18 | Boltz-2 | **0.7532** | **0.7158** | 1.515 | 285 | **meas** | no | **none** | 1 |
| 19 | Vina | **0.6076** | **0.5105** | 2.411 | 285 | **meas** | **yes** | full | empirical |
| 20 | AK-Score2 | 0.531 | — | — | 285 | lit | **yes** | full | 1 |

`meas` = measured here, 285 complexes, our poses and labels ·
`lit` = copied from the paper · `lit²` = **secondary source only, primary
inaccessible** · `int` = uses explicit interaction features ·
`pose` = consumes the crystal pose · *italic RMSE* = from the pK-mean
aggregation (the z-sum headline is not on the pK scale).

## Five reasons this ordering should not be read as a ranking

1. **rho is missing for every literature row.** Papers report scoring power
   (R); ranking power is a separate CASF metric most of them never publish. Any
   ranking claim can only be made across the `meas` rows.
2. **n is 290 for five literature rows** (TopBP, HAC-Net, DeepAtom, FAST,
   Pafnucy), which call it "CASF-2016" or "the PDBbind v2016 core set". Strict
   CASF-2016 is 285 (Su et al., JCIM 2019). 290 is the pre-curation release.
   Systematic convention, not five mistakes.
3. **The top of the table is ensembles.** TopBP averages a GBT ensemble with a
   50-model DL ensemble, 1000x. AK-Score v1's 0.827 is **30 networks**; its
   single model is **0.760**, which would place it below every measured row
   except Boltz-2 and Vina. PointTransformer's 0.852 is 5 models (single 0.833).
   AEScore95 is 5 (plain 0.80).
4. **HAC-Net's 0.846 did not reproduce.** A 2025 audit (Zhang et al., JCIM,
   PMC13370855) re-ran it and got **0.777**, tracing the gap to a broken
   ACC2-API dependency.
5. **KDEEP and OctSurf are `lit²`.** KDEEP's 0.82 comes from a review
   paraphrasing a paywalled primary, and a separate 2026 reproduction got
   ~0.75. OctSurf's 0.875 could not be confirmed from primary text at all, and
   its complex count is unknown.

## What survives all of that

Restricted to rows that are measured on identical inputs (`meas`), with the two
design axes visible:

| class | best | R | rho |
|---|---|---:|---:|
| explicit interactions + pose | GIGN (3 models) | 0.8550 | 0.7702 |
| **no explicit interactions + pose** | **OURS (36 models)** | **0.8229** | **0.7316** |
| no explicit interactions, pose-blind | Boltz-2 | 0.7532 | 0.7158 |

Our margin over the best *independent* interaction-free peer (Sfcnn, 0.7880) is
**not statistically significant**: dR +0.0348 (p=0.073), drho +0.0807
(p=0.155), Williams/Steiger and Wilcoxon respectively. Only GIGN significantly
beats Sfcnn (p=0.0002 / 0.018).
