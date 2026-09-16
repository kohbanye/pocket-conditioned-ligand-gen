# Literature-reported CASF-2016 scoring power (NOT measured here)

2026-09-16. **These numbers are copied from papers. They are not comparable to
the measured table** and must never be merged into it.

This bench states in three separate files that "published numbers are never
copied into the table: a row exists because it was measured here", because a
measured row is scored on the SAME crystal poses and the SAME labels as our
arms. The rows below were produced by their authors on their own inputs,
preprocessing, and — as it turns out — frequently on a different number of
complexes. They are context for a related-work section, not competitors in the
results table.

## The 290-vs-285 problem, which affects five of these rows

**DeepAtom, Pafnucy (2016), FAST, HAC-Net and TopBP all report on 290 complexes
while calling it "CASF-2016" or "the PDBbind v2016 core set".** Strict CASF-2016
is **285** (57 clusters x 5; Su et al., JCIM 2019, DOI 10.1021/acs.jcim.8b00545).
290 is the PDBbind-v2016 core-set release before CASF-2016's own curation
dropped five. This is a systematic convention in this literature, not five
independent mistakes, and deserves one footnote rather than five.

## Interaction-FREE (the class this project belongs to)

| model | R | RMSE | n | source | single/ensemble | runnable? |
|---|---:|---:|---:|---|---|---|
| AK-Score v1 | **0.827** | ~1.22–1.29 | 285 | primary, IJMS 2020 21(22):8424 | **ensemble of 30** (single 0.760) | no code anywhere |
| OctSurf | 0.875 | not found | not confirmed | **secondary snippets only**, primary paywalled | unknown | repo enterprise-only |
| DeepAtom | 0.83 ±0.003 | 1.23 | **290** | primary, arXiv:1912.00318 Table II | mean of 5 training runs | supplementary Excel+PDF only |
| AEScore95 | 0.83 [0.79–0.86] | 1.22 | 285 | primary, J Cheminform 13:59 (2021) | **ensemble of 5** (plain AEScore 0.80/1.30) | **code public**; audit flags ambiguous checkpoint |
| PointTransformer | 0.833 / **0.852** | 1.26 / 1.19 | 285 | primary, BiB 23(1):bbab474 Table 4 | single / **5-model ensemble** | weights 404 (verified by us) |
| Pafnucy | **0.78** | 1.42 | **290** | primary, Bioinformatics 34(21):3666 Table 1 | single | **code+weights public** |
| KDEEP | 0.82 | 1.27 | not confirmed | **SECONDARY** (WIREs review); primary paywalled | not stated | closed PlayMolecule platform |

## Interaction-EXPLICIT (fails this project's criterion; listed for contrast)

| model | R | RMSE | n | source | single/ensemble | featurisation |
|---|---:|---:|---:|---|---|---|
| TopBP | 0.861 | 1.65 | **290** | primary, PLoS CB 14(1):e1005929 | GBT + 50-model DL, x1000, median | persistent homology on cross atom-pair distances |
| HAC-Net | 0.846 (CNN-only 0.750, GCN-only 0.806) | 1.205 | **290** | primary, JCIM 63(7):1947 | **3-branch ensemble** | GCN branches use 3.5 A contact graph |
| FAST | 0.810 | 1.308 | **290** | primary, JCIM 61:1583 Table 1 | 2-branch fusion | SG-CNN: 1.5 A covalent / 4.5 A non-covalent graph |
| AK-Score2 (AK-Score-C) | 0.531 | not stated | 285 | primary, J Cheminform 16:121 Table 2 | single GNN + physics term | contact edges, protein-ligand pairs <8 A |

## Caveats that change how these should be read

* **Ensembles dominate the top.** AK-Score v1's 0.827 is a **30-network**
  ensemble; its single model is **0.760**. TopBP averages a GBT ensemble with a
  50-model DL ensemble, repeated 1000x. HAC-Net and AEScore95 are ensembles too.
  The same trap this bench already documented for GIGN (3-checkpoint mean worth
  +0.0157 R and +0.0608 rho over its single-model mean) applies across this
  whole literature.
* **HAC-Net does not reproduce.** A 2025 audit (Zhang et al., JCIM,
  PMC13370855) re-ran it and got **R = 0.777 against the claimed 0.846**, traced
  to a broken ACC2-API dependency.
* **KDEEP is secondary-sourced and contested.** 0.82 comes from a review
  paraphrasing the paywalled primary; a separate 2026 review's own reproduction
  got **R ~ 0.75** on an unclear subset. Do not quote 0.82 as settled.
* **AK-Score v1 has a third circulating value**: HAC-Net's table lists
  "AK-score*" at 0.812, footnoted as having used no validation set and therefore
  "expected to present overly optimistic results". Prefer the primary 0.827.
* **PointTransformer's preprint and published numbers are different
  quantities**, not a shrinkage: preprint 0.859 (single) was revised to 0.833
  (single), and 0.852 (5-model ensemble) is new in the published version.
* **AK-Score2's headline is worse than its own component**: the raw GNN variant
  AK-Score-DockC scores 0.696 against the combined model's 0.531.

## Not verified

KDEEP's primary text (paywalled everywhere tried); OctSurf's R/RMSE/n/leak
handling (paywalled, no OA copy); DeepAtom's peer-reviewed IEEE version vs its
arXiv preprint; TopologyNet (2017) as distinct from TopBP (2018); AK-Score v1's
canonical ensemble RMSE (1.22 vs 1.29 ambiguity within the same paper).

## Leak handling, quoted

Every interaction-free row above except KDEEP and OctSurf states that core-set
complexes were excluded from training. Examples: Pafnucy "we excluded a total of
87 overlapping complexes"; PointTransformer "All data sets of the test set were
removed from the training set"; DeepAtom "we exclude the complexes shared with
the core subset to prevent the data leakage"; AEScore "Systems present in both
PDBbind and CASF datasets are removed from the training and validation sets".
KDEEP: not stated / not verifiable. OctSurf: not accessible.
