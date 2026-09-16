# Pre-registration: blending a same-protein ligand-similarity kNN into the ensemble

2026-09-15 ~15:45 JST. Written BEFORE any validation prediction for the blend
exists and before the blend's (k, w) is chosen.

## Origin -- disclosed because it contaminates the idea, not the test

An exploratory scan ON CASF (`experiments/affinity_loop_20260914/ligand_knn_diagnostic.py`)
put a ceiling on this: blending the kNN prior into the `e250_unionnc_corr`
family moved R 0.8126 -> 0.8236 and rho by +0.02..+0.04 at the best CASF-chosen
weights. Those numbers chose k and w on CASF and are not results. The test below
removes every choice from CASF; it cannot remove the fact that the idea was
suggested by CASF, so a pass is weaker evidence than a lever predicted blind.

The companion diagnostic `target_prior_diagnostic.py` found that a
UniProt-mean prior (no ligand information) gains only +0.003 even on CASF, so
what the kNN adds is ligand-level, not "which protein".

## Method (fixed now)

* **Retrieval base:** the PDBbind half of `union_noconf`'s training documents
  (7,201; labels from its `train.rmsd`), restricted to the query's UniProt
  (PDBbind `INDEX_general_PL_name.2020`). No BioLiP-half ligands (no structures
  at hand), no validation or CASF complexes.
* **Similarity:** RDKit Morgan radius 2, 2048 bits, Tanimoto, from
  `{pdb}_ligand.sdf` (mol2 fallback).
* **Prior:** similarity-weighted mean pK of the top-k neighbours (weights
  Tanimoto + 1e-3). Fallback: UniProt mean over all union training labels, then
  the global training mean.
* **Blend:** `z(ensemble) + w * z(prior)`, z computed within the evaluated set
  (the protocol `zsum` already uses for heads).
* **Ensemble:** `e250_unionnc_corr`, all 8 seeds.
* **No contacts, no interaction features:** the prior sees ligand
  fingerprints, a protein identity, and training labels only.
* **Leak check done before this note:** 0 of 268 CASF complexes with neighbours
  have a retrieval neighbour at Tanimoto >= 0.9 (median max 0.186).

## Selection (on validation only)

`(k, w)` over k in {1, 3, 5} x w in {0.1, 0.2, 0.3, 0.5, 0.8}, chosen by the
Pearson R of the blend on **PDBbind val of `pdbbind_noconf`** (the builder's
386-complex validation draw, minus any pdbid that also appears in the BioLiP
training half; disjoint from PDBbind training and from CASF, verified; list in
`data/pdbbind_noconf_val_pdbs.txt`) using the f16 family ensemble. Ties within
0.001 go to the smaller w, then the smaller k. If no (k, w) beats the ensemble
alone on val, the blend is declared not selected and CASF is not scored with it.

These complexes were never used for early stopping: `union_noconf`'s val files
are BioLiP r1's 431 documents (its meta.json's 817 is stale), so the PDBbind
val draw was never written into any split the heads saw.

## Predictions (CASF, the val-chosen (k, w), applied once)

* **KN1 (scoring).** f32 family: R(blend) - R(ensemble) >= +0.005.
* **KN2 (ranking).** f32 family: rho(blend) - rho(ensemble) >= 0.
* f16 reported beside. The goal line (R > 0.8127, rho > 0.7281, R > 0.83) is
  printed as exploratory.

**Works only if KN1 and KN2 hold.** A headline built on it would need its own
pre-registration and must state the model is an 8-seed ensemble plus a
retrieval prior over training labels.

## Cost

8 validation inference jobs (gpu_1, 0:30) ~= 0.8 pt. CPU otherwise.
