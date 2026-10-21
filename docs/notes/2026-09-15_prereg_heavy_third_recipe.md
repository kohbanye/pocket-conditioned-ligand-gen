# Pre-registration: a third recipe (heavy-atom pocket) in the mix

2026-09-15 ~16:45 JST. Written BEFORE the cross-recipe confirmation (M1/M2) was
read, before either heavy corpus finished building, and before any head of this
arm exists.

## Why a third recipe, and why this one

Family residuals (LS-calibrated, CASF, exploratory) correlate:

| pair | residual corr |
|---|---|
| seeds within one family | 0.93 |
| `e250_unionnc_corr` vs `e250_union_noconf` (both ca, no rotation) | 0.984 |
| `e250_unionnc_corr` vs `e250_unionr8_ens` (rotation) | 0.945 |
| dev `ca_legacy` vs dev `heavy_legacy` (pocket definition) | **0.82-0.88** |

The pocket definition decorrelates our errors more than anything else measured,
including rotation. On the dev cohort, mixing ca and heavy families gained
+0.0065 R at n=8 where same-family seeds gain +0.0008.

## Arms

* **Treatment** `e250_unionheavy` (NEW, 8 seeds 26-33): corpus =
  `data/lm_tokens_e250aff_kdki_heavy` (BioLiP, `--pocket-distance heavy`) +
  `data/lm_tokens_e250aff_pdbbind_noconf_heavy` (PDBbind, same), same head
  recipe as every recent arm (`--label-cap 13 --max-epochs 20
  --early-stop-patience 4 --block-size 640`), scored with the arm's own
  `pocket_distance="heavy"`.
* **Baseline mix (16):** the fresh cross-recipe mix
  (`e250_unionnc_corr_v2` + `e250_unionr8_ens_v2`, seeds 18-25).
* **Size-matched control (24):** baseline mix + 8 already-scored models of the
  same two recipes (`e250_unionnc_corr` seeds 7, 11, 12, 13 and
  `e250_unionr8_ens` seeds 7, 11, 12, 13).

## Predictions (f32 basis)

* **T1.** R(16 fresh + 8 heavy) - R(16 fresh + 8 same-recipe) >= **+0.004**.
  This is the diversity claim with size held fixed at 24 models.
* **T2.** rho(16 fresh + 8 heavy) >= rho(16 fresh + 8 same-recipe) - 0.005.

**Works only if T1 and T2 hold.** f16 reported beside.

## Disclosed now

* The heavy corpus's BioLiP half is the `--validated-cohort` build (8,206
  complexes) while the ca arms' BioLiP half is `r1` (8,581). The third recipe
  therefore changes **two** things -- pocket definition and BioLiP cohort -- and
  a pass cannot be attributed to the pocket definition alone.
* Still no contact pairs, interaction edges, distance bins or contact counts:
  "heavy" only selects which residues enter the pocket, by minimum heavy-atom
  distance instead of CA distance.
* No headline from this note. A headline needs its own registration and must
  state the model count on both sides (GenScore's 3-checkpoint ensemble is the
  like-for-like reference: R 0.8387, rho 0.7211).

## Cost

2 corpus builds (already submitted, ~1 pt) + 8 heads (gpu_1, 1:00) ~= 1.6 pt +
16 inferences ~= 1.6 pt ~= **4.2 pt total**, ledger to about 132 of 200.
