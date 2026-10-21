# Pre-registration: a fourth recipe (all-atom readout) in the mix

2026-09-15 ~17:10 JST. Written BEFORE any head of this arm exists and before its
corpus round is submitted. Same fixed-size design as the heavy round
(`docs/notes/2026-09-15_prereg_heavy_third_recipe.md`), run in parallel with it.

## Why this axis

The confirmed lever (`docs/results/2026-09-15_cross_recipe_mix_confirmed.md`) is
mixing recipes whose between/within profiles differ. The readout is the third
axis where our arms differ that way, measured on the dev cohort at f32:

| dev family (4 seeds) | R | between | within | rho |
|---|---|---|---|---|
| `e250_dev_ca_legacy` (ligand readout) | 0.7896 | 0.3525 | 0.4371 | 0.6825 |
| `e250_dev_allatoms` (all-atom readout) | 0.7863 | **0.3780** | 0.4083 | **0.7000** |

The all-atom readout mean-pools every atom token, pocket and ligand, instead of
ligand tokens only. It trades within for between exactly like the rotation
recipe does, and it needs no new corpus. It is parameter-free (unlike `concat`),
so it is the same head size.

**No contacts:** pooling all atom representations is not a contact pair, an
interaction edge, a distance bin, or a contact count. The concat readout was
refuted as a stand-alone arm
(`docs/results/2026-09-14_concat_readout_refuted.md`); this round makes no claim
about the readout alone, only about what it adds to a mix.

## Arms

* **Treatment** `e250_unionallatoms` (NEW, 8 seeds 34-41): corpus
  `data/lm_tokens_e250aff_union_noconf`, `--readout allatoms`, otherwise the
  standard recipe (`--label-cap 13 --max-epochs 20 --early-stop-patience 4
  --block-size 640`), scored with the arm's own readout.
* **Baseline mix (16):** `e250_unionnc_corr_v2` + `e250_unionr8_ens_v2`, seeds 18-25.
* **Size-matched control (24):** baseline mix + 8 already-scored models of those
  two recipes (seeds 7, 11, 12, 13 of each).

## Predictions (f32 basis)

* **A1.** R(16 fresh + 8 allatoms) - R(16 fresh + 8 same-recipe) >= **+0.004**.
* **A2.** rho(16 fresh + 8 allatoms) >= rho(16 fresh + 8 same-recipe) - 0.005.

**Works only if A1 and A2 hold.** f16 reported beside. The family's residual
correlation against the two mixed recipes is reported with the verdict, not used
as a criterion.

Measured before submitting, on the dev cohort (f32, LS-calibrated residuals of
4-seed family ensembles): ca_legacy vs allatoms **0.947**, ca_legacy vs
heavy_legacy 0.936, allatoms vs heavy_legacy 0.896, ca_legacy vs balanced 0.981.
So this axis decorrelates about as much as the rotation recipe already in the
mix (0.945) and less than the heavy recipe. Stated now: the expected gain is
smaller than the heavy round's, and the round is run because it needs no corpus
and can train in parallel, not because it is the stronger bet.

If both this round and the heavy round pass, the combined 32-model mix is
exploratory until its own headline pre-registration exists.

## Cost

8 heads (gpu_1, 1:00) ~= 1.6 pt + 16 inferences ~= 1.6 pt ~= **3.2 pt**, ledger
to about 131 of 200.
