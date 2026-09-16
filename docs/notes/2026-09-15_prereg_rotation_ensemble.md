# Pre-registration: rotation augmentation on the union corpus, judged as an ensemble

2026-09-15 ~12:20 JST. Written BEFORE the corpora are built, before any head is
trained, and before any CASF number for these arms exists.

## Why this revisits a refuted lever, and what changed

`docs/results/2026-09-15_rotation_sampling_refuted.md` refuted rotation
sampling on 2026-09-15 02:30. Its family ensembles cleared both bars
(**dR +0.0143, drho +0.0193** at f16) but the paired seeds did not (2/4 and
1/4), and the pre-registration required per-seed consistency. That verdict
stands and is not being reopened.

What changed is the external evidence. A literature pass (2026-09-15) found:

* **AK-Score** (Kwon 2020, IJMS): single network **R < 0.74**, five networks
  **R > 0.80**, thirty networks **R = 0.827** -- the largest measured effect of
  any single design choice in this family -- while training with **24-fold
  rotation augmentation**.
* **Pafnucy** (2018): 24-fold rotation augmentation at training time, and
  explicitly **no test-time averaging**.

So in the models that use it, rotation augmentation's benefit shows up as
**ensemble diversity**, not as a better single net. That is exactly the shape we
measured and rejected: family gain, no per-seed gain. Phase 1 was run on the
BioLiP-only corpus (families at R 0.79); it was never run on the union corpus
that carries our best numbers.

**This is a different claim from Phase 1's**, and a weaker kind: family-level
only. A family mean is noisier than a paired per-seed test, so the design
compensates by **doubling the seeds, 4 -> 8**, which halves the standard error
of the family mean. Stated here, before any number.

## Arms

| | treatment `e250_unionr8_ens` | control `e250_union_noconf` |
|---|---|---|
| corpus | NEW: union with 8 orientations per complex | `data/lm_tokens_e250aff_union_noconf` |
| training | `--rotations-per-complex 8` (one orientation drawn per access) | default |
| seeds | 7, 11, 12, 13, 14, 15, 16, 17 | 7, 11, 12, 13, 14, 15 (already trained) |

The corpus is rebuilt from the same two sources with `--num-rotations 8` on both
halves (the PDBbind builder has no such flag yet and gets the BioLiP builder's
implementation; val stays at one orientation, as it already does on the BioLiP
side). Everything else is the recipe every recent arm used: `--label-cap 13
--max-epochs 20 --early-stop-patience 4 --block-size 640`, backbone
`mlm_e250lig3` e02, ligand readout, f16 and f32 inference.

Epochs stay complex-sized: `--rotations-per-complex 8` indexes one entry per
complex and draws one of its 8 orientations per access (verified at 722
steps/epoch on the union corpus), so this is not the 8x-documents run that
memorised in `e250_rot8`.

## Predictions, fixed now

Judged on the **f32 family ensemble** of 8 treatment seeds against the control's
6 seeds (all seeds of each family; no seed selection).

* **E1 (scoring).** family dR >= +0.010.
* **E2 (ranking).** family drho >= +0.010.
* **E3 (it is diversity, not per-seed quality).** The mean per-seed R of the
  treatment is NOT required to beat the control's; this is recorded, not
  required. If per-seed mean also improves by >= +0.010, that is reported as a
  stronger result than the claim being made.

**Lever verdict:** works only if E1 AND E2 hold.

**Stated in advance so it cannot be rationalised later:** a family-level-only
criterion is the weaker test, and this project has been fooled by family means
before (the f32 round, the 3-seed rot1 rho). Eight seeds is the mitigation. If
E1/E2 pass, the result is a claim about ensembles of this size, not about the
model being better per seed, and any headline built on it must say so.

## What this round does not do

No headline. If the lever works, a headline needs its own pre-registration
before the ensemble is scored -- and it would have to state that the number is
an 8-seed ensemble, since the literature pass showed that several published
numbers (AK-Score 0.827, HAC-Net 0.846, GIGN 0.855) are ensembles reported
without that qualifier being obvious.

## Cost

2 corpus builds (gpu_1, ~2-3 h each) ~= 1.4 pt + 8 head jobs (gpu_1, 2:00)
~= 3.2 pt + 16 inference jobs ~= 1.6 pt ~= **6.2 pt**, taking the ledger to
about 124 of the 200 pt cap.
