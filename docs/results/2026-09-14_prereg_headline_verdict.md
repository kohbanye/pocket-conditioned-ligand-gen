# Pre-registered headline: the verdict fails on all three conditions

2026-09-14 20:09 JST. Computed mechanically by
`experiments/affinity_loop_20260914/headline_prereg.py` under
`docs/notes/2026-09-14_prereg_final_ensemble.md` (rules 1-7, Amendments 1-2, the
disclosure, and the user's goal change adding R > 0.83). Membership, frame
counts, twin set and verdict were all fixed before the numbers existed; the
script refused at every earlier run and computed only once all 16 families
(65 arms) were complete at both frame counts. Registry sha256 matched the
pre-registration.

## Verdict

Goal: R > 0.8127 AND rho > 0.7281 (both GenScore, measured here) AND R > 0.83.

| basis | R | rho | vs GenScore R | vs GenScore rho | vs R 0.83 | verdict |
|---|---:|---:|---:|---:|---:|---|
| **f32 ensemble (headline)** | **0.8100** | **0.7000** | -0.0027 | -0.0281 | -0.0200 | **FAIL** |
| f32 best single seed | 0.8090 | 0.7246 | -0.0037 | -0.0035 | -0.0210 | FAIL |
| f16 ensemble | 0.8116 | 0.7140 | -0.0011 | -0.0141 | -0.0184 | FAIL |
| f16 best single seed | 0.8098 | 0.6754 | -0.0029 | -0.0527 | -0.0202 | FAIL |

No basis passes any one of the three conditions except none. The closest
single number is the f32 best single seed's rho, 0.0035 short.

## Full tables

`clean` = the leak-sensitivity set, 285 minus the 12 CASF complexes with a
training twin (Amendment 2), removed identically for every method.

### f32 (headline)

| method | kind | R_all | rho_all | R_clean | rho_clean |
|---|---|---:|---:|---:|---:|
| OURS all 16 families | ensemble | 0.8100 | 0.7000 | 0.8127 | 0.7250 |
| OURS best family (`e250_union_noconf`) | ensemble | 0.8101 | 0.6930 | 0.8125 | 0.7214 |
| OURS best single seed (`e250_union_noconf`) | single | 0.8090 | 0.7246 | 0.8112 | 0.7446 |
| GenScore | single | 0.8127 | 0.7281 | 0.8198 | 0.7429 |
| GIGN (mean of 3) | ensemble | 0.8550 | 0.7702 | 0.8664 | 0.7875 |
| OnionNet-2 | single | 0.8513 | 0.7175 | 0.8515 | 0.7286 |
| GIGN single-model mean | single | 0.8393 | 0.7094 | 0.8507 | 0.7226 |

### f16

| method | kind | R_all | rho_all | R_clean | rho_clean |
|---|---|---:|---:|---:|---:|
| OURS all 16 families | ensemble | 0.8116 | 0.7140 | 0.8139 | 0.7375 |
| OURS best family (`e250_union_noconf`) | ensemble | 0.8107 | 0.6737 | 0.8129 | 0.7000 |
| OURS best single seed (`e250_unionnc_balanced`) | single | 0.8098 | 0.6754 | 0.8139 | 0.6946 |

(Baselines are frame-independent; same rows as above.)

The ordering against every baseline is the same on the clean set as on all
285. Nothing here depends on the overlap.

## What the new families did to the ensemble

The 16-family f32 headline (rho 0.7000) sits BELOW the 12-family f16 pool
measured earlier (0.7140) and the 45-arm f16 pool (0.7175). The family-count
curve ("+0.002 rho per family, no saturation") did not continue: the four
families added last -- two dev-round corners, the IC50 family and the KD/KI
control -- did not raise rho, and f32 lowered it further. That curve was
measured on families already in hand; extrapolating it was the same error as
extrapolating f32 from one family.

## Rule 6: IC50 labels trade ranking for scoring

`e250_aff_ic50` (IC50, KD, KI; 15,047 sites) vs `e250_aff_kdki_ctl` (KD, KI;
8,644 sites), same builder, tokenizer, exclusions and head recipe, paired by
seed:

| | mean dR | better | mean drho | better |
|---|---:|---:|---:|---:|
| f32 | +0.0044 | 3/4 | **-0.0175** | **0/4** |
| f16 | +0.0064 | 4/4 | **-0.0325** | **0/4** |

74% more labelled sites buys a small, consistent R gain and costs ranking on
every seed. IC50 is assay-dependent, and noisier labels blur the ordering within
a congeneric series even while more examples help absolute fit. So "more data"
is not a free R lever: this source of it spends rho, which is already the side
furthest from GenScore.

## Exploratory, not a result: the head recipe on an identical corpus

Seen after the fact, so it cannot change the verdict. `e250_aff_kdki_ctl` and
`e250_dev_ca_legacy` train on the byte-identical corpus (see the disclosure),
differing only in the head-training recipe (`pipelines/train/scoring_head.py`
with `--early-stop-patience 4` vs the dev round's runner). Verified, not assumed: the
token streams are byte-identical (`train.bin` sha256 `c9d905596c085f37` in both) and
both load `mlm_e250lig3` e02 (`fresh_head_tasks_consistent.json` `--init`). Their f32 family
ensembles differ by rho 0.7509 vs 0.6825 -- but paired by seed the effect is
much smaller and not consistent:

| seed | f32 dR | f32 drho | f16 dR | f16 drho |
|---:|---:|---:|---:|---:|
| 7 | -0.0208 | -0.0386 | -0.0212 | -0.0316 |
| 11 | +0.0029 | +0.0719 | -0.0025 | +0.0772 |
| 12 | -0.0051 | +0.0193 | -0.0060 | +0.0088 |
| 13 | +0.0190 | +0.0561 | +0.0173 | +0.0491 |
| **mean** | **-0.0010** (2/4) | **+0.0272** (3/4) | **-0.0031** (1/4) | **+0.0259** (3/4) |

**Resolved: indistinguishable from training-seed noise.** The effective configs
match field for field (lr 1e-4, warmup 500, min-lr ratio 0.1, weight decay 0.01,
micro-batch 32, 8 complexes per batch, 20 max epochs, head dropout 0.1, mean
pooling, label cap 13, block 640, grad clip 1.0, bf16). The one real difference
is `--early-stop-patience 4`, and it does not track the gap: on seed 13 both
recipes selected a checkpoint near epoch 4 yet differ by +0.056 rho, and on seed
12 `scoring_head.py` trained longer and still ranked better.

The real flaw is the pairing. "Seed 7" in `train_round.py` and "seed 7" in
`scoring_head.py` drive different code and consume different RNG streams, so
equal seed numbers do not pair training trajectories -- the comparison is
effectively unpaired. With per-seed rho sd ~0.04, four runs a side give a
standard error of ~0.028 on the mean difference; +0.027 is about one standard
error. No recipe lead exists. Seed pairing only removes variance when both arms
run the same driver.

## Where this leaves the goal

* **rho** is 0.028 short on the headline. The single-model f32 row is 0.0035
  short, but that row is the best of 65 seeds, i.e. a maximum over the test set.
* **R > 0.83** is 0.020 short and nothing tested today moved R by that much.
  Source union moved it ~+0.02 historically, and the union scale-up is blocked
  (PDBbind raw data gone; val split not pinnable).
* Ten levers are now refuted or neutral: interface contacts, pocket context,
  within-cluster compression, congeneric difficulty, metal coordination,
  zero-shot `pll`, cross-generation ensembling, MLM continuation, frame count,
  and IC50 labels (R up, rho down).

## "Change the test": the alternatives, reported after the fact

The user asked (20:25) to pursue every option in priority order, including
changing the comparison. The pre-registered headline had already been computed,
so **no alternative comparison chosen now can be a pre-registration** -- it
would be selected knowing the numbers. What can be done honestly is to report
the alternatives that the pre-registered tables already contain, all of them
computed under the fixed membership, and to decline any comparison picked
because it would pass.

| comparison | ours | reference | R pass | rho pass | R > 0.83 |
|---|---|---|---|---|---|
| ensemble vs single, all 285 (the verdict) | 0.8100 / 0.7000 | GenScore 0.8127 / 0.7281 | no | no | no |
| best single seed vs single, all 285 | 0.8090 / 0.7246 | GenScore 0.8127 / 0.7281 | no | no | no |
| ensemble vs single, leak-clean 273 | 0.8127 / 0.7250 | GenScore 0.8198 / 0.7429 | no | no | no |
| ensemble vs ensemble, all 285 | 0.8100 / 0.7000 | GIGN mean-of-3 0.8550 / 0.7702 | no | no | no |
| single vs single mean, all 285 | 0.8090 / 0.7246 | GIGN single-model mean 0.8393 / 0.7094 | no | yes | no |

(f32 throughout.) The one "yes" -- our best single seed's rho against GIGN's
single-model mean -- compares the best of 65 seeds to a 3-checkpoint average,
which is selection on the test set, and it is not the goal's comparison. It is
listed for completeness and claims nothing.

## Addendum (2026-09-14 22:42): the clean column above is superseded

The `union_noconf` corpus had never been leak-audited on its own; its audit
(`leak_audit_general/out_union_noconf/TWINS.json`) raised the twin set from 12
to 19 by adding 7 PDBbind-side twins. **The verdict above is on all 285 and does
not change.** The clean column does: ours f32 0.8146 / 0.7446 and GenScore
0.8184 / 0.7589 on the 266-complex set (full table in
`2026-09-14_concat_headline_and_the_twin_screen_is_a_lower_bound.md`). Still
below GenScore on both metrics, ordering unchanged.

That audit also showed the frozen twin criteria are a **lower bound**: a CASF
complex screened against its own source files self-flags only 190/285 (PDBbind
path) and 73/279 (BioLiP path), because coreset files split chains at gaps. So
every clean column here is a sensitivity analysis on an incomplete screen, not a
certification. Pdbid-level exclusion of all 285 core complexes remains exact.
