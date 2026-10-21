# Pre-registration: the final affinity ensemble (2026-09-14)

Written 2026-09-14T18:39:31+09:00, BEFORE any of the following exist: the all-families f32 ensemble,
any CASF number for `e250_aff_ic50` or `e250_aff_kdki_ctl`. Registry file
`benchmarks/affinity-bench/affinity_bench/variants.py` sha256 at writing:
`fe07fbb3239b315f56b70573c96f58d8661c31059b751f5bdd7432acd158bf94`.

## Why this exists

Families have been joining the ensemble one at a time -- `e250_unionnc_lo`, then
`e250_aff_ic50` and `e250_aff_kdki_ctl` -- while the ensemble sits 0.0020 R and
0.0106 rho under GenScore. Choosing membership after seeing the f32 and IC50
results would be a search over subsets on the test set: "add families until it
crosses 0.7281". Fixing membership now removes that freedom.

Measured just before writing, to show the choice is not free: adding the
refuted lo family to the 45-arm f16 ensemble moves R 0.8107 -> 0.8115 and
rho 0.7175 -> 0.7140. Including or excluding a family changes the headline.

## Rules

1. **Headline = z-sum over every arm of every family listed below**, each member
   standardised then summed with equal weight (`zsum_ensemble`). No family is
   dropped for its score, including `e250_unionnc_lo`, which lost its own paired
   comparison 1/4 seeds.
2. **Frame count: f32 is the headline, f16 reported beside it.** Justified before
   the all-families result: 6 paired seeds of `e250_union_noconf`, drho +0.0155
   (5/6), dR +0.0001. Applied to every arm or none.
3. **A family missing any member's dump is reported as incomplete, not silently
   dropped.** The headline is computed only when every family below is complete
   at that frame count; until then no headline number exists.
4. **Families registered after this note are excluded from the headline** and go
   to a secondary table, however they score.
5. **Verdict = beats GenScore on BOTH** R > 0.8127 and rho > 0.7281 (n=285,
   measured here). Also reported: the best single training seed per family and
   the best single family mean, because the original goal was a single model
   and an ensemble win does not satisfy it.
6. Paired family comparisons use the control in `SEED_REFERENCE_FOR`, seed by
   seed, reporting n_better/n_seeds.

## Membership (16 families, 65 arms)

Dump counts are as of writing and are recorded only to show what existed.

| family | arms | f16 dumps | f32 dumps | paired control |
|---|---:|---:|---:|---|
| `e250_aff_ic50` | 4 | 0 | 0 | `e250_aff_kdki_ctl` |
| `e250_aff_kdki_ctl` | 4 | 0 | 0 | `e250_rot1` |
| `e250_dev_allatoms` | 4 | 4 | 3 | `e250_dev_ca_legacy` |
| `e250_dev_balanced` | 4 | 4 | 0 | `e250_dev_ca_legacy` |
| `e250_dev_ca_legacy` | 4 | 4 | 0 | `e250_rot1` |
| `e250_dev_ca_validated` | 4 | 0 | 0 | `e250_dev_ca_legacy` |
| `e250_dev_heavy_legacy` | 4 | 0 | 0 | `e250_dev_ca_legacy` |
| `e250_dev_heavy_validated` | 4 | 4 | 0 | `e250_dev_ca_legacy` |
| `e250_pdbbind` | 3 | 3 | 0 | `e250_rot1` |
| `e250_rot1` | 6 | 6 | 0 | `e250_rot1` |
| `e250_union_clean` | 3 | 3 | 0 | `e250_rot1` |
| `e250_union_lw` | 3 | 3 | 0 | `e250_union_clean` |
| `e250_union_noconf` | 6 | 6 | 6 | `e250_union_clean` |
| `e250_unionnc_allatoms` | 4 | 4 | 0 | `e250_union_noconf` |
| `e250_unionnc_balanced` | 4 | 4 | 0 | `e250_union_noconf` |
| `e250_unionnc_lo` | 4 | 4 | 0 | `e250_union_noconf` |

Families with fewer arms than listed at f16/f32 are either still running or
were never scored at that frame count; rule 3 applies.

## Amendment 1 (same day, still before any f32 / IC50 result)

Added after reading `final_source_overlap_audit/` and `final_near_duplicate_audit/`,
and before any of the results this note governs exist.

**The leak flag is corpus-wide, not a dev-cohort property.** The dev round's
note flags its own CASF numbers because 20 of its 7,714 training sites share a
protein sequence and ligand with a CASF core complex. Measured here: **none of
those 20 training PDBs are excluded by `eval_holdout_pdbs.txt` or
`affinity_union_holdout_pdbs.txt`** (only `6ql2`, by the noconf list). They are
BioLiP sites, so they reach `e250aff_r1` and therefore `union_noconf` too.
Dropping the six dev-cohort families from the headline would not remove the
flag; it would only hide where it was measured. All 16 families stay.

**Rule 7 (new). Every headline number is reported twice, on the same eligible
set for every method:**

* **Protocol set** = all 285 core complexes, PDB-id exclusion only. This is the
  protocol GenScore, GIGN and OnionNet-2 were trained under, and the only basis
  on which their published numbers mean anything.
* **Leak-sensitivity set** = 285 minus the CASF complexes with a training twin,
  where a twin is (a) exact protein-chain sequence + ligand identity by
  canonical isomeric SMILES or RDKit InChIKey, or (b) near-duplicate under the
  frozen policy in `final_near_duplicate_audit/policy.json` (Morgan r=2/2048
  Tanimoto >= 0.9, global-alignment sequence identity >= 0.9, both coverages
  >= 0.9). Derived from the audit artifacts, not transcribed.

The same removed set is applied to OUR predictions and to every baseline dump.
Removing cases from only one side would be the "evaluation 側から都合の悪い症例を
落とす" failure both audit reports warn against.

**Exact protein sequence alone is NOT treated as a leak** (103 of 285 core
complexes have a same-sequence training site). Same target with a different
ligand is ordinary in every PDBbind-trained method, and the audit declines to
call it leakage. It is disclosed, not excluded.

**What this does not certify:** `cohort.json` carries
`complete_leakage_certification: false` and both audits state that homology,
pocket similarity, scaffold similarity and public-pretraining contamination
were not examined. The sensitivity set bounds one specific, measured overlap.
It is not an independent-test certification, and no claim here should be
written as one.

## Disclosure (19:33, before any KD/KI-control or IC50 CASF number exists)

Not an amendment: membership and rules are unchanged. Recorded because it
changes how two of the listed families should be read.

**`e250_aff_kdki_ctl` trains on the same corpus as `e250_dev_ca_legacy`.**
`corpus_kdki_ctl.sh` passed exactly the flags of `corpus.sh`'s `ca_legacy`
build (`--pocket-distance ca --chemistry-mode legacy --validated-cohort
--affinity-types KD,KI --casf-pdbs data/eval_holdout_pdbs.txt --seed 0
--val-frac 0.05`, same tokenizer), and the builder is deterministic: its
`complex_manifest.jsonl` sha256 is `cecfc875a6fcf54f...`, byte-identical to the
dev cohort's (8,206 accepted = 7,714 train + 492 val). The two families differ
only in the head-training recipe. The headline therefore weights that corpus
twice. Rule 4 forbids removing either, so both stay, and this is disclosed.

**The label-type pair is still a valid one-flag comparison.** `--validated-cohort`
does not pin the site set (it sets manifest writing and `preserve_chemistry`),
so `--affinity-types` does act. From the builds' own logs:

| build | affinity sites before cohort filter | protein groups |
|---|---:|---:|
| `kdki_ctl` (KD,KI) | 8,644 | 2,166 |
| `ic50` (IC50,KD,KI) | 15,047 | 2,722 |

Both use the same head recipe, so `e250_aff_ic50` vs `e250_aff_kdki_ctl` isolates
the label restriction. What they do NOT isolate is "more data" from "noisier
labels": IC50 is assay-dependent, and the two effects arrive together.

**Leak audit status for the new corpora.** The generalized audit
(`experiments/affinity_loop_20260914/leak_audit_general/`) reproduces every
data artifact of the frozen dev-cohort audit byte for byte. Because the KD/KI
control manifest is the dev cohort's, auditing it can only re-confirm the 10
pre-registered twins. The IC50 corpus is a strict superset (15,047 sites) and is
the audit that can change rule 7's set; it runs when its manifest lands and
before its CASF numbers exist.

## Amendment 2 (19:55, before any IC50 CASF number exists)

Existence check run immediately before writing this: **0** `pk-mean.csv` dumps
for any `e250_aff_ic50*` arm, and **0** IC50 head checkpoints (heads still
training). No IC50 score had been written, let alone read.

**Rule 7's leak-sensitivity set grows from 10 to 12 CASF complexes.** The IC50
corpus is a strict superset of the KD/KI corpus (13,590 accepted training sites)
and was audited with the generalized audit
(`experiments/affinity_loop_20260914/leak_audit_general/`, which reproduces
every data artifact of the frozen dev-cohort audit byte for byte). Near-
duplicate preflight passed within the frozen caps (133 alignments, 12.5M DP
cells). Two twins not in the pre-registered set:

| CASF | rule | responsible training sites | Tanimoto | identity | coverage (casf/train) |
|---|---|---|---:|---:|---|
| `1qkt` | near-duplicate | 13 `EST` sites: 1a52, 1ere, 1g50, 1gwr, 1pcg, 1qku, 2ocf, 2yja, 3uud, 4pxm, 5gs4, 5wgq, 6cbz | 1.0000 | 0.9825-1.0000 | 0.92-1.00 / 0.99-1.00 |
| `3wtj` | near-duplicate | `3wtk\|TH4\|A\|A\|1` | 1.0000 | 0.9952 | 1.0000 / 1.0000 |

**All 13 `1qkt` sources are IC50-only** -- absent from the KD/KI corpus. Restoring
IC50 labels imported estradiol / estrogen-receptor complexes that are
near-identical to a CASF core complex. This is the specific leak the IC50 audit
existed to catch, and it would not have been visible from the dev-cohort audit.

**`3wtj` is an unresolved inconsistency, included conservatively.** Its source
site `3wtk|TH4|A|A|1` IS in the KD/KI corpus, yet the audit of that corpus --
whose manifest is byte-identical to the dev cohort's -- reported exactly the
original 10 without `3wtj`. The same training site is flagged on one corpus and
not the other. Until that is explained, the frozen dev-cohort set of 10 cannot
be assumed complete either. Including `3wtj` can only remove a complex from
EVERY method equally; omitting a real twin would hide a leak. So it is in.

The set is applied, as before, identically to our dumps and every baseline dump.
Final set (12): `1o5b 1p1q 1qkt 2al5 2iwx 2wbg 2wer 2yge 3d4z 3dd0 3wtj 4jsz`.

### Amendment 2, resolution note (19:57, still before any IC50 CASF number)

The `3wtj` inconsistency is explained and is **not** an audit defect. Its source
site `3wtk|TH4|A|A|1` sits in the **val** split of the KD/KI corpus and in the
**train** split of the IC50 corpus. The audit screens `split == 'train'` only, so
it correctly did not flag a val site. The site moved because the builder draws
val by UniProt protein group over a pool derived from the site set, and adding
IC50 sites changes that pool. So the frozen dev-cohort set of 10 is not in doubt,
and the set of 12 stands on its own terms: `3wtj` is a genuine TRAINING twin for
the IC50 family.

**Caveat this exposes, recorded rather than acted on:** a CASF twin in **val**
still influences checkpoint selection (heads pick the lowest-val-loss
checkpoint), which is an indirect channel no audit here screens -- the dev-cohort
audit, the KD/KI audit and the IC50 audit all read the train split only. For
the KD/KI-control family, `3wtj` is exactly such a val twin. It is inside the set
of 12, so the sensitivity analysis removes it for every method regardless.
Val-side twins in other corpora were not searched.

## Goal change from the user (~20:00, before any headline was computed)

The user added a third condition to the loop's goal: **scoring R > 0.83**, in
addition to beating GenScore on both R (> 0.8127) and rho (> 0.7281). Recorded
here before the headline exists; `headline_prereg.py` has never produced a
headline number (it has refused at every run for lack of the IC50 family).

**Rule 5, as amended: the verdict requires all three**:

1. R > 0.8127 (GenScore, n=285, measured here)
2. rho > 0.7281 (GenScore)
3. **R > 0.83** (user-set absolute bar)

Same basis as before -- the protocol set (all 285), f32 headline with f16
beside it, and the single-model row reported next to the ensemble. The
leak-sensitivity set (rule 7) is reported for context but is not a second
chance to pass: the verdict is read on the protocol set.

Stated before seeing the result, so it cannot be tuned afterwards: the best
numbers measured so far are ensemble R 0.8115 and best single model R 0.8098,
so R > 0.83 needs about +0.02. No lever tested today moved R by that much;
family count is flat in R and f32 lowered it. The only effect of that size seen
is source union (+0.02 over either source alone), so the IC50 family is the one
pending result with a plausible R effect.

### Disclosure, verification (20:20)

The claim that `e250_aff_kdki_ctl` and `e250_dev_ca_legacy` "differ only in the
head-training recipe" originally rested on the manifest sha256 alone, which
records site identities, not tokens. Now verified at both levels that could
have broken it:

* **Token streams are byte-identical** across `data/lm_tokens_e250aff_kdki_ctl`,
  `experiments/affinity_loop_20260914/corpora/ca_legacy` and
  `corpora_consistent/ca_legacy` (the corpus the dev heads trained on): same
  sha256 for `train.bin` (`c9d905596c085f37`), `val.bin`, `train.rmsd`,
  `val.rmsd`, `train.len`, `val.len` and the manifest. The "consistent" repair
  pass (`repair_dative_corpus.py`) left `ca_legacy` unchanged.
* **The MLM backbone is the same.** `fresh_head_tasks_consistent.json` passes
  `--init pocket-ligand-mlm/mlm_e250lig3/checkpoints/mlm-e02-vl0.8009.ckpt` for
  every dev head -- the checkpoint the `scoring_head.py` heads load.

So the only difference is the head-training driver and its settings. This
changes nothing in the verdict; it establishes that the post-hoc rho contrast
between these two families is a recipe contrast and not a data or backbone one.
