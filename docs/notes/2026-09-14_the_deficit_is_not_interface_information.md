# The deficit is not interface information

2026-09-14. Five hypotheses tested and refuted in one afternoon, all on CPU, all
against the CASF-2016 core set. Recorded because four of them are attractive
enough to be re-proposed by anyone reading the size decomposition, and one of
them I had half-written up as a finding before the measurement contradicted it.

The standing puzzle: our size-free scoring R is 0.7368 against GIGN's 0.8339,
while we are *better* size-calibrated than either winner. So roughly +0.097 of
non-size signal is missing, and the question is what it is.

## 1. Ligand tokens are interface-blind -- true, and it does not matter

`LigandAtomDescriptor.compute` fills a ligand atom's four knn slots from ligand
atoms only, unless `pocket_canonical`/`pocket_elements` are passed. They are
passed in exactly one place: `data/atom_descriptors.py`, gated on
`PocketExtractionConfig.pocket_context`, which defaults to False and is set only
by `build_descriptors.py` and `train/vqvae.py`.

Both affinity corpus builders and affinity-bench inference tokenize through
`PoseEncoder`, which has no pocket-context path at all. So train and test agree
-- there is no tokenization mismatch corrupting our CASF numbers -- and a ligand
atom's token genuinely cannot see the protein it binds.

That is a real property of the tokenizer. It is not the affinity deficit. See 2.

Note for anyone tempted to grep this: `compute`'s pocket arguments are
positional, and the one caller passes them positionally. A keyword-only search
for `pocket_canonical=` finds nothing and reads as "never wired".

## 2. Contacts are the missing signal -- refuted, and backwards

Residualise the label on prediction, residualise contact features on the same
prediction, and regress. A residual that still correlates with contacts is
signal the model failed to use.

| model | R | R(residual ~ contacts) |
|---|---:|---:|
| ours (`e250_union_noconf_f16`) | 0.8096 | 0.1607 |
| ours (`e250_rot1_f16`) | 0.7957 | 0.1817 |
| GenScore | 0.8127 | 0.3608 |
| GIGN | 0.8550 | 0.3933 |

Contacts do carry signal beyond size: `R(label_resid ~ contacts | size) =
0.3700`. But **we have already extracted more of it than GIGN or GenScore
have.** Both of them win while leaving more contact-explainable variance in
their residuals than we do. Feeding contacts into the tokenizer cannot be the
lever, because we are not short of that information.

Features: heavy-atom contacts within 4 A, contacts per ligand atom, buried
fraction, polar-polar pairs within 3.5 A, apolar contacts, minimum distance.

## 3. Pocket context costs reconstruction anyway

`vq_e250_pctx` vs `vq_e250_lig3` differ by exactly one flag, `--pocket-context`
(and the cache that flag requires). The affinity stack is built on `lig3`.

| run | best val `atom_coord` |
|---|---:|
| `vq_e250_lig3` | 0.1021 |
| `vq_e250_pctx` | 0.1071 |

Worse. Do not compare `pctx` against `vq_e250` (0.1204): that run also lacks
`--ligand-source-weight 3.0` and `--ligand-ema-weight 3.0`, so the difference is
three flags, not one, and flatters `pctx` badly.

`vq_e250_pctx` has existed since 2026-08-20 and nothing downstream consumes it.
That is the right outcome, for two independent reasons.

## 4. We compress predictions within a congeneric series -- refuted

If a head emitted nearly the same value for five near-identical ligands, ranking
rho would collapse regardless of representation quality. Within-cluster spread,
in global z units, mean over 57 clusters:

| | label | ours | GIGN | GenScore |
|---|---:|---:|---:|---:|
| spread | 0.835 | 0.691 | 0.677 | 0.706 |
| ratio to label | -- | 0.827 | 0.811 | 0.845 |

We spread *more* within a cluster than GIGN. The head is not saturating. Our
five congeners get distinct values; they are in the wrong order.

## 5. We are specifically bad at congeneric series -- refuted, and inverted

`corr(mean within-cluster Tanimoto, our rho) = +0.1335`. Split at the median:

| half | ours | GIGN | gap |
|---|---:|---:|---:|
| high similarity (n=29) | 0.759 | 0.797 | 0.038 |
| low similarity (n=28) | 0.671 | 0.743 | 0.072 |

We do *better* where ligands are most alike, and the gap to GIGN is nearly twice
as wide on the chemically diverse clusters. The "cannot rank small
substitutions" story is exactly backwards.

Also ruled out by the same run: label-spread noise. All 57 clusters span more
than 1.5 log units, so no cluster's rho is a coin flip on a flat label range.

## 6. Metal coordination is where we lose -- refuted, and we TIE there

Plausible on its face: `LIGAND_ELEMENT_VOCAB` is
`C N O S F Cl Br I P B Si OTHER` -- **no metal has its own code**, on either
side of the shared codebook, so Zn, Mg and Mn are all the same undifferentiated
`OTHER`. GIGN and OnionNet-2 featurise metal contacts directly.

23 of 285 complexes have a metal within 3 A of a ligand heavy atom.

| stratum | n | ours | GIGN | GenScore |
|---|---:|---:|---:|---:|
| abs error, no metal | 262 | 1.011 | 0.830 | 0.948 |
| abs error, metal | 23 | 1.246 | 1.057 | 1.264 |
| scoring R, no metal | 262 | 0.8188 | 0.8636 | 0.8252 |
| scoring R, metal | 23 | 0.7235 | 0.7562 | 0.6494 |

Metal sites are harder for everyone, and the GAP is flat (0.235 vs 0.227 in
error; 0.045 vs 0.033 in R -- slightly *smaller* on metals). Per cluster it is
starker still: on the 6 clusters containing metals we tie GIGN exactly, 0.683
to 0.683, and the whole ranking deficit sits in the 51 metal-free clusters
(0.720 vs 0.780).

So the missing metal codes are a real limitation of the tokenizer that is
demonstrably *not* what the affinity gap is made of. Worth fixing on its own
merits some day; worthless as a lever here.

## 7. The zero-shot `pll` column carries usable signal -- refuted

Our dumps have always carried a pseudo-log-likelihood column that no analysis
used. `R(pll, label) = 0.1501`, and `head + 0.25 * pll` scores **0.8040** against
the head's own 0.8096. It makes things worse.

## What survived

Nothing about the interface. Two things point elsewhere:

**Complementarity.** GIGN explains 0.5590 of our residual; we explain 0.3458 of
GIGN's. Neither subsumes the other -- z-summing reaches R=0.8701, and adding
GenScore 0.8746. We carry real signal the contact-featurised models lack. That
is diagnostic only; their predictions are not ours to use.

**The backbone is undertrained.** `mlm_e250lig3` stopped at `--max-epochs 3`
with val loss still falling (1.1034 -> 0.8847 -> 0.8009), and 0.8009 is what the
registry pins for every affinity number in this project. Its own continuation
`mlm_e250lig3_lo` (lr 1e-4, one epoch, same token dir, same 536-batch val set)
reached **0.7867** on 2026-08-20 and has never been used by an affinity head.
The lr is load-bearing: `mlm_e250lig3_x2` ran the same continuation at the
default lr and went backwards to 0.9235.

That is the arm trained as `e250_unionnc_lo` (seeds 7/11/12/13), paired against
`e250_union_noconf` seed by seed, differing only in `--mlm-ckpt`.

### ...and it did not work either (final, 4 paired seeds)

| seed | dR | drho |
|---:|---:|---:|
| 7 | -0.0120 | **-0.0825** |
| 11 | +0.0115 | +0.0351 |
| 12 | -0.0048 | -0.0263 |
| 13 | -0.0107 | -0.0211 |
| **mean** | **-0.0040** (1/4) | **-0.0237** (1/4) |

Family mean: new R 0.7947 / rho 0.6579 against control R 0.7987 / rho 0.6816.
Seed 7 -- whose control was the strongest of the four (head val 0.7849) --
lost 0.08 of rho outright.

The head's own val loss said the same thing first, before CASF was touched: the
continued backbone gave WORSE head val loss on 3/4 seeds (mean +0.0126), better
only on seed 11 -- the same seed that then won on CASF.

**So MLM val loss does not predict affinity.** A backbone that models the token
stream better does not produce a better affinity head. This is worth more than
the arm it killed: it says "pretrain the LM longer" is not the affinity lever,
and it retires the whole class of "improve the backbone objective" moves unless
someone finds a different backbone metric that does correlate.

Consequence: the queued 3-epoch continuation (`mlm_e250lig3_lo2`) is no longer
expected to move affinity. It is left running because a better backbone still
has value for pose rescoring and generation, and because its reservation is not
recyclable.

## 8. Ensembling across tokenizer GENERATIONS -- refuted

Every e250 arm shares one VQ and one MLM, so their errors are correlated by
construction. The `v1` arms use a different VQ (`xzkjxu9q`) AND a different MLM,
which is as much architectural diversity as this project has.

| ensemble | n | R | rho |
|---|---:|---:|---:|
| all e250 families | 45 | 0.8107 | 0.7175 |
| all e250 + v1 | 48 | 0.8110 | 0.7175 |
| e250 + 0.25 x v1 (per generation) | -- | 0.8107 | 0.7193 |
| e250 + 1.00 x v1 (per generation) | -- | 0.8054 | 0.6965 |

Different tokenizer, correlated errors. Nothing there.

## What IS still alive

**Family count has not saturated.** Ensembling by number of families, averaged
over random subsets:

| k families | pooled R | pooled rho |
|---:|---:|---:|
| 1 | 0.7918 | 0.6718 |
| 6 | 0.8101 | 0.7053 |
| 12 | 0.8112 | 0.7175 |

Still climbing at 12, roughly +0.002 rho per family, and every single family
sits below rho 0.711 while the pool reaches 0.7175. The deficit is rho-shaped
(-0.0106), so *more diverse families* is a measured lever rather than a guess.

**Frame count is rho-shaped too, and it works.** Frame averaging estimates an
expectation over rotations, so more samples is a better estimate of the SAME
model. That makes it variance reduction rather than selection -- and it obliges
reporting f32 for every arm or none, which is what the 45-arm batch is for.

`e250_union_noconf`, all six seeds, f16 vs f32 paired:

| seed | dR | drho |
|---:|---:|---:|
| 7 | -0.0006 | +0.0088 |
| 11 | +0.0007 | +0.0088 |
| 12 | -0.0004 | -0.0035 |
| 13 | -0.0026 | +0.0140 |
| 14 | +0.0004 | +0.0298 |
| 15 | +0.0029 | +0.0351 |
| **mean** | **+0.0001** (3/6) | **+0.0155** (5/6) |

Family z-sum: rho **0.6737 -> 0.6930** (+0.0193), R 0.8107 -> 0.8101 (flat).
The weakest seeds gain the most (s14, s15), which is what noise reduction on a
noisy estimator looks like.

Earlier evidence pointing the same way, on `v1_kdki_mean`: f4 -> f16 moved rho
+0.0298 with R -0.0045, against an inference-seed noise floor of rho sd 0.0046
/ R sd 0.0013 (measured from `_f16`, `_f16s1`, `_f16s2`: the same head, three
rotation draws).

**CORRECTION (same day, after the full f32 batch landed): it did not carry.**
The projection below was wrong and the lever is dead. Measured over all 49 arms
and 12 complete families:

| level | mean dR | better | mean drho | better |
|---|---:|---:|---:|---:|
| family (12) | -0.0013 | 1/12 | +0.0028 | 8/12 |
| arm (49) | -0.0008 | 14/49 | +0.0025 (sd 0.0155) | 27/49 |

Per-arm drho is a coin flip (27 up / 22 down) and +0.0025 sits inside its own
standard error (0.0155/sqrt(49) = 0.0022). R is consistently slightly negative.
`union_noconf` (+0.0193) and `dev_ca_legacy` (+0.0211) carry almost all of the
positive mass; `union_clean`, `union_lw` and `unionnc_lo` are negative.

The original projection, kept for the record: *"if +0.0193 carries to the 45-arm
all-families ensemble, rho goes 0.7175 -> ~0.7368, above GenScore's 0.7281."*
It rested on a single family's six seeds, which is the same mistake as judging
an arm on one training seed -- one level up. See
`2026-09-14_f32_did_not_replicate.md`.

**And R needs something different from rho.** R is nearly flat in family count
(0.8086 at 4 families, 0.8112 at 12) while rho climbs, so more families will not
close R. R needs better individual arms, which points at training data -- the
IC50 corpora now building. The two live levers address the two halves of the
gap, and neither is a head change.

**Training data was halved on purpose.** `--affinity-types` defaults to
`KD,KI,IC50`; the e250 corpora passed `KD,KI`. The V1-era full corpora still
exist and show the size of it:

| corpus | label types | complexes |
|---|---|---:|
| `e250aff_pdbbind_noconf` (in use) | KD, KI | 7,587 |
| `affinity_pdbbind` (V1 era) | IC50, KD, KI | 14,047 |
| `e250aff_r1` (in use) | BioLiP2 KD/KI | 8,581 |
| `affinity` (V1 era) | BioLiP2 full | 15,078 |
| GIGN `train.csv` (measured) | -logKd/Ki | **11,904** |

We already train on more data than GIGN (15,351 vs 11,904) and still lose, so
scale alone is not the explanation -- but the IC50 half is free to recover on
the BioLiP side, and it arrives as a NEW FAMILY, which the curve above says is
worth rho independently of the extra data.

`data/pdbbind` is absent, so the PDBbind half cannot be rebuilt without
re-downloading. `data/biolip` (18 GB) is present, so the BioLiP half can.

### The union scale-up is blocked, and it is the half that would move R

R responds to SOURCE UNION, not to family count:

| family | R |
|---|---:|
| `e250_union_noconf` (BioLiP u PDBbind) | 0.8107 |
| `e250_union_clean` (BioLiP u PDBbind) | 0.8075 |
| `e250_rot1` (BioLiP alone) | 0.7913 |
| `e250_pdbbind` (PDBbind alone) | 0.7625 |

Union is worth ~+0.02 R over either source alone, while family count moves R
barely at all (0.8086 at 4 -> 0.8112 at 12). So the R lever is "union with more
data", i.e. mixing an IC50-restored BioLiP with the PDBbind token corpus.

**Two independent facts block it.**

*The val cannot be pinned.* `tokenize_affinity_biolip.py` splits val by UniProt
protein group: `rng.permutation(len(gid_of))`, accumulating groups until
`len(sites) * val_frac`. The pool is derived from the site set, so adding IC50
sites redraws the split, and there is no `--val-pdbs` flag. `mix.py` cannot
repair this -- it only concatenates `.bin`/`.len`, the `.rmsd`/`.comp`/`.dlen`/
`.disp` sidecars, and offsets `.grp`. It applies no exclusion of its own.

*The two halves do not even split on the same key.* `tokenize_affinity_biolip.py`
splits val by UniProt protein GROUP, with a comment saying PDB-level splitting
would let one protein sit on both sides and make early stopping optimistic.
`tokenize_affinity_pdbbind.py:158` splits by PDB:
`val_pdbs = set(pdbs[: int(len(pdbs) * args.val_frac)])`. So a protein can be in
BioLiP's val group and in PDBbind's train under a different PDB id, and only the
432-exclusion patches over it. This applies to the union arms ALREADY IN THE
TABLE, not just to the one cancelled here -- worth a look before any union
number is reported as final.

*PDBbind cannot be re-tokenized.* The 2026-09-13 contamination
(`2026-09-13_the_union_val_was_60_percent_contaminated.md`) was 258 of BioLiP's
432 val PDBs sitting in PDBbind's train, and the fix was to re-tokenize PDBbind
excluding them. That repair is no longer available: the builder needs
`refined-set/` + `v2020-other-PL/` structures, and `data/pdbbind` is gone.

A new BioLiP val would therefore overlap PDBbind's train again -- BioLiP and
PDBbind overlap ~49% -- and the resulting arm's early stopping would be
optimistic in a way no column in a dump reveals. `corpus_ic50_nc.sh` was
submitted for this and then cancelled once the mechanism was read.

So `affinity_union_holdout_pdbs.txt` = `eval_holdout` (9,329) u BioLiP's val 432
= 9,761, and `affinity_noconf_holdout` = that u the 151 contradictory labels =
9,912. Those 432 are excluded from the PDBbind side; BioLiP *contributes* them
as its val (`e250aff_r1` has `val_docs=431`). Excluding them from BioLiP -- which
the cancelled job did -- removes BioLiP's own val and is backwards.

**What remains available:** the matched pair `e250_aff_ic50` vs
`e250_aff_kdki_ctl`, standalone BioLiP corpora differing in exactly
`--affinity-types`. No PDBbind mixing, so no cross-corpus leak; their val is
drawn from their own pool. It answers what restricting labels to KD/KI costs,
and adds a family, but it is not the union arm and should not be expected to
reach `union_noconf`'s R.

## Leak check on the backbone swap

Done because the new arm introduces a backbone into the pipeline, and because
whatever is true here was already true of every affinity number in the project:
`mlm_e250lig3_lo` continues the pinned checkpoint on the SAME token dir.

**The head corpus is clean by construction.** Unlike the pretraining builders,
`tokenize_affinity_biolip.py` and `tokenize_affinity_pdbbind.py` default
`--casf-pdbs` to `data/casf2016_pdbs.txt` rather than to `None`, so the
exclusion happens unless someone actively points the flag elsewhere.

Stated at its true strength: these corpora's `meta.pt` does NOT carry the
`exclusion_ids_sha256` the current builder writes (they were built 2026-09-13,
before or beside that field), so there is no stored hash to match a list
against. What can be checked is that every list the flag could plausibly have
been given contains the entire core set:

| file | ids | CASF core ids absent |
|---|---:|---:|
| `affinity_noconf_holdout_pdbs.txt` | 9912 | **0** |
| `affinity_union_holdout_pdbs.txt` | 9761 | **0** |
| `eval_holdout_pdbs.txt` | 9329 | **0** |
| `casf2016_pdbs.txt` | 285 | 0 |
| `evaluation_pdbs()` | 9328 | **0** |

This is the leak that would matter most -- the head is the only stage that ever
sees a pK -- and it is closed by measurement rather than by reading a flag.

Do not mistake `pdbbind_holdout_eval.txt` (281 ids) for a CASF holdout: it
contains NONE of the 285. It is a disjoint PDBbind evaluation set.

**The pretraining corpus: opt-in exclusion, and the failure mode is narrower
than it first looks.** The MLM's token dir mixes five sources, and `--casf-pdbs`
defaults to `None` in `tokenize_biolip.py` and `tokenize_plinder.py`, so
exclusion there is opt-in. No surviving log carries either the positive line
(`"CASF-2016 exclusion: N PDBs"`) or the warning
(`"No --casf-pdbs given: CASF-2016 NOT excluded (leak risk)"`).

The archived build scripts in `jobs/archive/` are the PREVIOUS generation, not
this corpus: `job_tokenize_biolip.sh` and the two `job_tok_plinder_*_nocasf.sh`
write to `lm_tokens_complex_biolip` / `lm_tokens_{protein,complex}_plinder_nocasf`
against the `xzkjxu9q` tokenizer. None of them names an `e250lig3` directory.
They do show the flag being passed as routine practice -- and passed as the
NARROW list, `data/casf2016_pdbs.txt`.

`tokenize_plinder.py`'s help text warns about exactly that narrow-vs-wide
choice: the published corpus passed `data/eval_holdout_pdbs.txt` (9329), not the
285. **For CASF specifically the distinction does not matter** -- both files
contain all 285 core ids (measured above), so either list excludes the core set
entirely. The narrow list leaks *other* evaluation sets (CrossDocked test,
sbdd-bench), which is a real problem for those tables and not for this one.

So the only way CASF reaches this backbone is `--casf-pdbs` being omitted
outright.

Bounding the exposure rather than leaving it open: the MLM never sees an
affinity label, so the worst case is structural familiarity with a receptor, not
a pK leak, and the head corpus above is clean regardless. Recorded as
inferential, not verified.

## Method note

Sections 2, 4, 5 and 6 fit on the test set. They are diagnosis, not results, and
no arm was selected on them. Scripts are in the session scratchpad rather than
committed, deliberately: they answer "why are we losing", not "what do we
report".
