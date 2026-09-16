# The overlap does not flatter us, and the GIGN bar is an ensemble

2026-09-14. Two measurements, both about what the comparison means rather than
how well we do. Protocol fixed beforehand in
`docs/notes/2026-09-14_prereg_final_ensemble.md` (Amendment 1, rule 7), written
while no f32 or IC50 result existed.

## 1. Leak sensitivity: removing the overlapping complexes helps EVERY method

The twin set is 10 CASF core complexes, derived from the frozen audit artifacts
rather than transcribed:

* **exact protein-chain sequence + ligand identity** (canonical isomeric SMILES
  or RDKit InChIKey), from `final_source_overlap_audit/matches.json`:
  **3** -- `1p1q`, `2al5`, `3dd0`
* **near-duplicate** under `final_near_duplicate_audit/policy.json` (Morgan
  r=2/2048 Tanimoto >= 0.9, global-alignment identity >= 0.9, both coverages
  >= 0.9), from `sequence_qualified_candidates.json` (key `casf_pdb`):
  **8** -- `1o5b`, `2iwx`, `2wbg`, `2wer`, `2yge`, `3d4z`, `3dd0`, `4jsz`

The same 10 are removed from OUR dumps and from every baseline dump.

| method | R (285) | rho (285) | R (275) | rho (275) | dR | drho |
|---|---:|---:|---:|---:|---:|---:|
| ours ens(49) f16 | 0.8115 | 0.7140 | 0.8143 | 0.7339 | +0.0028 | +0.0199 |
| ours family `union_noconf` | 0.8107 | 0.6737 | 0.8126 | 0.6982 | +0.0019 | +0.0245 |
| ours single best (`unionnc_balanced`) | 0.8098 | 0.6754 | 0.8120 | 0.6839 | +0.0021 | +0.0085 |
| GIGN mean-of-3 | 0.8550 | 0.7702 | 0.8664 | 0.7821 | +0.0115 | +0.0120 |
| GenScore | 0.8127 | 0.7281 | 0.8202 | 0.7393 | +0.0076 | +0.0112 |
| OnionNet-2 | 0.8513 | 0.7175 | 0.8521 | 0.7286 | +0.0008 | +0.0110 |
| Boltz-2 | 0.7532 | 0.7158 | 0.7637 | 0.7196 | +0.0106 | +0.0039 |
| Vina | 0.6076 | 0.5105 | 0.6096 | 0.5214 | +0.0020 | +0.0109 |

**Every method gets better when the twins are dropped**, so those complexes are
harder than average for everyone -- the opposite of what a leak that inflates
our score would look like. Our ensemble gains the most rho (+0.0199 against
GIGN's +0.0120), i.e. the overlap was costing us slightly more than it cost
them. The ordering of methods is unchanged on either basis.

**The flag is corpus-wide, not a dev-cohort artifact.** None of the 20 twin
TRAINING pdbs are excluded by `eval_holdout_pdbs.txt` or
`affinity_union_holdout_pdbs.txt` (only `6ql2`, by the noconf list), and they
are BioLiP sites, so they reach `e250aff_r1` and therefore `union_noconf`.
The dev-round note flagged its own numbers; the same overlap sits under every
arm we have.

**Exact protein sequence alone is disclosed, not excluded**: 103 of 285 core
complexes share a sequence with a training site. Same target, different ligand
is ordinary for every PDBbind-trained method, and the audit declines to call it
leakage.

**Not certified:** `cohort.json` carries
`complete_leakage_certification: false`. Homology, pocket similarity, scaffold
similarity and public-pretraining contamination were never examined. This
bounds one measured overlap; it is not an independent-test certification.

## 2. The GIGN bar is a 3-model ensemble

`results/gign/scoring.csv` is the exact arithmetic mean of the three released
checkpoints' predictions (max |difference| 1.8e-15 against `mean3`).

| GIGN dump | R | rho |
|---|---:|---:|
| repeat0 | 0.8448 | 0.6860 |
| repeat1 | 0.8410 | 0.7316 |
| repeat2 | 0.8323 | 0.7105 |
| **single-model mean** | **0.8393** | **0.7094** |
| **scoring.csv (mean of 3)** | **0.8550** | **0.7702** |

Averaging three checkpoints buys GIGN +0.0157 R and **+0.0608 rho** over its own
single-model mean -- more rho than the entire gap we have been trying to close.

Consequences for the table:

* **Ensemble vs ensemble** is the honest comparison for `GIGN mean-of-3`. Our
  ens(49) at 0.8115/0.7140 loses on both, but to an ensemble, not a single model.
* **Single vs single**: GIGN 0.8393/0.7094 against our best single arm
  0.8098/0.6754. We lose R by 0.0295 and rho by 0.0340.
* **Our ens(49) rho 0.7140 already exceeds GIGN's single-model rho 0.7094**, and
  on the clean 275 our 0.7339 exceeds it further. That is an ensemble-vs-single
  comparison and must be labelled as such, not quoted as a win.
* GenScore, OnionNet-2, Boltz-2 and Vina are single models here; GenScore uses
  one released checkpoint (`GatedGCN_ft_1.0_1.pth`).

Every row must therefore carry whether it is a single model or an ensemble.
`report.py` already records that GIGN ships three checkpoints and that
`scoring.csv` is their mean; what had not been drawn is that the bar we were
chasing is not a single-model number.

## Method note

Two loader defects were found and fixed while producing this, and the first
version of the table was discarded:

1. The near-duplicate extraction guessed record keys (`casf`, `casf_pdbid`,
   `pdbid`, `casf_id`) and silently matched none -- the real key is `casf_pdb`.
   It reported a twin set of 3 instead of 10 without failing.
2. Baselines were loaded with `sorted(glob("*.csv"))[0]`, which picks
   `20221121_074758_GIGN_repeat0.csv` ahead of `scoring.csv` and reported GIGN
   as 0.8448/0.6860. GenScore survived only by luck (`.` sorts before `_`, so
   `scoring.csv` precedes `scoring_283_as_run.csv`).

Both would have produced a plausible table. Prefer named dumps
(`scoring.csv` / `kdki-mean.csv`) explicitly and assert the schema of any audit
artifact before trusting a set derived from it.
