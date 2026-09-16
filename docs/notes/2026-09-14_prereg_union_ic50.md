# Pre-registration: PDBbind IC50 in the source union (2026-09-14 ~22:20)

Written BEFORE either corpus is built, before any head is trained, and before
any CASF number for these arms exists.

## Why this lever, stated before the result

* The only intervention that ever moved scoring R here by ~0.02-0.035 was the
  **source union**: `union_clean` 0.7896 vs `rot1` (BioLiP only) 0.7544, with
  PDBbind alone worst at 0.7403. Twelve other levers are refuted.
* BioLiP IC50 alone bought R +0.005 and cost rho 0.018-0.033 on 0/4 seeds
  (headline verdict, rule 6). **PDBbind IC50 is a different source** --
  PDBbind-curated rather than BioLiP-scraped -- and the union's gain was
  cross-source, so whether PDBbind IC50 behaves like BioLiP IC50 is untested.
* The unused labels are real: PDBbind v2020 KD/KI/IC50 is 16,258 complexes
  against 9,936 for KD/KI, i.e. +64% before exclusions.
* It is corpus-side. Tokenizer, MLM, head architecture and head recipe are
  unchanged, so the head stays one MLP (the paper's contribution is the
  tokenizer, not the head).

## Arms: exactly one flag differs (`--affinity-types`)

| | treatment `e250_unionic50` | control `e250_unionkdki` |
|---|---|---|
| BioLiP half | `data/lm_tokens_e250aff_ic50` (KD/KI/IC50; 13,590 train / 720 val docs) | `data/lm_tokens_e250aff_kdki_ctl` (KD/KI; 7,714 / 492 docs) |
| PDBbind half (NEW) | `lm_tokens_e250aff_pdbbind_ic50`, `--affinity-types KD,KI,IC50` | `lm_tokens_e250aff_pdbbind_kdki`, `--affinity-types KD,KI` |
| exclusions | `data/affinity_unionic50_holdout_pdbs.txt` | `data/affinity_unionkdki_holdout_pdbs.txt` |

Both PDBbind halves: the same builder run from `snapshot_corpus` (verified
byte-identical to the live tree), the same VQ checkpoint (`vq_e250_lig3`, coord
0.1021), `--max-residues 50 --pocket-distance ca --val-frac 0.05 --seed 0`,
PDBbind v2020 read **read-only** from another member's tree. Mixed with
`pipelines/corpora/mix.py`, which offsets group ids per input.

Exclusion lists are built by `experiments/affinity_loop_20260914/union_ic50_holdouts.py`:
`eval_holdout_pdbs.txt` (contains all 285 CASF core, asserted) UNION its own
BioLiP half's val pdbids UNION every PDBbind complex sharing a UniProt with that
val split. The two lists differ only because adding IC50 sites redraws the
BioLiP protein-group val split.

Heads: seeds 7, 11, 12, 13; `pipelines/train/scoring_head.py` with the control
recipe (`--label-cap 13 --max-epochs 20 --early-stop-patience 4 --block-size
640`), backbone `mlm_e250lig3` e02, default (`ligand`) readout; f16 and f32
inference. Both arms run the same driver, so equal seed numbers pair
trajectories.

### Construction addendum (still before any build or outcome)

* **Val = the BioLiP half's val split only.** This is how `union_noconf` was
  built, verified from its files rather than its metadata: its `val.len` holds
  431 documents, the same as `r1`'s, and `pdbbind_noconf/val.len` was also
  replaced by those 431 (its `meta.pt` still says 386). Plain `mix.py` would
  concatenate both halves' val and silently change the checkpoint-selection
  signal, so the mix job mixes `train` only, copies the BioLiP half's val
  sidecars, and asserts val is byte-identical to that half. The PDBbind half's
  own random 5% val is dropped, as in `union_noconf`.
* Exclusion lists as built (22:25): `affinity_unionic50_holdout_pdbs.txt`
  10,304 ids (eval_holdout 9,329 + 747 val pdbids + 228 same-UniProt PDBbind);
  `affinity_unionkdki_holdout_pdbs.txt` 10,232 ids (9,329 + 523 + 380).
  The val pdbids come from each BioLiP half's `complex_manifest.jsonl`, which
  lists accepted SITES (8,644 / 15,047); the token stream holds fewer DOCUMENTS
  (7,714+492 / 13,590+720) because tokenization rejects some. Excluding by the
  larger set is the conservative direction.
* **Control arm as built (23:15, before any CASF number).** The builder's own
  counts: 8,369 labels after exclusion -> 7,016 ok (no-dir 0, bad-lig 0, bounds
  1,353, setup 0) -> 6,658 train / 358 val. Mixed union `e250aff_unionkdki`:
  14,372 train docs (7,714 + 6,658), val 492 and byte-identical to the BioLiP
  half's, `.rmsd` counts matching on both splits. The 1,353 dropped on bounds
  are mostly `--max-heavy 60`; that cutoff is immaterial for the benchmark,
  where 2 of 285 CASF core ligands exceed 60 heavy atoms (max 67, median 23).

### Disclosure addendum (22:45, builds running, no head trained, no number)

* **The new pair is not `union_noconf` rebuilt.** The `union_noconf` audit
  recovered its build flags: both of its halves used `--max-heavy 80 --seed 7`
  (`jobs/generated/affcorp_rot-1.sh`, `affnoconf.sh`). The new PDBbind halves use
  the builder defaults `--max-heavy 60 --seed 0`, which is what their own BioLiP
  halves (`e250aff_ic50`, `e250aff_kdki_ctl`, `corpus_ic50.sh`) were built with.
  Treatment and control therefore still differ only in `--affinity-types`; the
  comparison to `union_noconf` numbers is NOT paired and is not a claim here.
  The cutoff is not a material handicap on the benchmark: only 2 of 285 CASF
  core ligands exceed 60 heavy atoms (max 67, median 23) and none exceeds 80.
* **The twin screen is a lower bound.** The `union_noconf` audit's controls show
  the frozen criteria flag a CASF complex against its OWN source files only
  190/285 times (PDBbind files) and 73/279 (BioLiP rows) -- CASF coreset files
  split chains at gaps, PDBbind does not. Pdbid-level exclusion of all 285 core
  complexes is verified exactly; the near-duplicate "clean" set is a sensitivity
  analysis on an incomplete screen, applied identically to every method, and is
  not a certification.

## Predictions, fixed now

* **D1 (scoring).** Treatment R above control on >= 3/4 paired seeds AND
  f32 family-ensemble dR >= +0.010.
* **D2 (ranking).** f32 family-ensemble rho NOT below control by more than
  0.010, AND >= 2/4 paired seeds better. (This is where BioLiP IC50 failed:
  0/4 seeds, -0.018 to -0.033.)
* **Lever verdict:** works only if D1 AND D2 both hold. Anything less is
  reported as a refutation or partial result, with its numbers.

## Headline rules, fixed now

* The headline candidate is the **f32 z-sum ensemble of the 4 treatment arms**,
  judged against R > 0.8127 AND rho > 0.7281 AND R > 0.83 on all 285.
* Computed regardless of the lever verdict (as with the concat headline), so
  the verdict cannot decide whether a headline exists.
* **Leak sensitivity.** Both NEW PDBbind halves must be audited under the frozen
  policy before any headline for them is reported; the twin set is Amendment 2's
  12 UNION the `union_noconf` audit UNION these audits, applied identically to
  every method. If a CASF core pdbid appears among training candidates, that is
  a leak and the arm is reported INVALID, not merely sensitivity-adjusted.
* No pool other than the one named above may be compared against the goal.

## Cost

2 corpus builds (~2 pt) + 8 head jobs (~2.8 pt) + 16 inference jobs (~1.6 pt)
~= 6.4 pt of the 101.88 pt remaining under the 200 pt cap.
