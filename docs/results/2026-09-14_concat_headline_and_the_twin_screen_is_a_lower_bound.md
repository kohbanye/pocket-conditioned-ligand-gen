# The concat headline fails; the twin set grows to 19; the twin screen is a lower bound

2026-09-14 22:42 JST. Two pre-registered things resolved at once: the concat
headline (`docs/notes/2026-09-14_prereg_concat_headline.md`, frozen before
`concat_verdict.py` was ever run) and the `union_noconf` leak audit it waited
on. Computed by `experiments/affinity_loop_20260914/headline_concat_prereg.py`;
raw output in `headline_concat_output.txt`.

## 1. Concat headline: FAIL on all three conditions

Goal: R > 0.8127 AND rho > 0.7281 (GenScore, measured here) AND R > 0.83.

| basis | R_all | rho_all | R_clean(266) | rho_clean | vs GenScore R | vs 0.83 |
|---|---:|---:|---:|---:|---:|---:|
| **H1 f32, 8 concat arms (headline)** | **0.7970** | **0.6807** | 0.8024 | 0.7161 | -0.0157 | -0.0330 |
| H2 f32, concat + 65 prior arms | 0.8090 | 0.6965 | 0.8137 | 0.7393 | -0.0037 | -0.0210 |
| H1 f16 | 0.7994 | 0.6860 | 0.8041 | 0.7214 | -0.0133 | -0.0306 |
| H2 f16 | 0.8108 | 0.7140 | 0.8151 | 0.7554 | -0.0019 | -0.0192 |
| family `e250_unionnc_concat` f32 | 0.7942 | 0.6737 | 0.7992 | 0.7089 | | |
| family `e250_kdkictl_concat` f32 | 0.7869 | 0.6860 | 0.7932 | 0.7321 | | |
| best concat seed (max of 8, selection) | 0.7965 | 0.6737 | 0.8020 | 0.7196 | | |

Rule 4 required computing this regardless of the lever verdict, which was DOES
NOT WORK; both were computed and both are reported. The best single concat seed
is a maximum over 8 seeds chosen on the test set and claims nothing (rule 6).

**The concat arms are worse than the pool they were meant to improve**, so
pooling them into the 65 prior arms LOWERS it: H2 f32 0.8090 against the prior
headline's 0.8100. A pre-declared secondary that hurts is still reported; it was
declared before the numbers existed precisely so this could not be dropped.

## 2. The `union_noconf` audit: no direct leak, twin set 12 -> 19

`leak_audit_general/out_union_noconf/TWINS.json`, run under the frozen policy
(`policy_matches_frozen_policy_json: true`), reading no labels and no predictions.

* **`casf_pdbids_in_candidates` is empty.** No CASF core pdbid is itself a
  training candidate in either half. The headline is therefore valid, not INVALID.
* **19 twins**, up from the pre-registered 12. The 7 new ones are all from the
  PDBbind half: `1o0h 3aru 3arv 3tsk 4eor 4f2w 5tmn`.
* **Reconstruction was exact, not a superset**: r1 8,581 sites (8,150/431, 940
  pair groups), pdbbind_noconf 7,587 (7,201/386); every build-log count and the
  per-document `.grp` streams match. The derived scripts reproduce
  `out_kdki_ctl` byte-identically, and reproduce `kdki_ctl`'s builder-written
  manifest (0 of 8,644 rows differ).
* **Build flags recovered** (they were missing, like r1's): both halves used
  `--max-heavy 80 --seed 7`. "noconf" = `affinity_union_holdout_pdbs.txt`
  (eval_holdout 9,329 + exactly r1's 432 val pdbids) + 151 conflicting labels.
* **Deviations, all widening the screen**: every protein chain is screened for
  PDBbind sites (there is no single receptor chain), and ligand keys come from
  both sdf and mol2 -- their SMILES disagree for 3,203 of 6,455 complexes, and
  `1o0h 3tsk 5tmn` are twins only through the mol2 key. A sensitivity set may
  only grow, so widening is the safe direction; it is disclosed, not hidden.

## 3. The twin screen is a lower bound (the finding that matters)

The audit's own positive controls: take each CASF core complex and screen it
against **its own source files**.

| control | self-flagged as a twin | ligand key matches | exact chain sequence matches |
|---|---:|---:|---:|
| PDBbind path (own PDBbind v2020 files) | **190 / 285** | 284 | 181 |
| BioLiP path (own BioLiP rows) | **73 / 279** | 132 | 158 |

The cause is mechanical: CASF coreset files split chains at gaps and PDBbind's
do not, so exact sequence equality and the 90% coverage rule fail on complexes
that are literally the same entry.

**Consequence.** Every "clean" column in this project is a sensitivity analysis
on an incomplete screen, not a leakage certification -- and that includes the
pre-registered 12-complex set, which came from the same frozen BioLiP-path
criteria. What IS exact: pdbid-level exclusion of all 285 core complexes from
every training corpus (every candidate exclusion list contains all 285 with 0
missing, and 0 CASF pdbids appear among training candidates). Revising the
policy (joining chain segments, looser ligand keys) would have to be declared in
advance and applied identically to every method, including the baselines.

## 4. The prior headline's clean column is superseded

The verdict in `2026-09-14_prereg_headline_verdict.md` was on all 285 and does
not move. Its clean column was computed with 12 twins; with 19:

| method | all 285 | clean 273 (12 twins) | clean 266 (19 twins) |
|---|---|---|---|
| ours f32, 65 arms | 0.8100 / 0.7000 | 0.8127 / 0.7250 | 0.8146 / 0.7446 |
| ours f16, 65 arms | 0.8116 / 0.7140 | 0.8139 / 0.7375 | 0.8159 / 0.7554 |
| GenScore | 0.8127 / 0.7281 | 0.8198 / 0.7429 | 0.8184 / 0.7589 |
| GIGN (mean of 3) | 0.8550 / 0.7702 | 0.8664 / 0.7875 | 0.8648 / 0.8054 |
| OnionNet-2 | 0.8513 / 0.7175 | 0.8515 / 0.7286 | 0.8519 / 0.7464 |

On the wider clean set we are still below GenScore on both metrics (-0.0038 R,
-0.0143 rho). Removing twins helps every method; the ordering is unchanged. The
goal remains unmet on every basis.
