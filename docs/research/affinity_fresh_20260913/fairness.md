# Affinity benchmark independent fairness audit — 2026-09-13

Scope: code, binary corpus metadata, checkpoint hyperparameters, baseline harness and split IDs. No prior results/notes Markdown was read. Comments embedded in executable files occasionally contain history; they were treated as assertions to verify, not experimental evidence. This audit does not establish that current checkpoints are contaminated; it identifies what is demonstrated and what remains unproven.

## What is actually verified

- `data/casf2016_pdbs.txt` contains 285 distinct IDs; `data/eval_holdout_pdbs.txt` contains 9,329 and includes all 285.
- Current BioLIP affinity builder filters exclusion PDB IDs and splits validation by UniProt group (`pipelines/corpora/tokenize_affinity_biolip.py:220`, `:243`). This is better than a random complex split for early stopping.
- Actual binary `.len` streams (uint16) contain 8,150/431 train/val documents for `lm_tokens_e250aff_r1`, 7,359/431 for `lm_tokens_e250aff_pdbbind_clean`, and 15,509/431 for `lm_tokens_e250aff_union_clean`. Their validation token binaries have identical SHA256 `657ec1a000b0272e14657eb1d2b72b12bb82b69bb42796b0252b673ab768c715`. Sharing validation inputs is therefore confirmed, not merely a registry claim.
- Metadata drift exists: union `meta.json` says 806 validation documents, while its actual binary and `meta.pt` say 431; PDBbind-clean `meta.pt` says 375 validation documents while actual binary says 431. Use actual streams and reconstruct provenance before publishing a manifest. Counts are documents, not independent labels.
- E250 MLM checkpoint hyperparameters identify `data/lm_tokens_e250lig3_mlm`, 15% masking, block size 512, 3 pretraining epochs, seed 7. That corpus metadata identifies five source corpora and 14,623,540 train documents. It does not identify per-document PDB IDs or an exclusion-list checksum.
- GenScore official local `data/val_set{1,2,3}` each contains 1,500 IDs and has zero direct CASF ID intersection. This says nothing about its training near-duplicates or current model selection beyond those files.

## High-priority audit gaps

### 1. Exact PDB exclusion is not protein/ligand/complex novelty

`src/prolit/data/holdout.py:21` explicitly describes four-character PDB matching. Alternate crystal structures of the same target and ligand, homologous pockets, repeated assays and close ligand analogues remain possible. This is valid as a specifically declared CASF/PDB-ID exclusion protocol, but cannot support a claim of unseen-target or unseen-chemotype generalization.

Required ledger: one record per source complex and downstream document, with PDB ID, biological target/UniProt, sequence cluster, pocket similarity cluster, standardized isomeric ligand identity, scaffold, ligand fingerprint, affinity measurement/source/type, split, transform seed and source release. Deduplicate exact protein–ligand–measurement entities across BioLIP/PDBbind before sample weighting. Report nearest train-neighbor similarity for each evaluation complex. Treat protein novelty and ligand novelty as separate axes; also evaluate their joint holdout. Near-duplicate exposure in unlabeled pretraining is different from labeled leakage but must be disclosed and controlled.

### 2. Full tokenizer-to-head lineage is not yet auditable

The VQ CLI exclusion flag is opt-in (`pipelines/train/vqvae.py:272`); `src/prolit/data/atom_descriptors.py:294` applies it if enabled. The inspected `vq_e250_lig3` checkpoint stores training/model configuration but not the HubDatasetConfig containing this flag. Current MLM source `meta.json` files store source counts but no exact source IDs/exclusion hashes. A file/directory named `nocasf` and a registry comment saying “leak-free” do not prove what a checkpoint trained on.

Further, missing exclusion lists fail open: `src/prolit/data/holdout.py:85` returns an empty set; affinity builders guard filtering with `.exists()` (`tokenize_affinity_biolip.py:221`, `tokenize_affinity_pdbbind.py:143`). Require the requested file to exist and match a manifest hash in reproducible builds.

Normalization has a subtle lineage risk: `src/prolit/data/atom_descriptors.py:674` reuses cached stats unconditionally when the file exists. Newly computed statistics correctly restrict to `train_set` at `:683`, but an existing cache is not keyed by its training split. A newly excluded tokenizer can therefore reuse previously inclusive statistics. This is a risk, not proof of present leakage, and its likely impact is not equivalent to affinity-label exposure. Record/validate the statistics' training-ID hash, rebuild if unknown, and carry the exact stats artifact with each tokenizer.

### 3. Clean union needs a protein-level validation audit

BioLIP validation is UniProt-grouped, but the PDBbind builder combines its source `train`, `val`, `test` CSVs (`pipelines/corpora/tokenize_affinity_pdbbind.py:141`), excludes PDB IDs only (`:143`) and independently makes a random-PDB validation split (`:153`). Combining source CSVs is not automatically wrong when defining a new external CASF test, but those source splits must not later be claimed as held out. Excluding BioLIP val **PDB IDs** from the union does not necessarily exclude other PDB entries of the same validation UniProt or near-identical pocket. Shared validation bytes prove equal evaluation inputs, not equally clean training. Rebuild union exclusion at the same target-cluster granularity as the validation definition. Per-document IDs are currently absent from these corpus directories, so binary-level identity proof is unavailable without reconstruction.

### 4. CASF-directed improvement is exploratory model development

The user's requested failure analysis is scientifically useful. Once per-complex CASF residuals, difficult target types or benchmark correlations influence representation/loss/augmentation choices, CASF is functioning as development data even if no label ever enters gradient descent. Validation-only checkpoint selection does not erase this adaptive use. A future improvement derived from this analysis requires a locked second evaluation set (chronological, target/scaffold-aware and excluded from **all** representation training) for a confirmatory claim. Pre-register the decision rules now; use CASF results as exploratory evidence and historical comparability. Holm correction over the current table does not correct all historical adaptive experiments.

## Baseline comparison and statistics

- Local GenScore harness defaults to `GatedGCN_ft_1.0_1.pth` (`../baselines/run_casf_scoring.py:71` approximately; model filename in parser setup). Its official local example also uses that model, so the choice is not inherently unfair. However, three corresponding seed checkpoints plus GT variants exist. Say “this released GenScore checkpoint” until the intended model family and published protocol are established. Report the three corresponding seeds or an agreed released ensemble alongside our comparable single/ensemble treatment; never pick the weaker GenScore variant after seeing CASF.
- The baseline harness normalizes crystal MOL2 to SDF with OpenBabel, whereas ours parses supplied SDF directly (`benchmarks/affinity-bench/affinity_bench/inference.py:254`). Both use crystal poses, but protonation, aromaticity, stereochemistry, bond order, metals and atom completeness may differ. Verify chemical identity and coordinates, record transformations and use the official preprocessing recipe. Equal input information need not mean forcing every architecture to an identical pocket crop; disclose each input context/cost.
- Separate a **matched-data causal comparison** (same training complexes/labels, independent seeds, fixed simple head for tokenizer ablations) from an **as-released practical comparison** (released baseline's actual training data/compute). More BioLIP labels may improve performance but do not establish a better tokenizer. Boltz-2 is sequence+SMILES in this harness (`baselines.py:49`) while ours/GenScore see the crystal pose, so it belongs in a separately labeled input regime.
- `aggregate.py:54` aligns shared PDB IDs for scoring p-values, but `:74` computes deltas from each method's own available cases. Missing outputs can therefore make the reported delta and tested delta describe different populations. `inference.py:235` silently omits failed cases. Require all 285 unique IDs, identical labels/clusters and 57 complete five-ligand groups for the primary CASF table; separately report failures and a predeclared shared subset if necessary.
- `metrics.py:54` permits groups with only three members and drops constant-prediction Spearman groups. This can reward failure/degeneracy. Require full clusters; predeclare how undefined constant scores are penalized/reported rather than silently averaging them out.
- `metrics.py:75` z-sum sorts arrays but does not join/check matching IDs before adding. Equal-length different subsets could be misaligned. Test-distribution standardization also makes the ensemble transductive; fit means/std on validation if an inductive deployment claim is intended.
- `aggregate.py:111` “beats_both” means positive point estimates only, not statistically established superiority. Report delta confidence intervals and avoid translating this boolean into a significance claim.
- Per-complex correlation uncertainty should respect target clusters (57 CASF groups), not assume 285 unrelated observations. Use paired target-cluster bootstrap for ΔPearson and Δwithin-target metric, with training-seed variation reported separately or through a hierarchical bootstrap. Existing Williams/Steiger comparison accounts for shared predictions but not target dependence. Three paired training seeds are useful diagnostics, weak evidence for tiny gains; sharing seed numbers does not guarantee all stochastic effects cancel.
- Do not assume CASF groups are uniformly congeneric or that ligand size has no within-target signal: code documentation makes that assertion, but it needs actual chemical analysis. Include ligand-only descriptors, target-only and similarity-neighbor baselines to measure shortcut performance. Report between-target vs within-target decompositions and RMSE/MAE alongside correlation.

## Minimal credible path to a tokenizer-centered claim

1. Freeze a data/split/lineage specification; audit labeled and unlabeled exposure and source duplicates; reserve a fresh confirmatory set before developing fixes from CASF residuals.
2. Keep one fixed small pooling+MLP head and matched labeled data across tokenizer/LM ablations. Vary only the representation change, paired training seeds and controlled optimization budget. Report frozen-probe and full-finetuning modes as distinct questions.
3. Train/choose hyperparameters on target/scaffold-aware validation, including residue budget, rotation count, contact losses and mask schedule. Do not choose them by the final test errors.
4. Show that changes improve a mechanistic diagnostic (contact geometry, token stability, chemical identity, pocket sensitivity) **and** affinity under this controlled head. A reconstruction gain alone is not evidence of affinity relevance.
5. Publish complete per-complex predictions, failure counts, baseline checkpoint hashes, lineage manifests and clustered uncertainty. If only exploratory CASF improves, state that; a novel tokenizer paper can make a strong representation claim without asserting unsupported overall SOTA.

## Follow-up: missing predictions are present in the actual comparison

The missingness risk above is realized, not hypothetical. `../baselines/casf_work/scoring_power_genscore.csv` contains 283 cases; ours `results/e250_union_clean_f16/kdki-mean.csv` contains 285. The two missing GenScore IDs are:

| ID | CASF cluster | Actual recorded failure |
|---|---:|---|
| 4k18 | 6 | `scoring_genscore.log:256`: pocket graph cannot be generated; preceding RDKit error reports O atom 159 explicit valence 3. |
| 4mme | 17 | `scoring_genscore.log:268`: unpacking error; preceding RDKit error reports N atom 16 explicit valence 4, followed by ligand graph `None`. |

The log establishes failure stage, not the final chemical root cause. Repair should validate original structures and official preprocessing rather than arbitrarily neutralize atoms to force a prediction. Do not count the local 283-case GenScore correlation as a reproduced complete official CASF result.

Current numerical consequence for **e250_union_clean, seed 7, f16 only** (not a multi-seed summary):

| Evaluation support | Ours Pearson | GenScore Pearson | Ours mean rho | GenScore mean rho |
|---|---:|---:|---:|---:|
| Current unaligned: ours285 vs Gen283 | .787673 | .815898 | .636842 | .735088 |
| Shared283, 55 groups of5 + 2 groups of4 | .787017 | .815898 | .629825 | .735088 |
| Shared55 complete groups, 275 cases | .793160 | .814398 | .652727 | .729091 |

For scoring, shared283 is the most inclusive currently matched subset. For the canonical five-ligand ranking definition, use 55 complete shared groups and label the restricted coverage. Also provide mixed-size shared57 as a sensitivity analysis if useful. Ultimately restore/obtain verified GenScore outputs for both missing cases to make a complete CASF comparison. Missingness does not reverse this seed's deficit, but the apparent ranking gap changes from .09825 (unaligned) to .10526 (shared mixed-size) or .07636 (complete groups).

## Follow-up: critical qualification of the chemistry audit

`audit_chemistry.py` reports differences between an unsanitized RDKit interpretation of supplied SDF and the current heavy-atom reconstruction + sanitization/fallback pipeline. Its 156/285 charge-mismatch figure is an **input-pipeline disagreement count**, not a count of 156 chemically valid molecules corrupted by charge removal, and not evidence that fixing charge alone recovers affinity.

The code clearly discards SDF charge and stereochemical fields (`src/prolit/tokenizers/ligand.py:90` onward) and reconstructs atoms with neutral defaults (`:228` onward). However, raw SDF charge records can themselves be invalid; the present audit also changes hydrogen treatment and sanitization state. Atom-count equality by itself is weaker than explicit per-index element/coordinate mapping verification. A nonzero source RDKit chiral tag is not a complete audit of stereo retained in coordinates, since raw 3D source interpretation and rebuilt conformer handling differ.

Required corrected audit strata: source fully sanitizes; source fails but validated MOL2/CCD template provides a consistent reference; unresolved chemistry. Compare matched atoms under the same sanitization/hydrogen convention, verify charge/tautomer/protonation/stereo against provenance, and report failure reasons separately. Then run a one-change-at-a-time corpus-wide retraining ablation (representation distributions change, so an inference-only parser patch is not a valid efficacy comparison). Chemical canonicalization and preservation of **validated** annotations is the justified P0 recommendation; blindly retaining every supplied SDF charge is not.
