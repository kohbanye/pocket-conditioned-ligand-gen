# M02 heavy-atom pocket crop implementation

`PocketExtractionConfig.pocket_distance` accepts `ca` (unchanged default) and `heavy` (minimum distance of any residue heavy atom to ligand heavy atoms). The same distance selects the closest max_residues residues and drives optional distance order. Canonical frame remains PCA over the selected residue CAs. Legacy deserialized configs lacking the new field fall back to CA. Unknown metrics fail explicitly.

CLI `--pocket-distance {ca,heavy}` is wired in affinity BioLiP/PDBbind corpus builders, unlabeled BioLiP/PLINDER corpus builders, descriptor builder, and affinity inference. Affinity inference additionally exposes `--max-residues`. Affinity corpus metadata records crop metric and cap. These are experimental opt-ins; use distinct caches and run names. No production defaults changed.

CPU audit (`.venv/bin/python docs/research/affinity_fresh_20260913/audit_heavy_crop.py`) on all CASF-2016 crystal complexes:

|Metric|CA8Å/max50|heavy8Å/max50|
|---|---:|---:|
|omitted residue contacts within4Å|181|0|
|max emitted token length|441|472|
|documents over640|0|0|

285/285 crops changed. 124 complexes recovered a total181 contact residues, zero originally retained contact residues lost. This verifies the intended selection effect, not affinity improvement. All contacts here are standard amino-acid residues with CA; metals/waters/cofactors remain excluded by design. Every changed crop also changes the canonical frame, requiring matched retokenization/training.

Tests: `.venv/bin/python -m pytest tests/test_pocket_distance.py -q`: 2 passed. They recover a terminal side-chain contact with CA beyond8Å, verify closest heavy contacts survive max_residues=2, validate distance ordering, legacy no-field configuration, and invalid-option failure. Targeted ruff check passed.

## Corpus and training commands (root controls scheduling/accounting)

The commands below use independent output paths. Existing train/val partition logic and seed remain unchanged. Before production training, compare exact successful complex identities with the CA control: a crop-induced success/failure change must be reported or both arms restricted to common coverage. The PDBbind holdout file must include the BioLiP validation exclusions, as the existing clean union did. Do not replace it with CASF-only IDs.

```bash
M02_VQ='pocket-ligand-vqvae/vq_e250_lig3/checkpoints/atomvqvae-epoch=237-val/atom_coord=0.1021.ckpt'
M02_NORM='data/descriptor_cache_allatom/normalization_stats.pt'
.venv/bin/python pipelines/corpora/tokenize_affinity_biolip.py --ckpt "$M02_VQ" --norm-stats "$M02_NORM" --out-dir data/lm_tokens_m02_biolip_heavy --casf-pdbs data/eval_holdout_pdbs.txt --affinity-types KD,KI --num-rotations 1 --max-residues 50 --pocket-distance heavy --seed 0
.venv/bin/python pipelines/corpora/tokenize_affinity_pdbbind.py --ckpt "$M02_VQ" --norm-stats "$M02_NORM" --pdbbind-dir /gs/bs/tga-ohuelab/ysato/data/P-L --out-dir data/lm_tokens_m02_pdbbind_heavy_clean --casf-pdbs data/affinity_union_holdout_pdbs.txt --affinity-types KD,KI --max-residues 50 --pocket-distance heavy --seed 0
```

Union assembly should reuse the existing clean-union source-ID exclusion and validation-source recipe, replacing only source corpus paths. **Do not blindly invoke mix.py on both train+val outputs:** preserve the BioLiP-only 431-document validation and its exclusion from PDBbind training. Root owns that assembly and ledger.

After verifying union metadata max_len <=640 and coverage, a bounded same-head diagnostic is:

```bash
.venv/bin/python pipelines/train/scoring_head.py --token-dir data/lm_tokens_m02_union_heavy_clean --mlm-ckpt pocket-ligand-mlm/mlm_e250lig3/checkpoints/mlm-e02-vl0.8009.ckpt --label-cap 13 --max-epochs 20 --early-stop-patience 4 --block-size 640 --run-name aff_m02_heavy_s7 --seed 7
```

This reuses the old CA-crop MLM as warm start and tests adaptation to the new crop; it is not a fully matched crop-pretraining experiment. For the latter, rebuild unlabeled MLM corpus with `--pocket-distance heavy`, holdout IDs unchanged, and continue pretraining for the same update/token budget as the CA control before the same head training. Root chooses the bounded scope within cumulative200pt. No qsub was executed by this agent.


## Count correction and fresh paired BioLiP preparation

The initial179/122 crop count collapsed `(chain,residue_number)` identities that omit insertion codes. In 2zda and3utu, H60 TYR and inserted H60 TRP collided. Candidate-index-based counting is authoritative: **181 contacts in124 complexes**, exactly matching the prior analysis. The production crop always selected candidate indices correctly; only audit set identity was wrong. `audit_heavy_crop.py` now checks selected atom counts against actual extraction and uses candidate indices.

`m02_prepare_corpus.sh ca data/lm_tokens_m02_ca_fullholdout` and `m02_prepare_corpus.sh heavy data/lm_tokens_m02_heavy_fullholdout` are ready for root-controlled allocated execution, same fixed tokenizer and float32 path, legacy chemistry, seed0, full holdout+CD exclusion. `--write-site-audit` records source identities and statuses for both. Run `check_m02_corpora.py CA_DIR HEAVY_DIR` before training; it fails on cohort/status/split/label-order differences and lengths>640. Do not silently continue on a mismatched cohort.

CPU source-only audit (`audit_biolip_source_split.py`) finds10463 eligible KD/KI pK2–13 sites; exclusions leave8644 sites with train8119/val525 before structural filters. PDB, UniProt and group train/val intersections are empty; no excluded IDs remain. These counts do not promise emitted corpus counts: parser/structure/heavy-atom filters still apply. This fresh full-holdout split differs from the historical8150/431 emitted corpus, so both CA and heavy must be freshly built for a fair pair. See source CSV and summary JSON for identities and exclusion hash.

## Reconstruction and other-task regression guard

Read-only inspection: `benchmarks/recon-bench/recon_bench/adapters/own_allatom.py` delegates reconstruction to `benchmarks/recon-bench/scripts/own_allatom_reconstruct_cli.py`; `run_reconstruction.py` exposes `--allatom-arms`, `--limit`, `--targets`, `--pb-valid`. It does **not** expose a heavy-distance crop toggle, so an unchanged recon-bench command would silently assess the old crop. Do not edit user-modified recon-bench files for this experiment.

Before accepting M02, use a deterministic same-source development subset from the successful paired BioLiP manifest (e.g. first32 sorted IDs, fixed before scores), keep exact same ligand atom identities, and reconstruct CA/heavy views with the same VQ weights. Invert each respective PCA frame to world coordinates, then compute ligand atom-correspondence RMSD without independent alignment (placement), aligned ligand RMSD (internal shape), bond-length error, element/charge/aromatic accuracy, and contact fidelity. Compare protein reconstruction only on **common original candidate atoms** (track insertion-aware identity or coordinates+atom-name mapping), not all atoms from two different crops. Report newly recovered contacts separately; extra atoms must not mask degradation on common atoms. Freeze decoder/checkpoint and report failures.

A practical acceptance gate needs predeclared tolerances based on paired dev measurement noise; do not invent a favorable tolerance after observing outcomes. Since M02 is opt-in with unchanged weights/default, other existing tasks are behavior-preserved under default CA config, but promoting heavy crop to their defaults would require dedicated same-head pose-ranking and generation validations. Neither prediction improvements nor reconstructed geometry preservation has been established by the CPU selection audit.

Ready-to-run reconstruction probe (allocated execution owned by root):

```bash
.venv/bin/python docs/research/affinity_fresh_20260913/m02_reconstruction_gate.py --corpus data/lm_tokens_m02_ca_fullholdout --ckpt pocket-ligand-vqvae/vq_e250_lig3/checkpoints/atomvqvae-epoch=237-val/atom_coord=0.1021.ckpt --norm data/descriptor_cache_allatom/normalization_stats.pt --out docs/research/affinity_fresh_20260913/m02_reconstruction_dev --limit 32
```

This script reuses existing reconstruction encode/decode helpers through read-only import; it does not modify recon-bench. It selects lexicographically first accepted validation complexes, records them before scoring, and emits two rows per complex in `paired_reconstruction.csv`. Chemistry fidelity targets are the existing legacy input descriptors, **not independently validated chemistry ground truth**. The default `cuda` execution was prepared but not executed by this agent; `--device cpu` is available for a very small smoke. CLI import/help and AST parsing passed. Interface contacts are scored on common protein atoms, while crop recovery is recorded separately by the CPU audit. A single script's metrics do not replace pose-scoring/generation regression evaluation.

## Runtime smoke and four-corpus interface (2026-09-14)

`m02_reconstruction_gate.py` now requires `--paired-corpus`; `--corpus` is the CA corpus and `--paired-corpus` the heavy corpus. It verifies their metadata crop/chemistry/cohort settings when metadata exists, rejects duplicate source identities, selects only accepted validation sites, verifies those identities/splits/groups/labels are accepted in the counterpart, and checks identical ligand atom coordinate order after preparation. Each row carries full source-site fields. Output directories and selected-site JSON are created before metric evaluation.

Run the two within-chemistry comparisons as follows, using root's actual four corpus directories:

```bash
.venv/bin/python docs/research/affinity_fresh_20260913/check_m02_corpora.py "$CA_LEGACY" "$HEAVY_LEGACY" --block-size 640
.venv/bin/python docs/research/affinity_fresh_20260913/check_m02_corpora.py "$CA_VALIDATED" "$HEAVY_VALIDATED" --block-size 640
.venv/bin/python docs/research/affinity_fresh_20260913/m02_reconstruction_gate.py --corpus "$CA_LEGACY" --paired-corpus "$HEAVY_LEGACY" --chemistry-mode legacy --validated-cohort --ckpt "$M02_VQ" --norm "$M02_NORM" --out "$M02_OUT_LEGACY" --limit 32
.venv/bin/python docs/research/affinity_fresh_20260913/m02_reconstruction_gate.py --corpus "$CA_VALIDATED" --paired-corpus "$HEAVY_VALIDATED" --chemistry-mode validated-v1 --validated-cohort --ckpt "$M02_VQ" --norm "$M02_NORM" --out "$M02_OUT_VALIDATED" --limit 32
```

All four builders use `--validated-cohort`. The legacy pair parses the validated cohort and then discards chemistry metadata exactly as its corpus builder does; the validated pair passes atom-aligned chemistry to `LigandAtomDescriptor.compute`. Do not compare CAlegacy directly to heavyvalidated as the crop effect.

The corpus checker now defaults to block512 (override explicitly640 only when downstream actually uses640). It verifies first_document offsets, actual int32 `.grp` arrays, manifest-document coverage and labels, unique site identities, and no accepted train/val PDB/UniProt/group intersection. Identical pK arrays alone are insufficient to establish source order.

CPU real-VQ runtime smoke passed for both legacy and validated modes on source-validation complex1b5q/MD2, two crop rows per mode. Because the available tokenization smoke contains no accepted val documents, the reconstruction smoke used a clearly separate synthetic single-site manifest derived from the source split audit; this verifies runtime and correspondence, not acceptance rates of future corpora. Outputs are `/tmp/m02_gate_smoke_legacy/paired_reconstruction.csv` and `/tmp/m02_gate_smoke_validated/paired_reconstruction.csv`. No jobs submitted.

One legacy smoke complex showed ligand world RMSD0.905→0.726Å but common-protein RMSD0.327→0.393Å and common contact recall0.699→0.602. These values are not an efficacy estimate or a criterion selected after observation; they demonstrate that contact recovery alone does not establish reconstruction non-regression. Await the fixed paired development subset and downstream task checks before promotion.
