# M01 validated CCD chemistry — implementation and execution contract

2026-09-14. Opt-in prototype implemented; no model training or scheduler submission by this agent.

## Implemented

- `parse_ligand_pdb_text(..., preserve_chemistry=True)` requires a complete heavy-atom-count CCD template match and successful RDKit sanitization. No template/partial match/invalid match fails closed. Default parser behavior is unchanged.
- A versioned `chemistry` payload (`schema=validated_v1`, `source=ccd_template`, sanitized molblock) retains chemical annotations. `validated_heavy_mol` verifies atom element order, connectivity/bond orders and schema, removes explicit hydrogen atoms while retaining H counts, and reorders metadata to the descriptor heavy-atom order.
- `LigandAtomDescriptor.compute(..., chemistry=...)` reads validated features from that molecule instead of rebuilding neutral atoms. Coordinates still come from the original `atoms` array. It preserves existing descriptor dimensionality and default behavior. Stereo metadata survives storage but there is no new stereo categorical descriptor head; its presence alone is not evidence the model uses stereo.
- `PoseEncoder.ligand_seq` and `ligand_descs` forward the optional payload. Existing molecule dictionaries and third-party descriptor duck types keep the old call shape.
- BioLiP affinity builder `--chemistry-mode {legacy,validated-v1}` selects representation. `--validated-cohort` applies identical CCD validation eligibility to the legacy control. `--write-site-audit` independently records the legacy full-cohort audit for other experiments.
- Audited builds require fresh empty output directories and both exclusion sources. `complex_manifest.jsonl` and CSV record source identity, UniProt group/split, pK, acceptance/rejection and document offsets/counts. `meta.pt` records mode, seed, split fraction, source/exclusion/script hashes, model/stat paths and version `affinity_ccd_v1`.
- `--max-complexes` is a smoke limit applied **after** full-cohort split assignment. Never use a smoke corpus as the main training dataset.

The opt-in does **not** blindly preserve malformed SDF charges, does not yet provide a corrected SDF/MOL2 ingestion path, and does not retrofit every VQ/MLM corpus builder. Its first supported dataset is CCD-validated BioLiP. A frozen old tokenizer applied to corrected descriptors is a distribution-shift/adaptation experiment, not a fully corrected tokenizer/LM model.

## Checked

`pytest tests/test_validated_ligand_chemistry.py tests/test_pose_encoder.py tests/test_atom_descriptor.py tests/test_atom_order.py -q`: **23 passed**. Tests cover positive/negative/zwitterionic charge, explicit-H counts, heavy-atom reorder, stereo round-trip, missing/partial templates, stale metadata rejection, and existing batching/order/descriptor behavior. Targeted Ruff checks pass.

Two-site CPU corpus smoke using full 9,329-ID exclusion and seed0 reconstructs 8,644 eligible sites, 2,166 protein groups and 96 validation groups (~525 prescreen sites) before applying the smoke limit. It accepts 3ibl/O59 (one 212-token training document) and rejects 2ca2/SCN under existing heavy-atom bounds. See `m01_corpus_smoke.jsonl`. These are smoke observations, not full-corpus acceptance rates.

Real frozen E250 VQ synthetic CPU encode/decode smoke produces six paired rows in `m01_recon_smoke.csv`. Charged reference atoms are not faithfully decoded in those tiny examples even with corrected input; do not infer efficacy or adopt from this smoke. It demonstrates why reconstruction against a common validated reference is necessary instead of rewarding recovery of the old incorrect descriptor.

## Four-arm bounded diagnostic (root schedules and accounts)

Use the cross product `chemistry-mode={legacy,validated-v1}` × `pocket-distance={ca,heavy}` with **all four** using `--validated-cohort --write-site-audit`. Example one arm:

```bash
.venv/bin/python pipelines/corpora/tokenize_affinity_biolip.py \
  --ckpt pocket-ligand-vqvae/vq_e250_lig3/checkpoints/atomvqvae-epoch=237-val/atom_coord=0.1021.ckpt \
  --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
  --casf-pdbs data/eval_holdout_pdbs.txt --seed 0 --num-rotations 1 \
  --affinity-types KD,KI --max-residues 50 --validated-cohort --write-site-audit \
  --chemistry-mode validated-v1 --pocket-distance ca \
  --out-dir data/lm_tokens_m01_ccdv1_ca
```

Before head training, intersect successful **source-complex** identities across all arms, retaining original group/split assignment; create separate derived corpus directories and manifests for that intersection. Check pK equality, source identity order, unique-label count, lengths, label/length stream counts and train/val UniProt disjointness. The inclusion of identical source IDs in all token corpora does not by itself certify pretraining exclusion or homology novelty. Keep this BioLiP experiment separate from PDBbind union until its target-level validation exclusions and chemistry route are implemented.

For a first small-head adaptation run (root chooses epoch/compute cap):

```bash
.venv/bin/python pipelines/train/scoring_head.py \
  --token-dir data/lm_tokens_m01_ccdv1_ca_common \
  --mlm-ckpt pocket-ligand-mlm/mlm_e250lig3/checkpoints/mlm-e02-vl0.8009.ckpt \
  --label-cap 13 --block-size 640 --lr 0.0001 --max-epochs 20 \
  --early-stop-patience 4 --seed 7 --run-name aff_m01_ccdv1_ca_s7
```

Control uses its corresponding common-cohort corpus with otherwise identical parameters. Main interpretation: ability to adapt the same frozen tokenizer/old MLM initialization and fixed-head architecture to the corrected descriptor distribution. It cannot establish the best achievable chemically corrected VQ representation.

## Mandatory reconstruction gate before adoption

`audit_m01_reconstruction.py` reads accepted validation source identities from the audited manifest, optionally intersects an independent control manifest, rejects overlap with `eval_holdout_pdbs.txt`, and uses the same legacy CA crop/frame for both chemical representations. It emits no-alignment ligand coordinate RMSD; element/charge/hybrid/aromatic/ring/H-count accuracies against a **common validated target** and against each arm's own target; changed-atom and charged-atom accuracies; code-change fraction. Rejected examples stay in the output. No existing recon-bench files are edited.

```bash
.venv/bin/python docs/research/affinity_fresh_20260913/audit_m01_reconstruction.py \
  --ckpt pocket-ligand-vqvae/vq_e250_lig3/checkpoints/atomvqvae-epoch=237-val/atom_coord=0.1021.ckpt \
  --norm-stats data/descriptor_cache_allatom/normalization_stats.pt \
  --site-manifest data/lm_tokens_m01_ccdv1_ca_common/complex_manifest.jsonl \
  --control-manifest data/lm_tokens_m01_legacy_ca_common/complex_manifest.jsonl \
  --max-complexes 16 --device cuda --out outputs/m01/dev_reconstruction16.csv
```

This defaults to deterministic first16 common validation identities, a bounded screening diagnostic. Expand to all common dev cases for an adoption gate; include all rejects and use fixed acceptable noninferiority margins before inspecting results. Protein reconstruction is unchanged for the chemistry-only change because protein input is identical; a future retrained joint codebook can change it and requires both protein and ligand reconstruction checks. The combined heavy-crop change needs its own crop reconstruction control. Other downstream tasks (generation/pose etc.) still need fixed-dev regression tests before replacing shared model artifacts. Better affinity alone is insufficient.

Final backward-compatibility check: current default `parse_ligand_pdb_text` outputs equal the actual `HEAD` implementation on12 cases (four charged/aromatic/stereochemical ligands × valid/missing/invalid templates). Existing `test_sdf_counts.py` adds4 passing tests, for27 targeted tests overall. Matched legacy/validated two-site smoke manifests and label/length/group streams are byte-identical; neutral3ibl/O59's212 tokens are also identical (`m01_matched_smoke.json`). This confirms unchanged behavior where validated chemistry does not differ; it is not evidence of an affinity gain.
