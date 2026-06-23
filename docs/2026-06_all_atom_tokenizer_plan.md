# All-atom tokenizer redesign — implementation plan

Branch: `feat/all-atom-tokenizer` (cut from `feat/vqvae-clash-loss`).
Status: **Phase A (code) implemented** — see §8 for what landed and the exact
Phase B (compute) commands. The original design write-up follows unchanged
below for context.

---

## 1. Why we are doing this (diagnosis that led here)

The SBDD LM generated distorted / poorly-binding ligands. We ran a multi-faceted
diagnosis (`scripts/eval_sbdd_full.py`, 100 test pockets, scratch `g79let5b` vs
GEOM-finetune `cjp7e60q` vs GT). Findings:

- Molecules are mostly chemically valid (RDKit 94-95%), **but ~30% are
  fragmented** (connected 69-72% vs GT 96%) and **bond ANGLES are often wrong**
  (PoseBusters bond_angles 71-72% vs 87%). Clashes are NOT the main defect.
- **Vina as-is is positive (+8..+14)** because the generated geometry is
  strained, but **local minimisation recovers it to ~-4.6..-4.8** (the
  geometry cost is large but *recoverable by relaxation*).
- **Even relaxed, generated ligands dock ~2 kcal/mol worse than GT (-4.6 vs
  -6.5)** → a *molecule-fitness* gap that is NOT geometry → the LM is not
  generating ligands that complement the pocket.
- **GEOM pretraining did not help** (finetune ≈ slightly worse than scratch).

Root causes (verified against data):
1. **Pocket is backbone-only (N/CA/C) + residue type, no side chains**
   (`PROTEIN_LAYOUT` coord=12). Interactions happen at side chains → the model
   cannot see where they are → cannot learn interactions. Range (8 Å / 128 res)
   is fine; the problem is the missing atoms.
2. **80% of training poses are decoys.** CrossDocked `label=0` = bad/decoy pose
   (RMSD>2 Å to native), built as classifier negatives. cdonly fold0 train =
   1.57M poses, only 20% (303k) are `label=1` good poses. We trained a
   *generative* model on mostly-wrong arrangements → poor molecule fitness.
   (Using "all poses" to reach 1.4B tokens was the mistake.)
3. **token/param mismatch:** 0.3B LM vs ~18-40M clean tokens (good poses only)
   = badly undertrained from scratch. Compensated by GEOM pretrain + rotation
   augmentation + the ~5x token increase from all-atom pockets.

## 2. Agreed design

**Unified, all-atom (heavy atoms only, NO hydrogens) tokenizer** for protein +
ligand:

- One atom descriptor for **both** protein and ligand heavy atoms: element +
  spherical coord (pocket canonical frame) + KNN offsets/elements + atom
  features + a **protein/ligand flag**. → **one VQ-VAE, one codebook.**
- Protein pocket: **residue-level → all heavy atoms** (extract every heavy atom
  of the pocket residues, not just N/CA/C). This puts the interaction-relevant
  side-chain atoms into the representation.
- LM sequence stays `<bos><p> protein-atom-codes </p><l> ligand-atom-codes
  </l><eos>`; the LM generates only the `<l>` block conditioned on `<p>`. The
  `<p>`/`<l>` markers are LM-only; the tokenizer itself does not distinguish.
- **Data: `label=1` (good poses) only** + rotation augmentation (the existing
  GEOM machinery) to recover token count.
- **GEOM pretrain (ligand atoms only) → fine-tune on clean good poses.**

### Why no H (numbers)
Real receptors: **8.3 heavy atoms / residue**. So all-atom protein ≈ **8x** the
current per-residue token count. With pockets averaging 26 residues (max 55):
- heavy-only: pocket ~217 tok avg / ~457 max; total seq ~243 avg / ~490 max →
  **fits block_size 512.**
- with H (~16/res, ~2x): total ~466 avg / ~930 max → **exceeds 512**, needs
  ~1024-2048 context, ~100x attention. And H positions are determined by heavy
  atoms (added deterministically; docking adds them). → **drop H.**

Cost of the heavy-atom redesign: ~5x total tokens, ~25x attention (quadratic),
protein VQ-VAE goes residue→atom level. Bonus: ~5x tokens helps token/param.

## 3. Implementation cascade (order)

1. **Unified atom descriptor** (`src/tokenizers/`, `descriptor_schema.py`):
   - Define a single atom descriptor used for protein AND ligand heavy atoms:
     `element, coord(4 spherical), knn_offsets(4x4), knn_elements(4)` + a
     `source` flag (protein=0/ligand=1) + atom features (see open decision).
   - Protein extraction changes from per-residue backbone to **per heavy atom**
     (parse all heavy atoms of pocket residues; keep residue id / atom name as
     features if useful). Update `src/tokenizers/protein.py` /
     `src/data/descriptors.py` extraction accordingly.
2. **Regenerate descriptor cache** with (a) all-heavy-atom protein, (b)
   `label=1` filter. Add a label filter to `_load_pairs_from_manifest`
   (`src/data/descriptors.py`) — the manifest has a `label` column.
   Prep job like `scripts/prepare_descriptors_*.sh` (cpu, ~tens of min).
   Write to a NEW cache dir (e.g. `data/descriptor_cache_allatom`); keep v4.
3. **Train the unified VQ-VAE** (one codebook over all atoms). Reuse
   `TransformerVQVAE` / `VQVAEModule`; feed protein+ligand atoms (the descriptor
   carries the source flag). New run; `scripts/train_vqvae*.sh` pattern. Keep
   `train_vqvae.py`'s inode-safe guard (skip extraction when cache exists).
4. **Re-tokenize** complexes → `<bos><p> prot-atoms </p><l> lig-atoms </l><eos>`
   with rotation augmentation. Reuse `scripts/tokenize_dataset.py` +
   `src/data/token_io.py` + the rotation primitive
   (`src/tokenizers/ligand.py: rotate_ligand_descriptor`,
   `geometry.py: random_rotation_matrix`). **Also re-tokenize GEOM** with the
   new VQ-VAE's ligand-atom codes (`scripts/tokenize_geom.py`).
5. **LM:** new flat vocab = specials + ONE atom codebook range (no separate
   protein/ligand ranges; update `src/tokenizers/lm_vocab.py`). GEOM pretrain
   (ligand-only, empty pocket) → fine-tune on clean good poses
   (`scripts/train_lm.py --init-from`).
6. **Evaluate** with `scripts/eval_sbdd_full.py` (multi-faceted: validity /
   PoseBusters geometry / Vina) vs the current `g79let5b` / `cjp7e60q`, same 100
   pockets, seed 0.

## 4. Open decision — RESOLVED: Full (ligand-parity)

How rich should the **protein atom** features be? **Decided: Full.** Protein
atoms carry the same chemistry as ligand atoms — element + charge + hybrid +
aromatic + ring + numH — derived from one RDKit parse of the receptor per
receptor (`Chem.MolFromPDBFile`), keyed by `(chain, resnum, atom_name)`, with a
neutral/OTHER fallback when a key is missing. Plus protein-only `aa` (residue
type) and `bb_sc` (backbone/side-chain) context, and a `source` flag.
(Original recommendation was "light"; Full was chosen so the codebook sees the
same chemistry signal on both sides of the interface.)

## 5. Reusable assets
- Rotation augmentation: `random_rotation_matrix`, `rotate_ligand_descriptor`,
  batched spherical converters (`src/tokenizers/geometry.py`).
- Tokenize streaming + packing: `scripts/tokenize_dataset.py`,
  `scripts/tokenize_geom.py`, `src/data/token_io.py`.
- VQ-VAE + LM training: `src/model/vqvae_module.py`, `src/model/lm_module.py`,
  `scripts/train_vqvae.py`, `scripts/train_lm.py` (`--init-from` warm start).
- Multi-faceted eval: `scripts/eval_sbdd_full.py` (run with
  `uv run --with posebusters python`). Geometry-only localiser:
  `scripts/diagnose_geometry.py`.
- inode-safe data: stream from tars; write caches/tokens to Lustre `data/`.

## 6. Key paths / facts (so you don't re-derive)
- Current ligand VQ-VAE (residue-level protein): `pocket-ligand-vqvae/3dvcbp0h/checkpoints/vqvae-epoch=99-val/ligand_coord=0.1501.ckpt`
- VQ-VAE norm stats: `data/descriptor_cache_v4/normalization_stats.pt` (pass via `--norm-stats`)
- LMs: scratch `pocket-ligand-lm/g79let5b/checkpoints/lm-e09-vl1.4088.ckpt`;
  GEOM-finetune `pocket-ligand-lm/cjp7e60q/.../lm-e02-vl1.4965.ckpt`;
  GEOM pretrain `pocket-ligand-lm/gdnesyzx/checkpoints/lm-e01-vl1.8593.ckpt`
- GEOM data: `data/geom/rdkit_folder.tar.gz` (50 GB, served as plain tar; read
  with `tarfile r|*`). Tokenize: `scripts/tokenize_geom.py --geom-tar`.
- CrossDocked manifest: `data/hub_cache/repo/manifest.parquet` (columns incl
  `label`, `cdonly_fold0`, `complex_dir`, `shard_idx`, `pair_idx`). Receptors
  under `data/hub_cache/receptors/`, ligand tars under `data/hub_cache/repo/ligands/`.
- Pocket: `PocketExtractionConfig` 8 Å / 128 res. Pocket ~26 residues avg.
- Vina/obabel/prepare_receptor: `/home/5/uq02055/.local/bin/vina`,
  `/home/5/uq02055/usr/app/babel/bin/obabel`,
  `/home/5/uq02055/usr/app/ADFRsuite/bin/prepare_receptor`.
- TSUBAME: r3n11 = interactive GPU node (shared, watch for OOM contention);
  big jobs via `qsub -g tga-ohuelab`. Confirm node/time before qsub.
- Sibling branches (failed experiments, kept for reference): PR #4
  `feat/vqvae-knn-neighbor-embed` (knn-offset head), `feat/vqvae-clash-loss`
  (clash-hinge loss). Both regressed because adding a target/penalty to the
  small 8-D latent stole capacity.

## 7. Success criteria
Re-run `eval_sbdd_full.py` on the same 100 pockets and compare to current:
- connectivity ↑ (toward GT 96%), bond_angles ↑ (toward 87%),
- Vina local_only ↓ (toward GT -6.5; closing the ~2 kcal/mol molecule-fitness
  gap is the real win — it tests whether side chains + clean poses let the LM
  learn interactions),
- PB-valid ↑.
If the LM overfits (train good, val/test/Vina flat) with 0.3B, consider a
smaller LM (~50-100M, SOTA SBDD scale).

## 8. Implementation status

The redesign is **additive**: the legacy per-residue protein / per-atom ligand
two-codebook path is left intact (eval baselines `g79let5b` / `cjp7e60q` keep
working). Delete it only after Phase B validates the new path (task tracked).

### Phase A — code (DONE, all tests green: 54 passing)

Unified **33-D atom descriptor** `ATOM_LAYOUT` (`descriptor_schema.py`):
`coord(4) + source(1) + element(1) + charge(1) + hybrid(1) + aromatic(1) +
ring(1) + numH(1) + aa(1) + bb_sc(1) + knn_offsets(16) + knn_elements(4)`.
`source` is input-only (no recon head). Recon heads: coord + 6 chemistry on all
atoms; `aa`/`bb_sc` masked to protein rows; `clash` masked to ligand rows.

New / changed files:
- `src/tokenizers/descriptor_schema.py` — `SOURCE_VOCAB`, `BB_SC_VOCAB`,
  `ATOM_LAYOUT` (33-D), `ATOM_RECON_HEADS`, `ATOM_PROTEIN_ONLY_HEADS`.
- `src/tokenizers/atom.py` (new) — `LigandAtomDescriptor`,
  `ProteinAtomDescriptor`, `rotate_atom_descriptor`, `atom_descriptor_to_coords`,
  `precompute_receptor_atom_features` (RDKit receptor chem lookup).
- `src/tokenizers/protein.py` — all-heavy-atom pocket extraction
  (`PocketAtomData`, `precompute_pocket_atom_candidates`,
  `extract_pocket_atoms_from_candidates`); CA-PCA frame unchanged.
- `src/tokenizers/vqvae.py` — `domain="atom"` + per-source loss masking.
- `src/config.py` — `AtomVQVAEConfig` (latent 16, codebook 8192, max_seq_len
  1024), `AtomVQVAETrainingConfig` (mol_batch_size 256 — pockets are long),
  `HubDatasetConfig.good_poses_only`, `LigandLMConfig.atom_codebook_size`.
- `src/data/descriptors.py` — `label==1` filter in `_load_pairs_from_manifest`.
- `src/data/atom_descriptors.py` (new) — `AtomComplexDescriptorDataModule`
  (single pooled "atom" stream; each complex yields protein + ligand seqs),
  `AtomShardedDataset`.
- `src/data/atom_tar_prep.py` (new) — inode-safe tar-streaming cache builder
  with the `label==1` filter.
- `src/model/vqvae_module.py` — `AtomVQVAEModule` (single VQ-VAE, one stream).
- `src/tokenizers/lm_vocab.py` — `AtomLMVocab` (single code range; `<p>/<l>`
  markers retained; `split_sequence` splits by marker).
- scripts: `prepare_descriptors_atom.py`, `train_vqvae_atom.py`,
  `tokenize_dataset_atom.py`, `tokenize_geom_atom.py` (rotation aug; train split
  only).
- tests: `test_atom_descriptor.py`, `test_lm_vocab.py`, `test_atom_stream.py`,
  plus atom cases in `test_descriptor_schema.py` / `test_vqvae.py`.

### Phase B — compute progress

- **Step 1 — cache: DONE.** `data/descriptor_cache_allatom` = **351,006 complexes**
  (cdonly, label==1 ∧ `_min`), 14 GB / 35 shards. Fold0 split: train 203,760 /
  val 22,640 / test 124,606. `normalization_stats.pt` (`atom_mean`/`atom_std`,
  33-D) written. Built on r3n11 (~20 min, tar streaming).
- **Step 2 — VQ-VAE: RUNNING.** qsub `8002541` (gpu_1, h_rt 16h), `scripts/
  job_train_vqvae_atom.sh`: atom VQ-VAE 100 epochs, bs256, codebook 8192,
  latent 16, WANDB offline, ckpt top-3 by `val/atom_coord` under
  `pocket-ligand-vqvae/<id>/checkpoints/atomvqvae-*.ckpt`. ~7 min/epoch
  (GPU compute-bound; throughput flat in batch/workers ~970 samp/s) → ~13 h.
  Smoke confirmed loss 186→58 over 114 steps.
- **Next:** inspect VQ quality (val/atom_coord, codebook util) → pick best ckpt
  → `tokenize_dataset_atom.py` (rotation aug) + `tokenize_geom_atom.py` →
  LM GEOM-pretrain → fine-tune → `eval_sbdd_full.py`.

### Phase B — remaining steps (each qsub confirmed with node/time first)

1. **Cache** (CPU, tar streaming, inode-safe):
   `python scripts/prepare_descriptors_atom.py --source-types cdonly
   --max-residues 50 --cache-dir data/descriptor_cache_allatom --num-workers N`.
   **Good-pose definition (verified on the manifest):** `label` is per
   docking-run FILE, not per pose. `label==1` covers 351,020 `*_min.sdf.gz`
   (1 minimized near-native pose each) + 125,916 `*_docked.sdf.gz` (each holds
   ~20 poses: 1 near-native + 19 decoys). So the default is **`label==1` ∧
   `_min`** (`min_only`, ~226k cdonly fold0-train poses) — taking the docked
   files' 20 poses would re-admit the decoys the redesign exists to remove.
   `--keep-label1-docked` disables it (ablation). The norm-stats
   (`atom_mean`/`atom_std`) are written by the DataModule at train `setup()`,
   not by prep.
   **Measured (smoke, max-residues 50):** doc length (prot+lig atoms + 6
   markers) median 252 / p99 387 / max 431 → **0% exceed 512, so block_size
   512 is safe.** Protein ~221 atoms/complex avg, ligand ~28.
2. **VQ-VAE** (GPU): `python scripts/train_vqvae_atom.py --source-types cdonly
   --codebook-size 8192 --mol-batch-size 256` → `atomvqvae-*.ckpt`.
3. **Tokenize** (GPU): `tokenize_dataset_atom.py` (CrossDocked, `--num-rotations
   K`) + `tokenize_geom_atom.py` (GEOM pretrain). Pass `--norm-stats
   data/descriptor_cache_allatom/normalization_stats.pt`.
   **Measure the doc-length distribution here** and set the LM `block_size`
   (likely 512–640) so no complex doc is split across blocks.
4. **LM**: GEOM pretrain → fine-tune on clean good poses (`train_lm.py
   --init-from`). Set `LigandLMConfig.atom_codebook_size = 8192`.
5. **Generation/eval**: update `generate_ligands_3d.py` / `decoder.py` to the
   unified VQ + `AtomLMVocab`, then `eval_sbdd_full.py` vs `g79let5b` /
   `cjp7e60q` on the same 100 pockets, seed 0.

### Open knobs to tune in Phase B
- `--max-residues` (default 50) vs LM `block_size` — bound so docs fit one block.
- atom codebook size (8192 start) and `latent_dim` (16) — watch codebook util.
- `mol_batch_size` for the VQ-VAE — protein-atom seqs are ~10x longer than
  ligand; 256 is a starting point, raise/lower per GPU memory.
