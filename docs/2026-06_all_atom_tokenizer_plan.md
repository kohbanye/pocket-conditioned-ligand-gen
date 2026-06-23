# All-atom tokenizer redesign — implementation plan

Branch: `feat/all-atom-tokenizer` (cut from `feat/vqvae-clash-loss`).
Status at writing: design agreed, **not yet implemented**. This doc is the
hand-off so a future session can execute it.

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

## 4. Open decision (settle in step 1)

How rich should the **protein atom** features be?
- Light: element + residue type + backbone/side-chain flag.
- Full (ligand-parity): element + charge + hybrid + aromatic + ring + numH
  (needs RDKit on the protein, or a residue/atom-name lookup table).
Recommend starting light (element + residue type + bb/sc flag) and adding more
if interaction learning is weak.

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
