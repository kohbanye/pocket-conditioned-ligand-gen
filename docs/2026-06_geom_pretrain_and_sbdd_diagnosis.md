# Session log — GEOM pretraining, eval tooling, and SBDD diagnosis (2026-06)

What this session built, ran, found, and decided. Forward plan is in the
companion doc `docs/2026-06_all_atom_tokenizer_plan.md`.

## Goal we started from
The pocket-conditioned ligand LM generated distorted-shaped ligands. Plan:
pretrain the LM on GEOM (large set of valid 3D conformers) before training on
CrossDocked, with random-rotation augmentation, to teach good ligand geometry.

## 1. GEOM ligand pretraining (built + run)
- `src/data/geom.py`: stream conformers straight from `rdkit_folder.tar` (no
  extraction — inode-safe), molecule-level split (blake2b hash of SMILES),
  top-N conformers by Boltzmann weight.
- `scripts/tokenize_geom.py` + `src/data/token_io.py`: per conformer, **K random
  orientations** (rotation must be injected BEFORE tokenization — the
  spherical coord's θ/φ and KNN directions are frame-dependent; only r is
  rotation-invariant). Emits empty-pocket `<bos><p></p><l> ligand </l><eos>`
  sequences so fine-tuning is the same format with the pocket filled in.
  Cheap rotation primitive: `ligand.rotate_ligand_descriptor` (verified
  equivalent to recomputing the descriptor in the rotated frame).
- `scripts/download_geom.py` (Harvard Dataverse; needs a browser User-Agent),
  `train_lm.py --init-from` warm start, job scripts.
- **Data downloaded**: `data/geom/rdkit_folder.tar.gz` = 50 GB. Dataverse serves
  it as a *plain* tar despite the `.tar.gz` name → read with `tarfile r|*`.
- **Tokenized** (K=32, drugs, max-confs 5): `data/lm_tokens_geom`, 1.46M
  conformers → ~1.44B train tokens (job 7945864).
- **Pretrained** LM (run `gdnesyzx`): 3 epochs, val/loss ~1.86, best
  `pocket-ligand-lm/gdnesyzx/checkpoints/lm-e01-vl1.8593.ckpt`.
- **Fine-tuned** on CrossDocked (run `cjp7e60q`): 3 epochs from the GEOM ckpt,
  val/loss 1.50, best `lm-e02-vl1.4965.ckpt`.

## 2. Generation / geometry eval tooling (built)
- `scripts/eval_generation.py` extended with `--empty-pocket` / `--label`;
  notebooks `pretrain_vs_finetune_eval.py`, `geometry_diagnostics.py`.
- `scripts/diagnose_geometry.py`: GT / VQ-recon / LM-gen 3-arm localiser.
- `scripts/eval_sbdd_full.py`: the multi-faceted eval (validity / PoseBusters
  geometry / Vina score_only+local_only). Run with
  `uv run --with posebusters python`.

## 3. VQ-VAE fix attempts — BOTH FAILED (kept for reference)
Diagnostics localised the clash to the VQ-VAE decoder (per-atom absolute-coord
decode is pairwise-blind). Two attempts to fix it both regressed:
- **knn-offset head** (branch `feat/vqvae-knn-neighbor-embed`, PR #4, run
  `jlwr5r75`): decode KNN relative offsets as an extra head. The 8-D latent got
  overloaded → coord recon 0.15→0.59, clash 77→100%.
- **clash-hinge loss** (branch `feat/vqvae-clash-loss`, run `o0jie0n9`):
  `relu(1.2 - pairwise)^2` on ligand atoms. Also regressed (coord 0.15→0.37,
  clash 77→100%, connected 80→13% — d_floor 1.2 broke real short bonds).
- Lesson: the 8-D latent is near capacity; adding any target/penalty steals it.

## 4. Multi-faceted SBDD diagnosis (100 pockets: scratch g79let5b vs finetune cjp7e60q vs GT)
| metric | GT | scratch | finetune |
|---|---:|---:|---:|
| RDKit valid % | 97 | 95 | 94 |
| connected % | 96 | 72 | 69 |
| PB-valid % | 58 | 41 | 34 |
| PB bond-angle OK % | 87 | 72 | 71 |
| PB no-clash % | 92 | 80 | 84 |
| Vina as-is (median) | -5.5 | +8.1 | +14.3 |
| Vina relaxed (median) | -6.5 | -4.8 | -4.6 |

- "形が崩れる" = mostly **bad bond angles + ~30% fragmentation**, not clash.
- Vina as-is is positive (strained geometry) but **recovers to ~-4.6 on
  relaxation** → geometry cost is recoverable.
- **Relaxed, still ~2 kcal/mol worse than GT** → molecule-fitness gap (LM).
- **GEOM pretraining did not help** (finetune ≈ slightly worse than scratch).

### Root causes found (data-backed)
1. Pocket is **backbone-only (N/CA/C) + residue type — no side chains** →
   can't learn interactions. Range (8 Å / 128 res) is fine.
2. **80% of training poses are decoys** (CrossDocked `label=0`). We trained the
   generative model on mostly-wrong arrangements. Good poses (`label=1`) cdonly
   train = 303k (more than SOTA's ~100k; the 1.5M was inflated with decoys).
3. 0.3B LM is undertrained on clean-only tokens (mitigated by pretrain +
   rotation augment + all-atom's ~5x tokens).

## 5. Decision → all-atom tokenizer redesign
Unified all-atom (heavy only, no H) tokenizer: protein + ligand atoms in one
VQ-VAE/codebook, protein pocket extracted at full heavy-atom level (side chains
in), `label=1` filter, GEOM pretrain → clean fine-tune. Details + step-by-step
in `docs/2026-06_all_atom_tokenizer_plan.md`.

## 6. Branches / PRs / key runs
- `feat/vqvae-knn-neighbor-embed` → **PR #4** (knn experiment + all GEOM/eval
  infra). Pushed.
- `feat/vqvae-clash-loss` (off the above) — clash-hinge + `eval_sbdd_full.py`.
  Pushed.
- `feat/all-atom-tokenizer` (this branch, off clash-loss) — these two docs; the
  redesign will be built here.
- VQ-VAE runs: `3dvcbp0h` (baseline, kept), `jlwr5r75` (knn, bad),
  `o0jie0n9` (clash, bad). LM runs: `g79let5b` (scratch), `gdnesyzx` (GEOM
  pretrain), `cjp7e60q` (GEOM→CrossDocked finetune).
- Eval artifacts: `outputs/sbdd_eval/results.parquet`,
  `outputs/diagnostics/*.npz`, rendered HTML uploaded to Google Drive
  (`claude_sharing`).
