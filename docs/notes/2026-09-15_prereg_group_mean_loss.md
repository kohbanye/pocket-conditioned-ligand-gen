# Pre-registration: a loss term on the per-target mean (2026-09-15 ~03:10)

Written BEFORE any head is trained and before any CASF number for these arms
exists. The implementation landed first (config field, loss function, CLI flag,
group batching, 4 tests, plus a `.grp` rebuild verified against the corpus);
no arm has been run.

## Why this lever, from the measured diagnosis

`docs/results/` now has the stratified analysis (exploratory, reads CASF
labels). What it found:

* The error is **target-level, not ligand-level**. Per-target mean |error|
  (after a least-squares calibration onto pK): HIV-1 integrase 2.69 against
  GIGN's 0.44, transporter 1.63 vs 0.56, HIV-1 protease 1.64 vs 0.68. Whole
  clusters are offset -- integrase is overestimated by ~2.7 log on all five.
* The mechanism is **shrinkage of target-level differences**. Slope of
  cluster-mean error against the cluster's mean pK: **ours -0.190**, GenScore
  -0.193, GIGN -0.048, OnionNet-2 -0.014. We compress the spread of target
  levels by ~19%; the winners barely do.
* It is **not** ligand size (size-linked residual +0.022 after calibration,
  against -0.36/-0.28/-0.23 for the baselines), not any ligand descriptor
  (|corr| < 0.13), not protein family (bootstrap CIs include 0), not target
  coverage (corr(log training count, error) = -0.079), not resolution, not
  pocket context.
* Removing our 5 worst targets halves the between-target gap to GIGN
  (-0.042 -> -0.023) and flips the within-target part in our favour (+0.018).

**Per-complex smooth-L1 never penalises that failure shape.** A group whose
predictions are uniformly 1 log too low costs exactly what scattered error of
the same size costs. A term on the group's MEAN residual re-weights the
objective toward the between-target component, which is where the deficit is.

This adds **no contact, edge, distance-bin or contact-count input feature** --
the peer-group claim is untouched. It adds no parameters either: it is one
extra term in the loss.

## Prerequisite that landed first: the union corpus had no real groups

`union_noconf`'s PDBbind half was written without a `.grp` stream, so `mix.py`
gave its 7,201 documents singleton groups. PDBbind's
`INDEX_general_PL_name.2020` supplies a UniProt for every entry and the audit's
reconstruction supplies each document's pdbid, so
`experiments/affinity_loop_20260914/rebuild_union_groups.py` rewrote **only the
`.grp` sidecars** (tokens, labels and lengths copied byte for byte, verified by
sha256) into `data/lm_tokens_e250aff_unionnc_grp`:

| | before | after |
|---|---:|---:|
| groups | 9,278 | **3,426** |
| documents in groups of >= 3 | 41.4% | **80.2%** |
| PDBbind-half groups | 7,201 (all singletons) | **2,444** |
| groups spanning both halves | -- | 1,095 |

132 documents whose UniProt is unknown keep singleton groups.

## Arms (one flag differs: `--group-mean-weight`)

| | treatment `e250_unionnc_gm` | control `e250_unionnc_gm_ctl` |
|---|---|---|
| corpus | `lm_tokens_e250aff_unionnc_grp` | the same directory |
| `--group-mean-weight` | **1.0** | 0.0 |
| batching | `--complexes-per-batch 8 --max-per-group 4 --group-batch-partition` | identical |

Both arms use group batching, so batch composition is NOT a confound: the only
difference is whether the group-mean term is in the loss. The weight is one
point, not a sweep -- picking it by CASF score would be selection on the test
set -- and 1.0 is the neutral choice (equal weight with the regression term).
The cap of 4 keeps a batch at most 8x4 = 32 documents, the same size as the
current micro-batch.

**Group batching had to be fixed before it could be used.** The legacy sampler
SUBSAMPLES an oversized group: with real protein groups (max 826 ligands) an
epoch covered **7,133 of 15,351 documents (46.5%)**, so more than half the
corpus would go unseen each epoch for both arms. `--group-batch-partition`
splits a large group into consecutive chunks that land in different batches
instead; measured on this corpus it gives **15,351/15,351 (100%)**, batches of
at most 32 (mean 21.3), and every batch holds several ligands of one protein
(722/722). Both arms use it, so it is not a difference between them -- it is
what keeps the comparison from being run on half a corpus.

Everything else is the recipe every recent arm used: `--label-cap 13
--max-epochs 20 --early-stop-patience 4 --block-size 640`, backbone
`mlm_e250lig3` e02, default `ligand` readout, seeds 7, 11, 12, 13, f16 and f32
inference. Same driver for both arms, so equal seed numbers pair.

## Amendment 1 (2026-09-15 10:40): the first control was not group-batched

Written BEFORE any CASF number for this round was read. What happened, from the
artifacts rather than from memory:

* The treatment's heads ran **722 steps per epoch**, the control's **479**.
  15,351 documents / 32 = 479, i.e. the control used plain random batching.
* Both arms' checkpoints record `group_batch_partition=True, max_per_group=4,
  complexes_per_batch=8`, so the flags were passed. But
  `RescoreDataModule._loader` selected `GroupBatchSampler` only when
  `listwise_loss_weight > 0 or group_mean_weight > 0`. With weight 0 the
  control fell through to the random-batch loader.
* So the two arms differed in the loss term AND in batch composition -- the
  confound this pre-registration explicitly set out to avoid.

**Fix.** `--group-batch-partition` now turns grouped batching on by itself, and
a NEW control is trained with it at weight 0. The verdict below is computed
against that control (`e250_unionnc_gm_ctl2`).

**The mis-batched arm is kept and reported, not discarded.** It gives a third
arm that separates the two effects:

| arm | batching | `--group-mean-weight` |
|---|---|---|
| A `e250_unionnc_gm_ctl` (already trained) | random | 0.0 |
| B `e250_unionnc_gm_ctl2` (new control) | grouped, partitioned | 0.0 |
| C `e250_unionnc_gm` (already trained) | grouped, partitioned | 1.0 |

C - B is the loss term. B - A is the batching. Both are reported; only C - B
decides G1/G2/G3.

## Predictions, fixed now

* **G1 (the mechanism).** The slope of cluster-mean error against cluster-mean
  pK, measured exactly as in the diagnosis, moves from about -0.19 toward 0:
  treatment slope >= -0.12 on the f32 family ensemble. This is the claim being
  tested; if R improves while the slope does not, the mechanism is wrong even
  if the number is nice.
* **G2 (scoring).** f32 family-ensemble dR >= +0.010 AND treatment R above
  control on >= 3/4 paired seeds.
* **G3 (ranking, guard).** f32 family-ensemble drho >= -0.010. The term
  re-weights toward between-target accuracy, so within-target ranking must not
  pay for it.

**Lever verdict:** works only if G1 AND G2 AND G3 hold. Any subset is reported
as a partial result with its numbers.

## What this round does not do

No headline. If the lever works, a headline needs its own pre-registration,
written before the ensemble is scored, exactly as with the concat and
union-IC50 rounds.

## Cost

8 head jobs (gpu_1, 2:00) ~= 3.2 pt + 16 inference jobs (gpu_1, 0:30) ~= 1.6 pt
~= 4.8 pt, taking the ledger to ~113 of the 200 pt cap.
