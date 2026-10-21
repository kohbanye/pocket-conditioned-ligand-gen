# Four candidate levers, refuted before spending a single point

2026-09-15 ~00:30 JST. After the union-IC50 refutation, a flat re-analysis
produced four candidates. Each was killed by a diagnostic that cost no GPU time
and no budget, by testing the MECHANISM before buying the experiment. Each
diagnostic reads CASF labels, so all of it is exploratory: it can aim a lever,
never support a claim.

## 1. More pocket context -- no between-target signal in the outer shells

The between-target part is where we lose to GIGN (0.3704 vs 0.4128) and
OnionNet-2 (0.4290), and OnionNet-2 sees contact shells to 30 A while our pocket
caps at 50 residues. The plan would have been a corpus rebuild plus an MLM
retrain at longer context -- the most expensive item on the list.

Crude shell descriptors (atom count, residue count, carbon fraction, charged and
hydrophobic residue fractions) per shell, regressed on the 57 cluster-mean
affinities with leave-one-target-out validation:

| features | target-mean R (LOO) |
|---|---:|
| ligand heavy-atom count only | 0.1619 |
| 8 A shell | **0.5578** |
| 8 + 10 A | 0.5381 |
| 8 + 10 + 12 A | 0.5061 |

**Outer shells make it worse, not better.** Separately, the 50-residue cap
clips 0% of CASF complexes at 8 A, 47% at 10 A, 92% at 12 A -- so the current
pocket already holds the entire first contact shell. Dropped.

## 2. Capacity control (`--encoder-lr-scale`) -- already run, never recorded

The encoder memorizes: final `train/loss` 0.0182 against `val/loss` 0.695
(unionkdki s7; ic50 s7 is 0.151 vs 0.661). 99M parameters on 15-25k complexes,
while GIGN and OnionNet-2 win with 1-4M parameters plus hard-coded geometry.

But the knob was **implemented, run on 4 seeds, and left unjudged** in the dev
round (`experiments/affinity_loop_20260914/runs/lr01`, `runs/lr03`; the M12 note
documents the implementation and says "No job submissions"). Reading those logs,
paired by seed under the same driver:

| vs control (dev) | mean dR | better | mean d(macro-rho) | better |
|---|---:|---:|---:|---:|
| `lr01` (scale 0.1) | **-0.052** | 0/4 | -0.010 | 2/4 |
| `lr03` (scale 0.3) | -0.012 | 1/4 | +0.011 | 2/4 |

Monotone dose-response (1.0 > 0.3 > 0.1), agreeing with the CASF-measured frozen
encoder (R 0.709). **The memorization is real but is not the binding
constraint** -- restricting the encoder's adaptation removes useful fitting.
Dropped, and the 4 pt saved.

## 3. Pose-resolution filter -- no mechanism on this benchmark

Our tokens encode geometry, so poor-resolution poses should hurt us more than
methods that read coordinates directly. They do not:

| method | corr(abs error, resolution) |
|---|---:|
| ours (65-arm f32 ensemble) | **-0.046** |
| GIGN (mean of 3) | +0.012 |
| GenScore | +0.081 |
| OnionNet-2 | -0.025 |

And the premise was wrong anyway: **CASF core is curated to <= 2.5 A**
(median 1.93, max 2.50), so there are no low-resolution test complexes. A
training-set resolution cut would spend 20-50% of the data -- which we know
matters -- against no measurable effect. Dropped.

## 4. Monotone recalibration -- the headroom is test-set overfitting

Pearson R is invariant to affine rescaling but NOT to nonlinear monotone maps,
and our predictions are range-compressed, so isotonic recalibration looked like
free R.

| method | R | in-sample isotonic | 5-fold cross-fitted |
|---|---:|---:|---:|
| ours f32 (65 arms) | 0.8100 | 0.8361 (+0.0261) | **0.8008 (-0.0092)** |
| ours f16 (65 arms) | 0.8116 | 0.8359 (+0.0243) | 0.7949 (-0.0168) |
| GenScore | 0.8127 | 0.8391 (+0.0264) | 0.8068 (-0.0059) |
| GIGN (mean of 3) | 0.8550 | 0.8763 (+0.0213) | 0.8566 (+0.0016) |
| OnionNet-2 | 0.8513 | 0.8792 (+0.0279) | 0.8510 (-0.0003) |

Every method gains ~+0.02 in sample and ~0 cross-fitted. There is no
differential headroom, and a map fitted on validation data would gain nothing.
Dropped.

## Where the deficit actually sits

* **Against GenScore we are level, not behind on structure**: between-target
  ours 0.3704 vs 0.3682, within-target 0.4396 vs 0.4445. The R gap is -0.0027.
  The real gap to GenScore is **rho** (-0.028 on the headline ensemble).
* **Against GIGN and OnionNet-2** the gap is between-target, and diagnostic 1
  says it is not about seeing more of the protein. It is first-shell pairwise
  geometry -- precisely those models' architectural inductive bias.
* R > 0.83 needs +0.02. In this project only one intervention ever moved R that
  far: the original BioLiP+PDBbind source union, which is already in every arm.
