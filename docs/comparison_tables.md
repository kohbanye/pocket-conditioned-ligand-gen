# All-atom best vs. existing methods — comparison tables

Our best all-atom model (see `docs/best_allatom_configs.md`) against published
baselines, per task. Arrows give the "better" direction. Bold = our result.
Metrics verified 2026-07-23.

## 1. Generation — Vina & molecular quality
CASF/sbdd-bench targets (2ity, 1iep, 3pbl), mean over 150 samples/target.
Our model = placement LM `p6lpk7br` + refiner `bond1` + sampling T=0.85.

| Method | Vina Score ↓ | Vina Min ↓ | PB-valid ↑ | Clash-free ↑ | QED ↑ | SA ↓ | Scaffold div ↑ |
|---|---|---|---|---|---|---|---|
| DiffGui | **−6.54** | **−8.63** | **0.70** | 0.50 | 0.50 | **4.21** | 1.00 |
| TargetDiff | −4.76 | −6.68 | 0.50 | 0.60 | 0.37 | 5.16 | 0.97 |
| DiffSBDD | −4.40 | −6.48 | 0.49 | **0.70** | **0.52** | 4.73 | 0.96 |
| **Ours (all-atom)** | **−5.33** | −6.92 | 0.22 | 0.69 | 0.39 | 6.22 | 0.82 |

- Vina Score: we **beat DiffSBDD and TargetDiff**, below DiffGui. Across-target
  paired test vs DiffGui is not significant (n=3, p=0.39).
- We are competitive on clash-free; weaker on PB-valid and SA (VQ reconstruction
  fidelity). SA is 1–10, lower = easier to synthesize.

## 2. Pose rescoring — CASF-2016 docking power
285 core complexes, decoys-only (native excluded), identical pose set for all
methods. Our model = MLM `j90rlrgm` + head `v2` (mean-pool, RMSD regression),
optionally in z-sum consensus with Vina.

Numbers below are recomputed on the **shared 284-target pose set** (identical
decoys for every method) by `notebooks/paper_pose_rescoring.py` — reproducible
source of truth. They differ by ≤2 pt from the baselines' paper-reported values
(RTMScore 94.4 / GenScore 91.5) because those are on each paper's own pose set.

| Method | Docking power (top1<2Å) ↑ | Near-native (top1<1Å) ↑ | Spearman ρ ↑ |
|---|---|---|---|
| RTMScore | **94.0** | 75.4 | **0.86** |
| GenScore | 90.8 | 73.2 | 0.85 |
| **Ours — 3-head + Vina** | 90.5 | 75.8 | 0.80 |
| **Ours — v2 + Vina** | 89.8 | **77.9** | 0.68 |
| **Ours — 3-head ensemble** | 89.5 | 70.9 | 0.83 |
| **Ours — v2 (single)** | 88.8 | 66.7 | 0.83 |
| Vina | 84.6 | 71.6 | 0.31 |

- On **near-native (top1<1Å)** our Vina-fused ensembles **beat both baselines**
  (v2+Vina 77.9, 3-head+Vina 75.8 vs RTMScore 75.4 / GenScore 73.2).
- On the standard 2Å docking power we sit just under GenScore and below RTMScore
  (statistically tied, McNemar p=0.11–0.64).
- The learned heads give the best ranking ρ (0.83 ≈ GenScore); adding Vina trades
  ρ for near-native top-1. Pose ensemble = pose heads (v2/v6/v7) + Vina, **not**
  the affinity head.

## 3. Affinity — CASF-2016 scoring & ranking power
285 complexes (57 clusters × 5). Our model = leak-free MLM `wxlhgqx3` +
fixed 5-head ensemble (no test-set selection). `method_comparison.csv`.

| Method | Scoring R ↑ | Ranking ρ ↑ |
|---|---|---|
| GenScore | **0.816** | **0.735** |
| Boltz-2 | 0.753 | 0.716 |
| **Ours (LF5 ensemble)** | 0.790 | 0.674 |
| Vina | 0.608 | 0.511 |

- **Statistically tied with GenScore on both** metrics (scoring Steiger p=0.21;
  ranking Wilcoxon p=0.14).
- **Beats Boltz-2 on scoring** (point estimate 0.790 vs 0.753; a lightweight head
  on crystal poses vs a large structure-prediction model).
- RTMScore is not in this table — it has no scoring function (pose selector only).
- Best single head: mean × Kd/Ki (`tzqaubl4`) R=0.762 / ρ=0.630. Scoring-oriented
  variant (LF5 + xattn + Vina) reaches R=0.807.

---

### Notes
- Generation baselines: sbdd-bench `results/per_model.csv` (official).
- Pose/affinity baselines: RTMScore & GenScore reproduced under identical
  protocol (`baselines/`), Boltz-2 native mode (285/285). Same pose sets.
- All "Ours" ensembles are plain within-set z-score sums — no learned weights,
  no test-set selection (honest, paper-reportable aggregation).
