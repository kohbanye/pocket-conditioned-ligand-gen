# Best all-atom configurations (paper baseline)

Frozen record of the best-performing **all-atom tokenizer** setups for the three
tasks the paper reports. Everything below uses the single-book all-atom VQ-VAE
`xzkjxu9q` (vocab 8199). Checkpoint paths are relative to the repo root and were
verified to exist on 2026-07-23. Metrics are on CASF-2016 (285 core complexes,
same 3 generation targets 2ity/1iep/3pbl for Vina).

Shared assets:
- **All-atom VQ-VAE**: `pocket-ligand-vqvae/xzkjxu9q/checkpoints/atomvqvae-epoch=99-val/atom_coord=0.1073.ckpt` (vocab 8199, coord err 0.107 Å)
- **Normalization stats**: `data/descriptor_cache_allatom/normalization_stats.pt` (must accompany the VQ-VAE everywhere)

---

## 1. Generation — best raw Vina score

**Pipeline**: placement-refinetuned LM → all-atom VQ decode → e3nn pose refiner → Vina.

| Component | Value |
|---|---|
| LM checkpoint | `pocket-ligand-lm/p6lpk7br/checkpoints/lm-e02-vl1.0029.ckpt` |
| LM training | placement re-finetune of the all-poses LM `awdya0s8`, lr 5e-5, 4 ep, on `data/lm_tokens_atom_goodmix` (PLINDER + CrossDocked good poses, CASF held out) |
| Pose refiner | `pocket-ligand-refine/refine_atom_bond_v1/checkpoints/refine-e08-r0.9440.ckpt` (e3nn single-shot x1, jitter 0.3 + bond-graph feature, λ_bond 2.0) |
| **Sampling** | **temperature 0.85, top_p 0.95**, max_new_tokens 220 |
| Decode | `scratchpad/gen_atom_target.py` (env `GEN_TEMPERATURE=0.85 GEN_TOP_P=0.95`) |
| Eval | sbdd-bench `run_evaluation.py --dock-modes score min` |

**Result (3 targets, mean over 150 samples/target):**

| metric | ours | DiffGui | TargetDiff | DiffSBDD |
|---|---|---|---|---|
| vina_score (raw) | **-5.33** | -6.54 | -4.76 | -4.40 |
| vina_min | **-6.92** | -8.63 | -6.68 | -6.48 |
| PB valid | 0.22 | 0.70 | 0.50 | 0.49 |
| clash-free | 0.69 | 0.50 | 0.60 | 0.70 |

Beats DiffSBDD and TargetDiff on both vina_score and vina_min; below DiffGui.
Across-target paired test vs DiffGui is not significant (n=3, p=0.39).
Journey: raw vina_score went from +1.21 (pre-loop 2-codebook `own`) → -5.33.
Levers that worked: **placement re-finetune (decisive)** then **temperature 0.85**
(traded surplus diversity — div_uniqueness stayed 0.998 — for placement quality).
The refiner improves PoseBusters but NOT Vina (geometry-only lever). Per-target at
T=0.85: 1iep -9.78, 2ity -2.98, 3pbl -3.24.

Reproduce: `scripts/job_temp_sweep.sh` (arm T=0.85) → sbdd-bench eval.
Full loop log: `<scratchpad>/loop_ledger_40pt.md`.

---

## 2. Pose rescoring — CASF docking power

Encoder-only complex MLM → ligand-token pooling → RMSD-regression head; native-pose
selection among docking decoys. All heads share the docking-best MLM backbone.

| Component | Value |
|---|---|
| **MLM backbone** | `pocket-ligand-mlm/j90rlrgm/checkpoints/mlm-e01-vl0.8528.ckpt` (all-atom, leak-benign*, interface-rich corpus) |
| **Best single head (v2)** | `pocket-ligand-rescore/kdy9d8g3/checkpoints/rescore-e02-vl0.2075.ckpt` |
| v2 recipe | mean-pool, RMSD regression (smooth-L1), batch 32, warm-start j90rlrgm, trained on rigid+torsion synthetic decoys `data/lm_tokens_decoys_v2` |
| Ensemble partner (v6) | `pocket-ligand-rescore/w2usq187/checkpoints/rescore-e01-vl0.1958.ckpt` (meanmax-pool) |
| Physics consensus | Vina per-pose score (`outputs/casf/pose_scores_vina.csv`), z-sum fused |
| Eval | `scripts/eval_casf_rescore.py --score-mode head --exclude-native` |

**Result (CASF-2016, decoys-only, shared 284-target pose set), recomputed by
`notebooks/paper_pose_rescoring.py`:**

| system | top1<2Å | top1<1Å (near-native) | rho(score,rmsd) |
|---|---|---|---|
| **v2 head (single)** | 88.8% | 66.7% | 0.827 |
| **v2 + Vina consensus** | 89.8% | **77.9%** | 0.68 |
| **3-head + Vina** | 90.5% | 75.8% | 0.80 |
| Vina alone | 84.6% | 71.6% | 0.31 |
| GenScore | 90.8% | 73.2% | 0.85 |
| RTMScore | 94.0% | 75.4% | 0.86 |

On the strict **near-native (top1<1Å)** metric the Vina-fused ensemble **beats
both baselines** (77.9 vs 75.4 / 73.2); on standard top1<2Å we sit just under
GenScore and below RTMScore. Baseline top1<2Å here (94.0/90.8) are recomputed on
the identical pose set, ≤2 pt from the papers' own values (94.4/91.5).
Per-pose score dumps: `outputs/casf/pose_scores*.csv`. History: memory
`project_mlm_rescorer`.

\* leak diagnostic was benign: leaked vs clean docking power Δ≈-1.0 (no memorization);
independently reconfirmed with a fully leak-free MLM (`wxlhgqx3`) giving the same
ensemble numbers. j90rlrgm is preferred for pose (interface data ≈4.6× → +5% docking).

---

## 3. Affinity — CASF scoring & ranking power

Same encoder+pooling+MLP, regression target swapped RMSD → pK. Uses the **leak-free**
MLM so the reported affinity numbers carry no CASF contamination risk.

| Component | Value |
|---|---|
| **MLM backbone** | `pocket-ligand-mlm/wxlhgqx3/checkpoints/mlm-e02-vl0.8199.ckpt` (all-atom, leak-free, CASF fully excluded) |
| **Best single head** | `pocket-ligand-rescore/tzqaubl4/checkpoints/rescore-e09-vl0.6196.ckpt` (mean-pool, Kd/Ki-filtered labels) — R=0.762 / ρ=0.630 |
| Head recipe | mean-pool, pK regression (`--label-cap 13`), Kd/Ki-only labels (drop IC50) |
| Eval | `scripts/eval_casf_scoring.py --affinity-head` |

**Best number = fixed 5-model ensemble (no test selection), z-sum:** **R=0.790 / ρ=0.674** (verified 2026-07-23).

Five members (CSV = source of truth for each member's metric):

| member | pooling | labels | checkpoint | CSV | R / ρ |
|---|---|---|---|---|---|
| mean_ic50 | mean | IC50+ | `pocket-ligand-rescore/{b5h4d39z or i1k7d96p}` † | `affinity_power_lf.csv` | 0.764 / 0.577 |
| attn_ic50 | attn | IC50+ | `pocket-ligand-rescore/1djzd0pm` | `affinity_all_attn.csv` | 0.774 / 0.589 |
| mean_kdki | mean | Kd/Ki | `pocket-ligand-rescore/tzqaubl4` | `affinity_kdki_mean.csv` | 0.762 / 0.630 |
| attn_kdki | attn | Kd/Ki | `pocket-ligand-rescore/3c58a53e` | `affinity_kdki_attn.csv` | 0.730 / 0.604 |
| meanmax_kdki | meanmax | Kd/Ki | `pocket-ligand-rescore/ynzqjqvm` | `affinity_kdki_meanmax.csv` | 0.749 / 0.619 |

**Comparison (CASF-2016 scoring/ranking power, `outputs/casf/method_comparison.csv`):**

| method | scoring R | ranking ρ |
|---|---|---|
| GenScore | 0.816 | 0.735 |
| **OUR ensemble** | 0.790 | 0.674 |
| Boltz-2 | 0.753 | 0.716 |
| Vina | 0.608 | 0.511 |

Statistically **tied with GenScore on both** (Steiger p=0.21 scoring / Wilcoxon
p=0.14 ranking) and **beats Boltz-2 on scoring** point estimate. Scoring-oriented
variant (LF5 + xattn head + Vina consensus) reaches R=0.807. Ranking plateaus at
ρ≈0.675 (architecture ceiling under a fixed tokenizer; see memory for the full
9-lever diagnosis). The pose head has essentially zero affinity signal
(R=-0.036) and vice-versa — they are separate specialists.

† mean_ic50 is one of the two early mean-pool IC50 heads (b5h4d39z @19:55 or
i1k7d96p @19:16 on 2026-07-17); `affinity_power_lf.csv` is the authoritative
per-complex output regardless of which hash. All other hashes are confirmed.

---

## Notes for reproduction
- Rescorer eval builds the pocket once per target (native-ligand neighborhood),
  then only the ligand tokens change per pose — `eval_casf_rescore.py` /
  `eval_casf_scoring.py`.
- Ensembles are plain within-set z-score sums (no learned weights, no test-set
  selection) — this is the honest, paper-reportable aggregation.
- Point ledger / metric provenance: memory `project_pose_refiner`,
  `project_mlm_rescorer`, and `<scratchpad>/loop_ledger_40pt.md`.
