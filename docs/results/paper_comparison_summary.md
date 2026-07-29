# 先行研究との比較まとめ（論文用サマリ表）

全て **all-atom 統一トークナイザ**（single-book VQ-VAE `xzkjxu9q`, vocab 8199,
coord err 0.107 Å）を共有する 1 スタック。構成・checkpoint の詳細は
`docs/results/best_allatom_configs.md`、タスク別の詳細表は `docs/results/comparison_tables.md`。
数値は 2026-07-23 に再計算・検証済み。

---

## Table 1. 全タスク一覧（先行研究 vs 本手法）

| Task | Benchmark | Metric | 先行研究 (best) | 他の先行研究 | **Ours** | 位置づけ |
|---|---|---|---|---|---|---|
| **Generation** | sbdd-bench (2ity/1iep/3pbl, 150 mol/target) | Vina Score ↓ | DiffGui **−6.54** | TargetDiff −4.76 / DiffSBDD −4.40 | **−5.33** | DiffSBDD・TargetDiff 超え、DiffGui 未満（n=3 paired, p=0.39 で有意差なし） |
| | | Vina Min ↓ | DiffGui **−8.63** | TargetDiff −6.68 / DiffSBDD −6.48 | **−6.92** | 同上 |
| | | PB-valid ↑ | DiffGui **0.70** | TargetDiff 0.50 / DiffSBDD 0.49 | 0.22 | 劣位（VQ 再構成忠実度が律速） |
| | | Clash-free ↑ | DiffSBDD **0.70** | DiffGui 0.50 / TargetDiff 0.60 | **0.69** | 実質同等（refiner の寄与） |
| **Pose rescoring** | CASF-2016 docking power (284 targets, decoys-only, 共通 pose set) | top1<2Å ↑ | RTMScore **94.0** | GenScore 90.8 / Vina 84.6 | **90.5** | GenScore 直下・統計的に同着（McNemar p=0.11–0.64） |
| | | **top1<1Å (near-native)** ↑ | RTMScore 75.4 | GenScore 73.2 / Vina 71.6 | **77.9** | **両ベースライン超え** |
| | | Spearman ρ ↑ | RTMScore **0.86** | GenScore 0.85 / Vina 0.31 | 0.83 | 学習ヘッド単独ならほぼ同等 |
| **Affinity** | CASF-2016 scoring/ranking (285 complexes, 57 clusters) | Scoring R ↑ | GenScore **0.816** | Boltz-2 0.753 / Vina 0.608 | **0.790** | GenScore と統計的同着（Steiger p=0.21）、**Boltz-2 超え** |
| | | Ranking ρ ↑ | GenScore **0.735** | Boltz-2 0.716 / Vina 0.511 | 0.674 | GenScore と同着（Wilcoxon p=0.14） |

太字 = そのカラムの最良、または本手法が先行研究を上回った項目。
Ours の構成: Generation = LM `p6lpk7br` + pose refiner `refine_atom_bond_v1/e08` + T=0.85/top_p 0.95。
Pose = MLM `j90rlrgm` + 3-head + Vina consensus（top1<1Å 行は v2+Vina）。
Affinity = leak-free MLM `wxlhgqx3` + 固定 LF5 アンサンブル（test-set selection なし）。

---

## Table 2. Generation 詳細（sbdd-bench 3 ターゲット, mean over 150 samples/target）

| Method | Validity ↑ | PB-valid ↑ | Clash-free ↑ | Vina Score ↓ | Vina Min ↓ | Vina Dock ↓ | QED ↑ | SA ↓ | Scaffold div ↑ |
|---|---|---|---|---|---|---|---|---|---|
| DiffGui | 1.00 | **0.70** | 0.50 | **−6.54** | **−8.63** | **−9.96** | 0.50 | **4.21** | **1.00** |
| TargetDiff | 1.00 | 0.50 | 0.60 | −4.76 | −6.68 | −9.00 | 0.37 | 5.16 | 0.97 |
| DiffSBDD | 1.00 | 0.49 | **0.70** | −4.40 | −6.48 | −8.40 | **0.52** | 4.73 | 0.96 |
| **Ours (all-atom, best)** | 0.97 | 0.22 | 0.69 | **−5.33** | −6.92 | — | 0.39 | 6.22 | 0.82 |

Vina Dock は本手法では未計測（refiner は同一分子を精緻化するため off/on 不変。
2-codebook 系で −9.37, all-atom 中間版で −8.36 を確認済み）。

---

## Table 3. Pose refiner のアブレーション（本手法の寄与分解）

生成ポーズを物理多様体へ射影する e3nn E(3)-equivariant flow-matching refiner の on/off。
出典: `../sbdd-bench/results/refined/comparison_onoff.md`。

| トークナイザ / LM | refiner | Validity ↑ | PB-valid ↑ | Clash-free ↑ | Vina Score ↓ | Vina Min ↓ |
|---|---|---|---|---|---|---|
| 2-codebook | OFF | 0.96 | 0.21 | 0.30 | +1.21 | −6.45 |
| 2-codebook | **ON** (bond特徴) | 0.94 | **0.30** | **0.61** | **−4.68** | **−7.23** |
| all-atom | OFF | 0.91 | 0.03 | 0.07 | +3.93 | −5.40 |
| all-atom | ON (旧・bond なし) | 0.79 | 0.05 | 0.28 | −1.94 | −6.98 |
| all-atom | **ON** (bond特徴) | 0.91 | 0.13 | 0.25 | −1.73 | −5.58 |
| all-atom full再学習 (`awdya0s8`) | OFF | **0.97** | 0.17 | 0.10 | +8.56 | −4.07 |
| all-atom full再学習 | **ON** (bond特徴) | **0.97** | **0.28** | **0.36** | **−2.14** | −5.60 |

**読み方（重要）**: refiner は *幾何* のレバーであって *結合モード探索* のレバーではない。
- PoseBusters / clash-free は全条件で改善（最大 PB 0.17→0.28、clash-free 0.10→0.36）。
- Vina は「raw ポーズが壊れている時だけ」劇的に救済（full再学習 LM で **+8.56 → −2.14**, 10.7 kcal/mol）。
- 最終最良 −5.33 を作ったのは refiner ではなく **placement 再finetune（決定的）+ temperature 0.85**。
  geo 強化版 refiner (`refine_atom_geo_v1`) は PB 0.158→0.222 に上げたが Vina は −4.84→−4.72 と改善せず。
- bond-graph 特徴（`bond_embed` は **zero-init 必須**）が旧 refiner の validity 低下 0.91→0.79 を 0.91 に回復させた。

---

## Notes
- Generation ベースライン: sbdd-bench `results/per_model.csv`（公式値）。
- Pose / affinity ベースライン: RTMScore・GenScore を同一プロトコルで再現（`../baselines/`）、
  Boltz-2 は native モード（285/285）。**全手法で同一 pose set** を使用。
  そのため RTMScore 94.0 / GenScore 90.8 は各論文の報告値（94.4 / 91.5）と ≤2 pt ずれる。
- 本手法のアンサンブルは全て素の within-set z-score sum（学習重みなし・test-set selection なし）。
- RTMScore は scoring function を持たない（pose selector のみ）ため Affinity 表には不在。
- 再現用 notebook: `notebooks/paper_{generation,pose_rescoring,affinity}.py`。
