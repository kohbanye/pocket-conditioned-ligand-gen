# SBDD-bench 比較（refiner on/off × トークナイザ、3ターゲット 2ity/1iep/3pbl）

Vina は kcal/mol（低いほど良い）。2-codebook refiner ON = bond-graph特徴版（PoseBusters最良, refine_legacy_bond_v1）。
all-atom の bond特徴版 refiner (refine_atom_bond_v1/e08, val RMSD 0.944) は旧 refiner の validity 低下(0.91→0.79)を解消(→0.91)し PB も 0.05→0.13 に改善。

**all-atom full再学習 (awdya0s8, CrossDocked全ポーズ 11.1M docs/2.72B tokens) の結論 — 効果が2軸に分離した:**
- **分子内の質は大幅改善**: PB 0.03→**0.17**(OFF, 6倍) / 0.13→**0.28**(ON, 2.1倍)、validity 0.91→**0.97**、SA 6.31→**5.47**(合成容易化)。← リガンド側データ13.6倍が効いた。
- **ポーズ配置は悪化**: raw Vina Score +3.93→**+8.56**(OFF)、vina_min -5.40→-4.07。← 「legacyと同じ全ポーズ」= **~93%が decoy ポーズ**なので、歪んだ配置を学習してしまった。加えて CrossDocked は **1,638ポケットしかなく**未知ポケットへの条件付けが弱い(val は epoch1で頭打ち→epoch2で悪化)。
- refiner(bond特徴)はこの悪いポーズを強力に救済: Vina **+8.56→-2.14**(10.7 kcal/mol改善)、clash-free 0.10→0.36。
→ 次イテレーション: **good-poses-only(min) + PLINDER でポケット多様性**。分子内の質(PB/validity/SA)の伸びを保ったまま Vina を取り戻す狙い。
Vina Dock は精緻化で不変（同一分子）。all-atom は未収束 all-atom LM(8a7umbru)生成のため絶対値が低い（分子の質由来／refinerの効きは off→on 差で判断）。all-atom の Vina Dock=-8.36 は off/on 共通（同一分子, full redock exhaustiveness 8, n=50）。

| 群 | 手法 | Validity | PB-valid | Clash-free | Vina Score | Vina Min | Vina Dock | QED | SA |
|---|---|---|---|---|---|---|---|---|---|
| 既存手法 | DiffGui | 1.00 | 0.70 | 0.50 | -6.54 | -8.63 | -9.96 | 0.50 | 4.21 |
| 既存手法 | TargetDiff | 1.00 | 0.50 | 0.60 | -4.76 | -6.68 | -9.00 | 0.37 | 5.16 |
| 既存手法 | DiffSBDD | 1.00 | 0.49 | 0.70 | -4.40 | -6.48 | -8.40 | 0.52 | 4.73 |
| 2-codebook | refiner OFF | 0.96 | 0.21 | 0.30 | +1.21 | -6.45 | -9.37 | 0.39 | 5.74 |
| 2-codebook | refiner ON | 0.94 | 0.30 | 0.61 | -4.68 | -7.23 | -9.37 | 0.40 | 5.87 |
| all-atom | refiner OFF | 0.91 | 0.03 | 0.07 | +3.93 | -5.40 | -8.36 | 0.42 | 6.31 |
| all-atom | refiner ON (旧・bondなし) | 0.79 | 0.05 | 0.28 | -1.94 | -6.98 | -8.36 | 0.46 | 6.78 |
| all-atom | refiner ON (**bond特徴**) | 0.91 | 0.13 | 0.25 | -1.73 | -5.58 | -8.36 | 0.42 | 6.35 |
| all-atom **full再学習** | refiner OFF | **0.97** | **0.17** | 0.10 | **+8.56** | -4.07 | n/a | 0.40 | **5.47** |
| all-atom **full再学習** | refiner ON (bond特徴) | **0.97** | **0.28** | **0.36** | -2.14 | -5.60 | n/a | 0.44 | 5.92 |
