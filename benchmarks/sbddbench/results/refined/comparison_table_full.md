# SBDD-bench 比較表（3ターゲット 2ity/1iep/3pbl）

Vina は kcal/mol（低いほど良い）。refiner=単発 x1予測。Vina Dock は精緻化で不変。
all-atom は未収束の all-atom LM(8a7umbru)生成のため絶対値が低い（分子の質由来／リファイナーの効きは off→on の差で判断）。

| 手法 | Validity | PB-valid | Clash-free | Vina Score | Vina Min | Vina Dock | QED | SA |
|---|---|---|---|---|---|---|---|---|
| DiffGui | 1.00 | 0.70 | 0.50 | -6.54 | -8.63 | -9.96 | 0.50 | 4.21 |
| TargetDiff | 1.00 | 0.50 | 0.60 | -4.76 | -6.68 | -9.00 | 0.37 | 5.16 |
| DiffSBDD | 1.00 | 0.49 | 0.70 | -4.40 | -6.48 | -8.40 | 0.52 | 4.73 |
| Ours 2cb（生） | 0.96 | 0.21 | 0.30 | +1.21 | -6.45 | -9.37 | 0.39 | 5.74 |
| Ours 2cb + refiner | 0.95 | 0.21 | 0.56 | -4.03 | -7.34 | -9.37 | 0.40 | 5.83 |
| Ours all-atom（生） | 0.91 | 0.03 | 0.07 | +3.93 | -5.40 | n/a | 0.42 | 6.31 |
| Ours all-atom + refiner | 0.79 | 0.05 | 0.28 | -1.94 | -6.98 | n/a | 0.46 | 6.78 |
