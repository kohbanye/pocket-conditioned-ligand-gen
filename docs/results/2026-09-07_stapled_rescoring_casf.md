# Table 2: CASF-2016 pose rescoring — ProLIT 対 ESM3 × ConfSeq

日付: 2026-09-07 / dump: `benchmarks/pose-rescoring-bench/results/rescoring/{e250_div,stapled}`
/ 検定: `scripts/stapled_significance.py`（共通 275 ターゲットで対応を取る）

## 結果

| | DP@2Å | DP@1Å | ranking ρ | n |
|---|---|---|---|---|
| ProLIT (`e250_div`, n_frames=16) | **95.44** | **82.11** | **0.894** | 285 |
| ESM3 × ConfSeq (`stapled`, n_frames=1) | 48.73 | 30.91 | 0.432 | 275 |

| 差 | 値 | p | Holm 補正後 |
|---|---|---|---|
| DP@2Å | **−46.71** | 3.68e−35 (McNemar) | 3.68e−35 |
| ranking ρ | **−0.462** | 7.49e−47 (Wilcoxon) | 7.49e−47 |

## ベースラインは壊れてはいない

corr(head, 真の RMSD) = **−0.382**、ranking ρ = 0.432。乱数なら 0 なので、
**順位付けを学習してはいる**。ただ大きく劣る。

予測どおりの機構である。ESM3 のポケットコードは**同一ターゲットの全ポーズで
バイト単位で同一**（受容体は動かない）ので、ポーズを区別する情報は
**配置 4 トークン + ConfSeq の内部角度**にしか乗らない。ProLIT は共有フレーム上の
約 25 個の原子トークン全体にポーズ情報が分散する。この非対称は
`inference/stapled.py` の docstring に、数字を見る前に書いてある。

予測の精度も裏が取れた。GPU を使う前に CPU で符号化可能性だけ測った時点で
「275/285 ターゲット、21,345/22,492 ポーズ」と出しており、本実行の dump が
**寸分違わず 275 ターゲット 21,345 ポーズ**だった。

## 未解決の非対称: フレーム平均

**ProLIT は公表設定の `--n-frames 16`、貼り合わせ側は 1 で走っている。**
フレーム不変な表現にとって回転平均は恒等操作なので「1 回で平均済み」と
言えるのは正しいが、16 フレーム平均は ProLIT が得ていて**ベースラインが
構造上得られない分散低減**でもある。

差が大きいだけに、プロトコルの非対称に寄りかかった数字にはしたくない。
`--variant e250_div --n-frames 1` を 1 本回して、**プロトコルを揃えた
補助行**を出すべき。現状 `e250_div` の dump は `div_f16.csv` しか無い。

## ConfSeq が書けない 10 ターゲット

1lpg, 2y5h, 3arp, 3b68, 3jvs, 3oe4, 3oe5, 3ozs, 3ozt, 4dld。全て
「0/N ポーズ」で、部分的な失敗が 1 件も無い = 分子単位の失敗。8 件が ConfSeq
本体、2 件が RDKit の kekulize。詳細は
`docs/results/2026-09-06_stapled_casf_encodability.md`。
