# Table 3: 生成 — ProLIT 対 ESM3 × ConfSeq

日付: 2026-09-10
dump: `results/generation/stapled/{per_model.csv,per_target.csv,per_molecule.parquet}`
生成: job 8614785 (gpu_1, 1h40m) / 評価: job 8617298 (cpu_160, 11h25m)
ProLIT 側の数値: `docs/results/2026-08-22_canonical_100.md`

正典 100 ポケット、各 10,000 分子。生成は 100/100 標的が `ok`、
**全標的でちょうど 100 分子**（欠損標的ゼロ）。

## 数値

| 指標 | ProLIT | ESM3 × ConfSeq | 勝ち |
|---|---|---|---|
| validity | 0.9929 | **1.0000** | 貼り合わせ |
| connected | — | 1.0000 | |
| QED | 0.385 | **0.479** | 貼り合わせ |
| SA | 4.14 | **3.14** | 貼り合わせ |
| bond length W1 | 0.042 | **0.039** | 貼り合わせ |
| bond angle W1 | 5.59 | **4.07** | 貼り合わせ |
| **Vina dock (median)** | −7.04 | **−7.23** | 貼り合わせ |
| PB-valid | **0.9202** | 0.8574 | ProLIT |
| strain (median) | **58.3** | 78.3 | ProLIT |
| uniqueness / scaffold 多様性 | **0.640 / 0.385** | 0.456 / 0.322 | ProLIT |
| **clash-free** | **0.9630** | **0.0305** | **ProLIT (31 倍)** |
| **Vina score (median)** | **−1.66** | **+99.99** | **ProLIT** |
| **Vina min (median)** | **−4.34** | +2.60 | **ProLIT** |

参照リガンド自身: score −6.81 / min −6.87 / dock −7.74。

## 読み方: よい分子を、壊滅的な位置に置く

**貼り合わせ側は分子としては ProLIT より良い。** validity は 1.000、
SA 3.14 (ProLIT 4.14) で合成しやすく、QED も高く、結合長・結合角の W1 距離も
小さい。ConfSeq が局所幾何に強いのは Table 1 (結合長 MAE 0.040 対 0.122) と
同じ現象である。

**そして再ドッキング後の親和性でも負けていない** (Vina dock −7.23 対 −7.04)。
分子そのものは良い結合剤である。

**欠損は全て配置に集中している。** 生成ポーズの **96.9% が衝突**しており
(clash-free 0.0305)、そのままの Vina score は **+99.99** — 物理的に成立して
いない配置である。再ドッキングすると −7.23 まで戻る。

これは sbdd-bench README が名指しする形そのもの:

> Vina Dock だけ良いモデルは「生成 pose は悪いが再ドッキングで良く見える」。
> Score↔Dock の差がそれを暴く。

機構は 3 つの表で一貫している:

| 表 | 現象 |
|---|---|
| Table 1 再構成 | ConfSeq は結合長で勝ち、界面 (lDDT-PLI, Contact-F1) で負ける |
| Table 2 rescoring | ポケットコードが全ポーズで同一なので順位付けできない (DP@2Å 48.7 対 95.4) |
| **Table 3 生成** | **分子は良い、配置が壊れている** |

配置情報を 4 トークン (51.93 bits) に押し込むと、既知ポーズの**符号化**は
~0.06 Å でできるが、LM に**生成**させると成立しない。ProLIT は約 25 個の
原子トークンすべてが共有ポケットフレームの球面座標なので、配置がトークン列
全体に分散している。

## 未解決の公平性: ProLIT 側は後処理込み

**この 31 倍差をそのまま論文に書いてはいけない。**

ProLIT の正典値は refiner `refit_e250lig3` + 分子ごとの拘束半径での局所緩和
+ 剛体 + torsion 沈み込みを通した後の数値である (元 doc に「この 0.920 は
後処理込み」と明記)。貼り合わせ側はどれも通していない — 共有フレームが無い
ので refiner を当てる先が無い。

`feedback_ml_only_comparison` (数値最適化を挟んだ数字は不公平) に照らすと、
報告すべきは **ML のみの比較**である。レジストリの `e250_gen` は
`refiner=None` の腕で、貼り合わせ variant が写したのはまさにこの構成だが、
**このベンチではまだ走っていない**。

**これを埋めるには** `--variant e250_gen` を同じパイプラインに通す
(生成 gpu_1 ~2h + 評価 cpu_160 ~11h)。clash-free 0.963 のうち何割が後処理
由来かが分かるまで、「31 倍」は上限であって差そのものではない。

## 再現

```sh
# 生成
python jobs/submit.py --name genstap --resource gpu_1 --hours 8 \
    --env "RESCORING_BENCH_SBDD_PYTHON=$PWD/.venv/bin/python" -- \
    benchmarks/pose-rescoring-bench/scripts/infer_generation_crossdocked.py \
      --variant stapled --skip-eval --n-samples 100 --seed 7
# 評価 (obabel/vina は PATH に無いので両系統の環境変数を渡す)
python jobs/submit.py --name evalstap --resource cpu_160 --hours 12 \
    --env "RESCORING_BENCH_SBDD_PYTHON=$PWD/.venv/bin/python" \
    --env "PROLIT_OBABEL=$(which obabel)" --env "PROLIT_VINA=$(which vina)" \
    --env "SBDD_OBABEL=$(which obabel)" --env "SBDD_VINA=$(which vina)" -- \
    benchmarks/pose-rescoring-bench/scripts/infer_generation_crossdocked.py \
      --variant stapled --skip-gen --seed 7
```
