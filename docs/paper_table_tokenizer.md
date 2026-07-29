# Paper Table 1 — All-atom tokenizer comparison

論文（AAAI 投稿想定）の トークナイザ評価セクション用メイン表。列グループは
FoldToken Table 2 の構成（Global / Success Rates / Local）を本タスク向けに
**Ligand / Pocket / Interface / Cost** へ置き換えたもの。

- 数値の出所: `notebooks/paper_tokenizer.py` → `outputs/tokenizer_eval/`
- 評価集合: CrossDocked `cdonly` fold0 **test**, `label==1`, `_min` ポーズ、
  ランダム 1000 複合体（seed=42）。全 arm が同一複合体・同一 descriptor を見る
  完全な paired 比較。
- `—` は「その手法では構造的に測れない」欄、`TBD` は未実測。

---

## コスト列の定義（ここを間違えると比較が壊れる）

**`bits/atom` と `|V|` は別物**であり、両方を載せる必要がある。

トークン列は `<p>...</p><l>...</l>` とブロックが決定的なので、protein 原子は
protein book の中から、ligand 原子は ligand book の中から選ばれることが文脈から
分かる。したがって **1 原子あたりの情報量は「自分のモダリティの codebook」の
log2** であり、連結語彙の log2 ではない。

| | codebook ベクトル数 | `|V|`（LM の埋め込み行数） | `bits/atom`（レート） |
|---|---:|---:|---:|
| Joint 8192 | 8192 | 8192 | 13 |
| Separate 8192+8192 | 16384 | 16384 | **13** |
| Separate 4096+4096 | 8192 | 8192 | **12** |

**すべてを同時に揃える分割は存在しない。** 8192+8192 はレートを揃える代わりに
codebook ベクトルが 2 倍、4096+4096 はベクトル数と語彙を揃える代わりにレートが
joint より安くなる。この非対称性そのものが「共有」の効果であり、論文では両方を
挟み込む形で報告する。

裏付け（`codebook_stats.csv`, n=1000）: joint は単一 8192 予算の下で、リガンド
原子が 5819 コード・protein 原子が 8146 コードを使い、うち **5795 を共有**して
いる。ligand 専用コードはわずか 24。ハード分割の 4096+4096 ならリガンドは 4096
で頭打ちになるので、**共有により同じ総予算でどちらのモダリティも固定分割より
多くのコードにアクセスできている**。

---

## Table 1（main, 論文本体用）

| Method | `|V|` | bits/atom | pose bits | Lig RMSD↓ | Lig<0.25Å↑ | PB-valid↑ | Pocket RMSD↓ | Pocket lDDT↑ | Res.Rec↑ | lDDT-PLI↑ | Contact-F1↑ | Clash↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Ligand-only tokenizers* | | | | | | | | | | | | |
| Coordinate binning (0.25 Å) | TBD | TBD | TBD | TBD | TBD | TBD | — | — | — | — | — | — |
| Geo2Seq | TBD | TBD | TBD | TBD | TBD | TBD | — | — | — | — | — | — |
| Mol-StrucTok | TBD | TBD | TBD | TBD | TBD | TBD | — | — | — | — | — | — |
| *Protein-only tokenizers (residue-level)* | | | | | | | | | | | | |
| FoldSeek 3Di | TBD | TBD | — | — | — | — | TBD † | TBD † | TBD | — | — | — |
| FoldToken | TBD | TBD | — | — | — | — | TBD † | TBD † | TBD | — | — | — |
| ESM3 structure tokenizer | TBD | TBD | — | — | — | — | TBD † | TBD † | TBD | — | — | — |
| *Ligand-own-frame + pose transmission（＝先行研究の構成）* | | | | | | | | | | | | |
| Ligand-own-frame + oracle pose | 16384 | 13 | ∞ | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Ligand-own-frame + 3 tok pose | 16384 | 13 | 39 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Ligand-own-frame + 2 tok pose | 16384 | 13 | 26 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Ligand-own-frame + 1 tok pose | 16384 | 13 | 13 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| *Shared-frame complex tokenizers* | | | | | | | | | | | | |
| Separate 4096+4096 (capacity-matched) | 8192 | 12 | 0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Coordinate binning (10³ cells × 12 elem) | 12000 | 13.55 | 0 | 1.583 | 0.000 | TBD | 1.613 | 0.672 | — | 0.663 | 0.364 | 0.037 |
| Separate 8192+8192 (rate-matched) | 16384 | 13 | 0 | **0.277** | **0.742** | **0.770** | 0.388 | 0.978 | 0.9992 | **0.9871** | 0.816 | **0.015** |
| **Joint (ours)** | **8192** | 13 | **0** | 0.372 | 0.249 | 0.287 | **0.326** | **0.990** | **0.9995** | 0.9855 | **0.820** | 0.019 |
| *(reference: crystal geometry)* | — | — | — | 0 | 1.000 | 0.980 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |

PB-valid のみ n=300（PoseBusters の energy-ratio チェックが重いため）。
その他は n=1000。coordinate binning は n=8 の暫定値。

† 残基単位トークナイザは backbone (N/CA/C/O) しか復元しないため、all-heavy-atom の
Pocket RMSD / lDDT とは直接比較できない。これらの行には **backbone 限定**の
`BB RMSD` / `BB lDDT` を用い、脚注でその旨を明記する（notebook は両方を出力する）。

**指標の定義**

- `Lig RMSD` — **共通ポケットフレームでの per-atom RMSD**（superposition なし）。
  内部形状ではなく *結合ポーズそのもの* の再現度。Kabsch 版は付録に回す。
- `Lig<0.25Å` — Kabsch RMSD < 0.25 Å を満たした複合体の割合（success rate）。
- `PB-valid` — 再構成座標に**元の結合グラフ**を載せた RDKit mol が PoseBusters
  `mol` チェック全通過。生成ではなく再構成なので結合認識は不要。
- `Pocket RMSD / lDDT` — ポケット全 heavy atom。lDDT は同一残基内ペアを除外。
- `lDDT-PLI` — CASP15 準拠、protein 原子 × ligand 原子ペアのみ、R0 = 6 Å、
  閾値 0.5/1/2/4 Å の平均。
- `Contact-F1` — 4 Å 以内の protein–ligand heavy-atom pair 集合の F1。
- `Clash` — vdW 半径和の 0.75 倍未満に入った ligand 原子の割合。
- `pose bits` — リガンドの剛体変換の送信に要するビット。共通フレームの手法は
  配置を原子ごとに暗黙符号化するので **0**。

---

## ligand-own-frame 行の作り方（Interface 列を機能させる唯一の実験）

リガンド自身のフレームで符号化するトークナイザのトークン列は **SE(3) 不変**で、
ポケット内のどこに置かれるかを一切持たない。復元物を受容体に戻すには 6 自由度の
剛体変換を別途送る必要がある。この予算をビットで表して量子化する。

- 並進: ポケット bounding box を一辺 `2**(bits/3)` 分割した立方格子
- 回転: 決定的に生成した `2**bits` 個の単位四元数から最近傍（q と −q は同一視）

配線検証（n=6, 暫定 VQ）の結果:

| arm | Lig RMSD | **Lig Kabsch** | lDDT-PLI | Contact-F1 | Clash |
|---|---:|---:|---:|---:|---:|
| Ligand-own-frame + oracle | 0.267 | **0.212** | 0.9935 | 0.837 | 0.000 |
| Ligand-own-frame + 2 tok | 0.751 | **0.212** | 0.927 | 0.680 | 0.029 |
| Ligand-own-frame + 1 tok | 3.559 | **0.212** | 0.522 | 0.216 | 0.401 |

**Kabsch RMSD が 3 行とも同一**なのが実装の正しさの証拠。内部形状はポーズ予算に
依存せず、frame RMSD と界面指標だけが崩れる。

### oracle 行の扱い（重要）

oracle 行は joint を上回る。内部形状の再構成が専用 codebook のぶん優れており、
かつ配置が無料だからである。**これは実現不可能な行**なので、論文では

> joint は追加予算 **0 ビット**で lDDT-PLI = X を達成する。ligand-own-frame 系が
> これに並ぶには剛体変換に N ビット（= M トークン）を要する。

という **break-even のレート主張**として書く。「なぜ 3〜4 トークン使わないのか」
という反論に先回りするため、ポーズ予算は oracle / 3 / 2 / 1.5 / 1 トークンで
スイープしてある。

---

## 現時点の実測から言えること / 言えないこと

`outputs/tokenizer_eval/paired_tests.csv`（n=1000, paired Wilcoxon、joint vs
separate 8192+8192）より。

**joint が有意に優れる**

- Pocket RMSD 0.326 vs 0.388 Å（p = 7.8e-158）
- Pocket lDDT 0.990 vs 0.978（p = 1.1e-162）
- 語彙 `|V|` 8192 vs 16384（構造的、検定不要）

**separate が有意に優れる**

- Lig RMSD 0.277 vs 0.372 Å（p = 5.5e-147）、Lig<0.25Å 0.742 vs 0.249
- 結合長 MAE 0.083 vs 0.150 Å、結合角 MAE 6.33° vs 11.18°
- Clash 0.0148 vs 0.0191（p = 7.7e-8）

**差が無い / 無視できる**

- lDDT-PLI 0.9871 vs 0.9855（p = 5.8e-6 だが差 0.0017、0.98 台での 0.2% 差）
- Contact-F1 0.820 vs 0.816（p = 0.14、有意差なし）

### PB-valid が最大の弱点（n=300）

| | PB-valid |
|---|---:|
| 参照（結晶座標） | 0.980 |
| Separate 8192+8192 | 0.770 |
| **Joint** | **0.287** |

**joint の再構成は 7 割の分子で PoseBusters の幾何チェックを落とす。** separate の
0.770 との差は 2.7 倍で、`Lig<0.25Å`（0.249 vs 0.742）や結合角 MAE（11.18° vs
6.33°）と整合する。参照が 0.980 なので、この差はデータ側ではなく**トークナイザ
起因**である。

これは `docs/comparison_tables.md` の生成タスクで PB-valid が 0.22 と低かった
（DiffGui 0.70 / TargetDiff 0.50 / DiffSBDD 0.49）ことの直接の説明になる。
生成モデルではなく **all-atom トークナイザの再構成忠実度がボトルネック**である
ことが、これで独立に裏付けられた。

原因は容量配分と考えられる。joint は単一 8192 予算を protein 8146 コードと
ligand 5819 コードで分け合っており、リガンド専用 8192 を持つ separate に比べて
リガンドの幾何に割ける表現力が小さい。論文では
**「joint は語彙とポケット精度を得る代わりにリガンド幾何を失う」**というトレード
オフとして正直に書き、rate–distortion 曲線（joint 16384 など）でこの軸を示すのが
筋。隠すと生成側の PB-valid 0.22 と矛盾して見える。

界面が引き分けなのは、この 2 arm が**どちらも共通フレーム**を使っており、
相対配置が構成上保存されているため。「1 codebook vs 2 codebook」軸しか振られて
いない。上記 ligand-own-frame 行が入って初めて「共通フレーム vs モダリティ別
フレーム」軸が振られる。

---

## 埋めるための残作業

| 行 / 列 | 状態 |
|---|---|
| `PB-valid` / `BB RMSD` / `BB lDDT` | notebook 実装済み、n=1000 実行中 |
| Separate 4096+4096 | VQ 2 本を学習中（epoch guard 付きで自動登録） |
| Ligand-own-frame × pose sweep | local-frame cache 構築中 → VQ 学習 → 自動登録 |
| Coordinate binning | 自前で最も安く追加できる下界の参照 |
| Mol-StrucTok / Geo2Seq | 公開実装の有無を確認。無ければ論文報告値（分割が違う旨を明記） |
| ESM3 / FoldSeek / FoldToken | backbone 限定列で比較。ポケット切り出しを揃える |

### 公平性のための注意（査読対策）

- **`bits/atom` と `|V|` を必ず併記する。** どちらか一方だけでは比較が成立しない。
  可能なら rate–distortion 曲線も図で出す。
- 学習途中のチェックポイントが表に載らないよう、notebook は epoch >= 90 の
  checkpoint しか arm に登録しない（`TOKENIZER_EVAL_MIN_EPOCH`）。
- 先行研究は自分の test 分割で再計算するのが理想。不可能なら論文報告値である
  ことを明記し、自分の実測値と同じ行に混在させない。
- pose refiner を通した数値と生 decode の数値を混ぜない。本表は**生 decode**。
- 残基単位トークナイザとの比較は backbone 限定列でのみ行う。
