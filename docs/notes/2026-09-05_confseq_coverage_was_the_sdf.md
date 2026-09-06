# ConfSeq が CASF を読めないのは CASF の SDF のせいだった

2026-09-05。貼り合わせ baseline (ESM3 × ConfSeq) の実現可能性調査中。

## 症状

ConfSeq のラウンドトリップ (encode → decode) を評価集合で測ったら、
**PoseBusters は全通過、CASF は全滅**という極端な差が出た。

| 集合 | 読み方 | n | encode 成功 | 完全ラウンドトリップ |
|---|---|---:|---:|---:|
| PoseBusters Benchmark | `*_ligand.sdf` | 428 | 428 | **428** |
| CASF-2016 coreset | `*_ligand.sdf` | 285 | 190 | **0** |
| CASF-2016 coreset | `*_ligand.mol2` | 285 | 285 | **285** |

CASF を SDF で読むと RDKit のパース自体が 92/285 で失敗し、パースできた 190 件は
**全件** `decode: KeyError` で落ちる。同じリガンドを **mol2 から読むと 285/285 が
完全に通る**。

## 誤診しかけた道筋

`decode: KeyError` が 190/190 という「完全な」失敗率を見て、最初は分子側の性質
(CASF は古い PDB 由来でペプチド様・補因子が多い) を疑い、ポーズチャネルの定義を
ConfSeq のデコーダ出力から**リガンド自身の PCA 正準フレーム**へ変える設計変更を
検討した。encode だけで済ませてデコード依存を切る、という回避策である。

これは不要だった。**100% の失敗率は分子の性質ではなくファイルの性質の合図**で、
そこで止まって入力側を疑うべきだった。RDKit で読み直して書き戻す
(`MolToMolBlock` → `MolFromMolBlock`) 実験でも直らなかったので「分子が悪い」と
思い込みかけたが、それは*同じ壊れた読み取り結果*を書き戻していただけで、
入力を変えたことにはなっていない。

`feedback_measurement_traps` に 1 件足すなら: **失敗率が 0% か 100% に張り付いたら、
仮説ではなく入力を疑う。** 中間の失敗率だけが分子の性質でありうる。

## 教訓 (この repo では 2 回目)

`project_sdf_counts_bug` は「固定幅の counts 行を `split()` で誤読していた」で、
CASF の敗因 5 件の真因だった。今回は自前パーサではなく RDKit だが、**壊れていたのは
やはり CASF に同梱の SDF** である。

**CASF のリガンドは mol2 から読む。** SDF は使わない。`prolit.chem.mol2` に
`parse_mol2_multi` / `mol2_records` があり、CASF 系のコードは既にそちらを使って
いるものがある。新しく CASF リガンドを読むコードを書くときは合わせること。

## 貼り合わせ baseline への影響

無い。ConfSeq は評価集合・学習集合ともに実質 100% 被覆できる:

- CASF-2016 coreset (リスコアリングの評価集合) 285/285
- PoseBusters Benchmark (再構成の評価集合) 428/428
- トークン率も一致: 2.89 / 2.88 token per heavy atom

したがって**ポーズチャネルは当初設計のまま** (ConfSeq のデコード結果と実座標の
間の剛体変換を量子化) でよく、PCA フレームへの切り替えは不要。
