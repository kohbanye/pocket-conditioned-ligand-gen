# 生成ベンチの標的セットが正典と別物だった (2026-08-21)

「97 標的」と呼んでいたものは、**CrossDocked の標準テスト分割ではなかった**。

## 正典は 100 エントリ / 93 タンパク質

TargetDiff / DiffSBDD / FLOWR が報告するのは 100 ポケット。ディレクトリは 93 で、
7 つのタンパク質が受容体構造を 2 つずつ持つ:

```
CDK6_HUMAN   2f2c_B_rec / 4aua_A_rec      NOS3_HUMAN  1rs9_A_rec / 4kcq_A_rec
CHIB_SERMA   1h0i_A_rec / 4z2g_A_rec      NQO1_HUMAN  1dxo_C_rec / 1gg5_A_rec
LMBL1_HUMAN  2pqw_A_rec / 2rhy_A_rec      PYRD_TRYCC  2e6d_A_rec / 3w83_B_rec
NOS1_HUMAN   3tym_A_rec / 4d7o_A_rec
```

実体は `/gs/bs/tga-ohuelab/sakano/data/targetdiff/test_set` に最初から揃っていた
(93 ディレクトリ / 100 受容体)。`prepare_targets.py --crossdocked-test` はこの
レイアウト専用に書かれていて、`__<stem>` で 7 組に一意なタグを与える処理まで
入っている。**仕組みは在ったのに使われていなかった。**

## 何が起きていたか

使われていたのは `data/target_pairs.json` (97 件) で、これは HuggingFace の
CrossDocked2020 キャッシュ (`data/hub_cache`) から受容体を引いて作られていた。
結果:

1. **3 標的が欠けた** — `BGAT_HUMAN_63_353_0` / `GRK4_HUMAN_1_578_0` /
   `PHKG1_RABIT_6_296_ATPsite_0`。いずれも `data/refs/<tag>.sdf` が作られず脱落。
   7 つの複数ポケット標的は 97 にも両方入っており、これは原因ではない。
2. **97 のうち 87 で受容体構造が違った** — 同じポケット定義に対する別の PDB
   エントリ。例: `ABL2_HUMAN_274_551_0` は ours `2xyn_A_rec` / 正典 `4xli_B_rec`。
   manifest 上このポケットには 4 構造あり、HF から引くとどれが来るかは
   正典の選択と無関係だった。

## 内部のベースライン比較も成立していなかった

`results_diffsbdd` (100 標的) と我々の 97 で、参照リガンドの Vina が一致するのは
比較可能な 71 標的のうち **2 つだけ**、最大差 124.9 kcal/mol。そしてその 2 つは
**受容体がたまたま一致した 10 標的の中にあった**。ベースラインは正典で、我々は
別セットで走っていたということ。

## リークは、どちらでも無い

`holdout.sbdd_bench_receptor_pdbs` はポケット名から manifest 経由で
**そのポケットの全受容体 PDB id** を引いて除外する。`ABL2_HUMAN_274_551_0` なら
`2xyn` も `4xli` も両方落ちる。正典に切り替えても学習リークは入らない。
ここが塞がっていたので、HF 側に留まる技術的な理由は無かった。

## 正典を選んだ理由

- **FLOWR は自前で走らせられない** (アダプタは diffgui/diffsbdd/own/targetdiff の
  4 つだけ)。公表値を引くしかなく、それは正典 100 の値。HF セットに揃えると
  FLOWR との比較そのものが表から消える。
- **コストが逆**。`results_diffsbdd` と `results_td_a/b` は既に正典で走っている。
  正典に寄せれば再実行は我々の腕だけ。HF に寄せると全ベースラインを生成から
  やり直したうえ、FLOWR は依然として比較できない。

## いま何が有効で、何が無効か

- **有効**: 今日測った改善幅 (剛体緩和で clash-free 0.204 -> 0.876、
  place-first で生の clash-free 0.260 -> 0.619、剪定と接合、自動拘束半径)。
  すべて同一入力どうしの対比なので、相対的な結論は動かない。
- **無効**: 絶対値と、FLOWR / DiffSBDD / TargetDiff との比較。正典 100 で
  取り直す。
- 旧セットは `data/targets_hub97` に退避。受容体構造を替えたときのロバスト性を
  見る追試には使える (正典の表が固まってから)。

## 教訓

標的セットは**それを使う全員が同じ 1 つの定義から作る**こと。
`prepare_targets.py` という正しい入口があるのに、その脇から
`target_pairs.json` を作る経路が生えていたのが原因。ベンチの入力は
`variants.py` と同じ扱い — レジストリを 1 つにして迂回路を作らない。
