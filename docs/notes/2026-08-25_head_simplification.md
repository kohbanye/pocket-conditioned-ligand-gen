# 採点ヘッドを単一構成に畳む (2026-08-25)

`e250_div` を正として、実験の過程で足した読み出し・補助損失をコードから外した。
判断は「使っていない機構も削る」。checkpoint 互換より、コードが実際の構成を
表すことを優先した。

## 何を削ったか

| 機構 | 実装場所 | 削除理由 |
|---|---|---|
| `pooling` の 5 択 (`meanmax`/`attn`/`xattn`/`pairsum`) | `rescore_module._pool` | 全て測って mean に負けた |
| `decomposed_head` + `decomp_aux_weight` | 同 `_predict`、`_step` | 機構自体が反証済 |
| `atom_head` + `atom_aux_weight` | 同 `forward_with_atoms`、`_step` | 改善せず |
| `head_interaction_layers` + `pair_heads` | 同 `_encode_states` | 使う腕が無い |
| `.disp` / `.comp` の読み込み | `rescore_dataset` | 上の 2 つ専用だった |

残ったのはこれだけ:

```
リガンドトークンの mean  ->  Linear(768,768) -> GELU -> Dropout(0.1)
                              -> LayerNorm -> Linear(768,1)  ->  予測 RMSD
```

ヘッド 0.59M / encoder 105.3M。出力はスカラー 1 個。

## 各機構をなぜ落としてよいと判断したか

**プーリング。** 95 標的の標的単位交差検証で、mean 92.6/74.7/0.8793 に対し
max 89.5/56.8/0.8382、mean+max 89.5/61.1/0.8472 (DP@2A/DP@1A/rho)。
希釈を避けるつもりの max は、1 チャンネルにつき 1 原子しか見ないので
他の全原子を捨てる方が高くついた。`xattn` `pairsum` も登録腕の実測で勝てず。

**decomposed_head。** 学習済みヘッドの出力を成分に回帰すると
`0.904*rigid^2 + 0.118*internal^2` で内部変形が 3.5 倍軽い、という測定から
作った。しかし 30 標的の統制摂動 (純平行移動 / 純回転 / 純ねじれ、8 フレーム)
での傾き d(pred)/d(RMSD) は 0.790 / 0.756 / 0.893 でほぼ等しく、一様な減衰は
単調なので順序を変えない。トークンも内部変形を符号化している
(純ねじれでコードの 5.2% 変化、同 RMSD の剛体摂動で 5.6%)。機構が反証された。

**atom_head。** 1 ポーズ 1 ラベルより ~30 の per-atom ラベルの方が密、という
狙いだったが CASF で改善せず。

**読み出しそのものが律速でない、という確認。** プールされた 768 次元を取り出し、
標的単位の交差検証で線形/MLP プローブを学習させて学習済みヘッドと比較した:

| 読み出し | DP@2A | DP@1A | rho |
|---|---|---|---|
| 学習済み head | 92.6 | 78.9 | 0.8806 |
| Ridge (CV) | 92.6 | 74.7 | 0.8793 |
| MLP 256 (CV) | 91.6 | 77.9 | 0.8394 |

どちらも超えない。表現から汎化する形で取れる情報は 0.59M のヘッドが
取り切っている。(in-sample の MLP は 100% 出るが 7255 例に 20 万パラメータで暗記。)

## 捨てたもの

- 登録腕 3 本: `e250_pairsum` / `e250_aux03` / `e250_decomp`
  (state_dict に消した submodule のキーがあり `load_state_dict` が通らない)
- ディスク上の rescore checkpoint 約 150 run (mean 以外のプーリング 63、
  atom_aux 38、interaction 層 21、decomp 4)。**消してはいない。読めなくなっただけ。**

CLAUDE.md の「config dataclass を変えると checkpoint が読めなくなる」の通り。
フィールドの *削除* 自体は unpickle を壊さない (インスタンスの `__dict__` は
復元され、誰も読まないだけ) が、submodule が消えた分 state_dict が合わなくなる。

## 事故と復旧 (再発防止のため残す)

**1. `variants.py` を壊した。** 3 腕を消すのに
`s.rfind("\n\n\n", 0, i)` で「直前のコメントブロックごと」削ろうとしたら、
区切りが 3 連改行でない箇所を跨いで `_JOINT_VQVAE` 以下の定数ブロックと
`JOINT` / `JOINT_NOCASF` / `SEPARATE` / `E250_MEAN` / `_E250_VQ` / `_E250_MLM` まで
消えた。**このファイルの大半は未コミットだったので `git checkout` で戻せない。**

復旧: `__pycache__/variants.cpython-312.pyc` (15:37、編集前) を marshal で
読み、`sys.modules` に自分を登録してから exec して値を回収。HEAD 側と一致する
先頭部分は `git show HEAD:` から戻し、E250 定数は .pyc の値で書き直した。
復旧後、.pyc の REGISTRY と照合して**残る 18 腕が完全一致**することを確認した
(ckpt パス / label / description / codebook_size)。

**2. `tests/test_provenance.py` と `test_submit.py` を `git checkout HEAD --` で
戻して未コミット変更を消した。** 一括 sed で壊したのを戻すつもりだったが、
両ファイルには HEAD にない変更 (計 65 行) があった。`test_submit.py` は
再構成できた。`test_provenance.py` は `tune_vqvae.py` (git 未追跡) が
`RecordProvenance` でなく `write_manifest` を使う件の例外扱いだったと推定し、
どちらでも通る形に書き直した (理由を docstring に明記)。

**教訓: 未コミットの変更があるファイルに対して、範囲を計算で決める削除も
`git checkout` も使わない。** 消す対象は文字列で完全一致させ、
先に .pyc かコピーを取る。

## 副次的に直したもの

- `--sweep` の例が消した `--pooling` を指していた 4 箇所
  (`CLAUDE.md`、`tests/test_submit.py`、`provenance.py` の docstring、
  `eval_casf_scoring_power.py` の usage) を `--listwise-tau` に差し替え
- リポジトリ直下に取り残されていた `seedvar.py` (前セッションの調査スクリプト、
  git 未追跡、`ruff check .` の 49 件の全て) を scratchpad へ退避
- ベンチの再現テスト 2 本を単一ヘッドの現行値に書き直し
  (rescoring は `e250_div` 95.4/82.1/0.89 と `joint` 89.5/73.7/0.85 を固定、
  affinity は `joint` R=0.7711 / rho=0.6474。GenScore に負けている事実も
  テストとして固定した)
