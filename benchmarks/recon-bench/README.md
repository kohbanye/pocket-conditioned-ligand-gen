# prolit-recon-bench

タンパク質・低分子化合物の **3D 構造再構成（reconstruction）** を、離散構造トークナイザ
4 種で比較するためのベンチマーク。評価には **CASP16** の実験構造（pharma ligands）を使う
——どのモデルの学習データにも含まれない held-out 複合体なので、リークのない再構成テストになる。

| モデル | 種別 | 再構成対象 | 重みの入手 |
|--------|------|-----------|-----------|
| **ESM3** structure tokenizer | protein structure VQ-VAE | タンパク質背骨 (N, CA, C) | HuggingFace `biohub/esm3-sm-open-v1`（公開・非ゲート、structure enc/dec のみ ~1.3GB） |
| **FoldToken4** | protein structure VQ-VAE | タンパク質背骨 | Zenodo [13901445](https://zenodo.org/records/13901445)（`model_zoom.zip`、取得済み） |
| **Token-Mol 1.0** | ligand torsion tokenizer | リガンド（SMILES + 回転結合の torsion） | **重み不要**（再構成はトークナイザ往復のみ） |
| **pocket-ligand VQ-VAE**（自作, Ours） | protein pocket + ligand VQ-VAE | ポケット背骨 **＋** リガンド重原子 | 別ディレクトリの作業コピーから symlink（`weights/`） |

「再構成」とは、構造を離散トークンへ encode → そこから decode して 3D 構造を復元する
往復処理。復元構造と入力構造の **RMSD / TM-score / lDDT** で品質を測る。

- **タンパク質背骨**は ESM3 / FoldToken4 / Ours で比較可能（共通の評価軸）。
- **リガンド**は Ours（座標 VQ-VAE）と Token-Mol（torsion）が再構成可能。Ours はさらに
  ポケット＋リガンドを一括 align した **complex** の RMSD も報告する。

## リポジトリ構成

```
prolit-recon-bench/
├── third_party/                         # git submodules（ソースのみ）
│   ├── pocket-conditioned-ligand-gen/   # 自作モデル
│   ├── esm/                             # ESM3 (evolutionaryscale/esm)
│   ├── FoldToken_open/                  # FoldToken4/5 (A4Bio/FoldToken_open)
│   └── token-mol/                       # Token-Mol 1.0 (jkwang93/token-mol)
├── recon-bench/                             # ベンチ本体パッケージ
│   ├── adapters/                        # 各モデルの再構成アダプタ
│   │   ├── base.py · esm3.py · foldtoken.py · own_vqvae.py · token_mol.py
│   ├── datasets.py                      # casp16 / pdb-folder ローダ
│   ├── metrics.py                       # RMSD / TM-score / lDDT
│   ├── structio.py                      # PDB / SDF 入出力
│   ├── runner.py                        # 再構成ループ → results テーブル
│   └── paths.py                         # パス・環境変数の集約
├── scripts/
│   ├── prepare_casp.py                  # CASP16 展開 + ligand PDB→SDF + index
│   ├── fetch_weights.py                 # 重みの取得 / symlink
│   ├── setup_foldtoken_env.sh           # FoldToken 用 uv venv 構築（conda 不要）
│   ├── own_reconstruct_cli.py           # 自作モデル駆動（自作 venv で実行）
│   ├── foldtoken_reconstruct_cli.py     # FoldToken 駆動（バッチ=1, FoldToken venv）
│   └── run_reconstruction.py            # CLI ランナー
├── notebooks/comparison.py              # marimo: 結果集計・比較プロット
├── weights/  data/  outputs/  results/  # git 管理外（results は集計のみ追跡）
```

## セットアップ

```bash
git submodule update --init --recursive   # クローン直後の場合

uv sync                                   # コア環境（軽量）
uv sync --group esm3                       # ESM3 を in-process で動かす（torch + esm）
sh scripts/setup_foldtoken_env.sh          # FoldToken 用 uv venv（conda 不要, cu118/H100対応）

# 重み・評価データ
uv run python scripts/fetch_weights.py --foldtoken --esm3 --own
uv run python scripts/prepare_casp.py     # CASP16 を展開して index.json を作成
```

CASP16 構造の取得（未取得の場合）:

```bash
cd data/casp16 && for n in 1 2 3 4; do
  wget https://predictioncenter.org/download_area/CASP16/targets/pharma_ligands/L${n}000_exper_struct.tar.gz
done
```

**FoldToken** は依存が古いため**専用 uv venv**（`.venv-foldtoken`、`scripts/setup_foldtoken_env.sh`
で構築）で動かし、アダプタはサブプロセス経由で呼ぶ。再構成に必要なのは torch + PyG
（scatter/cluster）+ pytorch-lightning 等のみで、flash-attn / openfold / deepspeed は不要。
cu118 ビルドで H100(sm_90) 対応。GPU 必須。bench は `.venv-foldtoken` を自動検出する
（`RECON_BENCH_FOLDTOKEN_PYTHON` で上書き可）。

**自作モデル** は作業コピーの uv venv（`RECON_BENCH_OWN_MODEL_PYTHON`、既定で
`../pocket-conditioned-ligand-gen/.venv/bin/python`）でサブプロセス実行する。ソースは
submodule、重み・正規化統計は作業コピーから symlink。

## 実行

```bash
# CASP16: 自作モデルが pocket+ligand、ESM3/FoldToken は全長を再構成→pocket 残基で評価
# （HF_HUB_OFFLINE=1 で prefetch 済みの structure 重みのみ使用。未設定だと初回 5.5GB DL）
HF_HUB_OFFLINE=1 uv run python scripts/run_reconstruction.py \
    --models own_vqvae esm3 foldtoken token_mol \
    --dataset casp16 --limit 50 --out results/casp16.parquet

# ESM3/FoldToken を CASP の全長タンパク質で（各モデル本来のスコープ、自作モデルは対象外）
uv run python scripts/run_reconstruction.py \
    --models esm3 foldtoken --dataset casp16 --protein-scope full

# all-atom トークナイザの ablation arm（学習済みのものだけ自動で選ばれる）
uv run python scripts/run_reconstruction.py \
    --models own_allatom --dataset casp16 --protein-scope full
uv run python scripts/run_reconstruction.py \
    --models own_allatom --allatom-arms joint separate binning --dataset casp16

# 比較ノートブック
uv run marimo edit notebooks/comparison.py
```

ESM3 / FoldToken は**常に全長タンパク質を再構成**する（各モデル本来のタスク）。
`--protein-scope pocket`（既定）は、その再構成を **全長(full)** と **ポケット残基限定(pocket)**
の**両方**で集計する（同じ表に両方の行が出る）。`full` は全長のみ。pocket 行は自作モデル
（pocket native）と同一残基での直接比較になる。

> ⚠️ ポケット PDB を直接 ESM3/FoldToken に入力するのは避ける。ポケットは配列的に不連続な
> 残基の寄せ集めで全長前提のモデルには強い分布外となり、再構成が壊れる（実測 kabsch RMSD
> ~10 Å）。「全長で再構成 → ポケット残基で評価」が正しい比較。

## all-atom トークナイザの ablation arm

`own_vqvae`（残基単位）の後継である **all-atom トークナイザ**は、ポケット原子とリガンド
原子を 1 つの 33-D descriptor で表すので、単一 codebook で両方を覆える。`own_allatom`
アダプタは 1 インスタンス = 1 **arm** で、論文が主張する 2 つの設計軸を張る
（定義は `recon-bench/adapters/own_allatom.py` の `ARMS`）。

| arm | codebook | ligand frame | pose bits | 位置づけ |
|---|---|---|---|---|
| `joint` | 1 冊 8192 | 共有ポケット | 0 | 本命 |
| `separate` | 2 冊 4096+4096 | 共有ポケット | 0 | joint と総 codebook サイズ・LM 語彙が一致（論文の ablation） |
| `binning` | 格子 10³×12 元素 | 共有ポケット | 0 | **学習なし**の下限参照 |
| `localframe_{oracle,3tok,2tok,1.5tok,1tok}` | 2 冊 8192+8192 | **リガンド自身** | ∞/39/26/20/13 | 単一モダリティ型リガンドトークナイザの構成 |

**分割は共有 codebook に対して「codebook ベクトル数」と「bits/atom」を同時に一致させられない**
ため、`separate` を容量一致・レート一致の 2 本立てで挟む。

`localframe_*` はリガンドを自身の canonical frame で符号化する。トークン列が
**SE(3) 不変**になり配置情報を持たないので、受容体へ戻すには剛体変換を別送する必要があり、
その予算を `pose_bits` が課金する（`oracle` は無限精度＝到達不可能な上限）。予算を 1 点に
決め打ちせずスイープするのは、**共有フレーム型（`pose_bits=0`）に並ぶのに何トークン要るか**
という break-even を報告するため。

学習途中の checkpoint が黙って表に載らないよう、arm は epoch >= `--allatom-min-epoch`
（既定 90）の checkpoint がある場合のみ登録される。

### 追加した指標

`recon-bench/metrics.py` に、モダリティ別の RMSD だけでは見えない**界面**の指標を追加した。
protein と ligand を**再構成されたフレームのまま**まとめて評価する `complex` modality で計算する。

- `lddt_pli` — CASP15 準拠。protein 原子 × ligand 原子ペアのみ、R0 = 6 Å。superposition
  不要なので「相対配置が保たれたか」を直接測る。
- `contact_f1` / `contact_precision` / `contact_recall` — 4 Å 接触集合の一致度。
- `clash_lig_atom_frac` / `min_dist_ratio` — vdW 半径和の 0.75 倍を下回る衝突（PoseBusters 準拠）。
- `iface_lig_rmsd` — 参照で受容体に接触しているリガンド原子に限った RMSD。
- `bond_mae` / `bond_max` / `angle_mae` / `angle_max` — リガンド内部幾何の平均と**最悪値**。
  平均は良いのに分子あたり 1 本だけ壊すケースが化学的妥当性を落とすので、最悪値が要る。

### レート列

`bits_per_atom` / `pose_bits` / `total_bits` を全行に付ける。codebook を大きくすれば
再構成誤差はいくらでも下がるので、**レートを併記しない RMSD は比較として成立しない**。
`total_bits` は「評価した原子数」ではなく**実際に発行したトークン数**で課金する
（背骨 30 残基を評価していてもポケット 216 原子分のトークンを払っている、など、
再構成スコープが評価スコープより広いモデルを不当に有利にしないため）。

## ベンチマーク設計メモ

- 入力サンプルは `recon_bench.types.Sample`：タンパク質 PDB、リガンド SDF。
- 各アダプタは `reconstruct(sample) -> ReconResult`（modality ごとに整列済みの ref/rec 座標、
  トークン数）を返し、`recon_bench.metrics` が RMSD などを計算する。
- 結果は long 形式（`sample_id, model, modality, kabsch_rmsd, tm_score, lddt, n_tokens, ...`）
  で出力し、ノートブックで protein-backbone 行と ligand 行を分けて集計する。
- 自作モデルの `reconstruct_one` は任意の receptor PDB + ligand SDF からポケット抽出 →
  encode → decode → NeRF 復元まで行うため、CASP 複合体に直接適用できる（CrossDocked 不要）。

## 状態 / 実装メモ

- ✅ submodule 4 つ、uv 環境（コア + `esm3` group + FoldToken 用 venv）、CASP16 準備（303 複合体）、各モデルの重み。
- ✅ 4 モデルすべて CASP16 でエンドツーエンド実行可能（H100 GPU、n=303）。
  **結果の数値・図表は `notebooks/comparison.py`（marimo）と書き出した HTML/PDF を参照**
  （README には載せない）。

実装上の注意:

- **FoldToken4 のバッチ不具合**: upstream `reconstruct.py` は 32 件バッチで長さの異なる構造を
  混ぜると再構成が壊れる（単体で良好な構造がバッチで大きく劣化）。`scripts/foldtoken_reconstruct_cli.py`
  でバッチ=1（モデルは 1 回ロード）にして回避。
- **codebook**: FoldToken4 は `vq_space=12` で 2^12=4096 が上限（levels 5–12）。2^16 は FoldToken2。
  既定は level 12（ESM3 の 4096 と同等）。
- **集計**: 再構成 RMSD は重い裾を持つので mean ± std と median を併記（`runner.summarize`）。
  Kabsch 整列後の `kabsch_rmsd` が主指標（生 `rmsd` は全長再構成の global frame 差で大きく出る）。
- **modality**: protein_backbone（CA、ESM3/FoldToken は full と pocket の両方で集計）、ligand
  （Ours / Token-Mol）、complex（Ours のポケット CA + ligand 一括 align）。
