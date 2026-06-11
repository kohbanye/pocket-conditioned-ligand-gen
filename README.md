# protein-ligand-3d-reconstruction-bench

タンパク質・低分子化合物の **3D 構造再構成（reconstruction）** を、離散構造トークナイザ
3 種で比較するためのベンチマーク。評価には **CASP16** の実験構造（pharma ligands）を使う
——どのモデルの学習データにも含まれない held-out 複合体なので、リークのない再構成テストになる。

| モデル | 種別 | 再構成対象 | 重みの入手 |
|--------|------|-----------|-----------|
| **ESM3** structure tokenizer | protein structure VQ-VAE | タンパク質背骨 (N, CA, C) | HuggingFace `biohub/esm3-sm-open-v1`（公開・非ゲート、structure enc/dec のみ ~1.3GB） |
| **FoldToken4** | protein structure VQ-VAE | タンパク質背骨 | Zenodo [13901445](https://zenodo.org/records/13901445)（`model_zoom.zip`、取得済み） |
| **Token-Mol 1.0** | ligand torsion tokenizer | リガンド（SMILES + 回転結合の torsion） | **重み不要**（再構成はトークナイザ往復のみ） |
| **pocket-ligand VQ-VAE**（自作, Ours） | protein pocket + ligand VQ-VAE | ポケット背骨 **＋** リガンド重原子 | 別ディレクトリの作業コピーから symlink（`weights/`） |

「再構成」とは、構造を離散トークンへ encode → そこから decode して 3D 構造を復元する
往復処理。復元構造と入力構造の **RMSD / TM-score / lDDT** で品質を測る。

- **タンパク質背骨**は 3 モデルすべてで比較可能（共通の評価軸）。
- **リガンド**を再構成できるのは自作モデルのみ（リガンド列は単独で報告）。

## リポジトリ構成

```
protein-ligand-3d-reconstruction-bench/
├── third_party/                         # git submodules（ソースのみ）
│   ├── pocket-conditioned-ligand-gen/   # 自作モデル
│   ├── esm/                             # ESM3 (evolutionaryscale/esm)
│   ├── FoldToken_open/                  # FoldToken4/5 (A4Bio/FoldToken_open)
│   └── token-mol/                       # Token-Mol 1.0 (jkwang93/token-mol)
├── plbench/                             # ベンチ本体パッケージ
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
│   ├── own_reconstruct_cli.py           # 自作モデル駆動（自作 venv で実行）
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
（`PLBENCH_FOLDTOKEN_PYTHON` で上書き可）。

**自作モデル** は作業コピーの uv venv（`PLBENCH_OWN_MODEL_PYTHON`、既定で
`../pocket-conditioned-ligand-gen/.venv/bin/python`）でサブプロセス実行する。ソースは
submodule、重み・正規化統計は作業コピーから symlink。

## 実行

```bash
# CASP16: 自作モデルが pocket+ligand、ESM3/FoldToken は全長を再構成→pocket 残基で評価
# （HF_HUB_OFFLINE=1 で prefetch 済みの structure 重みのみ使用。未設定だと初回 5.5GB DL）
HF_HUB_OFFLINE=1 uv run python scripts/run_reconstruction.py \
    --models own_vqvae esm3 foldtoken \
    --dataset casp16 --limit 50 --out results/casp16.parquet

# ESM3/FoldToken を CASP の全長タンパク質で（各モデル本来のスコープ、自作モデルは対象外）
uv run python scripts/run_reconstruction.py \
    --models esm3 foldtoken --dataset casp16 --protein-scope full

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

## ベンチマーク設計メモ

- 入力サンプルは `plbench.types.Sample`：タンパク質 PDB、リガンド SDF。
- 各アダプタは `reconstruct(sample) -> ReconResult`（modality ごとに整列済みの ref/rec 座標、
  トークン数）を返し、`plbench.metrics` が RMSD などを計算する。
- 結果は long 形式（`sample_id, model, modality, kabsch_rmsd, tm_score, lddt, n_tokens, ...`）
  で出力し、ノートブックで protein-backbone 行と ligand 行を分けて集計する。
- 自作モデルの `reconstruct_one` は任意の receptor PDB + ligand SDF からポケット抽出 →
  encode → decode → NeRF 復元まで行うため、CASP 複合体に直接適用できる（CrossDocked 不要）。

## 状態（このコミット時点）

- ✅ submodule 3 つ、uv コア環境、FoldToken 用 uv venv、CASP16 準備（303 複合体）、各モデルの重み。
- ✅ **4 モデルすべて CASP16 でエンドツーエンド検証済み（H100 GPU、n=303）**。
  集計は **mean ± std**（再構成 RMSD は重い裾を持つので median も併記推奨）:

  | Model | modality | eval_scope | kabsch_rmsd (Å) | tm_score | lddt | n_tokens |
  |-------|----------|-----------|-----------------|----------|------|----------|
  | ESM3 | protein_backbone | full | 3.51 ± 6.11 | 0.88 ± 0.25 | 0.93 ± 0.06 | 731 |
  | ESM3 | protein_backbone | pocket | 1.16 ± 2.15 | 0.81 ± 0.25 | 0.95 ± 0.07 | 731 |
  | FoldToken4 | protein_backbone | full | 2.12 ± 0.58 | 0.95 ± 0.02 | 0.79 ± 0.07 | 731 |
  | FoldToken4 | protein_backbone | pocket | 1.37 ± 0.45 | 0.53 ± 0.16 | 0.77 ± 0.09 | 731 |
  | Ours | protein_backbone | native (pocket) | 0.85 ± 0.15 | 0.69 ± 0.13 | 0.87 ± 0.03 | — |
  | Ours | **complex** (pocket CA + ligand, joint align) | native | **0.69 ± 0.10** | — | — | — |
  | **Ours** | **ligand** | native | **0.35 ± 0.10** | — | — | 27 |
  | **Token-Mol** | **ligand** | native | **1.60 ± 0.82** | — | — | 6 |

  ESM3 の median は 0.88Å(full)/0.36Å(pocket) で論文の <1Å と一致（mean は外れ値で大）。
  **リガンド**: Ours（座標 VQ-VAE, 0.35Å, ~27 token/分子）は Token-Mol（torsion, 1.60Å, ~6 token/分子）
  より精密だが、Token-Mol は遥かにコンパクト——精度と圧縮率のトレードオフ。
  **complex** はポケット背骨(CA)とリガンドを連結して 1 回の Kabsch で重ねた RMSD で、別々に
  align した場合と違いリガンドのポケットに対する相対配置のズレも捉える（Ours 0.69Å）。
  - **重要な修正**: FoldToken4 の upstream `reconstruct.py` は 32 件バッチ再構成で長さの異なる
    構造を混ぜると再構成が壊れる（単体 1.8Å の構造がバッチで 15Å に）。`scripts/foldtoken_reconstruct_cli.py`
    でバッチ=1（モデルは 1 回ロード）にして解消。N128 平均 6.4Å→1.8Å。
  - **codebook**: FoldToken4 は `vq_space=12` で 2^12=4096 が上限（levels 5–12）。2^16 は FoldToken2 の話。
    既定は level 12（ESM3 の 4096 と同等）。
  - Kabsch 整列後の `kabsch_rmsd` が有効指標（生 `rmsd` は全長再構成の global frame 差で大きく出る）。
