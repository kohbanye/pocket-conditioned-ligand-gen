# CLAUDE.md — Pocket-Conditioned Ligand Generation

## Project Overview

標的タンパク質のポケット構造・配列情報を条件として、結合リガンドの3次元構造を自己回帰型Transformerで生成する研究プロジェクト。
創薬プロセスにおけるリード化合物設計の効率化を目指す。

### Research Items

1. **項目(1)**: タンパク質構造・配列、化合物の立体構造を離散トークンにエンコードするトークナイザの構築
2. **項目(2)**: トークン化したデータによる自己回帰型Transformerの学習
3. **項目(3)**: 複数の標的タンパク質に対して生成した分子セットの評価

### Architecture

- **タンパク質エンコーダ**: ESM3 — タンパク質の構造・配列をトークン化
- **リガンドエンコーダ**: Mol-StrucTok — 化合物の3次元構造を離散トークンにエンコード
- **生成モデル**: 自己回帰型Transformer (GPT/Llama ベース)
- **デコード戦略**: 2段階 — Step1: 粗い原子配置の復元、Step2: 拡散モデルによる構造微調整

### Token Sequence Format

```
<p>...protein pocket structure tokens...</p><s>...protein sequence tokens...</s><l>...ligand structure tokens...</l>
```

- `<p></p>`: タンパク質ポケット構造トークン
- `<s></s>`: タンパク質配列トークン
- `<l></l>`: リガンド構造トークン
- Retrieval augmentation: 類似複合体の構造・配列情報を入力先頭に追加

## Tech Stack

- **Language**: Python 3.12
- **Package Manager**: uv
- **Deep Learning**: PyTorch + PyTorch Lightning
- **Config**: Hydra (hydra-core)
- **Experiment Tracking**: Weights & Biases (wandb)
- **Linter**: Ruff (select = ALL, ignore = D, N812)
- **Type Checker**: ty
- **Testing**: pytest

## Project Structure

```
pocket-conditioned-ligand-gen/
├── src/
│   ├── tokenizers/           # Tokenization modules
│   │   ├── protein.py        # ESM3 wrapper for pocket structure/sequence
│   │   ├── ligand.py         # Mol-StrucTok wrapper for 3D ligand structure
│   │   └── sequence.py       # Assemble <p>...<s>...<l>... format
│   ├── data/                 # Data loading and preprocessing
│   │   ├── crossdocked.py    # CrossDocked2020 DataModule
│   │   └── preprocessing.py  # Data preprocessing utilities
│   ├── model/                # Model architecture
│   │   ├── transformer.py    # GPT/Llama-based autoregressive model
│   │   └── decoder.py        # Two-stage decoding (coarse + diffusion)
│   ├── evaluation/           # Evaluation utilities
│   │   └── metrics.py        # Validity, novelty, diversity, docking score
│   └── config.py             # Dataclass-based configuration
├── scripts/                  # CLI entry points
│   ├── tokenize.py           # Tokenization pipeline
│   └── train.py              # Training pipeline
├── notebooks/                # Jupyter notebooks for exploration/visualization
├── data/                     # Downloaded datasets (gitignored)
├── pyproject.toml            # Project config and dependencies
└── uv.lock                   # Dependency lock file
```

## Key Datasets

- **CrossDocked2020**: タンパク質–リガンドのクロスドッキング複合体データセット（主要な学習データ）
- **PDBbind**: タンパク質–リガンド複合体データベース
- 評価標的: SARS-CoV-2 メインプロテアーゼ、EGFR 等

## Evaluation Metrics

- 生成分子の有効性 (Validity)、新規性 (Novelty)、多様性 (Diversity)
- ドッキングスコア
- FEP（自由エネルギー摂動法）による結合親和性評価
- MD シミュレーションとの RMSD 比較

## Development Conventions

- Config は dataclass で定義する（`src/config.py`）
- モデルは `LightningModule` を継承して実装する
- データは `LightningDataModule` を継承して実装する
- Ruff の ALL ルール準拠（docstring 系 D, N812 は除外）
- コードとコミットメッセージは英語で書く

## Key References

- ESM3: Meta の大規模タンパク質言語モデル（構造・配列・機能のマルチモーダルトークナイザ）
- Mol-StrucTok: 分子の3次元構造を離散トークンにエンコードするトークナイザ
- HierDiff (Qiang+ ICML2023): 2段階拡散モデルによる構造微調整
- VQ-VAE (Van Den Oord+ NeurIPS2017): 離散潜在変数による再構成精度向上の検討

## Commands

```bash
# Install dependencies
uv sync

# Run training
uv run python train.py

# Run linter
uv run ruff check .

# Run formatter
uv run ruff format .

# Run tests
uv run pytest
```
