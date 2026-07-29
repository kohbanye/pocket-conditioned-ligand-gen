# CLAUDE.md — ProLIT

## Project Overview

タンパク質ポケットとリガンドを**ひとつの共有語彙**で離散トークン化する
**ProLIT** (Protein–Ligand Interface Tokenizer) の研究リポジトリ。
論文は `../aaai27-paper/main.tex`（AAAI 2027 投稿,
*Learning the Language of the Binding Interface*）。**論文が唯一の正**で、
そこに載る構成だけが本流。

### 構成

| 層 | 実体 |
|---|---|
| **トークナイザ (本流)** | joint all-atom VQ-VAE。33-D per-atom descriptor、単一 codebook 8192 (vocab 8199) |
| **トークナイザ (ablation)** | separate = protein 専用 VQ + ligand 専用 VQ を 1 code space に連結 |
| **ProLIT-MLM** | encoder-only 複合体 MLM (~99M)。pose rescoring / affinity head の backbone |
| **ProLIT-CLM** | Qwen3 系 causal LM (~298M) + e3nn flow-matching pose refiner。生成 |

### トークン列

```
<bos><p> pocket atom tokens </p><l> ligand atom tokens </l><eos>
```

protein 原子と ligand 原子は**同じ codebook**から引く。片方が空でもよい
（空の `<p></p>` = リガンド単独コーパス）ので、単独事前学習 → 複合体 finetune が
同じフォーマットで繋がる。

## 重要な前提

- **checkpoint と `normalization_stats.pt` は必ずセット**。取り違えるとエラーにならず、
  もっともらしいがスケールの狂った座標が出る。
- **どのアームがどの重みを指すかは
  `benchmarks/common/src/prolit_bench/variants.py` が唯一の定義**。ここを迂回しない。
  モジュール冒頭に、benchmark 間で checkpoint 選択が食い違っていた既知の問題が
  書いてあるので、触る前に読むこと。
- **`prolit/api.py` が公開 API**。`__all__` の外は内部実装。ベンチから private を
  掴まない。
- **旧 `src.*` import パス**は checkpoint の pickle 互換のために
  `prolit/_legacy_import_path.py` で生かしてある。消すと既存 checkpoint が全部読めなくなる。

## Tech Stack

- Python 3.12 / uv (workspace) / PyTorch + Lightning / wandb
- Config は **dataclass** (`prolit/config.py`)。**Hydra は使っていない**
- Notebooks は **marimo** (`.py` 形式)
- Lint: Ruff (`select = ["ALL"]`, ignore `D`, `N812`, `COM812`) / 型: ty / テスト: pytest

## Structure

```
src/prolit/        ライブラリ（副作用なし・argparse なし）。入口は prolit/api.py
  ├── chem/        RDKit / PDB / 幾何（モダリティ非依存）
  ├── tokenizers/  descriptor schema, VQ-VAE, vocab, checkpoint loaders
  ├── model/       VQ-VAE / CLM / MLM / scoring head / pose refiner
  └── data/        descriptor cache, token stream, datasets
pipelines/         コーパス構築 (corpora/) と学習 (train/) の CLI
benchmarks/        論文の表ごとに 1 つ。common/ が共有レジストリと有意差検定
jobs/              TSUBAME 投入ツール + 実行済みジョブの archive
scripts/           評価・生成の入口
third_party/       ベースラインの submodule。patches/ に当てている修正
docs/results/      凍結した結果記録（数値の出所はここが正）
docs/notes/        日付つき調査ログ（当時の記録。現在の仕様ではない）
```

## Commands

```bash
uv sync --all-packages         # ライブラリ + workspace 内ベンチ
uv run pytest                  # ライブラリ / pipelines / workspace 内ベンチ
uv run ruff check .
uv run ty check src
```

`benchmarks/plbench` だけは別環境（ESM3 が fork 版 transformers を要求するため）。
`cd benchmarks/plbench && uv sync` で個別に。

## Conventions

- **学習・トークン化は `.venv/bin/python` を直叩き**。`uv run` は毎回 editable install を
  解決し直すので遅く、ジョブ内では無意味。
- **ジョブスクリプトは `jobs/submit.py` 経由で作る**。`source ~/.bashrc` と
  `module load cuda` は**書かない**（0.3 秒で exit 0・無出力の無言死を招く。
  torch は CUDA 同梱なのでロード不要）。`jobs/lib.sh` にこの経緯がある。
- コードとコミットメッセージは英語。
- トークン列のバイト表現を変える変更は**全 checkpoint を無効化する**。
  `tests/test_token_stream.py` が既存実装との一致を固定しているので、
  意図的に変える場合はそのテストごと更新する。
