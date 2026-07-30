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

## Structure とレイヤ規約

```
                  pipelines/  ─┐
                  benchmarks/ ─┼──→  src/prolit/   （下向きのみ）
                  scripts/    ─┘
```

```
src/prolit/        ライブラリ。argparse も I/O ポリシーも持たない。入口は prolit/api.py
  ├── chem/        RDKit / PDB / MOL2 / Open Babel / docking / 幾何
  ├── tokenizers/  descriptor schema, VQ-VAE, vocab, loaders, PoseEncoder
  ├── model/       VQ-VAE / CLM / MLM / scoring head / pose refiner
  └── data/        descriptor cache, token stream, datasets
pipelines/         コーパス構築 (corpora/) と学習 (train/) の CLI
benchmarks/        論文の表ごとに 1 つ。common/ が共有レジストリと有意差検定
scripts/           **ベンチが subprocess で叩く ProLIT 側の入口だけ**を置く
jobs/              クラスタ投入ツール（lib.sh + submit.py。ジョブ本体は git 管理外）
third_party/       ベースラインの submodule。patches/ に当てている修正
```

**規約（`tests/test_layering.py` が強制する）**:

- `prolit` は上位レイヤを一切知らない。import も subprocess も禁止。
- `pipelines` / `benchmarks` / `scripts` は**兄弟**で、互いを import しない。
  2 つが同じものを必要としたら、それは `prolit` に置くもの。
  実際 `PoseEncoder` / `parse_mol2_multi` / `obabel_mol` / docking ヘルパは
  この規則で降りてきた（eval スクリプトに住んでいて、corpus builder が import し、
  ベンチが複製していた）。
- 同レイヤ内の兄弟 import は**裸のモジュール名**で書く
  (`from tokenize_biolip import ...`)。`pipelines.` / `scripts.` 接頭辞は
  リポジトリルートが `sys.path` にある時しか解決せず、cwd 依存になる。
- `scripts/` は 12 本を上限にテストで縛ってある。増やす前に、
  再利用可能なら `prolit`、コーパス・学習なら `pipelines/` を検討する。

## Commands

```bash
uv sync --all-packages         # ライブラリ + workspace 内ベンチ
uv run pytest                  # ライブラリ / pipelines / workspace 内ベンチ
uv run ruff check .
uv run ty check src
```

`benchmarks/recon-bench` だけは別環境（ESM3 が fork 版 transformers を要求するため）。
`cd benchmarks/recon-bench && uv sync` で個別に。

## Conventions

- **学習・トークン化は `.venv/bin/python` を直叩き**。`uv run` は毎回 editable install を
  解決し直すので遅く、ジョブ内では無意味。
- **git に載せるのはコードだけ**。重み・キャッシュ・トークン列・結果 dump・
  ジョブスクリプト・`docs/` は全て `.gitignore`（ローカルには残す）。
  数値の出所は `docs/results/`、調査ログは `docs/notes/` に、重みと同じ機械上で。
  ジョブの出所は下の「ジョブと出所」を参照。
- **サイト固有の絶対パスを書かない**。ルートは `__file__` から導出、外部バイナリ
  (vina / obabel / prepare_receptor) は `prolit.external_tools` が `PATH` と
  環境変数から実行時に解決する。
- **乱数は `prolit.seeding` 経由で一本化**。CLI は `add_seed_argument(parser)` +
  `seed_from_args(args)`、独立ストリームが要る所は `rng_for(seed, "名前")`。
  裸の `np.random.default_rng()`（引数なし）は禁止で、`tests/test_seeding.py`
  が検出する。学習の seed は config に載るので checkpoint が覚えている。
- コードとコミットメッセージは英語。
- トークン列のバイト表現を変える変更は**全 checkpoint を無効化する**。
  `tests/test_token_stream.py` が既存実装との一致を固定しているので、
  意図的に変える場合はそのテストごと更新する。

## ジョブと出所（provenance）

ジョブスクリプトは**手で書かない**。`jobs/submit.py` が生成し、`jobs/` は
git 管理外。`source ~/.bashrc` と `module load cuda` は**書かない**
（0.3 秒で exit 0・無出力の無言死を招く。torch は CUDA 同梱なのでロード不要）。
経緯は `jobs/lib.sh` にある。

```sh
python jobs/submit.py --name lm_pre --resource node_f --hours 8 \
    -- pipelines/train/clm.py --token-dir data/lm_tokens_allatom --seed 7
```

**スクリプトを git に残さない代わりに、run が自分の出所を書く。**
学習は `RecordProvenance` コールバックで、checkpoint と同じディレクトリに
`run.json` を落とす:

```json
{"command": ["pipelines/train/clm.py", "--seed", "7", ...],
 "git": {"sha": "...", "dirty": false, "branch": "main"},
 "seed": 7, "started": "...", "hostname": "r9n2",
 "job": {"name": "lm_pre", "resource": "node_f", "hours": "8", "id": "8299013"}}
```

- **git にはコード、run ディレクトリにはそれを作ったコマンド**。両方揃えば再現できる。
- run を消せば出所も消える ← 正しい（同じものなので）。
- `job` は `submit.py` が環境変数で渡す。`--run-name` 無しだと checkpoint 先が
  wandb の run id になり投入時には未確定なので、**投入側でなく run 側が記録する**。
  対話実行では `job` が無いだけで、コマンドは残る。
- **`dirty: true` は SHA より重要**。true なら git だけからその数値を再現できない。

新しく学習スクリプトを足したら `RecordProvenance(seed=args.seed)` を
callbacks に入れる。`tests/test_provenance.py` が入れ忘れを検出する。

**git に載せてよい実験設定**は「論文で報告するもの」だけ。
`benchmarks/common/src/prolit_bench/variants.py` のようなレジストリに
**データとして**足す。探索の枝番（`_v8` `_v10` …）は run ディレクトリの
`run.json` にだけ残せばよく、最良と分かったものを後からレジストリへ昇格させる。
