# sbdd-bench

**Structure-Based Drug Design（SBDD）の3Dリガンド生成モデル**を、**共通・モデル非依存の評価スイート**で比較するベンチマーク。

姉妹プロジェクト [`protein-ligand-3d-reconstruction-bench`](../protein-ligand-3d-reconstruction-bench) と同じ設計思想——**モデルのソースは git submodule**、**重みは作業コピーから symlink**、**各モデルは自分の環境でサブプロセス実行**——を、reconstruction ではなく **conditional generation** に対して適用する。

| モデル | 種別 | 条件づけ | 重みの入手 |
|--------|------|----------|-----------|
| **DiffSBDD** (Schneuing+ 2024) | E(3)-equivariant diffusion | full receptor + 参照リガンド | CrossDocked `crossdocked_fullatom_cond.ckpt`（symlink） |
| **TargetDiff** (Guan+ ICLR2023) | atom diffusion | ≤10 Å ポケット PDB | `pretrained_diffusion.pt`（Google Drive） |
| **DiffGui** (Hu+ 2024) | guided diffusion + bond | ≤10 Å ポケット PDB | `trained.pt` + `bond_trained.pt`（Google Drive） |
| **pocket-conditioned-ligand-gen**（自作, Ours） | 自己回帰 LM + VQ-VAE | full receptor + 参照リガンド | 別ディレクトリの作業コピーから symlink |

**設計の肝**：生成（モデルごとの重い環境）と評価（共通の軽い環境）を分離する。各モデルは自分のインタプリタで `generated.sdf`（1分子1エントリの3D構造）だけを吐き、bench 環境はそれを読んで RDKit + AutoDock Vina + PoseBusters で**全モデルを同一基準で**採点する。bench 環境は生成モデルを一切 import しない。

## リポジトリ構成

```
sbdd-bench/
├── third_party/                        # git submodules（ソースのみ）
│   ├── pocket-conditioned-ligand-gen/  # 自作モデル
│   ├── DiffSBDD/  targetdiff/  DiffGui/
├── sbddbench/                          # bench 本体（= 評価器）
│   ├── adapters/                       # 各モデルの生成アダプタ（subprocess）
│   │   ├── base.py · own.py · diffsbdd.py · targetdiff.py · diffgui.py
│   ├── molio.py        # generated.sdf → 統一表現（elements/coords/sanitized mol）
│   ├── chem.py         # ① 化学妥当性: validity, QED, SA, Lipinski/Veber/PAINS
│   ├── docking.py      # ② 親和性: Vina Score / Min / Dock
│   ├── pose.py         # ③ ポーズ品質: PoseBusters, 衝突数, strain
│   ├── interactions.py # ④ 相互作用: key-residue recovery, IFP（ProLIF, optional）
│   ├── diversity.py    # ⑤ 多様性: uniqueness, novelty, scaffold diversity
│   ├── metrics.py      # 統合 → per-molecule / per-target + 複合 hit-rate
│   ├── datasets.py     # ターゲット集合ローダ（index.json）
│   ├── types.py · paths.py
├── scripts/
│   ├── fetch_weights.py     # 重みの symlink / 取得
│   ├── prepare_targets.py   # receptor/pocket/ref-ligand/pdbqt/box を整備
│   ├── setup_envs.sh        # モデルごとの conda env を作成
│   ├── run_generation.py    # アダプタ駆動 → outputs/<model>/<target>/generated.sdf
│   └── run_evaluation.py    # 全 SDF を採点 → results/{per_molecule,per_target,per_model}
├── weights/  data/  outputs/  results/  envs/   # すべて git 管理外
```

## 評価指標の選択

> SBDD の評価は1スコアで決めない。特に Vina だけで SOTA 判定しない——3D 生成モデルは docking では強く見えても、**chemical validity / pose quality** で破綻しがち。

下表のとおり、依頼の6カテゴリから実用上重要なものを選んで実装した（`sbddbench/metrics.py` が統合）。

| カテゴリ | 実装した指標 | モジュール |
|----------|-------------|-----------|
| ① 化学妥当性 | **validity**（RDKit sanitize）, connectivity, QED, **SA**, Lipinski, Veber, PAINS | `chem.py` |
| ② 親和性 proxy | **Vina Score**（生成ポーズ）/ **Vina Min**（局所最小化）/ **Vina Dock**（再ドッキング）を分けて報告 | `docking.py` |
| ③ ポーズ品質 | **PoseBusters PB-validity**, タンパク質–リガンド **steric clash 数**, **strain energy**（PoseCheck 式） | `pose.py` |
| ④ 相互作用 | **key-residue recovery**, **interaction-fingerprint Tanimoto**（ProLIF, opt-in） | `interactions.py` |
| ⑤ 多様性 | uniqueness, **novelty**（学習集合比）, internal diversity, **scaffold diversity** | `diversity.py` |
| ⑥ 参照近さ／実用性 | 参照リガンドへの Tanimoto, 複合 **hit-rate**（下記） | `metrics.py` |

### 主指標：複合 hit-rate（valid & plausible & bindable & synthesizable）

単純な平均 Vina ではなく、**実用に近い成功率**を主指標に置く。生成分子が以下を**すべて**満たす割合：

1. RDKit valid
2. PoseBusters PB-valid（物理的に妥当な pose）
3. Vina Dock が**参照リガンドより良い**
4. 合成可能（SA ≤ 5）
5. drug-like（QED ≥ 0.4）

報告は **scaffold-unique** な hit の割合（`hit_scaffold_unique_rate`）——同じ骨格を量産する mode collapse で水増しできないようにする。閾値は `sbddbench/metrics.py` の `HIT` で変更可。

### 公平性メモ
- **生成ライブラリサイズ**は結果を歪める（1万生成と100万生成で best-k を取る比較は不公平）。`run_generation.py --n-samples` を全モデル固定にし、per-target の生成数を常に記録する。
- **Vina の3モード差分**を見る：Vina Dock だけ良いモデルは「生成 pose は悪いが再ドッキングで良く見える」。Score↔Dock の差がそれを暴く。
- 化学妥当性は**全モデルで同一の bond 再認識**（Open Babel）を通して採点する（`molio.py`）ので、SDF の綺麗さでは得しない。

## セットアップ

```bash
git submodule update --init --recursive   # クローン直後の場合

uv sync                                    # bench（評価）環境。軽量・GPU 不要
uv sync --group interactions               # ④ を使う場合のみ（ProLIF + MDAnalysis）

# 各生成モデルの環境（重い・CUDA 依存・互いに非互換）
sh scripts/setup_envs.sh diffsbdd targetdiff diffgui
#   Ours は作業コピーの既存 uv venv を使う（SBDD_OWN_PYTHON で上書き可）

# 重み
python scripts/fetch_weights.py --all
#   --own        : 作業コピーから LM/VQVAE ckpt + descriptor cache を symlink
#   --diffsbdd   : ローカルの crossdocked_fullatom_cond.ckpt を symlink
#   --targetdiff : pretrained_diffusion.pt の置き場所を表示（Google Drive）
#   --diffgui    : trained.pt + bond_trained.pt の置き場所を表示（Google Drive）
```

## ターゲットの準備

```bash
# 単一ターゲット（例: EGFR 2ITY）
python scripts/prepare_targets.py --pdb-id 2ITY --ligand-resname IRE --tag 2ity

# CrossDocked テスト集合（100 ポケット, 各モデル論文の標準評価集合）
#   *_pocket*.pdb と対応 *.sdf を含むフォルダを渡す
python scripts/prepare_targets.py --crossdocked-test data/crossdocked_test

# 自前リスト（receptor.pdb + ref_ligand.sdf のペア）
python scripts/prepare_targets.py --pairs my_targets.json
```

各ターゲットにつき receptor / ≤10 Å pocket / 参照リガンド SDF / receptor.pdbqt / docking box を生成し、`data/targets/index.json` に追記する。

## 実行

```bash
# 生成（GPU ノード, モデルごとに1回）
python scripts/run_generation.py --models own       --n-samples 100
python scripts/run_generation.py --models diffsbdd  --n-samples 100
python scripts/run_generation.py --models targetdiff --n-samples 100
python scripts/run_generation.py --models diffgui   --n-samples 100

# 評価（bench 環境, CPU 可。docking はコア並列）
python scripts/run_evaluation.py --models own diffsbdd targetdiff diffgui \
    --train-smiles data/crossdocked_train_smiles.txt   # novelty 用（任意）

# 速い確認（docking 抜き、または先頭 N 分子だけ docking）
python scripts/run_evaluation.py --models own --no-dock
python scripts/run_evaluation.py --models own --dock-limit 100
```

出力：
- `results/per_molecule.parquet` — 1分子1行、全指標
- `results/per_target.csv` — (model, target) ごとの集計
- `results/per_model.csv` — モデルごとの集計（ターゲット平均, headline 列）

高コストなモデル（DiffGui 等）を別途評価して後から統合する場合：
```bash
python scripts/run_evaluation.py --models diffgui --results results/diffgui
python scripts/merge_results.py --inputs results results/diffgui --out results
```

## 結果（名前付き3ターゲット: EGFR 2ity / ABL 1iep / D3 3pbl）

4モデルを同一の評価器で採点した実測値（ターゲット平均）。**↓ は小さいほど良い、↑ は大きいほど良い。**

| model | n生成 | valid↑ | PB-valid↑ | clash-free↑ | QED↑ | SA↓ | Vina **Score**↓ | Vina **Min**↓ | Vina **Dock**↓ | scaffold-div↑ | **hit (scaffold-uniq)**↑ |
|-------|------:|-------:|----------:|------------:|-----:|----:|------:|------:|------:|------:|------:|
| **DiffGui**¹ | 61 | 1.00 | **0.70** | 0.51 | 0.50 | **4.21** | **−6.54** | **−8.63** | **−9.96** | **1.00** | **0.133** |
| **DiffSBDD** | 290 | 1.00 | 0.49 | **0.70** | **0.52** | 4.73 | −4.40 | −6.48 | −8.40 | 0.96 | 0.056 |
| **TargetDiff** | 227 | 1.00 | 0.50 | 0.60 | 0.37 | 5.16 | −4.76 | −6.68 | −9.00 | 0.97 | 0.052 |
| **Ours** (pclg) | 300 | 0.96 | 0.21 | 0.30 | 0.39 | 5.74 | **+1.21** | −6.45 | −9.37 | **0.47** | **0.017** |

**読み取り（このベンチが捉える本質）:**
- **Ours は Vina Dock が良い（−9.37, 分子自体は結合可能）が Vina Score が +1.21（生成ポーズが物理的に悪い）。** この **Score↔Dock の大きな乖離**こそ「再ドッキングすれば良く見えるが、出したポーズは悪い」典型例。PB-valid 0.21・clash-free 0.30 も最低で、3D生成モデルの弱点（pose 品質）がそのまま出ている。
- **Ours は scaffold diversity 0.47 と低く（mode collapse）**、hit が出ても骨格が重複するため **scaffold-unique hit-rate が 0.017 に崩落**。単純な hit-rate（0.053）だけ見ると他と並ぶが、骨格多様性で割ると弱さが露呈する——主指標を scaffold-unique にした狙い通り。
- **DiffGui が全 Vina モードと PB-valid で最良**だが、**¹ 生成数が 20/target と他（~100）より大幅に少ない**点に注意（小さいライブラリは uniqueness/diversity/hit を過大評価しやすい——ベンチが警告する library-size 交絡）。公平比較には生成数を揃える必要がある。DiffGui は pool ベースで ~2-3分/分子と遅いため、この実行では数を絞った。
- **DiffSBDD は clash-free・QED・SA のバランスが良く**、**TargetDiff は Vina Score（生成ポーズの良さ）が拡散モデル中で最良**。

> このように「Vina が低い＝SOTA」では決まらない。**物理的に妥当な pose で・正しく結合し・合成可能で・多様な候補**を出せるかを多指標で見るのが本ベンチの主旨。生成数を揃えた本評価（CrossDocked テスト等）が次のステップ。

## システム要件

- **AutoDock Vina** + **Open Babel**（docking / リガンド prep）, **ADFRsuite prepare_receptor**（receptor pdbqt）。`sbddbench/paths.py` で場所を解決（`SBDD_VINA` 等で上書き可）。
- 生成は GPU（H100 で確認）。評価は CPU で可。
- TSUBAME ではジョブとして投げる（生成は GPU リソースタイプ、評価は CPU でも可）。

## メモ
- TargetDiff / DiffGui は upstream の env が CUDA 11.8 固定の exact-build export。新しい GPU では torch / pytorch-scatter を cu118 ビルドに緩める必要があるかもしれない（cu118 は H100/sm_90 対応）。
- DiffGui はサンプリング中に QuickVina で各分子を採点するため、生成 env に qvina バイナリが必要（`third_party/DiffGui/softwares/`）。
