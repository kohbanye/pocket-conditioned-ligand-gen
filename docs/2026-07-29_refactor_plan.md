# リファクタ計画 — ProLIT モノレポ化 (2026-07-29)

## 0. 前提: いま「効いている手法」= 論文 ProLIT のスタック

`../aaai27-paper/main.tex` (AAAI 2027 投稿, *Learning the Language of the Binding
Interface*) が唯一の正。ここに載る構成だけを本流とし、それ以外は落とす。

| 層 | 実体 | 主要 checkpoint | コード |
|---|---|---|---|
| **トークナイザ (本流)** | joint all-atom VQ-VAE, 33-D descriptor, 単一 codebook 8192 (vocab 8199) | `pocket-ligand-vqvae/xzkjxu9q/…atom_coord=0.1073.ckpt` | `src/tokenizers/{atom,vqvae,codebook,descriptor_schema,lm_vocab}.py`, `src/data/atom_descriptors.py` |
| **トークナイザ (ablation)** | separate = protein 専用 VQ + ligand 専用 VQ を 1 code space に連結 (8192+8192 / 4096+4096) | `pocket-ligand-vqvae/{protein,ligand}-vqvae[-4096]/last.ckpt` | `src/tokenizers/separate_vqvae.py` |
| **ProLIT-MLM** | encoder-only 複合体 MLM (自前 ESM3 流, 99M) + pose head | `pocket-ligand-mlm/j90rlrgm`, `pocket-ligand-rescore/*` | `src/model/{complex_mlm,mlm_module,rescore_module,mlm_score}.py` |
| **ProLIT-CLM** | Qwen3 系 causal LM (~298M) + e3nn flow-matching pose refiner | `pocket-ligand-lm/p6lpk7br`, `pocket-ligand-refine/refine_atom_bond_v1` | `src/model/{ligand_lm,lm_module,pose_refiner}.py` |

論文の結果表 3 本 = ベンチ 3 本に 1:1 対応する:

| 論文の表 | 内容 | 対応リポジトリ | 状態 |
|---|---|---|---|
| Table 1 `tab:recon-compare` | CASP16 303 複合体の再構成。vs ESM3 / FoldToken4 / Token-Mol / Bio2Token + separate ablation | `../protein-ligand-3d-reconstruction-bench` | **現役**。未コミット 50 件 (`bio2token.py`, `own_allatom.py`, `confseq.py`, `pb.py`) が論文の行そのもの |
| Table 2 `tab:docking` | CASF-2016 docking power。vs Vina / DeepRMSD(+Vina) / GenScore / RTMScore + separate ablation | `../complex-tokenizer-bench` (+ `../baselines`) | 現役。最新かつ設計が一番きれい |
| Table 3 `tab:gen` | 3 ターゲット生成。vs DiffSBDD / TargetDiff + separate ablation | `../sbdd-bench` + ctbench | **進行中**。`\TODO{}` が残り、`ctb_gen_tr` (job 8295960) が実行中 |

> **`protein-ligand-3d-reconstruction-bench` は「もう使っていない」わけではない** —
> 論文 Table 1 の実体です。ここが今回いちばん重要な認識のズレでした。
>
> `affinity-prediction-bench` は別プロジェクトとして対象外 (ユーザ確認済み)。
> なお本リポジトリの affinity head 系コードは動作するが論文からは落ちているので、
> 削除はせず「非 paper-critical」として `benchmarks/affinity/` に隔離する。

---

## 1. いま何が壊れているか

1. **ルート汚染** — SGE ログ 535 個 (`*.o<jobid>` / `*.e<jobid>`) がリポジトリ直下。ディレクトリエントリ 549 個。
2. **`scripts/` が実験の履歴書になっている** — `.py` 62 本 + job script **129 本**。`job_tok_decoys_{v2,v8,v8rot,v10,big,huge}.sh` のように 1 アーム 1 ファイル。
3. **トークナイザが 3 世代同居** — ①legacy 2-codebook (残基レベル protein spherical + ligand descriptor, `3dvcbp0h`) ②split-codebook (共有 encoder + 2 book, `ix6q6po0`) ③all-atom joint + separate。**論文に載るのは③だけ**。①②の分岐が 8 スクリプト・56 箇所の CLI フラグに残存。
4. **tokenize スクリプトの重複** — `_Encoder` クラス (`_norm`/`add`/`_encode`/`flush`/`flush_all`) が `tokenize_{plinder_protein,biolip,dataset_atom,geom_atom}.py` に 4 回コピーされている。計 ~2,300 行。
5. **評価コードが 3 箇所に散在** — 本リポジトリの `scripts/eval_*.py` / ctbench / sbddbench。Vina 呼び出しと PoseBusters がそれぞれに存在。
6. **循環 submodule** — `sbdd-bench` と `plbench` がそれぞれ `third_party/pocket-conditioned-ligand-gen` として**本リポジトリ自身**を submodule に持っている。
7. **`src` がパッケージ名になっていない** — ベンチ側が `from src.config import …` で参照し、さらに **private を掴んでいる** (`src.data.rescore_dataset._ligand_mask`, `src.tokenizers.protein._compute_canonical_frame`)。公開 API が未定義。
8. **`src/config.py` が 600 行 15 dataclass の一枚岩**。CLAUDE.md は Hydra を謳うが実際には未使用 (全 CLI が argparse)。
9. **衛生** — `ruff check` で 52 errors。`pytest --collect-only` に 130 秒 (import 時に torch/rdkit/e3nn を総なめ)。依存はフラット 1 リスト (ベンチ用の posebusters/prolif と学習用が同居)。
10. **明確な死骸** — `src/evaluation/` (空), `scripts/train.py` (boilerplate 雛形), `src/data/{crossdocked,hub_crossdocked}.py` (雛形からのみ参照), `src/tokenizers/sequence.py` + `scripts/tokenize_data.py` (テキスト形式トークン化、未使用)。stale worktree `.claude/worktrees/idempotent-watching-shore`。

---

## 2. ディレクトリ構成の提案

提示いただいた構成は骨格として妥当ですが、このプロジェクト固有の事情で 8 点変えることを勧めます。

```
prolit/                                  # = 現 pocket-conditioned-ligand-gen
├── pyproject.toml                       # base + [optional-dependencies] bench-* に分割
├── uv.lock / README.md / CLAUDE.md
│
├── src/prolit/                          # ライブラリ (import 可能・副作用なし・argparse なし)
│   ├── config/                          # 600行の一枚岩を用途別に分割
│   │   ├── data.py  vqvae.py  lm.py  mlm.py  head.py  refiner.py
│   ├── chem/                            # モダリティ非依存の RDKit / PDB / 幾何
│   │   ├── io.py                        # parse_sdf(_text), parse_ligand_pdb_text, receptor parse
│   │   ├── pocket.py                    # pocket candidates / extraction / canonical frame
│   │   └── geometry.py                  # spherical, KNN frame, Kabsch
│   ├── tokenizer/                       # ★ 論文の中核
│   │   ├── schema.py                    # ATOM_LAYOUT (33-D)
│   │   ├── descriptor.py                # 複合体 -> per-atom 33-D
│   │   ├── vqvae.py  codebook.py        # TransformerVQVAE + EMACodebook
│   │   ├── separate.py                  # SeparateVQVAE (ablation)
│   │   ├── vocab.py                     # AtomLMVocab
│   │   └── api.py                       # ★ 公開 API: encode_complex / encode_pose / decode_to_atoms
│   ├── models/
│   │   ├── clm.py + clm_module.py       # ProLIT-CLM
│   │   ├── mlm.py + mlm_module.py       # ProLIT-MLM
│   │   ├── heads.py                     # pose / affinity head (pooling 3 種)
│   │   ├── refiner.py                   # e3nn flow-matching
│   │   └── vqvae_module.py
│   ├── data/
│   │   ├── descriptor_cache.py          # shard I/O + Welford 正規化統計
│   │   ├── token_cache.py               # ★ .bin/.len/.rmsd + TokenStreamWriter (4重複を一本化)
│   │   └── datasets/{lm,mlm,rescore,pose_refine}.py
│   └── generate.py                      # ★ 公開 API: pocket -> ligand SDF (sample+decode+refine)
│
├── pipelines/                           # コーパス構築と学習の CLI (argparse)
│   ├── corpora/
│   │   ├── sources/{crossdocked,plinder,biolip,geom,casf}.py   # 「複合体を yield する」だけ
│   │   ├── build_descriptors.py
│   │   ├── tokenize.py                  # ★ 1 本の CLI, --source {crossdocked|plinder|…}
│   │   ├── build_decoys.py
│   │   └── mix.py
│   └── train/{vqvae,clm,mlm,head,refiner}.py
│
├── benchmarks/
│   ├── common/                          # docking(Vina) / posebusters / molio / stats / dumps / report / plotting
│   ├── variants.py                      # ★ joint | separate | separate_4096 のレジストリ
│   ├── reconstruction/                  # 論文 Table 1   <- plbench
│   │   ├── adapters/{prolit,esm3,foldtoken,token_mol,bio2token}.py
│   │   └── metrics.py  run.py
│   ├── rescoring/                       # 論文 Table 2   <- ctbench + baselines
│   │   ├── adapters/{prolit,rtmscore,genscore,vina,deeprmsd}.py
│   │   └── metrics.py  run.py
│   ├── generation/                      # 論文 Table 3   <- sbdd-bench + ctbench
│   │   ├── adapters/{prolit,diffsbdd,targetdiff,diffgui}.py
│   │   └── metrics.py  run.py
│   └── affinity/                        # 論文外。動くので残すが paper-critical ではない旨を README に明記
│
├── third_party/                         # submodule 群 (自己 submodule は除去)
├── envs/                                # ベースラインごとの環境仕様 (conda yaml / uv venv 構築スクリプト)
├── jobs/                                # SGE: templates/ + gen.py   (129 本 -> ~12 テンプレート)
├── results/                             # <task>/<variant>/ の per-sample dump + tables/ + figures/
├── notebooks/                           # marimo: 論文図のみ
├── docs/{results,notes}/                # 凍結記録 / 日付つき調査ログ
├── runs/                                # gitignore: pocket-ligand-* の checkpoint 群
└── tests/
```

### 提示案からの変更点と理由

1. **`benchmarks/` はモデル軸でなくタスク軸で切る。**
   提示案の `adapters/{ours,baseline_a,baseline_b}` は「1 タスク N モデル」の形。実際は
   **3 タスク × 別データ × 別ベースライン × 別実行環境**で、`reconstruction` は CASP16 と
   ESM3/FoldToken の venv、`generation` は CrossDocked ターゲットと conda の diffusion 環境、
   `rescoring` は CASF-2016 と micromamba の DGL 環境。共通化できるのは metrics/stats/dump/docking
   だけなので、そこを `common/` に集め、adapter はタスク配下に置くのが正しい粒度。

2. **`configs/` (Hydra yaml ツリー) をやめ、`benchmarks/variants.py` の dataclass レジストリにする。**
   ctbench に既に実装済みで機能している (`joint` / `joint_nocasf` / `separate` / `separate_4096`)。
   バリアントは 3〜4 個で固定、値は checkpoint パスと codebook サイズだけ。yaml 階層を挟むと
   「どの ckpt がどのアームか」が追いにくくなるだけです。**Hydra は依存から外す** (現状も未使用)。

3. **`environments/*/Dockerfile` → `envs/*.yaml`。**
   TSUBAME4 はユーザに Docker を提供していません (使えるのは Apptainer)。既に
   `sbdd-bench/envs_spec/{diffgui,targetdiff}.yaml` + `setup_envs.sh` で conda 環境を作る運用が
   動いているので、それを正式な形にするのが素直。コンテナ化したくなったら Apptainer def を足す。

4. ~~**`patches/` は作らない。**~~ → **訂正: 必要でした。**
   Phase 0 で `sbdd-bench/third_party/DiffGui/scripts/sample.py` に**未コミットの改変**
   (内部 Vina docking を optional 化) が見つかりました。submodule は upstream commit に
   pin されるので、この改変は git から見えず clone し直すと消えます。
   `patches/<baseline>/*.patch` + 冪等な `scripts/apply_patches.sh` として保存済み
   (sbdd-bench commit `c14dad3`)。提示案の `patches/` はそのまま採用します。

5. **`pipelines/` を新設する。** — 提示案に欠けている最大のもの。
   このリポジトリの体積と壊れやすさの大半は「生構造 → descriptor cache (2 TB) → VQ →
   token stream (.bin/.len)」のコーパス構築層にあります。ここは `scripts/` の雑多な CLI 群でなく
   一級市民として置き、`src/prolit/` (純ライブラリ) と明確に分ける。

6. **`jobs/` を新設する。** 129 本の SGE スクリプトはテンプレート + パラメータに畳む。
   このリポジトリで**実際に動く** job のパターン (`source ~/.bashrc` と `module load` を使わず
   `.venv/bin/python` を直叩き) をテンプレートに固定して、過去の踏み抜きを再発させない。

7. **`src/my_model/{model,training}.py` は平すぎる。**
   現行の `tokenizers / data / model` 分割自体は妥当で、問題は中の堆積物。そこに
   **`chem/`** (legacy と all-atom の両方が生えてきた RDKit 層) と
   **公開 API モジュール** (`tokenizer/api.py`, `generate.py`) を足す。後者が今回の肝で、
   これがあればベンチ側が `_ligand_mask` のような private を掴む必要がなくなります。

8. **`results/{manifests,summaries}` → `results/<task>/<variant>/…` + `tables/` + `figures/`。**
   ctbench の per-sample dump 規約 (`io_dumps.py`) と、その数値を検証する再現テスト
   (`test_reproduce_*.py`) が既にこの形。壊さずに引き継ぐ。

---

## 3. 実行計画

### Phase 0 — 凍結と地ならし ✅ 完了 (2026-07-30)
- `pre-refactor` タグを 4 リポジトリに打って push。本リポジトリには `legacy-2codebook` も
  (Phase 2 の削除対象を復元可能にするため)。
- ルート直下の SGE ログ・job 出力 **529 個を `logs/sge/` へ退避** → ルートのエントリ **549 → 28**。
- stale worktree `.claude/worktrees/idempotent-watching-shore` と対応ブランチを削除 (main にマージ済みを確認)。
- `.gitignore` に `/logs/`, `/runs/`, `pocket-ligand-*/` を追加。
- **各ベンチの未コミット作業を保全** (Phase 1 の前提):
  - plbench `20f277c` — all-atom アーム 9 種 (joint / separate / separate4096 / binning /
    localframe_{1,1.5,2,3}tok / oracle) + Bio2Token・ConfSeq アダプタ + lDDT-PLI/Contact-F1/
    PoseBusters + 全アームの結果 parquet。**論文 Table 1 の実体**。third_party に
    bio2token・ConfSeq を submodule 登録。
  - sbdd-bench `c14dad3` — DiffGui のパッチを `patches/` へ (上記訂正)。
  - ctbench `b1ed47c` — tp1000 生成スイープ 4 アーム。
- **基準値** (この数値を Phase 2〜5 の回帰判定に使う):

  | 項目 | 値 |
  |---|---|
  | pytest | **66 passed / 0 failed** (203 s) |
  | ruff check | **52 errors** (PLR2004 8, C901 6, PD011 5, F401 4, RUF100 4, 他) |
  | 追跡ファイル | 207 |
  | src+scripts+notebooks 行数 | 29,908 |

### Phase 1 — モノレポ統合 (1 日)
- **先に各ベンチをコミット**する。特に plbench の未コミット 50 件 (論文 Table 1 の実装) と
  sbdd-bench の submodule 更新。ここを取りこぼすと論文の行が復元できなくなる。
- `git subtree add --prefix=benchmarks/<name> <path> <branch>` で履歴ごと取り込む
  (plbench / sbddbench / ctbench の 3 本。affinity-prediction-bench は対象外)。
- 各ベンチの `third_party/pocket-conditioned-ligand-gen` **自己 submodule を除去**。
- `third_party/` をトップレベルに 1 つへ統合 (DiffSBDD, targetdiff, DiffGui, esm, FoldToken_open, token-mol)。
- `pyproject.toml` を統合: base 依存 + `bench-recon` / `bench-gen` / `bench-rescore` の extras。
- ⚠️ 追跡済み結果ファイルが計 ~220 MB (sbdd-bench 117M / ctbench 56M / plbench 44M)。
  取り込む前に「巨大 parquet を残すか、集計 CSV だけにするか」を決める。

### Phase 2 — トークナイザ世代の整理 (1.5 日)
**残すのは joint all-atom と separate の 2 つだけ。** legacy 2-codebook と split-codebook を削除。

削除:
- `src/tokenizers/{ligand,protein}.py` の descriptor クラス (`LigandDescriptor`, `BackboneSphericalDescriptor`)
- `src/data/descriptors.py` の legacy 計算部 (`_process_pose` 系) — ただし後述のとおり**分割**
- `src/tokenizers/sequence.py`, `src/data/{crossdocked,hub_crossdocked,tar_prep}.py`, `src/evaluation/`
- `src/config.py`: `ProteinVQVAEConfig` / `LigandVQVAEConfig` / `VQVAETrainingConfig`,
  `AtomVQVAEConfig.split_codebook` / `.ligand_codebook_size`
- `scripts/`: `train.py`, `tokenize_data.py`, `tokenize_dataset.py`, `tokenize_geom.py`,
  `prepare_descriptors{,_tar}.py`, `train_vqvae.py`, `generate_ligands.py`, `eval_generation.py`,
  `write_reconstruction_pdbs.py`, `diagnose_geometry.py`, `smoke_train_vqvae.py`
- `notebooks/visualization.py` (legacy 可視化)
- `--split-codebook` / `--ligand-codebook-size` フラグを 8 スクリプトから除去 (56 箇所)

**分割 (削除ではない)**: `src/data/descriptors.py` 984 行のうち shard I/O・Welford 統計・
`collate_molecules` / `ShardedMoleculeDataset` は all-atom 側が現に使っている →
`prolit/data/descriptor_cache.py` へ。`parse_sdf` / `precompute_pocket_atom_candidates` /
幾何ユーティリティも同様に `prolit/chem/` へ移す。

見込み: **−4,000〜5,000 行**、CLI の 3 分岐が 2 分岐に。

### Phase 3 — パッケージ化 `src/prolit/` (1 日 / 機械的)
- `src/` → `src/prolit/`、`from src.` → `from prolit.` を一括置換。`pyproject.toml` に `package-dir`。
- **公開 API を定義**: `prolit/tokenizer/api.py` (encode/decode) と `prolit/generate.py`。
  ctbench の `inference/encode.py` が今やっていることをライブラリ側に引き上げ、
  private 参照 (`_ligand_mask`, `_compute_canonical_frame`) を解消する。
- ctbench の `ensure_source_repo_importable()` は同一パッケージ化により不要 → 削除。
- ⚠️ **ジョブキューが空のときに実施すること。** 現在 `ctb_gen_tr` (job 8295960) が実行中で、
  job script は `PYTHONPATH` 直指定なので走行中のジョブを壊します。

### Phase 4 — pipelines/ 集約 (2 日)
- 4 重複している `_Encoder` を `prolit/data/token_cache.py` の `TokenStreamWriter` +
  `ComplexEncoder` に一本化。
- `tokenize_{dataset_atom,plinder_protein,biolip,geom_atom,decoys}.py` →
  `pipelines/corpora/tokenize.py --source X` + `sources/*.py` (各 source は「複合体を yield する」だけ)。
- 見込み: ~2,300 行 → ~1,000 行。**回帰検知として、既存 cache の先頭 N doc とバイト一致することを
  テストに入れる** (トークン列が変わると全 checkpoint が無効になるため、ここは慎重に)。

### Phase 5 — benchmarks 統合 (2〜3 日)
- ctbench の `metrics/` `stats.py` `report.py` `io_dumps.py` `plotting.py` → `benchmarks/common/`。
- sbddbench の `docking.py` `pose.py` `chem.py` `diversity.py` `molio.py` → `benchmarks/common/`
  (Vina と PoseBusters は generation と reconstruction の両方で使う)。
- 各タスクに `run.py`: `--variant joint|separate|separate_4096 --method prolit|rtmscore|…`。
- **ctbench の再現テスト (`test_reproduce_*.py`) は必ず残す** — 論文の数値が動いていないことの
  唯一の自動チェックです。

### Phase 6 — jobs/ テンプレート化 (1 日)
- 129 本 → ~12 テンプレート + `jobs/gen.py` (アーム名・ノード種別・時間をパラメータ化)。
- 動作実績のあるパターン (`.venv/bin/python` 直叩き、`source ~/.bashrc`/`module load` 禁止、
  `export PYTHONPATH`/`WANDB_MODE`) をテンプレートに固定。

### Phase 7 — 衛生 (1 日)
- ruff 52 errors → 0。`pytest --collect-only` の 130 秒を lazy import で短縮。
- 依存を base / bench-* に分割し、未使用の Hydra を削除。CLAUDE.md の記述も実態に合わせる。
- README を書き換え (現在「Boilerplate for ML projects」の 1 行)。ProLIT の説明・3 表・再現手順。
- `docs/` を `docs/results/` (凍結記録: `best_allatom_configs.md` 等) と
  `docs/notes/` (日付つき調査ログ) に整理。

**合計 8〜10 日相当。**

---

## 4. 順序に関する注意

- 論文 Table 3 (generation) にまだ `\TODO{}` があり、`ctb_gen_tr` が走行中。
  **Phase 3 (rename) と Phase 5 (benchmarks 統合) は、生成 ablation が終わってからが安全**。
  Phase 0 / 1 / 2 / 6 / 7 は先行して構いません。
- checkpoint パス (`pocket-ligand-*/…`) は `docs/best_allatom_configs.md` と
  `ctbench/variants.py` が実パス文字列で参照している。`runs/` へ移すなら両者を同時更新する
  (移さずルート据え置きのままでも構わない — 論文の再現性記録としては据え置きの方が安全)。
- トークン列のバイト表現を変える変更 (Phase 4) は、全 checkpoint を無効化しうる。
  既存 cache との一致テストを先に置いてから着手する。
