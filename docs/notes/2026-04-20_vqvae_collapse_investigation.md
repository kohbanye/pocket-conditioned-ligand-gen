# Protein VQ-VAE Codebook Collapse — 2026-04-17〜04-20 Investigation

4/7 の修正 (`docs/2026-04-07_vqvae_loss_divergence.md`) 後も protein 側の VQ-VAE は繰り返し崩壊した。本ドキュメントは 4/17–4/20 にかけての複数 run にまたがる調査・仮説検証・対応の全記録。

## 背景とスコープ

本プロジェクトの VQ-VAE は protein pocket backbone (ESM3 と同様の Z-matrix 記述子) と ligand 構造 (Z-matrix) を同時に学習する二系統構成。両系統とも `TransformerVQVAE` (同じクラス) を使い、`VQVAEModule` 内で共有オプティマイザで同時訓練する。**何度修正しても protein 側だけが特定 epoch で急性崩壊する**症状が出続けたのが本調査の契機。

## Run 一覧と崩壊タイミングの推移

| run id | 主な構成差分 | 崩壊 epoch | 失敗モード |
| --- | --- | --- | --- |
| `whit8eqh` | baseline: 4/7 の codebook L2 正規化 + `log_scale` + grad clip=1.0。LR=1e-3 固定。 | 25–30 | 緩やかな codebook 崩壊 (util 0.9→0、perplexity 1300→1) |
| `avp1nzme` | + 入力 descriptor の LayerNorm、`commitment_cost` を protein だけ 0.25→0.1、dead-code restart を `usage_count` から `ema_cluster_size < 0.1` に切替、`log_scale` は残す | ~20 | 同上 (1 epoch での急性崩壊)。perplexity 1380→3 を 86 ステップで踏破 |
| `k3cmtzso` | + LR scheduler (`LinearLR` warmup + `CosineAnnealingLR`)、診断メトリクス拡充 (後述) | ~20 | `z_pre_norm` の段階的成長 → 1 epoch で 82→770 に急増 → 崩壊 |
| `7bfljpvs` | + 出力 (latent) LayerNorm `nn.LayerNorm(latent_dim)`、codebook 内部の L2 正規化と `log_scale` を全撤去 | ~11 | `z_pre_norm` は抑えられたが `grad_norm` が 0.25 → 3,530 に 14,000 倍急増。optimizer の崖落ち |
| `oqdbacxx` (現行) | + peak LR 1e-3→3e-4、submodule 別の grad_norm/param_norm、`recon_max`、`latent_norm` γ、AdamW `exp_avg_sq` の平均をログ追加 | 観察中 | — |

崩壊が**早まる**傾向があった事実が重要。表面的な対処 (restart 発火条件の修正、入力正規化、commitment_cost 引き下げ) は individual には正しかったが、**本質 (commitment loss と encoder ノルムの decouple) には届かなかった**ため他の安定化要素 (正規化が弱い状態が逆に効いていたなど) を剥がすと崩壊時期が手前に現れた。

## 観測された症状

### `whit8eqh` / `avp1nzme` (4/17–4/18)

wandb screenshot から読み取った挙動:

- `train/protein_recon` が 0.2–0.4 で推移後、ある epoch で **1.0 付近に張り付く**
- `train/protein_codebook_util` が 0.8 → 0 付近に崩壊
- `train/protein_perplexity` が 1200–1500 → 0 付近
- `train/protein_commit` が 1e-3 → 1e-7 に急落 (コードが1つに集中して距離ゼロ)
- ligand 側は全く健全 (perplexity ~300、util > 0.9 維持)

### `k3cmtzso` (4/19 前半、診断メトリクス付き)

per-epoch の集計 (抜粋):

| epoch | protein_recon | z_pre_norm_mean | z_diversity | perplexity | util |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.175 | 21.3 | 5.0 | 955 | 0.88 |
| 10 | 0.091 | 66.4 | 16.6 | 1350 | 0.99 |
| 20 | 0.094 | 81.6 | 20.4 | 1410 | 0.99 |
| **21** | **0.666** | **770** | 28 | 534 | 0.49 |
| 22 | 1.00 | 2,310 | 53 | 1.8 | 0.02 |
| 31 | 0.991 | 13,700 | 168 | 1.0 | 0.0007 |

**`z_pre_norm_mean` は epoch 0→20 で 21→82 と単調増大**し、epoch 20→21 の 1 epoch で **82→770 に10倍**に跳ねた。この時点で grad_norm 自体は 0.45 と小さく、clip にかかっていない。つまり**崩壊の直接トリガーは encoder 出力ノルムの発散**だった。

### `7bfljpvs` (4/19 後半、出力 LayerNorm 版)

| epoch | protein_recon | z_pre_norm | z_diversity | ema_cluster_size_min | grad_norm |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.147 | 8.34 | 1.90 | 4.81 | 0.614 |
| 5 | 0.073 | 11.7 | 2.39 | 2.44 | 0.322 |
| 10 | 0.072 | 11.8 | 2.31 | 2.21 | 0.220 |
| **11** | **0.356** | 11.7 | 1.97 | 1.40 | **3,530** |
| 12 | 0.705 | 11.2 | 1.58 | 0.45 | **10,500** |
| 13 | 0.605 | 10.7 | 1.45 | 0.28 | **37,100** |

`z_pre_norm` は 11.7 前後で張り付いている (LayerNorm 効果) が、epoch 11 で **`grad_norm` が 14,000 倍に急増**。`commitment_loss` は崩壊中にむしろ下がる (0.054→0.017)。`z_diversity` は 10 epoch で緩やかに低下 (2.3→1.45)、`ema_cluster_size_min` も 4.8→2.2→0.28 と右肩下がり。前回とは異なる失敗モード。

## 仮説と検証プロセス

### 仮説 A: Dead-code restart 機構の破綻 (verified & fixed)

#### 原因

`EMACodebook._restart_dead_codes` は `usage_count` という累積カウンタで判定していた:

```python
# 旧コード
self.usage_count.add_((batch_cluster_size > 0).long())  # モノトニック増加
...
dead_mask = self.usage_count < self.dead_code_threshold  # threshold=2
```

問題点: 各 step でそのコードが1回でも使われると `usage_count` が増える。`threshold=2` のため**学習 2 step 目以降は事実上すべてのコードが永久に alive 判定**になり、後半で実際にコードが使われなくなっても restart が一度も発火しない。`_restart_dead_codes` がデッドコードだった。

#### 対応 (`src/tokenizers/codebook.py`)

判定を EMA ベースに変更:

```python
# 新コード
dead_mask = self.ema_cluster_size < self.dead_code_threshold  # threshold=0.1
```

`ema_cluster_size` は既に EMA 更新されており「最近使われていない」を自然に表現できる。`usage_count` バッファは削除。`_restart_dead_codes` は restart した件数を tensor で返すように変更し、diagnostics から監視可能にした。

#### 検証

run `avp1nzme` 以降、崩壊後に `num_restarted` が毎 step 数件発火することを wandb で確認。修正自体は正しく動作している。

#### しかし崩壊は解決しなかった

- 崩壊が**1 epoch (~86 step) で完了**するのに対し、`ema_cluster_size` が `0.1` を下回るには `ema_decay=0.99` で 700+ step かかる。restart が発火する頃には encoder 側が先に潰れている。
- `_restart_dead_codes` は現在のバッチの z をサンプルして置き換えるので、encoder が既に縮退していると restart 先も縮退した z になる。結果として restart がコードブックの健全性を回復できない。

この「restart が追いつかない」構造自体が次の仮説につながった (encoder を先に止める必要がある)。

### 仮説 B: Commitment pressure が encoder ノルムに効かない (root cause of `avp1nzme`/`k3cmtzso`)

#### 観測

`k3cmtzso` の診断メトリクスから `z_pre_norm_mean` が単調に育ち、臨界で破裂することを確認。崩壊前の `grad_norm` は小さい (0.2–0.45)、`log_scale` もほぼ 1.3 で一定、`lr` は cosine decay 正常動作。つまり**大域的な optimizer 不安定ではなく、encoder 出力のノルムが青天井に育つこと自体が現象を生んでいた**。

#### メカニズム

4/7 の修正で codebook 内部が以下のようになっていた:

```python
z_norm = F.normalize(z, p=2, dim=-1)          # ||z_norm|| = 1
embedding_norm = F.normalize(self.embedding, p=2, dim=-1)
distances = z_norm の Euclidean 距離 → embedding_norm
commitment_loss = cost * (z_norm - quantized.detach()).pow(2).mean()
quantized = z_norm + (quantized - z_norm).detach()  # STE
quantized = quantized * log_scale.exp()
```

この設計の問題:

1. **Commitment loss が `z_norm` (方向のみ) の関数**。encoder が `||z||` をいくら大きくしても commitment 値は変わらない。
2. **勾配の縮み**: `∂z_norm/∂z = (I - z̄z̄ᵀ)/||z||`。`||z||` が育つほど encoder に返る commitment loss / STE の勾配が `1/||z||` で縮む。
3. **正のフィードバック**: encoder は「`||z||` を大きくすれば commitment から逃げられる」方向に最適化される → `||z||` が育つ → さらに逃げやすくなる。
4. **臨界破裂**: ある点で `||z||` のドリフトが自己触媒的な暴走に入り、1 epoch で 10 倍に膨らんで `z_norm` の direction も不安定化 → コードブック分布と完全に decouple し崩壊。

なぜ ligand では起きないか:

- ligand: `descriptor_dim=4`, `latent_dim=8`, `codebook_size=1024`
- protein: `descriptor_dim=12`, `latent_dim=16`, `codebook_size=2048`

protein 側は潜在空間も入力も広く、逃げ道が多い。また protein backbone の二面角分布は α-helix/β-sheet に強く集中する (退化が大きい) ため、encoder が「全体を1〜数点に潰す」戦略を取りやすい。L2 正規化による decoupling がなかった時代 (4/7 以前) は commitment が magnitude を直接縛っていたため隠れていた。

#### 対応 (`src/tokenizers/vqvae.py` + `src/tokenizers/codebook.py`)

**Step 1: Encoder 出力に LayerNorm を置いて `z` を unit-std に固定する**

```python
# TransformerVQVAE.__init__
self.latent_proj = nn.Linear(h, latent_dim)
self.latent_norm = nn.LayerNorm(latent_dim)  # NEW

# TransformerVQVAE.forward
z = self.latent_norm(self.latent_proj(h))
```

`encode` メソッド側にも同じ変更を反映。LayerNorm の学習可能 gain `γ` で若干のスケール調整は許すが、`||z||` の不定な成長を構造的に防ぐ。

**Step 2: Codebook 内部の L2 正規化と `log_scale` を撤去**

encoder 出力がバウンドされたので内部正規化は冗長かつ有害。標準 VQ-VAE (Van Den Oord+ 2017) に戻す:

```python
# EMACodebook.forward (新)
distances = (
    z.pow(2).sum(dim=1, keepdim=True)
    - 2 * z @ self.embedding.t()
    + self.embedding.pow(2).sum(dim=1, keepdim=True).t()
)
indices = distances.argmin(dim=1)
quantized = self.embedding[indices]
commitment_loss = self.commitment_cost * (z - quantized.detach()).pow(2).mean()
quantized = z + (quantized - z).detach()
```

これで commitment が生の `z` に作用し、encoder に magnitude 抑制圧が戻る。`log_scale` Parameter は削除、`lookup` も `self.embedding[indices]` を返すだけに。

#### 検証

run `7bfljpvs` で `z_pre_norm_mean` は epoch 0–13 を通じて 8–12 で横ばい。`z_pre_norm_max` も暴走せず。encoder ノルム暴走は完全に抑えられた。

#### しかし崩壊は解決しなかった (別モードが現れる)

`7bfljpvs` は epoch 11 で grad_norm が爆発して崩壊。これは仮説 B では説明できない別の現象 → 仮説 C へ。

### 仮説 C: Optimizer 崖落ち + AdamW の二次モーメント過小 (suspect of `7bfljpvs`)

#### 観測

- epoch 0–10: `grad_norm` は 0.2–0.4 で非常に安定。`z_pre_norm` も一定。`z_diversity` は 2.3 付近でやや低下傾向。
- epoch 11: 1 step 内で `grad_norm` が 3,500 に跳ねる。`recon` が 0.07→0.36 に上昇。`commit_loss` は下がる (0.054→0.043)。
- epoch 12–13: `grad_norm` が 10,500 → 37,100 と発散。`z_diversity` も 1.97 → 1.58 → 1.45 と低下。`ema_cluster_size_min` が 1.4 → 0.45 → 0.28 と急低下。

注目点:

- `z_pre_norm` は崩壊中もほぼ横ばい (11.7→10.7) で、仮説 B のメカニズムは発動していない。
- grad clip (`max_norm=1.0`) は applied norm を 1.0 に抑えたが、**方向**は暴発した raw gradient に支配される。
- `commit_loss` が減るのは encoder が数点に潰れて commitment の分散が縮んだ結果 (posterior-collapse 的挙動)。

#### メカニズム (仮説)

AdamW の更新式は `Δθ = lr × m / (√v + ε)`。10 epoch ほぼ一定の `grad_norm ≈ 0.25` で学習が続くと、二次モーメント推定 `v` が相応に小さい値で平衡する (各成分で ~0.06)。ここで 1 step だけ raw gradient が跳ねると `√v + ε` が小さいままのパラメータで `lr/√v × g` が極端に大きくなる — 実効的な step size が何倍にも膨らみ、encoder がバッドランドスケープに飛ばされる。一度飛ぶと recon が悪化 → 次の grad はさらに大きい → runaway。

grad clip が effective norm を抑えても、Adam の `v` は raw gradient で更新されるので後続 step の分母も小さいまま (clip 後の値は使わない)。clip は現 step を抑えるが**将来の step の adaptive LR は暴走寄りにシフトする**。

#### 対応 (`src/config.py` + `src/model/vqvae_module.py`)

ユーザ判断により段階的に検証する方針。まずは **peak LR を 1e-3 → 3e-4 に下げる**:

```python
# src/config.py
@dataclass
class VQVAETrainingConfig:
    learning_rate: float = 3e-4  # was 1e-3
```

これで `lr / √v` の絶対値が下がり、1 step の flight distance を縮める。もし崩壊が十分遅延しない場合は warmup 延長、grad clip 締め付け、bf16→fp32 に進む (後述「残課題」参照)。

#### 検証 (run `oqdbacxx` で進行中)

diagnostic メトリクス大幅拡充 (次節)。仮説の最終判定は次回崩壊時の `recon_max` / submodule 別 `grad_norm` / `adam_v_mean` の推移で行う。

## 追加した診断メトリクス (時系列)

### 初期 (baseline)

- `train/protein_{recon,commit,perplexity,codebook_util}`
- `train/total_loss`, `train/global_step`

### `avp1nzme` (4/17) で追加 (commit: dead-code 修正と同時)

- なし (既存メトリクスの精査段階)

### `k3cmtzso` (4/19 前半) で追加 — Codebook / encoder の静的状態を可視化

- `train/{protein,ligand}_log_scale` — learnable scale の推移
- `train/{protein,ligand}_ema_cluster_size_{min,mean,max}` — EMA 使用統計
- `train/{protein,ligand}_num_dead_codes` — `ema_cluster_size < threshold` のコード数
- `train/{protein,ligand}_num_restarted` — 各 step で置換した件数
- `train/{protein,ligand}_z_pre_norm_{mean,max}` — L2 正規化**前**の encoder 出力ノルム (仮説 B の smoking gun)
- `train/{protein,ligand}_z_diversity` — `z.std(dim=0).mean()`, encoder 多様性
- `train/grad_norm` — clip 前の global gradient norm
- `train/lr` — scheduler 出力

### `oqdbacxx` (4/20) で追加 — 失敗源の局所化用

- `train/{protein,ligand}_recon_max` — バッチ内のトークン単位 MSE 最大値。**病的な単発サンプルがバッチ平均に埋もれる問題**を可視化
- `train/{protein,ligand}_grad_norm_{encoder,decoder,latent_proj,latent_norm}` — submodule 別 grad_norm (pre-clip)。爆発源を encoder か decoder か latent projection か特定
- `train/{protein,ligand}_param_norm_{encoder,decoder,latent_proj,latent_norm}` — submodule 別 weight norm。weight drift / decoder 成長 (仮説 D) 検出
- `train/{protein,ligand}_latent_norm_gamma_{mean,max}` — 出力 LayerNorm の学習可能 gain。育ち過ぎると LN が実質無効化される
- `train/adam_v_mean` — AdamW `exp_avg_sq` 全 parameter の平均。仮説 C 「v 過小 → adaptive LR 膨張」の直接検証

## まだ検証できていない他の仮説

### 仮説 D: Decoder weight の成長による勾配増幅

Decoder は 6 層 Transformer + output_proj、無制約。10 epoch かけて weight norm が育ち、`z` の小変動が `x_hat` の大変動に増幅されて recon gradient が大きくなる説。`oqdbacxx` の `param_norm_decoder` で観察予定。

### 仮説 E: 病的バッチ (単発異常サンプル)

特定の pocket が欠損残基による壊れた Z-matrix / 異常距離を含み、1 バッチで recon loss が跳ねる説。`recon_max` で検出予定。

### 仮説 F: Posterior collapse

Decoder が latent 情報を使わなくなり、encoder に有意義な勾配が返らなくなって不安定化する説。`commit_loss` が崩壊中に下がる現象と整合。もし顕在化すれば decoder の仕様検討 (容量絞り、再構成 loss の重み付け等) を検討。

### 仮説 G: bf16-mixed の数値起因

`precision='bf16-mixed'`。LayerNorm backward の std 割り算、AdamW の `m`/`v` 累積で bf16 のダイナミックレンジ不足が効きうる。切り分けには fp32 run が必要 (メモリ・速度コスト)。

### 仮説 H: EMA codebook と encoder の race condition

`ema_decay=0.99` なので codebook は encoder を ~100 step 遅れで追う。encoder が Adam kick で急に動くと一時的に z と code の距離が開き、commitment 勾配が大きくなって encoder をさらに動かす振動説。`ema_decay` を 0.995/0.999 にすると改善するか、あるいは悪化するか (codebook が追従できず多様性が失われる方向) は未検証。

## 現在のコード構成 (2026-04-20 時点)

### `src/tokenizers/codebook.py` — `EMACodebook`

- **ベクトル正規化なし**。raw z と raw embedding で Euclidean 距離、commitment, STE, EMA 更新。
- Dead-code restart は `ema_cluster_size < 0.1` 判定、z をサンプルして置換。restart 件数を返す。
- `log_scale` parameter, `usage_count` buffer は削除済み。
- Diagnostics: `log_scale` 以外の 7 指標 (ema_cluster_size stats, num_dead_codes, z_pre_norm stats, num_restarted)。

### `src/tokenizers/vqvae.py` — `TransformerVQVAE`

- `input_norm = nn.LayerNorm(descriptor_dim)` を forward 冒頭に適用 (入力 descriptor の正規化、4/17 追加)。
- `latent_proj` 直後に `latent_norm = nn.LayerNorm(latent_dim)` を適用し `z` をバウンド (4/19 追加)。
- forward は dict を返す: `reconstructed`, `indices`, `commitment_loss`, `reconstruction_loss`, `diagnostics`。
- `diagnostics` には codebook 側 + `z_diversity` (std across tokens) + `recon_max` (per-token MSE の最大値) を同梱。
- `encode` メソッドも同様に `input_norm` / `latent_norm` / 4要素 unpack に追随。

### `src/model/vqvae_module.py` — `VQVAEModule` (Lightning)

- `precision='bf16-mixed'`, `automatic_optimization=False`。
- `configure_optimizers`: `AdamW(lr=3e-4)` + `SequentialLR(LinearLR warmup=min(500, total/20) steps, CosineAnnealingLR eta_min=3e-6)`。dict 形式で返す。
- `training_step`:
  1. `total_loss` を protein/ligand 分累積。各系 diagnostics をログ。
  2. `opt.zero_grad()` → `manual_backward`。
  3. **Pre-clip**: `_log_submodule_grad_norms()` で submodule 別 raw gradient norm を記録。
  4. `clip_grad_norm_(max_norm=1.0)` で global を clip、`train/grad_norm` に pre-clip 値をログ。
  5. `opt.step()`。
  6. **Post-step**: `_log_submodule_param_norms()`, `_log_latent_norm_gain()`, `_log_adam_v_mean(opt)`。
  7. `sch.step()` を手動、`train/lr` を log。
- `validation_step` / `test_step` でも diagnostics を log (ただし submodule stats/Adam v は train のみ)。

### `src/config.py` (抜粋)

```python
@dataclass
class ProteinVQVAEConfig:
    descriptor_dim: int = 12
    hidden_dim: int = 256
    latent_dim: int = 16
    codebook_size: int = 2048
    commitment_cost: float = 0.1  # 4/17 に 0.25 から変更
    ema_decay: float = 0.99
    num_transformer_layers: int = 4
    ...

@dataclass
class VQVAETrainingConfig:
    learning_rate: float = 3e-4  # 4/20 に 1e-3 から変更
    mol_batch_size: int = 4096
    max_epochs: int = 100
    precision: str = 'bf16-mixed'
    ...
```

## 各修正の評価

| 修正 | 要否 | 理由 |
| --- | --- | --- |
| 4/7: codebook の L2 正規化 + `log_scale` | **撤去** | 仮説 B の元凶。encoder との decouple を引き起こしていた |
| 4/7: grad clip max_norm=1.0 | 維持 | 7bfljpvs の崩壊で grad_norm=37,100 を 1.0 に抑えた実績あり。方向問題は残るが applied step は抑制。保険として有効 |
| 4/17: 入力 descriptor の LayerNorm | 維持 | protein descriptor (距離・角度) の scale 混在を解消。直接の原因ではないが一般的に健全 |
| 4/17: `commitment_cost` 0.25→0.1 (protein のみ) | 維持 (要再検討) | encoder 出力が bounded になった今、0.25 に戻すと commitment 圧が強まり encoder 多様性を維持しやすい可能性。今後の実験で検討 |
| 4/17: dead-code restart を EMA ベースに | 維持 | 実際に restart が発火するようになったが急性崩壊には間に合わない。予防というより詳細解析用 |
| 4/19: LR scheduler (warmup + cosine) | 維持 | adaptive LR の late training 暴走抑制に寄与。`lr` が見えるメリットもあり |
| 4/19: 出力 LayerNorm + codebook 正規化撤去 | 維持 | 仮説 B の根本対策。z_pre_norm は実際に安定化 |
| 4/20: peak LR 1e-3→3e-4 | 検証中 | 仮説 C への対応 |
| 4/17–4/20: 診断メトリクス拡充 | **最重要資産** | z_pre_norm の暴走、grad の局所爆発といった具体的現象の特定はこのログがあって初めて可能になった |

## 残課題と次の打ち手候補

`oqdbacxx` (現行) で再度崩壊した場合の優先順位案:

1. **warmup を長く**: 現 500 step (~5% of total) → 2000 step (~10%)。Adam の `v` 推定に時間を与える。実装は `configure_optimizers` で warmup steps の計算式を変えるだけ。
2. **grad clip を 1.0→0.5**: outlier step の方向汚染を更に減衰。
3. **`bf16-mixed` → `'32'`**: メモリ・速度のトレードオフを受け入れ、LayerNorm / AdamW の数値安定性を確保。
4. **encoder に多様性正則化**: `z_diversity` が緩やかに下がる傾向への明示的対策。`-λ × z.std(dim=0).mean().log()` のようなエントロピー項を損失に追加。
5. **`latent_dim` 16→8**: protein 潜在空間を縮小してコードブック密度を上げる (ligand と揃える)。データ退化 (helix/sheet) への過剰表現力を抑制。
6. **`ema_decay` 0.99→0.995/0.999**: codebook の追従を遅らせて race を和らげる。逆に codebook 陳腐化の副作用あり。
7. **`commitment_cost` を 0.25 に戻す**: encoder 出力が bounded になった今、より強い commitment 圧が encoder 多様性維持に効く可能性。

## 参考

- 4/7 の経緯: `docs/2026-04-07_vqvae_loss_divergence.md`
- 関連 wandb run (all under `kohbanye/pocket-ligand-vqvae`):
  - `whit8eqh`, `avp1nzme`, `k3cmtzso`, `7bfljpvs`, `oqdbacxx`
- Van Den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
  — 標準 VQ-VAE。`avp1nzme` → `7bfljpvs` で事実上この定式に回帰した
- Yu et al., "Vector-quantized Image Modeling with Improved VQGAN", ICLR 2022
  — codebook 内部の L2 正規化を導入した論文。本プロジェクトでは encoder 出力に explicit な LayerNorm を置くことで内部正規化を不要にした
- Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2", NeurIPS 2019
  — EMA-based codebook update と dead-code restart の原型
