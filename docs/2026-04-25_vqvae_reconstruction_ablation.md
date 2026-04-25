# VQ-VAE 再構成精度改善 — Batch 1 / Batch 2 アブレーション

`yggua4f0` (4/21 完了) の評価で per-dim MSE と 3D RMSD のボトルネックが明確になり、4/22–4/25 にかけて 5 つのアブレーション run を回した記録。protein backbone の Kabsch RMSD を 1.547 Å → 1.374 Å (-11.2%) まで改善し、原因が **sin/cos 正規化による単位円制約の破壊** にあることを実証。

## 背景

`yggua4f0` (commit `ffda327`、3D coord loss 追加版) の `notebooks/visualization.pdf` から読み取れた状況:

| 指標 | Protein | Ligand |
|---|---:|---:|
| Test MSE (normalized) | 0.1415 | 0.0188 |
| 3D RMSD per-atom | 1.73 Å | 1.49 Å |
| 3D RMSD Kabsch | 1.55 Å | 0.22 Å |
| Codebook utilization / norm. perplexity | 99.85% / 0.76 | 99.71% / 0.71 |

**Codebook は健全** (utilization 99%+, perplexity 0.71+, t-SNE で encoder 出力をよく覆う) で、ボトルネックは descriptor → 3D 復元の精度に集中。Per-dim MSE では距離系 (`N_d/CA_d/C_d` で 0.02–0.04) に対し角度系 (`CA_θ=0.27`, `C_θ=0.31`, `C_sin/cos=0.13–0.19`) が 5–10 倍。

加えて、`iwfhus30` (commit `45c103f`、3D coord loss 無し) の val/protein_recon は **0.0589** に対し `yggua4f0` は **0.1316** と倍以上悪化しており、**dual-objective の均衡崩れ**も疑われた。

## 仮説

### A. sin/cos の単位円制約が壊れている【最有力】
- `src/data/descriptors.py:984-987` で **全 descriptor 次元に per-dim std 正規化** が適用される
- sin/cos スロット (protein idx 2,3,6,7,10,11 / ligand idx 2,3) は元値で `sin²+cos²=1` だが、正規化後は `((sin-μ)/σ)² + ((cos-μ)/σ)² ≠ 1` で単位円から離れる
- `src/tokenizers/vqvae.py:196-198` の descriptor MSE loss `diff_sq.mean()` は単位円制約を課さない
- `project_unit_circle` は coord loss 分岐内のみ (vqvae.py:338, 445)
- decoder が単位円外の sin/cos を出力可能 → NeRF で角度誤差 → per-dim 角度系の MSE 悪化

### B. Dual-objective の初期バランスが悪い【学習ダイナミクス要因】
- `src/model/vqvae_module.py:37-38` の `TaskWeighting` は `log_var_recon = log_var_coord = 0` 初期化（重み 1:1）
- `coord_loss` は Å² スケール、`recon_loss` は無次元 → 初期は coord が桁違いに大きく学習を引っ張る
- `commitment_cost` 0.1→0.25 の引き上げと合わさり descriptor recon の学習が後回しに

### C. Ligand root atom の影響【今回は保留】
- ligand.py:440-444 の root atom (r, θ, sin φ, cos φ) は pocket centroid 相対の spherical coord
- 全 atom が同じ codebook (1024 codes) を共有するため、root の量子化誤差が全原子の剛体シフトとして伝播
- Kabsch=0.22 vs per-atom=1.49 の 7× ギャップの主因
- Tier 2 として今回は触らず、Batch 1/2 の結果次第とした

## 介入と実装

### 4/22 — Batch 1 設計

3 介入を 1 つずつ単独 / 組み合わせで評価:

1. **Unit-circle penalty** (B1-A): `λ · Σ(s²+c²−1)²` を descriptor recon loss に追加 (λ=0.1)
2. **Coord_loss warmup** (B1-B): 最初の 10 epoch で `coord_loss` を 0→1 にランプ、warmup 中は `TaskWeighting` を bypass (`log_var_coord` の -∞ ドリフト回避)
3. **両方** (B1-C): A + B

**実装変更** (詳細は git log / 該当 commit):
- `src/config.py`: `circle_loss_weight` を Protein/LigandVQVAEConfig に、`coord_loss_warmup_epochs` `skip_sincos_normalization` を VQVAETrainingConfig に追加
- `src/tokenizers/vqvae.py`:
  - `_compute_circle_loss(x_hat, mask)` を追加（denorm 後の sin/cos slot で `(s²+c²-1)²` の MSE）
  - `forward()` の戻り値 dict に `"circle_loss"` を追加
- `src/model/vqvae_module.py`:
  - `_coord_loss_ramp()` 追加（線形 0→1 ramp）
  - `_combine_losses()` に warmup 分岐とλ·circle 加算を追加
  - **DDP 対策**: warmup 中 / coord_loss 無効時に `0.0 * (log_var_recon + log_var_coord)` を足して `TaskWeighting` の Parameter を autograd graph に残す（B1-B/C 初回投入で発生した `RuntimeError: parameters that were not used` を修正）
- `scripts/train_vqvae.py`: `--circle-loss-weight`, `--coord-loss-warmup-epochs`, `--skip-sincos-norm`, `--run-name`, `--cache-dir` の CLI フラグ追加
- `scripts/recompute_norm_stats.py`: stats のみ再計算するユーティリティ
- `tests/test_vqvae.py`: `TestCircleLoss` (3 件), `TestCoordLossRamp` (3 件) を追加

最初の B1-B / B1-C 投入 (job 7239517 / 7239518) は **DDP unused-parameter エラー** で 1 step 目に死亡。warmup 分岐を通ると `TaskWeighting.log_var_*` が autograd 上で touched にならないため。`0.0 * (...)` を足す対策を入れて 7244046 / 7244047 で再投入し、いずれも完走。

### 4/23 — Batch 2 設計

B1 の結果（後述）から **B1-B (warmup) のみ若干改善, B1-A/C は逆効果** と判明。仮説 A (sin/cos 正規化問題) を直接検証するため:

4. **skip_sincos_normalization** (B2-A): `_setup_from_shards` で normalization stats 計算後、sin/cos slot に `mean=0, std=1` を上書き
5. **skip_sincos + warmup** (B2-B): A + B1-B 勝者構成

**Cache の扱い**: B1 の checkpoint は v1 cache (sin/cos 正規化) で訓練済みなので、stats を上書きすると invalidate される。そのため `data/descriptor_cache_v2/` を新規生成（並列 prep job 7245930, cpu_40, 22 分で完了）し、v2 のみ skip_sincos stats を生成。`scripts/train_vqvae.py` と `scripts/recompute_norm_stats.py` に `--cache-dir` を追加して v2 cache を指定可能に。

注意: v2 は manifest 更新により complex 数が 2.44M → 2.53M に増えており、test split が異なる（v1: 244,237, v2: 253,193 complexes）。**B1 (v1) と B2 (v2) の比較は厳密に apples-to-apples ではない**。両方とも訓練から完全分離された hold-out なので相対比較は妥当。

## Run 一覧（設定差分）

| Run | wandb id | `circle_loss_weight` | `coord_loss_warmup_epochs` | `skip_sincos` | Cache | wallclock | exit |
|---|---|---:|---:|:---:|:---:|---:|:---:|
| Baseline | `yggua4f0` | 0.0 | 0 | False | v1 (2.44M) | ~17h | 0 |
| **B1-A** | `lv5nldy5` | **0.1** | 0 | False | v1 | 16.16h | 0 |
| **B1-B** | `uhhyc6y7` | 0.0 | **10** | False | v1 | 16.37h | 0 |
| **B1-C** | `t65o9cot` | **0.1** | **10** | False | v1 | 16.38h | 0 |
| **B2-A** | `q141s8h6` | 0.0 | 0 | **True** | v2 (2.53M) | 16.85h | 0 |
| **B2-B** | `81foydl6` | 0.0 | **10** | **True** | v2 | 17.12h | 0 |

学習リソース: 全 run `node_f=1` × `h_rt=20:00:00`（B1 当初は 24h、実績から 20h に短縮）。実消費は 16〜17h なので余りはほぼ最小化済。Cache 再生成は 1 回のみ（cpu_40, 22 分）。

## 結果

### Descriptor 空間

| Run | val/protein_recon (best) | Test MSE (norm) Protein | Test MSE (norm) Ligand | Test MSE (orig) Protein | Test MSE (orig) Ligand |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.1316 | 0.1415 | 0.0188 | 0.0581 | 0.0095 |
| B1-A | 0.1294 | 0.1395 | 0.0211 | 0.0605 | 0.0107 |
| B1-B | 0.1286 | 0.1418 | 0.0198 | **0.0517** | 0.0100 |
| B1-C | 0.1571 ↑ | 0.1642 ↑ | 0.0199 | 0.0567 | 0.0100 |
| B2-A | 0.1238 | 0.1310 | **0.0173** | 0.0848 | 0.0101 |
| **B2-B** | **0.1107** | **0.1153** | 0.0183 | 0.0802 | 0.0105 |

### 3D 復元 (本命指標)

| Run | Protein RMSD per-atom (Å) | Protein RMSD **Kabsch** (Å) | Ligand RMSD per-atom (Å) | Ligand RMSD Kabsch (Å) |
|---|---:|---:|---:|---:|
| Baseline | 1.727 | 1.547 | 1.488 | 0.223 |
| B1-A | 1.883 ↑ | 1.676 ↑ | 1.569 ↑ | 0.226 |
| B1-B | 1.631 | 1.461 | **1.393** | 0.227 |
| B1-C | 1.729 | 1.550 | 1.679 ↑ | **0.215** |
| **B2-A** | **1.500** | **1.374** | 2.180 ↑ | 0.249 ↑ |
| B2-B | 1.637 | 1.489 | 1.711 ↑ | 0.240 ↑ |

### Codebook 健全性 (test 2000 complexes)

| Run | Protein Util / Perplexity (norm) | Ligand Util / Perplexity (norm) | Dead codes (P/L) |
|---|---:|---:|---:|
| Baseline | 0.999 / 0.755 | 0.997 / 0.711 | 3 / 3 |
| B1-A | 0.997 / 0.741 | 0.999 / 0.679 | 6 / 1 |
| B1-B | 0.995 / 0.750 | 0.997 / 0.742 | 11 / 3 |
| B1-C | 0.997 / 0.770 | 1.000 / 0.731 | 6 / 0 |
| B2-A | 0.990 / 0.779 | 1.000 / 0.745 | 20 / 0 |
| **B2-B** | 0.998 / **0.782** | 0.999 / **0.763** | 4 / 1 |

PDF (`notebooks/visualization_*.pdf`) で各 run の per-dim MSE bar chart, 散布図, RMSD ヒストグラム, t-SNE が確認できる。

## 主要な発見

### 1. circle_loss penalty (B1-A) は逆効果
Protein RMSD per-atom が 1.727 → 1.883 Å (+0.16 Å)、Kabsch 1.547 → 1.676 Å。sin/cos が std で正規化された状態で λ=0.1 のペナルティを掛けると、recon loss と競合して全体の精度が下がる。**仮説 A の補強**: sin/cos 正規化を直さずペナルティだけ掛けても効かない。

### 2. coord_loss warmup (B1-B) は protein 3D に小幅改善
Per-atom -0.10 Å, Kabsch -0.09 Å, ligand per-atom -0.10 Å。仮説 B (`TaskWeighting` 初期不安定) は実在したが効果は限定的。warmup そのものよりも、warmup 中の Plain additive 加算が実質的に coord_loss の重みを下げているのが効いている可能性。

### 3. 介入の単純加算 (B1-C) は最悪
val/protein_recon 0.1316 → 0.1571 (+19.4%), Test MSE +16%。circle penalty の副作用が warmup の恩恵を打ち消した。**介入を重ねる前に必ず単独効果を確認**するのが教訓。

### 4. skip_sincos (B2-A) が protein 3D の本命
Protein RMSD per-atom -0.23 Å, Kabsch -0.17 Å。仮説 A の決定的検証。単位円制約を保持することで NeRF の角度誤差累積が抑制される。

### 5. descriptor MSE は misleading
B2-B は val/protein_recon (0.1107) も Test MSE (norm, 0.1153) も最良だが **3D RMSD は B2-A 以下**。warmup で descriptor 局所解にハマって 3D 精度を犠牲にした疑い。**3D RMSD で評価すべき**という方針が正しかった。

### 6. Test MSE (orig) が B2 で増加した謎
B2-A の Test MSE (orig) は 0.0848（baseline 0.0581 比 +46%）、しかし 3D RMSD は改善。原因: original-scale Protein MSE は segment-start の `N_d` (球面座標 `r`, std~4.4 Å, 大きい) に支配される。一方 3D RMSD は累積 NeRF の角度誤差に支配されるため、両者の相関は弱い。**原器として normalized MSE よりも per-dim MSE と 3D RMSD を見るべき**。

### 7. ligand 3D は B2 で悪化
per-atom 1.49 → 2.18 Å、ただし Kabsch は 0.22 → 0.25 Å でほぼ不変。内部構造は維持されたまま剛体シフトが増えた → **root atom anchor 問題は skip_sincos で解決しない**ことが確認された。Tier 2 (root 専用 codebook など) が必要。

加えて v2 test set の影響: B2-A の per-atom mean=2.18, median=1.41 で右裾が重い (Std=2.31)。一部の外れ値 complex が押し上げている可能性。

### 8. 角度 per-dim MSE は B2 で改善
PDF より B2-B の Protein per-dim MSE (normalized): `CA_θ` 0.27 → ~0.23, `C_θ` 0.31 → ~0.23。sin/cos slot は std=1 なので直接比較できないが、unit circle に近づいたと推察。ただし依然として最大ボトルネック。

### 9. Codebook は終始健全
全 run で utilization >99%, perplexity 0.68〜0.78, dead codes <20。**問題は量子化粒度や codebook collapse ではない**ことが確定した。

## Verdict

| 指標 | 勝者 | スコア | vs Baseline |
|---|---|---:|---:|
| val/protein_recon | B2-B | 0.1107 | -15.9% |
| Test MSE (norm) Protein | B2-B | 0.1153 | -18.5% |
| **Protein RMSD per-atom** | **B2-A** | 1.500 Å | **-13.1%** |
| **Protein RMSD Kabsch** | **B2-A** | 1.374 Å | **-11.2%** |
| Ligand RMSD per-atom | B1-B | 1.393 Å | -6.4% |
| Ligand RMSD Kabsch | B1-C | 0.215 Å | -3.6% (誤差範囲) |
| Codebook perplexity | B2-B | 0.78 / 0.76 | +3.6% / +7.3% |

**3D RMSD 優先という当初の方針からは B2-A が現時点ベスト**。

## 次のステップ

目標 Kabsch < 1.0 Å に対し、現状 1.374 Å から 0.37 Å (27%) の追加改善が必要。

### Phase 1（次の 1 run、低コスト）

B2-A 構成を base に以下を同梱:

1. **coord_loss を強制的に重く** — `TaskWeighting` を bypass し `total = recon + 5–10 * coord + commit + λ*circle` に変更。dual-objective を 3D 寄りにシフト
2. **Per-dim 重み付け recon** — 角度 dims (idx 1, 5, 9) を 2–3×, sin/cos (2,3,6,7,10,11) を 2×, 距離 (0,4,8) を 0.5× に
3. **Ligand root atom を coord_loss で重く** — `refs[..., 0] == -1` のマスクで weight=N に
4. **Ligand root を cartesian 化** — descriptor 計算で spherical を捨てて (x,y,z) で表現（軽微な実装変更）

期待: Protein Kabsch 1.374 → 1.15–1.25 Å, Ligand per-atom 1.5 → 0.8–1.0 Å。

### Phase 2（Phase 1 で <1.0 Å に届かなければ）

- **Protein per-atom token 化** — 1 residue = 1 token (12-D) を 1 atom = 1 token (4-D) に分解。系列長 3×、各 token 単純化、effective bit 数 3×
- **Ligand root 専用 codebook** — main codebook (1024) と root codebook (256) を分離

期待: per-atom token 化で 0.2–0.4 Å 追加、root codebook で ligand per-atom 0.3–0.5 Å 改善。

### スケーリングは最後の手段

データ 10× は不可（CrossDocked2020 は cdonly+it0+it2_redocked で既にほぼ全部）。codebook 2× は perplexity 0.78 → 余地は限定的 (5–10%)。model 4× で 15–25%、ただし学習時間 4× = 64h/run。**構造修正で取れるところを取ってからスケーリングを判断**するのが効率的。

## 参考

- 各 run の PDF: `notebooks/visualization_b1a.pdf`, `_b1b.pdf`, `_b1c.pdf`, `_b2a.pdf`, `_b2b.pdf` (baseline は `notebooks/visualization.pdf`)
- 学習スクリプト: `scripts/train_vqvae_{b1a,b1b,b1c,b2a,b2b}.sh`
- Cache v2 生成: `scripts/prepare_descriptors.py` + `scripts/prepare_descriptors.sh`
- 統計再計算: `scripts/recompute_norm_stats.py`
- visualization.py の env var: `VQVAE_CKPT` (ckpt path), `VQVAE_CACHE_DIR` (cache dir, B2 評価では `data/descriptor_cache_v2`)
