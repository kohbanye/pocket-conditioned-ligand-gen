# VQ-VAE Training Loss Divergence — 2026-04-07

WandB run: https://wandb.ai/kohbanye/pocket-ligand-vqvae/runs/yv2afg4k

## 症状

- `train/total_loss` が epoch 30 付近から発散（最大 4e+10）
- `train/protein_recon` が 5M → 20M+ に増大
- `train/protein_commit` が 8e+10 に爆発
- `train/protein_codebook_util` が 0.8 → 0.4 に低下（codebook collapse）
- `train/protein_perplexity` が 400 → 200 に低下
- **Ligand 側は安定**（recon ≈ 0.5–1.5, perplexity ≈ 125–175）

4 run（trill-directive-4, nemesis-quark-3, dax-unimatrix-2, flowing-smoke-1）で再現。

## 原因: Encoder-Codebook Divergence

EMA-based VQ-VAE の既知の不安定性パターン。

### 発生メカニズム

1. Protein descriptor は kNN 距離（3〜50+ Å）を含み、ligand descriptor と比べて値のスケールが大きい
2. Encoder 出力 `z` に正規化がなく、magnitude が際限なく増大可能
3. Codebook は EMA（decay=0.99）で更新されるため、encoder の急速な変化に追従が遅れる
4. `z` と codebook vector の距離が開く → commitment loss が増大 → 勾配が大きくなる → encoder がさらに大きく動く（正のフィードバックループ）
5. Codebook が encoder に追いつけなくなると、一部の code しか使われなくなる（codebook collapse）

Ligand 側は descriptor の値域が小さい（角度は [0, π]、bond length は 1〜2 Å 程度）ため、同じ問題が顕在化しなかった。

### 根本原因の要約

- Encoder 出力の L2 正規化が欠如
- Gradient clipping がなく発散を止める安全弁がない

## 修正内容

### 1. L2 正規化の導入 (`src/tokenizers/codebook.py`)

ViT-VQGAN (Yu+ ICLR 2022) に倣い、encoder 出力と codebook vector を量子化前に L2 正規化する。

- `z` → `z / ||z||` に正規化してから nearest-neighbor lookup
- Codebook vector も正規化してから距離計算
- 正規化により両者が単位超球面上に制約され、距離が有界（最大 2）になる
- Learnable scale parameter (`log_scale`) を追加し、decoder に渡す quantized vector のスケールを学習可能に

### 2. Gradient clipping の追加 (`src/model/vqvae_module.py`)

`training_step` で `torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)` を追加。
万一の勾配爆発に対する安全弁。

## 変更ファイル

- `src/tokenizers/codebook.py` — L2 正規化 + learnable scale
- `src/model/vqvae_module.py` — gradient clipping 追加

## 参考文献

- Yu et al., "Vector-quantized Image Modeling with Improved VQGAN", ICLR 2022
  — L2 正規化による VQ-VAE 学習安定化の手法を提案
- Van Den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
  — 元の VQ-VAE 論文
