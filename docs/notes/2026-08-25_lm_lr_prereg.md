# 事前登録: fullft 段を低い学習率でやり直す (2026-08-25)

## 根拠

5 kcal/mol のうち **3.81 が LM のコード選択**
(`2026-08-25_where_the_kcal_go.md`)。トークナイザは 1.28 しか奪っておらず、
VQ 往復 (-5.60) は FLOWR (-5.73) より上。**全部 LM にある。**

配備 LM (`clm_e250lig3_fullft`) の val loss:

| epoch | val loss |
|---|---|
| 0 | **5.3568** |
| 1 | 5.4648 (悪化 -> early stop) |

**4.1B トークンのコーパスに 298M のモデルが 1 epoch で過学習することは
考えにくい。**lr 3e-4 が高すぎたとみるのが自然。事前学習からの finetune で
3e-4 は大きい。

## 腕

| | init | lr | epochs | 資源 |
|---|---|---|---|---|
| `clm_e250lig3_fullft` (配備、対照) | `pretrain/last` | 3e-4 | 2 (early stop) | node_f x4 |
| **`clm_fullft_lr1e-4`** | 同じ | **1e-4** | 最大 3 | node_f x4 |

コーパス・batch・seed は同じ。**変えるのは学習率だけ。**

## 判定規則

**主要**: `refiner_on_known_error.py --source clm` の
**teacher-forced argmax の RMSD**、100 標的、配備 LM との対応のある差、
Wilcoxon p < 0.05。

**副次** (必ず報告): 生の Vina score / min、val loss の曲線
(epoch 1 でも下がり続けたか)。

**val loss だけで判定しない。**同じコーパス・同じトークン列なので比較可能だが、
`nnft` / `clean` のときに val loss とポーズ誤差が逆を向いた前例がある。

## 予測

- val loss は epoch 1 でも**下がる** (今回の仮説が正しければ)。
- teacher-forced RMSD は 1.902 から **1.7-1.8 台**に下がる。
- **1.5 A を切ることはない**と予測する。学習率だけで 5 倍の精度は出ない。
- Vina score (生) は -1.78 から **-2.5 前後**。天井 -5.60 には届かない。

**予測が当たっても FLOWR には届かない。**当たれば「LM 側にまだ余地がある」
ことの証拠になり、外れれば「学習率ではない」と分かる。どちらでも次が決まる。
