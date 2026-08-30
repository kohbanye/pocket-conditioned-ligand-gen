# Generation, 97 CrossDocked targets x 100 molecules (2026-08-19)

Tokenizer `vq_e250_lig3` (atom_coord 0.1021), CLM `clm_e250lig3_fullft` e00,
refiner `refit_e250lig3` e16. Sampling T=1.0, top_p=0.95, seed 0. Evaluation
pockets excluded from tokenizer and corpus training (`data/sbdd_bench_pockets.txt`,
54 of 93 overlapped ProLIT's fold split and were dropped).

| | raw | +relax | +refiner+relax | FLOWR (published) |
|---|---|---|---|---|
| validity | 0.965 | 0.965 | 0.971 | — |
| **PB-valid** | 0.722 | **0.818** | 0.786 | **0.92** |
| clash-free | 0.117 | 0.117 | **0.204** | — |
| **Vina score (median)** | +11.12 | +11.11 | **+3.88** | **−6.29** |
| Vina min (median) | −3.98 | −4.00 | −4.14 | — |
| Vina dock (median) | — | −7.21 | −7.16 | — |
| strain (median) | 302 | 128 | **125** | — |
| bond length W1 | 0.051 | **0.035** | 0.047 | — |
| bond angle W1 | 7.59 | 6.57 | **6.34** | — |
| QED | 0.439 | 0.439 | 0.437 | — |
| scaffold diversity | 0.395 | 0.395 | 0.412 | — |

Earlier reference point: PB-valid was **0.237** before two generation defects were
fixed (every bond written single; local geometry never repaired at decode).

## What these say

**The molecules are good; the poses are broken.** Vina *dock* -- redock the
molecule from scratch, which ignores the generated pose -- is -7.16, competitive.
Vina *score*, which reads the pose as generated, is **positive**: that is the
repulsion term, and it is the entire gap to FLOWR.

**Relaxation does exactly what it claims and nothing more.** It moves PB-valid
0.722 -> 0.818 and strain 302 -> 128, and leaves clash-free (0.1166 -> 0.1169)
and Vina score (11.119 -> 11.109) untouched. That is the designed behaviour --
it repairs local geometry inside a flat-bottomed restraint at the tokenizer's own
coordinate error -- and it is the evidence that it is not quietly improving the
pose.

**The refiner trades PB for pose.** It is the only thing that moves clash-free
(0.117 -> 0.204) and Vina score (+11.1 -> +3.9), and it costs PB-valid
(0.818 -> 0.786) because pushing atoms out of the receptor bends the ligand.

**Vina min only reaches -4.14 while redocking reaches -7.16.** Local
minimisation cannot rescue the generated pose, so this is not "a few atoms
slightly clashing" -- a substantial fraction of poses are in the wrong basin.
Centroid offset from the reference ligand is a median 1.34 A but 3.28 A at p90.

## Two causes identified, both being fixed

1. **The CLM's LR never annealed.** Its cosine was sized for `--max-epochs 3`
   and the job hit its 20 h walltime after ~2, so every molecule here was
   sampled from a mid-schedule, high-LR checkpoint. Re-running one full epoch
   with a completing schedule (`clm_e250lig3_anneal`).
2. **The refiner was trained on the wrong corruption.** Its corpus used
   `resample_frac=0` -- VQ round-trip plus isotropic jitter -- while the LM's
   error is token substitution. Calibrated against the LM's observed clash:

   | resample_frac | clash-free | mean clashes |
   |---|---|---|
   | 0.0 (what it was trained on) | 0.840 | 1.16 |
   | 0.2 | 0.160 | 4.80 |
   | 0.3 | 0.080 | 7.20 |
   | **LM at deployment** | **0.124** | **8.48** |

   Seven times milder than deployment. Corpus rebuilt with `--resample-frac 0.4`
   (graded ladder -> 0.0/0.1/0.2/0.3, bracketing the LM).

Sampling temperature was tested and is *not* a lever: 1.0 -> 0.6 moves clash-free
only 0.269 -> 0.308 and costs diversity (597 -> 537 unique SMILES over 12 targets).

## Fairness

FLOWR's numbers are its published ones and are comparable: it busts its own
predicted bond orders, which is what this pipeline now does too. The in-house
TargetDiff/DiffSBDD numbers are **not** comparable any more -- their molecules
were deleted and their old PB came from the Open Babel re-perception path that
has since been fixed -- so they are omitted here rather than quoted. They need
regenerating before they go in a table; their weights and envs are both gone.

## Update: two causes measured, two fixes landed (same day, later)

### The clash splits in two, and diversity is the smaller half

Generating for pockets the LM **saw in training** (6 CrossDocked train pockets,
identical model/refiner/seed/sample count as the eval run):

| | clash-free | mean clashes |
|---|---|---|
| crystal ligands | 0.850 | 1.50 |
| tokenizer round-trip (encode->decode reference ligands) | 0.800 | 1.45 |
| **LM, pockets it trained on** | **0.427** | 3.51 |
| **LM, unseen eval pockets** | **0.254** | 5.22 |

Memorisation is real (1.7x better on seen pockets) but the *intrinsic* gap
0.427 -> 0.800 is larger than the generalisation gap 0.254 -> 0.427. Pocket
diversity alone cannot close this.

### The LM was trained on docking decoys

`_build_pocket_plans` reads `pair_idx, complex_dir, source_type, cdonly_fold0,
receptor_pdb` -- **never `label`** -- and `--cache-dir` pointed at a prebuilt
`descriptor_cache_atom_full` holding all 28,228,531 poses. So the 16.5M training
docs are 1,565,002 fold0-train rows x ~10.6 poses each, of which:

| | share | median RMSD to native |
|---|---|---|
| label==1 | 19.4% | 1.16 A |
| **label==0 (decoys)** | **80.6%** | **5.39 A** |

`label` is a property of the FILE, and a label==1 `*_docked.sdf.gz` still holds
~20 poses spread a median 4-7 A apart (measured). The `--include-decoys` flag
governs how the descriptor *cache* is built and does nothing on this path.
`--near-native-only` (label==1 & _min, 226,411 rows over 1,433 pockets) was added
to fix it.

### The refiner was trained on the wrong corruption -- fixed, and it works

Its corpus used `resample_frac=0`: VQ round-trip plus isotropic jitter, topping
out at 0.93 A RMSD, while deployment is several times worse. Rebuilt with
`--resample-frac 0.4` (graded ladder 0.66 -> 3.15 A, calibrated so its clash
statistics bracket the LM's). Deployment, same 12 eval targets:

| refiner | clash-free | mean clashes | closest contact |
|---|---|---|---|
| old (`refit_e250lig3` e16, 18 epochs) | 0.247 | 5.19 | 2.00 A |
| **new (matched corruption, e03)** | **0.350** | **2.94** | **2.25 A** |

`val/rmsd_gain` reached +0.419 by epoch 3 against the old refiner's +0.198 peak
at epoch 18, and was still climbing. (Its first validation was -0.168, which
looks like failure; the old run's first two were -0.545 and -0.231. Early
negative gain is warm-up, not divergence.)

### Ruled out

* **Learning rate / schedule.** One full annealed epoch from the 2-epoch
  checkpoint (`clm_e250lig3_anneal`, lr 1e-4, cosine completing inside walltime)
  gave **val 5.7029, worse than the 5.3568 it started from**. More optimisation
  on this corpus hurts. Optimisation was never the limit.
* **Sampling temperature.** 1.0 -> 0.6 moves clash-free only 0.269 -> 0.308 and
  costs diversity (597 -> 537 unique SMILES over 12 targets).
* **Constrained decoding against the receptor.** A code has no fixed position:
  the decoder is a transformer, and the same code lands 1.383 A from its own
  mean depending on context, against a 1.456 A median spacing between distinct
  codes. See `docs/notes/2026-08-19_no_clash_masking_at_decode.md`.

## 確定版: 97/97 標的での完全ベースライン (2026-08-20)

それまでの表は **74 標的でしか計算されていなかった**。altLoc を持つ受容体で
`prepare_receptor` が壊れた pdbqt を出し、Vina が受容体ごと拒否していたため
（`prepare_targets.py` の `_drop_altlocs` で恒久修正済み）。exit status は 0 の
まま、欠けた Vina 列だけが症状だった。修正後、参照リガンド込みで 97/97 が復旧。
既存 74 標的のスコアは**最大差 0.0000 で不変**（修正が壊れた標的だけを直した証拠）、
復旧 23 標的の Vina 中央値 +2.51 は全体 +2.70 と同等で、難易度に偏りはなかった。

構成: CLM `clm_e250lig3_fullft` e00 + refiner `refit_e250lig3` e16 + 局所緩和。

| 指標 | ProLIT (97/97) | FLOWR |
|---|---|---|
| validity | 0.971 | — |
| **PB-valid** | **0.786** | **0.92** |
| clash-free | 0.204 | — |
| **Vina score (median)** | **+3.36** | **−6.29** |
| Vina min (median) | −4.00 | — |
| Vina dock (median) | −7.00 | — |
| strain (median) | 124.6 | — |
| bond length / angle W1 | 0.047 / 6.34 | — |
| QED / SA | 0.437 / 4.31 | — |
| scaffold diversity | 0.412 | — |

**残る唯一の障害は Vina で、Vina は衝突そのもの。** 剛体変位の較正がそれを
定量化した: 結晶リガンドを剛体的に 2.0 A ずらすと clash-free 0.170 / 平均衝突
8.24 となり、LM 実測の 0.103-0.117 / 7.3-8.5 とほぼ一致する。つまり LM は
**化学的にまともな分子を約 2 A ずれた場所に置いている**。PB がそこそこ良く、
ポーズを保つ設計の緩和が効かず、Vina min が -4.00 止まりで再ドッキングが -7.00
に届くのは全てこれで説明がつく。

### 近傍コーパスの入れ替えは val loss を大きく下げたが生成は改善しなかった

decoy 80% を near-native + 結晶錯体に入れ替えた `clm_e250lig3_clean` は
val 5.3568 -> 3.7281（perplexity 212 -> 41.6）。だが 12 標的プローブでは
PB 0.734 (旧 CLM 0.858)、refiner なし clash-free 0.103 (旧 0.117) で、
**衝突は改善しなかった**。BioLiP/PLINDER の結晶錯体は化学的に多様で難しく、
ユニーク SMILES は 706 -> 1081 に増えている。val loss の改善はその分布に
当たるようになったことを示すだけで、ポーズ配置の精度とは別物だった。

## 最終確定値と、FLOWR に届かなかった理由 (2026-08-20)

| 指標 | ProLIT (97/97) | FLOWR | 差 |
|---|---|---|---|
| validity | 0.971 | — | — |
| **PB-valid** | **0.795** | **0.92** | **-0.125** |
| clash-free | 0.204 | — | — |
| **Vina score (median)** | **+3.36** | **-6.29** | **~10 kcal/mol** |
| Vina min (median) | -4.00 | — | — |
| Vina dock (median) | -7.00 | — | — |
| QED / SA | 0.437 / 4.31 | — | — |

局所緩和に UFF フォールバックを足して MMFF が型付けできない分子も直すように
した (緩和成功 78% -> 86%)。16 標的サンプルでは PB 0.739 -> 0.777 だったが、
**97 標的全体では 0.7863 -> 0.7948 (+0.0085) にとどまった**。サンプルからの
外挿 (0.82 見込み) は楽観的すぎた。

### Vina が届かない理由は「後処理では直せない」ことが確定した

Vina score が正なのは斥力項そのもの、つまり衝突。衝突の正体は剛体較正で
**LM が化学的にまともな分子を約 2 A ずれた場所に置いている**ことと判明した
(結晶リガンドを 2.0 A 剛体変位させると clash-free 0.170 / 平均衝突 8.24 で、
LM 実測 0.103-0.117 / 7.3-8.5 と一致)。これを後処理で直す手を 4 通り試し、
すべて失敗した:

| 手法 | 結果 |
|---|---|
| 局所緩和 (ポーズ保存) | clash-free 0.117 -> 0.117。設計通り効かない |
| pocket-aware 緩和 | PB 0.932 -> 0.851、clash-free は 0.145 のまま |
| refiner: jitter 腐敗 (既存) | 腐敗上限 0.93 A。deployment の半分で補正不足 |
| refiner: code resample 腐敗 | 局所幾何を壊す。PB 0.858 -> 0.463 |
| refiner: 剛体腐敗 (実効 2.11 A) | **推論が発散** (train/loss 0.419 に対し val 2.76e7、RMSD 829 A) |
| 剛体アライメント | box 中心 0.188 / ポケット重心 0.019。**そのまま (0.272) より悪い** |

決定的なのは **Vina min (局所最適化) ですら -4.00 止まりで、再ドッキングの
-7.00 に届かない**こと。2 A の誤配置は局所的手法では原理的に直せない。refiner
も局所手法なので同じ壁に当たる。3 回の refiner 失敗は手法選択自体の誤りだった。

### 届かせるなら変えるべきもの

後処理ではなく **LM のポーズ配置精度そのもの**。根拠は、トークナイザの
round-trip は clash-free 0.800 (結晶 0.850) を達成しており、表現能力は足りて
いる。足りないのは LM が自己整合的なトークン列を出す能力で、デコーダが文脈
依存 (同じコードが文脈で 1.383 A 動く) なため誤差が累積する。

近傍コーパス化 (decoy 80% を排除、val 5.357 -> 3.728) は val を劇的に改善した
が衝突は 0.117 -> 0.103 で不変だった。つまりデータではなく、
**離散トークン列から座標を復元する経路そのもの**が律速。

## 剛体立体緩和を入れた 97 標的の確定値 (2026-08-21)

| 指標 | 基準 (relax) | +rigid+settle | FLOWR |
|---|---|---|---|
| validity | 0.971 | 0.971 | — |
| PB-valid | 0.789 | **0.797** | 0.92 |
| clash-free | 0.204 | **0.876** | — |
| **Vina score (median)** | **+3.36** | **-1.03** | **-6.29** |
| Vina min (median) | -4.00 | -4.22 | — |
| Vina dock (median) | -7.00 | -7.00 | — |
| QED / SA | 0.438 / 4.295 | 0.438 / 4.294 | — |

9,700 分子・97 標的の対比。分子そのものは変わっていない (QED/SA が一致) ので、
差はポーズだけ。**Vina score が正から負に転じた**。

手法は `prolit/chem/rigid_fit.py`: 受容体しか使わず、スコア関数もリガンド参照も
使わない 2 段階の剛体変換。段 1 で vdW 重なりを最小化し、段 2 でその水準を上限に
固定したまま Lennard-Jones の谷に落とす。調整する重みは無く、境界は 2 つ
(並進 2.5 A / 回転 30 度) で、どちらも測定から決めた。

**公平性**: FLOWR の公表値は後処理なしなので、表では raw / +relax /
+relax+rigid を別行として並べること。片方だけを載せて比べない。

導出と否定された仮説は `docs/notes/2026-08-21_pose_error_is_rigid.md`、
記述子側の 2 案が両方悪くなった件は
`docs/notes/2026-08-21_descriptor_ablations_negative.md`。
