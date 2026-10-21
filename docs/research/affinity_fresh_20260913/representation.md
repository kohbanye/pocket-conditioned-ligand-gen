# Tokenizer / LM の独立コード監査（2026-09-13）

過去の結果 Markdown は読まず、実装・設定定義・CASF SDF を調査した。実装中のコメントに過去の評価が埋め込まれているものは目に入るが、本提案の採否の根拠には使っていない。以下では観測事実と改善仮説を区別する。学習ジョブや本番コードの変更はしていない。

## 最優先の実測：chemistry source の不整合と formal charge の情報損失

`src/prolit/tokenizers/ligand.py:125–147` の SDF パーサは xyz と元素、結合次数だけを保存し、原子行の charge、M CHG、stereo 等を保存しない。`:235–258` は元素だけから RDKit Atom を作り直す。`parse_ligand_pdb_text` も CCD テンプレートで構築した mol を最終的に元素・座標・結合次数だけへ縮約する（`:210–220`）。したがって charge を持つ descriptor スロットがあっても、元入力の電荷を運べない。サニタイズの結果として一部の電荷が生成される可能性はあり、「すべて必ずゼロ」とまでは言えない。

実際の CASF-2016 crystal SDF 全 285 件を current parser と `_build_rdkit_mol` の heavy-atom 経路に通した。

|検査|実測|
|---|---:|
|原 SDF に charged atom を持つ複合体|150 / 285|
|原 SDF の charged atoms|212|
|再構築 mol の charged atoms|14|
|formal charge が一致しない原子|226|
|formal charge が一致しない複合体|156 / 285 = 54.7%|
|net charge がゼロだが charged atom を持つ複合体|15|
|heavy-atom 数の不一致|0|

再現：`.venv/bin/python docs/research/affinity_fresh_20260913/audit_chemistry.py`。明細は `chemistry_audit.csv`、集計は `chemistry_audit_summary.json`。原 SDF を sanitize=False で読むことでファイル中の charge を比較基準にし、heavy atom 順序は current parser と一致している。差分原子数 226 が原 charged atoms 212 を上回るのは再構築後に別の電荷が導入されるため。

**追検証で重要な留保が見つかった。原 CASF SDF 自体が 92 / 285 件で RDKit sanitization に失敗する。したがって raw SDF の charge を盲目的にコピーする修正は推奨しない。** 例えば 1a30 SDF は valence 不整合を持つ C- を含む。一方、SDF が sanitize できた 193 件に限定しても charge mismatch は 75 件存在する。全 285 件の対応 original MOL2 は sanitize に成功し、heavy atom の元素と座標が SDF と同順に対応する 282 件では current SDF rebuild と電荷が異なるのが 167 複合体 / 257 原子だった（3 件は atom mapping を追加検証するまで比較から除外）。MOL2 の partial charge と formal charge は別なので、ここでは RDKit が MOL2 の atom/bond types から構成した formal charge を比較している。MOL2 が assay 条件下の protonation の正解である保証もない。

これは修正優先度の高い**chemistry ingestion の不整合と確定した情報損失**である。ただし affinity 誤差の因果原因とまではまだ証明していない。最初の実験は SDF/MOL2/CCD の同一分子・原子対応、valence、bond order、charge の正規化監査で、全モデルに同じ chemically valid input を渡すこと。電荷メタデータの保存、pH に依存する化学状態推定、化学状態の multi-view 学習は別々に比較すべき。正規化ルールを train/validation/test 共通に適用する。古い token cache と checkpoint に新しい chemistry を混ぜず、cache fingerprint を更新して tokenizer→LM→head を一貫して作り直す。

Stereo tag の欠落もあるが、3D 座標は残るので「立体化学を完全に失った」は不正確。立体中心周辺の quantization による向きの反転率、chiral volume の符号保存率を別に測定する。

## 構造上の制約と監査すべき契約

1. **Affinity backbone は既に bidirectional MLM。causal attention が原因ではない。** `src/prolit/model/mlm.py:120–130` の SDPA に causal 指定はなく、`:269–285` は padding bias のみ。CLM と混同して bidirectional 化を改善案に挙げる必要はない。
2. **共有 codebook と複合体 joint encoding は異なる。** `PoseEncoder.setup_pocket` は protein descriptor のみを encode し（`pose_encoder.py:97–101`）、ligand も独立に encode（`:115–120`）。共通座標系と同じ量子化器を使うが、tokenizer encoder 内で ligand–protein cross-attention は行わない。現在の表現が interface を明示的に保存する保証はない。
3. **Pocket-context/atom-order オプションの経路不一致。** `PoseEncoder.__init__` は `LigandAtomDescriptor()`（`:67`）を初期化し `pocket_cfg.atom_order` を渡さない。`ligand_seq`（`:115–117`）、`ligand_descs`（`:161`）は pocket_canonical / pocket_elements を渡さない。一方 descriptor corpus 側は `atom_descriptors.py:146–156` に pocket_context 経路がある。default 設定がすべて False/file なら現在の既存 arm のバグではないが、将来これを切り替えるだけでは train/inference が一致しない。
4. **実際の KNN は「空スロット充填」ではない。** `_knn_offsets_and_elements`（`ligand.py:401–404`）は context 使用時に 4 slots を self=2/context=2 に分割する。self KNN は結合グラフではなく全 ligand 原子間距離。terminal atom でも ligand 全体が 5 原子以上なら通常 self slots は埋まる。コメントから「terminal の空いた近傍を埋める」効果だと説明しない。比較すべきは intramolecular second shell を cross-interface へ置換するトレードオフ。
5. **Metals/waters/cofactors を表現しない。** `protein.py:140–141` は AA_3TO1 にあり CA を持つ残基だけを保持する。金属配位、water-mediated interactions、cofactor stacking 等は周囲 protein/ligand の間接的情報しかない。入力で欠けたものは複雑な head でも復元できない。ただし新たに cofactor/water 入力を与える場合、他方式にも同じ available information を与える比較が必要。
6. **Frame の連続安定性と回転不変性は別。** `protein.py:258–296` は pocket CA の PCA + 最大投影点による符号決定。理想的な非縮退 PCA は rigid rotation に対応するが、固有値近接、最大投影点交代、pocket membership 変化に敏感になり得る。canonical frame 固定後に追加回転する評価は token の frame 感度を測る。raw complex を回転して canonical frame を再計算する評価は pipeline の rigid invariance を測る。両者を区別する。
7. **Token order に意味が入る。** tokenizer encoder/decoder は sinusoidal index を足す（`vqvae.py:364,391,447`）。LM は通常の token-index RoPE（`mlm.py:74–79,121`）。同じ原子集合でも SDF 順序や残基順序に依存し得る。正しい距離関係を token の線形位置から推測する負担がある。ただし RoPE を外すだけで改善するとは未検証。
8. **Training は prefix truncation、inference は全長。** `rescore_dataset.py:126` と `mlm_dataset.py:124` は block_size まで先頭切断。`affinity_bench/inference.py:294–303` は全長を渡す。長い pocket が ligand を押し出す件数を train/val 別に集計し、ligand_mask.sum==0 は fail-fast にする。実際に何件存在するか未測定なので、現状スコア低下の断定は不可。
9. **Config にある head/aux オプションが実装されているとは限らない。** `RescoreTrainingConfig` に pooling、head_interaction_layers、mlm_aux_weight、label_divide_by_size 等があるが、現行 `ComplexRescoreModule` は mean と固定 MLP（`:51–57,108–109`）、smooth L1 と optional listwise のみ（`:173–193`）。`_predict` の size 再乗算もない。efficiency 使用時は dataset が label を原子数で割る（`rescore_dataset.py:130–131`）のに benchmark は raw head を pK と扱う（`inference.py:285–289`）。これらを既存主 arm の問題と断定せず、使用した checkpoint config を確認する。
10. **Affinity の default clipping には注意。** head は pose 用実装を共有し、`rmsd_cap=8.0` が default、`rescore_module.py:176` で max clip。affinity CLI は `--label-cap` を持つ（`scoring_head.py:182–183`）。checkpoint ごとに cap が pK 最大値を超えることを確認する。デフォルトの存在だけで実際の affinity run が clip されたとは言えない。

## 何を学ばせるか：head を固定した優先実験

以下は**予測される改善の仮説**。既存失敗履歴に従って捨てたり、成功とみなしたりしない。各段階で encoder frozen linear probe、frozen existing MLP、同じ fine-tuning MLP を置く。主要比較は同じ MLP のまま行い、モデル主張を tokenizer/LM に帰属させる。

### P0：入力 chemistry と execution contract

まず上記 source normalization と charge retention を全データへ適用。sanitize に失敗する raw CASF SDF charge をそのまま採用しない。タンパク質 feature fallback の率、ligand bond-template 成功率、high-charge/metal/salt subset を測る。SDF から直接 RDKit mol を保持する方法が最小情報損失だが、BioLiP PDB 経路と統一するため atom-aligned charge 等の metadata schema を明示する。token ids が新旧で変わるので既存 weights をそのまま評価するのは入力分布シフト試験にしかならない。報告は parser-only decoder chemistry fidelity と、retrained downstream affinity の二段階。

### P1：interface を量子化後にも読めるようにする

目標を atom RMSD だけにせず、**intermolecular distance の局所誤差と接触タイプ保存**に置く。現行 VQ reconstruction は原子 xyz MSE（`vqvae.py:749–757`）が中心で、ligand 単独 sample の pair loss は相手 protein を見ない（`:816–819`）。同じ RMSD でも水素結合/塩橋/π geometry の破壊は異なる。

最小変更は paired protein/ligand sample を共有 frame で encode した後、両方の decoded xyz から cross-distance loss を計算すること。head は不要。全遠距離 pair を平均すると interface が薄まるため、距離帯ごとに正規化し近接 pair を均等に扱う。距離 loss のみでは donor/acceptor と方向を保証しないので、charge/numH/aromatic と角度 fidelity を併記する。optional pocket_context=2 slots は distinct ablation にする。

評価：held-out reconstruction の interface distance MAE、contact precision/recall、salt-bridge charge correctness、chiral volume preservation、linear-probe contact predictability、同じ affinity head の cluster-heldout R。decoded RMSD の改善だけでは採用しない。

### P1：MLM が pocket を使わなければ解けない事前学習

現在 MLM は 15% independent token mask と通常 CE（`mlm_dataset.py:131–172`, `mlm.py:305–310`）。protein tokens が ligand より多い場合、全 token 平均では protein 内補完が目的の大半を占める。ligand の同一部分の近傍からも復元できるので、低 MLM loss は interaction 学習の証拠ではない。

まず mask を (a)従来、(b)protein/ligand 別 loss 正規化、(c)ligand の連続 chemical fragment / spatial neighborhood block mask の固定予算比較にする。mask 対象数だけでなく tokens processed を一致させる。さらに ligand mask 状態で pocket を交換/除去した時の復元 loss 差を validation で測る。protein token の偶然の順序まで変えると副作用が混ざるので token 長・組成を合わせた交換対照を使う。

次に token LM hidden state に training-only の小さな distance-bin/typed-contact auxiliary prediction を置き、推論時には捨てる。これは prediction head の複雑化ではなく表現事前学習で、最終 readout は同じ。幾何 labels は train structures だけから生成し test structures の自己教師学習混入も除外する。native vs decoy は pose plausibility の補助に使えるが、decoy に「低 affinity」の偽ラベルは与えない。

### P2：invariance を tokenizer/LM の訓練へ

1) canonical-frame perturbation に対する token/contextual embedding stability を測定、2)同じ complex の global rotations を paired views とした pooled representation consistency、3)atom-order augmentation を tokenizer と LM の両方へ適用、の順。Consistency は同一複合体のみを正例にし、似た別 ligand を強制一致させない。誤差の pK 標準偏差、rank flip、atom identity fidelity を出す。

大きい変更として、chemistry と position を一つの code に押し込む代わりに factorized/residual quantization がある。chemistry-stable code と residual geometry code を分離すると joint token の圧縮率主張に影響するため、token count/bitrate/LM FLOPs を揃えて比較する。これを初手にするより、既存一 token/atom のまま cross-interface preservation が改善するかを先に確かめる。

### P2：LM の adaptation を小データに合わせる

同じ 0.59M 程度の MLP のまま、full fine-tune vs last k layers vs LoRA、backbone/head の別 learning rate、事前学習重みからの逸脱抑制を小さい preregistered grid で比較する。これは追加 head interaction layers より帰属が明快。validation を protein/ligand cluster ごとに分け、全体相関だけでなく target 内差分を監視。rotation views は独立 label 数を増やさないので、effective unique complexes と optimizer steps を必ず併記する。

### P3：head の変更は固定数個に限定

現在の ligand mean は相互作用原子の情報を希釈し得るが、mean にすればサイズ情報が消えるわけではない（contextual states 自体が token 数・位置に依存）。比較するなら (a)現行 mean MLP、(b)linear readout、(c)mean に log heavy-atom count という一つの scalar を足す、程度。count を加えると size shortcut が強くなるリスクもあり、サイズで matched した validation を同時に見る。meanmax/pairwise interaction の大規模探索は論文の主因を不鮮明にするため後回し。

Loss は smooth L1、MSE、少量 target内 pairwise difference loss を固定して比較。ListNet は高い pK へ確率質量を寄せるので全ペア順位や pooled Pearson の直接目的とは違う。target 内差分のラベル uncertainty が大きいペアは除外/重み低下し、Kd/Ki/IC50 混在を扱う前に endpoint を揃える。損失を変えた run 間では val/loss の数値を横断比較せず、共通 validation metric で選ぶ。

## 反証できる診断設計

- Representation ceiling：charge/contact/atom identity を input descriptors、quantized tokens、LM states の各段階から同程度の小 probe で復元。初めから消えるなら parser、量子化で消えるなら tokenizer、token では読めるが LM で不安定なら pretraining/readout が標的。
- Geometry dependence：同じ ligand を同じ pocket で少量並進/回転したときの score 変化は感度試験。crystal 周囲の微小変動の安定性と大きい破綻への感度を別々に測る。native の全摂動に同じ pK を教師として与えると pose を無視するのでしない。
- Target use：protein masking/swapping の予測変化。これは非物理的入力であり絶対 accuracy の主評価ではない。タンパク質長・ligand size・target frequency matched controls を使う。
- Chemistry matched pairs：同じ target、近縁 ligand、charge/hetero atom/burial が変わる pair の ΔpK。学習ラベルと assay metadata の信頼性を先に確認。
- Test error inspection は仮説生成として明示し、CASF の同じ誤差集合で hyperparameters を最適化しない。上記 P0 のような source correctness の修正も、下流性能の採否は新しい untouched holdout で評価する。

## 不確実性

コードの default は実際の checkpoint hyperparameter を保証しない。charge loss は全 CASF 入力で実測したが affinity への改善量を予測する根拠はまだない。metals/water の追加、interface loss、MLM auxiliary の順位は機序に基づく優先順位であり、GenScore 超えの保証ではない。主論文では「同じ supervised data・same head・same LM budget で tokenizer を改善するとどの化学/構造情報が保持され affinity が改善するか」を因果鎖として示すのが最も説明しやすい。

## 追補：実 checkpoint と token corpus による切り分け

コード上の default と実運用を混同しないため、seed 7 の `e250_rot1` と `e250_union_clean` の lowest-val-loss checkpoint を `torch.load(map_location='cpu', mmap=True)` で読み、train/val の全 `.len`、ligand markers、`.rmsd` を検査した。再現スクリプトは `audit_execution_contract.py`、詳細は `execution_contract_audit.json`。

|項目|e250_rot1|e250_union_clean|
|---|---:|---:|
|selected checkpoint|rescore-e04-vl0.8378|rescore-e04-vl0.8094|
|block_size|512|640|
|label cap|13|13|
|train documents|8,150|15,509|
|train length median / p95 / max|231.5 / 365 / 509|241 / 376 / 526|
|val documents|431|431|
|val max length|412|412|
|train/val block 超過件数|0 / 0|0 / 0|
|切断で ligand が全欠落/一部欠落|0 / 0|0 / 0|
|train label maximum|13|15|
|train cap 超過件数|0|4|
|val cap 超過件数|0|0|

**従って現在のこの 2 arm では sequence truncation はスコア低下の原因ではない。** zero-ligand fallback も発生していない。`rmsd_cap=8` の default が使われているという疑いも否定された。ただし union_clean では 15,509 件中 4 件、zero-based document indices 10471 / 10530 / 11614 / 12546 の pK 15 / 13.96 / 15 / 13.82 が実際に 13 へ clipped される。規模は 0.026% と小さく、全体相関差を説明する主要原因とは考えにくい。変更する場合は当該ラベルの provenance/endpoint を確認し、スコア改善のため恣意的に残す・除くのではなく一般的な clipping 方針を決める。

両 checkpoint は mean pooling、full encoder fine-tuning、learning_rate=1e-4、max_epochs=20、seed=7。mlm_aux_weight=0、atom_aux_weight=0、head_interaction_layers=0、label_divide_by_size=False、listwise_loss_weight=0、ranking_loss_weight=0。よって未接続 aux/efficiency config のリスクも、この 2 arm の既存結果を無効にする問題ではない。

Chemistry audit の invalid 値取り扱いについても明示する。SDF charge mismatch の 156/285 は**入力メタデータ差分**であり正解誤り数ではない。sanitize 成功した SDF だけの 75/193 を別に報告。MOL2 比較では sanitize 全成功を確認した上で元素と座標の原子対応が成立した 282 件のみを比較し、対応失敗 3 件の mismatch は -1 sentinel、集計では除外する。MOL2 形式の partial charge を formal charge として比較していない。
