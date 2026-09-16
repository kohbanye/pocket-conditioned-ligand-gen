**affinity-bench 改善調査：複合体 tokenizer と LM の情報損失から考える**

2026-09-13。対象：現行 affinity 実装、CASF crystal 構造、保存済み生予測、学習 checkpoint／token corpus、一次文献。過去の結果・失敗考察 Markdown は読まずに調査した。コード内に埋め込まれた過去の説明は一部目に入っているが、採否の証拠にはせず再確認した。本調査は仮説生成と監査であり、新しいモデルを学習して改善を実証したものではない。

具体的な改善機序・期待する効果・反証条件を14案で整理した[改善方法の整理版](MECHANISMS_JA.md)も参照。

**現時点の判断**

最も有効と思われる順番は、**化学構造の入力整合性 → 実接触を落とさない pocket 抽出 → interface の情報を保存する tokenizer → pocket を使うことを要求する LM 事前学習**である。head は現在の ligand mean + 小さい MLP を主比較で固定する。モデルを大きくする前に、入力に存在しない情報と量子化で失われる情報を切り分ける価値が高い。

理由は一般論だけではない。原子電荷の取り込みに実際の不整合があり、Cα 距離による pocket 選択は ligand に接触する側鎖を落とす。VHL、COMT、GluK1 の具体的失敗対でこの欠落と系統的な順位逆転を確認した。ただし、これらを修正すれば GenScore を超えるという因果関係・改善幅はまだ分からない。**観測された欠落、予測誤差との関連、改善の実証は別の証拠**として扱う。[実装監査](representation.md)、[構造の実測](structure_case_contacts.csv)、[全件解析コード](analyze.py)

同時に、現在の比較は学習条件まで揃った対決ではない。GenScore の欠測、入力前処理の違い、tokenizer／LM の学習来歴不足がある。CASF を見ながら新方式を選ぶ以上、CASF は探索用として扱い、別の固定テストで確証を取る設計が必要になる。[公平性監査](fairness.md)

**1．どの程度負けているかを同じ集合で再計算した**

GenScore の生 CSV は 283 件、こちらは各 run 285 件。欠ける 4k18 と 4mme は、それぞれ pocket／ligand graph の生成失敗であり、直前に原子価エラーが記録されている。Pearson は共通 283 件、標的内 Spearman は 5 ligand が完全に揃う共通 55 clusters＝275 件で計算した。これは full CASF 285/57 の再現値ではない。欠測は別途回復・報告すべきで、難しい例を黙って落とした比較にはしない。[失敗ログの所在と再計算](fairness.md)

|モデル／集計|Pearson：共通283件|標的内 Spearman 平均：完全55群|
|---|---:|---:|
|GenScore：ローカルの公開 checkpoint 1個|0.8159|0.7291|
|e250_rot1 seed 7|0.7952|0.7291|
|e250_rot1 seed 11|0.7688|0.7345|
|e250_rot1 seed 12|0.6998|0.6364|
|rot1：3 seed の指標平均 ± SD|0.7546 ± 0.0493|0.7000 ± 0.0552|
|union_clean seed 7|0.7870|0.6527|
|union_clean seed 11|0.7875|0.6727|
|union_clean seed 12|0.7920|0.6818|
|union_clean：3 seed の指標平均 ± SD|0.7888 ± 0.0027|0.6691 ± 0.0149|

出典：[metrics.csv](metrics.csv)。rot1 と union_clean はレジストリで意味の分かる対照群として固定し、多数の arm から CASF 最良を選んだものではない。いずれも保存済み f16 の予測を使用。全 arm・全履歴を調べた leaderboard ではない。

全体相関の不足と標的内ランキングの不足は同じ問題ではない。rot1 seed 7/11 は共通55群の平均順位では GenScore と同程度だが、全体相関で下回る。union_clean は seed 間の全体相関が安定している一方、順位は下回る。このデータだけから「ランキング loss を強くすれば全問題が解ける」とは言えない。

3 seed の予測平均は誤差原因の安定した可視化に使う。rot1 の平均予測は R=0.7841、union_clean は0.8066。ただし、これは**3モデル ensemble**であり、GenScore single に勝ったという主張には使わない。rot1 seed7 と GenScore の ΔR は −0.0207、target-cluster bootstrap 3,000回の95%区間は [−0.0766, +0.0380]。union ensemble の ΔR は −0.0093、区間 [−0.0517, +0.0357]。差が有意でないことは同等性の証明でもない。この区間は固定済み予測に対する test cluster の不確実性で、再学習の不確実性は含まない。またCASF群間の相同target family依存までは調整していない。[bootstrap.json](bootstrap.json)

![予測の散布図・標的別順位・サイズ別誤差](diagnostics.png)

**2．入力 chemistry に、head の前で直すべき不整合がある**

現行 `parse_sdf` は元素・xyz・結合次数を保存するが、形式電荷と stereo metadata を渡さない。RDKit molecule を元素から再構築するので、charge descriptor のスロットが存在しても元の化学状態を忠実に運べない。BioLiP の CCD template から辞書へ変換する経路にも同様の縮約がある。3D 座標は保持されるため「立体化学が完全に消える」とは言わない。[コード箇所と再現方法](representation.md)

CASF 全285件で調べると、raw SDF と再構築後の電荷不一致は156件だった。しかし raw SDF の92件は RDKit sanitization に失敗する。したがって156件を「正しい構造を壊した数」と解釈するのは誤りである。

より限定した結果は次の通り。

|検査|実測|
|---|---:|
|SDF が sanitization を通る|193 / 285|
|その193件で現行経路後に電荷不一致|75 / 193|
|同梱 crystal MOL2 が sanitization を通る|285 / 285|
|MOL2 と再構築経路の原子の元素・座標順序が対応する|282 / 285|
|対応282件で電荷不一致|167複合体、257原子|

出典：[chemistry_audit_summary.json](chemistry_audit_summary.json)、[原子対応とvalidityの明細](chemistry_audit.csv)。sanitizable な MOL2 も実験時 protonation の真値を保証しない。これは入力の不一致が大規模である証拠であり、修正後の性能を保証する結果ではない。

最初の実験は、validated MOL2／CCD と atom mapping を使った**共通の chemistry 正規化**にする。芳香族性、結合次数、charge、hydrogen count、stereo、salt/fragment の扱いを保存し、train/val/test と baseline で入力情報を揃える。元ファイルの不正電荷をそのままコピーしない。変換不能例は reason code を残す。これを直した token と古い tokenizer／LM を混ぜると別の入力分布になるので、cache と統計の fingerprint を更新し、tokenizer→LM→同一head を一貫して再学習する。

さらに共通283件のSDF/MOL2重原子座標集合を照合すると、3aryと4ekyにそれぞれ最大1.74/1.30 Åの最近傍距離差があった。他281件は0.002 Å以内だった。これは原子対応後RMSDではなく、重原子座標集合の双方向最大最近傍距離である。化学解釈だけでなく入力座標も同一と仮定できない例があるため、特に4ekyを含む失敗対の機序は前処理差を解消してから評価する。[全件の座標照合](per_complex.csv)

論文ではまず共通の正しい入力基盤として導入し、その上で tokenizer のアブレーションを比較する。前処理修正による増分を新 tokenizer の原理だけに帰属させない。

**3．実際の接触側鎖が pocket から消える**

現在の affinity は、ligand heavy atoms から **Cα が8 Å以内**の残基を候補にして最大50残基を取る。長い側鎖の末端は ligand と3 Å前後で接触していても、その Cα は8 Åより遠くなり得る。これは単なる「pocket を大きくすればよい」ではなく、**残基を選ぶ距離の定義**の問題である。実装は `src/prolit/tokenizers/protein.py` の `extract_pocket_atoms_from_candidates`。[コード監査](representation.md)

共通283件中124件、合計181残基で、ligandの4 Å以内にある標準アミノ酸残基が選択されなかった。この124件はすべてCα8 Å基準による欠落を持ち、50残基capだけが原因となる欠落は0件だった。欠落残基はARG54、TYR46、TRP45が多い。上限を増やすだけでは今回の接触欠落は解消しない。[per_complex.csv](per_complex.csv)

実際の入力座標から以下を確認した。距離は protein/ligand heavy atom 間最短距離で、水素結合の角度判定や結合エネルギーを意味しない。

|複合体|token 化されない近接残基|ligandへの最短距離|Cαへの最短距離|
|---|---|---:|---:|
|4w9h|PHE C91|3.41 Å|8.29 Å|
|4w9h|TRP C117|3.45 Å|8.19 Å|
|3oe4|TRP A38|3.51 Å|9.27 Å|
|3fv2|ARG A508|2.79 Å|9.41 Å|
|3gbb|ARG A95|2.82 Å|9.63 Å|

出典：[structure_case_contacts.csv](structure_case_contacts.csv)。図は実データの ligand と4 Å以内の protein atoms。灰色が入力に含まれる接触、赤い注釈が落ちる残基、紫が金属。各構造を ligand 主軸へ投影したもので、receptor を重ね合わせた差分図ではない。

![実複合体の接触と欠落](structure_cases.png)

検証すべき最小変更は、**残基内の任意 heavy atom と ligand の最短距離**で候補選択し、同じ残基数／token予算で比較すること。接触原子を確実に残す方式と、単に上限を50→大きくする方式を区別する。抽出を変えると canonical frame も変わるため、frame を固定した診断と全 pipeline の再学習を別に評価する。ポケット数増加と情報選択改善を混ぜない。

金属・water・cofactor は標準アミノ酸かつCAを持つ残基という選択条件では token にならない。COMT 3oe4/3ozs の Mg–ligand heavy atom 最短距離は2.11/2.15 Åだが、金属を表す token はない。一方、両 ligand に共通する金属欠落だけでその2.14 pK差の失敗を説明したことにはならない。金属配位は対象を限定した次段階の仮説とする。入力に水が既に削除されている場合、tokenizer側の変更だけでは水を回復できない。

**4．どの標的・ligand で負けるのか**

次の表は抽出した6標的のうち5標的を詳述し、残るFactor XIaの4ty7/4x6p対もCSVとligand図に含めた。候補抽出時点では3seedで頑健に逆転する1 pK以上の対が14対あり、その中で標的重複を除いている。次の対は、rot1 の3 seed すべてが1 pK以上の真値差を逆転し、GenScore は向きを正しく予測したものから、fingerprint類似度の高い順に標的重複を避けて選んだ。都合のよい手動選択ではなく、抽出ルールをコードに残した。ただしCASFの事後選択なので機序の確証には使わない。予測欄の ours は3 seed平均、GenScore はraw scoreであり、互いに同じ pK尺度ではない。[全550標的内ペア](pairs.csv)、[抽出結果](selected_cases.csv)、[ligand図](failure_ligands.svg)

|対象|複合体対 A→B|実測 pK A→B|rot1平均予測 A→B|GenScore score A→B|解くべき仮説|
|---|---|---|---|---|---|
|VHL|4w9h→4w9l|6.73→5.02|6.95→7.38|48.41→47.54|サイズ・柔軟性増加を有利に数えすぎる／溶媒露出・配座コスト不足／接触側鎖欠落|
|COMT|3oe4→3ozs|7.47→5.33|7.07→7.29|68.69→56.96|Mg配位環境、塩基置換の化学情報、脱溶媒和、TRP接触の表現|
|GluK1/GluR5|3fv2→3gbb|8.11→6.90|7.10→7.37|71.38→66.34|charged ligand と近接ARGの保持、局所接触・極性の違い|
|glycogen phosphorylase|3syr→4eky|5.10→3.52|4.97→5.68|39.77→34.85|極性相互作用の利得と脱溶媒和コスト、TYR側鎖欠落|
|thrombin|2zda→3bv9|8.40→5.36|7.50→8.14|68.71→42.11|カチオン性・柔軟性・接触密度の局所表現|

標的同定の一次資料：[VHL 4W9H](https://www.rcsb.org/structure/4W9H)、[COMT 3OE4](https://www.rcsb.org/structure/3OE4)、[GluR5 3FV2](https://www.rcsb.org/structure/3FV2)、[glycogen phosphorylase 3SYR](https://www.rcsb.org/structure/3SYR)、[thrombin 2ZDA](https://www.rcsb.org/structure/2ZDA)。表のaffinity値は外部ページではなくローカル CASF labels と生予測から計算した。

VHL 対では大きい ligand が弱いのに予測が逆になる。COMT 対とともに union_clean の3 seedでも全て逆転しており、単なる特定seedの不運ではない。対照的に、GluK1 対では union_clean の平均予測が僅かに正しい向きへ動くが、正しいseedは1/3に留まる。thrombin 対では2/3で正しい。改善の再現性を平均値だけで見ない。

COMT の元論文は塩基置換と solvation の重要性を論じる。3SYR の元論文にも、追加した相互作用の利得を脱溶媒和コストが上回る例がある。これらは「接触数を増やせば常に高 affinity」という教師設計への反例になる。ただし、3SYR 論文中の比較対を今回の3syr/4eky対と取り違えてはいけない。[COMT一次論文](https://doi.org/10.1002/chem.201003648)、[3SYR一次論文](https://doi.org/10.1111/j.1747-0285.2012.01349.x)

探索的なsubset集計でも候補を確認した。chargedはnet chargeではなく「形式電荷を持つ原子が1つ以上」であり、zwitterionを含む。化学記述子は全件でsanitized MOL2を用いた。

|subset|複合体数|rot1平均予測のPearson|GenScoreのPearson|
|---|---:|---:|---:|
|charged atomsあり|167|0.7447|0.8379|
|charged atomsなし|116|0.8363|0.7875|
|回転可能結合8本以上|49|0.6007|0.8379|
|ligand接触に複数protein chainsが関与|33|0.7258|0.8298|
|4 Å以内の金属あり|24|0.7335|0.6764|

出典：[groups.csv](groups.csv)。金属subsetではむしろrot1が上回るという反証もある。「金属が欠けるからGenScore差の主因」とは言わない。上表のoursは診断用3seed平均でありsingle対singleの勝利判定には使わない。

大きい誤差を「大きい ligand」「charged ligand」とだけ呼ぶと標的、pK範囲、サイズが交絡する。全件の化学特徴、接触、seed SD、標的別順位を[per_complex.csv](per_complex.csv)、[groups.csv](groups.csv)、[clusters.csv](clusters.csv)に残した。subsetの相関は小標本・range restriction・target compositionの影響を受け、特徴と誤差の関連は原因の特定ではない。今後は開発foldでtarget/size/label-rangeを揃えた matched pair によって検証する。

**5．tokenizer の何を改善すべきか**

現状の「joint」は共有 codebook と共通座標系を意味するが、PoseEncoder は protein と ligand の descriptor を別々に量子化する。tokenizer encoder が複合体全体の cross-interface interactions を直接 encode しているわけではない。LMは離散化された各成分から相互作用を組み立てる必要がある。[representation.md](representation.md)

最初に **raw descriptor → 量子化前 embedding → discrete code → LM hidden state** の各段階から、同程度の小さい probe で次を復元する。atom element/charge/donor/acceptor/aromaticity、短距離 contact type、距離帯、chiral volume の符号。probe もtrainで学習し、target/ligand-cluster held-outで評価する。codebookを全データで集計してからsplitする手順にはしない。

連続表現なら読めて code から読めない場合は量子化のボトルネック、descriptorから既に読めない場合は入力、codeから読めるのにLMで弱くなる場合は事前学習やreadoutを疑える。probe容量の違いで結論が変わらないかも確認する。affinityで continuous を使う対照は診断用に分離し、本提案の離散token主モデルと混ぜない。

その後の第一候補は **paired complex の cross-distance／typed-contact fidelity を tokenizer loss に入れる**こと。同じframeで双方をdecodeし、近接pairの距離・化学情報を重み付きで保存する。遠距離pairが多数を占める単純全平均は避け、距離帯別の正規化を行う。atom RMSDが同じでも塩橋や芳香環相互配置が異なるため、座標復元だけを選択基準にしない。推論時のheadは変えない。

次点として interface-context descriptor、factorized/residual quantization、原子順序とframeに対する安定性がある。ただし context KNN の実装は自己近傍4枠に追加するのでなく **self2/context2に置換**する。分子内情報を削る効果を含むので独立アブレーションとする。設定をONにするだけではPoseEncoderに渡らない経路もある。train/inference同一性のテストが先である。[詳細コード監査](representation.md)

**6．LM は既に双方向である。学習目的を interface に向ける**

実装を確認すると affinity はbidirectional MLMであり、causal→bidirectional化は今回の改善案にならない。通常の独立15%maskと全token平均CEでは、protein内部やligand内部の補完で良いlossを得られる。これだけでは相手分子を使っている証明にならない。[mlm実装の監査](representation.md)

優先実験は、(a)protein/ligand別loss正規化、(b)ligandのfragment/spatial block masking、(c)train-onlyの距離帯／typed-contact補助目的。補助decoderは事前学習時だけ使い、affinity推論では捨てる。mask token数、処理token数、独立複合体数と計算予算を揃える。fragmentを強くmaskした時、正しいpocketと長さ・組成を合わせて交換したpocketで復元性能が変わるかを開発セットで測る。

これはGenScoreの複雑なheadをコピーする提案ではない。GenScoreの距離分布による相互作用学習から、**表現へ与える教師信号**を借りる発想である。BindNetはatomic pair distance predictionとmasked ligand reconstructionを事前学習に使う。ProFSAはfrozen encoder + MLPのaffinity評価を示しており、headを固定して表現改善を示す先例になる。[GenScore](https://pubs.rsc.org/en/content/articlelanding/2023/sc/d3sc02044d)、[BindNet](https://arxiv.org/abs/2311.16160)、[ProFSA](https://github.com/bowen-gao/ProFSA)

注意点は、pose correctnessとaffinityを混同しないこと。decoyへ機械的に低pKを与えない。native周辺の全座標摂動へ同じpKを強く課すと、肝心のgeometryを無視する方向に働き得る。同一complexのglobal rotation/serializationは不変性の正例、異なる化学構造や結合配置は区別する。

**7．head と fine-tuning の改善は、比較可能な小さい範囲に置く**

実checkpoint seed7を確認した。rot1はblock512・train最大509、union_cleanはblock640・最大526で、train/valのligand切断はゼロ。両者のcapは13で、rot1にclippingはなく、unionで15,509文書中4件のみがclipされる。従って「長さでligandが消えている」「pKが一律8でclipされている」を主要因として扱わない。両者ともmean MLP、full encoder fine-tuning、LR=1e-4、追加interaction/aux/rankingは不使用だった。[実checkpoint監査](execution_contract_audit.json)

主headを固定したまま、backbone/head別LR、last-k-layer tuning、LoRAなど適応量を抑える方法を少数条件で試す価値がある。frozen linear、frozen同一MLP、同一MLPでfine-tuneの3評価は役割が違う。frozen probe改善があれば「head側が頑張っただけ」という批判への説明が明快になる。過学習抑制の効果は現在のseed差だけでは確定しない。

head自身は現行mean、linear、mean+log heavy-atom count程度に限定する。count追加はサイズshortcutを強める可能性があるのでmatched-size評価を伴わせる。lossはSmoothL1対MSE、必要ならtrainの同一target・整合するassay内の小さいpairwise ΔpK項を比較する。Kd/Ki/IC50の混在と近いラベルの測定誤差を先に扱う。

rot1平均予測ではpK<5の平均誤差が+1.80、pK≥9では−1.80、予測SD1.20に対し真値SD2.18でrange compressionがある。しかし **正の線形再校正ではPearsonは変わらない**。傾きを広げるだけの修正を相関改善案にしない。RMSE改善と、正しい順位／標的間offsetを学ぶ改善を分ける。[metrics.csv](metrics.csv)、[groups.csv](groups.csv)

**8．既存手法に勝つための比較設計**

文献上のCASF数値を一つの表に並べるだけでは不公平になる。CleanSplitで再学習したGenScoreの報告値はR=.780、GEMS=.803であり、今回のローカル公開GenScore .8159とはtraining/protocolが違う。これを「実装が悪くないこと」の参考にはできても、CleanSplitで今回のモデルが既に勝ったとは言えない。2026年のBA-PredはCleanSplitで.818を報告するが5-fold ensembleで、公開default checkpointはrandom split。singleとensembleを分ける必要がある。[一次文献・版・数値条件の一覧](literature.md)、[GEMS](https://www.nature.com/articles/s42256-025-01124-5)、[BA-Pred](https://doi.org/10.1021/acs.jcim.5c02591)

二つの比較trackを分けるのがよい。一つは同じ学習複合体・label・前処理・split・tuning予算でGenScore等を再学習するtrack。もう一つは外部pretrainingを含む公開best-availableモデルとの実用比較track。tokenizerの因果的主張は前者で、公開モデルへの到達度は後者で示す。sequence+SMILES入力のBoltz-2とcrystal-inputモデルも同一情報条件としては扱わない。

リーク監査はtokenizer fit、正規化統計、LM pretraining、affinity training、validation/model selectionの全段階を対象とする。全285 CASF IDが除外リストにあることは確認できたが、別PDBの同一ligand–target、相同pocket、同じassayの再収録まで除外できた証拠はない。tokenizer/MLMのsource ID manifestや除外hashが不足し、正規化cacheはsplitに紐付かない再利用が可能。現時点でリークを断定するのも「完全にleak-free」と認定するのも不適切である。[fairness.md](fairness.md)

評価は用途を二つに分ける。既知target系列内のligand最適化なら、同一target内の近縁分子・activity cliffを測る。未知target/ligandへの転移なら両軸をhold outする。どちらも共通のprotein/ligand entityのduplicateを管理し、testごとの最近傍train類似度を報告する。CleanSplitはaffinity差をfilterに使うので純粋なinput-only splitと区別する。LP-PDBBind v3のligand条件はMorgan Dice≤.99で、厳しいcold-scaffold保証ではない。[CleanSplit/LP-PDBBind/DataSAILの原典と注意](literature.md)

**9．次の実験をこの順で実施する**

以下はこれから固定する計画であり、今回のCASF解析を遡って事前登録と呼ぶものではない。採用判断はdevelopment foldsで行い、最終holdoutは選択に使わない。予算は相対的な見積もりで、実ジョブ時間・ポイントは測定していない。

|順序|実験|変更するもの／固定するもの|採否の証拠|相対コスト|
|---|---|---|---|---|
|0|比較と来歴を修復|共通ID・missingness・化学前処理・manifest、モデルは固定|欠測理由と同じ母集団の指標、全stage exclusion監査|低〜中|
|1|情報の所在をprobe|input/continuous/code/LMでprobe家族・split固定|どの段階でcharge/contactが失われるか|低〜中|
|2|chemistry正規化|valid atom mappingとmetadata保存、同じarchitecture/head|validity改善＋同じ開発foldでaffinity再学習|中〜高：cache/VQ/LM更新|
|3|heavy-atom基準pocket|同じtoken予算・head、抽出定義のみ|接触残基recallとaffinity、frame変化対照|中〜高|
|4|interface保存tokenizer|cross-distance/typed-contact loss、headとLM予算固定|量子化後contact fidelity＋frozen probe＋affinity|高|
|5|interface MLM|mask/loss正規化とtrain-only auxiliary、tokenizer/head固定|pocket依存性＋frozen probe＋affinity|中〜高|
|6|適応量の制御|別LR/last layers/LoRA、tokenizer/head固定|複数seedのtarget-heldout性能・variance|中|
|7|不変性|atom order/global views、処理予算固定|token/embedding安定性、rank flip減少、chemistry保持|中〜高|
|8|小head/loss対照|linear/mean/count、MSE/少量ΔpK|改善がrepresentation変更を代替するか確認|中|
|9|pseudo fragment pretraining等|source親proteinも除外、同計算量対照|上位変更後にも追加価値が残るか|高・後段|

最小の帰属実験は新旧tokenizer × 通常/interface MLM の2×2、headは同一。tokenizerが変わる場合はcode意味が変わるためLMもそのtoken空間で学習し、旧LMをそのまま組み替えない。frozen probeと同じfine-tuning recipeを併記する。3seedは最初のscreeningとして使い、僅差の勝利主張では必要なseed数を開発データの分散から決める。モデル数、token bitrate、params、処理tokens、独立複合体数を明示する。

ligand-only、protein-only、target-identity有無、類似ligand nearest-neighbourを同じsplitで置く。tokenizerの改善が接触probeを改善し、相手pocketへの依存性が生まれ、単純headでaffinityへ繋がるという連鎖を示す。単に再構築RMSDやcodebook使用率が良いだけでは不足する。

**限界・再現性**

この調査はAIによるコード監査・文献探索・数値解析を用いた。target/ligandの定量解析は共通283件、chemistry監査は全285件。構造の接触判定は距離ベースで、H-bond角度、結晶水、溶媒自由エネルギー、配座entropyを実測したものではない。各PDBの化学状態・assay整合性を実験文献まで完全に監査したわけではない。全pretraining sourceの類似性監査とGenScore再学習、token再構築probe、新方式の学習は未実施である。したがって現状は原因候補の優先順位であり、勝利保証ではない。

生予測は[input_manifest.json](input_manifest.json)にSHA256を保存した。再現はrepo rootから以下を実行する。構造特徴のchemistryはsanitize済みcrystal MOL2、距離解析は実際のinference用SDF heavy座標。両ファイルの座標集合の最大最近傍距離もCSVに記録する。fingerprintはMorgan radius2/2048-bit Tanimoto、接触は4 Å、現行pocketはCα8 Å/50残基。予測の3seed平均は診断用、Pearson比較は共通283、順位比較は完全55群、bootstrapはcluster単位でseed20260913固定。subset解析は探索的で多重検定を経た因果発見ではない。

```bash
.venv/bin/python docs/research/affinity_fresh_20260913/audit_chemistry.py
.venv/bin/python docs/research/affinity_fresh_20260913/audit_execution_contract.py
MPLCONFIGDIR=/tmp/affinity_mpl .venv/bin/python docs/research/affinity_fresh_20260913/analyze.py
MPLCONFIGDIR=/tmp/affinity_mpl .venv/bin/python docs/research/affinity_fresh_20260913/plot_structures.py
```

本番コードと既存結果は変更せず、このディレクトリに調査資料と再現スクリプトを追加した。新しい学習ジョブの投入はしていない。
