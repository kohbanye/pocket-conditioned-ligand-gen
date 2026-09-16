**改善の理由を説明できる14案：何を変えると、何が改善するはずか**

この資料は「こういう改善をすればここがよくなる、と論理的に説明できる方法を色々と見つけてまとめる」という依頼への整理版。数値・構造の調査は[詳細レポート](REPORT_JA.md)、実装の所在は[representation.md](representation.md)、比較条件は[fairness.md](fairness.md)にまとめた。2026-09-13に保存済み予測のSHA256を現在のファイルと照合し、以下の根拠が参照する入力が一致することを再確認した。

**結論は、情報が不足する段階に対応した変更を行い、その情報が回復したこととaffinityが改善したことを別々に示す、という方針である。** 最終headは原則として現行ligand mean＋MLPに固定する。補助学習器は事前学習時だけに置ける。大きなaffinity headで不足を補うことを主眼としない。

下記の「改善するはず」は条件付きの仮説であり、改善が実証済みという意味ではない。**A＝ローカルの欠落を実測、B＝実装・誤差から動機づけられるが機序は未実証、C＝追加診断が成立すれば進める候補**とする。この等級は予測される改善量の大きさでも、論文の新規性評価でもない。

|ID|変更する場所と方法|改善を期待する対象|根拠|最初の優先度|
|---|---|---|---|---|
|M01|化学情報を正規化して保持|電荷・結合・donor/acceptorの解釈|A|前提整備|
|M02|heavy-atom接触でpocketを選ぶ|長い側鎖の接触欠落|A|最優先|
|M03|tokenizerのcross-interface復元損失|量子化後の相対配置・接触|B|主研究候補|
|M04|化学状態と幾何を分けて量子化|化学的に違う環境のcode衝突|C|probe後|
|M05|interface文脈をtokenizerへ渡す|相互作用環境の識別|B|M03との代替比較|
|M06|MLMのprotein/ligand損失を分ける|ligand側へ届く学習信号|B|低変更量の対照|
|M07|ligandのまとまった領域をmask|pocketを使う文脈表現|B|主研究候補|
|M08|LMにtrain-only接触・距離教師|小headで読み出せる相互作用|B|主研究候補|
|M09|剛体回転・原子順序の整合性学習|同じ複合体の表現のばらつき|C|variance診断後|
|M10|金属・cofactorをtoken化|金属配位等の入力欠落|A/C|対象限定の後段|
|M11|埋没・露出・内部自由度を補助学習|サイズ増加を利得と誤認する例|C|探索的後段|
|M12|LMの更新量を制御|少数labelでの表現破壊・seed差|B|並行する学習対照|
|M13|同一assay/targetのΔpKを学習|標的内ligand順位|B|label整備後|
|M14|readoutの最小アブレーション|平均poolingによる情報希釈|C|原因切り分け用|

**M01．化学情報の正規化と保持**

**根拠：** 現行SDF→辞書→RDKit再構築経路は形式電荷等を保持しない。sanitization成功SDF193件でも75件に電荷不一致がある。一方raw SDF92件はsanitizeできず、無条件に原ファイルを真値にするのも誤り。[chemistry_audit_summary.json](chemistry_audit_summary.json)、[audit_chemistry.py](audit_chemistry.py)

**変更 → 中間効果 → 予測への期待：** MOL2/CCD等で同一分子・原子対応を確認してbond order、charge、hydrogen count、stereoを保存する → 異なる化学状態を誤って同じdescriptorへ潰すことを減らす → 荷電ligandと極性残基の組合せを学ぶための情報が増える。情報が入力前段で消えていれば、LMやheadの容量だけを増やしても元情報を一意には復元できない。

**最小検証：** 同じ化学参照・sanitization条件でのatom-level一致率をまず確認し、次に同じheadで一貫して再token化・再学習する。入力忠実性の修正と、実験pHにおけるprotonation推定は別の実験にする。

**反証・限界：** 化学一致率だけ上がりheld-out affinityが改善しなければ、正当な入力修正ではあっても性能差の主要因ではない。sanitizableであることは実験状態が正しいことを保証しない。修正による増分をtokenizer固有の新規性と呼ばず、以降の全armの共通入力基盤として扱う。

**M02．Cαでなく実際の接触に基づいてpocketを選ぶ**

**根拠：** 共通283件中124件、計181残基で4 Å以内のheavy-atom接触が入力から落ちる。いずれもCα8 Å基準の欠落で、50残基capによる欠落は0件。GluK1 3fv2ではARG末端が2.79 Å、Cαは9.41 Åに位置する。[per_complex.csv](per_complex.csv)、[structure_case_contacts.csv](structure_case_contacts.csv)

**変更 → 中間効果 → 予測への期待：** 残基内heavy atomとligandの最短距離で選択する → 長い側鎖が形成する局所環境を保持する → 芳香族・極性・荷電接触を区別する材料がLMに入る。現状の問題に対して残基上限だけ増やしても、候補に入らない残基は回復しない。

**最小検証：** まず座標だけでcontact recallが上がるかを調べる。学習比較は同じ総token予算で接触優先選択と現行選択を比べる。残基数を揃えても残基ごとの原子数は異なるので、実token数・計算量も報告する。抽出変更でPCA frameも変わるため、旧frame固定の診断と新frameを含む全pipelineを分ける。

**反証・限界：** 接触recallが上がってもaffinityが改善しなければ、後段が接触を利用できていない可能性が残る。保持した残基を消した際の予測変化だけではOOD入力への感度と区別できないため、frozen contact probeと再学習比較を併用する。新しいcropの閾値はCASF誤差で選ばない。

**M03．tokenizerの目的を、interfaceの相対配置の保存まで拡張する**

**根拠：** 現行PoseEncoderはproteinとligandを別々に量子化する。共通codebookがあることと、量子化が分子間接触を保存することは同義ではない。[実装監査](representation.md)

**変更 → 中間効果 → 予測への期待：** paired complexの両側を共有frameでdecodeし、cross-distanceとtyped-contactの損失を加える → 相互作用に関係する相対配置の誤差を直接減らす → 少ない教師labelと小headでも近縁ligand間の接触差を読み出しやすくする。

数学的に、原子位置誤差をeᵢ、eⱼとすると距離誤差は `|d̂ᵢⱼ−dᵢⱼ| ≤ ||eᵢ||+||eⱼ||`。ただし平均座標誤差が小さいことは、特定の重要接触の誤差が小さいことを保証しない。遠距離pairが多数の全平均でなく、近接帯と非接触帯を層化して損失を計算する理由がここにある。距離だけでは水素結合方向や化学的相補性は決まらないため、typed chemistryと角度指標も必要になる。

**最小検証：** まずcontinuous embedding対codeのcontact probeを比較する。量子化時の低下があれば、同じcodebook/bitrateで現行損失とcross-interface損失を比較し、contact fidelity、frozen affinity probe、固定headでのaffinityを順に確認する。

**反証・限界：** contact fidelityが改善してもaffinityが変わらなければ、必要情報が別にあるか、下流が使えていない。最終headに追加interaction networkを足して救済するとこの仮説の評価にならない。訓練時の補助decoderの容量・計算量も対照に含める。

**M04．化学状態と幾何を分けた量子化**

**根拠：** code衝突が現在の主要因かは未測定。M01を直した後、同じcodeが異なるcharge/atom typeを過度に共有するか調べてから進める。[probe設計](REPORT_JA.md)

**変更 → 中間効果 → 予測への期待：** chemistry用codeとgeometry residual用codeを分ける、あるいは量子化にchemistry保持の制約を入れる → 座標の細かな変化と引き換えに化学状態を取り違える圧縮を減らす → 薬効差に重要な置換・電荷を安定して表す。

**最小検証：** `H(chemical label | code)`等の混同量とgeometry歪みを同じheld-out splitで測る。まず一tokenのまま損失重みを変える対照、その次にfactorized codeを試す。二token化するなら総bitrate、sequence長、LM計算量を揃えた対照を置く。

**反証・限界：** codeから既に化学情報が容易に読めるなら優先度は低い。codebookを増やしただけの利得やLMの入力情報量の増加と区別する。raw charge tagを追加する方式は有用な診断だが、tokenizerがその情報を学習したという主張にはならない。

**M05．tokenizer自身に相手分子の局所文脈を与える**

**根拠：** 現行affinity経路はligand descriptor計算へpocket contextを渡さない。別経路のcontext KNNはself4をself2/context2に置換する実装で、単なる追加ではない。[representation.md](representation.md)

**変更 → 中間効果 → 予測への期待：** 相手の近接元素・距離等を少量descriptorへ渡す、あるいはpaired encodingを行う → 同じligand原子でも異なるbinding environmentをtoken段階で区別する → LMが粗い独立codeから全相互作用を再構成する負担を減らす。

**最小検証：** 同一分子内geometryと異なるinterfaceを持つtrain/development例で、contact-type probeを比較する。self-neighbour削減のみの対照も置き、cross-contextの効果と情報置換の損失を分ける。train/inferenceのdescriptor契約を一致させる。

**反証・限界：** codeの文脈依存性が強すぎると共通化が難しくなり、少ないlabelでは逆効果もあり得る。相手pocketが変更されればligand codeも変わるので、M07のpocket swap診断ではこの変化を考慮する。M03との大規模併用より、まず代替案として比較する。

**M06．protein/ligand別にMLM損失を正規化する**

**根拠：** 現行は通常のmask CEで、protein tokenが多ければlossの多くをprotein側が占める。実際に勾配がどちらへ配分されているかは未測定。[mlmコード監査](representation.md)

**変更 → 中間効果 → 予測への期待：** `L = α mean(CE_protein) + (1−α) mean(CE_ligand)`として、masked token数の偶然の差と重みを切り離す → ligand復元へ意図した学習量を配分できる → ligand hidden stateを使う既存readoutの情報を増やす可能性がある。

**最小検証：** 両成分のmasked-token数、loss、gradient norm、held-out復元精度を記録する。mask数・処理token数を揃え、αはdevelopmentで少数候補から選ぶ。

**反証・限界：** lossの数値だけ変わっても進歩ではない。ligand復元は良くなるがpocketを変えても性能が変わらない場合、相互作用学習の改善とは言えない。M07/M08とは別の小変更対照として扱う。

**M07．独立atom maskからfragment／spatial block maskへ**

**根拠：** 独立15%maskでは隣のligand情報だけで補完できる可能性がある。これは現行がpocketを無視しているという実証ではなく、検証すべきshortcut仮説。[representation.md](representation.md)

**変更 → 中間効果 → 予測への期待：** ligand内の近傍情報をまとめて隠す → ligand内部だけで解ける補完を減らす → 正しいpocketが予測に寄与する余地を増やす → affinityに使えるcross-interface hidden stateを育てる。BindNetのmasked ligand reconstructionは関連する一次研究だが、この具体的mask方式の有効性を保証しない。[BindNet](https://arxiv.org/abs/2311.16160)

**最小検証：** 同じmask対象数で独立maskとblock maskを比べる。`Δcontext = loss(交換pocket) − loss(正しいpocket)`に加え、正しいpocketでの絶対復元精度とfrozen affinityを測る。交換は長さ・組成を合わせたtrain/development上の診断に留める。

**反証・限界：** Δcontextが大きいだけでは正しく相互作用を学んだとは言えない。交換pocketが非物理的であることや難易度増加でも差が出る。pocket-conditioned tokenizerではligand code自体にpocket情報が入るため、交換条件ごとの再token化または独立tokenizer対照が必要。ほぼ復元不能なmaskを増やすだけなら採用しない。

**M08．LMへ距離帯・typed contactの自己教師を与える**

**根拠：** GenScoreは相互作用の距離分布を扱い、BindNetはatomic pair distanceを事前学習に利用する。今回借りるのは教師信号で、affinity headの構造ではない。[GenScore作者実装](https://github.com/sc8668/GenScore)、[BindNet](https://arxiv.org/abs/2311.16160)

**変更 → 中間効果 → 予測への期待：** LM hidden stateにtrain-onlyの小さなpair decoderを付ける → 表現が原子対の接触関係を含むように学習される → 推論時にpair decoderを捨てても、同じmean＋MLPで相互作用情報を読みやすくする。

**最小検証：** 元CEとCE＋補助lossを同一corpus/computeで比較する。補助decoderが強いだけでなく、小さい独立probeでも接触が読めるかを見る。補助精度、frozen affinity、同一fine-tuningの順で評価する。

**反証・限界：** 独立probeが良くならず補助decoderだけが解けるなら、LM表現が改善した根拠は弱い。結晶poseの接触だけでbinding free energy全体は決まらない。native/decoyを使う場合もdecoyへ偽の低pKラベルを与えない。tokenizer/LMの自己教師用構造からも最終testを除外する。

**M09．同一複合体の回転・serializationに対する整合性**

**根拠：** tokenizer/LMは原子順序の位置情報を使い、frameと量子化に依存し得る。ただし現在のノイズ量・誤差との関係は追加測定が必要。training-seed SDと回転SDは別物である。[representation.md](representation.md)

**変更 → 中間効果 → 予測への期待：** 同一複合体のSO(3)回転や正しく再indexした原子順序をpaired viewsとして学ぶ → 物理的に同じ入力のpooled representationを安定化する → affinityと無関係な変動を減らす。説明用モデルで予測を`s+ε`としεが真値・sと独立なら、`Corr(y,s+ε)=Corr(y,s)/sqrt(1+Var(ε)/Var(s))`。この仮定下ではnoise低下が相関を上げる。

**最小検証：** raw座標を回転してframeを再計算する試験と、frame後のdescriptorを回転する試験を分ける。複合体内prediction SD、近いligand間rank flip、chemical probeを測る。view数で独立label数が増えたと数えず、同じoptimizer steps/処理量で比較する。

**反証・限界：** 実際の誤差が真値に依存すれば上式の結論はそのまま使えない。SDが既に小さければ優先度は低い。全表現を一定にするcollapseでも整合性は達成できるため、復元・識別能力を同時に保つ。mirror reflectionはchiralityを変えるので回転と同一視しない。異なるligandや大きなpose変化を同じ正例にしない。

**M10．金属・cofactorを限定的に追加する**

**根拠：** 現行protein tokenは標準アミノ酸かつCAありに限定され、COMT例では2.11 Å近傍のMgが入らない。ただし金属4 Å以内subsetではrot1平均予測R=.7335、GenScore=.6764で、金属欠落が全体の敗因という説明には反証がある。[構造距離](structure_case_contacts.csv)、[subset集計](groups.csv)

**変更 → 中間効果 → 予測への期待：** 元構造にある金属/cofactorを明示的なtyped tokenとして保持する → 配位環境が可視化される → 金属依存targetの化学置換を区別できる可能性がある。

**最小検証：** 金属別・標的別の少数例から、入力参照と配位距離の妥当性を確認する。共通の追加情報をbaselineにも許可し、金属あり/なしの両群を同じ固定headで評価する。

**反証・限界：** 両ligandで同じ金属が欠けるというだけでは、そのaffinity差を説明しない。追加が全体で効かなくても対象限定の研究は成立するが、一般改善と呼ばない。水が入力から既に除去されている場合はwater-tokenの追加だけで解決しない。欠測水の推定は別task・別不確実性になる。

**M11．接触利得だけでなく露出・内部自由度を表現に残す**

**根拠：** VHL 4w9h→4w9lは33→41 heavy atoms、回転可能結合6→8、4 Å接触を持たないligand原子割合約.18→.32、実測pK6.73→5.02だがrot1平均予測6.95→7.38。この相関は機序の確定ではない。[per_complex.csv](per_complex.csv)、[selected_cases.csv](selected_cases.csv)

**変更 → 中間効果 → 予測への期待：** tokenizer/LMにatomごとの埋没・露出、接触密度、ligand内部geometryや自由度proxyを補助復元させる → 「原子が増えた」「接触が増えた」以外の状態を表現に残す → 小headでも大きく柔軟なligandの利得と不利を区別する可能性がある。最終headへ物理スコアを多数足すことを第一案にしない。

**最小検証：** 同一targetでサイズ/chemistryを揃えた開発ペアを用い、現行token/LMから既にこれらが読めるか測る。必要ならtrain構造から得たproxyのみを補助教師とし、情報保持とΔpKの双方を見る。

**反証・限界：** rotatable-bond数はentropyそのもの、距離ベース露出は脱溶媒和自由エネルギーそのものではない。「接触数−柔軟性」で正しい物理モデルができたと主張しない。proxyが十分読めるのに誤る場合はtoken不足よりlabel量や学習目的を疑う。unbound ensembleの教師を使う拡張は高コストで、conformer sampling誤差を伴うため後段に置く。

**M12．少数labelに合わせてLMの更新量を抑える**

**根拠：** rot1の3seed Rは.7952/.7688/.6998。一方union_cleanは.7870/.7875/.7920。これは不安定性の存在を示すが、過学習やcatastrophic forgettingが原因という証明ではない。実checkpointの両seed7はfull fine-tune、LR1e-4。[metrics.csv](metrics.csv)、[execution_contract_audit.json](execution_contract_audit.json)

**変更 → 中間効果 → 予測への期待：** backbone/head別LR、last-k-layer更新、LoRA、元重みからの逸脱抑制を少数条件で比較する → ラベルの少ない領域で有用な事前学習表現が崩れる自由度を減らす → held-out targetでの再現性と平均性能を上げる可能性がある。

**最小検証：** frozen linear、frozen現行MLP、full fine-tune、制限したfine-tuneを同じhead familyで比較する。affinityの平均・SDに加え、fine-tune前後のcontact probeとvalidationの動きを見る。更新可能paramsと調整予算を報告する。

**反証・限界：** freeze/LoRAは必要なadaptationも制約する。trainもvalも弱くなればunderfittingが考えられる。seed差が減っただけで平均が悪化する場合、性能改善とは言えない。random-init LM対照は事前学習の寄与を示すが、計算量差も明記する。

**M13．同一target・整合するassay内のΔpKを教師にする**

**根拠：** 全体Rと標的内順位は一致しない。rot1 seed7は共通55群の順位がGenScoreと同じ.7291でも全体Rは下回る。従ってこの変更はまず標的内順位を狙い、全体R改善を自動的には期待しない。[metrics.csv](metrics.csv)

**変更 → 中間効果 → 予測への期待：** 絶対pK回帰に小さいpair差分損失を加える → ligand置換が生む相対変化を直接学ぶ → 標的間offsetで全体指標を稼ぐだけでは改善しない学習信号になる。`y_ti=b_t+u_ti+ε_ti`なら、同一target内の差分で共通b_tは消える。

**最小検証：** 同一targetだけでなくmeasurement type/assay条件を揃える。差が測定不確実性と同程度のpairは事前に定めた扱いにする。同一assay内のpair差分と絶対回帰の重みをdevelopmentで固定し、macro順位、target-centered相関、全体Rを併記する。

**反証・限界：** 異なるassay由来のoffsetはtargetが同じでも相殺しない。独立label noiseなら差分noise分散は両者の和になり、かえって悪化し得る。pair数は独立label数ではなく、多数ligandを持つtargetが支配しない重みが必要。ΔpK項のみでは絶対offsetを学べないので回帰を残す。test対の正解順位を訓練へ戻さない。

**M14．headの変更を原因切り分けに使う**

**根拠：** 現行はligand hidden stateのmean＋MLP。meanが重要な局所特徴を希釈する可能性はあるが、contextual state自体がサイズ等を含めるため、必ず情報を失うとは言えない。[rescore実装監査](representation.md)

**変更 → 中間効果 → 予測への期待：** 同じLM表現でlinear、現行mean MLP、mean＋log atom-count程度を比較する → 表現から読み出せないのか、readoutが足りないのかを分ける。説明用に局所信号がk原子だけにあり他原子がゼロならmean振幅はk/nになるが、実LMではこの仮定が成立するかprobeが必要。

**最小検証：** tokenizer/LM変更の主比較は現行headに固定し、head対照は別表にする。readoutを変えても新tokenizerの増分が残るかを見る。サイズを追加する場合はmatched-size/ligand-only対照を置く。

**反証・限界：** count追加だけで差が消えたら、tokenizerが相互作用を改善したという説明は弱くなる。相互作用GNNや大規模cross-attentionをheadへ足すと、何が効いたかの帰属が難しくなるため、この論文の第一選択にしない。

**「どこを直すか」を決める診断の順番**

まずM01/M02で入力欠落を評価し、次にraw descriptor→continuous embedding→code→LM hidden stateの同容量probeを取る。入力から読めないなら前処理、continuousから読めてcodeで失うならM03/M04/M05、codeで読めてLMで弱いならM06/M07/M08、view間のばらつきが大きいならM09、fine-tuneで崩れるならM12、表現にあるがmeanから読めないならM14へ進む。probeの容量不足や分布差でも見かけのボトルネックが生まれるので、単一probeだけで原因確定しない。

全案に共通して、**中間指標だけの改善とaffinity改善は別判定**にする。最も明快な成功パターンは「固定予算でcontact/chemistryが保存される → frozen probeでも読める → 同じ小headでheld-out affinityが改善する」。機序を切り分けるには一変更ずつの対照が必要で、この連鎖だけで厳密なcausal mediationを証明したとは呼ばない。

**組み合わせる前の最小実験系列**

共通のchemistry/crop基盤を整えた後、M03型の旧/新tokenizer × 通常/M07–M08型LM目的を比較する。まずM07とM08は別々に効果を確認してから組み合わせる。tokenizerが違えばcodeの意味が違うため、それぞれに対応するLMを学習し、旧LMのembeddingを無条件に流用しない。head、supervised IDs、独立complex数、選択metric、seed、tuning予算を固定する。M12は主representation比較の両側へ同じように適用する。

ProFSAのような表現事前学習研究は関連先例だが、別benchmarkの改善値を今回の期待増分として転用しない。新規性は共有codebookの存在だけでなく「相互作用に必要な情報を、どの圧縮予算で保持できるか」で説明する。[ProFSA作者実装](https://github.com/bowen-gao/ProFSA)、[検証済み文献比較](literature.md)

**公平性とリークについて、この資料から主張できる範囲**

既存CASFの構造・誤差を見て選んだ案は探索的であり、この同じ集合を未使用testと呼ばない。確証用holdoutはtokenizer fit、normalization、LM pretraining、affinity training、モデル選択の全段階から除外する。exact PDB IDだけでなく、protein/ligand identity・近縁性・assay重複と用途に対応したsplitを監査する。現checkpointを完全leak-freeと認定した結果ではない。[fairness.md](fairness.md)

GenScoreとの比較は欠測を明示し、同じIDと完全なranking群で行う。今回の共通集合はPearson283件、順位55群275件。singleとensemble、追加pretrainingの有無、入力chemistryとpose、tuning予算を区別する。baselineへ同じ学習条件を与える比較と公開weightsの実用比較を分ける。全体相関と標的内順位のどちらを改善したかを報告し、片方の改善をもう片方の改善として説明しない。

**優先しない説明も明記する**

「cap50を大きくすれば接触欠落が解消」「causal attentionを双方向へ直す」「pKが8でclipされているから直す」は、今回確認した実装・checkpointに合わない。「正の線形校正でPearsonが上がる」も数学的に成り立たない。codebook使用率やMLM lossが良いだけでaffinity表現が良いとも言えない。これらを外すことで、根拠が残る14案へ実験資源を向けられる。[詳細の反証](REPORT_JA.md)、[実checkpoint監査](execution_contract_audit.json)

**完了範囲：** 改善候補の探索、論理的機序、反証条件、優先順位、公平な検証設計までをまとめた。提案の新規学習やGenScore超えの実証は含まない。一次文献、コード、保存済み構造・予測をAI支援で解析した資料であり、未検証のmechanismを発見事実としては扱っていない。
