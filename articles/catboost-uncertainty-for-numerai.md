---
title: "Catboost UncertaintyをNumeraiで活用する"
emoji: "👻"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["numerai"]
published: true
---

この記事は[Numerai AdventCalendar 2022](https://adventar.org/calendars/7631)の23日目の記事です。
https://adventar.org/calendars/7631

色々な意味で冬に入ってますね。自分も今年はほとんどモデルを作っておらず、観戦の日々が続いております。  
今回はマケデコなどで少し話がでていたCatboost Uncertaintyを使った知見を共有できればと思います。

## Catboost Uncertaintyとは
勾配ブースティング系のライブラリの一つでCatboostには予測の不確実性(Uncertainty)も推定できる機能があります。  
ここでは理論的な説明などは（自分が完全に理解をしていないため）割愛させていただきますが、一言でいうと **「確率的勾配ランジュバン動力学法を勾配ブースティングで使えるようにした手法:SGLBなどを利用して学習をして、予測と同時に2種類の不確実性に分解して推論する」** ということをしています。  
理論的に知りたければ以下の原著を読むといいかと思います。
https://arxiv.org/abs/2006.10562
https://arxiv.org/abs/2001.07248
「確率的勾配ランジュバン動力学法」についてはベイズ深層学習の本に説明などがあります。
https://amzn.to/3PNm8fs
また、応用では公式ドキュメントにもreferenceされているこちらのチュートリアル記事がかなり有用だったので、今回はこちらを元に説明していきいます。
https://towardsdatascience.com/tutorial-uncertainty-estimation-with-catboost-255805ff217e

### 2種類の不確実性
勾配ブースティングの不確実性を扱えるライブラリとしてはNgboostなどがありますが、Catboostでは **知識の不確実性（Knowledge Uncertainty）** と **データの不確実性（Data Uncertainty）** の２種類の不確実性に分けて推論できます。2種類の不確実性については原著の例をみるとわかりやすかったので紹介します。以下の図は2つのカテゴリ変数を使った回帰問題の合成データを用意し、目的変数やその予測の各不確実性をヒートマップで表した図です。  

![](https://storage.googleapis.com/zenn-user-upload/c1effb9879cc-20221223.png)  
*"Uncertainty in Gradient Boosting via Ensembles"で紹介された合成データによる不確実性推定*

カテゴリ変数自体に順序性はないですが、不確実性を幾何学的に把握できるように横軸がカテゴリ変数 $x_1$ 、縦軸が $x_2$ として可視化しています。(a)は **学習データのなかでそれぞれの変数で観測されたtargetの分散(データの不確実性)** を表しており、ハートの輪郭がもっとも分散が大きいです。 一方で輪郭の内部は真っ白になっていますが、こちらは学習データを用意できなかった領域としており、観測できなかったため０になっています。
(b), (c)はそれぞれモデルで推定した不確実性になっています。(b)は(a)の不確実性のようにデータの不確実性をうまく推定できており※、(c)では逆に **学習データになかった部分の予測の分散(知識の不確実性)** を推定できているといえます。論文では、 **out-of-domainを検出するために利用するデータの不確実性と、Active Learningなどで利用できる知識の不確実性を分解できることが応用では重要である** ことが主張されています。

> ※ただ、論文中では "Total Uncertainty"と記述しているので、このヒートマップは知識不確実性も含まれてるかもしれないです。

### 利用方法
上記の不確実性を含めて推論するためのモデル構築は、適切なconfigを設定してfit/predictするだけで可能です。お手軽ですね。  
```python
params = {
  # Uncertaintyを推論するための追加設定
  "objective": "RMSEWithUncertainty",
  "posterior_sampling": True,

  "iterations": 1000,
  "learning_rate": 0.01,
  
  # GPUは未対応
  "task_type": "CPU",
}

model = CatBoost(params)
model.fit(X, y)
```
不確実性を含めた予測をするときには、専用の関数として`virtual_ensembles_predict()`が用意されており、予測値だけでなくそれぞれの不確実性を加えた`(データ数 , 3)`の配列が返ってきます。Knowledge Uncertaintyを計算するときには複数モデルを構築する必要があるのですが、本論文で提案されているVirtual Ensemblesという勾配ブースティングの特性を活かした近似計算手法を利用しています。
```python
# Virtual Ensembleを行う学習木の数を指定
n_esm = 200
preds = model.virtual_ensembles_predict(
  X_valid, 
  prediction_type="TotalUncertainty", 
  virtual_ensembles_count=n_esm
)
```


## Numeraiでの活用
不確実性を活用する方法は様々議論されているかと思いますが、一番わかりやすいのはドメイン外検出（out-of-domain）などに則って不確実性が高いときに予測を運用時に利用しないという後処理かなと思います。  
以前自分が投稿した[ベイズNNでの活用](https://www.kaggle.com/kansukehabano/numerai-parameter-distribution-analysis-using-bnn)のように、不確実性を利用してパフォーマンスが悪かったroundを検出できるか分析してみます。さらに、今回は2種類の不確実性で評価を比較し、考察を行います。

### 評価設計
Numeraiでは、予測の評価として各roundの予測値とtargetの相関が指標の一つになっています。（最近TCが出てきたのでややこしくなってしまいましたが。。。）
データが優秀な分、あまりおきることは少ないですが、相関が負になった場合はよく「Burn Era」などと呼ばれ、いかにBurn Eraを少なくするかがモデリングのポイントの一つになっています。今回は、 **「era内で平均したそれぞれの不確実性をeraの不確実性としてBurn Eraかどうか２値分類できるか」** をAUCで評価します。  

実装手順としては以下のとおりです。  
1. あるtargetに対して、Catboost Uncertaintyで学習
2. Validation Dataに対して、Virtual Ensemblesで予測値と2種類の不確実性を推論
3. 指定したtargetをもとに、Correlationを計算し、負の値かどうかを算出
4. それぞれの不確実性でeraごとに平均をとり、aucを計算

実装は以下のNotebookで雑に実装しています。
https://www.kaggle.com/code/kansukehabano/numerai-multitarget-distribution-catboost  
学習した設定は以下のとおりです。  
- v4データの`fncv3_features`を特徴量とする
- 学習で利用するeraは12周期に間引く（後程の追加実験の事情で間引いています）
- validationはそのまま利用する

### AUCの結果
以下はそれぞれの不確実性で算出したROC曲線とAUCになります。  
![](https://storage.googleapis.com/zenn-user-upload/f8cd0a835a8e-20221223.png)
Data Uncertaintyは高く、それなりにBurnを検出できてそうです。  
一方でKnowledge Uncertaintyはチャンスレート以下となっており、分類がうまくいっていないことがわかります。  
ここから言えることとしては、今回の学習条件では**Burn Eraは「学習データになかったデータでの予測値がうまく推定できていないこと（知識の不確実性）」で起きているというより、「似たような特徴量でもeraや銘柄によってtargetの値がばらつきがあること（データの不確実性）」で発生する傾向がある**ということだと思います。
まあ0.01くらいが相関のスケールになっているように、このタスクはそもそも予測が難しいタスクなので当たり前なのかもしれませんが、個人的に面白いなと思いました。  
また、eraを間引いても知識の不確実性がそこまで影響していないと考察するのであれば、このモデル的にデータ量は十分なのかもしれませんね。  

### Multi targetの活用
次に試してみたこととして、複数targetを利用して学習してみましたが微妙でした。  
データの不確実性が効いているのであれば「複数targetを利用して学習したらさらによくなるんじゃね？」と思い、複数targetを単純に混ぜて学習をしてみました。固く言うと以下の仮説を立てています。  

- targetは運営が恣意的に選んだものと考え、targetがサンプルされる確率分布をモデル化する
- 複数targetでばらつきがあるデータは予測がしにくく、そのデータによってBurnがおきている

targetは種類の違い、20・60日後含めて全部利用しました。データは既に間引いているので、そのままのデータで学習しています。
以下結果です。Data Uncertaintyが微妙に下がり、Knowledge Uncertaintyが上がりました。
![](https://storage.googleapis.com/zenn-user-upload/2e23dc84357f-20221223.png)
まあ、評価が一つのtargetのみを利用しているので、複数targetで評価してみると変わるのかもしれません。

## 終わりに
今回はCatboostの２つの不確実性をもとに、eraの不確実性を算出してみましたが、他にも色々と応用できると思います。  
例えば、不確実性の高いデータ自体を使わない（Numeraiの場合は予測を中央値で埋めるなど）などはできそうですし、システムトレードはやってないので適当なことを言ってるかもですが、そもそもトレードしない判定として使えそうです。また運用以外にも、Data Uncertaintyによるタスクの難易度の確認やKnowledge Uncertaintyは学習データが不足しているかどうかを定量的に見ることもできるのでActive Learningやデータを増やす意思決定などにも使えそうです。
実は[昨年の記事](https://zenn.dev/habakan/articles/numerai-tournament-modelling-2021)で共有していたLiveデータから重み付けして学習したCatboostも中身はUncertaintyで学習していまして、もしBurnがくるようであれば「不確実性が高ければ提出しない」などの後処理を加えると良さそうだなと思っていたのですが、このモデルが幸いにもBurnが少なく今でもCorrのメインモデルではあるので後処理は実際に運用はできてないです笑。別のTCモデルも悪くはないので正直今後も観戦するだけになりそうなので、効いたらこっそり教えてください。