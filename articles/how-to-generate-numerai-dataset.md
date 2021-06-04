---
title: "Numeraiトーナメントのデータを構築過程から分析してみる"
emoji: "💭"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: [numerai]
published: true
---

# はじめに
この記事ではNumeraiトーナメントが提供しているデータセットに関するEDAをまとめています。  
僕は個人的にモデリングして良いスコアを出すこともモチベーションにありますが、プロダクトとしてトーナメントの構造自体にとても興味があります。
最初はKaggleの匿名化データのコンペティションと同様に、データに対する深いが理解が難しいのではないかと思っていたのですが、データの設計過程から考えて分析をすることで思った以上に考察ができ、EDAがとても楽しかったです。  
また、僕はスタートアップで機械学習を使ったプロダクトを開発しているエンジニアですが、金融ドメインなどは片手間にやっているNumeraiを通して勉強している程度なので、今回の分析は推測が多分に含まれています。
そこを含めた指摘や考察などはご意見いただけますと幸いです！  

# データの構築方法
## 運営側はどのようにデータを用意しているか？  
データを実際に分析する前に、そのデータがどのように用意されているか大枠だけ整理しておきましょう。  
運営側が明言しているわけではないですが、おそらくこのようなフローでデータが提供されています。  
1. 株価に関連するデータの収集（株価、財務諸表 etc...）
2. データを元にした特徴抽出（テクニカル指標 etc...）
3. **難読化**  

他のデータコンペティションと比較して特徴的なところは**難読化**だと思います。  
こちらの難読化はビジネス上の要因として発生する処理だと思いますが、難読化をする理由をいくつかあげてみます。  

#### 難読化理由(1) Numeraiのデータとして知財を守りたい 

参加している方ならわかるかと思いますが、**Numeraiのデータセットはそもそも優秀**で、特徴量をそのまま利用するチュートリアルのスクリプトでさえかなりのパフォーマンスが出ています。  
そのようなデータを知財として守るというのは開催する上で重要な要件でしょう。  

#### 難読化理由(2) メタモデルに寄与する予測の収集が第一優先

運営側のモチベーションは「**メタモデルに寄与する予測を集める**」ことです。  
なのでデータを理解しやすくしても、優秀なデータサイエンティストの参加はあまり増えないという判断だと思います。
運営側としても一応QAで似たようなニュアンスのことを書かれています。  

> **Why so vague about what is being predicted?** 
>
> Numerai purposely releases very little information about the data in order to maintain the integrity of the tournament. Additionally, the only way to provide hedge-fund quality data for free is to ensure that the data is encrypted and obfuscated. Providing participants with more information only introduces bias and potentially decreases the quality of the predictions submitted.
>
> https://docs.numer.ai/tournament/faq

というのが僕が推測する理由ですが、難読化はデータサイエンティスト側としては若干相反しているところでもあります。  
データの理解を深めることで良い予測を作成でき、メタモデルに貢献できると個人的には思っています。
Prado氏なども**難読化**と明言しているあたり、匿名化ではないのでまだ分析をする余地はあると思い、EDAを始めています。

# データ分析
ここから実際に分析結果をまとめていきます。  
以下の記事などがとても勉強になりましたので、こちらを踏まえての分析をしています。

- https://qiita.com/blog_UKI/items/fb401725288e58c92bd6
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3478927
- https://speakerdeck.com/yohrn/an-encouragement-of-numerai

## (1) レコード分析
レコードはある銘柄を表しており、どの時系列に対する銘柄情報かはeraで表現されています。  
上記の記事の中には、idは日時と銘柄名によって生成された識別子であると考察しているものもあります。  
まずは、レコード自体がどのような構造でどのように作成されたかを分析します。

### era内の銘柄に重複はあるか？
こちらがeraごとのレコード数の推移です。

![](https://storage.googleapis.com/zenn-user-upload/cf346cb2cef50fe0bebf8cac.png)
レコード数の推移は定常的ではなく、連続した変化が見られます。  
また、このeraはtraining, validationはMonthlyでtest, liveはWeeklyです。  
ここでまず、Monthlyのeraというのは、ある月の複数Weeklyのレコードが入っているのか、ある月の１時点のレコードが入っているのか分析します。    
仮に複数のWeeklyのレコードが入っている場合以下のようなデータになることが考えられます。  

- 週次に関連する特徴量が入っている可能性がある
- era内に同一銘柄のレコードが存在する
- 特定eraのレコード数は欠損などがない限りは4, 5の倍数  

分析前の時点では、銘柄の重複をなくすために１時点のみだと考えていましたが、いくつかの記事を見ると「IDは銘柄と日時を表した識別子」という記述があったので、表現によってはありえるかと思ったので一応、調査分析しました。

-  [Numerai はいいぞ / An encouragement of Numerai - Speaker Deck p.5](https://speakerdeck.com/yohrn/an-encouragement-of-numerai?slide=5 ) 

- https://qiita.com/blog_UKI/items/fb401725288e58c92bd6#データセット  

  まずは上記の3番目に記したレコード数が4, 5の倍数に該当するeraの比率を調べると以下の結果になります。  

```python
mod4 = (training_data.groupby('era').size() % 4 == 0).values
mod5 = (training_data.groupby('era').size() % 5 == 0).values
print(sum(mod4 | mod5) / len(mod4))
# 0.35
```
こちらを見ると3割程度で、半分以上のeraは規則にしたがってなさそうです。  
次にvalidation, live, testのそれぞれの銘柄数の推移を見ても銘柄数のオーダーはすべて同じです。  
以上のことを考えると、era内に銘柄の重複はないと個人的には考えています。  


### 推移に関する分析
次にtraining dataの各eraがどのような時期に対応するのかはわかりませんが、ひとまず年次の世界上場会社数と比較してみます。  

![](https://storage.googleapis.com/zenn-user-upload/78dff9c710eec9dfdfae6836.png)numeraiのレコード数と比較するとオーダー数が違いますが、全体的に増加しているのはどちらにも言えそうです。  
ここで少なくともいえるのはオーダーが違うところから、**era内の銘柄は何かしらの方法でユニバース選定をしている**ということです。  

ここから、推移を元に各eraがどの時点を表し、どのような銘柄・セクターが入っているかを２つのグラフを比較して検証してみましたが、残念ながら明確に理解するところまではいきませんでした。  

### Signalsとの銘柄数の一致

他にユニバースを特定する方法として、Signalsとの関係性があるかを調べたところ実は**Numeraiトーナメントのliveレコード数とNumerai Signalsのuniverse数が一致している**ことがわかりました。  

```python
import pandas as pd

tournament_data = pd.read_csv('https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz')
signals_df = pd.read_csv('https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/universe/latest.csv')

print('tournament live data: ', (tournament_data.data_type == 'live').sum())
print('signals universe: ', len(signals_df))

# tournament live data: 5414
# signals universe: 5414
```

運営的にもユニバースを変更する理由がないと思うので、個人的にはSignalsのユニバースと一緒なのではないかと思います。
なので、Signalsのユニバースを参考にクラスタリングなどで、ユニバースの選定をモデリングに組み込むなどもできるかもしれません。

## (2) 特徴量の分析

次に特徴量について分析をします。

特徴量は匿名化されていて、それぞれの意味合いが何かを理解するにはドメインの知識がかなり必要になります。

僕自身はドメインの知識はほとんどないので、特徴量自身のドメイン的な考察はせず、今回はグルーピングなどを活用した統計量を中心に分析をしています。

### 分位化

targetを含めた特徴量は[0.0, 0.25, 0.5, 0.75, 1.0]の要素で構成されています。  

これは分位化（ビニング処理）によるものであると考えられます。

pandasでは分位化前のデータフレームを`raw_df`とした時に以下の処理で分位化ができます。

```python
pd.qcut(raw_df, 5, labels=False) / 4
```

ポイントとしてはこの分位化をNumeraiではどのように行なっているかです。
以下の選択肢があると考えられます。

- すべてのデータをまとめて分位化
- eraごとに分位化

個人的にはeraによる確率分布の変化（Concept Drift）を考慮するために、eraごとに行なっていると考えます。

どちらかを確かめるために統計量を分析をします。
すべての統計量を出すと大変なので、それぞれの特徴量の１つ目の統計量を見てみます。

```python
cols = ["feature_intelligence1", "feature_dexterity1", "feature_charisma1", "feature_strength1", "feature_constitution1", "feature_wisdom1"]
training_data[cols].describe()
```

| statistics | feature_intelligence1 | feature_dexterity1 | feature_charisma1 | feature_strength1 | feature_constitution1 | feature_wisdom1 |
| ---------- | --------------------: | -----------------: | ----------------: | ----------------: | --------------------: | --------------- |
| count      |                501808 |             501808 |            501808 |            501808 |                501808 | 501808          |
| mean       |              0.499981 |           0.499976 |          0.499968 |          0.499978 |              0.499968 | 0.499981        |
| std        |              0.353596 |           0.352544 |          0.352986 |          0.352650 |              0.342626 | 0.353596        |

次に、ある確率分布から同じ数をサンプルし、そのデータに対して分位化すると同じ統計量が獲得できるかを検証します。

今回は簡単に正規分布と一様分布からサンプルしたデータを利用し、それに対して分位化してみます。

```python
sample_df = pd.DataFrame()
sample_df['era'] = training_data['era']
sample_df['norm'] = np.random.normal(0, 1, len(sample_df))
sample_df['norm_quantile'] = pd.qcut(sample_df['norm'], 5, labels=False) / 4
sample_df['uniform'] = np.random.uniform(0, 1, len(sample_df))
sample_df['uniform_quantile'] = pd.qcut(sample_df['uniform'], 5, labels=False) / 4
sample_df.describe()
```

| statistics | norm          | norm_quantile | uniform      | uniform_quantile |
| ---------- | ------------- | ------------- | ------------ | ---------------- |
| count      | 501808.000000 | 501808.000000 | 5.018080e+05 | 501808.000000    |
| mean       | -0.000170     | 0.500000      | 5.000475e-01 | 0.500000         |
| std        | 1.001453      | 0.353554      | 2.885991e-01 | 0.353554         |

結果を見ると平均が0.5ちょうどになっているのに対し、Numeraiのデータは端数が入っているのがわかります。

これに対して、乱数データをeraごとに分位化したものを確認してみます。

```python
col_name = 'norm'
new_col = col_name + '_era_quantile'
for era in sample_df['era'].unique():
    sample_df.loc[sample_df['era'] == era, new_col] = pd.qcut(sample_df.loc[sample_df['era'] == era, col_name], 5, labels=False) / 4

col_name = 'uniform'
new_col = col_name + '_era_quantile'
for era in sample_df['era'].unique():
    sample_df.loc[sample_df['era'] == era, new_col] = pd.qcut(sample_df.loc[sample_df['era'] == era, col_name], 5, labels=False) / 4
    
sample_df.describe()
```

| statistics | norm          | norm_quantile | norm_era_quantile | uniform      | uniform_quantile | uniform_era_quantile |
| ---------- | ------------- | ------------- | ----------------: | ------------ | ---------------- | -------------------: |
| count      | 501808.000000 | 501808.000000 |     501808.000000 | 5.018080e+05 | 501808.000000    |        501808.000000 |
| mean       | -0.000170     | 0.500000      |          0.499981 | 5.000475e-01 | 0.500000         |             0.499981 |
| std        | 1.001453      | 0.353554      |          0.353596 | 2.885991e-01 | 0.353554         |             0.353596 |

平均だけで言うと、eraごとに分位化したほうが特徴量に近いものになっています。
例えば`feature_intelligence1`と端数まで一致しています。

次にeraごとの平均の推移を見てみます。こちらは`feature_intelligence1`の推移です。

```python
training_data.groupby('era')['feature_intelligence1'].mean().plot()
plt.show()
```

![](https://storage.googleapis.com/zenn-user-upload/86ab140d5e4415b1d2f282c1.png)

こちらと`norm_quantile`, `norm_era_quantile`の推移を比較してみます。

```python
plt.figure(figsize=(10, 5))
df.groupby('era')['norm_quantile'].mean().plot(label="norm_quantile")
df.groupby('era')['norm_era_quantile'].mean().plot(label="norm_era_quantile")
plt.legend()
plt.show()
```

![](https://storage.googleapis.com/zenn-user-upload/be0d5ca6530ce4c9d70e3b4e.png)

まあ当然と言ってしまえば当然ですが、全体に対して分位化したものは平均推移のばらつきが大きくなっています。
それに対して、`feature_intelligence1`,`norm_era_quantile`,`uniform_era_quantile`を並べると推移が一致しています。

```python
plt.figure(figsize=(10, 5))
training_data.groupby('era')['feature_intelligence1'].mean().plot(label="feature_intelligence1")
df.groupby('era')['norm_era_quantile'].mean().plot(label="norm_era_quantile")
df.groupby('era')['uniform_era_quantile'].mean().plot(label="uniform_era_quantile")
plt.legend()
plt.show()
```

![](https://storage.googleapis.com/zenn-user-upload/fe72fbc93c736820ea4466ec.png)

ちなみに`feature_dexterity1`と比較すると推移としては似ていますが、完全一致していません。  
任意の分布について一致するわけではないのかもしれません。

![](https://storage.googleapis.com/zenn-user-upload/526a486de65688cdc82747e0.png)

### 非定常なデータでの分位化

ここまでは、同一の分布からデータをサンプルして検証してきましたが、時系列データとしては強い仮定になると思うので、非定常な分布(i.i.dではない)からサンプルされたデータでも検証してみました。

ここでは、各eraのデータ数を平均とした正規分布からサンプルすることを考えてみます。

```python
for era in sample_df['era'].unique():
    num_record = len(sample_df.loc[sample_df['era'] == era, :])
    sample_df.loc[sample_df['era'] == era, 'nonstationary'] = np.random.normal(num_record, len(sample_df)/num_record, num_record)
sample_df['nonstationary_quantile'] = pd.qcut(sample_df['nonstationary'], 5, labels=False) / 4
col_name = 'nonstationary'
new_col = 'nonstationary_era_quantile'
for era in sample_df['era'].unique():
    sample_df.loc[sample_df['era'] == era, new_col] = pd.qcut(sample_df.loc[sample_df['era'] == era, col_name], 5, labels=False) / 4
```

![](https://storage.googleapis.com/zenn-user-upload/ac7f476b01cd22cc75cf749e.png)

やはり、非定常なデータだと全体での分位化をしてしまうと、eraの分布変化が出てしまっていますね。

分位化によって、**各特徴量の単一の統計量の変化を少なくする**ことができています。

###  分位化による情報量の影響　

分位化により、分布の差を少なくしていることはわかりましたが、同時に情報量も少なくなっているのでそれについても分析してみます。

各特徴量は５種類の数値のみで構成されているので、重複した特徴量を持つレコードがあるかどうか分析してみます。

すべての特徴量が重複しているレコードは一つもなかったです。

ちなみに学習データの数501808の中で、１レコードでも重複する確率は誕生日のパラドックスの問題として重複する確率を計算することができ、その近似値を計算しましたが、0.0でした（オーダーが大きいので近似計算が間違っている可能性があります）。
恣意的に重複をなくしているわけではなさそうです。

```python
import numpy as np

'''
近似計算はこちらを参考
https://azapen6.hatenablog.com/entry/2013/04/30/234806
'''
n = 5 ** 310
m = 501808
p = 1 - np.exp(- (m * (m - 1 )) / (2 * n))
print(p)
```


次に特徴量名のグループごとに分けて重複するかどうか検証してみます。

しかし、そのままグループごとの重複量を確認しても、グループ内の特徴量数が異なるためそれを考慮します。

1. 各グループでランダムに10個の特徴量を選択
2. 選択した特徴量を元に重複したレコードを削除
3. この試行を100回繰り返し、平均レコード数を計算

|                | intelligence | dexterity | charisma  | strength  | constitution | wisdom    |
| -------------- | ------------ | --------- | --------- | --------- | ------------ | --------- |
| 特徴量数       | 12           | 14        | 86        | 38        | 114          | 36        |
| 平均レコード数 | 199342.91    | 132807.53 | 349578.37 | 259864.82 | 261146.96    | 235495.79 |

`dexterity`は特徴量数のより少ない`intelligence`と比較しても、重複が多そうです。
騰落系指標があると考察されていることから、銘柄固有やセクターに依存せず騰落系が似たレコードがあるということでしょうか。
`constitution`も特徴量数が多いのに対し、重複が多そうです。
セクター情報が入っているという点から重複がおきているとも考えられます。

eraごとの重複の推移をみてみます。
それぞれのグループの1~10の特徴を元にeraごとの重複数を計算してみます。

```python
def get_feature(name):
  return  [col for col in training_data.columns if name in col]

names = ["dexterity", "constitution", "intelligence", "charisma", "strength", "wisdom"]

fig, axes= plt.subplots(2,3, figsize=(20, 10))
for i, name in enumerate(names):
  cols = get_feature(name)[:10]
  print(cols)
  n_duplicates = [len(training_data.loc[training_data.era == era, cols]) - len(training_data.loc[training_data.era == era, cols].drop_duplicates()) for era in range(1, 121)]
  axes[i // 3, i % 3].set_title('num of duplicates by era: '+name)
  axes[i // 3, i % 3].plot(n_duplicates)
plt.show()
```

![](https://storage.googleapis.com/zenn-user-upload/cb37ad2399df7564ac896f11.png)

`dexterity`, `intelligence`, `wisdom`は全体的に増加をしているように見えますが、これは銘柄数の増加に伴う影響と考えられます。
逆にそれ以外の`constitution`, `charisma`, `strength`は影響しにくい特徴量と言えそうですね。
また、全体的にera80付近で重複数が減少しているのも何か理由がありそうで、まだ考察の余地があります。

### 特徴量が離散値（カテゴリカル）ではない

データをいじることに注力をしていましたが、全て分位化されているとするとどれも元は連続値の可能性が高いです。
しかし、銘柄データには国やセクター情報などのカテゴリカルな情報も存在します。
ここの分析はまだしきれていませんが何かしらの方法でカテゴリをエンコーディングしていると考えられます。
考察段階ですが仮説を列挙しておきます。

- セクターごとの前月平均リターン（Target Encodingに近い）
- セクター内の銘柄数（Count Encoding）
- セクターでのGrouby特徴（セクターごとの平均PERなど）
- 逆にPCAなどは使っていない可能性が高い
  - PCAを利用している場合は各変数が直交になるはずだが、相関が高い

## (3) targetの分析

最後にtargetについて、分析してみます。
targetは`nomi`と`kazutsugi`の2種類ありました。targetも数値が5種類なところもあり、分位化されていそうです。
2つのtargetをplotしてみます。

![](https://storage.googleapis.com/zenn-user-upload/0ac8a194f014c6988882a74e.png)

これを見ると、`kazutsugi`は一様分布、`nomi`は正規分布っぽくなっています。
個人的にはこちらは分位点の設定を変更することで、分位化後の分布を正規分布に変更していると思っています。
`kazutsugi`と`nomi`のような2種類のtargetを作成できるか検討してみます。
トーナメントの銘柄がSIgnalsのユニバースと一緒で、分位化により作成されていることを前提でyfinanceで取得したデータから作成してみます。
価格からsingalsのtargetに近い値を作成する方法はこちらのForumで議論しています。

- https://forum.numer.ai/t/decoding-the-signals-target/2501

ただ再現はできておらず、提供されているtargetとはまだ程遠い状況ですが、元のデータはほぼこちらを利用しているだろういうことと、今回は`nomi`と`kazutsugi`の関係性を再現することが目的なので、こちらのコードを参考に生成してみました。
コアとなるコードはこちらです。

```python
# pricesはyfinanceよりダウンロードした価格DataFrame
# トーナメントでは一ヶ月後なので、それに対応するリターンを計算
period_returns = prices.pct_change(19)

# リターンをランク化
period_ranks = period_returns.rank(axis=1, pct=True, method="first")

# 一様分布に近くなるような分位化
period_ranks_bins_uniform = pd.cut(
    period_ranks.stack(),
    5,
    right=True,
    labels=[0, 0.25, 0.50, 0.75, 1],
    include_lowest=True
).reset_index()

# 正規分布に近くなるような分位化
period_ranks_bins_norm = pd.cut(
    period_ranks.stack(),
    bins=[0, 0.05, 0.25, 0.75, 0.95, 1],
    right=True,
    labels=[0, 0.25, 0.50, 0.75, 1],
    include_lowest=True
).reset_index()
```

こちらを元にplotしてみた結果がこちらです。

![](https://storage.googleapis.com/zenn-user-upload/7564dd043f2685181aebfe56.png)

各targetの分布はそれぞれ一様分布、正規分布になりましたが、一様分布で0.5になっているデータが正規分布でも0.5のみとなっています。
`nomi`と`kazutsugi`ではここは0.5のみではなく、むしろ0.0や1.0の値になっているデータも存在するので、分位化以外で何か処理がある or 異なっている可能性があります。

# 感想

普段から機械学習を活用したシステムトレードをやっている方々には当たり前なのかもしれませんが、こうして特徴量の設計視点から分析をすることで株価というデータのドメイン知識とそれに対するモデリング、評価方法を深く理解することができて非常に面白かったです。
最近はモデルを変えすぎて成績が落ちていますが、今回のようにSignalsとトーナメントの知見を相互に活かしながら、引き続き自分のペースで分析、モデリングできればと思っています！