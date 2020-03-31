---
title: "종목분석 리포트 분석"
date: 2020-03-20 16:14
categories: projects
---

## 주가 및 종목분석 리포트 분석
## euphoria, 2019.07.31

~~~
subtitle: Stock Analysis from NAVER Finance
tags: #stock #finance #analysis #nlp #textmining
period: 2019.05.01 ~ 2019.07.31
tools: Python, Tensorflow, Ubuntu
Summary: 
   1. Data Preparation
     - NAVER 주가 크롤링: 시가,종가,고가,저가,일자,거래량
     - NAVER 금융에서 증권사 종목분석 리포트 크롤링: 텍스트,증권사,투자의견,목표주가
   2. Stock Analysis
     - 현재 주가와 증권사의 목표 주가 사이의 차이 비교
   3. Text Analysis
     - 시각화: 증권사 별 리포트의 긍부정 단어 비율
     - Doc2Vec과 t-SNE을 이용한 리포트 유사도 비교
   4. Modeling
     - 
~~~

## 1. 분석 프로젝트 요약
 - Data Science Competition 2019 from NAVER
 - 금융시장에서 데이터 과학을 이용한 투자전략 이끌어내기
 - NAVER Finance에서 주가 및 종목분석 리포트 분석하기

## 2. 분석 과정 요약

1. Data Preparation: 일 단위 주가, 증권사 종목분석 리포트
2. Analysis and Modeling
  - Doc2Vec과 t-SNE을 이용한 리포트 텍스트 유사도 비교
  - 증권사 별 리포트의 긍부정 단어 비율 분석
  - 문장 단위 문서 유사도 비교 및 예측

## 3. 분석 과정

- 분석배경: 증권사에서는 투자자를 위해 종목부석 리포트를 제공한다 . 그러나 금융 투자업계 구조 상 각 기업, 증권사 등의 이해관계가 얽혀있어 매도 의견을 자유롭게 낼 수 없다. 따라서 애널리스트의 매수추천 의견이 미래수익률과는 무관하게 발생되어 투자가치 및 신뢰성을 떨어뜨리는 결과를 낳기도 했다.

- 분석 주제: 보고서의 신뢰성, 주가정보, 텍스트 표현 등의 요소가 투자자의 결정에 영향을 미친다고 가정한다. 그리고 리포트의 텍스트와 증권사의 투자의견을 이용하여 종목 선택에 유용성에 미치는 요인들을 탐색하고, 이를 바탕으로 유용성을 지표화 하고자 한다.

- 활용 데이터: 네이버 금융에서 제공하는 증권사별 종목분석 리포트를 크롤링하여 텍스트 데이터를 추출
  - 시가 총액 상위 50개 종목 중 우선주를 제외한 48개 종목
  - 2009년 1월 1일 이후부터 2019년 6월 30일 까지의 10년 6개월의 보고서를 수집하여 총 13,508 개의 종목분석 리포트를 이용
  - 리포트의 애널리스트 투자의견의 하향 건수는 20, 중립 혹은 의견없음은 164, 나머지 13324는 매수추천 의견이었다. 또한 한달 후 주가 상승 여부에서 상승한 경우는 5116건, 내린 경우는 8392건
  
- 분석 방법
  - Dimension Reduction with AutoEncoder & Doc2Vec
  - Prediction with Logistic Regression
  

## 1. Data Preparation
 - 모든 코드는 euphoria0-0 github 내 존재
```python
def crawler(crpname, start_date='20090101', end_date='20190630'):
   '''
   1. crawling of analysis report from NAVER Finance
   2. download pdf
   3. read text of pdf
   '''
   return df
```
```python
def crawler_naverfinance_stock(itemcode, start_date, end_date):
   '''
   crawler of stock price from NAVER Finance
   '''
   return df
```
```python
if __name__ == "__main__":
   crawler(crp_list)
```

## 2. Analysis & Modeling

### 1. Doc2Vec 시각화
 - 리포트의 텍스트와 투자의견 , 그리고 한달 후 주가 등락여부 데이터를 이용 하여 Doc2Vec 알고리즘 사용 후 t SNE 시각화
 - 아래 그림은 실제 문서의 투자의견 매수 , 의견없음 , 매도 과 한달 후 주가 상승 여부 상승 , 하락
 - 실제로 서로 다른 두 의견을 가진 리포트는 Doc2Vec 의 임베딩으로는 비슷하게 나타나나 오른쪽에는 주가가 상승한 리포트들이 모였다
 - t-SNE 를 이용한 Doc2Vec 시각화
 - 그 의미는 
   (1) 리포트에 담긴 임베딩 단어의 분포는 매수리포트와 중립 혹은 매도 리포트 모두 비슷하다 . 즉 , 매수추천 의견을 가진 리포트는 중
립 혹은 매도 의견의 텍스트 어조와 크게 다르지 않다 .
   (2) 리포트의 텍스트 정보 및 투자의견이 한달 후 주가 등락 여부와는 뚜렷한 상관관계가 없다

![process](/assets/images/5_img1.png)

```python
from txt_preprocess import *
from soynlp.noun import LRNounExtractor_v2
from gensim.models import Doc2Vec

text = txt_preprocess(raw_text)
text = noun_extractor.train_extract(text)
d2v_corpus = Doc2VecCorpus(text)
d2v_model = Doc2Vec(d2v_corpus)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X[:100])
fig = plt.figure()
```
  
### 2. AutoEncoder & Classification
 - 사용한 변수 : 리포트의 투자의견 , 리포트 발간 날짜 기준 up_down, 증권사와의 베타계수 , 리포트 감정(긍/부/중), 강조 단어
 - 변수 파생 : 리포트는 약 한 달에서 두 달마다 발간된다 . 이를 기준으로 주가는 상승한 일수 , 발간일 기준 증권사와의 베타계수 , 서울대 감성
사전 기준 긍부정 , 강조단어의 빈도를 계산하였다
 - 분석 방법 : 로지스틱 회귀를 이용해 주가 상승을 설명하는 변수들을 찾아낸다. 보고서의 텍스트를 전처리하고 Auto Encoder 를 이용해 워드 임베딩의 차원을 축소시켜 투자 의견이 다른 두 문서의 유사도를 비교
 - 로지스틱 회귀 결과 , 모델 설명력이 낮았다 . 변수들은 강조어조의 수, 투자 의견이 주가가 상승한 일수에 영향을 주었다.
 - 오토인코더 결과 , 이를 시각화하여 표현하였다 . 매수 의견을 지닌 보고서의 수는 매매 의견의 수보다 25% 였으며 매매가 매수보다 더 넓게 퍼져 있다 . 문서들 간 잘 뭉쳐져 보이진 않는다
 - 오른쪽 파이차트는 각 증권사 별 베타계수를 기준으로 감성(긍부정) 추이를 표현하였다. 감성에 대한 차이를 보이진 않았다
 - 각 증권사에서 pos, neg 값의 차가 크지 않기 때문에 중요한 변수가 아님을 확인할 수 있다

![process](/assets/images/5_img2.png)

```python
text = text_cleaning()
def count_pos_neg(textdata,pn_dict=None):
   '''
   count numbers of positive/negative/neutral/intensity words
   '''
   return pos, neg, neut, intn
pos,neg,neut,intn = count_pos_neg(df_text['text'])
```

```python
from bokeh.plotting import figure, show, output_file
# using bokeh library
```



### 3. Analysis Sentence Similarity & Predict Stock Price
 - 

## 3. Conclusion
 - 본 연구의 목적은 보고서의 투자판단에의 유용성을 탐색하는 것이었다.
 - 주가 상승에 영향을 주는 보고서의 요인을 찾고자 한 로지스틱 회귀는 모델 설명력이 낮았다. 생성한 파생변수가 주가 상승 수를 설명하지 못했다. 오토인코더는 input data의 30% 정도만 재현했다. 
  - 그 이유는 첫째, 텍스트 분석 시 시퀀스가 아닌 단일 단어만을 고려하여 빈도를 계산하였다. 
  - 둘째 금융 도메인인 텍스트만의 특성을 고려할 필요가 있다.
  - 마지막으로 본 연구에서 유용성이라는 없는 지표를 찾으면서 옳은 모델인지 확인하기 어려웠다.
 - 따라서, 보고서의 텍스트만으로는 보고서의 투자 유용성을 분석하기엔 부족했다
 

