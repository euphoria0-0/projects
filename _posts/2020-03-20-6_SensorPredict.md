---
title: "SDS: 제품 불량률 예측"
date: 2020-03-20 16:15
categories: projects
comments: true
---

## 제품 불량률 예측
## euphoria , 2019.08.31

~~~
subtitle: Quality Defective Prediction using Time Series Sensor Data
tags: #SensorData #Prediction #DeepLearning #TimeSeries
period: 2019.07.01 ~ 2019.08.31
tools: Python, Tensorflow, sklearn
Summary:
   1. EDA
   2. Dimension Reduction
   3. 1st Model
      - LGBM
      - XGBoost
   4. 2nd Model
      - BiLSTM
   5. Results
~~~

## 1. 분석 프로젝트 요약
 - Samsung SDS Brightics AI Academy 공모전: 데이터 분석
 - 제품을 생산하는 과정에서 발생하는 설비의 센서 측정 데이터와 해당 제품의 불량률 사이의 유의미한 연관성 분석 및 불량률 예측

## 2. 분석 과정 요약

![process](/assets/images/6_process.png)

#### 1. EDA

- Outlier
  - Tukey's method & boxplot으로 상위 0.5% 제거
- 85개의 센서 변수간 다중공선성
  - 변수 선택
  - 차원 축소
  -> 모든 센서가 유의미한 역할을 가진다고 가정하고 , 85 개의 센서 데이터에 대하여 차원 축소를 수행
  
#### 2. Model

(1) Dimension Reduction
 
  - kernel PCA를 이용하여 k=5의 주성분 생성
  - GridSearchCV를 이용한 최적 파라미터 선정
  
(2) xgb, lgbm으로 예측
 
  - GridSearchCV를 이용한 최적 파라미터 선정
  - 평균으로 앙상블
  -> WMAE 1.29
    
(3) 시계열 특성 이용을 위한 LSTM 모형
 
  - Bi-directed LSTM
  - CallBacks
  -> WMAE 0.94
  
#### 3. Prediction

- Final Model: Bi-Directional LSTM model
- 평가지표: 검증 데이터의 각 제품별 불량률에 대한 가중평균절대오차 WMAE:Weighted Mean Absolute Error) 측정값을 산출
 
#### 4. Comments

- Good
  - 가장 좋은 성능을 내는 모델 하나를 선택하기 보다 여러 모델을 Ensemble 한 모델을 선택하여 보다 모델의 bias 를 보완하고 딥러닝의 오버피팅 문제를 해소한다
  - 차원축소를 통해 85 개의 센서데이터가 설명하는 양 분산량 을 주성분을 통해 나타낼 수 있어 단순하면서 재사용가능한 알고리즘이다
- Bad
  - Deep Learning 모델이 복잡해짐에 따라 학습시간이 크게 증가한다. 그 때문에 적합한 모델을 찾고 Hyper Parameter 를 찾는데 Grid Search 를 사용하였고 많은 시간을 소요하였다
  - 차원축소 기법에서는 시계열을 고려하지 않았다
- Further work
  - 모델의 하이퍼파라미터를 찾는 데에 베이지안 최적화 방법을 이용하는 등 빠른 파라미터 서치가 필요하다
  - 투영 차원축소를 이용하였고 , 더 높은 성능을 확인할 수 있다면 매니폴드 방법도 시도할 수 있다
  - BiLSTM 모델의 성능을 
 
 
## 3. 분석 과정

## 1. 라이브러리 및 데이터
### 1-1. Import libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.scorer import make_scorer
from sklearn.preprocessing import LabelEncoder, scale
from sklearn import utils
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.manifold import LocallyLinearEmbedding

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import warnings
warnings.filterwarnings("ignore")
```

### 1-2. Load data

```python
raw_data = pd.read_csv('data/qualityData_train.csv')
data = raw_data.copy()
print(data.shape)
data.head(3)
```

    (3507, 87)
    




<div>


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time_Sequence</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>x5</th>
      <th>x6</th>
      <th>x7</th>
      <th>x8</th>
      <th>x9</th>
      <th>...</th>
      <th>x77</th>
      <th>x78</th>
      <th>x79</th>
      <th>x80</th>
      <th>x81</th>
      <th>x82</th>
      <th>x83</th>
      <th>x84</th>
      <th>x85</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.028837</td>
      <td>-0.649824</td>
      <td>0.363124</td>
      <td>-0.752145</td>
      <td>-0.631316</td>
      <td>-0.588815</td>
      <td>-0.904200</td>
      <td>-0.577024</td>
      <td>0.715939</td>
      <td>...</td>
      <td>1.394348</td>
      <td>1.350135</td>
      <td>1.774360</td>
      <td>1.741823</td>
      <td>1.705609</td>
      <td>-1.175603</td>
      <td>0.705534</td>
      <td>1.350135</td>
      <td>0.625413</td>
      <td>2.185269</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.053882</td>
      <td>-0.649824</td>
      <td>0.347376</td>
      <td>-0.499595</td>
      <td>-0.528668</td>
      <td>-0.487181</td>
      <td>-0.904200</td>
      <td>-0.576967</td>
      <td>0.702009</td>
      <td>...</td>
      <td>1.277810</td>
      <td>1.404929</td>
      <td>1.701051</td>
      <td>1.469954</td>
      <td>1.705365</td>
      <td>-2.588321</td>
      <td>0.810728</td>
      <td>1.404929</td>
      <td>0.050509</td>
      <td>1.572313</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.065778</td>
      <td>-0.649824</td>
      <td>0.347376</td>
      <td>-0.732795</td>
      <td>-0.586186</td>
      <td>-0.578574</td>
      <td>-0.904200</td>
      <td>-0.577844</td>
      <td>0.703790</td>
      <td>...</td>
      <td>1.428878</td>
      <td>1.573875</td>
      <td>1.758069</td>
      <td>1.741823</td>
      <td>1.705609</td>
      <td>-1.371211</td>
      <td>0.747611</td>
      <td>1.573875</td>
      <td>0.625413</td>
      <td>2.581797</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 87 columns</p>
</div>


## 2. 데이터 전처리

## 2-1. Outlier Detection

- 상위 0.5% 제거


```python
plt.figure(figsize=(12, 5)) 
plt.subplot(1,4,1); plt.boxplot(data['Y'], sym="bo")
plt.subplot(1,4,2); plt.boxplot(data[data.Y < data.Y.quantile(0.995)]['Y'], sym="bo")
plt.subplot(1,4,3); plt.boxplot(np.log1p(data['Y']), sym="bo")
plt.subplot(1,4,4); plt.boxplot(np.log1p(data[data.Y < data.Y.quantile(0.995)]['Y']), sym="bo")
plt.show()
```


![png](/assets/images/6_OutputImage1.png)



```python
data = data[data.Y < data.Y.quantile(0.995)]
data.reset_index(drop=True, inplace=True)
data.shape
```




    (3489, 87)



### 2-2. Train / Validation data

- 80% : 20%, RANDOM_SEED = 41


```python
RANDOM_SEED = 41
```


```python
X_col = data.columns[1:86]
X_train = data[data.columns[:86]]
X0_train = data[X_col]
y_train = data['Y']
print(X_train.shape, X0_train.shape, y_train.shape)
```

    (3489, 86) (3489, 85) (3489,)
    

## 3. Dimension Reduction

### 3-1. kernel PCA
 
 - 최적 파라미터 찾기


```python
def wmae(y, y_pred):
    return sum((y/sum(y))*(np.abs(y-y_pred)))
```


```python
def GridSearch_kPC(n, X, y):
    clf = Pipeline([
            ("kpca", KernelPCA(random_state=RANDOM_SEED, fit_inverse_transform=True, n_jobs=1)),
            ("reg", LinearRegression())
        ])

    param_grid = [{
            "kpca__n_components": [n],
            "kpca__gamma": np.linspace(0.01, 0.99, 99),
            "kpca__kernel": ["rbf"] #, "sigmoid", "linear", "poly", "cosine"]
        }]

    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=make_scorer(wmae, greater_is_better=False))

    grid_search.fit(X, y)

    pm = grid_search.best_params_
    print(pm)

    rbf_pca = KernelPCA(n_components = pm['kpca__n_components'], kernel=pm['kpca__kernel'], gamma=pm['kpca__gamma'], 
                        fit_inverse_transform=True, n_jobs = 1, random_state = RANDOM_SEED)
    X_kpc = rbf_pca.fit_transform(X)
    X_kpc_inv = rbf_pca.inverse_transform(X_kpc)
    print('WMAE: ', wmae(X, X_kpc_inv))
```


```python
GridSearch_kPC(3, X0_train, y_train)
```

### 3-2. 최종 X (PC)

 - Principal Components 로 new X 생성


```python
mse = []
for n in range(2,21):
    rbf_pca = KernelPCA(n_components = n, kernel="rbf", gamma=0.012, fit_inverse_transform=True)
    X_pc = rbf_pca.fit_transform(X0_train)
    X_pc_inv = rbf_pca.inverse_transform(X_pc)
    mse.append(mean_squared_error(X0_train, X_pc_inv))
```


```python
n = np.arange(21)
mse_list = [np.nan, np.nan] + mse
plt.plot(n,mse_list)
```




    [<matplotlib.lines.Line2D at 0x2959ad06c50>]




![png](/assets/images/6_OutputImage2.png)



```python
rbf_pca = KernelPCA(n_components = 5, kernel="rbf", gamma=0.274, fit_inverse_transform=True)
X_pc = rbf_pca.fit_transform(X0_train)
X_pc_inv = rbf_pca.inverse_transform(X_pc)
mean_squared_error(X0_train, X_pc_inv)
```




    0.3955572634513758




```python
'''rbf_pca = KernelPCA(n_components = 10, kernel="rbf", gamma=0.011, fit_inverse_transform=True)
X_pc = rbf_pca.fit_transform(X0_train)
X_pc_inv = rbf_pca.inverse_transform(X_pc)
mean_squared_error(X0_train, X_pc_inv)'''
```




    0.10530485041170284

## 4. 변수 선택 및 모델 구축
### 4-1. Data Preparation


```python
X_train = pd.concat([data['Time_Sequence'], pd.DataFrame(X_pc_inv)], axis=1)
X_train.shape
```




    (3489, 86)




```python
y_train = data['Y']
y_train.shape
```




    (3489,)



### 4-2. Prediction


```python
def get_cv_score(models, X, y):
    kfold = KFold(n_splits=5, random_state=RANDOM_SEED).get_n_splits(X) # X.values
    for m in models:
        print("Model {} CV score : {:.4f}".format(m['name'], np.mean(cross_val_score(m['model'], X, y)), kf=kfold))
        
def print_best_params(model, params, X, y, wmae=True):
    if wmae == True:
        scorer = make_scorer(wmae, greater_is_better=False)
    else:
        scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    
    grid_model = GridSearchCV(
        model, 
        param_grid = params,
        cv=5,
        scoring= scorer)
    
    grid_model.fit(X, y)
    rmse = np.sqrt(-1*grid_model.best_score_)
    print('{0} 5 CV 시 \n 최적 평균 RMSE 값 {1} \n 최적 alpha:{2} \n '.format(model.__class__.__name__, np.round(rmse, 4), grid_model.best_params_))
    return grid_model.best_estimator_
```


```python
xgb_params ={
    'max_depth':np.arange(10,100,10),
    'n_estimators':np.arange(100,600,100)
}

xgb_model = xgb.XGBRegressor(seed=RANDOM_SEED, eta=0.01, max_depth=20, subsample=0.8, colsample_bytree=0.5, silent=True, 
                             gpu_id=0,predictor='gpu_predictor', refit=True, learning_rate=0.05, n_jobs=2)
xgb_estimator = print_best_params(xgb_model, xgb_params, X_train, y_train, wmae=False)
```

    XGBRegressor 5 CV 시 
     최적 평균 RMSE 값 0.964 
     최적 alpha:{'max_depth': 10, 'n_estimators': 100} 
     
    


```python
xgb_params ={
    'n_estimators':np.arange(10,100,5)
}

xgb_model = xgb.XGBRegressor(seed=RANDOM_SEED, eta=0.01, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, silent=True, 
                             gpu_id=0,predictor='gpu_predictor', refit=True)
xgb_estimator = print_best_params(xgb_model, xgb_params, X_train, y_train, wmae=False)
```

    XGBRegressor 5 CV 시 
     최적 평균 RMSE 값 0.964 
     최적 alpha:{'max_depth': 10, 'n_estimators': 10} 
     
    


```python
lgb_params = {
#    'num_leave' : np.arange(1,11),
#    'learning_rate' : [0.05, 0.01, 0.005],
    'n_estimators':np.arange(100,600,100)
#    'max_bin' : np.arange(10,100,10)
}

lgb_model = lgb.LGBMRegressor(seed=RANDOM_SEED, objective='regression', gpu_id=0, tree_method='gpu_hist',
                              predictor='gpu_predictor',refit=True, num_leave=10, learning_rate=0.01, max_bin=20)
lgb_estimator = print_best_params(lgb_model, lgb_params, X_train, y_train, wmae=False)
```

    LGBMRegressor 5 CV 시 
     최적 평균 RMSE 값 0.5686 
     최적 alpha:{'n_estimators': 100} 
     
    


```python
lgb_params = {
    'num_leave' : np.arange(1,6),
    'n_estimators':np.arange(300,400,10),
    'max_bin' : np.arange(10,30,5)
}

lgb_model = lgb.LGBMRegressor(seed=RANDOM_SEED, learning_rate=0.005,objective='regression', gpu_id=0, tree_method='gpu_hist', predictor='gpu_predictor',refit=True)
lgb_estimator = print_best_params(lgb_model, lgb_params, X_train, y_train, wmae=False)
```

    LGBMRegressor 5 CV 시 
     최적 평균 RMSE 값 0.5241 
     최적 alpha:{'max_bin': 20, 'n_estimators': 390, 'num_leave': 1} 
     
    


```python
xgb_model = xgb.XGBRegressor(seed=RANDOM_SEED, eta=0.01, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, silent=True, 
                             gpu_id=0,predictor='gpu_predictor', refit=True, gamma=0.0001, eval_metric='rmse', max_depth=10,
                            n_estimators=100, reg_alpha=0.0001)

lgb_model = lgb.LGBMRegressor(seed=RANDOM_SEED, learning_rate=0.005,objective='regression', gpu_id=0, tree_method='gpu_hist', 
                              predictor='gpu_predictor',refit=True, max_bin=20, n_estimators=100, num_leave=10, boosting='gbdt',
                             metric='rmse', bagging_fraction=0.9, bagging_freq=5, feature_fraction=0.8, feature_fraction_seed=RANDOM_SEED,
                             bagging_seed=RANDOM_SEED, min_data_in_leaf=10, min_sum_hessian_in_leaf=30)
```


```python
def get_cv_score(models, X, y):
    kfold = KFold(n_splits=5, random_state=41).get_n_splits(X) 
    for m in models:
        print("Model {} CV score : {:.4f}".format(m['name'], np.mean(cross_val_score(m['model'], X, y)), kf=kfold))
        
def AveragingBlending(models, X, y, sub_X):
    for m in models : 
        m['model'].fit(X, y)
    
    predictions = np.column_stack([
        m['model'].predict(sub_X) for m in models
    ])
    return np.mean(predictions, axis=1)
```


```python
'''cbr_params = {
    'num_leaves' : np.arange(1,11),
    'max_depth':np.arange(10,100,10),
    'n_estimators':np.arange(10,100,10),
    'max_bin' : np.arange(10,100,10)
}

cbr_model = cb.CatBoostRegressor(random_state=RANDOM_SEED, logging_level='Silent', early_stopping_rounds=300, learning_rate=0.01)
cbr_estimator = print_best_params(cbr_model, cbr_params, X_train_lle, y_train, wmae=False)'''
```

### 4-3. Ensemble Modeling

```python
# Average Blending
models = [{'model':xgb_model, 'name':'XGBoost'}, 
          {'model':lgb_model, 'name':'LightGBM'}]

## train test
get_cv_score(models, X_train, y_train)
```

    Model XGBoost CV score : -2.8592
    Model LightGBM CV score : -0.0328
    

```python
df = pd.concat([X_train, y_train], axis=1)

df_train, df_val = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
print(df_train.shape, df_val.shape)

df_X_train = df_train[df_train.columns[:-1]]
df_Y_train = df_train['Y']
df_X_val = df_val[df_val.columns[:-1]]
df_Y_val = df_val['Y']
```

    (2791, 87) (698, 87)
    
## 5. 모델 학습 및 검증

```python
## validation test
y_val_pred = AveragingBlending(models, df_X_train, df_Y_train, df_X_val)
# 가중평균절대오차
print(sum((df_Y_val/sum(df_Y_val))*(np.abs(df_Y_val-y_val_pred))))
# MSE
print(np.mean((df_Y_val-y_val_pred)**2))
# MAE
print(np.mean(abs(df_Y_val-y_val_pred)))
```

    0.6617596418838108
    0.48310632170135215
    0.5090702644184196
    

## 6. 예측


```python
raw_test = pd.read_csv('data/qualityData_test.csv')
test = raw_test.copy()
```


```python
#test[test.columns[1:86]] = test[test.columns[1:86]].apply(minmaxscaler, axis=0)
#test['Time_Sequence'] = (test['Time_Sequence']-1)/(3807-1)
```


```python
test.describe()
```



```python
X0_test = test[X_col]
X_test = test[test.columns[:86]]
```


```python
rbf_pca = KernelPCA(n_components = 5, kernel="rbf", gamma=0.274, fit_inverse_transform=True)
X_test_pc = rbf_pca.fit_transform(X0_test)
X_test_pc_inv = rbf_pca.inverse_transform(X_test_pc)
mean_squared_error(X0_test, X_test_pc_inv)
```




    0.4009954254743861




```python
df_X_train.head(3)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time_Sequence</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>75</th>
      <th>76</th>
      <th>77</th>
      <th>78</th>
      <th>79</th>
      <th>80</th>
      <th>81</th>
      <th>82</th>
      <th>83</th>
      <th>84</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2729</th>
      <td>2988</td>
      <td>-0.075255</td>
      <td>-0.670662</td>
      <td>0.554325</td>
      <td>-0.987781</td>
      <td>0.296512</td>
      <td>-0.513375</td>
      <td>0.034924</td>
      <td>-0.591695</td>
      <td>0.521103</td>
      <td>...</td>
      <td>1.322268</td>
      <td>-0.703414</td>
      <td>-0.676013</td>
      <td>-0.849892</td>
      <td>1.753639</td>
      <td>1.753596</td>
      <td>1.054071</td>
      <td>-0.592789</td>
      <td>-0.713732</td>
      <td>0.592326</td>
    </tr>
    <tr>
      <th>3341</th>
      <td>3647</td>
      <td>-0.043314</td>
      <td>-0.693356</td>
      <td>0.539805</td>
      <td>-0.575558</td>
      <td>-0.297256</td>
      <td>-0.496055</td>
      <td>-0.148273</td>
      <td>-0.609136</td>
      <td>0.536950</td>
      <td>...</td>
      <td>-0.057272</td>
      <td>0.135934</td>
      <td>0.251381</td>
      <td>0.217531</td>
      <td>1.763925</td>
      <td>1.808351</td>
      <td>-0.130871</td>
      <td>0.265545</td>
      <td>0.183189</td>
      <td>0.571972</td>
    </tr>
    <tr>
      <th>1607</th>
      <td>1753</td>
      <td>-0.031633</td>
      <td>-0.691115</td>
      <td>0.551150</td>
      <td>-0.600187</td>
      <td>-0.019595</td>
      <td>-0.506974</td>
      <td>0.330753</td>
      <td>-0.607473</td>
      <td>0.521085</td>
      <td>...</td>
      <td>0.015898</td>
      <td>-0.243354</td>
      <td>-0.123675</td>
      <td>-0.073419</td>
      <td>1.765924</td>
      <td>1.803083</td>
      <td>0.059135</td>
      <td>-0.092753</td>
      <td>-0.188646</td>
      <td>0.574404</td>
    </tr>

  </tbody>
</table>
<p>5 rows × 86 columns</p>
</div>




```python
X_test_KPC = pd.concat([test['Time_Sequence'], pd.DataFrame(X_test_pc_inv)], axis=1)
X_test_KPC.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time_Sequence</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>75</th>
      <th>76</th>
      <th>77</th>
      <th>78</th>
      <th>79</th>
      <th>80</th>
      <th>81</th>
      <th>82</th>
      <th>83</th>
      <th>84</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>-0.054129</td>
      <td>-0.662981</td>
      <td>0.592394</td>
      <td>-0.649704</td>
      <td>-0.264775</td>
      <td>-0.513557</td>
      <td>-0.150291</td>
      <td>-0.585189</td>
      <td>0.526572</td>
      <td>...</td>
      <td>0.110629</td>
      <td>-0.032189</td>
      <td>0.097198</td>
      <td>0.057434</td>
      <td>1.708933</td>
      <td>1.809054</td>
      <td>0.089652</td>
      <td>0.123341</td>
      <td>0.014893</td>
      <td>0.594546</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>-0.054132</td>
      <td>-0.662972</td>
      <td>0.592404</td>
      <td>-0.649600</td>
      <td>-0.264995</td>
      <td>-0.513538</td>
      <td>-0.150599</td>
      <td>-0.585180</td>
      <td>0.526552</td>
      <td>...</td>
      <td>0.110461</td>
      <td>-0.031880</td>
      <td>0.097562</td>
      <td>0.057787</td>
      <td>1.708894</td>
      <td>1.809051</td>
      <td>0.089392</td>
      <td>0.123704</td>
      <td>0.015233</td>
      <td>0.594525</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>-0.054122</td>
      <td>-0.663227</td>
      <td>0.592119</td>
      <td>-0.653270</td>
      <td>-0.257838</td>
      <td>-0.514010</td>
      <td>-0.141879</td>
      <td>-0.585429</td>
      <td>0.527109</td>
      <td>...</td>
      <td>0.116154</td>
      <td>-0.041925</td>
      <td>0.085922</td>
      <td>0.046383</td>
      <td>1.710016</td>
      <td>1.809066</td>
      <td>0.097676</td>
      <td>0.112336</td>
      <td>0.004361</td>
      <td>0.595136</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19</td>
      <td>-0.054121</td>
      <td>-0.663221</td>
      <td>0.592126</td>
      <td>-0.653174</td>
      <td>-0.258021</td>
      <td>-0.514000</td>
      <td>-0.142089</td>
      <td>-0.585423</td>
      <td>0.527096</td>
      <td>...</td>
      <td>0.116006</td>
      <td>-0.041669</td>
      <td>0.086218</td>
      <td>0.046674</td>
      <td>1.709989</td>
      <td>1.809067</td>
      <td>0.097466</td>
      <td>0.112622</td>
      <td>0.004637</td>
      <td>0.595120</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
      <td>-0.054121</td>
      <td>-0.663235</td>
      <td>0.592110</td>
      <td>-0.653380</td>
      <td>-0.257623</td>
      <td>-0.514025</td>
      <td>-0.141615</td>
      <td>-0.585437</td>
      <td>0.527126</td>
      <td>...</td>
      <td>0.116325</td>
      <td>-0.042228</td>
      <td>0.085572</td>
      <td>0.046040</td>
      <td>1.710050</td>
      <td>1.809067</td>
      <td>0.097925</td>
      <td>0.111994</td>
      <td>0.004034</td>
      <td>0.595154</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 86 columns</p>
</div>




```python
y_test_pred = AveragingBlending(models, df_X_train, df_Y_train, X_test_KPC)
```


```python
y_test_pred = pd.DataFrame(y_test_pred, columns=['Y'])

submission = pd.concat([raw_test['Time_Sequence'], y_test_pred], axis=1)
submission.to_csv('data/quality_result0.csv', index=False)
submission.head(3)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time_Sequence</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>1.085601</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1.085601</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>1.086483</td>
    </tr>
  </tbody>
</table>
</div>




```python

```



