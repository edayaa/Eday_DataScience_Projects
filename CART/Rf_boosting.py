# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 07:15:52 2020

@author: Mohamed.Imran
"""
import os
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier , RandomForestRegressor
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

cwd = os.getcwd()
cwd


data = pd.read_csv('C:\\Users\\IQBALE\\ClassRoom\\CART\\Automobile_data.xls',sep=',')

#data = pd.read_csv('Automobile_data.csv')

data.replace('?', np.nan, inplace = True)

string_cols = data.select_dtypes(exclude = np.number).columns.tolist()

#Num cols
num_cols = ['normalized-losses', 'bore', 'stroke', 'horsepower','peak-rpm','price']

for i in num_cols:
    data[i] = pd.to_numeric(data[i], errors = 'raise')
    
#category conversion
for i in data:
    if is_string_dtype(data[i]):
        data[i] = data[i].astype('category').cat.as_unordered()

#cat code conversion (int)
for i in data:
    if(str(data[i].dtype) == 'category'):
        data[i] = data[i].cat.codes

data.fillna(data.median(), inplace = True)

#model
X= data.drop('symboling', 1)
y = data['symboling']
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 100)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

#print(rf.score(x_train, y_train))
print(rf.score(x_val, y_val))

lr = LogisticRegression()
lr.fit(x_train, y_train)

print(lr.score(x_val, y_val)) # 0.6585365853658537


NoofEstimator = [5, 10, 15, 20]
MinSampleLeaf = [1, 3, 5, 7]
MaxFeature = np.arange(0.1, 1.1, 0.1)
best_score = []

for i in NoofEstimator:
    for j in MinSampleLeaf:
        for k in MaxFeature:
            result = [i, j, k]
            rfc = RandomForestClassifier(n_estimators = i, min_samples_leaf = j, max_features = k) #random_state = 100 (can be used)
            rfc.fit(x_train, y_train)
            result.append(rfc.score(x_train, y_train))
            result.append(rfc.score(x_val, y_val))
            if len(best_score) == 0:
                best_score = result
            elif best_score[4] < result[4]:
                best_score = result
                print(best_score)
                
print('The final best result is: ', best_score)

#[5, 1, 0.2, 0.9573170731707317, 0.7804878048780488]
#[5, 1, 0.6, 0.9939024390243902, 0.8536585365853658]
#[15, 1, 0.4, 1.0, 0.8780487804878049]

#Grid search
rf = RandomForestClassifier()
rf_grid = GridSearchCV(estimator=rf, param_grid = dict(n_estimators = NoofEstimator,
                                                       min_samples_leaf = MinSampleLeaf,
                                                       max_features = MaxFeature))

rf_grid.fit(x_train, y_train)
print(rf_grid.best_estimator_)
print(rf_grid.score(x_val, y_val))

"""RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features=0.7000000000000001,
                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=15, n_jobs=None, oob_score=False,
                       random_state=None, verbose=0, warm_start=False)
0.7804878048780488"""

#Randomized search
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions=dict(n_estimators = NoofEstimator,
                                                       min_samples_leaf = MinSampleLeaf,
                                                       max_features = MaxFeature))

rf_random.fit(x_train, y_train)
print(rf_random.best_estimator_)
print(rf_random.score(x_val, y_val))    

"""RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features=0.2, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=15,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
0.8048780487804879"""

#Checking out of bag score
rf_o = RandomForestClassifier(oob_score=True)
rf_o.fit(x_train, y_train)
rf_o.oob_score_

"""C:\ProgramData\Anaconda3\lib\site-packages\sklearn\ensemble\forest.py:460: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable oob estimates.
  warn("Some inputs do not have OOB scores. "
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\ensemble\forest.py:465: RuntimeWarning: invalid value encountered in true_divide
  predictions[k].sum(axis=1)[:, np.newaxis])
Out[11]: 0.6951219512195121"""

#Important features
imp_features = rf_grid.best_estimator_.feature_importances_
feature_list = list(X.columns)
feature_importance = sorted(zip(imp_features, feature_list), reverse=True)
df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])


#Visaulization
plt.style.use('ggplot') #ggplot

x_values = list(range(len(feature_importance)))
importance = list(df['importance'])
feature = list(df['feature'])

plt.figure(figsize = (15, 10))
plt.bar(x_values, importance, orientation = 'vertical')
plt.xticks(x_values, feature, rotation = 'vertical')

plt.xlabel('Variable')
plt.ylabel('Importance')
plt.title('Variable importance')


#Adaboost 
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=0.8, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


ab_model = AdaBoostClassifier(n_estimators = 50,  base_estimator = rf)
ab_model.fit(x_train, y_train)
print('Ada test score: ', ab_model.score(x_val, y_val))

#Ada test score:  0.8048780487804879

#xgboost
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
print('XGB test score: ', xgb.score(x_val, y_val))
#XGB test score:  0.8536585365853658