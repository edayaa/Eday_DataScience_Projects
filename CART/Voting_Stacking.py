# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:07:47 2020

@author: Mohamed.Imran
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from mlxtend.classifier import StackingCVClassifier
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('diabetes_data.csv')

X = df.drop(columns = ['diabetes'])
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

df['diabetes'].value_counts()
y_train.value_counts()
y_test.value_counts()

#create a knn model
knn = KNeighborsClassifier()
params_knn = {'n_neighbors': np.arange(1, 25)}
knn_gs = GridSearchCV(knn, params_knn, cv=5)
knn_gs.fit(X_train, y_train)

knn_best = knn_gs.best_estimator_
print(knn_gs.best_params_)

#create a rf classifier
rf = RandomForestClassifier()
params_rf = {'n_estimators': [50, 100, 200]}
rf_gs = GridSearchCV(rf, params_rf, cv=5)

rf_gs.fit(X_train, y_train)

rf_best = rf_gs.best_estimator_
print(rf_gs.best_params_)

#create a logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

#Test accuracy
print('knn: {}'.format(knn_best.score(X_test, y_test)))
print('rf: {}'.format(rf_best.score(X_test, y_test)))
print('log_reg: {}'.format(log_reg.score(X_test, y_test)))

#create a dictionary of our models
estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg)]

#create voting classifier
vc = VotingClassifier(estimators)
vc.fit(X_train, y_train)
vc.score(X_test, y_test)


#Simple Stacking CV classifier
clf1 = KNeighborsClassifier(n_neighbors=10)
clf2 = RandomForestClassifier(random_state=42)
clf3 = GaussianNB()
lr = LogisticRegression()

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                            meta_classifier=lr,
                            random_state=42)
sclf.fit(X_train, y_train)
sclf.score(X_test, y_test)

#Stacking classifier using probabilities as Meta-Features
clf1 = KNeighborsClassifier(n_neighbors=10)
clf2 = RandomForestClassifier(random_state=42)
clf3 = GaussianNB()
lr = LogisticRegression()

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                            meta_classifier=lr, use_probas=True,
                            random_state=42)
sclf.fit(X_train, y_train)
sclf.score(X_test, y_test)


#vif
variance_inflation_factor(X_train.values, 0) #for testing vif for one variable

#vif for all variables
for i in range(len(X_train.columns)):
    print(variance_inflation_factor(X_train.values, i))
