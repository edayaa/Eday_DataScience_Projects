# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:03:23 2020

@author: Mohamed.Imran
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('diabetes_data.csv')

X = data.drop(columns = ['diabetes'])
y = data['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

pca = PCA(n_components = 3)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

pca.explained_variance_ratio_

lr = LogisticRegression()
lr.fit(X_train, y_train)

lr.score(X_test, y_test)