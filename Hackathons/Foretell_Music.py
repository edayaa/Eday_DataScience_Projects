# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 07:20:24 2019

@author: IQBALE
"""

import os
import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import datetime as dt 
import pandas_profiling
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

pwd = os.getcwd()

print(pwd)

data_Train = pd.read_csv('C:\\Users\\IQBALE\\ClassRoom\\Hackathons\\ChartbustersParticipantsData\\Data_Train.csv')
data_Test = pd.read_csv('C:\\Users\\IQBALE\\ClassRoom\\Hackathons\\ChartbustersParticipantsData\\Data_Test.csv')

data_Test.shape
data_Train.shape
data_Train.describe()
data= pd.concat([data_Train,data_Test],axis=0)
data.shape
data.isna().sum() 
data.duplicated().sum() # 0 
data[pd.isnull(data["Song_Name"])]
data.shape
data = data[data.isna()['Song_Name'] == False]
data.isna().sum()
data["Date"] = pd.to_datetime(data["Timestamp"])
data["Year"] = data["Date"].dt.year
data["Year"] = 2000 - data["Year"]
data["Year"].unique()
data["Comments"].unique() # number values 
data["Country"].unique()
data["Followers"].unique()
data["Genre"].unique()
data["Likes"].unique()
data["Views"].isna().sum()

#pandas_profiling.ProfileReport(data)
#data.describe(include='all')
data.columns

data_input = data[['Comments', 'Followers', 'Genre', 'Likes', 
       'Popularity', 'Views', 'Year']]

data_input.dtypes
data_input["Likes"] = data_input.Likes.str.replace(',','')
data_input["Popularity"] = data_input.Popularity.str.replace(',','')

"""if data_input["Likes"][-1] == 'K':
    data_input["Likes"] = float(data_input["Likes"][:-1]) * 1000
else if data_input["Likes"][-1] == 'M':
    data_input["Likes"] = float(data_input["Likes"][:-1]) * 1000000
else:
    data_input["Likes"] = float(data_input["Likes"])"""

def convert_num(x):
    if x[-1] == 'K':
        return float(x[:-1]) * 1000
    elif x[-1] == 'M':
        return float(x[:-1]) * 1000000
    else: 
        return float(x)

data_input["Likes"] = data_input["Likes"].apply(convert_num)
data_input["Popularity"] = data_input["Popularity"].apply(convert_num)

Train_input = data_input[data_input.isna()['Views'] == False] 
Test_input = data_input[data_input.isna()['Views'] == True]

data_input["Popularity"] .unique()

data_input["Genre"].unique()

le = preprocessing.LabelEncoder()
data_input["Genre"] = le.fit_transform(data_input["Genre"])

data_input["Genre"].unique()

#     print(df[i])

Train_input.shape
Test_input.shape

X_Train = Train_input.drop(columns=['Views'])
Y_Train = Train_input[['Views']]
X_Test  = Test_input.drop(columns=['Views'])

X_Train.dtypes
X_Train.Likes.unique()

X_Train.Popularity.unique()
X_Train.Genre.unique()
X_Train.shape 
X_Test.shape

X_train_train, X_train_test, Y_train_train, Y_train_test = train_test_split(X_Train,Y_Train,test_size=0.3,random_state= 8)



def model_fit(model,X_train_train,X_train_test,Y_train_train,Y_train_test,X_test):
    
    model.fit(X_train_train,Y_train_train)
    #pd.DataFrame(model.coef_,X_train_train.columns).plot(kind="bar")
    train_predict = model.predict(X_train_train)
    test_predict = model.predict(X_train_test)
    print("Model:" , model)
    print("Train RMSE : ",np.sqrt(mean_squared_error(train_predict,Y_train_train)))
    print("Test RMSE : ",np.sqrt(mean_squared_error(test_predict,Y_train_test)))
    print ( 'number of features used: ' ,  np.sum(model.coef_!=0) )    
    #test_final_predict = model.predict(X_test)
    #test_final_predict["Unique_ID"] = X_test[["Unique_ID"]]
    #test_final_predict.to_excel (r'C:\Eday\docs\Datascience\Foretell_music.xlsx', index = None, header=True)
        
model = LinearRegression() 
model_fit(model,X_train_train,X_train_test,Y_train_train,Y_train_test,X_Test)

model = Lasso(alpha=0.1)
model_fit(model,X_train_train,X_train_test,Y_train_train,Y_train_test,X_Test)




