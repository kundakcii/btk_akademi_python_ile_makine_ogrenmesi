# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:44:15 2021

@author: kundakci
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

veriler=pd.read_csv('tenis.csv')

"""LabelEncoding işlemi """
veriler2=veriler.apply(preprocessing.LabelEncoder().fit_transform)
"""OneHotEncoder"""
outlook=veriler2.iloc[:,:1]
ohe=preprocessing.OneHotEncoder();
outlook=ohe.fit_transform(outlook).toarray()
"""İlgili verilerin ayıklanması"""
sonucOutlook=pd.DataFrame(data=outlook,index=range(14),columns=['o','r','s'])

"""verilerin birleştirilmesi"""
sonucVeri=pd.concat([veriler2.iloc[:,-2:],sonucOutlook,veriler.iloc[:,1:3]],axis=1)

"""verilerin bölünmesi"""

x_train,x_test,y_train,y_test=train_test_split(sonucVeri.iloc[:,:-1],sonucVeri.iloc[:,-1:],test_size=0.33,random_state=0)


"""Lİneaer Regression kullanarak verilerin tahmini"""
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

"""backward elimination p value bulunması"""

X=np.append(arr=np.ones((14,1)).astype(int),values=sonucVeri.iloc[:,:-1],axis=1)

X_l=sonucVeri.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(sonucVeri.iloc[:,-1:],X_l).fit()
print(model.summary())

sonveri=sonucVeri.iloc[:,1:]

X_l=sonucVeri.iloc[:,[0,1,2,3,4]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(sonucVeri.iloc[:,-1:],X_l).fit()
print(model.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)







