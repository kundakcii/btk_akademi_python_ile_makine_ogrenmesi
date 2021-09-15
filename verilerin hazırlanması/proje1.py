# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 15:20:40 2021

@author: kundakci
"""


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import  SimpleImputer

#verilerin içeri alınması
veriler = pd.read_csv('eksikveriler.csv')


#Nan değerlerin ortalama ile doldurulması
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
yasVerisi=veriler.iloc[:,1:4].values
imputer = imputer.fit(yasVerisi)
yasVerisi =  imputer.transform(yasVerisi)
print(yasVerisi)


"""one hot encoder kullanımı"""
ulkeVeri=veriler.iloc[:,0:1]
ohe=OneHotEncoder()
oheUlke=ohe.fit_transform(ulkeVeri).toarray()
print(oheUlke)

"""verilerin birleştirilmeden önce DataFrame'e dönüştürülüp hazırlanması."""
sonuc1 = pd.DataFrame(data=oheUlke,index=range(22),columns=['fr','tr','us'])
print(sonuc1)

sonuc2=pd.DataFrame(data=yasVerisi,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyetVerisi =veriler.iloc[:,-1].values# -1 sondon 1. kolon demektir
le=LabelEncoder()
cinsiyetVerisi=le.fit_transform(cinsiyetVerisi)# 0=erkek
sonuc3 = pd.DataFrame(data=cinsiyetVerisi,index=range(22),columns=['cinsiyet'])
print(sonuc3)


"""hazırlanan sonuç varilerinintek bir tabloda birleştirilmesi."""
s=pd.concat([sonuc1,sonuc2],axis=1)
hazırlananVeri=pd.concat([sonuc1,sonuc2,sonuc3],axis=1) 
"""axis=1 yanya ekler. axis 'ın default değeri sıfırdır ve alt alta ekler ve 
nan değerler oluşur."""




"""Hazırlanan verilerin eğitim ve test olarak bölünmesi"""
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(s,sonuc3,test_size=0.33,random_state=0)
"""s=x bağımsız sonuc3=y bağımlı değişkenler"""

"""verinin standartlandırılması"""
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test);

















