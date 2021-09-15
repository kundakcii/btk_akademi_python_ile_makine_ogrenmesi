# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:10:45 2021

@author: kundakci
"""

import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

veriler =pd.read_csv('veriler.csv')


lecinsiyet=veriler.iloc[:,-1:].values
le=preprocessing.LabelEncoder();
lecinsiyet[:,-1]=le.fit_transform(veriler.iloc[:,-1])

"""burda  ikitane cinsiyet colonu dobby variable durumuna yol açar birini temizle"""
ohe=preprocessing.OneHotEncoder()
oheCinsiyet=veriler.iloc[:,-1:].values
oheCinsiyet=ohe.fit_transform(oheCinsiyet).toarray()

"""burada mecbur ohe  yapıyorum çünkü aynı durumu etkileyen birden fazla collumn var ve
n columndan n-1 tanesini almak zorundayım"""

ulke=veriler.iloc[:,:1]
oheUlke=ohe.fit_transform(veriler.iloc[:,:1]).toarray()

"""geriye kalan veri"""
digerVeriler=veriler.iloc[:,1:4]
"""cinsiyetin DataFrame e dönüştürlmesi"""
sonucCinsiyet=pd.DataFrame(data=lecinsiyet,index=range(22),columns=['Cinsiyet'])
"""ulke verisinin dataframe çevirlmesi"""
sonucUlke = pd.DataFrame(data=oheUlke,index=range(22),columns=['fr','tr','us'])

"""Verilerin birleştirilmesi"""
veri=pd.concat([sonucUlke,digerVeriler],axis=1)
tumveriler=pd.concat([sonucUlke,digerVeriler,sonucCinsiyet],axis=1)

"""verilerin eğitim için bölünmesi"""
x_train,x_test,y_train,y_test=train_test_split(veri,sonucCinsiyet,test_size=0.33,random_state=0)



"""modelin oluşturulması ve eğitilmesi"""
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

boyVerisi=tumveriler.iloc[:,3:4].values
sol=tumveriler.iloc[:,:3]
sag=tumveriler.iloc[:,4:]

birlesenVeri = pd.concat([sol,sag],axis=1)

x_train,x_test,y_train,y_test=train_test_split(birlesenVeri,boyVerisi,test_size=0.33,random_state=0)
r2=LinearRegression()
r2.fit(x_train,y_train)
y_pred=r2.predict(x_test)


"""başarı kriterlerinin belirlenmesi modelin başarısını test etme p value bulma 
p value su  yüksek olan sistemden çıkarılmalı
""" 
X =np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)

X_l=birlesenVeri.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model =sm.OLS(boyVerisi,X_l).fit()
print(model.summary())


"""p si en büyük olan 4. elemandı eledik 
p degeri 0.5 den küçükse kabul edilebilir bu değişebilir."""

X_l=birlesenVeri.iloc[:,[0,1,2,3,5]].values
X_l=np.array(X_l,dtype=float)
model =sm.OLS(boyVerisi,X_l).fit()
print(model.summary())






    
