# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:01:18 2021

@author: kundakci
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
veriler=pd.read_csv('maaslar.csv')
"""Linear regression"""

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,-1:]
X=x.values
Y=y.values
lg= LinearRegression()
lg.fit(X, Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lg.predict(X))
plt.show()

"""polynomial regression"""
poly_reg=PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)

lin_reg2= LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(X,Y,color='red')
plt.plot(X,Y)
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()




poly_reg=PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)

lin_reg2= LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(X,Y,color='red')
plt.plot(X,Y)
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()









