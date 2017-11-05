#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

sampleNum=200

X=np.arange(-5,5,0.05)
Y=X+2
Y+=X**2+np.random.uniform(-0.5,0.5,size=sampleNum)

X_train,X_test,Y_train,Y_test=train_test_split(
						X.reshape(-1,1),
						Y.reshape(-1,1),
						test_size=0.2)
plt.scatter(X,Y,s=5)

lr=LinearRegression(normalize=True)
lr.fit(X_train,Y_train)
print('Linear regression score is:'+str(lr.score(X_test,Y_test)))
print(lr.intercept_)
print(lr.coef_)
X_lr=np.linspace(-5,5,100)
Y_lr=X_lr*lr.coef_+lr.intercept_
plt.plot(X_lr.reshape(-1,1),Y_lr.reshape(-1,1),'black')

pf=PolynomialFeatures(degree=2)
X_train=pf.fit_transform(X_train)
X_test=pf.fit_transform(X_test)
lr.fit(X_train,Y_train)
print('PolynomialFeatures score is:'+str(lr.score(X_test,Y_test)))
print(lr.intercept_)
print(lr.coef_)

X_pf=X_lr
Y_pf=lr.intercept_+X_pf*lr.coef_[0,1]+X_pf**2*lr.coef_[0,2]
plt.plot(X_pf.reshape(-1,1),Y_pf.reshape(-1,1),c='r')

plt.show()




