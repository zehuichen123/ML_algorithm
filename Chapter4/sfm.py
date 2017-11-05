#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

boston=load_boston()
X_train,X_test,Y_train,Y_test=train_test_split(
							boston.data,
							boston.target,
							test_size=0.2)
pf=PolynomialFeatures(degree=2)
X_train_pf=pf.fit_transform(X_train)
X_test_pf=pf.fit_transform(X_test)

lr=LinearRegression(normalize=True)
lr.fit(X_train,Y_train)
print('the LinearRegression model score:')
print(lr.score(X_test,Y_test))

sfm=SelectFromModel(lr,threshold=60)
sfm.fit_transform(X_train_pf,Y_train)
print('the SelectFromModel model score:')
temp=sfm.estimator_.score(X_test_pf,Y_test)
print(temp)





