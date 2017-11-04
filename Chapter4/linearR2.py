#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
import numpy as np
from sklearn.datasets import load_boston
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

boston=load_boston()
'''
for i in range(12):
	plt.subplot(3,4,i+1)
	plt.plot(boston.data[:,i])
plt.show()
'''
X_train,X_test,Y_train,Y_test=train_test_split(
			boston.data,
			boston.target,
			test_size=0.1)
lr=LinearRegression(normalize=True)

lr.fit(X_train,Y_train)
# print(lr.score(X_test,Y_test))
# for here,it is no use to use score here because 
# the predict value is continuous instead of discrete

# if we add some guissan noise here
X_noise=boston.data[0:10]+np.random.normal(0.0,0.1)
print(lr.predict(X_noise))
print(boston.target[0:10])

print('\n')

print('y='+str(lr.intercept_)+' ')
for i,c in enumerate(lr.coef_):
	print('x'+str(i)+' is :'+str(c))

lr1=LinearRegression(normalize=True)
print('using cross-validation method:')
scores=cross_val_score(lr1,boston.data,boston.target,cv=10,scoring='neg_mean_squared_error')
# what is the neg_mean_squared_error method??? 
print(scores.mean())
# using coefficent of destination and cross-validation

