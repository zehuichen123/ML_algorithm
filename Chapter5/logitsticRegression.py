#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

sampleNum=500
plt.subplots(figsize=(8,8))
X,Y=make_classification(n_samples=sampleNum,
						n_features=2,
						n_informative=2,
						n_redundant=0,
						n_clusters_per_class=1)

for i in range(sampleNum):
	if(Y[i]==0):
		plt.scatter(X[i,0],X[i,1],marker='o',c='r',s=8)
	else:
		plt.scatter(X[i,0],X[i,1],marker='^',c='b',s=8)


lr=LogisticRegression()
lr.fit(X,Y)
score=cross_val_score(lr,X,Y,cv=10,scoring='accuracy')
print('LogisticRegression score is:')
print(score.mean())

print('the coefficient of model:')
print(lr.coef_)
print(lr.intercept_)

X_line=np.linspace(X[:,0].min(),X[:,0].max(),10)
Y_line=(-lr.coef_[0,0]*X_line-lr.intercept_)/lr.coef_[0,1]
plt.plot(X_line,Y_line)

plt.show()

# here the coef_ is for each feature. say this example,
# it is like coef[0]*feature1+coef[1]*feature2+intercept=0
