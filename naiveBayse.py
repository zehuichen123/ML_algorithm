#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest

from sklearn.datasets import make_classification
#from sklearn.bayes import BernoulliNB
import matplotlib.pyplot as plt
sampleNum=300
X,Y=make_classification(n_samples=sampleNum,
						n_features=2,
						n_informative=2,
						n_redundant=0)

for i in range(sampleNum):
	if Y[i]==0:
		plt.scatter(X[i,0],X[i,1],c='r',marker='0')
	else:
		plt.scatter(X[i,0],X[i,1],c='blue',marker='x')
plt.show()