#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression,RANSACRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1000)
sampleNum=200
noiseNum=20

X=np.arange(-5,5,0.05)
Y=X+2

Y+=np.random.uniform(-0.5,0.5,size=sampleNum)

for i in range(sampleNum-noiseNum,sampleNum):
	Y[i]+=np.random.uniform(4,7)

plt.scatter(X,Y,s=5)

lr=LinearRegression(normalize=True)
lr.fit(X.reshape(-1,1),Y.reshape(-1,1))
print('w[0]='+str(lr.intercept_))
print('w[1]='+str(lr.coef_))
lrX=np.linspace(-5,5,100)
lrY=(lrX*lr.coef_+lr.intercept_).reshape(100,-1)
plt.plot(lrX,lrY,c='r')

rr=RANSACRegressor(lr)
rr.fit(X.reshape(-1,1),Y.reshape(-1,1))
print('w[0]='+str(rr.estimator_.intercept_))
print('w[1]='+str(rr.estimator_.coef_))
rrX=np.linspace(-5,5,100)
rrY=(rrX*rr.estimator_.coef_+rr.estimator_.intercept_).reshape(100,-1)
plt.plot(rrX,rrY,c='black')

plt.show()
# in this case I find that if the noise is not so overt,then 
# RANSACRegressor may not be so useful