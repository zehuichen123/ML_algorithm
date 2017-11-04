#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def getData():
	X=np.linspace(0,5,20)
	Y=X+np.random.uniform(-0.5,0.5,size=20)
	return X,Y

X,Y=getData()
plt.subplot(1,2,1)
plt.scatter(X,Y,marker='x',c='r')
plt.xlabel('the origin data')
plt.plot(X,X)

def lossFunc(v):
	f=0
	for (i,j) in zip(X,Y):
		f+=np.square(v[0]+v[1]*i-j)
	return 0.5*f

def getGradient(v):
	g=np.zeros(shape=2)
	for i in range(X.size):
		g[0]+=v[0]+v[1]*X[i]-Y[i]
		g[1]+=(v[0]+v[1]*X[i]-Y[i])*X[i]
	return g

res=minimize(fun=lossFunc,x0=[0,0],jac=getGradient,method='L-BFGS-B')
para=res.x
xPlot=np.linspace(0,5,20)
yPlot=xPlot*para[1]+para[0]
plt.subplot(1,2,2)
plt.scatter(X,Y,marker='x',c='r')
plt.plot(xPlot,yPlot)
plt.xlabel('using simple liner regression')
plt.show()


# the figure plot can be viewed through: http://r.photo.store.qq.com/psb?/V13ngZ4i1rva8i/a2i3g7ckPneDgSTHvirHpF7VG9NmB3iLn8EzA4eVB7A!/r/dPIAAAAAAAAA