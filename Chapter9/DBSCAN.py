#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,DBSCAN

nb_samples=1000
X,Y=make_moons(n_samples=nb_samples,noise=0.05)
plt.subplot(3,1,1)

for i,y in enumerate(Y):
	if y==1:
		plt.scatter(X[i,0],X[i,1],s=3,c='red')
	else:
		plt.scatter(X[i,0],X[i,1],s=3,c='blue')

km=KMeans(n_clusters=2)
Y_km=km.fit_predict(X)
plt.subplot(3,1,2)
for i,y in enumerate(Y_km):
	if y==1:
		plt.scatter(X[i,0],X[i,1],s=3,c='red')
	else:
		plt.scatter(X[i,0],X[i,1],s=3,c='blue')

dbs=DBSCAN(eps=0.1)
Y_dbs=dbs.fit_predict(X)
plt.subplot(3,1,3)
for i,y in enumerate(Y_dbs):
	if y==1:
		plt.scatter(X[i,0],X[i,1],s=3,c='red')
	else:
		plt.scatter(X[i,0],X[i,1],s=3,c='blue')

plt.show()