#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
#from scipy.cluster.hierarchy import linkage,dendrogram
nb_samples=1000
X,_=make_blobs(n_samples=nb_samples,n_features=2,
			centers=8,cluster_std=2.0)
plt.figure(1)
plt.scatter(X[:,0],X[:,1],s=3)

plt.figure(2)
ac=AgglomerativeClustering(n_clusters=8)
Y=ac.fit_predict(X)

color=['red','green','blue','black','pink','yellow','brown','grey']
for i,y in enumerate(Y):
	plt.scatter(X[i,0],X[i,1],c=color[y],s=3)
plt.show()