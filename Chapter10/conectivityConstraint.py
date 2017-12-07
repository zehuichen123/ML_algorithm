#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
#from scipy.spatial.distance import pdist
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
#from scipy.cluster.hierarchy import linkage,dendrogram
nb_samples=1000
X,_=make_circles(n_samples=nb_samples,noise=0.05)
plt.figure(1)
plt.scatter(X[:,0],X[:,1],s=3)

plt.figure(2)
ac=AgglomerativeClustering(n_clusters=20)
Y1=ac.fit_predict(X)

acc=[]
k=[50,100,200,500]

for i in range(4):
	kng=kneighbors_graph(X,k[i])
	ac1=AgglomerativeClustering(n_clusters=20,connectivity=kng,
							linkage='average')
	ac1.fit(X)
	acc.append(ac1)
