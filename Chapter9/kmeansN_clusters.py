#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,calinski_harabaz_score

nb_samples=1000
X,_=make_blobs(n_samples=nb_samples,n_features=2,
			centers=5,cluster_std=1.5)

# here we introduce how to find the best
# n_clusters for your kmeans model
plt.figure(1)
plt.scatter(X[:,0],X[:,1],s=3)


nb_clusters=[2,3,4,5,6,7,8,9,10]
inertias=[]

for n in nb_clusters:
	km=KMeans(n_clusters=n)
	km.fit(X)
	inertias.append(km.inertia_)

plt.figure(1)
plt.plot(nb_clusters,inertias)
plt.xlabel('Number of clusters')
plt.ylabel('Inertias')

avg_silhouettes=[]

for n in nb_clusters:
	km=KMeans(n_clusters=n)
	Y=km.fit_predict(X)
	avg_silhouettes.append(silhouette_score(X,Y))
plt.figure(2)
plt.plot(nb_clusters,avg_silhouettes)
plt.xlabel('Number of clusters')
plt.ylabel('Average silhouette score')

ch_scores=[]
for n in nb_clusters:
	km=KMeans(n_clusters=n)
	Y=km.fit_predict(X)
	ch_scores.append(calinski_harabaz_score(X,Y))
plt.figure(3)
plt.plot(nb_clusters,ch_scores)
plt.xlabel('Number of clusters')
plt.ylabel('calinski_harabaz_score')

plt.show()