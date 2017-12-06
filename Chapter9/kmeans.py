#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
nb_samples=1000
X,_=make_blobs(n_samples=nb_samples,n_features=2,
			centers=3,cluster_std=1.5)
plt.scatter(X[:,0],X[:,1],s=3)

km=KMeans(n_clusters=3)
km.fit(X)
centers=km.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],s=3,c='red')
plt.show()

