#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage,dendrogram
nb_samples=25
X,Y=make_blobs(n_samples=nb_samples,n_features=2,
			centers=3,cluster_std=1.5)
plt.figure(1)
plt.scatter(X[:,0],X[:,1],s=3)

plt.figure(2)
Xdist=pdist(X,metric='euclidean')
Xl=linkage(Xdist,method='ward')
Xd=dendrogram(Xl)
plt.show()