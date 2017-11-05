#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest

from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

sampleNum=500
plt.subplots(figsize=(8,8))
X,Y=make_classification(n_samples=sampleNum,
						n_features=2,
						n_informative=2,
						n_redundant=0,
						n_clusters_per_class=1)

sgd=SGDClassifier(loss='log',learning_rate='optimal',max_iter=10)
print('using log loss function score:')
print(cross_val_score(sgd,X,Y,cv=10,scoring='accuracy').mean())
print('using perceptron loss function score:')
sgdp=SGDClassifier(loss='perceptron',learning_rate='optimal',max_iter=10)
print(cross_val_score(sgdp,X,Y,cv=10,scoring='accuracy').mean())