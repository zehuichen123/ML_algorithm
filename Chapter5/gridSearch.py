#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest

import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score,GridSearchCV
# import matplotlib.pyplot as plt
import numpy as np

iris=load_iris()
paramGrid=[
	{
		'penalty':['l1','l2'],  			# penalty here is l1(l1 正则）,and l2（l2 正则）
		'C':[0.5,1.0,1.5,1.8,2.0,2.5]
	}
]
gs=GridSearchCV(estimator=LogisticRegression(),
				param_grid=paramGrid,
				cv=10,
				n_jobs=multiprocessing.cpu_count())
gs.fit(iris.data,iris.target)

lr=LogisticRegression()
lr.fit(iris.data,iris.target)
print('using the LogisticRegression model scores:')
print(cross_val_score(lr,iris.data,iris.target,cv=10).mean())

print('finding the best param grid:')
print(gs.best_estimator_)
print('the score of this is:')
print(cross_val_score(gs.best_estimator_,iris.data,iris.target,cv=10,scoring='accuracy').mean())



