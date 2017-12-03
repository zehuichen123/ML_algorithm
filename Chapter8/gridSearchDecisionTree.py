#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
import numpy as np
from sklearn.datasets import make_classification,load_digits
# import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import cross_val_score,GridSearchCV

digits=load_digits()
param_grid=[
	{
		'criterion':['gini','entropy'],
		'max_features':['auto','log2',None],
		'min_samples_split':[2,10,25,100,200],
		'max_depth':[5,10,15,None]
	}
]

gs=GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=param_grid,
				scoring='accuracy',cv=10)
gs.fit(digits.data,digits.target)
print(gs.best_estimator_)
print(gs.best_score_)
