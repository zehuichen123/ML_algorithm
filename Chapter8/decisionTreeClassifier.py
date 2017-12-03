#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
import numpy as np
from sklearn.datasets import make_classification
# import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import cross_val_score

n_samples=500
X,Y=make_classification(n_samples=n_samples,n_features=3,
						n_informative=3,n_redundant=0,
						n_classes=3,n_clusters_per_class=1)
dtc=DecisionTreeClassifier()
print('using DecisionTreeClassifier:')
print(cross_val_score(dtc,X,Y,cv=10,scoring='accuracy').mean())

dtc.fit(X,Y)
'''
with open('dtc.dot','w') as df:
	df=export_graphviz(dtc,out_file=df,
					   feature_names=['A','B','C'],
					   class_names=['C1','C2','C3'])
'''

print('feature importances given by decision tree:')
print(dtc.feature_importances_)
print('name of feature importances:')
print(np.argsort(dtc.feature_importances_))