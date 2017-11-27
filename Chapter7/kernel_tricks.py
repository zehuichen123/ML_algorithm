#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest

import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.model_selection import GridSearchCV

nb_samples=500
X,Y=make_circles(n_samples=nb_samples,noise=0.1)

for i in range(nb_samples):
	if Y[i] == 1:
		plt.scatter(X[i,0],X[i,1],c='green',s=3)
	else:
		plt.scatter(X[i,0],X[i,1],c='blue',s=3)

grid_param={
	'kernel':['linear','poly','rbf','sigmoid'],
	'C':[0.1,0.2,0.4,0.5,1,1.5,1.8,2.0,2.5,3.0]
}
gs=GridSearchCV(estimator=SVC(),param_grid=grid_param,
				scoring='accuracy',cv=5)

gs.fit(X,Y)
#print(gs.best_estimator_)
best_svm=gs.best_estimator_
sv=best_svm.support_vectors_
for i in sv:
	plt.scatter(i[0],i[1],c='red',s=3)

print(gs.best_score_)

plt.show()





