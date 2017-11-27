#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest

import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import NuSVC
from sklearn.model_selection import GridSearchCV

nb_samples=500
X,Y=make_classification(n_samples=nb_samples,n_features=2,
						n_informative=2,n_redundant=0,n_clusters_per_class=1)

for i in range(nb_samples):
	if Y[i] == 1:
		plt.scatter(X[i,0],X[i,1],c='green',s=3)
	else:
		plt.scatter(X[i,0],X[i,1],c='blue',s=3)

grid_param={
	'nu':np.arange(0.05,0.5,0.05)
}
gs=GridSearchCV(estimator=NuSVC(),param_grid=grid_param,
				scoring='accuracy',cv=5)

gs.fit(X,Y)
#print(gs.best_estimator_)
best_svm=gs.best_estimator_
print(best_svm)
sv=best_svm.support_vectors_
for i in sv:
	plt.scatter(i[0],i[1],c='red',s=3)

print(gs.best_score_)

plt.show()





