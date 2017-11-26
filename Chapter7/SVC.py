#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest

import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

nb_samples=500
X,Y=make_classification(n_samples=nb_samples,n_features=2,
						n_informative=2,n_redundant=0,n_clusters_per_class=1)

for i in range(nb_samples):
	if Y[i]==1:
		plt.scatter(X[i,0],X[i,1],c='green',s=3)
	else:
		plt.scatter(X[i,0],X[i,1],c='blue',s=3)

svc=SVC(kernel='linear')
print('linear svm scores:')
print(cross_val_score(svc,X,Y,scoring='accuracy',cv=10).mean())

svc.fit(X,Y)
for i in svc.support_vectors_:
	plt.scatter(i[0],i[1],color='red',s=3)

a=svc.coef_[0,0]
b=svc.coef_[0,1]
c=svc.intercept_
X=np.array([-1,1])
Y=np.array([a/b-c/b,-a/b-c/b])
plt.plot(X,Y,c='black')
plt.show()








