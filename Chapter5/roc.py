#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import numpy as np

sampleNum=200
X,Y=make_classification(
		n_samples=sampleNum,
		n_features=2,
		n_informative=2,
		n_redundant=0,
		n_clusters_per_class=1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

lr=LogisticRegression()
lr.fit(X_train,Y_train)
Y_scores=lr.decision_function(X_test)

fpr,tpr,threshold=roc_curve(Y_test,Y_scores)
aucArea=auc(fpr,tpr)

plt.plot(fpr,tpr,c='r',label='LogisticRegression(AUC:%.2f)'%aucArea)
plt.plot([0,1],[0,1],c='blue',linestyle='--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()
