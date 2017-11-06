#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import confusion_matrix,precision_score
# import matplotlib.pyplot as plt
import numpy as np

iris=load_iris()
print(iris.target)
X_train,X_test,Y_train,Y_test=train_test_split(
				iris.data,
				iris.target,
				test_size=0.2)
lr=LogisticRegression()
lr.fit(X_train,Y_train)
print('LogisticRegression scores:')
print(cross_val_score(lr,X_test,Y_test,cv=10,scoring='accuracy').mean())

# using confusion matrix
print('the confusion matrix is:')
print(confusion_matrix(y_true=Y_train,y_pred=lr.predict(X_train)))

# ************* Attention ***************
# iris dataset is not a binary dataset,so 
# it can not implement precision_score with
# binary paramter for average, so you need 
# to specify the average to 'micro'(though
# I don't know what is micro...)
print('using the precision_score:')
print(precision_score(Y_test,lr.predict(X_test),average='micro'))