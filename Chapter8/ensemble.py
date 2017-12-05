#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
import numpy as np
from sklearn.datasets import make_classification,load_digits
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.model_selection import cross_val_score

digits=load_digits()
nb_classifications=100

accuracy_rf=[]

for i in range(1,nb_classifications):
	a=cross_val_score(RandomForestClassifier(n_estimators=i),
		digits.data,digits.target,scoring='accuracy',cv=10).mean()
	accuracy_rf.append(a)

accuracy_et=[]

for i in range(1,nb_classifications):
	b=cross_val_score(ExtraTreesClassifier(n_estimators=i),
		digits.data,digits.target,scoring='accuracy',cv=10).mean()
	accuracy_et.append(b)

accuracy_ab=[]

for i in range(1,nb_classifications):
	c=cross_val_score(AdaBoostClassifier(n_estimators=i),
		learning_rate=1.0,digits.data,digits.target,scoring='accuracy',cv=10).mean()
	accuracy_ab.append(c)

print('AdaBoostClassifier with n_estimators and learning_rate=1 scores:')
print(accuracy_ab)
print('RandomForestClassifier with n_estimators scores:')
print(accuracy_rf)
print('ExtraTreesClassifier with n_estimators scores:')
print(accuracy_et)
plt.plot(range(1,nb_classifications),accuracy_et,c='red',label='ExtraTreesClassifier')
plt.plot(range(1,nb_classifications),accuracy_rf,c='blue',label='RandomForestClassifier')
plt.plot(range(1,nb_classifications),accuracy_ab,c='green',label='AdaBoostClassifier')
plt.ylabel('Accuracy')
plt.xlabel('Number of trees')
plt.legend(loc='upper left')
plt.show()
