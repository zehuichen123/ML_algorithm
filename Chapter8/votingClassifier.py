#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
import numpy as np
from sklearn.datasets import make_classification,load_digits
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

nb_samples=500
X,Y=make_classification(n_samples=nb_samples,n_features=2,
	n_redundant=0,n_classes=2)

lr=LogisticRegression()
svc=SVC(kernel='poly',probability=True)
dt=DecisionTreeClassifier()

classifier=[('lr',lr),
			('svc',svc),
			('dt',dt)]

vc=VotingClassifier(estimators=classifier,voting='hard')

print('LogisticRegression scores:')
print(cross_val_score(lr,X,Y,cv=10).mean())
print('SVC scores:')
print(cross_val_score(svc,X,Y,cv=10).mean())
print('DecisionTreeClassifier scores:')
print(cross_val_score(dt,X,Y,cv=10).mean())
print('VotingClassifier scores:')
print(cross_val_score(vc,X,Y,cv=10).mean())

# for soft voting
weights=[1.5,0.5,0.75]
vc=VotingClassifier(estimators=classifier,weights=weights,
	voting='soft')
