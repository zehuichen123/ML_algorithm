#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest

import numpy as np
from sklearn.naive_bayes import BernoulliNB,MultinomialNB,GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer

nb_samples=300
X,Y=make_classification(n_samples=nb_samples,n_features=2,
						n_informative=2,n_redundant=0)
'''
for i,y in enumerate(Y):
	if y==0:
		plt.scatter(X[i,0],X[i,1],c='red',s=5)
	else:
		plt.scatter(X[i,0],X[i,1],c='blue',s=5)

'''
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)


bnb=BernoulliNB(binarize=0.0)
bnb.fit(X_train,Y_train)
bnbScore=bnb.score(X_test,Y_test)
print('BernoulliNB scores:')
print(bnbScore)
'''
plt.title('BernoulliNB scores '+str(bnbScore))
plt.show()
'''

data = [
      {'house': 100, 'street': 50, 'shop': 25, 'car': 100, 'tree': 20},
      {'house': 5, 'street': 5, 'shop': 0, 'car': 10, 'tree': 500, 'river': 1}
]
dv=DictVectorizer()
X=dv.fit_transform(data)
Y=np.array([1,0])
print(data)
print('after using DictVectorizer:')
print(X)

mnb=MultinomialNB()
print(mnb.fit(X,Y))

test_data = data = [
      {'house': 80, 'street': 20, 'shop': 15, 'car': 70, 'tree': 10, 'river':
   1},
      {'house': 10, 'street': 5, 'shop': 1, 'car': 8, 'tree': 300, 'river': 0}
]
print('test data is:')
print(test_data)
print('using MultinomialNB:')
print(mnb.predict(dv.fit_transform(test_data)))


# continue to use Bernoilli data
gnb=GaussianNB()
gnb.fit(X_train,Y_train)
print('GuassianNB scores:')
print(gnb.score(X_test,Y_test))

lr=LogisticRegression()
lr.fit(X_train,Y_train)
print('LogisticRegression scores:')
print(lr.score(X_test,Y_test))















