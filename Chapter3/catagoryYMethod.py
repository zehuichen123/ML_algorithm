#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction import DictVectorizer,FeatureHasher
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# scikit-learn toy datasets
boston=load_boston()
X=boston.data
Y=boston.target
print('the size of X is '+str(X.shape))
print('the size of Y is '+str(Y.shape))

# generate new randomState generator
rs=check_random_state(1000)

# datasets train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=rs)
print('the size of X_test is'+ str(X_test.shape))
print('the size of X_train is'+ str(X_train.shape))

X = np.random.uniform(0.0,1.0, size=(10,2))
Y = np.random.choice(('Male','Female','Unknown'), size=(10))
print('the sex array is:')
print(Y)

# manage catagorical data Method_1
le=LabelEncoder()
yt=le.fit_transform(Y)
print('the encode array by labelEncoder is:')
print(yt)

# how to inverse the encode/ say decode:
decoded_yt=[le.classes_[i] for i in yt]
print('the decode array is:')
print(decoded_yt)

# manage catagorical data Method_2
lb=LabelBinarizer()
yb=lb.fit_transform(Y)
print('the encode array by LabelBinarizer is')
print(yb)

# here, how to inverse the encode/ say decode different with method1
decoded_yb=lb.inverse_transform(yb)
print('the decode array is:')
print(decoded_yb)

# here we show another method for structure like a list of dictionaries
data = [
      { 'feature_1': 10.0, 'feature_2': 15.0 },
      { 'feature_1': -5.0, 'feature_3': 22.0 },
      { 'feature_3': -2.0, 'feature_4': 10.0 }
]

# use DictVectorizer
dv=DictVectorizer()
Ydict=dv.fit_transform(data)
print('the encode array by DictVectorizer is:')
print(Ydict)
print('after to dense array:')
print(Ydict.todense())
print('dv class `vocabulary_` is:')
print(dv.vocabulary_)

# use FeatureHasher
dh=FeatureHasher()
Yhash=dh.fit_transform(data)
print('the encode array by FeatureHasher is:')
print(Yhash)
print('after to dense array:')
YhashArray=Yhash.todense()
print(YhashArray)
print('the shape of dense array by FeatureHasher is:')
print(YhashArray.shape)

# use one-hot encoder to extend
data = [
      [0, 10],
      [1, 11],
      [2, 8],
      [3, 12],
      [0, 15]]
oh=OneHotEncoder(categorical_features=[0])
Yoh=oh.fit_transform(data)
print('the encode array by OneHotEncoder is:')
print(Yoh)
print('the to dense array is:')
print(Yoh.todense())

# use feature indice to show onehot area
print(oh.feature_indices_)








