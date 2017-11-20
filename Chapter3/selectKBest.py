#!/usr/bin/env python3
# -*- coding: utf-8 -*-





from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler,RobustScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest,f_regression


data=np.array([[1,2,np.nan],
			  [2,np.nan,4],
			  [-1,4,2]])
print('the origin dataset is:')
print(data)

# managing the missing data
imp=Imputer(strategy='mean') # other strategies:median,most_frequent
data_imp=imp.fit_transform(data)
print('using mean to managing missing data:')
print(data_imp)

data=np.random.uniform(-10,10,size=(100,2))
plt.subplot(2,2,1)
plt.scatter(data[:,1],data[:,0])
# using StandardScaler to normalize the dataset
ss=StandardScaler()
dataSS=ss.fit_transform(data)
plt.subplot(2,2,2)
plt.scatter(dataSS[:,1],dataSS[:,0])

# using RobustScaler to normalize the dataset
rs1=RobustScaler(quantile_range=(15,85))
dataRs1=rs1.fit_transform(data)
plt.subplot(2,2,3)
plt.scatter(dataRs1[:,1],dataRs1[:,0])

rs2=RobustScaler(quantile_range=(30,60))
dataRs2=rs2.fit_transform(data)
plt.subplot(2,2,4)
plt.scatter(dataRs2[:,1],dataRs2[:,0])

plt.show()

# an example about how to use Normalizer
data=np.array([1,2])
print('the origin data is:')
print(data)
nl=Normalizer(norm='max')
dataNl=nl.fit_transform(data.reshape(1,-1)) # here Normalizer want to get 2-D array
print('after normalizer data with max norm:') # there are max,l1,l2 options for chooing
print(dataNl)

# using VarianceThreshold
data1=np.random.uniform(-10,10,size=(3,2))
data2=np.random.uniform(-1,1,size=(3,1))
data=np.hstack((data1,data2))
print('the origin data is:')
print(data)

vt=VarianceThreshold(threshold=1)
dataVt=vt.fit_transform(data)
print('after variance select:')
print(dataVt)

# using SelectKBest to find the top best features to be used
regr_data=load_boston()
print('the origin regr_data shape is:'+str(regr_data.data.shape))
skb=SelectKBest(f_regression)
regr_dataSkb=skb.fit_transform(regr_data.data,regr_data.target)
print('after selections, the data shape is:'+str(regr_dataSkb.shape))
print('the scores for each feature:')
print(skb.scores_)

