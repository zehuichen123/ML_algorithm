#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile,chi2,f_regression,SelectKBest
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression,RANSACRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

rawData=pd.read_csv('train.csv')
testData=pd.read_csv('test.csv')
testId=testData['Id']
testData=testData.drop(['Id'],axis=1)

Y=rawData['SalePrice']
rawData=rawData.drop(['SalePrice','Id'],axis=1)

rawData_d=pd.get_dummies(rawData)

keep_cols=rawData_d.select_dtypes(include=['number']).columns
train_d=rawData_d[keep_cols]

train_d=train_d.fillna(train_d.mean())

test_d=pd.get_dummies(testData)


test_d=test_d.fillna(test_d.mean())

for col in keep_cols:
    if col not in test_d:
        test_d[col] = 0
test_d = test_d[keep_cols]

X_test=test_d
X_train=train_d


ss=StandardScaler()
X_scale=ss.fit_transform(X_train)
X_test_scale=ss.transform(X_test)

ls1=LassoCV()


#skb=SelectKBest(f_regression)
#X_scale_k=skb.fit_transform(X_scale,Y)
print(cross_val_score(ls1,X_scale,Y,cv=5))

'''
ls1=LassoCV()
rr=RANSACRegressor(ls1)
rr.fit(X_scale,Y)
#print(cross_val_score(lr,X_train,Y,scoring='neg_mean_squared_error',cv=5))
predict=np.array(rr.predict(X_test_scale))
final=np.hstack((testId.reshape(-1,1),predict.reshape(-1,1)))
np.savetxt('new.csv',final,delimiter=',',fmt='%d')
''' 