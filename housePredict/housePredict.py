#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile,chi2,f_regression
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

rawData=pd.read_csv('train.csv')
testData=pd.read_csv('test.csv')

testId=testData['Id']
testData=testData.drop(['Id'],axis=1)

X=rawData.drop(['SalePrice','Id'],axis=1)
Y=rawData['SalePrice']

miss_count=X.isnull().sum().sort_values(ascending=False)
miss_rate=miss_count/len(Y)
miss_data=pd.concat([miss_count,miss_rate],axis=1,keys=['count','ratio'])

X=X.drop(miss_data[miss_data['count']>1].index,axis=1)
X_test=testData.drop(miss_data[miss_data['count']>1].index,axis=1)
Y=Y.drop(X.loc[X['Electrical'].isnull()].index)
X=X.drop(X.loc[X['Electrical'].isnull()].index)
'''
quantity=[attr for attr in X.columns if X.dtypes[attr]!='object']
quality=[attr for attr in X.columns if X.dtypes[attr]=='object']

for c in quantity:
	X[c]=X[c].fillna(X[c].median(),inplace=True)
	X_test[c]=X_test[c].fillna(X[c].median(),inplace=True)
'''
X.fillna('UNKNOWN',inplace=True)
X_test.fillna('UNKNOWN',inplace=True)
dv=DictVectorizer()
X_train=dv.fit_transform(X.to_dict(orient='record'))
X_test=dv.transform(X_test.to_dict(orient='record'))
#X_train,X_test,Y_train,Y_test=train_test_split(X_train,Y,test_size=0.25)
'''
lr=LinearRegression(normalize=True)
rr=RANSACRegressor(lr)
rr.fit(X_train,Y_train)
print('using RANSACRegressor:')
print(rr.score(X_test,Y_test))

svc=SVC(kernel='linear')
svc.fit(X_train,Y_train)
print('using svm')
print(svc.score(X_test,Y_test))
'''

ss=StandardScaler()
X_scale=ss.fit_transform(X_train.todense())
X_test_scale=ss.transform(X_test.todense())

ls=LassoCV(alphas=(10,3,1),normalize=True)
ls.fit(X_scale,Y)
ls1=LassoCV(alphas=ls.alphas_,normalize=True)
ls1.fit(X_scale,Y)
'''
rr= RandomForestRegressor(max_depth=30, n_estimators=800, max_features = 100, oob_score=True, random_state=1234)
#cv_score = cross_val_score(rf_test, train_d.drop('SalePrice', axis = 1), train_d['SalePrice'], cv = 5, n_jobs = -1)
#print('using RandomForestRegressor')
#print(rf.score(X_test,Y_test))
rr.fit(X_train,Y)

#print(rf.feature_importances_)
'''
#print(cross_val_score(lr,X_train,Y,scoring='neg_mean_squared_error',cv=5))
predict=np.array(ls1.predict(X_test_scale))
final=np.hstack((testId.reshape(-1,1),predict.reshape(-1,1)))
np.savetxt('new.csv',final,delimiter=',',fmt='%d')








