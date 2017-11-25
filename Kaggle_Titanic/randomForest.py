#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectPercentile,chi2,f_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score


rawData=pd.read_csv('train.csv')
testData=pd.read_csv('test.csv')
y=rawData['Survived']
X=rawData.drop(['PassengerId','Survived','Name'],axis=1)
X_test_Id=testData['PassengerId']
X_test=testData.drop(['PassengerId','Name'],axis=1)


X['Age'].fillna(X['Age'].mean(),inplace=True)
X_test['Age'].fillna(X['Age'].mean(),inplace=True)
X.fillna('UNKNOWN',inplace=True)
X_test.fillna('UNKNOWN',inplace=True)

#X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.25)









dv=DictVectorizer()
X_train=dv.fit_transform(X.to_dict(orient='record'))
X_test=dv.transform(X_test.to_dict(orient='record'))


#lr=LogisticRegression()
#lr.fit(X,y)
#predict=lr.predict(X_test)
sp=SelectPercentile(chi2,percentile=92)
X_train_sp=sp.fit_transform(X_train,y)
X_test_sp=sp.transform(X_test)




	
rfc=RandomForestClassifier(n_estimators=30, max_features='sqrt')

 


rfc.fit(X_train_sp,y)



predict=rfc.predict(X_test_sp)

final=np.hstack((X_test_Id.reshape(-1,1),predict.reshape(-1,1)))
#finalSub=np.vstack((name.reshape(1,2),final))
np.savetxt('new.csv', final, delimiter = ',',fmt='%d')  
#print(Predict)
