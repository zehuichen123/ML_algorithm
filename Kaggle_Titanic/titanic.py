# -*- coding: utf-8 -*-
# A Simple Kaggle Tutorial
# author: lovesnowbest

# import tools you need: scikit-learn, pandas, numpy
# there are some other useful tools: lightgbm, xgboost, tensorflow
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectPercentile,chi2,f_regression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

# read your data from files
rawData=pd.read_csv('train.csv')
testData=pd.read_csv('test.csv')

# Feature Engineering
# Usually based on your background knowledge or your inner view on dataset 
rawData['Family'] = rawData["Parch"] + rawData["SibSp"]
rawData['Family'].loc[rawData['Family'] > 0] = 1
rawData['Family'].loc[rawData['Family'] == 0] = 0

testData['Family'] = testData["Parch"] + testData["SibSp"]
testData['Family'].loc[testData['Family'] > 0] = 1
testData['Family'].loc[testData['Family'] == 0] = 0

rawData = rawData.drop(['SibSp','Parch'], axis=1)
testData = testData.drop(['SibSp','Parch'], axis=1)

# First split the dataset rawData into two part: X and y(survived 0/1).
# Feature `name` seems useless for our prediction, just drop it out.
y = rawData['Survived']
X = rawData.drop(['PassengerId','Survived','Name'],axis=1)
X_test_Id = testData['PassengerId']
X_test = testData.drop(['PassengerId','Name'],axis=1)

# Handle with Missing Values
# first let's have a look at the distribution of missing values
total = X.isnull().sum().sort_values(ascending=False)
percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data.head()

# Handle with NULL value
X['Age'].fillna(X['Age'].mean(),inplace=True)
X_test['Age'].fillna(X['Age'].mean(),inplace=True)
X.fillna('UNKNOWN',inplace=True)
X_test.fillna('UNKNOWN',inplace=True)

#X.head()

# Tackle with categorical features(also the UNKNOWN value)
dv = DictVectorizer()
X_train = dv.fit_transform(X.to_dict(orient='record'))
X_test = dv.transform(X_test.to_dict(orient='record'))

# select the best k percentiles features to get a better score
sp = SelectPercentile(chi2,percentile=92)
X_train_sp = sp.fit_transform(X_train,y)
X_test_sp = sp.transform(X_test)

rf_params = {
    'n_estimators': 800,
    'max_features' : 'sqrt',
}
rf = RandomForestClassifier(**rf_params)
rf.fit(X_train_sp,y)
predict_rf = rf.predict(X_test_sp)

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}
et = ExtraTreesClassifier(**et_params)
et.fit(X_train_sp,y)
predict_et = et.predict(X_test_sp)

# AdaBoost parameters
ada_params = {
    'n_estimators': 800,
    'learning_rate' : 0.5
}

ada = AdaBoostClassifier(**ada_params)
ada.fit(X_train_sp,y)
predict_ada = ada.predict(X_test_sp)

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 800,
     #'max_features': 0.2,
    'max_depth': 5,
    #'min_samples_leaf': 2,
}

gb = GradientBoostingClassifier(**gb_params)
gb.fit(X_train_sp,y)
predict_gb = gb.predict(X_test_sp)

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }
svc = SVC(**svc_params)
svc.fit(X_train_sp,y)
predict_svc = svc.predict(X_test_sp)

# ensemble the prediction by all classifiers
# you can also use VotingClassifier in sklearn
one = np.ones(X_test.shape[0])
zero = np.ones(X_test.shape[0])
for i,j in enumerate(predict_rf):
	if j == 1:
		one[i] += 1
	else:
		zero[i] += 1
for i,j in enumerate(predict_ada):
	if j == 1:
		one[i] += 1
	else:
		zero[i] += 1
for i,j in enumerate(predict_gb):
	if j == 1:
		one[i] += 1
	else:
		zero[i] += 1
for i,j in enumerate(predict_et):
    if j == 1:
        one[i] += 1
    else:
        zero[i] += 1
for i,j in enumerate(predict_svc):
    if j == 1:
        one[i] += 1
    else:
        zero[i] += 1

predict = []
for i,j in zip(one,zero):
	if i < j:
		predict.append(0)
	else:
		predict.append(1)

predict_pd = pd.DataFrame(predict, columns=['Survived'])

final_predict = pd.concat([X_test_Id, predict_pd], axis=1)

#final_predict.head()

# save your prediction to file predict.csv
final_predict.to_csv('predict.csv', index=False)

#from google.colab import files
#files.download('predict.csv')
# BTW, upload a file to colab: `files.upload()`

