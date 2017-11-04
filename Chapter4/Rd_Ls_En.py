#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.linear_model import RidgeCV,Lasso,LassoCV
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.model_selection import cross_val_score

diabetes=load_diabetes()
lr=LinearRegression(normalize=True)
rd=Ridge(0.005,normalize=True)

print('the lr score:')
lr_scores=cross_val_score(lr,diabetes.data,diabetes.target,cv=10)
print(lr_scores.mean())
print('the Ridge method score:')
rd_scores=cross_val_score(rd,diabetes.data,diabetes.target,cv=10)
print(rd_scores.mean())

print('using RidgeCV:')
rd_cv=RidgeCV(alphas=(1,0.3,0.1,0.03,0.01,0.005,0.001),normalize=True)
rd_cv.fit(diabetes.data,diabetes.target)
print('find the best alphas is:'+str(rd_cv.alpha_))


print('using lasso score:')
ls=Lasso(alpha=0.01,normalize=True)
ls_score=cross_val_score(ls,diabetes.data,diabetes.target,cv=10)
print(ls_score.mean())
print('using lassoCV to find the best alpha:')
ls_cv=LassoCV(alphas=(1,0.3,0.1,0.03,0.01,0.005,0.001),normalize=True)
ls_cv.fit(diabetes.data,diabetes.target)
print(ls_cv.alpha_)


print('using ElasticNet and ElasticNetCV')
en=ElasticNet(alpha=0.001,l1_ratio=0.75,normalize=True)
en_scores=cross_val_score(en,diabetes.data,diabetes.target,cv=10)
print('using ElasticNet score:')
print(en_scores.mean())
en_cv=ElasticNetCV(alphas=(0.1, 0.01, 0.005, 0.0025, 0.001),
				   l1_ratio=(0.1, 0.25, 0.5, 0.75, 0.8),
				   normalize=True)
en_cv.fit(diabetes.data,diabetes.target)
print('find the best ElasticNet alpha is:'+str(en_cv.alpha_))
print('find the best ElasticNet l1_ratio is:'+str(en_cv.l1_ratio_))