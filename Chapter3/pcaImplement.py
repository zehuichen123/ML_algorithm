#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
import numpy as np
import matplotlib.pyplot as plt
data=np.array([[0.9,0.8],[2.01,1.9],[1.8,1.73],[3.5,3.2],
		   	  [4.2,4.5],[6.0,5.73],[6.8,7.0],[4.33,4.5],
		   	  [5.4,5.23],[4.32,4.5],[2.33,2.4],[6.23,6.4]])

#def showPlot(dataMat):
#	plt.scatter(dataMat[:,0],dataMat[:,1])
#	plt.show()
def getSubMean(dataMat):
	meanVal=np.mean(dataMat,axis=0)
	return dataMat-meanVal,meanVal

def getCovMat(dataMat):
	return np.cov(dataMat,rowvar=0)  # here if rowvar=0 means a row is a sample, or a column

def getEigen(covMat):
	eigenVal,eigenVec=np.linalg.eig(covMat)
	return eigenVal,eigenVec

def getNBest(eigenVal,eigenVec,n):
	eigenValIndex=np.argsort(eigenVal)
	n_eigenValIndex=eigenValIndex[-1:-(n+1):-1]
	n_eigenVec=eigenVec[:,n_eigenValIndex]
	return n_eigenVec

print('the origin data is:')
print(data)
plt.subplot(2,2,1)
plt.scatter(data[:,0],data[:,1],marker='x')
dataSubMean,meanVal=getSubMean(data)
print('get the mean of each feature(here is X,Y):')
print(meanVal)
print('first step :substract the mean:')
print(dataSubMean)

plt.subplot(2,2,2)
plt.scatter(dataSubMean[:,0],dataSubMean[:,1],marker='x')

dataCovMat=getCovMat(dataSubMean)
print('second step :get the covariance matrix:')
print(dataCovMat)

dataEigenVal,dataEigenVec=getEigen(dataCovMat)
print('third step :get the eigenVectors and eigenValues:')
print('eigenVectors is:')
print(dataEigenVec)
print('eigenValues is')
print(dataEigenVal)

# you can plot the two lines here in the plot

n_dataEigenVec=getNBest(dataEigenVal,dataEigenVec,2)
# here we can plot the largest eigenvector  in the original plot
largestVec=n_dataEigenVec[0,:]
print('the largetestVec is:')
print(largestVec)
vecX=np.linspace(-4,4,100)
vecY=vecX*largestVec[1]/largestVec[0]
plt.plot(vecX,vecY)
print('fifth step: sort the n largetest eigenVectors:')
print(n_dataEigenVec)

finalData=np.dot(n_dataEigenVec.T,dataSubMean.T)
print('the final data is:')
print(finalData)
plt.subplot(2,2,3)
plt.scatter(finalData[0,:],finalData[1,:],marker='x')
plt.ylim(-4,4)

print('show the pca compression')
# of course, if we use pca to compress the dimensionals
n_dataEigenVec1=getNBest(dataEigenVal,dataEigenVec,1)
# here we compress the dataset to 1 dimensional
finalData=np.dot(n_dataEigenVec1.T,dataSubMean.T)
# now we want to get the original data
origData=np.dot(n_dataEigenVec1,finalData)+meanVal.reshape(2,-1)
print('get the original data:')
print(origData)
plt.subplot(2,2,4)
plt.scatter(origData[0,:],origData[1,:],marker='x')

plt.show()