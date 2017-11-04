#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: loveSnowBest
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
digits=load_digits()
plt.gray()
plt.subplot(1,2,1)
plt.xlabel('the origin pic')
plt.imshow(digits.images[0])

# use pca to compress the dimentionals in pic
pca=PCA(n_components=36,whiten=True)
X_pca=pca.fit_transform(digits.data)
X_rebuilt=pca.inverse_transform(X_pca)
plt.subplot(1,2,2)
plt.xlabel('the rebuilt pic')
plt.imshow(X_rebuilt[0].reshape(8,8))
plt.show()
