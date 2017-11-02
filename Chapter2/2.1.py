#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
nb_sample=100
Y=np.arange(0,100,1)
X_data=np.random.normal(loc=0.0,scale=np.sqrt(2.0),size=nb_sample)

def negative_log_likelihood(v):
	l=0
	f1=1.0/np.sqrt(2.0*np.pi*v[1])
	f2=2.0*v[1]
	for x in X_data:
		l+=np.log(f1*np.exp(-np.square(x-v[0])/f2))
	return -l

print(minimize(fun=negative_log_likelihood,x0=[0.0,1.0]))
plt.plot(Y,X_data)
plt.show()
