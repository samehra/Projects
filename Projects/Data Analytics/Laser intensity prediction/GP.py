# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 19:15:10 2018

@author: saura
"""
import matplotlib.pylab as plt
import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.io
import pandas as pd

#np.random.seed(1)
fifo = scipy.io.loadmat('FiFo.mat')

X = fifo['X']
y = fifo['Y']
X = X[0][0].T
y = y[0][0].T
data = np.hstack((X,y))

np.savetxt("./data.csv", data, delimiter=",")
        
#x = [1.]
#y = [np.random.normal(scale=σ_0)]

#X = [[14, 25],[28, 25],[41, 25],[55, 25],[69, 25],[83, 25],[97, 25],[111, 25],[125, 25],[139, 25],[153, 25],[14, 27],[28, 27],[41, 27],[55, 27]]
#y = [620,1315,2120,2600,3110,3535,3935,4465,4530,4570,4600,625,1215,2110,2805]
#y_ = np.hstack((yarr.reshape(-1,1),yarr.reshape(-1,1)))
#diabetes = np.loadtxt("diabetes.data.txt", skiprows=1)   #http://www.stanford.edu/~hastie/Papers/LARS/diabetes.data
#x = diabetes[:, :-1]
#y = diabetes[:, -1]
time_index = np.arange(1,len(X)+1,1).reshape(-1,1)
X = np.hstack((X, time_index))
y = np.hstack((y, time_index))
#Xarr = np.asarray(X,dtype = np.float64)
#yarr = np.asarray(y,dtype = np.float64)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.99,random_state=1)

X_train = X_train[X_train[:,5].argsort()[::1]]
X_test = X_test[X_test[:,5].argsort()[::1]]
y_train = y_train[y_train[:,1].argsort()[::1]]
y_test = y_test[y_test[:,1].argsort()[::1]]

#X_train = X[0:900,:]; X_test = X[900:1000,:]
#y_train = y[0:900,:]; y_test = y[900:1000,:]
kernel = ConstantKernel() + WhiteKernel(noise_level=1) + RBF(length_scale = 5*np.ones(X_train.shape[1]), length_scale_bounds=(1e-3, 1e3))

gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 3, normalize_y = 'T', random_state = 1)
gp.fit(X_train, y_train[:,0])
gp.kernel_
gp.log_marginal_likelihood_value_
y_pred_train, sigma_train = gp.predict(X_train, return_std=True)
y_pred, sigma = gp.predict(X_test, return_std=True)
train_residual = y_pred_train-y_train[:,0]
RMSE = sqrt(mean_squared_error(y_test[:,0], y_pred))
RMSE_train = sqrt(mean_squared_error(y_train[:,0], y_pred_train))
RMSE
RMSE_train
y_pred_plot, sigma_plot = gp.predict(X, return_std=True)
# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure(1)
plt.subplot(211)
plt.plot(Xarr, y, 'r.', markersize=10, label=u'Observations')
plt.subplot(212)
plt.plot(X_test, y_test, 'r.', markersize=10, label=u'Test Data')
plt.figure(2)
plt.subplot(221)
plt.plot(X_test[:,0], y_test, 'b.', X_test[:,0], y_pred, 'r.', label=u'Prediction')
plt.subplot(222)
plt.plot(X_test[:,1], y_test, 'b.', X_test[:,1], y_pred, 'r.', label=u'Prediction')


plt.fill(np.concatenate([X_test, X_test[::-1]]), 
         np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]), 
         alpha=.5, fc='grey', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()

plt.figure(1)
plt.subplot(611)
plt.plot(X[:,0])
plt.subplot(612)
plt.plot(X[:,1])
plt.subplot(613)
plt.plot(X[:,2])
plt.subplot(614)
plt.plot(X[:,3])
plt.subplot(615)
plt.plot(X[:,4])
plt.subplot(616)
plt.plot(y)

#y_plot = np.hstack((y_test, y_pred.reshape(-1,1)))
X_axis_train = y_train[y_train[:,1].argsort()[::1]][:,1]
X_axis_test = y_test[y_test[:,1].argsort()[::1]][:,1]
X_axis = np.arange(1, len(y)+1,1)
plt.figure(2)
plt.plot(X_axis, y_pred_plot, 'b-', label=u'Prediction')
plt.fill(np.concatenate([X_axis, X_axis[::-1]]), 
         np.concatenate([y_pred_plot - 1.9600 * sigma_plot, (y_pred_plot + 1.9600 * sigma_plot)[::-1]]), 
         alpha=.5, fc='grey', ec='None', label='95% confidence interval')
plt.plot(X_axis_train, y_train[:,0],'r.', markersize=5, label=u'Observations')

# ================================================================================== #
# Building PyFLUX Gaussian Process Model

import pyflux as pf
from pyflux import GPNARX

#growthdata = pd.read_csv('http://www.pyflux.com/notebooks/GDPC1.csv')
#USgrowth = pd.DataFrame(np.diff(np.log(growthdata['VALUE']))[149:len(growthdata['VALUE'])])
#USgrowth.index = pd.to_datetime(growthdata['DATE'].values[1+149:len(growthdata)])
#USgrowth.columns = ['US Real GDP Growth']



plt.figure(1)
plt.subplot(511)
plt.plot(X[:,0])
plt.subplot(512)
plt.plot(X[:,1])
plt.subplot(513)
plt.plot(X[:,2])
plt.subplot(514)
plt.plot(X[:,3])
plt.subplot(515)
plt.plot(X[:,4])

rng = pd.date_range('1/1/2018', periods=1000, freq='S')
train = pd.DataFrame(X[:,0], index=rng,columns=['Values1'])
model = pf.GPNARX(train,ar=10,kernel=pf.SquaredExponential())
fit = model.fit()
fit.summary()