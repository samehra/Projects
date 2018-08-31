# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 16:52:44 2018

@author: Saurabh Mehra
"""

# Import required libraries
import os # used for manipulating directory paths
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from scipy import optimize
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


#  training data stored in arrays X, y
nba_train = pd.read_csv('./Data/NBA_train.csv')
nba_test = pd.read_csv('./Data/NBA_test.csv')

def to_matrix(df,start,end):
    X = df.as_matrix(columns = df.columns[start-1:end])
    return X

def normalize(X):
    m, n = X.shape
    X_mean = np.zeros((1,n))
    X_std = np.zeros((1,n))
    X_norm = X
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean)/X_std
    
    return X_norm, X_mean, X_std

def CostFunc(Theta, X, Y, num_rows, num_features, lamda):
    Theta = Theta.reshape(num_features,1)
    J = (1/(2*num_rows))*np.sum((X.dot(Theta) - Y)**2) + (lamda/(2*num_rows))*np.sum(Theta[1:]**2)

    return J

def GradFunc(Theta, X, Y, num_rows, num_features, lamda):
    Theta = Theta.reshape(num_features,1)
    Theta_grad = np.zeros(Theta.shape)
    Theta_grad[0] = (1/num_rows)*np.sum(X.dot(Theta) - Y)
    Theta_grad[1:] = (1/num_rows)*((X[:,1:].T).dot(X.dot(Theta) - Y)) + (lamda/num_rows)*Theta[1:]
    Theta_grad = Theta_grad.flatten()

    return Theta_grad

def batch_gradient_descent(X, Y, Theta, alpha, lamda, num_rows, num_features, num_iters):
    m = Y.size
    j_history = np.zeros((num_iters+1))
    Theta = Theta.reshape(num_features,1)
    Theta_grad = np.zeros(Theta.shape)
    j_history[0] = (1/(2*m))*np.sum((X.dot(Theta) - Y)**2) + (lamda/(2*num_rows))*np.sum(Theta[1:]**2)

    for i in range(num_iters):
        Theta_grad[0] = (1/num_rows)*np.sum(X.dot(Theta) - Y)
        Theta_grad[1:] = (1/num_rows)*((X[:,1:].T).dot(X.dot(Theta) - Y)) + (lamda/num_rows)*Theta[1:]
        Theta -= alpha * Theta_grad 
        j_history[i+1] = (1/(2*m))*np.sum((X.dot(Theta) - Y)**2) + (lamda/(2*num_rows))*np.sum(Theta[1:]**2)

    return Theta, j_history

# Input data and convert to matrix form
X = to_matrix(nba_train,8,21)
Y = to_matrix(nba_train,7,7)
X_test = to_matrix(nba_test,8,21)
Y_test = to_matrix(nba_test,7,7)

m,n = X.shape
m_test, n_test = X_test.shape

# Normalize data
X_norm, X_mean, X_std = normalize(X)
X_test_norm, X_test_mean, X_test_std = normalize(X_test)

## Add intercept term to X
X_0 = np.concatenate((np.ones((m, 1)), X_norm), axis=1)
X_test_0 = np.concatenate((np.ones((m_test, 1)), X_test_norm), axis=1)

# Initial value of the Theta parameter
Theta_0 = np.zeros((n+1))

# Defining system parameter values
lamda = 100
args = (X_0, Y, m, n+1, lamda)  # arguments values

# Batch gradient descent algorithm
theta_, j_batch_gd = batch_gradient_descent(X_0, Y, Theta_0, alpha=0.2, lamda=lamda, num_rows = m, num_features = n+1, num_iters=500)
Theta_0 = np.zeros((n+1))

n_iters = 501

# Modified Powell's optimization algorithm
j_powell = np.zeros((n_iters))
j_powell[0] = CostFunc(Theta_0, *args)

for i in range(1, n_iters):
    res = optimize.fmin_powell(CostFunc, Theta_0, args=args, maxiter = i, disp = 0)
    j_powell[i] = CostFunc(res, *args)

# BFGS optimization algorithm
j_bfgs = np.zeros((n_iters))
j_bfgs[0] = CostFunc(Theta_0, *args)

for i in range(1, n_iters):
    res = optimize.fmin_bfgs(CostFunc, Theta_0, fprime=GradFunc, args=args, maxiter = i, disp = 0)
    j_bfgs[i] = CostFunc(res, *args)
    
# Newton Conjugate gradient optimization algorithm
j_ncg = np.zeros((n_iters))
j_ncg[0] = CostFunc(Theta_0, *args)

for i in range(1, n_iters):
    res = optimize.fmin_ncg(CostFunc, Theta_0, fprime=GradFunc, args=args, maxiter = i, disp = 0)
    j_ncg[i] = CostFunc(res, *args)

# Conjugate gradient optimization algorithm
j_cg = np.zeros((n_iters))
j_cg[0] = CostFunc(Theta_0, *args)

for i in range(1, n_iters):
    res = optimize.fmin_cg(CostFunc, Theta_0, fprime=GradFunc, args=args, maxiter = i, disp = 0)
    j_cg[i] = CostFunc(res, *args)

# Plot the cost function for each model
X_ax = np.arange(0, 501, 1)
plt.plot(X_ax, j_cg, X_ax, j_ncg, X_ax, j_bfgs, X_ax, j_batch_gd, X_ax, j_powell)
plt.title('Optimization Algorithms')
plt.xlabel('Iterations')
plt.ylabel('Cost function')
plt.legend()
plt.show()

# Check model accuracy
J = CostFunc(res, *args)
Y_pred_train = X_0.dot(res.reshape(n+1,1))
Y_pred_test = X_test_0.dot(res.reshape(n_test+1,1))

RMSE_cg_train = sqrt(mean_squared_error(Y, Y_pred_train))
RMSE_cg_test = sqrt(mean_squared_error(Y_test, Y_pred_test))

########################### Scikit Ridge regression model ############################
# Create linear regression object
regr = linear_model.Ridge(normalize=True, fit_intercept=False, solver = 'sparse_cg')

# Train the model using the training sets
regr.fit(X, Y)

# Make predictions using the testing set
y_pred = regr.predict(X_test)


RMSE_lr = sqrt(mean_squared_error(Y_test, y_pred))
r2_score_lr = r2_score(Y_test, y_pred)

clf = Ridge()
coefs = []
errors = []

alphas = np.logspace(1, 10, 200)

# Train the model with different regularisation strengths
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, Y)
    coefs.append(clf.coef_.flatten())

# Display results
ax = plt.gca()
plt.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('lambda')
plt.ylabel('coefs')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')