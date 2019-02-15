# import matlab.engine
import numpy as np
from sklearn.linear_model import Ridge
from scipy.linalg import block_diag
import math
# raw_data = open('X.csv', 'rt')
# X = np.loadtxt(raw_data, delimiter=',', dtype=float)
# raw_data = open('Y.csv', 'rt')
# Y = np.loadtxt(raw_data, delimiter=',', dtype=float)
# raw_data = open('Z.csv', 'rt')
# Z = np.array([np.loadtxt(raw_data, delimiter=',', dtype=float)]).T
# q = float(Y.shape[1])
# eng = matlab.engine.start_matlab()
# f, eta, yy, sigma, sigma_z_square, s, L, B_zero, theta_zero, residue_eta_yy=eng.ADMM_AJP_Joint_Fun(matlab.double(X.tolist()),matlab.double(Y.tolist()),matlab.double(Z.tolist()),0.0,10e-2,3,10e-5,nargout=10)
# y_predicted, z_predicted = eng.predict(matlab.double(X.tolist()),B_zero, theta_zero,eta, yy, q, nargout=2)

def ADMM_AJP_Joint_Fun(X,Y,Z,M,rho,MAXITER,TOL):
    n, p = X.shape
    q = Y.shape[1]
    B_zero = []
    for i in range(q):
        beta_i = Ridge(alpha=10e-10).fit(X, Y[:,i]).coef_.tolist()
        B_zero_i = [0]*p*i+beta_i+[0]*p*(q-i-1)
        B_zero.append(B_zero_i)
    B_zero=np.array(B_zero)
    theta_zero = Ridge(alpha=10e-10).fit(Y,Z).coef_
    theta_zero = np.array(theta_zero)
    ita_y = Y
    z = Z
    mu1 = np.zeros((q,1))
    mu2 = 0

    ita_X = []
    for k in range(n):
        A = X[k].tolist()
        for i in range(q):
            ita_X_kq = [0]*p*i+A+[0]*p*(q-i-1)
            ita_X.append(ita_X_kq)
    ita_X=np.array(ita_X)
    B_zero_stack = np.tile(B_zero, (n, 1))
    twist_N = np.multiply(ita_X, B_zero_stack)
    ita_I = []
    for i in range(q):
        ita_I_i = [0]*p*i+[1,1,1,1,1]+[0]*p*(q-i-1)
        ita_I.append(ita_I_i)
    ita_I = np.array(ita_I).T

    one_pq_pq=np.ones((p*q,p*q))
    one_pq_1=np.ones((p*q,1))
    one_q_1 = np.ones((q, 1))
    one_q_q = np.ones((q,q))
    I_mat_q_q = np.eye(int(q))

    theta_zero_stack = np.tile(theta_zero, (n, 1))
    twist_g = np.multiply(ita_y, theta_zero_stack)

    y=ita_y.T
    y=np.reshape(y,(n*q,1),order='F')

    cov_Y = np.cov(Y.T)
    sigma = cov_Y
    diag_sigma_inv=sigma
    for i in range(n-1):
        diag_sigma_inv =block_diag(diag_sigma_inv,sigma)
    eta = np.zeros((p*q,1))
    yy = np.zeros((q,1))
    s = np.zeros((q,1))
    L = 0
    sigma_z_square = 1

    ita_I_ita_I_T = np.dot(ita_I, ita_I.T)
    twist_g_T_twist_g = np.dot(twist_g.T, twist_g)
    f=[]
    for iter in range(MAXITER):
        # print('testing')
        # print('sigma',np.sum(sigma))
        # print('eta',np.sum(eta))  #not correct
        # print('sigma_z_square',sigma_z_square)
        # print('yy',np.sum(yy))   #not correct
        # print('mu1',np.sum(mu1))
        # print('mu2',mu2)  #not correct
        # print('s',np.sum(s))  #not correct
        # print('L',L)
        f.append(n * math.log(np.linalg.det(sigma)) \
                 + np.dot(np.dot((y - np.dot(twist_N, eta)).T, diag_sigma_inv), (y - np.dot(twist_N, eta))) \
                 + n * math.log(sigma_z_square)\
                 + (np.dot((z - np.dot(twist_g, yy)).T / sigma_z_square, (z - np.dot(twist_g, yy))))\
                  + np.dot(mu1.T,(np.dot(ita_I.T, eta) - yy - s))\
                  + rho / 2 * (np.linalg.norm(np.dot(ita_I.T,eta)-yy-s))**2\
                  + mu2 * (np.dot(one_q_1.T,yy)+np.dot(one_pq_1.T, eta) + L - M)\
                  + rho / 2 * (np.linalg.norm(np.dot(one_q_1.T,yy)+np.dot(one_pq_1.T, eta) + L - M))** 2)

        large_inv_for_eta = np.dot(np.dot(2*twist_N.T,diag_sigma_inv),twist_N)+rho*ita_I_ita_I_T+rho*one_pq_pq


        eta = np.linalg.lstsq(large_inv_for_eta,np.array(((np.dot(np.dot(np.dot(2, twist_N.T),diag_sigma_inv),y)-np.dot(ita_I,mu1)+ np.dot(rho* ita_I, yy) + np.dot(rho* ita_I, s) - mu2* one_pq_1 - (np.dot(rho* one_q_1.T,yy).T*one_pq_1)-rho*L*one_pq_1+ rho * M* one_pq_1)).T.tolist()[0]))[0]
        # eta = np.linalg.lstsq(large_inv_for_eta,(np.dot(np.dot(np.dot(2, twist_N.T),diag_sigma_inv),y)-np.dot(ita_I,mu1)+ np.dot(np.dot(rho, ita_I), yy) + np.dot(np.dot(rho, ita_I), s) - np.dot(mu2, one_pq_1) - (np.dot(rho* one_q_1.T,yy).T*one_pq_1)-rho*L*one_pq_1+ np.dot(rho * M, one_pq_1)))[0]
        eta = np.reshape(eta.clip(0),(-1,1))

        large_inv_for_yy = 2 * (sigma_z_square**(-1)) * twist_g_T_twist_g + rho* I_mat_q_q + rho* one_q_q

        yy = np.linalg.lstsq(large_inv_for_yy, (np.dot(2 * (sigma_z_square**(-1))* twist_g.T,z)+mu1+np.reshape((np.dot(np.dot(rho*one_q_q, ita_I.T),eta)),(-1,1))-mu2*one_q_1-np.reshape(np.dot(np.dot(rho*one_q_1,one_pq_1.T), eta),(-1,1)) - rho * L * one_q_1+ rho * M* one_q_1 - np.dot(rho* I_mat_q_q, s)))[0]
        yy = yy.clip(0)
        sum_ATA = 0

        for i in range(n):
            ita_X_i = X[i]
            for k in range(q - 1):
                ita_X_i = block_diag(ita_X_i, X[i])
            ita_v_i = np.multiply(ita_X_i, B_zero)
            # sum_ATA = sum_ATA + (np.dot((ita_y[i].T-np.dot(ita_v_i,eta)),(ita_y[i].T - np.dot(ita_v_i , eta)).T))
            sum_ATA = sum_ATA + (np.dot((np.reshape(ita_y[i].T,(-1,1))-np.dot(ita_v_i,eta)),((np.reshape(ita_y[i].T,(-1,1)) - np.dot(ita_v_i,eta)).T)))
        sigma = (sum_ATA / n).T
        sigma_inv = np.linalg.inv(sigma)
        diag_sigma_inv = sigma_inv
        for i in range(n - 1):
            diag_sigma_inv = block_diag(diag_sigma_inv, sigma_inv)
        ita_W = (z - np.dot((np.multiply(ita_y, theta_zero_stack)), yy))
        sigma_z_square = (np.dot(ita_W.T,ita_W)/n).tolist()[0][0]
        s = (mu1 + np.dot(rho * ita_I.T,eta)-rho*yy)/rho
        s = s.clip(0)
        L = (-mu2 + rho * M - np.dot(rho * one_pq_1.T,eta)-np.dot(rho*one_q_1.T, yy)) / rho
        L = (L.clip(0)).tolist()[0][0]
        mu1 = mu1 + rho * (np.dot(ita_I.T,eta)-yy-s)
        mu2 = (mu2 + rho * (np.dot(one_q_1.T,yy)+np.dot(one_pq_1.T, eta) + L - M)).tolist()[0][0]
        if iter > 0:
            if abs(f[iter] - f[iter - 1]) < TOL:
                break
    residue_eta_yy = np.dot(ita_I.T,eta)-yy
    return f,eta.tolist(),yy.tolist(),sigma.tolist(),sigma_z_square,s.tolist(),L,B_zero.tolist(),theta_zero.T.tolist(),residue_eta_yy.tolist()

def predict(Train_Data_X_Test,B_zero,theta_zero,eta,yy,q):
    [m,n] = Train_Data_X_Test.shape
    q=int(q)
    ita_X = []
    for k in range(m):
        A = Train_Data_X_Test[k].tolist()
        for l in range(q):
            ita_X_kq = [0]*n*l+A+[0]*n*(q-l-1)
            ita_X.append(ita_X_kq)
    ita_X=np.array(ita_X)
    B_zero=np.array(B_zero)
    eta=np.array(eta)
    yy=np.array(yy)
    theta_zero = np.array(theta_zero)
    B_zero_stack = np.tile(B_zero,(m,1))
    theta_zero_stack = np.tile(np.transpose(theta_zero),(m,1))
    Y_hat = np.reshape(np.dot(np.multiply(ita_X, B_zero_stack),eta),(m,q))
    Z_hat = np.dot(np.multiply(Y_hat,theta_zero_stack), yy)
    return Y_hat,Z_hat

class InsituEnsemble(object):
    def __init__(self,M,X,Y,Z):
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        self.q = float(Y.shape[1])
        self.f, self.eta, self.yy, self.sigma, self.sigma_z_square, self.s, self.L, self.B_zero, self.theta_zero, self.residue_eta_yy=ADMM_AJP_Joint_Fun(X,Y,Z,M.item(),10e-2,3,10e-5)
        # self.f, self.eta, self.yy, self.sigma, self.sigma_z_square, self.s, self.L, self.B_zero, self.theta_zero, self.residue_eta_yy = self.eng.ADMM_AJP_Joint_Fun(
        #     matlab.double(X.tolist()), matlab.double(Y.tolist()), matlab.double(Z.tolist()), M.item(), 10e-2, 3, 10e-5,
        #     nargout=10)

    def predict(self,X_Test):
        X_Test = np.array(X_Test)
        y_predicted, z_predicted = predict(X_Test,self.B_zero, self.theta_zero,self.eta, self.yy, self.q)

        z_predicted=z_predicted.T[0]
        return {'y_predicted': y_predicted, 'z_predicted': z_predicted}