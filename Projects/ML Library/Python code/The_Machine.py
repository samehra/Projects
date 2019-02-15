# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:37:23 2018

@author: liyifu
"""
from modelSelection import *
from modelTuning import *
from modelInference import *
from inputData import *
from modelReport import *
#import time
#import pymysql
import numpy as np
class The_Machine(object):
    def __init__(self):
        print("Choose Options by Entering Corresponding Number")
        self.mode = str(input('Choose mode: (1)normal mode (2)easy mode:'))
        self.model_name = str(input('Choose model: (1)lasso (2)elastic net (3)Least Angle Regression (4)Kriging (5)Gaussian Process (6)Logistic Regression (7)MTL-Lasso (8)Conv Nets (9)Time Series (10)Model Trees (11)Tensor Regression (12)In-situ Ensemble (13)SVM : '))
        if self.model_name not in ['11']:    
            if self.model_name == '9':
                self.partition_method = '3'
            elif self.model_name == '12':
                self.partition_method = '4'
            elif self.mode == '1':
                self.partition_method = str(input('Choose partition method: (1)0.9 partition (2)0.01 partition:'))
            else:
                self.partition_method = '1'
        if (self.model_name not in ['4','5','8','9','10','11']):
            if self.mode == '1':
                self.tuning_method = str(input('Choose tuning method: (1)5-fold cv :'))
            else:
                self.tuning_method = '1'
                
    def model_selection(self):
        if self.model_name == '1':
            model_lasso(self)
        elif self.model_name == '2':
            model_elasticNet(self)
        elif self.model_name == '3':
            model_LAR(self)
        elif self.model_name == '4':
            model_Kriging(self)
        elif self.model_name == '5':
            model_GaussianProcess(self)
        elif self.model_name == '6':
            model_Logistic(self)
        elif self.model_name == '7':
            model_MTLLasso(self)
        elif self.model_name == '8':
            model_ConvNets(self)
        elif self.model_name == '9':
            model_TSeries(self)
        elif self.model_name == '10':
            model_ModelTrees(self)
        elif self.model_name == '11':
            model_TensorReg(self)
        elif self.model_name == '12':
            model_InsituEnsemble(self)
        elif self.model_name == '13':
            model_SVM(self)
        #add new models here

        else:
            print("input error1")

    def input_data(self):
        if self.model_name in ['8','11','12']:
            loadDataset(self)
        else:
            if model.runMode == '1':
                i = str(input('Choose input method: (1)local file (2)database:'))
            else:
                i = '2'
            if i == '1':
                inputData(self)
            elif i == '2':
                self.dbname = str(input('Input database name:'))
                self.Xtablename = str(input('Input X table name:'))
                self.Ytablename = str(input('Input Y table name:'))
                self.Xa = int(input('Input idex of first column of X(start from 1):')) - 1
                self.Xb = int(input('Input idex of last column of X:')) - 1
                if self.model_name == '7':
                    self.Ya = int(input('Input idex of the first column of Y:')) - 1
                    self.Yb = int(input('Input idex of the last column of Y:')) - 1
                else:
                    self.Ya = int(input('Input idex of the column of Y:')) - 1
                loadDB(self)   #uncomment this line when database is needed
            else:
                print("input error")

    def data_partition(self):
        if self.partition_method == '1':
            partition_09(self)
        elif self.partition_method == '2':
            partition_001(self)
        elif self.partition_method == '3':
            partition_TSeries(self)
        elif self.partition_method == '4':
            partition_09_insitu(self)
        else:
            print("input error2")

    def parameter_tuning(self):
        print('Training')
        if self.model_name == '1':
            tuning_lasso(self)
        elif self.model_name == '2':
            tuning_elasticNet(self)
        elif self.model_name == '3':
            tuning_LAR(self)
        elif self.model_name == '4':
            tuning_Kriging(self)
        elif self.model_name == '5':
            tuning_GaussianProcess(self)
        elif self.model_name == '6':
            tuning_Logistic(self)
        elif self.model_name == '7':
            tuning_MTLLasso(self)
        elif self.model_name == '8':
            tuning_ConvNets(self)
        elif self.model_name == '9':
            tuning_TSeries(self)
        elif self.model_name == '10':
            tuning_ModelTrees(self)
        elif self.model_name == '11':
            tuning_TensorReg(self)
        elif self.model_name == '12':
            tuning_InsituEnsemble(self)
        elif self.model_name == '13':
            tuning_SVM(self)
        else:
            print("input error3")

#add new tuning methods here
            
    def model_inference(self):
        print('Inferencing')
        if self.model_name in ['7','8','11','12']:
            model_inference_category(self)
        else:
            model_inference_general(self)

    def report_generation(self):
        print('Generating report')
        if self.model_name in ['6','13']:
            report_generation_categorical(self)
        else:
            report_generation_continuous(self)



model = The_Machine()
model.model_selection()
model.runMode = str(input('Choose input method: (1)one-time run (2)continuous run(DB only):'))
if model.runMode == '1':
    #this is for one-time run mode
    model.input_data()
    if model.model_name not in ['8', '11']:
        model.data_partition()
    model.parameter_tuning()
    model.model_inference()
    if model.model_name not in ['8', '11', '12']:
        model.report_generation()
#elif model.runMode == '2':
#    #this is for continuous run mode
#    model.input_data()
#    lastTime1 = 0
#    lastTime2 = 0
#    newTime = time.time()
#    while 1:
#        newTime = time.time()
#        if newTime - lastTime1 > 10:
#            lastTime1 = newTime
#            loadDB(model)
#            model.data_partition()
#            model.parameter_tuning()
#            print('trained at time:', round(lastTime1, 2))
#        else:
#            if newTime - lastTime2 < 1:
#                time.sleep(1 - newTime + lastTime2)
#            lastTime2 = time.time()
#            loadDB(model)
#            model.X = (model.X - model.meanX) / model.stdX
#            y_predicted = model.fitted_final.predict(model.X)
#            print("beta",model.fitted_final.coef_)
#            print(model.y)
#            print(y_predicted)
#            conn = pymysql.connect(host="192.168.1.133", user="rtejada", passwd="ProjectP112358", db="dbtest")
#            myCursor = conn.cursor()
#            i=0
#            for p in y_predicted:
#                sql = f"UPDATE {model.dbname}.{model.Ytablename} SET Prediction = {p} WHERE TIME = {model.dbTime[i]}"
#                myCursor.execute(sql)
#                i+=1
#            conn.commit()
#            conn.close()
#            print('predicted at time:', round(lastTime2, 2))
