
from sklearn.metrics import mean_squared_error
import sklearn.linear_model as sl
from math import sqrt
from sklearn.model_selection import KFold

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential

import keras.backend as K
from InsituEnsemble import *

K.set_image_data_format('channels_last')
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
mlbench = importr('mlbench')
lattice = importr("lattice", lib_loc = "C:/Program Files/Microsoft/R Open/R-3.4.3/library")
Cubist = importr("Cubist")

from tensorly.regression.kruskal_regression import KruskalRegressor
from tensorly.regression.tucker_regression import TuckerRegressor

## Models should be trained on mormalized values    
def model_fit(model_options, X, y,*args, **kwargs):
    if model_options.model_name == '1':
        clf = sl.Lasso(alpha=model_options.lambda_value)
        clf.fit(X, y)
        return clf
    elif model_options.model_name == '2':
        clf = sl.ElasticNet(alpha=model_options.lambda_value, l1_ratio=model_options.ratio_value)
        clf.fit(X, y)
        return clf
    elif model_options.model_name == '3':
        clf = sl.Lars(copy_X=True, eps=model_options.lambda_value, fit_intercept=True, fit_path=True,
                      normalize=True, positive=False, precompute='auto', verbose=False)
        clf.fit(X, y)
        return clf
    elif model_options.model_name == '4':
        from sklearn.gaussian_process import GaussianProcessRegressor
        clf = GaussianProcessRegressor(kernel = model_options.kernel, n_restarts_optimizer = 3, random_state = 2018)
        clf.fit(X, y)
        return clf
    elif model_options.model_name == '5':
        from sklearn.gaussian_process import GaussianProcessRegressor
        clf = GaussianProcessRegressor(kernel = model_options.kernel, normalize_y = 'T', n_restarts_optimizer = 3, random_state = 2018)
        clf.fit(X, y[:,0])
        return clf
    elif model_options.model_name == '6':
        clf = sl.LogisticRegression(penalty=model_options.normSelection, C=1/model_options.lambda_value)
        clf.fit(X, y)
        return clf
    elif model_options.model_name == '7':
        clf = sl.MultiTaskLasso(alpha=model_options.lambda_value)
        clf.fit(X, y)
        return clf
    elif model_options.model_name == '8':
        X_input = Input(model_options.input_shape)
        
        X_ = ZeroPadding2D((3, 3))(X_input)
        X_ = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X_)
        X_ = BatchNormalization(axis = 3, name = 'bn0')(X_)
        X_ = Activation('relu')(X_)
        X_ = MaxPooling2D((2, 2), name='max_pool')(X_)
        X_ = Flatten()(X_)
        X_ = Dense(1, activation='sigmoid', name='fc')(X_)
        
        clf = Model(inputs = X_input, outputs = X_)
        
        clf.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        clf.fit(X, y, epochs=20, batch_size=50, verbose=1, validation_data=(model_options.X_train_valid, model_options.y_train_valid))
        return clf
    elif model_options.model_name == '9':
        from statsmodels.tsa.ar_model import AR
        clf = AR(X).fit(maxlag=int(model_options.lambda_value))
        return clf
    elif model_options.model_name == '10':
        clf = robjects.r('''mod1 = cubist(x = trainx, y = trainy, committees = 10)''')
        return clf
    elif model_options.model_name == '11':
        # Create a tensor Regressor estimator
        if model_options.tensorReg_type == '1':
            clf = KruskalRegressor(weight_rank=model_options.rank+1, tol=10e-7, n_iter_max=100, reg_W=1, verbose=0)
        elif model_options.tensorReg_type == '2':
             clf = TuckerRegressor(weight_ranks=[model_options.rank+1, model_options.rank+1], tol=10e-7, n_iter_max=100, reg_W=1, verbose=0)
        # Fit the estimator to the data
        clf.fit(X, y)
        return clf
    elif model_options.model_name == '12':
        Z = kwargs.get('Z', None)
        clf = InsituEnsemble(model_options.lambda_value,X,y,Z)
        return clf
    elif model_options.model_name == '13':
        from sklearn import svm
        clf = svm.SVC(C=model_options.lambda_value)
        clf.fit(X, y)
        return clf
    
    #add new elif for each new model

def tuningMethods(option):
    if(option.tuning_method == '1'):
        tuning_5fold(option)

    

def tuning_5fold(option):
    if option.model_name == '12':
        option.report_tuning = '5-fold cv'
        kf = KFold(n_splits=5, random_state=3333, shuffle=True)
        scoreList = []
        for train_index, test_index in kf.split(option.X_train):

            X_train_w_fold = [option.X_train[j] for j in train_index]
            X_test_w_fold = [option.X_train[j] for j in test_index]
            Y_train_w_fold = [option.Y_train[j] for j in train_index]
            Y_test_w_fold = [option.Y_train[j] for j in test_index]
            Z_train_w_fold = [option.Z_train[j] for j in train_index]
            Z_test_w_fold = [option.Z_train[j] for j in test_index]

            fitted = model_fit(option, X_train_w_fold, Y_train_w_fold, Z=Z_train_w_fold)
            Z_predicted = fitted.predict(X_test_w_fold)['z_predicted']
            score = sqrt(mean_squared_error(np.array(Z_test_w_fold).T[0], Z_predicted))
            scoreList.append(score)
        option.corresponding_rmse.append(scoreList)
        score_mean = sum(scoreList) / 5
        option.corresponding_rmse_mean.append(score_mean)
    else:
        option.report_tuning = '5-fold cv'
        kf = KFold(n_splits=5, random_state=3333, shuffle=True)
        scoreList = []
        for train_index, test_index in kf.split(option.X_train):

            X_train_w_fold = [option.X_train[j] for j in train_index]
            X_test_w_fold = [option.X_train[j] for j in test_index]
            y_train_w_fold = [option.y_train[j] for j in train_index]
            y_test_w_fold = [option.y_train[j] for j in test_index]

            fitted = model_fit(option, X_train_w_fold, y_train_w_fold)
            if option.model_name not in  ['6','13']:
                y_predicted = fitted.predict(X_test_w_fold)
                score = sqrt(mean_squared_error(y_test_w_fold, y_predicted))
            else:
                score = fitted.score(X_test_w_fold, y_test_w_fold)
            scoreList.append(score)
        option.corresponding_rmse.append(scoreList)
        score_mean = sum(scoreList) /5
        option.corresponding_rmse_mean.append(score_mean)

'''
add a new function for each new tuning method
'''
