
from tensorly.base import tensor_to_vec, partial_tensor_to_vec
from tensorly.random import check_random_state
from tuningMethods import *
from partitionMethods import *
def tuning_lasso(option):
    dataNormalization(option)
    option.candidate_lambda = np.linspace(0, option.max_lambda, option.lambda_range_space)
    option.corresponding_rmse_mean = []

    for lambda_curr in option.candidate_lambda:
        option.corresponding_rmse = []
        option.lambda_value = lambda_curr
        tuningMethods(option)

    option.corresponding_rmse_mean = np.asarray(option.corresponding_rmse_mean)
    option.opt_lambda = option.candidate_lambda[np.where(option.corresponding_rmse_mean == min(option.corresponding_rmse_mean))][0]
    option.lambda_value = option.opt_lambda
    option.fitted_final = model_fit(option, option.X_train, option.y_train)

def tuning_elasticNet(option):
    dataNormalization(option)
    option.candidate_lambda = np.linspace(0, option.max_lambda, option.lambda_range_space)
    option.candidate_ratio = np.linspace(0.001, 1, option.ratio_range_space)
    option.corresponding_rmse_mean = []
    option.plot3dX = []
    option.plot3dY = []
    for lambda_curr in option.candidate_lambda:
        for ratio_curr in option.candidate_ratio:
            option.corresponding_rmse = []
            option.ratio_value = ratio_curr
            option.lambda_value = lambda_curr
            tuningMethods(option)
            option.plot3dX.append(lambda_curr*ratio_curr)
            option.plot3dY.append(lambda_curr*(1-ratio_curr))
    option.corresponding_rmse_mean = np.asarray(option.corresponding_rmse_mean)

    opt_index = np.where(option.corresponding_rmse_mean == min(option.corresponding_rmse_mean))
    option.opt_lambda = option.candidate_lambda[int(opt_index[0][0]/option.ratio_range_space)]
    option.opt_ratio = option.candidate_ratio[int(opt_index[0][0]%option.ratio_range_space)]
    option.lambda_value = option.opt_lambda
    option.ratio_value = option.opt_ratio
    option.fitted_final = model_fit(option, option.X_train, option.y_train)

def tuning_LAR(option):
    option.candidate_lambda = np.linspace(0, option.max_lambda, option.lambda_range_space)
    option.corresponding_rmse_mean = []

    for lambda_curr in option.candidate_lambda:
        option.corresponding_rmse = []
        option.lambda_value = lambda_curr
        tuningMethods(option)

    option.corresponding_rmse_mean = np.asarray(option.corresponding_rmse_mean)
    option.opt_lambda = option.candidate_lambda[np.where(option.corresponding_rmse_mean == min(option.corresponding_rmse_mean))][0]
    option.lambda_value = option.opt_lambda
    option.fitted_final = model_fit(option, option.X_train, option.y_train)

def tuning_Kriging(option):
    dataNormalization(option)
    option.fitted_final = model_fit(option, option.X_train, option.y_train)
    
def tuning_GaussianProcess(option):
#    option.corresponding_rmse = []
#    option.corresponding_rmse_mean = []
    option.fitted_final = model_fit(option, option.X_train, option.y_train)
    
#    option.y_predicted, option.sigma = option.fitted_final.predict(option.X_test, return_std=True)
#    option.optKernel = option.fitted_final.kernel_
#    option.LogLikelihoodValue = option.fitted_final.log_marginal_likelihood_value_
#    rmsErr = sqrt(mean_squared_error(option.y_test, y_predicted))
#    option.corresponding_rmse = rmsErr
#    option.corresponding_rmse_mean = rmsErr

def tuning_Logistic(option):
    dataNormalization(option)
    option.candidate_lambda = np.linspace(0, option.max_lambda, option.lambda_range_space)
    option.candidate_lambda = np.delete(option.candidate_lambda, 0)
    option.corresponding_rmse_mean = []

    for lambda_curr in option.candidate_lambda:
        option.corresponding_rmse = []
        option.lambda_value = lambda_curr
        tuningMethods(option)

    option.corresponding_rmse_mean = np.asarray(option.corresponding_rmse_mean)
    option.opt_lambda = \
    option.candidate_lambda[np.where(option.corresponding_rmse_mean == max(option.corresponding_rmse_mean))][0]
    option.lambda_value = option.opt_lambda
    option.fitted_final = model_fit(option, option.X_train, option.y_train)

def tuning_MTLLasso(option):
    dataNormalization(option)
    option.candidate_lambda = np.linspace(0, option.max_lambda, option.lambda_range_space)
    option.corresponding_rmse_mean = []

    for lambda_curr in option.candidate_lambda:
        option.corresponding_rmse = []
        option.lambda_value = lambda_curr
        tuningMethods(option)

    option.corresponding_rmse_mean = np.asarray(option.corresponding_rmse_mean)
    option.opt_lambda = option.candidate_lambda[np.where(option.corresponding_rmse_mean == min(option.corresponding_rmse_mean))][0]
    option.lambda_value = option.opt_lambda
    option.fitted_final = model_fit(option, option.X_train, option.y_train)
    
def tuning_ConvNets(option):
    option.fitted_final = model_fit(option, option.X_train_train, option.y_train_train)


def tuning_TSeries(option):
    option.report_tuning = "Time Series Tuning"
    dataNormalization(option)
    option.candidate_lambda = np.arange(1, option.max_lambda+1)
    option.corresponding_rmse_mean = []
    X_test_w_fold, y_test_w_fold = option.X_train[1:round(len(option.X_train) - option.futureNum)], option.X_train[round(len(option.X_train) - option.futureNum):]
    for lambda_curr in option.candidate_lambda:
        scoreList = []
        option.corresponding_rmse = []
        option.lambda_value = lambda_curr
        fitted = model_fit(model_options = option, X = X_test_w_fold, y=None)
        y_predicted = fitted.predict(start=len(X_test_w_fold), end=len(X_test_w_fold) + len(y_test_w_fold) - 1, dynamic=False)
        score = sqrt(mean_squared_error(y_test_w_fold, y_predicted))
        scoreList.append(score)
        option.corresponding_rmse_mean.append(scoreList)

    option.corresponding_rmse_mean = np.asarray(option.corresponding_rmse_mean)
    option.opt_lambda = option.candidate_lambda[np.where(option.corresponding_rmse_mean == min(option.corresponding_rmse_mean))[0]]
    option.lambda_value = option.opt_lambda

    option.fitted_final = model_fit(model_options = option, X = option.X_train, y=None)

def tuning_ModelTrees(option):
    option.fitted_final = model_fit(option, option.X_train, option.y_train)
    
def tuning_TensorReg(option):
    image_height = 25
    image_width = 25
    rng = check_random_state(1)
    option.fitted_final_weights = [0]*option.max_rank
    option.X_train = T.tensor(rng.normal(size=(1000, image_height, image_width), loc=0, scale=1))
    option.weight_img = T.tensor(option.X)
    option.y_train = T.dot(partial_tensor_to_vec(option.X_train, skip_begin=1), tensor_to_vec(option.weight_img))
    for option.rank in range(option.max_rank):
        option.fitted_final = model_fit(option, option.X_train, option.y_train)
        option.fitted_final_weights[option.rank] = option.fitted_final.weight_tensor_

def tuning_InsituEnsemble(option):
    # dataNormalization(option)
    option.candidate_lambda = np.linspace(0, option.max_lambda, option.lambda_range_space)
    option.corresponding_rmse_mean = []
    for lambda_curr in option.candidate_lambda:
        option.corresponding_rmse = []
        option.lambda_value = lambda_curr
        tuningMethods(option)

    option.corresponding_rmse_mean = np.asarray(option.corresponding_rmse_mean)
    option.opt_lambda = \
    option.candidate_lambda[np.where(option.corresponding_rmse_mean == min(option.corresponding_rmse_mean))][0]
    print('opt'+str(option.opt_lambda))
    option.lambda_value = option.opt_lambda
    option.fitted_final = model_fit(option, option.X_train, option.Y_train, Z=option.Z_train)

def tuning_SVM(option):
    dataNormalization(option)
    option.candidate_lambda = np.linspace(0.001, option.max_lambda, option.lambda_range_space)
    option.corresponding_rmse_mean = []

    for lambda_curr in option.candidate_lambda:
        option.corresponding_rmse = []
        option.lambda_value = lambda_curr
        tuningMethods(option)

    option.corresponding_rmse_mean = np.asarray(option.corresponding_rmse_mean)
    option.opt_lambda = option.candidate_lambda[np.where(option.corresponding_rmse_mean == max(option.corresponding_rmse_mean))][0]
    option.lambda_value = option.opt_lambda
    option.fitted_final = model_fit(option, option.X_train, option.y_train)
'''

add new method for each model
'''