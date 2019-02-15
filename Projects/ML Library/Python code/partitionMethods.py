import numpy as np
from sklearn.model_selection import train_test_split

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()

def dataNormalization(option):
    print(option.X_train)
    meanX = np.mean(option.X_train)
    stdX = np.std(option.X_train)
    if(option.model_name == '9'):
        option.X_train = (option.X_train - meanX)
        option.X_test = (option.X_test - meanX)
        option.X = option.X - meanX
        return 0
    option.X_train = (option.X_train - meanX)/stdX
    option.X_test = (option.X_test - meanX) / stdX
    option.X = (option.X - meanX) / stdX
    option.meanX = meanX
    option.stdX = stdX
    if(option.model_name not in ['6','13']):
        meanY = np.mean(option.y_train)
        # option.y_train = option.y_train - meanY
        # option.y_test = option.y_test - meanY

def partition_001(option):
    option.report_partition = '0.01 partition'
    if len(option.y) < 10:
        raise ValueError("sample size is too small for such parition (n<10)")
    if option.model_name == '8':
        option.X_train_train, option.X_train_valid, option.y_train_train, option.y_train_valid = train_test_split(option.X_train, option.y_train, test_size = 0.99, random_state = 1)
    else:
        option.X_train, option.X_test, option.y_train, option.y_test = train_test_split(option.X, option.y, test_size = 0.99, random_state = 1)

    if option.model_name == '5':
        time_index_x = option.X_train.shape[1]-1
        time_index_y = option.y_train.shape[1]-1
        option.X_train = option.X_train[option.X_train[:,time_index_x].argsort()[::1]]
        option.X_test = option.X_test[option.X_test[:,time_index_x].argsort()[::1]]
        option.y_train = option.y_train[option.y_train[:,time_index_y].argsort()[::1]]
        option.y_train = option.y_train[option.y_train[:,time_index_y].argsort()[::1]]
        
    if option.model_name == '10':
        robjects.globalenv['trainx'] = option.X_train
        robjects.globalenv['trainy'] = option.y_train
        robjects.globalenv['testx'] = option.X_test
        robjects.globalenv['testy'] = option.y_test
        trainx = robjects.r('''trainx = as.data.frame(trainx)''')
        testx = robjects.r('''testx = as.data.frame(testx)''')
    
def partition_09(option):
    option.report_partition = '0.9 partition'
    if len(option.y) < 10:
        raise ValueError("sample size is too small for such parition (n<10)")
    if option.model_name == '8':
        option.X_train_train, option.X_train_valid, option.y_train_train, option.y_train_valid = train_test_split(option.X_train, option.y_train, test_size = 0.1, random_state = 1)
    else:
        option.X_train, option.X_test, option.y_train, option.y_test = train_test_split(option.X, option.y, test_size = 0.1, random_state = 1)
    
    if option.model_name == '5':
        time_index_x = option.X_train.shape[1]-1
        time_index_y = option.y_train.shape[1]-1
        option.X_train = option.X_train[option.X_train[:,time_index_x].argsort()[::1]]
        option.X_test = option.X_test[option.X_test[:,time_index_x].argsort()[::1]]
        option.y_train = option.y_train[option.y_train[:,time_index_y].argsort()[::1]]
        option.y_train = option.y_train[option.y_train[:,time_index_y].argsort()[::1]]
    
    if option.model_name == '10':
        robjects.globalenv['trainx'] = option.X_train
        robjects.globalenv['trainy'] = option.y_train
        robjects.globalenv['testx'] = option.X_test
        robjects.globalenv['testy'] = option.y_test
        trainx = robjects.r('''trainx = as.data.frame(trainx)''')
        testx = robjects.r('''testx = as.data.frame(testx)''')
    
def partition_TSeries(option):
    option.report_partition = 'time series partition'
    option.X_train, option.X_test = option.X[1:round(len(option.X) - option.futureNum)], option.X[round(len(option.X) - option.futureNum):]


def partition_09_insitu(option):
    option.report_partition = '0.9 partition'
    if len(option.Z) < 10:
        raise ValueError("sample size is too small for such parition (n<10)")

    option.X_train, option.X_test, option.Y_train, option.Y_test,option.Z_train, option.Z_test = train_test_split(option.X, option.Y,option.Z,test_size=0.1, random_state=1)


'''
add a new function for each new partition method
'''