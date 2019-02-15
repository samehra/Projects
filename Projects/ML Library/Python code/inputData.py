from pandas import Series
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import h5py
#import pymysql
def inputData(option):
    # filename = 'data.csv'
    filename = str(input('Input data file name:'))
    if option.model_name == '9':
        series = Series.from_csv(filename, header=0)
        f, axarr = plt.subplots(2, sharex=True)
        plot_acf(series, lags=50, ax=axarr[0])
        plot_pacf(series, lags=50, ax=axarr[1])
        plt.show()
        option.X = series.values
        option.y = option.X
        return 0
    Xa = int(input('Input idex of first column of X(start from 1):') ) -1
    Xb = int(input('Input idex of last column of X:') ) -1
    if option.model_name == '7':
        Ya = int(input('Input idex of the first column of Y:') ) -1
        Yb = int(input('Input idex of the last column of Y:') ) -1
    else:
        Ya = int(input('Input idex of the column of Y:') ) -1
    print('Loading')
    raw_data = open(filename, 'rt')
    data = np.loadtxt(raw_data, delimiter=',', dtype=float, skiprows=1)
    X = data[:, Xa:( Xb +1)]
    if option.model_name == '5':
        Y = data[:, Ya]
        time_index = np.arange(1 ,len(X ) +1 ,1).reshape(-1 ,1)
        option.X = np.hstack((X, time_index))
        option.y = np.hstack((Y.reshape(-1 ,1), time_index))
    elif option.model_name == '7':
        Y = data[:, Ya:( Yb +1)]
        option.X = X
        option.y = Y
    else:
        Y = data[:, Ya]
        option.X = X
        option.y = Y

def loadDataset(option):
    if option.model_name == '8':
        train_filename = str(input('Input data file name for training data:'))
        test_filename = str(input('Input data file name for test data:'))
        print('Loading')
        
        train_dataset = h5py.File(train_filename, "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    
        test_dataset = h5py.File(test_filename, "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    
        option.X = train_set_x_orig
        option.y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0])).T
        option.classes = np.array(test_dataset["list_classes"][:])
    
        option.y_train = train_set_y_orig.reshape((1, train_set_y_orig.shape[0])).T
        option.y_test = test_set_y_orig.reshape((1, test_set_y_orig.shape[0])).T
        option.X_train = train_set_x_orig/ 255.
        option.X_test = test_set_x_orig / 255.
        option.input_shape = option.X_train.shape[1:]
    elif option.model_name == '11':
        filename = str(input('Input data file name:'))
        Xa = int(input('Input idex of first column of X(start from 1):') ) -1
        Xb = int(input('Input idex of last column of X:') ) -1
        print('Loading')
        raw_data = open(filename, 'rt')
        data = np.loadtxt(raw_data, delimiter=',', dtype=float, skiprows=0)
        X = data[:, Xa:( Xb +1)]
        option.X = X
    elif option.model_name == '12':
        raw_data = open('./datasets/X.csv', 'rt')
        option.X = np.loadtxt(raw_data, delimiter=',', dtype=float)
        raw_data = open('./datasets/Y.csv', 'rt')
        option.Y = np.loadtxt(raw_data, delimiter=',', dtype=float)
        raw_data = open('./datasets/Z.csv', 'rt')
        option.Z = np.array([np.loadtxt(raw_data, delimiter=',', dtype=float)]).T

def loadDB(option):

    conn = pymysql.connect(host="192.168.1.133", user="rtejada", passwd="ProjectP112358", db=option.dbname)
    myCursor = conn.cursor()
    sql = f"SELECT * FROM {option.Xtablename} ORDER BY TIME DESC LIMIT 100"
    myCursor.execute(sql)
    results = myCursor.fetchall()
    Xdata = np.asarray(results)
    sql = f"SELECT * FROM {option.Ytablename} ORDER BY TIME DESC LIMIT 100"
    myCursor.execute(sql)
    results = myCursor.fetchall()
    Ydata = np.asarray(results)
    conn.close()

    X = Xdata[:, option.Xa:(option.Xb + 1)]
    option.dbTime = Ydata[:,0]
    if option.model_name == '5':
        Y = Ydata[:, option.Ya]
        time_index = np.arange(1, len(X) + 1, 1).reshape(-1, 1)
        option.X = np.hstack((X, time_index))
        option.y = np.hstack((Y.reshape(-1, 1), time_index))
    elif option.model_name == '7':
        Y = Ydata[:, option.Ya:(option.Yb + 1)]
        option.X = X
        option.y = Y
    else:
        Y = Ydata[:, option.Ya]
        option.X = X
        option.y = Y


