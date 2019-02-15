from scipy import stats
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import rpy2.robjects as robjects
import tensorly.backend as T

def model_inference_general(option):
    option.folderName = str(input('Please type in the folder name:'))
    if not os.path.exists('./' + option.folderName):
        os.makedirs('./' + option.folderName)
    if option.model_name == '5':
        y_train_predicted, sigma_train = option.fitted_final.predict(option.X_train, return_std=True)
        y_test_predicted, sigma_test = option.fitted_final.predict(option.X_test, return_std=True)
        y_pred_plot, sigma_plot = option.fitted_final.predict(option.X, return_std=True)
        option.kernel = option.fitted_final.kernel_
        option.log_likelihood = option.fitted_final.log_marginal_likelihood_value_
        print(option.kernel)
        print(option.log_likelihood)

        option.train_residual = y_train_predicted - option.y_train[:, 0]
        option.test_residual = y_test_predicted - option.y_test[:, 0]
        option.rmse_train_final = sqrt(mean_squared_error(option.y_train[:, 0], y_train_predicted))
        option.rmse_test_final = sqrt(mean_squared_error(option.y_test[:, 0], y_test_predicted))

        res = sum(np.square(option.y_train[:, 0] - y_train_predicted))
        tot = sum(np.square(option.y_train[:, 0] - np.mean(option.y_train[:, 0])))
    else:
        if option.model_name == '9':
            y_train_predicted = option.fitted_final.predict(start=option.fitted_final.k_ar, end=len(option.X_train) - 1,
                                                          dynamic=False)
            y_test_predicted = option.fitted_final.predict(start=len(option.X_train),
                                                         end=len(option.X_train) + len(option.X_test) - 1, dynamic=False)
            option.y_train = option.X_train[option.fitted_final.k_ar:len(option.X_train)]
            option.y_test = option.X_test
        elif option.model_name == '10':
            y_train_predicted = np.array(robjects.r('''predict(mod1, newdata = trainx)'''))
            y_test_predicted = np.array(robjects.r('''predict(mod1, newdata = testx)'''))
        else:
            y_train_predicted = option.fitted_final.predict(option.X_train)
            y_test_predicted = option.fitted_final.predict(option.X_test)
            print('y_test',option.y_test)
            print('y_test_predicted',y_test_predicted)

        if option.model_name == '6':
            option.trainscore = option.fitted_final.score(option.X_train, option.y_train)
            option.testscore = option.fitted_final.score(option.X_test, option.y_test)
            tn, fp, fn, tp = confusion_matrix(option.y_train, y_train_predicted).ravel()
            option.T1Error = float(fp / (fp + tn))
            option.T2Error = float(fn / (fn + tp))
            tn1, fp1, fn1, tp1 = confusion_matrix(option.y_test, y_test_predicted).ravel()
            option.T1Error1 = float(fp1 / (fp1 + tn1))
            option.T2Error1 = float(fn1 / (fn1 + tp1))
        elif option.model_name == '13':
            option.trainscore = option.fitted_final.score(option.X_train, option.y_train)
            option.testscore = option.fitted_final.score(option.X_test, option.y_test)

        option.train_residual = y_train_predicted - option.y_train
        option.test_residual = y_test_predicted - option.y_test
        option.rmse_train_final = sqrt(mean_squared_error(option.y_train, y_train_predicted))
        option.rmse_test_final = sqrt(mean_squared_error(option.y_test, y_test_predicted))

        res = sum(np.square(option.y_train - y_train_predicted))
        tot = sum(np.square(option.y_train - np.mean(option.y_train)))

    n = len(option.y_train)
    if option.model_name != '9':
        p = len(option.X_train[0])
    else:
        p = 1
    option.rsquared = 1 - res / tot
    option.adj_rsquared = 1 - ((res / (n - p - 1)) / (tot / (n - 1)))

    plt.figure(1)
    plt.title('Residual Plot for Resistance')

    plt.subplot(221)
    plt.plot(y_train_predicted, option.train_residual, 'ro')
    plt.xlabel('Predicted')
    plt.ylabel('Residual')

    plt.subplot(222)
    plt.plot(option.train_residual[0:-2], option.train_residual[1:-1], 'ro')
    plt.xlabel('Residual Lag 1')
    plt.ylabel('Residual')

    plt.subplot(223)
    plt.hist(option.train_residual, bins='auto')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')

    plt.subplot(224)
    if option.model_name != '7':
        if (0 in np.nditer(np.std(option.train_residual))):
            z = option.train_residual
        else:
            z = (option.train_residual - np.mean(option.train_residual)) / np.std(option.train_residual)
        stats.probplot(z, dist="norm", plot=plt)

    plt.xlabel('Standard Quantile')
    plt.ylabel('Residual Quantile')
    plt.tight_layout()

    plt.savefig('./' + option.folderName + '/residual_plot.png')
    plt.close()

    if option.model_name == '5':
        plt.figure(2)

        X_axis = np.arange(1, len(option.y) + 1, 1)
        plt.plot(X_axis, y_pred_plot, 'b-', label=u'Prediction')
        plt.fill(np.concatenate([X_axis, X_axis[::-1]]),
                 np.concatenate([y_pred_plot - 1.9600 * sigma_plot, (y_pred_plot + 1.9600 * sigma_plot)[::-1]]),
                 alpha=.5, fc='grey', ec='None', label='95% confidence interval')
        plt.plot(option.y_train[:, 1], option.y_train[:, 0], 'r.', markersize=5)
        plt.xlabel('Observations')
        plt.ylabel('Predicted Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./' + option.folderName + '/Predictions.png')
        plt.close()

    # number of penalty coefficient is one, do 2D plot
    if (option.lamNum == 1):
        plt.figure(2)
        plt.title('Parameter Tuning Plot')
        plt.scatter(option.candidate_lambda, option.corresponding_rmse_mean)
        plt.xlabel('Candidate Lambda')
        if option.model_name not in ['6','13']:
            plt.ylabel('Root Mean Squared Error')
        else:
            plt.ylabel('Accuracy')
        plt.savefig('./' + option.folderName + '/parameter_tuning_plot.png')
        plt.close()

    # number of penalty coefficient is two, do 3D plot
    elif (option.lamNum == 2):
        plt.figure(2)
        plt.title('Parameter Tuning Plot')
        plt.scatter(option.plot3dX, option.plot3dY, c=option.corresponding_rmse_mean, cmap='gray', marker='o')
        plt.xlabel('L1 coefficient a')
        plt.ylabel('L2 coefficient b')
        plt.savefig('./' + option.folderName + '/parameter_tuning_plot.png')
        plt.close()


def model_inference_category(option):
    option.folderName = str(input('Please type in the folder name:'))
    if not os.path.exists('./' + option.folderName):
        os.makedirs('./' + option.folderName)

    if option.model_name == '7':
        y_train_predicted = option.fitted_final.predict(option.X_train)
        y_test_predicted = option.fitted_final.predict(option.X_test)
        option.train_residual = y_train_predicted - option.y_train
        option.test_residual = y_test_predicted - option.y_test
        option.rmse_train_final = sqrt(mean_squared_error(option.y_train, y_train_predicted))
        option.rmse_test_final = sqrt(mean_squared_error(option.y_test, y_test_predicted))
        res = sum(np.square(option.y_train - y_train_predicted))
        tot = sum(np.square(option.y_train - np.mean(option.y_train)))
        n = len(option.y_train)
        p = 1
        option.rsquared = 1 - res / tot
        option.adj_rsquared = 1 - ((res / (n - p - 1)) / (tot / (n - 1)))
        for i in range(option.y_train.shape[1]):

            plt.figure(1)
            plt.title('Residual Plot for Resistance')

            plt.subplot(221)
            plt.plot(y_train_predicted[:, i], option.train_residual[:, i], 'ro')
            plt.xlabel('Predicted')
            plt.ylabel('Residual')

            plt.subplot(222)
            plt.plot(option.train_residual[:, i][0:-2], option.train_residual[:, i][1:-1], 'ro')
            plt.xlabel('Residual Lag 1')
            plt.ylabel('Residual')

            plt.subplot(223)
            plt.hist(option.train_residual[:, i], bins='auto')
            plt.xlabel('Residual')
            plt.ylabel('Frequency')

            plt.subplot(224)
            if (0 in np.nditer(np.std(option.train_residual[:, i]))):
                z = option.train_residual[:, i]
            else:
                z = (option.train_residual[:, i] - np.mean(option.train_residual[:, i])) / np.std(option.train_residual[:, i])
            stats.probplot(z, dist="norm", plot=plt)

            plt.xlabel('Standard Quantile')
            plt.ylabel('Residual Quantile')
            plt.tight_layout()

            plt.savefig('./' + option.folderName + '/residual_plot' + str(i) + '.png')
            plt.close()

        plt.figure(2)
        plt.title('Parameter Tuning Plot')
        plt.scatter(option.candidate_lambda, option.corresponding_rmse_mean)
        plt.xlabel('Candidate Lambda')
        if option.model_name not in ['6','13']:
            plt.ylabel('Root Mean Squared Error')
        else:
            plt.ylabel('Accuracy')
        plt.savefig('./' + option.folderName + '/parameter_tuning_plot.png')
        plt.close()

    elif option.model_name == '8':
        y_train_predicted = option.fitted_final.evaluate(option.X_train, option.y_train, batch_size=32, verbose=1,
                                                       sample_weight=None)
        y_test_predicted = option.fitted_final.evaluate(option.X_test, option.y_test, batch_size=32, verbose=1,
                                                      sample_weight=None)
        print()
        print("Loss = " + str(y_test_predicted[0]))
        print("Test Accuracy = " + str(y_test_predicted[1]))

        accuracy = option.fitted_final.history.history['acc']
        val_accuracy = option.fitted_final.history.history['val_acc']
        loss = option.fitted_final.history.history['loss']
        val_loss = option.fitted_final.history.history['val_loss']
        epochs = range(len(accuracy))

        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig('./' + option.folderName + '/accuracy_plot.png')
        plt.close()

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.savefig('./' + option.folderName + '/loss_plot.png')
        plt.close()
    elif option.model_name == '11':
        #true regression weight
        n_rows = 1
        n_columns = option.max_rank + 1
        fig = plt.figure()
        ax = fig.add_subplot(n_rows, n_columns, 1)
        ax.imshow(T.to_numpy(option.weight_img), cmap=plt.cm.OrRd, interpolation='nearest')
        ax.set_axis_off()
        ax.set_title('Original\nweights')
        y_train_predicted = option.fitted_final.predict(option.X_train)
        
        for j in range(option.max_rank):
            
            #Visualise the learned weights
            ax = fig.add_subplot(n_rows, n_columns, j+2)
            ax.imshow(T.to_numpy(option.fitted_final_weights[j]), cmap=plt.cm.OrRd, interpolation='nearest')
            ax.set_axis_off()
            ax.set_title('Learned\nrank = {}'.format(j+1))
        
        plt.suptitle(option.tensorReg_name)
        plt.savefig('./' + option.folderName + '/Learned_regression_weights.png')
        plt.close()

    elif option.model_name == '12':
        Z_train_predicted = option.fitted_final.predict(option.X_train)['z_predicted']
        Z_test_predicted = option.fitted_final.predict(option.X_test)['z_predicted']

        option.Z_train=option.Z_train.T[0]
        option.Z_test=option.Z_test.T[0]

        print('Z_test' + str(type(option.Z_test)) + str(option.Z_test))
        print('Z_test_predicted' + str(type(Z_test_predicted)) + str(Z_test_predicted))

        option.train_residual = Z_train_predicted - option.Z_train
        option.test_residual = Z_test_predicted - option.Z_test
        option.rmse_train_final = sqrt(mean_squared_error(option.Z_train, Z_train_predicted))
        option.rmse_test_final = sqrt(mean_squared_error(option.Z_test, Z_test_predicted))

        res = sum(np.square(option.Z_train - Z_train_predicted))
        tot = sum(np.square(option.Z_train - np.mean(option.Z_train)))

        n = len(option.Z_train)
        p = len(option.X_train[0])

        option.rsquared = 1 - res / tot
        option.adj_rsquared = 1 - ((res / (n - p - 1)) / (tot / (n - 1)))

        plt.figure(1)
        plt.title('Residual Plot for Resistance')

        plt.subplot(221)
        plt.plot(Z_train_predicted, option.train_residual, 'ro')
        plt.xlabel('Predicted')
        plt.ylabel('Residual')

        plt.subplot(222)
        plt.plot(option.train_residual[0:-2], option.train_residual[1:-1], 'ro')
        plt.xlabel('Residual Lag 1')
        plt.ylabel('Residual')

        plt.subplot(223)
        plt.hist(option.train_residual, bins='auto')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')

        plt.subplot(224)

        z = (option.train_residual - np.mean(option.train_residual)) / np.std(option.train_residual)
        stats.probplot(z, dist="norm", plot=plt)

        plt.xlabel('Standard Quantile')
        plt.ylabel('Residual Quantile')
        plt.tight_layout()

        plt.savefig('./' + option.folderName + '/residual_plot.png')
        plt.close()

        plt.figure(2)
        plt.title('Parameter Tuning Plot')
        plt.scatter(option.candidate_lambda, option.corresponding_rmse_mean)
        plt.xlabel('Candidate Lambda')
        plt.ylabel('Root Mean Squared Error')
        plt.savefig('./' + option.folderName + '/parameter_tuning_plot.png')
        plt.close()

        print("goodness-of-fit: "+ str(option.adj_rsquared))
        print("training rmse: "+str(option.rmse_train_final))
        print("testing rmse: "+str(option.rmse_test_final))

