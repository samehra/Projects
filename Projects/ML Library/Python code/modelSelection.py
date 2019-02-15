import numpy as np
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel, Matern, DotProduct


def model_lasso(option):
    option.report_model = 'Lasso'
    option.lamNum=1
    if option.mode == '1':
        option.max_lambda = float(input('Please type in the maximum lambda value. max_lambda:'))
        option.lambda_range_space = float(input('Please select the number of tuning parameters evenly distributed among the valid range. lambda_range_space:'))
    else:
        option.max_lambda = 10
        option.lambda_range_space = 20

def model_elasticNet(option):
    option.report_model = 'Elastic net'
    option.lamNum = 2
    print('lambda = a+b where a is L1 coefficient and b is L2 coefficient')
    print('l1_ratio = a/(a+b), 0 <= l1_ratio <= 1.')
    if option.mode == '1':
        option.max_lambda = float(input('Please type in the maximum lambda value. max_lambda:'))
        option.lambda_range_space = float(input('Please select the number of tuning parameter sets evenly distributed among the valid range. lambda_range_space:'))
        option.ratio_range_space = float(input('Please select the number of ratio evenly distributed among [0,1]. ratio_range_space:'))
    else:
        option.max_lambda = 40
        option.lambda_range_space = 20
        option.ratio_range_space =20

def model_LAR(option):
    option.report_model = 'Least Angle Regression'
    option.lamNum=1
    option.max_lambda = float(input('Please type in the maximum lambda value. max_lambda:'))
    option.lambda_range_space = float(input('Please select the number of tuning parameters evenly distributed among the valid range. lambda_range_space:'))

def model_Kriging(option):
    option.report_model = 'Kriging'
    option.lamNum=0
    sign = -1
    kernel = None
    if option.mode == '1':
        while (sign != 0):
            choice = float(input('Choose kernel: 1.RBF 2.ExpSineSquared 3.RationalQuadratic 4.WhiteKernel:'))
            if (choice == 1):
                nextkernel = 1.0 * RBF(1.0)
            elif (choice == 2):
                nextkernel = 1.0 * ExpSineSquared(length_scale=1.3, periodicity=1)
            elif (choice == 3):
                nextkernel = 1.0 * RationalQuadratic(alpha=0.7, length_scale=1.2)
            elif (choice == 4):
                nextkernel = 1.0 * WhiteKernel(noise_level=0.0361)
            else:
                print("Bad input.")
            if (sign == 1 ):
                kernel = kernel + nextkernel
            elif (sign == 2):
                kernel = kernel * nextkernel
            elif (sign == 3):
                kernel = kernel ** nextkernel
            elif(sign == -1):
                kernel = nextkernel
            else:
                print('Bad input')
            sign = float(input('Choose sign: 1.add 2.multiply 3.exp 0.done:'))
    else:
        kernel = 1.0 * RBF(1.0)
    print('Your kernel:')
    print(kernel)
    option.kernel = kernel

def model_GaussianProcess(option):
    option.report_model = 'GaussianProcess'
    option.lamNum = 0
    sign = -1
    kernel = None
    dim = np.asarray(option.X).shape[1]
    if option.mode == '1':
        while (sign != 0):
            choice = float(input('Choose kernel: 1. RBF 2. ExpSineSquared 3. RationalQuadratic 4. WhiteKernel 5. Dotproduct 6. ConstantKernel 7. Matern:'))
            if (choice == 1):
                nextkernel = RBF(5*np.ones(dim), length_scale_bounds=(1e-3, 1e3))
            elif (choice == 2):
                nextkernel = ExpSineSquared(length_scale=1.3, periodicity=1)
            elif (choice == 3):
                nextkernel = RationalQuadratic(alpha=0.7, length_scale=1.2)
            elif (choice == 4):
                nextkernel = WhiteKernel(noise_level=1)
            elif (choice == 5):
                nextkernel = DotProduct(sigma_0=1)
            elif (choice == 6):
                nextkernel = ConstantKernel()
            elif (choice == 7):
                nextkernel = Matern(length_scale=1.0, nu=1.5)
            else:
                print("Bad input.")
            if (sign == 1 ):
                kernel = kernel + nextkernel
            elif (sign == 2):
                kernel = kernel * nextkernel
            elif (sign == 3):
                kernel = kernel ** nextkernel
            elif(sign == -1):
                kernel = nextkernel
            else:
                print('Bad input')
            sign = float(input('Choose sign: 1.add 2.multiply 3.exp 0.done:'))
    else:
        kernel = 1.0 * RBF(1.0)
    print('Your kernel:')
    print(kernel)
    option.kernel = kernel

def model_Logistic(option):
    option.report_model = 'Logistic regression'
    option.lamNum = 1

    if option.mode == '1':
        penalty = str(input('Please choose the norm used in the penalization:(1)l1 norm (2)l2 norm:'))
        option.max_lambda = float(input('Please type in the maximum lambda value. max_lambda:'))
        option.lambda_range_space = float(input('Please select the number of tuning parameters evenly distributed among the valid range. lambda_range_space:'))
    else:
        penalty = '2'
        option.max_lambda = 3000
        option.lambda_range_space = 20

    if (penalty == '1'):
        option.normSelection = 'l1'
    elif (penalty == '2'):
        option.normSelection = 'l2'
    else:
        print('Bad input')

def model_MTLLasso(option):
    option.report_model = 'MTL-Lasso'
    option.lamNum=1
    option.max_lambda = float(input('Please type in the maximum lambda value. max_lambda:'))
    option.lambda_range_space = float(input('Please select the number of tuning parameters evenly distributed among the valid range. lambda_range_space:'))

def model_ConvNets(option):
    option.report_model = 'Conv Nets'

def model_TSeries(option):
    option.report_model = 'Time Series'
    option.lamNum=1
    if option.mode == '1':
        option.futureNum = int(input('Please input the number of future data need to be predicted:'))
        option.max_lambda = float(input('Please type in the maximum lag value:'))
    else:
        option.futureNum = 7
        option.max_lambda = 50
    
def model_ModelTrees(option):
    option.report_model = 'Model Trees'
    option.lamNum=0
    
def model_TensorReg(option):
    option.report_model = 'Tensor Regression'
    option.lamNum=0
    if option.mode == '1':
        option.max_rank = int(input('Please specify the max rank to be considered for Tensor Regression (usually a values less than 10):'))
        option.tensorReg_type = str(input('Please specify the type of regression: (1)Kruskal Regression (2)Tucker Regression:'))
    else:
        option.max_rank = 5
        option.tensorReg_type = '1'
    if option.tensorReg_type == '1':
        option.tensorReg_name = 'Kruskal regression'
    elif option.tensorReg_type == '2':
        option.tensorReg_name = 'Tucker regression'

def model_InsituEnsemble(option):
    option.report_model = 'In-situ Ensemble Modeling'
    option.lamNum=1
    if option.mode == '1':
        option.max_lambda = float(input('Please type in the maximum lambda value. max_lambda:'))
        option.lambda_range_space = float(input('Please select the number of tuning parameters evenly distributed among the valid range. lambda_range_space:'))
    else:
        option.max_lambda = 100
        option.lambda_range_space = 20

def model_SVM(option):
    option.report_model = 'SVM'
    option.lamNum=1
    if option.mode == '1':
        option.max_lambda = float(input('Please type in the maximum penalty parameter C value. max_C:'))
        option.lambda_range_space = float(input('Please select the number of tuning parameters evenly distributed among the valid range. lambda_range_space:'))
    else:
        option.max_lambda = 10
        option.lambda_range_space = 20
'''
add a new function for each new model
'''