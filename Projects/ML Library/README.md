I. Basic Steps:
Open the The_Machine.py file which is the main module that calls all the other submodules. Keep all the .py files in the same folder and follow the instructions below for the various models that have been implemented by this library.

1.	Mode Selection:
There are two modes available: (1) normal mode, (2) easy mode. The normal mode will ask user for specific methods of partition and tuning, and the easy mode will use default settings.
Example: 
Choose Options by Entering Corresponding Number
Choose mode: (1)normal mode (2)easy mode:1
By entering 1, the normal mode will be selected.

2.	Model Selection: 
Example:
Choose model: (1)lasso (2)elastic net (3)Least Angle Regression (4)Kriging (5)Gaussian Process (6)Logistic Regression (7)MTL-Lasso (8)Conv Nets (9)Time Series: 1
By entering 1, Lasso will be selected.

3.	Partition Method Selection
Example:
Choose partition method: (1)0.9 partition (2)0.01 partition: 1
By entering 1, 0.9 partition will be selected.


4.	Tuning Method 
Example:
Choose tuning method: (1)5-fold cv: 1
By entering 1, 5-fold cv will be selected.

5.	Choose Input Data File
The user need to type in the name of the file including the extension. If the data file is not in the current folder, the user need to include the path before the file.
Example:
Input data file name: data.csv
Input data file name: C:\Users\Saurabh\Desktop\data.csv

6.	Choose Column Index of Predictor Variable and Response Variable
The counting of column begin from 1. Take following data for example:
 

In this data file, the index from X1 to X5 is 1 to 5, and the index of y is 6. Therefore, the input should be：
Input idex of first column of X (start from 1): 1
Input idex of last column of X: 5
Input idex of the column of Y: 6
After the indexes are selected, the data will be loaded. This might take some time if the dataset is very large.
P.S. The input interface could be different depending on the model selected. For example, the multi-task lasso will need to select multiple columns of response Y. 

II. Model Parameters:
1.	Lasso:
The program will first ask for the maximum lambda value (max_lambda), and then ask for number of lambdas(lambda_range_space) evenly distributed among 0 to max_lambda. The program will train the model by using these lambdas and find the optimized lambda.
Example:
Please type in the maximum lambda value. max_lambda: 10
Please select the number of tuning parameters evenly distributed among the valid range. lambda_range_space: 11
	In this case, the lambda candidates would be integers 0 to 10
After this process, the training will begin.

2.	Elastic Net:
In this model, the variable lambda is the sum of l1 coefficient a and l2 coefficient b, and the variable l1_ratio is the ratio a/(a+b). The program will ask for the maximum value of lambda, then ask for number of lambda evenly distributed from 0 to max_lambda, and number of ratios evenly distributed from 0 to 1.
Example:
lambda = a+b where a is L1 coefficient and b is L2 coefficient
l1_ratio = a/(a+b), 0 <= l1_ratio <= 1.
Please type in the maximum lambda value. max_lambda:10
Please select the number of tuning parameter sets evenly distributed among the valid range. lambda_range_space:11
Please select the number of ratio evenly distributed among [0,1]. ratio_range_space:20
In this case, the lambda candidates would be integers 0 to 10, and for each lambda, 20 ratio combinations will be used to find the optimized lambda and ratio.
After this process, the training will begin.

3.	Least Angle Regression:

The program will first ask for the maximum lambda value (max_lambda), and then ask for number of lambdas(lambda_range_space) evenly distributed among 0 to max_lambda. The program will train the model by using these lambdas and find the optimized lambda.
Example:
Please type in the maximum lambda value. max_lambda: 10
Please select the number of tuning parameters evenly distributed among the valid range. lambda_range_space: 11
In this case, the lambda candidates would be integers 0 to 10
After this process, the training will begin.

4.	Kriging
This model will ask for the kernel will be used. There are four basic kernels available and the user could combine these kernels if needed.
Example: 
Choose kernel: 1.RBF 2.ExpSineSquared 3.RationalQuadratic 4.WhiteKernel:1
Choose sign: 1.add 2.multiply 3.exp 0.done:1
Choose kernel: 1.RBF 2.ExpSineSquared 3.RationalQuadratic 4.WhiteKernel:2
Choose sign: 1.add 2.multiply 3.exp 0.done:2
Choose kernel: 1.RBF 2.ExpSineSquared 3.RationalQuadratic 4.WhiteKernel:3
Choose sign: 1.add 2.multiply 3.exp 0.done:0
Your kernel:
1**2 * RBF(length_scale=1) + 1**2 * ExpSineSquared(length_scale=1.3, periodicity=1) * 1**2 * RationalQuadratic(alpha=0.7, length_scale=1.2)
In this case the kernel is RBF + ExpSineSquared * RationalQuadratic
After this process, the training will begin.

5.	Gaussian Process

This model will ask for the kernel will be used. There are seven basic kernels available and the user could combine these kernels if needed.
Example: 
Choose kernel: 1. RBF 2. ExpSineSquared 3. RationalQuadratic 4. WhiteKernel 5. Dotproduct 6. ConstantKernel 7. Matern:1
Choose sign: 1.add 2.multiply 3.exp 0.done:1
Choose kernel: 1. RBF 2. ExpSineSquared 3. RationalQuadratic 4. WhiteKernel 5. Dotproduct 6. ConstantKernel 7. Matern:2
Choose sign: 1.add 2.multiply 3.exp 0.done:2
Choose kernel: 1. RBF 2. ExpSineSquared 3. RationalQuadratic 4. WhiteKernel 5. Dotproduct 6. ConstantKernel 7. Matern:3
Choose sign: 1.add 2.multiply 3.exp 0.done:0
Your kernel:
1**2 * RBF([5,5,5,5]) + 1**2 * ExpSineSquared(length_scale=1.3, periodicity=1) * 1**2 * RationalQuadratic(alpha=0.7, length_scale=1.2)
In this case the kernel is RBF + ExpSineSquared * RationalQuadratic
After this process, the training will begin.

6.	Logistic Regression
This model will ask to choose between l1 norm and l2 norm. Then, it will ask for the maximum value of lambda (max_lambda) and number of lambda evenly distributed from 0 to max_lambda. In this model the lambda is the inverse of regularization strength, and smaller values specify stronger regularization.
Example:
Please choose the norm used in the penalization:(1)l1 norm (2)l2 norm:1
Please type in the maximum lambda value. max_lambda:1000
Please select the number of tuning parameters evenly distributed among the valid range. lambda_range_space:11
	In this case, the lambda candidates would be integers 0, 100, 200, …, 900, 1000.
After this process, the training will begin.

7.	MTL-Lasso
Same with Lasso.

8.	Conv Nets
This model has been primarily built for image classification. It asks for image datasets of train and test image separately and then creates the training and test datasets. The model parameters has currently been kept fixed and since a random set of parameters could really effect the run time. Certain range for parameters and model enhancements might be added later.

9.	Time Series
This model will ask for the number of future data the user would like to predict, and the maximum lag value (max_lambda). Then the program will train the model with lag values from 1 to lambda and find the optimized lag value.
Example:
Please input the number of future data need to be predicted:7
Please type in the maximum lag value:30
In this case the lag candidates would be integers from 1 to 30. 
After this process, the training will begin.
10.	Model Trees
No user input apart from model selection. The model automatically computes the appropriate partition using the error metric below.
 
III. Report Output
	After training the user need to type in the folder name which will be used to store relative images and report. If the folder does not exist, the program will create one automatically.
Example:
	Please type in the folder name: TestFolder
In this case the report and images will be stored in the folder called TestFolder in current path.

