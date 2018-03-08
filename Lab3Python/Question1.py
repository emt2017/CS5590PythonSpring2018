#1)Pick any dataset from the dataset sheet in class sheet and make one
# prediction model using your imagination with Linear Discriminant Analysis*.
# Some examples are:
# a.In the report provide convincible explanations about the difference
# of logistic regression and Linear Discriminant Analysis.
# b.You can also pick dataset of your own.

from sklearn.datasets import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()

#extract data from iris dataset
data = iris.data #store the flower data
target = iris.target #store the target data

#apply cross validation to randomly create training and test data sets for better coverage
xTrain, xTest, yTrain, yTest = train_test_split(data, target, test_size=0.2)

#create a model using linear discriminant analysis, fit the training data in the model, predict test values.
model = LinearDiscriminantAnalysis(n_components=2).fit(xTrain,yTrain).predict(xTest)

#compare predicted test values with the test labels to calculate the accuracy.
error = np.mean(yTest != model) #calculate accuracy

#print value of error
print(error)

#logistic regression:
'''
The betas in the equation:
	log((p1(x))/(1-p1(x)))=B_0+B_1 x
	Are estimated using maximum likelihood estimate, while:

'''
#Discriminant analysis:
'''
maximizes the difference by using the estimated mean/variance of the normal distribution
'''

