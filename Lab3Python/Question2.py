#2)Implement Support Vector Machineclassification,
# 1)Choose one of the dataset using the datasets features in the scikit-learn#####
# 2)Load the dataset##############################################################
# 3)According to your dataset, split the data to 20% testing data, 80% training data(you can also use any other number)#
# 4)Apply SVC with Linear kernel
# 5)Apply SVC with RBF kernel
# 6)Report the accuracy of the model on both models separately and report their differences if there is
# 7)Report your view how can you increase the accuracy and which kernel is the best for your dataset and why

from sklearn.datasets import *
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split

iris = load_iris()
data = iris.data
targets = iris.target

#apply cross validation to data
x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size = 0.2, random_state = 5)

#create SVC model with linear kernel, fit the training data to the model, predict targets for test
model = SVC(kernel='linear').fit(x_train, y_train).predict(x_test)

#compare predicted test values with the test labels to calculate the accuracy.
error = np.mean(y_test != model) #calculate accuracy
print('linear kernel:')
print(error)

#create SVC model with radial basis function kernel, fit the training data to the model, predict targets for test
model = SVC(kernel='rbf').fit(x_train, y_train).predict(x_test)

#compare predicted test values with the test labels to calculate the accuracy.
error = np.mean(y_test != model) #calculate accuracy
print('rbf kernel:')
print(error)

#we can normalize the data and use ensembling to improve the accuracy
#determine if the model is linear or non-linear. If the model is non-linear then the radial basis function would be used.
#split into a test, training, and validation set.
