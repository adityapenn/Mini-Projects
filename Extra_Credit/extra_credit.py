import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

class KNN():
    def __init__(self, X, y, k):
        self.dataset = X
        self.label = y
        self.number = k
        x_train,x_test,y_train,y_test=cross_validation.train_test_split(self.dataset,self.label,test_size=0.2) #Cross-validating data
        print('Dataset:', self.dataset[:5])
        print('Labels:', self.label[:5])
        clf=neighbors.KNeighborsClassifier(n_neighbors = self.number, algorithm = 'kd_tree') #Maximum accuracy achieved with 7 neighbors
        clf.fit(x_train,y_train)
        accuracy=clf.score(x_test,y_test) #Calculating accuracy
        print("Accuracy is:", accuracy)
        #Making all variables in 'init' function global so they can be used in the 'predict' function
        global a
        global b
        global c
        global lf
        a = self.number
        b = self.dataset
        c = self.label
        lf = clf.fit(x_train, y_train)
    def predict(data_point):
        lf=neighbors.KNeighborsClassifier(n_neighbors = a, algorithm = 'kd_tree') #Maximum accuracy achieved with 7 neighbors
        x_train,x_test,y_train,y_test=cross_validation.train_test_split(b,c,test_size=0.2)
        lf.fit(x_train,y_train)
        prediction=lf.predict(data_point) #Returns the label as an array
        x = " ".join(str(x) for x in prediction) #Returns the label as a string
        print(type(x))
        posterior = len(x)/a #Finds the posterior probability
        print(posterior)
        print(type(posterior))

dataf=pd.read_csv('C:\\Users\\adity\\Desktop\\bloodxls1.csv') #Reading CSV file that consists the following information
data=np.array(dataf.drop(['march'],1)) #Dropping unnecessary columns
labels=np.array(dataf['march']) #Taking 'March' as label
k = 7
KNN(data, labels, k)
dp=np.array([2,50,10200,98])
dp=dp.reshape(1,-1)
KNN.predict(dp)

#Task 2: Gradient Descent

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import math
get_ipython().magic('matplotlib inline')

def sigmoid(weight_vector, x_vector, return_deriv=True):
    sig = float(1) / (1 + math.e**(-x_vector.dot(weight_vector)))
    deriv = sig * (1 - sig)
    if return_deriv:
        return sig, deriv
    else:
        return sig


from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=26)


X = np.concatenate([np.ones((100,1)), X], axis=1)
X.shape

X[:3]


w_vector = np.zeros(3)
w_vector = np.random.uniform(1, 3, 3)
weight_vector = weight_vector.reshape(3,1)
weight_vector

from sklearn.metrics import accuracy_score
print(X.dot(weight_vector)[:5])
accuracy_score(np.round(X.dot(weight_vector)), y)

def error(weight_vector, x_vector, y):
    log_func_v = sigmoid(weight_vector, x_vector)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1-y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return np.mean(final)

#Task 2.1 - Stochastic Gradient Descent
n_epochs = 50
t0, t1 = 5, 50 # learning schedule hyperparameters
def perform_stochastic_gradient_descent(weight_vector, X, y, n_epochs, m):
    for epoch in range(n_epochs):
        for i in range(m):
            random_number = np.random.randint(m)
            x_vector = X[random_number:random_number+1]
            yi = y[random_number:random_number+1]
            gradients = 2 * x_vector.T.dot(x_vector.dot(weight_vector) - yi)
            eta = 0.1 #learning rate
            weight_vector += eta * error(weight_vector, x_vector) * sigmoid(weight_vector, x_vector) * x_vector
            print(accuracy_score(np.round(X.dot(weight_vector)), y))
        return weight_vector
        
#Task 2.2 - Batch Gradient Descent
def perform_batch_gradient_descent(weight_vector, X, y, batch_size=100):
        n_iterations = 1000
        for iteration in range(n_iterations):
            gradients = 2/m * X.T.dot(X.dot(weight_vector) - y)
            eta = 0.1 #learning rate
            weight_vector += eta * error(weight_vector, X, y) * sigmoid(weight_vector, X) * X
            print(accuracy_score(np.round(X.dot(weight_vector)), y))
        return weight_vector


#Task 2.3 - Batch Gradient Descent with new data
def perform_batch_gradient_descent_new(weight_vector, batch_size=100):
    iris = datasets.load_iris()
    m=100
    X = iris.data [:3, 1:]  #Batch size reduced
    y = iris.target [:1]
    n_iterations = 1000
    for iteration in range(n_iterations):
        gradients = 2/m * X.T.dot(X.dot(weight_vector) - y)
        eta = 0.1 #learning rate
        weight_vector += eta * error(weight_vector, X, y) * sigmoid(weight_vector, X) * X
        print(accuracy_score(np.round(X.dot(weight_vector)), y))
    return weight_vector

perform_batch_gradient_descent_new(weight_vector)
perform_batch_gradient_descent(weight_vector, X, y)
perform_stochastic_gradient_descent(weight_vector, X, y, n_epochs, 50)
