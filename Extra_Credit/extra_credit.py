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
