import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

class KNN():
    def __init__(self, X, y, k):
        self.data = X
        self.labels = y
        self.neighbors = k

        self.kneigh = NearestNeighbors(n_neighbors = 4, algorithm = 'kd_tree').fit(X) #Fitting our data X to the nearest neighbors KD-Tree algorithm

        if type(self.labels[1]) == object: #Classifiers will have objects/strings as labels, as opposed to regressors which will have numerical values as labels
            typ = 'C'
        else:
            typ = 'R' #The usage of 'typ' will be observed later in the program

        data_point = np.array([0, 1, 2, 2]) #input argument to the following function
    def predict(data_point):

        data1 = data_point.reshape(1,-1)
        distances, indices = self.kneigh.kneighbors(data1, n_neighbors = self.neighbors, metric = 'euclidean') #Fitting our reshaped argument array to the k-neighbors algorithm

        for d in indices:
            ind = Counter(self.labels[d]) #Dict with each label and number of times it occurs

        if typ == 'C': #For classifiers only
            value = max(ind.values)
            plabel = [sorted(key) for key, value in ind] #Get sorted labels
            prob = max(ind.values)/self.neighbors #Finding posterior probability
            return(plabel[0], prob)
        else:
            lab = [self.labels[d] for d in indices]
            u = sum(lab)/len(lab)
            for d2 in distances:
                    absolute = 0
                    absolute = absolute + abs(d2 - u)
            avgd = absolute/k
            print(u, avgd)

df = pd.read_csv('breast-cancer-wisconsin.data.txt') #Data set taken from UCI's sample database, thanks to a Youtuber's tutorial video
df.replace('?', -99999, inplace=True) #treat empty data as outliers
df.drop(['id'], 1, inplace=True) #remove unnecessary columns that don't help determine whether the tumor is malignant or benign
x = df.values
y = df.columns.values
k = 3
df2 = KNN(x, y, k)
