import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

class KNN():
    def __init__(self, X, y, k):
        self.data = X
        self.labels = y
        self.neighbors = k

        self.kneigh = NearestNeighbors(n_neighbors = 4, algorithm = 'kd_tree').fit(X)

        if type(self.labels[1]) == object:
            typ = 'C'
        else:
            typ = 'R'

    def predict():
        data_point = np.array([0, 1, 2, 2])
        data1 = data.reshape(1,-1)
        distances, indices = self.kneigh.kneighbors(data_point, n_neighbors = self.neighbors, metric = 'euclidean')

        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)

        for d in indices:
            ind = Counter(self.labels[d])
            lab = self.labels[d]

        if value == max(ind.values):
            plabel = [sorted(key) for key, value in ind]
        prob = max(ind.values)/self.neighbors
        return(plabel[0], prob)

        u = sum(lab)/len(lab)
        for d2 in distances:
                absolute = 0
                absolute = absolute + abs(d2 - u)
        avgd = absolute/k

        if typ == 'C':
            print(plabel[0], prob)
        else:
            print(u, avgd)

df = pd.read_csv('breast-cancer-wisconsin.data.txt') #Data set taken from UCI's sample database, thanks to a Youtuber's tutorial video
df.replace('?', -99999, inplace=True) #treat empty data as outliers
df.drop(['id'], 1, inplace=True) #remove unnecessary columns that don't help determine whether the tumor is malignant or benign
x = df.values
y = df.columns.values
k = 3
df2 = KNN(x, y, k)

