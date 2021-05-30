from hashlib import new
from os import curdir, name
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import select
from pandas.core.algorithms import SelectNFrame, mode
from sklearn import datasets
import pandas as pd
import random

data = datasets.load_iris()

iris = pd.DataFrame(data.data, columns=data.feature_names)
print(iris)


class Kmeans :

    def __init__(self, k = 2, max_iter = 300, tol = 0.001) :
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        
    def init_centeres(self, X):
        return [X.loc[random.randint(0, X.shape[0])] for _ in range(self.k)]



    def assign_centers(self, X, cents):
        clusters = []
        for i in range(X.shape[0]):
            dist = []
            for j in cents :
                dist.append(np.linalg.norm(X.loc[i] - j))
            
            clusters.append(np.argmin(dist))
        
        return clusters



    def centroid_recalc(self, X, clusters, centers):
        #create dataframe of datapoints and assigned cluster
        X['clusters'] = clusters
        new_cents  =[]
        for c in range(self.k):
            x_cen = X[X.columns[:-1]].loc[X['clusters'] == c]
            new_cents.append(np.mean(x_cen))

            

        return new_cents

        
            
    def fit(self, X):
       
        cur_cents = self.init_centeres(X)
        old_cs =  []
        clusters = self.assign_centers(X, cur_cents)
        
        
        optimized = True
        for i in range(self.max_iter):
            if i > 0 and np.sum((cur_cents - old_cs)/old_cs*100.0) > self.tol :
                old_cs = cur_cents
                cur_cents = self.centroid_recalc(X, clusters, 0)
                clusters = self.assign_centers(X, cur_cents)
                optimized = False
            if optimized :
                break
            
        

        return clusters
            

if __name__ == '__main__':
    model = Kmeans(3)
    x = np.array(model.fit(iris))
    print(x)

    plt.scatter(iris.values[x == 0,0], iris.values[x == 0,1], c = 'red')
    plt.scatter(iris.values[x == 1,0], iris.values[x == 1,1],  c = 'green')
    plt.scatter(iris.values[x == 2,0], iris.values[x == 2,1], c = 'yellow')
    
    plt.show()


    
   
            




