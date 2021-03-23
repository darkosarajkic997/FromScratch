
import pandas as pd
import numpy as np
import random

class KMeans:

    def __init__(self, max_inter=100, n_clusters=2):
        self.n_clusters=n_clusters
        self.max_iter=max_inter
        self.centroids=[]


    def fit(self,dataframe):
        rand_centroids = random.sample(range(0,  len(dataframe.index)), self.n_clusters)
        for i in rand_centroids:
            self.centroids.append(dataframe.loc[i])

        dataframe['cluster'] = dataframe.apply(lambda row : self.find_cluster(row,self.centroids), axis = 1)
        for index in range(0,self.max_iter):
            self.find_new_centroids(dataframe)
            dataframe['cluster'] = dataframe.drop(['cluster'],axis=1).apply(lambda row : self.find_cluster(row,self.centroids), axis = 1)

        return dataframe
            

    def calc_distance(self, centroid, data_row):
        return np.sum(np.square(np.subtract(centroid.to_numpy(),data_row.to_numpy())))**0.5    


    def find_cluster(self, row, centroids):
        distance=[]
        for centroid in centroids:
            distance.append(self.calc_distance(centroid, row))
        
        return np.argmin(distance)

    def find_new_centroids(self, dataframe):
        new_centroids=dataframe.groupby(['cluster']).mean().values.tolist()
        self.centroids=[]
        for centroid in new_centroids:
            self.centroids.append(pd.Series(centroid))

    
    def find_error(self, dataframe):
        tmp_dataframe=dataframe.drop(['cluster'],axis=1)
        clusters=dataframe['cluster']
        return sum(tmp_dataframe.apply(lambda row: (self.calc_distance(self.centroids[clusters[row.name]], row))**2, axis=1))

