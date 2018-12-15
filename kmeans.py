import pandas as pd
from random import sample
from math import sqrt
from numpy import mean
import copy
import csv
from pandas import DataFrame

class KMeans:
    """ To calculate K-means without GA """

    def __init__ (self, k, maxIterations):
        self.k = k
        self.maxIterations = maxIterations


    def initializeCenters(self, df, k):
        random_indices = sample(range(len(df)), k)
        centers = [list(df.iloc[idx]) for idx in random_indices]
        print("Random Indices : " + str(random_indices))
        return centers


    def computeCenter(self, df, k, cluster_labels):
        cluster_centers = list()
        data_points = list()
        for i in range(k):
            for idx, val in enumerate(cluster_labels):
                if val == i:
                    data_points.append(list(df.iloc[idx]))
            cluster_centers.append(list(map(mean, zip(*data_points))))
        return cluster_centers


    def euclidean_distance(self, x, y):
        summ = 0
        for i in range(len(x)):
            # print(x[i] - y[i])
            term = (x[i]-y[i])**2
            summ += term
        return sqrt(summ)


    def assignCluster(self, df, k, cluster_centers):
        cluster_assigned = list()
        # print(cluster_centers)
        for i in range(len(df)):
            distances = [self.euclidean_distance(list(df.iloc[i]), center) for center in cluster_centers]
            min_dist, idx = min((val, idx) for (idx, val) in enumerate(distances))
            # print(idx)
            cluster_assigned.append(idx)
        return cluster_assigned


    def kmeans(self, df, k):
        cluster_centers = self.initializeCenters(df, k)
        curr = 1
        while curr <= self.maxIterations:
            cluster_labels = self.assignCluster(df, k, cluster_centers)
            print("cl: " + str(cluster_labels))
            prev_centers = copy.deepcopy(cluster_centers)
            print('pc: ' + str(prev_centers))
            cluster_centers = self.computeCenter(df, k, cluster_labels)
            print('cc: ' + str(cluster_centers) + str("\n"))
            curr += 1

        dataInCluster0 = []
        dataInCluster1 = []
        dataInCluster2 = []
        dataInCluster3 = []

        for i in range(len(cluster_labels)):
            dataRow = [df.iloc[i][0], df.iloc[i][1], df.iloc[i][2], df.iloc[i][3], df.iloc[i][4], df.iloc[i][5]] #stores current data in row i

            target = cluster_labels[i] #get target class of row i
            # store data points according to cluster
            if target == 0:
                dataInCluster0.append(dataRow)
            elif target == 1:
                dataInCluster1.append(dataRow)
            elif target == 2:
                dataInCluster2.append(dataRow)
            elif target == 3:
                dataInCluster3.append(dataRow)

        print("SSE Cluster 0 " + str(self.SSE(cluster_centers[0], dataInCluster0)))
        print("SSE Cluster 1 " + str(self.SSE(cluster_centers[1], dataInCluster1)))
        print("SSE Cluster 2 " + str(self.SSE(cluster_centers[2], dataInCluster2)))
        print("SSE Cluster 3 " + str(self.SSE(cluster_centers[3], dataInCluster3)))


    def SSE(self, clusterAttribute, dataInCluster):
        sse = 0
        for i in range(len(dataInCluster)):
            for j in range(len(dataInCluster[i])):
                sse += (abs(dataInCluster[i][j] - clusterAttribute[j]))**2
        return sse