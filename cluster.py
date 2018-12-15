import pandas as pd
import math
import json
import numpy as np

pd.options.mode.chained_assignment = None

class Point:
    def __init__(self, pattern_id):
        self.length = len(pattern_id)
        self.pattern_id = pattern_id
        self.z = -1

    def __str__(self):
        return str(self.pattern_id)

    def toJSON(self):
        return {
            'pattern_id':self.pattern_id
        }


class Cluster:
    def __init__(self, dim, centroid):
        self.dim = dim
        self.centroid = centroid
        self.points = []
        self.distances = []

    # this method finds the average distance of all elements in cluster to its centroid
    def computeS(self):
        n = len(self.points)
        if n == 0:
            return 0
        s = 0
        for x in self.distances:
            s += x
        return float(s / n)


class Clustering:
    def __init__(self, generation, data, kmax):
        self.generation = generation
        self.data = data
        self.dim = data.shape[1]
        self.penalty = 1000000
        self.kmax = kmax
        # print self.dim

    def daviesBouldin(self, clusters):
        sigmaR = 0.0
        nc = len(clusters)
        for i in range(nc):
            sigmaR = sigmaR + self.computeR(clusters)
        DBIndex = float(sigmaR) / float(nc)
        return DBIndex

    def computeR(self, clusters):
        listR = []
        for i, iCluster in enumerate(clusters):
            for j, jCluster in enumerate(clusters):
                if(i != j):
                    temp = self.computeRij(iCluster, jCluster)
                    listR.append(temp)
        return max(listR)

    def computeRij(self, iCluster, jCluster):
        Rij = 0

        d = self.euclidianDistance(
            iCluster.centroid, jCluster.centroid)
        #print("d",d)
        #print("icluster",iCluster.computeS())
        Rij = (iCluster.computeS() + jCluster.computeS()) / d

        #print("Rij:", Rij)
        return Rij

    def euclidianDistance(self, point1, point2):
        sum = 0
        for i in range(0, point1.length):
            square = pow(
                point1.pattern_id[i] - point2.pattern_id[i], 2)
            sum += square

        sqr = math.sqrt(sum)
        return sqr

    def calcDistance(self, clusters):
        kmax = self.kmax
        dim = self.dim
        data = self.data
        dis = 0
        disSet = []

        for z in range(data.shape[0]):  #count no of rows
            point = Point(data.loc[z][0:dim]) #new point from row z
            point.z = z

            for i in range(kmax):
                dis = self.euclidianDistance(clusters[i].centroid, point) #calculate distance from point to each cluster centroids
                disSet.append(dis) #add distance to array
                dis = 0

            clusters = self.findMin(
                disSet, clusters, point)
            disSet = []  # clear disSet	# calculate distance

        return clusters

    def findMin(self, disSet, clusters, point):
        n = disSet.index(min(disSet))  # n is index
        minDis = disSet[n]
        clusters[n].points.append(point)
        clusters[n].distances.append(minDis)

        return clusters

    # childChromosome, kmax
    def calcChildFit(self, childChromosome):
        kmax = self.kmax
        dim = self.dim
        clusters = []
        for j in range(kmax):
            point = Point(childChromosome.genes[j * dim: (j + 1) * dim])
            c = Cluster(dim, point)
            clusters.append(c)
            # print(c.centroid)

        clusters = self.calcDistance(clusters)
        DBIndex = self.daviesBouldin(clusters)

        childChromosome.fitness = 1 / DBIndex

        return childChromosome

    def calcChromosomesFit(self):
        kmax = self.kmax
        generation = self.generation
        numOfInd = generation.numberOfIndividual
        # print "num of ind " + str(numOfInd)
        data = self.data
        chromo = generation.chromosomes
        # print chromo[0].genes

        for i in range(0, numOfInd):

            dim = self.dim
            clusters = []
            for j in range(kmax):
                point = Point(chromo[i].genes[j * dim: (j + 1) * dim])
                # print (("point is ") + str(point))
                clusters.append(Cluster(dim, point))

            #assign DBIndex to each chromosome fitness value
            clusters = self.calcDistance(clusters)
            DBIndex = self.daviesBouldin(clusters)
            generation.chromosomes[i].fitness = 1 / DBIndex

        return generation

    def printIBest(self, iBest):
        kmax = self.kmax
        dim = self.dim
        clusters = []
        for j in range(kmax):
            point = Point(iBest.genes[j * dim: (j + 1) * dim])
            clusters.append(Cluster(dim, point))

        clusters = self.calcDistance(clusters)
        DBIndex = self.daviesBouldin(clusters)
        z = (np.zeros(1728)).tolist() #default 150
        for i, cluster in enumerate(clusters):
            for j in cluster.points:
                # print (i)
                z[j.z] = i

        correct_answer = 0
        for i in range(0, 50):
            # print ("z[i] is: " + str(z[i]))
            if z[i] == 2:
                correct_answer += 1
        for i in range(50, 100):
            # print ("z[i] is: " + str(z[i]))
            if z[i] == 1:
                correct_answer += 1
        for i in range(100, 150):
            # print ("z[i] is: " + str(z[i]))
            if z[i] == 0:
                correct_answer += 1

        # accuracy = (correct_answer / 150) * 100

        # print("accuracy :", accuracy)
        print("iBest Fitness:", 1 / DBIndex)
        print("all index:", z)
        print("Clusters centroid:")

        for i, cluster in enumerate(clusters):
            print("centroid", i, " :", cluster.centroid)

        return clusters

    def output_result(self, iBest, data, clusters2):
        print("Saving the result...")
        kmax = self.kmax
        dim = self.dim
        clusters = []
        for j in range(kmax):
            point = Point(iBest.genes[j * dim: (j + 1) * dim])
            clusters.append(Cluster(dim, point))

        clusters = self.calcDistance(clusters)
        centroids = []
        for i in range(kmax):
            centroids.append(clusters[i].centroid)
        z = (np.zeros(1728)).tolist() #default 150
        for i, cluster in enumerate(clusters):
            for j in cluster.points:
                z[j.z] = i

        with open('result/cluster_center.json', 'w') as outfile:
            json.dump([e.toJSON() for e in centroids], outfile, sort_keys=True,
                      indent=4, separators=(',', ': '))

        # rename df header
        col_name = list()
        for i in range(data.shape[1]):
            col_name.append("f{0}".format(i))
        data.columns = col_name


        # for i in range(data.shape[0]):
        #     print(i)

        # insert cluster result
        data['Cluster Index'] = pd.Series(z, index=data.index)
        data.to_csv('result/result.csv', index=None)

        dataInCluster0 = []
        dataInCluster1 = []
        dataInCluster2 = []
        dataInCluster3 = []

        #print(data.shape[0])
        # iterate number of rows
        for i in range(data.shape[0]):
            dataRow = [data.iloc[i][0], data.iloc[i][1], data.iloc[i][2], data.iloc[i][3], data.iloc[i][4], data.iloc[i][5]] #stores current data in row i

            target = data.iloc[i][6] #get target class of row i
            # store data points according to cluster
            if target == 0:
                dataInCluster0.append(dataRow)
            elif target == 1:
                dataInCluster1.append(dataRow)
            elif target == 2:
                dataInCluster2.append(dataRow)
            elif target == 3:
                dataInCluster3.append(dataRow)


        print("SSE Cluster 0 " + str(self.SSE(clusters2[0].centroid, dataInCluster0)))
        print("SSE Cluster 1 " + str(self.SSE(clusters2[1].centroid, dataInCluster1)))
        print("SSE Cluster 2 " + str(self.SSE(clusters2[2].centroid, dataInCluster2)))
        print("SSE Cluster 3 " + str(self.SSE(clusters2[3].centroid, dataInCluster3)))


        print("Done.")

    def SSE(self, clusterAttribute, dataInCluster):
        sse = 0
        for i in range(len(dataInCluster)):
            for j in range(len(dataInCluster[i])):
                sse += (abs(dataInCluster[i][j] - clusterAttribute.pattern_id[j]))**2
        return sse