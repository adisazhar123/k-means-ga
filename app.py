import configparser
import numpy as np
import pandas as pd
import csv

from cluster import Clustering
from genetic import Genetic
from generation import Generation
from pandas import DataFrame
from kmeans import KMeans

NORMALIZATION = True


def readVars(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    budget = int(config.get("vars", "budget"))
    kmax = int(config.get("vars", "kmax"))  # Maximum number of Clusters
    numOfInd = int(config.get("vars", "numOfInd"))  # number of individual
    Ps = float(config.get("vars", "Ps"))
    Pm = float(config.get("vars", "Pm"))
    Pc = float(config.get("vars", "Pc"))

    return budget, kmax, Ps, Pm, Pc, numOfInd


# minmax normalization, input is dataset with Panda DataFrame format
def minMax(data):
    data[0] = (data[0] - data[0].min()) / (data[0].max() - data[0].min())
    data[1] = (data[1] - data[1].min()) / (data[1].max() - data[1].min())
    data[2] = (data[2] - data[2].min()) / (data[2].max() - data[2].min())
    data[3] = (data[3] - data[3].min()) / (data[3].max() - data[3].min())
    data[4] = (data[4] - data[4].min()) / (data[4].max() - data[4].min())
    data[5] = (data[5] - data[5].min()) / (data[5].max() - data[5].min())

    return data




def loadDatasetCV(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)	#, quoting=csv.QUOTE_NONNUMERIC
        dataset = []
        dataset2 = []
        line = []
        # this object is used to change
        alias = {
            '0': {
                'low': 1, 'med': 2, 'high': 3, 'vhigh': 4
                },
            '1': {
                'low': 1, 'med': 2, 'high': 3, 'vhigh': 4
                },
            '2': {
                '2': 2, '3': 3, '4': 4, '5more': 5
                },
            '3': {
                '2': 2, '4': 4, 'more': 5
                },
            '4': {
                'small': 1, 'med': 2, 'big': 3
                },
            '5': {
                'low': 1, 'med': 2, 'high': 3
                },
            '6': {
                 'unacc\n': 1, 'acc\n': 2, 'good\n': 3, 'vgood\n': 4
                 }
            }

        for line in csvfile:
            data = line.split(',')
            ok = data
            row = []
            row2 = []
            for i in range(len(data) - 1):
                row.append(alias[str(i)][ok[i]])
            dataset.append(row)

            for i in range(len(data)):
                row2.append(alias[str(i)][ok[i]])
            dataset2.append(row2)

        return dataset, dataset2


if __name__ == '__main__':
    d, d2 = loadDatasetCV('car.csv')
    dataset = DataFrame.from_records(d)
    normData = minMax(dataset)
    normData.to_csv('result/norm_data2.csv', index=None, header=None)
    data = normData

    columns = list(data.columns)
    features = columns[:len(columns)]

    classLabels = DataFrame.from_records(d2)[6]  # get class label, in the last column
    df = data[features]

    #K-means Without GA
    kmeans = KMeans(4, 5)
    #
    kmeans.kmeans(df, kmeans.k)




    config_file = "config2.txt"
    dim = data.shape[1]
    # kmeans parameters & GA parameters
    generationCount = 0
    budget, kmax, Ps, Pm, Pc, numOfInd = readVars(config_file)
    #
    #
    print("-------------GA Info-------------------")
    print("budget", budget)
    print("kmax", kmax)
    print("numOfInd", numOfInd)
    print("Ps", Ps)
    print("Pm", Pm)
    print("Pc", Pc)
    print("---------------------------------------")

    # dim or pattern id
    chromosome_length = kmax * dim
    # print chromosome_length
    # #-------------------------------------------------------#
                        # main 						#
    # #-------------------------------------------------------#
    initial = Generation(numOfInd, 0)
    initial.randomGenerateChromosomes(
        chromosome_length)  # initial generate chromosome

    clustering = Clustering(initial, data, kmax)  # eval fit of chromosomes

    # ------------------calc fitness------------------#
    generation = clustering.calcChromosomesFit()

    # ------------------------GA----------------------#
    while generationCount <= budget:
        GA = Genetic(numOfInd, Ps, Pm, Pc, budget, data, generationCount, kmax) #init constructor
        generation, generationCount = GA.geneticProcess(
            generation)
        iBest = generation.chromosomes[0]
        clusters = clustering.printIBest(iBest)


    # ------------------output result-------------------#
    clustering.output_result(iBest, data, clusters)
