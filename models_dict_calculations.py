from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.feature import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark import Row

import matplotlib.pyplot as plt
import matplotlib
import io
import numpy as np
import json
from math import sqrt
from functools import partial

from KStest import ks2d2s, ks2d2s_2d_points
from keywords import *





# create KMeans objects
def createKMeansObjects(k, iniMode, maxIter, distMeasure, set_name, models_dict):
    parameters = models_dict[parametersDict]
    models_dict[k][iniMode][maxIter][distMeasure][set_name][kmean_instance] = KMeans(featuresCol='features',
                                                                                     predictionCol='prediction',
                                                                                     k=k,
                                                                                     initMode=iniMode,
                                                                                     initSteps=parameters['initSteps'],
                                                                                     tol=parameters['tol'],
                                                                                     maxIter=maxIter,
                                                                                     seed=parameters['seed'],
                                                                                     distanceMeasure=distMeasure
                                                                                    )


# fit models
def fitModels(k, iniMode, maxIter, distMeasure, set_name, models_dict):
    paramDict = models_dict[k][iniMode][maxIter][distMeasure][set_name]
    paramDict[kmean_model] = paramDict[kmean_instance].fit(models_dict[points_sets][set_name])


# points for test
def calculatePointsForTest(k, iniMode, maxIter, distMeasure, set_name, models_dict):
    paramDict = models_dict[k][iniMode][maxIter][distMeasure][set_name]
    clusterCenters = paramDict[kmean_model].clusterCenters()
    clusterSizes = paramDict[kmean_model].summary.clusterSizes
    paramDict[points_for_test] = [(center_x * cluster_size, center_y * cluster_size) for ((center_x, center_y), cluster_size) in zip(clusterCenters, clusterSizes)]


# def plotPointsSets(k, iniMode, maxIter, distMeasure, set_name, models_dict):
#     plot_points(models_dict[k][iniMode][maxIter][distMeasure][set_name][points_for_test])


# todo przeksztalc na operacje na df
def calculateMeanSquareError(k, iniMode, maxIter, distMeasure, set_name, models_dict):
    param_dict = models_dict[k][iniMode][maxIter][distMeasure][set_name]
    kmeans_model = param_dict[kmean_model]

    pointsVectors = [row[0] for row in models_dict[points_sets][set_name].select('features').collect()]

    clustersCenters = [tuple(center) for center in kmeans_model.clusterCenters()]


    getDist = lambda p1, p2: sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # kmeans_model.predict(pointVector)
    MSE = sum([sqrt(getDist(pointVector, clustersCenters[kmeans_model.predict(pointVector)])) for pointVector in pointsVectors]) / len(pointsVectors)

    param_dict[mse] = MSE


def calculateSihouette(k, iniMode, maxIter, distMeasure, set_name, models_dict):
    param_dict = models_dict[k][iniMode][maxIter][distMeasure][set_name]
    kmeans_model = param_dict[kmean_model]
    evaluator = ClusteringEvaluator()
    points = models_dict[points_sets][set_name]
    predictions = kmeans_model.transform(points)
    param_dict[silhouette] = evaluator.evaluate(predictions)


def calculateClustersSplit(k, iniMode, maxIter, distMeasure, set_name, models_dict):
    param_dict = models_dict[k][iniMode][maxIter][distMeasure][set_name]
    kmeans_model = param_dict[kmean_model]
    param_dict[clustersSplit] = {}

    for i in range(k):
        param_dict[clustersSplit][i] = []

    points = [row[0] for row in  models_dict[points_sets][set_name].select('features').collect()]
    clustersCenters = kmeans_model.clusterCenters()

    for pointVector in points:
        clusterCenter = kmeans_model.predict(pointVector)
        param_dict[clustersSplit][clusterCenter].append(tuple([round(val, 2) for val in pointVector.toArray()]))


def printLastParam(paramKey, k, iniMode, maxIter, distMeasure, set_name, models_dict):
    print(models_dict[k][iniMode][maxIter][distMeasure][set_name][paramKey])


printPoints = partial(printLastParam, points_for_test)
printMSE = partial(printLastParam, mse)
printSilhouette = partial(printLastParam, silhouette)

# tak musi byc chyba wiec trzeba zrobic collect ale moze mozna by wziasc tylko jakas czesc punktow a nie wszystkie
# def plotClusters(k, iniMode, maxIter, distMeasure, set_name, models_dict):
#     clusters = models_dict[k][iniMode][maxIter][distMeasure][set_name][clustersSplit]
#     plot_clusters(k, clusters)


def deleteModelsAndDataframes(k, iniMode, maxIter, distMeasure, set_name, models_dict):
    del models_dict[k][iniMode][maxIter][distMeasure][set_name][kmean_model]
    del models_dict[k][iniMode][maxIter][distMeasure][set_name][kmean_instance]



def calculateKStest(k, iniMode, maxIter, distMeasure, models_dict):
    paramDict =  models_dict[k][iniMode][maxIter][distMeasure]

    getPoints = lambda set_name: paramDict[set_name][points_for_test]

    points_set1 = getPoints(set1)
    points_set2 = getPoints(set2)
    
    paramDict[KS_test] = ks2d2s_2d_points(points_set1, points_set2)


def printKStest(k, iniMode, maxIter, distMeasure, models_dict):
    print(models_dict[k][iniMode][maxIter][distMeasure][KS_test])
