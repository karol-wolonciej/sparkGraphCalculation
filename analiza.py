from __future__ import division
from distutils import core

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.feature import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark import Row
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
# from pyspark import SparkConf

from numpy import array
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib
import io
import numpy as np
from numpy import random
from scipy.spatial.distance import pdist, cdist
from scipy.stats import kstwobign, pearsonr
from scipy.stats import genextreme

# from pyspark.mllib.clustering import KMeans, KMeansModel



import functools
from functools import partial


from pyspark.ml.evaluation import ClusteringEvaluator

import json


sc = SparkContext()
spark = SparkSession(sc)

spark.sparkContext.setLogLevel("ERROR")



# spark.zos.master.app.alwaysScheduleApps=True

def plot_dataset(set_):
    x_l = set_.map(lambda pair: pair[0]).collect()
    y_l = set_.map(lambda pair: pair[1]).collect()
    
    plt.figure()
    plt.scatter(x_l, y_l, s = 0.1)
    plt.show()
    # z.show(plt)
    plt.close()


def plot_points(points):
    x_l = np.array([p[0] for p in points])
    y_l = np.array([p[1] for p in points])
    
    plt.figure()
    plt.scatter(x_l, y_l, s = 0.1)
    plt.show()
    z.show(plt)
    plt.close()


def plot_clusters(k, clusters):
    u_labels = np.unique(k)

    plt.figure()


    for (clusterId, label) in zip(list(range(k)), u_labels):
        x_l = np.array([p[0] for p in clusters[clusterId]])
        y_l = np.array([p[1] for p in clusters[clusterId]])
        
        plt.scatter(x_l, y_l, label = label)
    
    plt.show()
    z.show(plt)
    plt.close()
    

def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))
    

def compose(*fs):
    return functools.reduce(compose2, fs)


f = open('/home/karol/pdd/duzeZadanie2/grafy/parameters.json')
parameters = json.load(f)
f.close()


for i in parameters:
    print(parameters[i])

# todo: wczytuj z json
initSteps = parameters['initSteps']
tol=parameters['tol']
seed = parameters['seed']
ratioOfInputPoints = parameters['ratioOfInputPoints']


models_dict = {}
kmean_instance = 'kmean_instance'
kmean_model = 'kmean_model'
points_for_test = 'points_for_test'
KS_test = 'KS_test'
mse = 'mse'
silhouette = 'silhouette'
clustersSplit = 'clustersSplit'


set1 = 'set1'
set2 = 'set2'

print(models_dict)


df = spark.read.text('/home/karol/pdd/duzeZadanie2/grafy/birch3.txt')
td = df.rdd #transformer df to rdd
tr_data = td.map(lambda line: line[0].split()).map(lambda line: Row(X=int(line[0]), Y=int(line[1]))).toDF().sort('X', 'Y').sample(withReplacement=False, fraction=ratioOfInputPoints).cache()
# tr_data.cache()


tr_data.describe().show()

tr_data.show(5)

half = 0.5

set1_points = tr_data.sample(withReplacement=False, fraction=half).cache()
# set1_points.cache()

set2_points = tr_data.subtract(set1_points).cache()
# set2_points.cache()



assemble = VectorAssembler(inputCols=['X', 'Y'], outputCol = 'before_scaling_features')
scaler = StandardScaler(inputCol='before_scaling_features', outputCol='features')

data_transformation_pipeline = Pipeline(stages= [assemble, scaler]) #wyrzucilem scaler

transformed_data_model = data_transformation_pipeline.fit(tr_data)

transformed_data_set1 = transformed_data_model.transform(set1_points).cache()
transformed_data_set2 = transformed_data_model.transform(set2_points).cache()

# transformed_data_set1.cache()
# transformed_data_set2.cache()



points_sets = { 'set1' : transformed_data_set1, 'set2' : transformed_data_set2 }



def initialize_model_dict(models_dict=models_dict):
    for k in parameters['k_set']:
        models_dict[k] = {}
        
        for iniMode in parameters['initializationMode']:
            models_dict[k][iniMode] = {}
            
            for maxIter in parameters['maxIterations']:
                models_dict[k][iniMode][maxIter] = {}
                
                for distMeasure in parameters['distanceMeasures']:
                    models_dict[k][iniMode][maxIter][distMeasure] = {}
                    
                    for set_name in ['set1', 'set2']:
                        models_dict[k][iniMode][maxIter][distMeasure][set_name] = {}


# foo bierze wszystkie parametry i byc moze wiecej
def operate_dictionary(f_all, g):
    for k in parameters['k_set']:
        for iniMode in parameters['initializationMode']:
            for maxIter in parameters['maxIterations']:
                for distMeasure in parameters['distanceMeasures']:
                    partial_f_all = partial(f_all, k, iniMode, maxIter, distMeasure)
                    g(partial_f_all)


# foo pierwotnie bralo wszystko ale tu bierze juz tylko set_name 
def operate_on_parameters(foo):
    operate_dictionary(foo, lambda f: f())
    

def operate_on_models(foo):
    for set_name in ['set1', 'set2']:
        foo(set_name)
        
def operate_on_parameters_and_models(foo):
    operate_dictionary(foo, operate_on_models)



# create KMeans objects
def createKMeansObjects(k, iniMode, maxIter, distMeasure, set_name):
    models_dict[k][iniMode][maxIter][distMeasure][set_name][kmean_instance] = KMeans(featuresCol='features',
                                                                                     predictionCol='prediction',
                                                                                     k=k,
                                                                                     initMode=iniMode,
                                                                                     initSteps=initSteps,
                                                                                     tol=tol,
                                                                                     maxIter=maxIter,
                                                                                     seed=seed,
                                                                                     distanceMeasure=distMeasure
                                                                                    )


# fit models
def fitModels(k, iniMode, maxIter, distMeasure, set_name):
    paramDict = models_dict[k][iniMode][maxIter][distMeasure][set_name]
    paramDict[kmean_model] = paramDict[kmean_instance].fit(points_sets[set_name])


# points for test
def calculatePointsForTest(k, iniMode, maxIter, distMeasure, set_name):
    paramDict = models_dict[k][iniMode][maxIter][distMeasure][set_name]
    clusterCenters = paramDict[kmean_model].clusterCenters()
    clusterSizes = paramDict[kmean_model].summary.clusterSizes
    paramDict[points_for_test] = [(center_x * cluster_size, center_y * cluster_size) for ((center_x, center_y), cluster_size) in zip(clusterCenters, clusterSizes)]
    

def plotPointsSets(k, iniMode, maxIter, distMeasure, set_name):
    plot_points(models_dict[k][iniMode][maxIter][distMeasure][set_name][points_for_test])
    

# todo przeksztalc na operacje na df
def calculateMeanSquareError(k, iniMode, maxIter, distMeasure, set_name):
    param_dict = models_dict[k][iniMode][maxIter][distMeasure][set_name]
    kmeans_model = param_dict[kmean_model]
    
    pointsVectors = [row[0] for row in points_sets[set_name].select('features').collect()]
    
    clustersCenters = [tuple(center) for center in kmeans_model.clusterCenters()]
    
    
    getDist = lambda p1, p2: sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    
    # kmeans_model.predict(pointVector)
    MSE = sum([sqrt(getDist(pointVector, clustersCenters[kmeans_model.predict(pointVector)])) for pointVector in pointsVectors]) / len(pointsVectors)
    
    param_dict[mse] = MSE
    

def calculateSihouette(k, iniMode, maxIter, distMeasure, set_name):
    param_dict = models_dict[k][iniMode][maxIter][distMeasure][set_name]
    kmeans_model = param_dict[kmean_model]
    evaluator = ClusteringEvaluator()
    points = points_sets[set_name]
    predictions = kmeans_model.transform(points)
    param_dict[silhouette] = evaluator.evaluate(predictions)
    

# todo przeksztalc na operacje na df
def calculateClustersSplit(k, iniMode, maxIter, distMeasure, set_name):
    param_dict = models_dict[k][iniMode][maxIter][distMeasure][set_name]
    kmeans_model = param_dict[kmean_model]
    param_dict[clustersSplit] = {}

    for i in range(k):
        param_dict[clustersSplit][i] = []
    
    points = [row[0] for row in  points_sets[set_name].select("features").collect()]
    clustersCenters = kmeans_model.clusterCenters()
    
    for pointVector in points:
        clusterCenter = kmeans_model.predict(pointVector)
        param_dict[clustersSplit][clusterCenter].append(pointVector)


def printLastParam(paramKey, k, iniMode, maxIter, distMeasure, set_name):
    print(models_dict[k][iniMode][maxIter][distMeasure][set_name][paramKey])
    

printPoints = partial(printLastParam, points_for_test)
printMSE = partial(printLastParam, mse)
printSilhouette = partial(printLastParam, silhouette)

# tak musi byc chyba wiec trzeba zrobic collect ale moze mozna by wziasc tylko jakas czesc punktow a nie wszystkie
def plotClusters(k, iniMode, maxIter, distMeasure, set_name):
    clusters = models_dict[k][iniMode][maxIter][distMeasure][set_name][clustersSplit]
    plot_clusters(k, clusters)


def calculateKStest(k, iniMode, maxIter, distMeasure):
    paramDict =  models_dict[k][iniMode][maxIter][distMeasure]
    
    getPoints = lambda set_name: paramDict[set_name][points_for_test]
    
    points_set1 = getPoints(set1)
    points_set2 = getPoints(set2)
    
    pos = { 'x' : 0, 'y' : 1 }
    
    get_arr = lambda coordinate, points: np.array([point[coordinate] for point in points])
    
    params = [(pos['x'], points_set1), 
              (pos['y'], points_set1),
              (pos['x'], points_set2),
              (pos['y'], points_set2)]

    x1, y1, x2, y2 = [get_arr(coordinate, point) for (coordinate, point) in params]
    
    paramDict[KS_test] = ks2d2s(x1, y1, x2, y2)
    
def printKStest(k, iniMode, maxIter, distMeasure):
    print(models_dict[k][iniMode][maxIter][distMeasure][KS_test])



initialize_model_dict(models_dict)

operate_on_parameters_and_models(createKMeansObjects)
operate_on_parameters_and_models(fitModels)
operate_on_parameters_and_models(calculatePointsForTest)
operate_on_parameters_and_models(calculateMeanSquareError)
operate_on_parameters_and_models(calculateSihouette)
operate_on_parameters_and_models(calculateClustersSplit)


# operate_on_parameters_and_models(plotClusters)



# operate_on_parameters_and_models(printPoints)
operate_on_parameters_and_models(printMSE)
operate_on_parameters_and_models(printSilhouette)