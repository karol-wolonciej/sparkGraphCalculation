#!/usr/bin/python

from __future__ import division
from distutils import core

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.feature import *
from pyspark.ml import Pipeline


import io
import numpy as np
import json
from math import sqrt
from functools import partial
import pickle
import sys

# from plots import *
from models_dict_calculations import *
from keywords import *
from KStest import ks2d2s, ks2d2s_2d_points
from auxiliary import *


sc = SparkContext()
spark = SparkSession(sc)

path_to_parameters = sys.argv[1]

parameters_file = open(path_to_parameters)
parameters = json.load(parameters_file)
parameters_file.close()

models_dict = {}
models_dict[parametersDict] = parameters
models_dict[sparkContext] = sc

ratioOfInputPoints = parameters[ratioOfInputPoints]
dataset_path = parameters[dataset_path_key]
logLevel = parameters[logLevel_key]
resultDictionaryPath = parameters[resultDictionaryPath_key]

spark.sparkContext.setLogLevel(logLevel)

df = spark.read.text(dataset_path)

td = df.rdd

int_float = compose(int, float)

tr_data = td.map(lambda line: line[0].split()) \
            .map(lambda line: Row(X=int_float(line[0]), Y=int_float(line[1]))) \
            .toDF().sort(X, Y) \
            .sample(withReplacement=False, fraction=ratioOfInputPoints) \
            .cache()

half = 0.5
set1_points = tr_data.sample(withReplacement=False, fraction=half).cache()
set2_points = tr_data.subtract(set1_points).cache()

models_dict[sets_counts] = { set1 : set1_points.count(), set2 : set2_points.count() }

assemble = VectorAssembler(inputCols=[X,  Y], outputCol = 'before_scaling_features')
scaler = StandardScaler(inputCol='before_scaling_features', outputCol='features')
data_transformation_pipeline = Pipeline(stages= [assemble, scaler])

transformed_data_model = data_transformation_pipeline.fit(tr_data)
transformed_data_set1 = transformed_data_model.transform(set1_points).cache()
transformed_data_set2 = transformed_data_model.transform(set2_points).cache()

models_dict[originalSetSummary] = tr_data.summary().toPandas().to_string()
models_dict[originalSet1Summary] = set1_points.summary().toPandas().to_string()
models_dict[originalSet2Summary] = set2_points.summary().toPandas().to_string()

models_dict[transformedSet1Summary] = transformed_data_set1.summary().toPandas().to_string()
models_dict[transformedSet2Summary] = transformed_data_set2.summary().toPandas().to_string()

set_point_to_tuples = lambda pointsDF: [tuple([round(val, 3) for val in denseVector[0]]) for denseVector in pointsDF.select('features').collect()]

set1PointsTuplesForTest = set_point_to_tuples(transformed_data_set1)
set2PointsTuplesForTest = set_point_to_tuples(transformed_data_set2)

oryginalSetPointsTuples = set1PointsTuplesForTest + set2PointsTuplesForTest

models_dict[original_KS_test] = ks2d2s_2d_points(set1PointsTuplesForTest, set2PointsTuplesForTest)
models_dict[points_sets] = { set1 : transformed_data_set1, 
                             set2 : transformed_data_set2 }

models_dict[points_tuples_list] = { set1 : set1PointsTuplesForTest, 
                                    set2 : set2PointsTuplesForTest,
                                    original_set : oryginalSetPointsTuples }



initialize_model_dict(models_dict)

operate_on_k_iniMode_maxIter_distMeasure_set_name = partial(operate_on_clustering_k_iniMode_maxIter_distMeasure_setName, models_dict=models_dict)
operate_on_k_iniMode_maxIter_distMeasure = partial(operate_on_clustering_k_iniMode_maxIter_distMeasure, models_dict=models_dict)
operate_on_iniMode_maxIter_distMeasure = partial(operate_on_clustering_iniMode_maxIter_distMeasure, models_dict=models_dict)
operate_on_iniMode_maxIter_distMeasute_set_name = partial(operate_on_clustering_iniMode_maxIter_distMeasure_set_name, models_dict=models_dict)

parametersKeywords = [k_set, initializationMode, maxIterations, distanceMeasures]
parametersSetsKeywords = parametersKeywords + [set_name]

gatherMSE = partial(gatherDataForEveryK, mse)
gatherSilhouette = partial(gatherDataForEveryK, silhouette)
gatherKStest = partial(gatherDataForEveryK, KS_test)


operate_on_k_iniMode_maxIter_distMeasure_set_name(createKMeansObjects)
operate_on_k_iniMode_maxIter_distMeasure_set_name(fitModels)
operate_on_k_iniMode_maxIter_distMeasure_set_name(calculatePointsForTest)
operate_on_k_iniMode_maxIter_distMeasure_set_name(calculateMeanSquareError)
operate_on_k_iniMode_maxIter_distMeasure_set_name(calculateSihouette)
operate_on_k_iniMode_maxIter_distMeasure_set_name(calculateClustersSplit)
operate_on_k_iniMode_maxIter_distMeasure(calculateKStest)
operate_on_k_iniMode_maxIter_distMeasure_set_name(gatherMSE)
operate_on_k_iniMode_maxIter_distMeasure_set_name(gatherSilhouette)
operate_on_k_iniMode_maxIter_distMeasure(gatherKStest)

operate_on_iniMode_maxIter_distMeasute_set_name(add_k_to_mse)
operate_on_iniMode_maxIter_distMeasute_set_name(add_k_to_silhoutte)
operate_on_iniMode_maxIter_distMeasure(add_k_to_ks_test)

operate_on_k_iniMode_maxIter_distMeasure_set_name(deleteModelsAndDataframes)

del models_dict[sparkContext]
del models_dict[points_sets]

with open(resultDictionaryPath, 'wb') as f:
    pickle.dump(models_dict, f)