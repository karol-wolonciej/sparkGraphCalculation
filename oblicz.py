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

from plots import *
from models_dict_calculations import *
from keywords import *
from KStest import ks2d2s, ks2d2s_2d_points



sc = SparkContext()
spark = SparkSession(sc)

path_to_parameters = sys.argv[1]

model_dict_file = open(path_to_parameters)
parameters = json.load(model_dict_file)
model_dict_file.close()


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
tr_data = td.map(lambda line: line[0].split()) \
            .map(lambda line: Row(X=int(line[0]), Y=int(line[1]))) \
            .toDF().sort(X, Y) \
            .sample(withReplacement=False, fraction=ratioOfInputPoints) \
            .cache()

half = 0.5
set1_points = tr_data.sample(withReplacement=False, fraction=half).cache()
set2_points = tr_data.subtract(set1_points).cache()

assemble = VectorAssembler(inputCols=[X,  Y], outputCol = 'before_scaling_features')
scaler = StandardScaler(inputCol='before_scaling_features', outputCol='features')
data_transformation_pipeline = Pipeline(stages= [assemble, scaler])

transformed_data_model = data_transformation_pipeline.fit(tr_data)
transformed_data_set1 = transformed_data_model.transform(set1_points).cache()
transformed_data_set2 = transformed_data_model.transform(set2_points).cache()

set_point_to_tuples = lambda pointsDF: [tuple([val for val in denseVector[0]]) for denseVector in pointsDF.select('features').collect()]

set1PointsTuplesForTest = set_point_to_tuples(transformed_data_set1)
set2PointsTuplesForTest = set_point_to_tuples(transformed_data_set2)

models_dict[original_KS_test] = ks2d2s_2d_points(set1PointsTuplesForTest, set2PointsTuplesForTest)
models_dict[points_sets] = { set1 : transformed_data_set1, set2 : transformed_data_set2 }


initialize_model_dict(models_dict)

operateParametersWithSets = partial(operate_on_parameters_and_sets, models_dict=models_dict)
operateParametersOnly = partial(operate_on_parameters, models_dict=models_dict)

operateParametersWithSets(createKMeansObjects)
operateParametersWithSets(fitModels)
operateParametersWithSets(calculatePointsForTest)
operateParametersWithSets(calculateMeanSquareError)
operateParametersWithSets(calculateSihouette)
operateParametersWithSets(calculateClustersSplit)
operateParametersOnly(calculateKStest)
operateParametersWithSets(deleteModelsAndDataframes)

del models_dict[sparkContext]
del models_dict[points_sets]


with open(resultDictionaryPath, 'wb') as f:
    pickle.dump(models_dict, f)