from __future__ import division
from distutils import core

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.feature import *
from pyspark.ml import Pipeline

import matplotlib.pyplot as plt
import matplotlib
import io
import numpy as np
import json
from math import sqrt
from functools import partial


import functionalLib
import plots
from models_dict_calculations import *




sc = SparkContext()
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("ERROR")


f = open('/home/karol/pdd/duzeZadanie2/grafy/parameters.json')
parameters = json.load(f)
f.close()


# todo: wczytuj z json
initSteps = parameters['initSteps']
tol = parameters['tol']
seed = parameters['seed']
ratioOfInputPoints = parameters['ratioOfInputPoints']

models_dict = {}
models_dict["parameters"] = parameters


set1 = 'set1'
set2 = 'set2'



df = spark.read.text('/home/karol/pdd/duzeZadanie2/grafy/birch3.txt')
td = df.rdd #transformer df to rdd
tr_data = td.map(lambda line: line[0].split()).map(lambda line: Row(X=int(line[0]), Y=int(line[1]))).toDF().sort('X', 'Y').sample(withReplacement=False, fraction=ratioOfInputPoints).cache()

# tr_data.describe().show()
# tr_data.show(5)

half = 0.5
set1_points = tr_data.sample(withReplacement=False, fraction=half).cache()
set2_points = tr_data.subtract(set1_points).cache()


assemble = VectorAssembler(inputCols=['X', 'Y'], outputCol = 'before_scaling_features')
scaler = StandardScaler(inputCol='before_scaling_features', outputCol='features')
data_transformation_pipeline = Pipeline(stages= [assemble, scaler]) #wyrzucilem scaler

transformed_data_model = data_transformation_pipeline.fit(tr_data)
transformed_data_set1 = transformed_data_model.transform(set1_points).cache()
transformed_data_set2 = transformed_data_model.transform(set2_points).cache()


points_sets = { 'set1' : transformed_data_set1, 'set2' : transformed_data_set2 }

models_dict['points_sets'] = points_sets



initialize_model_dict(parameters, models_dict)

# print(parameters)
# print(models_dict)

operate_on_parameters_and_models(createKMeansObjects, parameters, models_dict)
operate_on_parameters_and_models(fitModels, parameters, models_dict)
operate_on_parameters_and_models(calculatePointsForTest, parameters, models_dict)
operate_on_parameters_and_models(calculateMeanSquareError, parameters, models_dict)
operate_on_parameters_and_models(calculateSihouette, parameters, models_dict)
operate_on_parameters_and_models(calculateClustersSplit, parameters, models_dict)
operate_on_parameters_and_models(printMSE, parameters, models_dict)
operate_on_parameters_and_models(printSilhouette, parameters, models_dict)