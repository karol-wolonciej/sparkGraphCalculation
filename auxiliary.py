from unittest.mock import sentinel
import numpy as np
from keywords import *
from functools import partial
from functionalLib import compose



def getArraysFromTupleList(pointsList):
    pos = { 'x' : 0, 'y' : 1 }
    
    get_arr = lambda coordinate, points: np.array([point[coordinate] for point in points])
    
    params = [(pos['x'], pointsList), 
              (pos['y'], pointsList)]

    x, y = [get_arr(coordinate, point) for (coordinate, point) in params]

    return (x, y)


def initialize_model_dict(models_dict):
    parameters = models_dict[parametersDict]
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


def loop(f, *params):
    if len(params):
        for p in params[0]:
            loop(partial(f, p), *params[1:])
    else:
        f()


def getParamsLists(models_dict, *keywords):
    parameters = models_dict[parametersDict]
    return compose(tuple, map)(lambda key: parameters[key], keywords)


def operate_on_parameters(f, models_dict, *paramsKeys):
    paramsLists = getParamsLists(models_dict, *paramsKeys)
    foo = partial(f, models_dict)
    loop(foo, *paramsLists)


def operate_on_clustering_k_iniMode_maxIter_distMeasure_setName(f, models_dict):
    operate_on_parameters(f, 
                          models_dict,
                          k_set,
                          initializationMode, 
                          maxIterations, 
                          distanceMeasures,
                          set_name) 


def operate_on_clustering_k_iniMode_maxIter_distMeasure(f, models_dict):
    operate_on_parameters(f, 
                          models_dict,
                          k_set,
                          initializationMode, 
                          maxIterations, 
                          distanceMeasures) 



def operate_on_clustering_iniMode_maxIter_distMeasure(f, models_dict):
    operate_on_parameters(f, 
                          models_dict, 
                          initializationMode, 
                          maxIterations, 
                          distanceMeasures)


def operate_on_k_iniMode_maxIter_distMeasure(f, models_dict):
    operate_on_parameters(f, 
                          models_dict,
                          k_set, 
                          initializationMode, 
                          maxIterations, 
                          distanceMeasures)

def getStringKey(lastDataKey, *args):
    newDataKeyword = '_'.join([str(arg) for arg in args]) + '_' + lastDataKey #dodac do srodka
    return newDataKeyword