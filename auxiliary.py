import numpy as np
from keywords import *
from functools import partial


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


# foo bierze wszystkie parametry i byc moze wiecej
def operate_dictionary(f_all, g, models_dict):
    parameters = models_dict[parametersDict]
    for iniMode in parameters['initializationMode']:
        for maxIter in parameters['maxIterations']:
            for distMeasure in parameters['distanceMeasures']:
                for k in parameters['k_set']:
                    partial_f_all = partial(f_all, k, iniMode, maxIter, distMeasure)
                    g(partial_f_all, models_dict)


# foo pierwotnie bralo wszystko ale tu bierze juz tylko set_name
def operate_on_parameters(foo, models_dict):
    operate_dictionary(foo, lambda f, md: f(md), models_dict)


def operate_on_models(foo, models_dict):
    for set_name in ['set1', 'set2']:
        foo(set_name, models_dict)


def operate_on_parameters_and_sets(foo, models_dict):
    operate_dictionary(foo, operate_on_models, models_dict)