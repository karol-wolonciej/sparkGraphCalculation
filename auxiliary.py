from unittest.mock import sentinel
import numpy as np
from keywords import *
from functools import partial
from functionalLib import compose
from scipy import spatial
from scipy.spatial import distance


x_two_columns_plot_size = 12
single_subplot_y_size = 6
x_plot_size = 6
y_plot_size = 6


def setNothing():
    pass


cosineMeasure = lambda point1, point2: 1 - spatial.distance.cosine(list(point1), list(point2))
euclideanMeasure = distance.euclidean


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


def operate_on_clustering_iniMode_maxIter_distMeasure_set_name(f, models_dict):
    operate_on_parameters(f, 
                          models_dict, 
                          initializationMode, 
                          maxIterations, 
                          distanceMeasures,
                          set_name)


def operate_on_k_iniMode_maxIter_distMeasure(f, models_dict):
    operate_on_parameters(f, 
                          models_dict,
                          k_set, 
                          initializationMode, 
                          maxIterations, 
                          distanceMeasures)


def getStringKey(lastDataKey, *args):
    newDataKeyword = '_'.join([str(arg) for arg in args]) + '_' + lastDataKey
    return newDataKeyword


def get_path_to_partial_pdf(models_dict, *args):
    folderPath = get_partial_PDFs_path(models_dict)
    pdf_name = '_'.join([str(arg) for arg in args] + [pdf_extension])
    return folderPath + pdf_name


def get_k_list(models_dict):
    return models_dict[parametersDict][k_set]


def get_partial_PDFs_path(models_dict):
    return models_dict[parametersDict]['partialPDFsPath']


def addElementToDict(key, elem, dict):
    if key in dict.keys():
        dict[key].append(elem)
    else:
        dict[key] = [elem]


def getParamDict(models_dict, *params):
    getNextElement = lambda key, dict: dict[key]
    functions = compose(tuple, map)(lambda keyword: partial(getNextElement, keyword), params[::-1])
    param_dict = compose(*functions)(models_dict)
    return param_dict