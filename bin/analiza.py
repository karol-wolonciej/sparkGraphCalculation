#!/usr/bin/python

import pickle
from statistics import mode
import matplotlib.pyplot as plt
import numpy as np
from prometheus_client import Summary
from yaml import compose

from auxiliary import *
from keywords import *
from functionalLib import compose
from plots import *
from textPDFs import *
from raport_generation import *
import sys


import json



path_to_parameters = sys.argv[1]

parameters_file = open(path_to_parameters)
parameters = json.load(parameters_file)
parameters_file.close()

model_dict_path = parameters["resultDictionaryPath"]
partial_PDFs_path = parameters["partialPDFsPath"]


with open(model_dict_path, 'rb') as f:
    models_dict = pickle.load(f)


operate_on_iniMode_maxIter_distMeasure = partial(operate_on_clustering_iniMode_maxIter_distMeasure, models_dict=models_dict)
operate_on_iniMode_maxIter_distMeasute_set_name = partial(operate_on_clustering_iniMode_maxIter_distMeasure_set_name, models_dict=models_dict)


drawPlot = partial(operate_on_clustering_iniMode_maxIter_distMeasure, models_dict=models_dict)

drawMSEPlot = partial(draw2DLinePlotsComparisionForSetsGathered, mse)
drawSilhouettePlot = partial(draw2DLinePlotsComparisionForSetsGathered, silhouette)


createRaportHeader(models_dict)
operate_on_iniMode_maxIter_distMeasure(createSubHeaders)

writeUsualLine(txt='Clustering comparision', path=parameters[partialPDFsPath] + clusteringComparisionHeaderPdf)
writeUsualLine(txt='New points comparision', path=parameters[partialPDFsPath] + pointsToTestComparisionHeaderPdf)
writeUsualLine(txt='Original two sets KS: ' + str(models_dict[original_KS_test]), path=parameters[partialPDFsPath] + originalSets_KS_PDF)
 
createSummary(originalSetSummary, models_dict)
createSummary(originalSet1Summary, models_dict)
createSummary(originalSet2Summary, models_dict)
createSummary(transformedSet1Summary, models_dict)
createSummary(transformedSet2Summary, models_dict)


drawPlot(drawMSEPlot)
drawPlot(drawSilhouettePlot)
drawPlot(draw2DPlotKStest)

operate_on_clustering_iniMode_maxIter_distMeasure(drawClustersComparisionFigure, models_dict=models_dict)
operate_on_clustering_iniMode_maxIter_distMeasure(drawNewPointsComparisionFigure, models_dict=models_dict)

drawOriginalSubsetsComparision(models_dict)
plotSet2D(models_dict, models_dict[points_tuples_list][original_set], 'original set')

partialPDFsPaths = []
getRaportHeaderPaths(partialPDFsPaths, models_dict)
getPaths = partial(getPathsForClusteringAnalysis, partialPDFsPaths)
operate_on_clustering_iniMode_maxIter_distMeasure(getPaths, models_dict=models_dict)

createReport(models_dict, partialPDFsPaths)