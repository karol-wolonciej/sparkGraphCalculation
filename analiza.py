import pickle
import matplotlib.pyplot as plt
import numpy as np
from yaml import compose

from auxiliary import *
from keywords import *
from functionalLib import compose
from plots import *

import json


model_dict_path = "/home/karol/pdd/duzeZadanie2/grafy/dictTestowy.pkl"

path_to_parameters = "/home/karol/pdd/duzeZadanie2/grafy/parameters.json"

parameters_file = open(path_to_parameters)
parameters = json.load(parameters_file)
parameters_file.close()



with open(model_dict_path, 'rb') as f:
    models_dict = pickle.load(f)




drawPlot = partial(operate_on_clustering_iniMode_maxIter_distMeasure, models_dict=models_dict)

drawMSEPlot = partial(draw2DPlotsComparision, mse)
drawSilhouettePlot = partial(draw2DPlotsComparision, silhouette)
drawKStestPlot = partial(draw2DPlotKStest, KS_test)


drawPlot(drawMSEPlot)
drawPlot(drawSilhouettePlot)
drawPlot(drawKStestPlot)

operate_on_clustering_iniMode_maxIter_distMeasure(drawClustersFigure, models_dict=models_dict)



drawOriginalSubsetsComparision(models_dict)
plotSet2D(models_dict[points_tuples_list][original_set], 'original set')



