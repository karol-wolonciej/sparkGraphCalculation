import pickle
import matplotlib.pyplot as plt
import numpy as np
from yaml import compose

from auxiliary import *
from keywords import *
from functionalLib import compose

import json


model_dict_path = "/home/karol/pdd/duzeZadanie2/grafy/dictTestowy.pkl"

path_to_parameters = "/home/karol/pdd/duzeZadanie2/grafy/parameters.json"




parameters_file = open(path_to_parameters)
parameters = json.load(parameters_file)
parameters_file.close()



k_list = parameters['k_set']


get_tile_x = { set1 : 1, set2 : 2 }
get_tile_y = dict(zip(k_list, range(len(k_list))))


with open(model_dict_path, 'rb') as f:
    models_dict = pickle.load(f)


#Getting unique labels
n = 30
 
u_labels = np.unique(n)
 
#plotting the results:

# cluster_split = models_dict[n]['random'][10]['euclidean']['set2']['clustersSplit']

s = 2


green = (0, 0.75, 0)
blue  = (0, 0, 0.75)

def plotSet2D(pointsSet, title):
    x, y = getArraysFromTupleList(pointsSet)
    plt.scatter(x, y, s=s)
    plt.title(title)
    plt.show()


def plotSubplot(plt_def, s_x, s_y, t_number):
    plt.subplot(s_x, s_y, t_number)
    plt_def()


def plotCluster(x , y , s_x, s_y, t_number, label, s, k):
    plt.subplot(s_x, s_y, t_number)
    plt.scatter(x , y , label=label, s=s)
    
    if t_number in (1,2):
        plt.title(set1 if t_number == 1 else set2)

    if t_number % 2 == 1:
        plt.ylabel('k value: ' + str(k))


def plotClusters(clusters, s_x, s_y, t_number, k):
    clusterNumbers = clusters.keys()
    
    for i in range(len(clusterNumbers)):
        x, y = getArraysFromTupleList(clusters[i])
        plotCluster(x , y , s_x, s_y, t_number, label=i, s=0.1, k=k)


def drawClusterFigure(k, iniMode, maxIter, distMeasure, set_name, models_dict):
    paramDict = models_dict[k][iniMode][maxIter][distMeasure][set_name]
    clusters = paramDict[clustersSplit]
    tile_number = get_tile_y[k] * 2 + get_tile_x[set_name]
    x_tiles = 2
    y_tiles = len(k_list)
    plotClusters(clusters, y_tiles, x_tiles, tile_number, k)

def drawClustersFigure(models_dict, iniMode, maxIter, distMeasure):
    # print(iniMode)
    for set_name in [set1, set2]:
        for k in k_list:
            drawClusterFigure(k, iniMode, maxIter, distMeasure, set_name, models_dict)
    plt.show()   

def createLinePlot(iniMode, maxIter, distMeasure, set_name, dataLastKey, t_number, x_label, y_label, label, models_dict):
    fullDataKey = getStringKey(dataLastKey, iniMode, maxIter, distMeasure, set_name)
    xy_list = compose(list, zip)(k_list, compose(list, map)(partial(round, ndigits=2), models_dict[fullDataKey]))
    x, y = getArraysFromTupleList(xy_list)
    plt.subplot(1, 2, t_number)
    plt.title(label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y)


def draw2DPlotsComparision(dataLastKey, models_dict, iniMode, maxIter, distMeasure):
    create2DPlot = partial(createLinePlot, iniMode, maxIter, distMeasure, models_dict=models_dict)
    params = zip([1,2], [dataLastKey]*2, [set1, set2])
    x_label = 'k value'
    y_label = dataLastKey
    for (t_number, key, set_name) in params:
        create2DPlot(set_name=set_name, dataLastKey=key, t_number=t_number, x_label=x_label, y_label=y_label, label=set_name)
    plt.show()

def draw2DPlotsComparisionKStest(dataLastKey, models_dict, iniMode, maxIter, distMeasure):
    create2DPlot = partial(createLinePlot, iniMode, maxIter, distMeasure, models_dict=models_dict)
    params = zip([1,2], [dataLastKey]*2)
    x_label = 'k value'
    y_label = dataLastKey
    for (t_number, key) in params:
        create2DPlot(set_name=set_name, dataLastKey=key, t_number=t_number, x_label=x_label, y_label=y_label, label=set_name)
    plt.show()


def drawOriginalSubsetsComparision(models_dict):
    sets_dict = models_dict[points_tuples_list]
    for (t_number, pointSets, color, set_name) in zip(range(1,3), (sets_dict[set1], sets_dict[set2]), (blue, green), [set1, set2]):
        x, y = getArraysFromTupleList(pointSets)
        draw = lambda: plt.scatter(x, y, color=color, s=s)
        plotSubplot(draw, 1, 2, t_number)
        title = 'original ' + set_name
        plt.title(title)
    plt.show()

print(models_dict)

drawPlotsComparision = partial(operate_on_clustering_iniMode_maxIter_distMeasure, models_dict=models_dict)

drawMSEPlot = partial(draw2DPlotsComparision, mse)
drawSilhouettePlot = partial(draw2DPlotsComparision, silhouette)
drawKStestPlot = partial(draw2DPlotsComparision, KS_test)


# drawPlotsComparision(drawMSEPlot)
# drawPlotsComparision(drawSilhouettePlot)
drawPlotsComparision(drawKStestPlot)

# operate_on_clustering_iniMode_maxIter_distMeasure(drawClustersFigure, models_dict=models_dict)



# drawOriginalSubsetsComparision(models_dict)
# plotSet2D(models_dict[points_tuples_list][original_set], 'original set')



