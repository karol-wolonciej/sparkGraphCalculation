import numpy as np
import matplotlib.pyplot as plt
from auxiliary import *
from keywords import *

s = 2

green = (0, 0.75, 0)
blue  = (0, 0, 0.75)


get_tile_x = { set1 : 1, set2 : 2 }
def get_tile_y(models_dict):
    k_list = get_k_list(models_dict)
    return dict(zip(k_list, range(len(k_list))))


def plot_dataset(set_):
    x_l = set_.map(lambda pair: pair[0]).collect()
    y_l = set_.map(lambda pair: pair[1]).collect()
    
    plt.figure()
    plt.scatter(x_l, y_l, s = 0.1)
    plt.show()
    plt.close()


def plot_points(points):
    x_l = np.array([p[0] for p in points])
    y_l = np.array([p[1] for p in points])
    
    plt.figure()
    plt.scatter(x_l, y_l, s = 0.1)
    plt.show()
    plt.close()


def plot_clusters(k, clusters):
    u_labels = np.unique(k)

    plt.figure()

    for (clusterId, label) in zip(list(range(k)), u_labels):
        x_l = np.array([p[0] for p in clusters[clusterId]])
        y_l = np.array([p[1] for p in clusters[clusterId]])
        
        plt.scatter(x_l, y_l, label = label)
    
    plt.show()
    plt.close()


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
    tile_number = get_tile_y(models_dict)[k] * 2 + get_tile_x[set_name]
    k_list = get_k_list(models_dict)
    x_tiles = 2
    y_tiles = len(k_list)
    plotClusters(clusters, y_tiles, x_tiles, tile_number, k)


def drawClustersFigure(models_dict, iniMode, maxIter, distMeasure):
    k_list = get_k_list(models_dict)
    for set_name in [set1, set2]:
        for k in k_list:
            drawClusterFigure(k, iniMode, maxIter, distMeasure, set_name, models_dict)
    plt.show()   


def createLineSubplot(dataLastKey, models_dict, t_number, x_label, y_label, label, *params):
    k_list = get_k_list(models_dict)
    fullDataKey = getStringKey(dataLastKey, *params)
    xy_list = compose(list, zip)(k_list, compose(list, map)(partial(round, ndigits=2), models_dict[fullDataKey]))
    x, y = getArraysFromTupleList(xy_list)
    plt.subplot(1, 2, t_number)
    plt.title(label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y)


def createLinePlot(dataLastKey, models_dict, x_label, y_label, label, *params):
    k_list = get_k_list(models_dict)
    fullDataKey = getStringKey(dataLastKey, *params)
    xy_list = compose(list, zip)(k_list, compose(list, map)(partial(round, ndigits=2), models_dict[fullDataKey]))
    x, y = getArraysFromTupleList(xy_list)
    plt.title(label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y)


def draw2DPlotsComparision(dataLastKey, models_dict, iniMode, maxIter, distMeasure):
    create2DPlot = partial(createLineSubplot, dataLastKey, models_dict)
    params = zip([1,2], [set1, set2])
    x_label = x_label_k_value
    y_label = dataLastKey
    for (t_number, set_name) in params:
        create2DPlot(t_number, x_label, y_label, set_name, iniMode, maxIter, distMeasure, set_name)
    plt.show()


def draw2DPlotKStest(dataLastKey, models_dict, iniMode, maxIter, distMeasure):
    x_label = 'k value'
    y_label = dataLastKey
    createLinePlot(dataLastKey, models_dict, x_label, y_label, 'set1 & set2 ks', iniMode, maxIter, distMeasure)
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