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




def plotSet2D(models_dict, pointsSet, title):
    x, y = getArraysFromTupleList(pointsSet)
    plt.figure(figsize=(x_plot_size, y_plot_size))
    plt.scatter(x, y, s=s)
    plt.title(title)
    pdf_path = get_path_to_partial_pdf(models_dict, 'original_set')
    plt.savefig(pdf_path)


def plotSubplot(plt_def, s_x, s_y, t_number):
    plt.subplot(s_x, s_y, t_number)
    plt_def()


def set_k_subfigure_title(k, t_number):
    if t_number in (1,2):
        plt.title(set1 if t_number == 1 else set2)

    if t_number % 2 == 1:
        plt.ylabel('k value: ' + str(k))


def getTileParams(k, set_name, models_dict):
    k_list = get_k_list(models_dict)
    x_tiles = 2
    y_tiles = len(k_list)
    tile_number = get_tile_y(models_dict)[k] * 2 + get_tile_x[set_name]
    return (x_tiles, y_tiles, tile_number)


def plotCluster(x , y , s_x, s_y, t_number, label, s, k):
    plt.subplot(s_x, s_y, t_number)
    plt.scatter(x , y , label=label, s=s)
    set_k_subfigure_title(k, t_number)


def plotClusters(clusters, clusterCenters, s_x, s_y, t_number, k):
    clusterNumbers = clusters.keys()
    
    for i in range(len(clusterNumbers)):
        x, y = getArraysFromTupleList(clusters[i])
        plotCluster(x , y , s_x, s_y, t_number, label=i, s=0.1, k=k)
    x_cluster_centers, y_cluster_centers = getArraysFromTupleList(clusterCenters)
    plt.scatter(x_cluster_centers , y_cluster_centers , s = 10, color = 'k')


def drawClusterFigure(k, iniMode, maxIter, distMeasure, set_name, models_dict):
    paramDict = getParamDict(models_dict, k, iniMode, maxIter, distMeasure, set_name)
    clusters = paramDict[clustersSplit]
    centers = paramDict[clusterCenters]
    x_tiles, y_tiles, tile_number = getTileParams(k, set_name, models_dict)
    plotClusters(clusters, centers, y_tiles, x_tiles, tile_number, k)


def drawPointsForTestsFigure(k, iniMode, maxIter, distMeasure, set_name, models_dict):
    paramDict = getParamDict(models_dict, k, iniMode, maxIter, distMeasure, set_name)
    points = paramDict[points_for_test]
    x_tiles, y_tiles, tile_number = getTileParams(k, set_name, models_dict)
    paramsFuncSubplot = partial(plt.subplot, y_tiles, x_tiles, tile_number)
    paramsFuncLabel = partial(set_k_subfigure_title, k, tile_number) 
    scatter = partial(plt.scatter, label='k value: ' + str(k), s=10)
    createPlot(scatter, [paramsFuncSubplot, paramsFuncLabel], points)


def drawKComparisionFigure(drawSingleFigure, figureName, models_dict, iniMode, maxIter, distMeasure):
    k_list = get_k_list(models_dict)
    plt.figure(figsize=(x_two_columns_plot_size, single_subplot_y_size * len(k_list)))
    for set_name in [set1, set2]:
        for k in k_list:
            drawSingleFigure(k, iniMode, maxIter, distMeasure, set_name, models_dict)
    pdf_path = get_path_to_partial_pdf(models_dict, iniMode, maxIter, distMeasure, figureName)
    plt.savefig(pdf_path)


drawClustersComparisionFigure = partial(drawKComparisionFigure, drawClusterFigure, 'clusters')
drawNewPointsComparisionFigure = partial(drawKComparisionFigure, drawPointsForTestsFigure, 'points_to_test')



def createPlot(plotFunc, paramsFunc, xy_list):
    x, y = getArraysFromTupleList(xy_list)
    for func in paramsFunc:
        func()
    plotFunc(x, y)


def calculateMeanSquareError(models_dict, k, iniMode, maxIter, distMeasure, set_name):
    param_dict = getParamDict(models_dict, k, iniMode, maxIter, distMeasure, set_name)
    kmeans_model = param_dict[kmean_model]
    pointsVectors = [row[0] for row in models_dict[points_sets][set_name].select('features').collect()]
    clustersCenters = [tuple(center) for center in kmeans_model.clusterCenters()]
    getDist = lambda p1, p2: sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    MSE = sum([sqrt(getDist(pointVector, clustersCenters[kmeans_model.predict(pointVector)])) for pointVector in pointsVectors]) / len(pointsVectors)
    param_dict[mse] = MSE


get_xy_list_from_gathered = lambda models_dict, dataLastKey, *params: models_dict[getStringKey(params[-1] + '_' + dataLastKey, *params[:-1])]
get_xy_list_from_dict = lambda models_dict, dataLastKey, *params: getParamDict(models_dict, *params, dataLastKey)


def draw2DPlotsComparision(plotFunc, lastPlotKeys, get_xy_list, dataLastKey, models_dict, *params):
    plt.figure(figsize=(x_two_columns_plot_size, y_plot_size))
    plt.title('plot_comparision')
    plotsParams = zip([1,2], lastPlotKeys)
    for (t_number, set_name) in plotsParams:
        subplot = partial(plt.subplot, 1, 2, t_number)
        xy_list = get_xy_list(models_dict, dataLastKey, *params, set_name)
        setXLabel = partial(plt.xlabel ,x_label_k_value)
        setYLabel = partial(plt.ylabel, dataLastKey)
        createPlot(plotFunc, [subplot, setXLabel, setYLabel], xy_list)
    pdf_path = get_path_to_partial_pdf(models_dict, *params, 'plot_comparision', dataLastKey)
    plt.savefig(pdf_path)


draw2DLinePlotsComparision = partial(draw2DPlotsComparision, plt.plot)
draw2DScatterPlotsComparision = partial(draw2DPlotsComparision, plt.scatter)


draw2DLinePlotsComparisionForSetsGathered = partial(draw2DLinePlotsComparision, [set1, set2], get_xy_list_from_gathered)
draw2DScatterPlotsComparisionForSetsDict = partial(draw2DScatterPlotsComparision, [set1, set2], get_xy_list_from_dict)


def draw2DPlotKStest(models_dict, iniMode, maxIter, distMeasure):
    plt.figure()
    key = getStringKey(KS_test, iniMode, maxIter, distMeasure)
    xy_list = models_dict[key]
    setTitle = partial(plt.title, 'set1 & set2 ks')
    setXLabel = partial(plt.xlabel ,x_label_k_value)
    setYLabel = partial(plt.ylabel, KS_test)
    createPlot(plt.plot, [setTitle, setXLabel, setYLabel], xy_list)
    pdf_path = get_path_to_partial_pdf(models_dict, iniMode, maxIter, distMeasure, 'ks_plot', KS_test)
    plt.savefig(pdf_path)
    plt.figure(figsize=(x_plot_size, y_plot_size))


def drawOriginalSubsetsComparision(models_dict):
    plt.figure(figsize=(x_two_columns_plot_size, y_plot_size))
    sets_dict = models_dict[points_tuples_list]
    for (t_number, pointSets, color, set_name) in zip(range(1,3), (sets_dict[set1], sets_dict[set2]), (blue, green), [set1, set2]):
        x, y = getArraysFromTupleList(pointSets)
        draw = lambda: plt.scatter(x, y, color=color, s=s)
        plotSubplot(draw, 1, 2, t_number)
        title = 'original ' + set_name
        plt.title(title)
    pdf_path = get_path_to_partial_pdf(models_dict, 'original_subsets_comparision')
    plt.savefig(pdf_path)
