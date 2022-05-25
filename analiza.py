import pickle
import matplotlib.pyplot as plt
import numpy as np

from auxiliary import *
from keywords import *

import json



model_dict_path = "/home/karol/pdd/duzeZadanie2/grafy/resTest.pkl"

path_to_parameters = "/home/karol/pdd/duzeZadanie2/grafy/parameters.json"




parameters_file = open(path_to_parameters)
parameters = json.load(parameters_file)
parameters_file.close()


k_list = parameters['k_set']

get_tile_y = { set1 : 0, set2 : 1 }
get_tile_x = dict(zip(k_list, range(len(k_list))))



with open(model_dict_path, 'rb') as f:
    models_dict = pickle.load(f)

# print(model_dict)


#Getting unique labels
n = 30
 
u_labels = np.unique(n)
 
#plotting the results:

# cluster_split = models_dict[n]['random'][10]['euclidean']['set2']['clustersSplit']


def plotSubplot(plt_def, s_x, s_y, t_number):
    plt.subplot(s_x, s_y, t_number)
    plt_def()


def plotCluster(clusterSplit, s_x, s_y, t_number):
    clusterNumbers = clusterSplit.keys()
    
    for i in range(clusterNumbers):
        x, y = getArraysFromTupleList(clusterSplit[i])
        draw = lambda: plt.scatter(x , y , label = i)
        plotSubplot(draw, s_x, s_y, t_number)


def drawClustersFigure(k, iniMode, maxIter, distMeasure, set_name, models_dict):
    paramDict = models_dict[k][iniMode][maxIter][distMeasure][set_name]
    clusters = paramDict[clustersSplit]
    tile_number = get_tile_y[k] * 2 + get_tile_x[set_name]
    plotCluster(clusterSplit, get_tile_x[set_name], get_tile_y[k], tile_number)


operate_on_parameters_and_sets(drawClustersFigure, models_dict)
plt.show()




# for i in range(n):
#     x = np.array([cor[0] for cor in cluster_split[i]])
#     y = np.array([cor[1] for cor in cluster_split[i]])
#     plt.scatter(x , y , label = i)
# plt.legend()
# plt.show()



# x = np.array([0, 1, 2, 3])
# y = np.array([3, 8, 1, 10])

# plt.subplot(2, 2, 1)
# plt.plot(x,y)

# x = np.array([0, 1, 2, 3])
# y = np.array([10, 20, 30, 40])

# plt.subplot(2, 2, 2)
# plt.plot(x,y)


# x = np.array([0, 1, 2, 3])
# y = np.array([10, 20, 30, 40])

# plt.subplot(2, 2, 3)
# plt.plot(x,y)

# x = np.array([0, 1, 2, 3])
# y = np.array([3, 8, 1, 10])

# plt.subplot(2, 2, 4)
# plt.plot(x,y)



# plt.show() 

