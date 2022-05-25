import pickle
import matplotlib.pyplot as plt
import numpy as np

model_dict_path = "/home/karol/pdd/duzeZadanie2/grafy/resultDictionary.pkl"


with open(model_dict_path, 'rb') as f:
    model_dict = pickle.load(f)

# print(model_dict)


#Getting unique labels
n = 60
 
u_labels = np.unique(n)
 
#plotting the results:

cluster_split = model_dict[n]['random'][10]['euclidean']['set2']['clustersSplit']


for i in range(n):
    x = np.array([cor[0] for cor in cluster_split[i]])
    y = np.array([cor[1] for cor in cluster_split[i]])
    plt.scatter(x , y , label = i)
plt.legend()
plt.show()

