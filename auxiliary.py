import numpy as np



def getArraysFromTupleList(pointsList):
    pos = { 'x' : 0, 'y' : 1 }
    
    get_arr = lambda coordinate, points: np.array([point[coordinate] for point in points])
    
    params = [(pos['x'], pointsList), 
              (pos['y'], pointsList)]

    x, y = [get_arr(coordinate, point) for (coordinate, point) in params]

    return (x, y)