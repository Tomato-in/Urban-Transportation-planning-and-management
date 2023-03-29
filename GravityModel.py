'''
@Author: Michael 2023-03-29 11:15:49
This code is about Unconstrained Gravity Model.

'''

import numpy as np
from TrafficDistribution import Fratar, AverageGrowth
from utilis import load_data, load_timedata

def AverageGrowth(Gap, MaxItera, Cur_dis, Pre_O, Pre_D):
    """Average growth coefficient method.

    Args:
        Gap (float): Expected accuracy.
        MaxIter (int): Maximum number of iterations.
    Returns:
        array: Forecast of Traffic Distribution.
    """
    # Cur_dis, Pre_O, Pre_D = load_data()
    Itera = 0
    while MaxItera - Itera:

        Itera += 1
        Cof_O = Pre_O / Cur_dis.sum(axis = 1)
        Cof_D = Pre_D / Cur_dis.sum(axis = 0)
        Cur_dis = Cur_dis * (Cof_O.reshape((3, 1)) + Cof_D.reshape((1, 3))) / 2

        if np.max(np.abs(Cof_O - 1)) <= Gap and np.max(np.abs(Cof_D - 1)) <= Gap:
            print(f'The number of iterations is {Itera}')
            break

    return Cur_dis


def GravityModel(Gap, MaxItera):
    Cur_dis, Pre_O, Pre_D = load_data()
    Cur_time, Pre_time = load_timedata()

    Y = np.log(Cur_dis).reshape((-1, 1))

    X_1 = np.ones(Y.shape)
    X_2 = np.log(Cur_dis.sum(axis = 1, keepdims = True) @ Cur_dis.sum(axis = 0, keepdims = True)).reshape((-1, 1))
    X_3 = np.log(Cur_time).reshape((-1, 1))
    X = np.hstack((X_1, X_2, X_3))

    Coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond = None)
    alpha, beta, gama = np.exp(Coeffs[0]), Coeffs[1], - Coeffs[2]

    Cur_dis = alpha * ((Pre_O.reshape((3, 1)) @ Pre_D.reshape((1, 3))) ** beta) / (Pre_time ** gama)
    Cur_dis = AverageGrowth(Gap, MaxItera, Cur_dis, Pre_O, Pre_D)

    return Cur_dis