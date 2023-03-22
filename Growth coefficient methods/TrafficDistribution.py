'''
@Author: Michael 2023-03-22 21:50:33
This code is about the forecast of Traffic Distribution.

'''

from utilis import load_data
import numpy as np

def AverageGrowth(Gap, MaxItera):
    """Average growth coefficient method.

    Args:
        Gap (float): Expected accuracy.
        MaxIter (int): Maximum number of iterations.
    Returns:
        array: Forecast of Traffic Distribution.
    """
    Cur_dis, Pre_O, Pre_D = load_data()
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

def Detroit(Gap, MaxItera):
    """Detroit method.

    Args:
        Gap (float): Expected accuracy.
        MaxIter (int): Maximum number of iterations.
    Returns:
        array: Forecast of Traffic Distribution.
    """
    Cur_dis, Pre_O, Pre_D = load_data()
    Itera = 0
    while MaxItera - Itera:

        Itera += 1
        Cof_O = Pre_O / Cur_dis.sum(axis = 1)
        Cof_D = Pre_D / Cur_dis.sum(axis = 0)
        Cur_dis = Cur_dis * (Cof_O.reshape((3, 1)) @ Cof_D.reshape((1, 3))) * Cur_dis.sum() / Pre_O.sum()

        if np.max(np.abs(Cof_O - 1)) <= Gap and np.max(np.abs(Cof_D - 1)) <= Gap:
            print(f'The number of iterations is {Itera}')
            break

    return Cur_dis


def Fratar(Gap, MaxItera):
    """Fratar method.

    Args:
        Gap (float): Expected accuracy.
        MaxIter (int): Maximum number of iterations.
    Returns:
        array: Forecast of Traffic Distribution.
    """
    Cur_dis, Pre_O, Pre_D = load_data()
    Itera = 0
    while MaxItera - Itera:

        Itera += 1
        Cof_O = Pre_O / Cur_dis.sum(axis = 1)
        Cof_D = Pre_D / Cur_dis.sum(axis = 0)
        L = Cur_dis.sum(axis = 1) / (Cur_dis @ Cof_D)
        L_bar = Cur_dis.sum(axis = 0) / (Cur_dis.T @ Cof_O)
        Cur_dis = Cur_dis * (Cof_O.reshape((3, 1)) @ Cof_D.reshape((1, 3))) * (L.reshape((3, 1)) + L_bar.reshape((1, 3))) / 2
        if np.max(np.abs(Cof_O - 1)) <= Gap and np.max(np.abs(Cof_D - 1)) <= Gap:
            print(f'The number of iterations is {Itera}')
            break

    return Cur_dis

def Furness(Gap, MaxItera):
    """Furness method.

    Args:
        Gap (float): Expected accuracy.
        MaxIter (int): Maximum number of iterations.
    Returns:
        array: Forecast of Traffic Distribution.
    """
    Cur_dis, Pre_O, Pre_D = load_data()
    Itera = 0
    while MaxItera - Itera:

        Itera += 1
        Cof_O = Pre_O / Cur_dis.sum(axis = 1)
        Cof_D = Pre_D / Cur_dis.sum(axis = 0)

        if Itera % 2: Cof_D = np.zeros((1,3))
        else: Cof_O = np.zeros((3,1))

        Cur_dis = Cur_dis * (Cof_O.reshape((3, 1)) + Cof_D.reshape((1, 3)))
        if np.max(np.abs(Cof_O - 1)) <= Gap or np.max(np.abs(Cof_D - 1)) <= Gap:
            print(f'The number of iterations is {Itera}')
            break

    return Cur_dis