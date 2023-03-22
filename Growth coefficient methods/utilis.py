'''
@Author: Michael 2023-03-22 21:51:59
This code is about useful functions for TrafficDistribution.py.

'''
import numpy as np


def load_data():
    """Load the data.

    Returns:
        Cur_dis (array): Traffic distribution.
        Pre_O (array): Future traffic generation.
        Pre_D (array): Future traffic attraction.

    """
    Cur_dis = np.array([[17, 7, 4],
                        [7, 38, 6],
                        [4, 5, 17]])
    Pre_O = np.array([38.6, 91.9, 36])
    Pre_D = np.array([39.3, 90.3, 36.9])

    return Cur_dis, Pre_O, Pre_D
