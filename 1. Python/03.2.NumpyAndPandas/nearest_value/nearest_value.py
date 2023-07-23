import typing as tp
import numpy as np


def nearest_value(matrix: np.ndarray, value: float) -> tp.Optional[float]:
    """
    Find nearest value in matrix.
    If matrix is empty return None
    :param matrix: input matrix
    :param value: value to find
    :return: nearest value in matrix or None
    """
    flat = matrix.flatten()
    if len(flat) == 0:
        return None
    array = sorted(np.append(flat, np.array(value)))
    i = array.index(value)
    if len(flat) == 1:
        return array[0]
    else:
        if i+1 < len(array) and array[i+1] - value < value - array[i-1]:
            return array[i+1]
        else:
            return array[i-1]
