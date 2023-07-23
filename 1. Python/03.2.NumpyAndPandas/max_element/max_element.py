import typing as tp
import numpy as np


def max_element(array: np.ndarray) -> tp.Optional[float]:
    """
    Return max element before zero for input array.
    If appropriate elements are absent, then return None
    :param array: array,
    :return: max element value or None
    """
    zeros_indices = np.where(array == 0)[0]
    if len(zeros_indices):
        true_indices = (zeros_indices + 1)[(zeros_indices < len(array) - 1)]
        if len(true_indices):
            return np.max(array[true_indices])
    return None
