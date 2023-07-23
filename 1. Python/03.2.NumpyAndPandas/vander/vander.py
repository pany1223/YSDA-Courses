import numpy as np


def vander(array: np.ndarray) -> np.ndarray:
    """
    Create a Vandermod matrix from the given vector.
    :param array: input array,
    :return: vandermonde matrix
    """
    N = array.shape[0]
    basic = np.hstack([list(array.reshape(-1, 1))] * N)
    powers = np.array(list(range(N)) * N).reshape(N, N)
    return np.power(basic, powers)
