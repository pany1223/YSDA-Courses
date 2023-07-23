import typing as tp
import numpy as np


def nonzero_product(matrix: np.ndarray) -> tp.Optional[float]:
    """
    Compute product of nonzero diagonal elements of matrix
    If all diagonal elements are zeros, then return None
    :param matrix: array,
    :return: product value or None
    """
    diag = np.diag(matrix)
    nonzeros = diag[np.nonzero(diag)]
    if len(nonzeros):
        return np.product(nonzeros)
    else:
        return None
