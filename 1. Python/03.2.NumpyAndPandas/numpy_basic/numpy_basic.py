import numpy as np
import numpy.typing as npt


def construct_array(
        matrix: np.ndarray,
        row_indices: npt.ArrayLike,
        col_indices: npt.ArrayLike) -> np.ndarray:
    """
    Construct slice of given matrix by indices row_indices and col_indices:
    [matrix[row_indices[0], col_indices[0]], ... , matrix[row_indices[N-1], col_indices[N-1]]]
    :param matrix: input matrix
    :param row_indices: list of row indices
    :param col_indices: list of column indices
    :return: matrix slice
    """
    return matrix[row_indices, col_indices]


def detect_identic(
        lhs_array: npt.ArrayLike,
        rhs_array: npt.ArrayLike) -> bool:
    """
    Check whether two arrays are equal or not
    :param lhs_array: first array
    :param rhs_array: second array
    :return: True if input arrays are equal, False otherwise
    """
    if np.isscalar(lhs_array) and np.isscalar(rhs_array):
        return lhs_array == rhs_array

    if isinstance(lhs_array, np.ndarray) and isinstance(rhs_array, np.ndarray):
        if lhs_array.shape == rhs_array.shape:
            return all(lhs_array == rhs_array)
        else:
            return False
    return False


def mean_channel(X: np.ndarray) -> np.ndarray:
    """
    Given color image (3-dimensional vector of size (n, m, 3).
    Compute average value for all 3 channels
    :param X: color image
    :return: array of size 3 with average values
    """
    return np.nanmean(X, axis=(0, 1))


def get_unique_rows(X: np.ndarray) -> np.ndarray:
    """
    Compute unique rows of matrix
    :param X: matrix
    :return: matrix of unique rows
    """
    return np.unique(X, axis=0)


def construct_matrix(first_array: np.ndarray, second_array: np.ndarray) -> np.ndarray:
    """
    Construct matrix from pair of arrays
    :param first_array: first array
    :param second_array: second array
    :return: constructed matrix
    """
    return np.hstack([first_array.reshape(-1, 1),
                      second_array.reshape(-1, 1)])
