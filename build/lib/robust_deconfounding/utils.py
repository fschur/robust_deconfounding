import numpy as np
from numpy.typing import NDArray


def is_power_of_two(n: int) -> bool:
    """ Check if n is a power of 2
    Attributes:
        n: number to check
    Returns:
        True if n is a power of 2, False otherwise
    """
    if n <= 0:
        return False
    return (n & (n - 1)) == 0


def haarMatrix(n: int, normalized: bool = True) -> NDArray:
    """
    Calculate the Haar matrix of size n

    Arguments:
        n: size of the Haar matrix
        normalized: if True, normalize the matrix
    Returns:
        Haar matrix

    Parts of the code taken from:
    https://stackoverflow.com/questions/23869694/create-nxn-haar-matrix
    """
    # Allow only size n of power 2
    if not is_power_of_two(n):
        raise ValueError("n is not a power of 2. Haar basis can only be calculated for n = 2^k.")

    if n == 1:
        return np.array([1])
    if n > 2:
        h = haarMatrix(n // 2)
    else:
        return np.array([[1, 1], [1, -1]])

    # calculate upper haar part
    h_n = np.kron(h, [1, 1])
    # calculate lower haar part
    if normalized:
        h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])
    else:
        h_i = np.kron(np.eye(len(h)), [1, -1])
    # combine parts
    h = np.vstack((h_n, h_i))
    return h


def cosine_basis(n: int) -> NDArray:
    """
    Generate a cosine matrix of size n with equally spaced sample points

    Arguments:
        n: size of the cosine matrix
    Returns:
        cosine matrix
    """
    sample_points = np.array([i / n for i in range(1, n)])
    tmp = [np.cos(np.pi * sample_points * (k + 1 / 2)) for k in range(n)]
    basis = np.hstack((np.ones((n, 1)), np.sqrt(2) * np.vstack(tmp))).T
    return basis
