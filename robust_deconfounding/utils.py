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


def get_funcbasis(x:NDArray, L:int, type="cosine_cont", intercept=True)->NDArray:
    """
    Returns the first L basis vectors evaluated at x. 
    Arguments:
        L: number of basis vectors
        x: points where the basis vectors are evluated
        type: type of basis spanning the L^2-space "cosine_cont" | "cosine_disc" | "poly"
    Returns: 
        matrix of the first L basis functions evaluated at x
    """

    L_0=(0 if intercept else 1)
   
    if type=="cosine_cont":
        tmp = [np.cos(np.pi * x * k)  for k in range(L_0,L+1)] 
        basis = np.vstack(tmp).T
    elif type=="cosine_disc":
        n=len(x)
        tmp = [np.cos(np.pi * x * (k + 1/2)) for k in range(L_0, L+1)]
        basis = np.vstack(tmp).T
        ind_0=np.arange(0,n)[list(x)==0]
        basis[ind_0, :]=1
    elif type=="poly":
        tmp=[x**k for k in range(L_0, L+1)]
        basis= np.vstack(tmp).T
    else:
        raise ValueError("Invalid basis type.")
    
    return basis


def get_funcbasis_multivariate(x:NDArray, L:NDArray, type="cosine_cont")->NDArray:
    """
    Multivariate version of get_funcbasis, returns the first L basis vectors evaluated at x=(x_1, x_2, ...)
    Arguments:
        L: NDArray of number of basis vectors
        x: points where the basis vectors are evluated
        type: type of basis spanning the L^2-space "cosine_cont" | "cosine_disc" | "poly"
    Returns: 
        matrix of the first L basis functions evaluated at x
    """

    if len(L)!=x.shape[0]:
        raise ValueError("Dimensions of L and x don't coincide.")
    
    basis=get_funcbasis(x=x[0, :], L=L[0], type=type, intercept=True)
    for i in range(1,len(L)):
        basis_add=get_funcbasis(x=x[i, :], L=L[i], type=type, intercept=False)
        basis=np.concatenate((basis,basis_add), axis=1)

    return basis