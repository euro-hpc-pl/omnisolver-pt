"""Common testing utilities."""
import numba
import numpy as np


@numba.njit
def numba_seed(seed):
    np.random.seed(seed)


@numba.njit
def numba_rand():
    return np.random.rand()
