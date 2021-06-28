"""Helper functions for converting BQM into a more PT friendly representation."""
from typing import Tuple

import dimod
import numpy as np


def vectorize_bqm(bqm: dimod.BQM) -> Tuple[np.ndarray, np.ndarray]:
    """Convert BQM into dense vector of biases and matrix of couplings.

    .. warning::
        This function assumes that variables are labelled 0,...,N-1,
        otherwise it will behave unexpectedly.

    :param bqm: Binary Quadratic Model to be vectorized
    :return: A tuple (h_vec, j_mat) comprising vector of biases and a matrix
     of quadratic coefficients. Missing biases and couplings are set to 0.
    """
    biases, couplings, _ = bqm.to_ising()
    n = bqm.num_variables
    h_vec = np.zeros(n)
    j_mat = np.zeros((n, n))
    for i, bias in biases.items():
        h_vec[i] = bias

    for (i, j), coupling in couplings.items():
        j_mat[i, j] = j_mat[j, i] = coupling

    return h_vec, j_mat


def adjacency_list_from_couplings(j_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Produce adjacency-list representation of given Ising graph.

    :param j_mat: symmetric matrix of Ising model couplings.
    :return: a tuple (adjacency_list, neighbours_count) such that:
     - neighbours_count.shape == (N,) and neighbours_count[i] contains number of
       neighbours of i-th spin
     - adjacency_list.shape == (N, max(neighbours_count)) and adjacency_list[i, j]
       is i-th spin j-th neighbour (for j < neighbours_count[j]
    """
    neighbours_count = (j_mat != 0).sum(axis=1).astype(int)
    adjacency_list = np.zeros((j_mat.shape[0], neighbours_count.max()), dtype=int)

    for i in range(j_mat.shape[0]):
        counter = 0
        for j in range(j_mat.shape[1]):
            if j_mat[i, j] != 0:
                adjacency_list[i, counter] = j
                counter += 1

    return adjacency_list, neighbours_count
