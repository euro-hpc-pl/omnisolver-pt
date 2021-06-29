"""Implementation of Parallel Tempering algorithm.

The following choice of data structures has been made to make the implementation
compatible with numba JIT (N denotes number of spins, M denotes number of replicas):

- The Ising Model being solved is represented as bias vector h_vec and
  couplings matrix j_mat.
- Connectivity information is stored as adjacency list. Adjacency list consists of:
  - a numpy array `neighbours_count` of shape (N,) declaring number of neighbours
    of each spin
  - a numpy array `adjacency_list` of shape (N, max(neighbours_count) such that
    adjacency_list[i, j] is index of j-th neighbour of i-th spin.
    If j >= neighbours_count[i], the adjacency_list[i, j] is undefined.
- State of replicas is stored in (M, N) shaped array.
- Energies of replicas are stored in (M,) shaped array.

The choice of storage format for replicas allows for an efficient parallel access
and makes it suitable to be used with numba.prange.
"""
import numba
import numpy as np


@numba.njit(fastmath=True)
def energy_diff(
    h_vec: np.ndarray,
    j_mat: np.ndarray,
    state: np.ndarray,
    position: int,
    neighbours_count: np.ndarray,
    adjacency_list: np.ndarray,
):
    """Compute energy difference between two configurations differing by a spin flip.

    Let H(s_1, ..., s_N) be Ising hamiltonian. Suppose we are given a configuration
    (s_1, s_2,..., s_pos,...,s_N). This function computes the difference:

    delta = H(s_1, s_2,...,s_pos,...,s_N) - H(s_1, s_2,...,-s_pos,...,s_N)

    :param h_vec: vector of biases in Ising hamiltonian.
    :param j_mat: a symmetric matrix of couplings in Ising hamiltonian.
    :param state: a configuration of the system.
    :param position: position of the spin to be flipped.
    :param neighbours_count: array with number of neighbours of each spin. Refer to
     this module's docstring for exact explanation of this argument.
    :param adjacency_list: adjacency list for the graph the model is defined on.
     Refer to this module's docstring for exact explanation of this argument.
    :return: an energy change obtained from current configuration by flipping a
     spin at `position`.
    """
    total = h_vec[position] * state[position]
    for i in range(neighbours_count[position]):
        j = adjacency_list[position, i]
        total += state[position] * j_mat[position, j] * state[j]
    return 2 * total