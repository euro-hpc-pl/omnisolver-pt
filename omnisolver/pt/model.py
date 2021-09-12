from functools import lru_cache

import numba
import numpy as np

from omnisolver.pt.bqm_tools import adjacency_list_from_couplings


class IsingModel:
    def __init__(self, _h_vec, _j_mat, _adjacency_list, _neighbours_count):
        self.h_vec = _h_vec
        self.j_mat = _j_mat
        self.adjacency_list = _adjacency_list
        self.neighbours_count = _neighbours_count

    def energy(self, state):
        total = 0.0
        for i in range(self.j_mat.shape[0]):
            total += state[i] * self.h_vec[i]
            for j in range(i + 1, self.j_mat.shape[1]):
                total += state[i] * state[j] * self.j_mat[i, j]
        return total

    def energy_diff(self, state, position):
        total = self.h_vec[position] * state[position]
        for i in range(self.neighbours_count[position]):
            j = self.adjacency_list[position, i]
            total += state[position] * self.j_mat[position, j] * state[j]
        return 2 * total

    @property
    def num_spins(self):
        return self.h_vec.shape[0]

    def is_equal(self, other):
        return np.array_equal(self.h_vec, other.h_vec) and np.array_equal(
            self.j_mat, other.j_mat
        )


@lru_cache
def _create_ising_model(spec):
    return numba.experimental.jitclass(spec)(IsingModel)


def ising_model(h_vec, j_mat) -> IsingModel:
    if h_vec.dtype != j_mat.dtype:
        raise ValueError(
            "Biases vector and couplings matrix need to have the same dtypes. "
            f"dtypes of received arrays: {h_vec.dtype}, {j_mat.dtype}."
        )
    if h_vec.shape != (h_vec.shape[0],):
        raise ValueError(
            f"Biases need to be 1D array, passed array of shape {h_vec.shape}"
        )
    if j_mat.shape != (h_vec.shape[0], h_vec.shape[0]):
        raise ValueError(
            "Couplings need to be a 2D array of shape NxN, where N is the length "
            f"of biases vector. Received array of shape {j_mat.shape} instead."
        )

    adjacency_list, neighbours_count = adjacency_list_from_couplings(j_mat)

    spec = (
        ("h_vec", numba.typeof(h_vec)),
        ("j_mat", numba.typeof(j_mat)),
        ("adjacency_list", numba.typeof(adjacency_list)),
        ("neighbours_count", numba.typeof(neighbours_count)),
    )

    model_cls = _create_ising_model(spec)

    return model_cls(h_vec, j_mat, adjacency_list, neighbours_count)
