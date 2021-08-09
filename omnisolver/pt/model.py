from typing import Literal

import numba
import numpy as np

from omnisolver.pt.bqm_tools import adjacency_list_from_couplings


def _create_ising_model(spec):
    @numba.experimental.jitclass(spec)
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

        def is_equal(self, other):
            return np.array_equal(self.h_vec, other.h_vec) and np.array_equal(
                self.j_mat, other.j_mat
            )

    return IsingModel


SPEC = tuple[
    tuple[Literal["h_vec"], numba.types.Type],
    tuple[Literal["j_mat"], numba.types.Type],
    tuple[Literal["adjacency_list"], numba.types.Type],
    tuple[Literal["neighbours_count"], numba.types.Type],
]

_MODEL_CLS_CACHE: dict[SPEC, type] = {}


def ising_model(h_vec, j_mat):
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

    spec: SPEC = (
        ("h_vec", numba.typeof(h_vec)),
        ("j_mat", numba.typeof(j_mat)),
        ("adjacency_list", numba.typeof(adjacency_list)),
        ("neighbours_count", numba.typeof(neighbours_count)),
    )

    try:
        model_cls = _MODEL_CLS_CACHE[spec]
    except KeyError:
        model_cls = _create_ising_model(spec)
        _MODEL_CLS_CACHE[tuple(spec)] = model_cls

    return model_cls(h_vec, j_mat, adjacency_list, neighbours_count)
