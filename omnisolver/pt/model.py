import numba

from omnisolver.pt.bqm_tools import adjacency_list_from_couplings


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

    @numba.experimental.jitclass(
        [
            ("h_vec", numba.typeof(h_vec)),
            ("j_mat", numba.typeof(j_mat)),
            ("adjacency_list", numba.typeof(adjacency_list)),
            ("neighbours_count", numba.typeof(neighbours_count)),
        ]
    )
    class IsingModel:
        def __init__(self, _h_vec, _j_mat, _adjacency_list, _neighbours_count):
            self.h_vec = _h_vec
            self.j_mat = _j_mat
            self.adjacency_list = _adjacency_list
            self.neighbours_count = _neighbours_count

        def energy(self, state):
            total = 0.0
            for i in range(j_mat.shape[0]):
                total += state[i] * self.h_vec[i]
                for j in range(i + 1, self.j_mat.shape[1]):
                    total += state[i] * state[j] * self.j_mat[i, j]
            return total

    return IsingModel(h_vec, j_mat, adjacency_list, neighbours_count)
