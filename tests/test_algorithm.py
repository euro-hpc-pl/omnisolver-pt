import numpy as np
import pytest

from omnisolver.pt.algorithm import energy_diff
from omnisolver.pt.bqm_tools import adjacency_list_from_couplings


def _energy(h_vec, j_mat, state):
    return (h_vec * state).sum() + 0.5 * state @ j_mat @ state


def _random_ising(rng, size):
    h_vec = rng.uniform(-2, 2, size)
    j_mat = rng.uniform(-1, 1, (size, size))
    j_mat += j_mat.T
    j_mat[np.diag_indices(size)] = 0
    return h_vec, j_mat


class TestComputingEnergyDifference:
    def test_difference_is_computed_correctly_for_position_of_isolated_spin(self):
        # In the test case below spin 1 is isolated, and therefore the energy difference
        # is equal, up to sign, to 2 * h_vec[1]
        h_vec = np.array([1, 2, 3])
        j_mat = np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        neighbours_count = np.array([1, 0, 1])
        adjacency_list = np.array([[2], [0], [0]])
        state = np.array([1, -1, 1])

        assert (
            energy_diff(h_vec, j_mat, state, 1, neighbours_count, adjacency_list)
            == -2 * h_vec[1]
        )

    @pytest.mark.parametrize("position", list(range(20)))
    def test_difference_is_computed_correctly_for_position_of_non_isolated_spin(
        self, position
    ):
        rng = np.random.default_rng(42)
        h_vec, j_mat = _random_ising(rng, 20)
        adjacency_list, neighbours_count = adjacency_list_from_couplings(j_mat)

        state = 2 * (np.arange(20) % 2) - 1

        flipped_state = state.copy()
        flipped_state[position] = -flipped_state[position]

        initial_energy = _energy(h_vec, j_mat, state)
        flipped_energy = _energy(h_vec, j_mat, flipped_state)

        assert energy_diff(
            h_vec, j_mat, state, position, neighbours_count, adjacency_list
        ) == pytest.approx(initial_energy - flipped_energy)

        assert energy_diff(
            h_vec, j_mat, flipped_state, position, neighbours_count, adjacency_list
        ) == pytest.approx(flipped_energy - initial_energy)
