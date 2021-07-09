import numba
import numpy as np
import pytest

from omnisolver.pt.algorithm import accept_solution, energy_diff
from omnisolver.pt.bqm_tools import adjacency_list_from_couplings


@numba.njit
def _numba_seed(seed):
    np.random.seed(seed)


@numba.njit
def _numba_random():
    return np.random.rand()


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


class TestMonteCarloMove:
    def test_solution_with_lower_energy_is_always_accepted(self):
        _numba_seed(42)
        # Try several times, we should accept each time (since we shouldn't
        # really use random numbers to decide it.
        assert all(accept_solution(4.0, 10.0) for _ in range(100))

    def test_solution_with_worse_energy_is_randomly_accepted_with_correct_probability(
        self,
    ):
        _numba_seed(42)
        # This is tricky, because we cannot inject random generator (see notes
        # in docstring). The idea is, that instead we first generate random sample,
        # and then construct negative and positive case for it.
        beta = 0.1
        u = _numba_random()

        # Negative case
        diff_negative = np.log(u) / beta - 1.0

        # Positive case
        diff_positive = np.log(u) / beta + 1.0

        _numba_seed(42)
        assert not accept_solution(diff_negative, beta)

        _numba_seed(42)
        assert accept_solution(diff_positive, beta)
