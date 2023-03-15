import numpy as np
import pytest

from omnisolver.pt.model import ising_model
from omnisolver.pt.replica import initialize_replica
from omnisolver.pt.testing import numba_rand, numba_seed


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
class TestNewReplica:
    @staticmethod
    def create_model(dtype):
        h_vec_list = [0.5, 0.25, -1.0]
        j_mat_list = [[0.0, 0.3, 0.2], [0.3, 0.0, -2.5], [0.2, -2.5, 0.0]]
        return ising_model(np.array(h_vec_list, dtype=dtype), np.array(j_mat_list, dtype=dtype))

    def test_cannot_be_initialized_when_state_mismatches_model(self, dtype):
        model = self.create_model(dtype)

        with pytest.raises(ValueError):
            initialize_replica(model, np.ones(2), 0.1)

    def test_retains_initial_state_and_its_energy_as_its_current_state_and_energy(self, dtype):
        model = self.create_model(dtype)
        beta = 1.0
        initial_state = np.array([-1, 1, 1], dtype=np.int8)

        replica = initialize_replica(model, initial_state, beta)

        np.testing.assert_array_equal(replica.current_state, initial_state)
        np.testing.assert_almost_equal(
            replica.current_energy, dtype(-0.5 + 0.25 - 1.0 - 0.3 - 0.2 - 2.5)
        )

    def test_considers_initial_state_and_energy_to_be_best_so_far(self, dtype):
        model = self.create_model(dtype)
        beta = 0.1
        initial_state = np.array([-1, -1, 1], dtype=np.int8)

        replica = initialize_replica(model, initial_state, beta)

        np.testing.assert_array_equal(replica.best_state_so_far, initial_state)
        np.testing.assert_almost_equal(
            replica.best_energy_so_far, dtype(-0.5 - 0.25 - 1.0 + 0.3 - 0.2 + 2.5)
        )

    def test_retains_beta_passed_during_initialization(self, dtype):
        model = self.create_model(dtype)
        beta = dtype(0.013)
        initial_state = np.ones(3, dtype=np.int8)

        replica = initialize_replica(model, initial_state, beta)

        assert replica.beta == beta


class TestMonteCarloSweep:
    @pytest.mark.parametrize(
        "model, beta",
        [
            (ising_model(np.array([0.3, 0.4]), np.array([[0, -2.5], [-2.5, 0]])), 0.1),
            (ising_model(np.array([1.0, -1.5]), np.array([[0, 0.5], [0.5, 0]])), 0.001),
        ],
    )
    def test_does_not_disturb_beta_or_the_model(self, model, beta):
        initial_state = np.array([-1, 1], dtype=np.int8)
        replica = initialize_replica(model, initial_state, beta)

        replica.perform_mc_sweep()

        assert replica.model.is_equal(model)
        assert replica.beta == beta

    def test_does_not_worsen_best_solution_found_so_far_and_maintains_solution_consistency(
        self,
    ):
        h_vec = np.array([0.2, 1.3, -1.5])
        j_mat = np.array([[0.0, 2.0, -0.7], [2.0, 0.0, 0.3], [-0.7, 0.3, 0.0]])
        model = ising_model(h_vec, j_mat)
        initial_state = np.array([-1, 1, -1], dtype=np.int8)
        initial_energy = model.energy(initial_state)
        numba_seed(1234)
        replica = initialize_replica(model, initial_state, 0.01)

        replica.perform_mc_sweep()

        assert replica.best_energy_so_far <= initial_energy
        np.testing.assert_almost_equal(
            model.energy(replica.best_state_so_far), replica.best_energy_so_far
        )

    def test_maintains_consistency_between_current_solution_and_current_energy(self):
        h_vec = np.array([0.5, -0.25, 0.75])
        j_mat = np.array([[0.0, 1.0, -0.75], [1.0, 0.0, 0.3], [-0.75, 0.3, 0.0]])
        model = ising_model(h_vec, j_mat)
        initial_state = np.array([1, 1, 1], dtype=np.int8)
        numba_seed(42)
        replica = initialize_replica(model, initial_state, 0.0001)

        for _ in range(10):
            replica.perform_mc_sweep()

        np.testing.assert_almost_equal(model.energy(replica.current_state), replica.current_energy)

    @pytest.mark.parametrize("beta", [10, 1, 0.1])
    @pytest.mark.parametrize("seed", [42, 123, 128])
    def test_better_solutions_are_always_accepted(self, beta, seed):
        h_vec = np.array([-0.5, -0.25, 0.75])
        j_mat = np.zeros((3, 3), dtype=float)
        model = ising_model(h_vec, j_mat)
        initial_state = np.ones(3, dtype=np.int8)
        numba_seed(1234)
        replica = initialize_replica(model, initial_state, beta)

        assert replica.should_accept_flip(2)

    # The energy difference in below two tests is -1.5, which explains
    # how we compute threshold beta.

    def test_worse_solution_is_accepted_with_correct_threshold_probability(self):
        h_vec = np.array([-0.5, -0.25, 0.75])
        j_mat = np.zeros((3, 3), dtype=float)
        model = ising_model(h_vec, j_mat)
        initial_state = np.array([1, 1, -1], dtype=np.int8)
        numba_seed(42)
        threshold_beta = np.log(numba_rand()) / -1.5
        numba_seed(42)  # reseed so we get the same sample
        replica = initialize_replica(model, initial_state, threshold_beta - 0.0001)

        assert replica.should_accept_flip(-1.5)

    def test_worse_solution_is_rejected_with_correct_threshold_probability(self):
        h_vec = np.array([-0.5, 0.75, -0.25])
        j_mat = np.zeros((3, 3), dtype=float)
        model = ising_model(h_vec, j_mat)
        initial_state = np.array([1, -1, 1], dtype=np.int8)
        numba_seed(1234)
        threshold_beta = np.log(numba_rand()) / -1.5
        numba_seed(1234)
        replica = initialize_replica(model, initial_state, threshold_beta + 0.01)

        assert not replica.should_accept_flip(-1.5)
