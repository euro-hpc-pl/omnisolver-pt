import numba
import numpy as np

from omnisolver.pt.algorithm import perform_monte_carlo_sweeps, should_exchange_states
from omnisolver.pt.model import ising_model
from omnisolver.pt.replica import initialize_replica


@numba.njit
def numba_seed(seed):
    np.random.seed(seed)


@numba.njit
def numba_rand():
    return np.random.rand()


class TestPerformingMonteCarloSweeps:
    def test_updates_all_replicas(self):
        model = ising_model(np.array([1.0, 2.0, 3.0]), np.zeros((3, 3)))
        # We choose the initial state with the worst energy. Hence, every sweep is guaranteed to
        # improve the solution.
        initial_state = np.ones(3, dtype=np.int8)

        replicas = [
            initialize_replica(model, initial_state, beta)
            for beta in np.linspace(0.1, 1.0, 10)
        ]

        perform_monte_carlo_sweeps(replicas, 1)

        assert all(
            replica.current_energy < model.energy(initial_state) for replica in replicas
        )


class TestReplicaExchangeCriterion:
    def test_exchange_shifting_better_solution_to_colder_replica_is_always_accepted(
        self,
    ):
        numba_seed(42)
        model = ising_model(np.ones(3), np.zeros((3, 3)))
        replica_1 = initialize_replica(model, np.ones(3, dtype=np.int8), beta=0.1)
        replica_2 = initialize_replica(
            model, np.array([-1, 1, -1], dtype=np.int8), beta=0.01
        )

        assert should_exchange_states(replica_1, replica_2)
        assert should_exchange_states(replica_2, replica_1)

    def test_moving_better_solution_to_hotter_replica_is_accepted_with_correct_probability(
        self,
    ):
        # Idea of this test is similar to ones in test_replica.py
        numba_seed(42)
        model = ising_model(np.ones(3), np.zeros((3, 3)))
        initial_state_1 = np.array([-1, -1, -1], dtype=np.int8)
        initial_state_2 = np.array([1, 1, 1], dtype=np.int8)
        energy_diff = model.energy(initial_state_1) - model.energy(initial_state_2)
        threshold_beta_difference = np.log(numba_rand()) / energy_diff
        beta_1 = 1.0
        beta_2 = beta_1 - (threshold_beta_difference - 0.1)
        numba_seed(42)

        replica_1 = initialize_replica(model, initial_state_1, beta_1)
        replica_2 = initialize_replica(model, initial_state_2, beta_2)

        assert should_exchange_states(replica_1, replica_2)

    def test_moving_better_solution_to_hotter_replica_is_rejected_with_correct_probability(
        self,
    ):
        numba_seed(42)
        model = ising_model(np.ones(3), np.zeros((3, 3)))
        initial_state_1 = np.array([-1, -1, -1], dtype=np.int8)
        initial_state_2 = np.array([1, -1, -1], dtype=np.int8)
        energy_diff = model.energy(initial_state_1) - model.energy(initial_state_2)
        threshold_beta_difference = np.log(numba_rand()) / energy_diff
        beta_1 = 1.0
        beta_2 = beta_1 - (threshold_beta_difference + 0.1)
        numba_seed(42)

        replica_1 = initialize_replica(model, initial_state_1, beta_1)
        replica_2 = initialize_replica(model, initial_state_2, beta_2)

        assert not should_exchange_states(replica_1, replica_2)
