import numpy as np

from omnisolver.pt.algorithm import perform_monte_carlo_sweeps
from omnisolver.pt.model import ising_model
from omnisolver.pt.replica import initialize_replica


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
