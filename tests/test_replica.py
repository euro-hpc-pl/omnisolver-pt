import numpy as np
import pytest

from omnisolver.pt.model import ising_model
from omnisolver.pt.replica import initialize_replica


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
class TestNewReplica:
    @staticmethod
    def create_model(dtype):
        h_vec_list = [0.5, 0.25, -1.0]
        j_mat_list = [[0.0, 0.3, 0.2], [0.3, 0.0, -2.5], [0.2, -2.5, 0.0]]
        return ising_model(
            np.array(h_vec_list, dtype=dtype), np.array(j_mat_list, dtype=dtype)
        )

    def test_retains_initial_state_and_its_energy_as_its_current_state_and_energy(
        self, dtype
    ):
        model = self.create_model(dtype)
        beta = 1.0
        initial_state = np.array([-1, 1, 1], dtype=np.int8)

        replica = initialize_replica(model, initial_state, beta)

        np.testing.assert_array_equal(replica.current_state, initial_state)
        assert replica.current_energy == dtype(-0.5 + 0.25 - 1.0 - 0.3 - 0.2 - 2.5)

    def test_considers_initial_state_and_energy_to_be_best_so_far(self, dtype):
        model = self.create_model(dtype)
        beta = 0.1
        initial_state = np.array([-1, -1, 1], dtype=np.int8)

        replica = initialize_replica(model, initial_state, beta)

        np.testing.assert_array_equal(replica.best_state_so_far, initial_state)
        assert replica.best_energy_so_far == dtype(-0.5 - 0.25 - 1.0 + 0.3 - 0.2 + 2.5)
