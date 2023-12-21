"""Dimod-compatible PT sampler."""
import dimod
import numba
import numpy as np

from omnisolver.pt.algorithm import (
    exchange_states,
    perform_monte_carlo_sweeps,
    should_exchange_states,
)
from omnisolver.pt.bqm_tools import vectorize_bqm
from omnisolver.pt.model import ising_model
from omnisolver.pt.replica import initialize_replica


class PTSampler(dimod.Sampler):
    """Parallel tempering samplers."""

    def sample_ising(
        self,
        h,
        J,
        num_replicas=10,
        num_pt_steps=100,
        num_sweeps=100,
        beta_min=0.01,
        beta_max=1.0,
        num_states=1,
        **parameters,
    ):
        """Solve given Ising problem.

        :param h: dictionary of biases.
        :param J: dictionary of couplings.
        :param num_replicas: number of system replicas.
        :param num_pt_steps: number of parallel tempering steps.
        :param num_sweeps: number of Monte Carlo sweeps per parallel tempering step.
        :param beta_min: inverse temperature of the hottest replica.
        :param beta_max: inverse temperature of the coldest replica.
        :returns: single-element sample set with the best solution found.
        """
        bqm = dimod.BQM.from_ising(h, J, 0)

        h_vec, j_mat = vectorize_bqm(bqm)
        model = ising_model(h_vec, j_mat)

        betas = np.geomspace(beta_min, beta_max)

        initial_states = (
            np.random.randint(0, 2, size=(num_replicas, model.num_spins), dtype=np.int8) * 2 - 1
        )

        replicas = numba.typed.List(
            [
                initialize_replica(model, initial_state, beta, num_states)
                for initial_state, beta in zip(initial_states, betas)
            ]
        )

        for _ in range(num_pt_steps):
            perform_monte_carlo_sweeps(replicas, num_sweeps)

            for i in range(num_replicas - 1):
                if should_exchange_states(replicas[i], replicas[i + 1]):
                    exchange_states(replicas[i], replicas[i + 1])

        all_records = [replica.tracker.records() for replica in replicas]
        recorded_states = sum((r[0] for r in all_records), start=[])
        recorded_energies = sum((r[1] for r in all_records), start=[])

        indices = np.argsort(recorded_energies)
        states_to_return = set()

        for i in indices:
            states_to_return.add(tuple(recorded_states[i]))
            if len(states_to_return) == num_states:
                break

        return dimod.SampleSet.from_samples_bqm(list(states_to_return), bqm)

    @property
    def parameters(self):
        return {
            "num_replicas": [],
            "num_pt_steps": [],
            "num_sweeps": [],
            "beta_min": [],
            "beta_max": [],
        }

    @property
    def properties(self):
        return {}
