"""Implementation of replica used in parallel tempering."""
from functools import lru_cache

import numba
import numpy as np

from .model import IsingModel


class Replica:
    """Replica of Ising spin-glass used in parallel tempering algorithm.

    :ivar model: Instance of Ising model this replica uses.
    :ivar beta: inverse temperature used for sampling.
    :ivar current_state: current state of the replica.
    :ivar current_energy: current energy of the replica. This energy always corresponds to
     replica's current state.
    :ivar best_state_so_far: best state the replica was in so far. This information is updated
     (if necessary) in every Monte-Carlo iteration.
    :ivar best_energy_so_far: energy corresponding to the best state seen by this replica so far.
    """

    def __init__(self, model: IsingModel, initial_state, beta):
        """Create new replica using given model, initial state and inverse temperature.

        :param model: Instance of Ising model this replica uses.
        :param initial_state: initial configuration of the system. It is
         expected that this array has N +/-1 integers, where N is the number of spins in the model.
        :param beta: inverse temperature used for sampling in this replica.

        .. note::
           Initializer of Replica does not perform any validation. If you wish to validate the
           input, use `initialize_replica` function from this module instead.
        """
        self.model = model
        self.beta = beta
        self.current_state = initial_state.copy()
        self.current_energy = model.energy(initial_state)

        self.best_state_so_far = self.current_state.copy()
        self.best_energy_so_far = self.current_energy

    def should_accept_flip(self, energy_diff: float) -> bool:
        """Determine if this replica should accept spin flip resulting in given energy difference.

        :param energy_diff: energy difference between current state of replica and the
         state with flipped spin.
        :return: True if replica should accept the change and False otherwise.

        .. note::
           Obviously, this method is non-deterministic. Energy differences greater than 0
           (i.e. changes that improve energy) are always accepted, while flips worsening
           current energy are accepted with the usual Metropolis-Hastings probability.
        """
        return energy_diff > 0 or np.random.rand() < np.exp(self.beta * energy_diff)

    def perform_mc_sweep(self) -> None:
        """Perform single Monte-Carlo sweep in this replica.

        Performing MC sweep involves triaging every possible spin flip and deciding whether it
        should be accepted or not. The replica's current state and energy, as well as
        replica's best state and energy are update accordingly.
        """
        for i in range(self.model.num_spins):
            energy_diff = self.model.energy_diff(self.current_state, i)
            if self.should_accept_flip(energy_diff):
                self.current_energy -= energy_diff
                self.current_state[i] = -self.current_state[i]
                if self.current_energy < self.best_energy_so_far:
                    self.best_energy_so_far = self.current_energy
                    self.best_state_so_far = self.current_state.copy()


@lru_cache
def _create_replica_cls(spec):
    return numba.experimental.jitclass(spec)(Replica)


def initialize_replica(model: IsingModel, initial_state, beta) -> Replica:
    """Initialize an instance of jit-compiled Replica class.

    :param model: Ising model used by the replica.
    :param initial_state: initial state of the replica. This is expected to be
     a 1D array of +/-1 integers of size N, where N is the number of spins
     in the model. If this requirement is not met, `ValueError` is raised.
    :param beta: inverse temperature of the replica.
    :raises ValueError: If initial state is not of shape (N,), where N
     is the number of spins in the system.
    :return: An instance of jit-compiled Replica class.

    .. note::
       Jit-compiled class definitions are reused, provided that the types
       of model coefficients are the same. Hence, in typical run all replicas
       are of the same numba type.
    """
    if initial_state.shape != (model.num_spins,):
        raise ValueError(
            f"Passed initial state of shape {initial_state.shape}, "
            f"expected ({model.num_spins},) instead."
        )

    scalar_dtype = numba.typeof(model.h_vec).dtype
    state_dtype = numba.types.npytypes.Array(numba.types.int8, 1, "C")

    spec = (
        ("model", getattr(model, "_numba_type_", None)),
        ("beta", scalar_dtype),
        ("current_state", state_dtype),
        ("current_energy", scalar_dtype),
        ("best_state_so_far", state_dtype),
        ("best_energy_so_far", scalar_dtype),
    )

    replica_cls = _create_replica_cls(spec)

    return replica_cls(model, initial_state, beta)
