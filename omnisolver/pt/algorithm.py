from typing import Sequence

import numba
import numpy as np

from omnisolver.pt.replica import Replica


@numba.njit(parallel=True)
def perform_monte_carlo_sweeps(replicas: Sequence[Replica], num_sweeps) -> None:
    """Perform given number of Monte Carlo sweeps for each replica.

    The operation is parallelized over replicas.

    :param replicas: replicas for which to perform sweeps. The caller should make
     sure that replicas don't share common data (e.g. their states don't point
     to the same place in memory.
    :param num_sweeps: number of sweeps to perform.
    """
    for i in numba.prange(len(replicas)):
        replica = replicas[i]
        for _ in range(num_sweeps):
            replica.perform_mc_sweep()


@numba.njit
def should_exchange_states(replica_1: Replica, replica_2: Replica) -> bool:
    """Determine if given replicas should exchange their states.

    The probability of exchanging states depends on current energy and temperatures
    of replicas. in particular, colder replica should always exchange state
    with hotter replica if its energy is lower.
    """
    exponent = (replica_1.current_energy - replica_2.current_energy) * (
        replica_1.beta - replica_2.beta
    )
    return exponent > 0 or np.random.rand() < np.exp(exponent)


@numba.njit
def exchange_states(replica_1, replica_2):
    """Exchange state and energies of two replicas.

    This function exchanges current states and energies between replicas.
    All other properties of replicas stay the same. In particular, best energies
    and best states found so far by each replica remain unchanged. Temperatures (beta)
    also remain unchanged. This is important for the implementation of PTSampler.
    """
    replica_1.current_energy, replica_2.current_energy = (
        replica_2.current_energy,
        replica_1.current_energy,
    )
    replica_1.current_state, replica_2.current_state = (
        replica_2.current_state,
        replica_1.current_state,
    )
