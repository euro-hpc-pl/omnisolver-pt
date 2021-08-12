from typing import Sequence

import numba
import numpy as np

from omnisolver.pt.replica import Replica


@numba.njit(parallel=True)
def perform_monte_carlo_sweeps(replicas: Sequence[Replica], num_sweeps):
    for i in numba.prange(len(replicas)):
        replica = replicas[i]
        for _ in range(num_sweeps):
            replica.perform_mc_sweep()


@numba.njit
def should_exchange_states(replica_1: Replica, replica_2: Replica):
    exponent = (replica_1.current_energy - replica_2.current_energy) * (
        replica_1.beta - replica_2.beta
    )
    return exponent > 0 or np.random.rand() < np.exp(exponent)
