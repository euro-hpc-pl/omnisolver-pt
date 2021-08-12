from typing import Sequence

import numba

from omnisolver.pt.replica import Replica


@numba.njit(parallel=True)
def perform_monte_carlo_sweeps(replicas: Sequence[Replica], num_sweeps):
    for i in numba.prange(len(replicas)):
        replica = replicas[i]
        for _ in range(num_sweeps):
            replica.perform_mc_sweep()
