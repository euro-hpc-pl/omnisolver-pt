from functools import lru_cache

import numba

from .model import IsingModel


class Replica:
    def __init__(self, model: IsingModel, initial_state, beta):
        self.model = model
        self.beta = beta
        self.current_state = initial_state
        self.current_energy = model.energy(initial_state)

        self.best_state_so_far = self.current_state.copy()
        self.best_energy_so_far = self.current_energy


@lru_cache
def _create_replica_cls(spec):
    return numba.experimental.jitclass(spec)(Replica)


def initialize_replica(model: IsingModel, initial_state, beta) -> Replica:
    if initial_state.shape != (model.num_spins,):
        raise ValueError(
            f"Passed initial state of shape {initial_state.shape}, "
            f"expected ({model.num_spins},) instead."
        )

    scalar_dtype = numba.typeof(model.h_vec).dtype
    state_dtype = numba.types.npytypes.Array(numba.types.int8, 1, "C")

    spec = (
        ("model", numba.typeof(model)),
        ("beta", scalar_dtype),
        ("current_state", state_dtype),
        ("current_energy", scalar_dtype),
        ("best_state_so_far", state_dtype),
        ("best_energy_so_far", scalar_dtype),
    )

    replica_cls = _create_replica_cls(spec)

    return replica_cls(model, initial_state, beta)
