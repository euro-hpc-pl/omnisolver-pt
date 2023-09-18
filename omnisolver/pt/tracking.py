from functools import lru_cache
from typing import Type

from numba.experimental import jitclass
from numba.types import int8


class GroundOnlyTracker:
    def __init__(self, initial_state, initial_energy):
        self.best_state_so_far = initial_state.copy()
        self.best_energy_so_far = initial_energy

    def records(self):
        return [self.best_state_so_far], [self.best_energy_so_far]

    def store(self, new_state, new_energy):
        if new_energy < self.best_energy_so_far:
            self.best_energy_so_far = new_energy
            self.best_state_so_far = new_state


@lru_cache
def construct_tracker(energy_dtype, max_num_states) -> Type[GroundOnlyTracker]:
    if max_num_states == 1:
        return jitclass((("best_state_so_far", int8[:]), ("best_energy_so_far", energy_dtype)))(
            GroundOnlyTracker
        )
    raise NotImplementedError("Only single state tracking is implemented so far.")