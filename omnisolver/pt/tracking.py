from functools import lru_cache
from heapq import heappop, heappush
from typing import Protocol, Sequence, Tuple

import numpy as np
from numba.experimental import jitclass
from numba.types import List, int8, int64

from ._numba_helpers import numba_type_of_cls

Records = Tuple[Sequence[np.ndarray], Sequence[float]]


class Tracker(Protocol):
    def __init__(self, initial_state: np.ndarray, initial_energy: float):  # pragma: no cover
        raise NotImplementedError()

    def records(self) -> Records:  # pragma: no cover
        raise NotImplementedError()

    def store(self, new_state: np.ndarray, new_energy: float):  # pragma: no cover
        raise NotImplementedError()


class TrackerFactory(Protocol):
    def __call__(
        self, initial_state: np.ndarray, initial_energy: np.ndarray
    ) -> Tracker:  # pragma: no cover
        raise NotImplementedError()


class _GroundOnlyTracker:
    def __init__(self, initial_state, initial_energy):
        self.best_state_so_far = initial_state.copy()
        self.best_energy_so_far = initial_energy

    def records(self) -> Records:
        return [self.best_state_so_far], [self.best_energy_so_far]

    def store(self, new_state, new_energy) -> None:
        if new_energy < self.best_energy_so_far:
            self.best_energy_so_far = new_energy
            self.best_state_so_far = new_state.copy()


@lru_cache
def _low_energy_spectrum_tracker(energy_dtype):
    @jitclass((("state", int8[:]), ("energy", energy_dtype)))
    class _HeapItem:
        def __init__(self, state, energy):
            self.state = state
            self.energy = energy

        def __lt__(self, other) -> bool:
            return self.energy < other.energy

    @jitclass((("heap", List(numba_type_of_cls(_HeapItem))), ("num_states", int64)))
    class _LowEnergySpectrumTracker:
        def __init__(self, initial_state, initial_energy, num_states):
            self.heap = [_HeapItem(initial_state.copy(), -initial_energy)]
            self.num_states = num_states

        def records(self) -> Tuple[Sequence[np.ndarray], Sequence[float]]:
            energies = np.array([-item.energy for item in self.heap])
            sorted_indices = np.argsort(energies)
            states = [self.heap[i].state for i in sorted_indices]
            return states, [energies[i] for i in sorted_indices]

        def store(self, new_state, new_energy):
            if self._is_already_stored(new_state):
                return
            new_item = _HeapItem(new_state.copy(), -new_energy)
            heappush(self.heap, new_item)
            if len(self.heap) > self.num_states:
                heappop(self.heap)

        def _is_already_stored(self, state):
            for item in self.heap:
                if np.array_equal(state, item.state):
                    return True
            return False

    return _LowEnergySpectrumTracker


@lru_cache
def tracker_factory(energy_dtype, num_states) -> TrackerFactory:
    if num_states == 1:
        return jitclass((("best_state_so_far", int8[:]), ("best_energy_so_far", energy_dtype)))(
            _GroundOnlyTracker
        )
    else:

        def _tracker_factory(initial_state: np.ndarray, initial_energy: float) -> Tracker:
            return _low_energy_spectrum_tracker(energy_dtype)(
                initial_state, initial_energy, num_states
            )

        return _tracker_factory
